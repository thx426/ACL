# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: move this file to verl.experimental.agent_loop after V1 is stable
"""TransferQueue adapter for AgentLoopManager and AgentLoopWorker"""

import asyncio
import logging
import os
from typing import Any

import ray
import torch
import transfer_queue as tq
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl.experimental.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    get_trajectory_info,
)
from verl.utils.ray_utils import auto_await
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
    params["top_p"] = 1.0
    params["top_k"] = -1
    params["temperature"] = 0


@ray.remote
class AgentLoopWorkerTQ(AgentLoopWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()
        self.background_tasks = set()
        # trace-opd (2026-08-10): rows each prompt's sessions have written to
        # TransferQueue, keyed (partition_id, uid) — the purge-on-retry path
        # (_purge_prompt_rows) needs them to clear residual groups.
        self._written_rows: dict[tuple[str, str], list[str]] = {}

    async def generate_sequences(self, batch: TensorDict) -> None:
        """Spawn agent loop for each sample in the batch without waiting for the results."""
        validate = batch["validate"] if "validate" in batch else False
        batch.pop("validate", None)
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch:
            default_agent_loop = config.agent.default_agent_loop
            batch["agent_name"] = NonTensorData(default_agent_loop)

        trajectory_info = await get_trajectory_info(batch["global_steps"], batch["index"], validate)

        # create background tasks for each sample in the batch
        for i in range(len(batch)):
            # TODO(wuxibin): add trace support
            trace_this_sample = False
            prompt = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    prompt[k] = v[i]
                elif isinstance(v, NonTensorStack):
                    prompt[k] = v[i].data
                elif isinstance(v, NonTensorData):
                    prompt[k] = v.data
                else:
                    logger.exception(f"Unsupported type {type(v)} for key {k}")

            # “fire-and-forget” background tasks
            task = asyncio.create_task(
                self._run_prompt(prompt, sampling_params, trajectory=trajectory_info[i], trace=trace_this_sample)
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def _run_prompt(self, prompt: dict, sampling_params: dict, trajectory: dict, trace: bool = False) -> None:
        """Spawn multiple agent loops in parallel according to rollout.n or rollout.val_kwargs.n."""
        uid, partition_id = prompt["uid"], "train" if not trajectory["validate"] else "val"
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
        # NOTE: user can dynamically adjust n for each sample here, e.g according to task difficulty.
        config = self.config.actor_rollout_ref.rollout
        n = prompt.pop("__rollout_n__", config.n if not trajectory["validate"] else config.val_kwargs.n)
        do_sample = prompt.pop("__do_sample__", True)

        run_sampling_params = dict(sampling_params)
        if not trajectory["validate"] and not do_sample:
            apply_greedy_sampling_params(run_sampling_params)

        # CD-006 R3 (2026-07-25, user-approved): wall-clock timeout per prompt.
        # The `except Exception` below catches raised errors but NOT a permanent
        # hang — e.g. awaiting a vLLM instance whose EngineCore died: the await
        # never returns, `running` never drains, and the whole training loop
        # deadlocks silently (observed 2026-07-25, second A800 incident).
        # On timeout the pending agent loops are cancelled (asyncio.TimeoutError
        # is builtins TimeoutError on py3.11+, an Exception subclass).
        # Default 2700s >> per-task wall p100;
        # set TRACE_AGENT_TASK_TIMEOUT_S=0 to disable.
        _timeout_s = float(os.environ.get("TRACE_AGENT_TASK_TIMEOUT_S", "2700"))

        # trace-opd (2026-08-10, user ruling): a timed-out/failed prompt group
        # must NEVER be trained on as a residual group. Completed sessions have
        # already written their rows to TransferQueue independently, and the
        # rows still missing are systematically the SLOWEST trajectories
        # (long/hard tasks) — training on the remainder is non-random
        # right-censoring of the sampling distribution (observed 2026-08-09,
        # SSB step-1: 191/192 rows, the lost row was task 42515's slowest
        # steered sample). Policy: purge the group's rows and retry the whole
        # group ONCE under the same frozen weights (rollout weights do not move
        # mid-step, so the retry is distributionally identical); on a second
        # failure mark `failure` and let ReplayBuffer.sample fail loud
        # (replay_buffer.py) instead of silently training on a biased batch.
        for attempt in range(2):
            tasks: list[asyncio.Task] = []
            try:
                for i in range(n):
                    task = asyncio.create_task(
                        self._run_agent_loop(
                            run_sampling_params, trajectory=trajectory, trace=trace, session_id=i, **prompt
                        )
                    )
                    tasks.append(task)
                if _timeout_s > 0:
                    await asyncio.wait_for(asyncio.gather(*tasks), timeout=_timeout_s)
                else:
                    await asyncio.gather(*tasks)
                await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "finished"})
                self._written_rows.pop((partition_id, uid), None)  # bookkeeping no longer needed
                return
            except Exception as e:
                logger.exception(f"Error in _run_prompt (attempt {attempt + 1}/2, uid={uid}): {e}")
                # codex review P1-5: gather() does NOT cancel siblings on a
                # non-timeout exception — cancel explicitly, then wait a
                # BOUNDED time for the loops to unwind before purging (a
                # straggler re-writing rows after the purge would resurrect
                # the residual group).
                for t in tasks:
                    t.cancel()
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True), timeout=120.0
                    )
                except (asyncio.TimeoutError, TimeoutError):
                    # A session refused to die within the grace window. The
                    # retry path is no longer safe: a late straggler could
                    # write stale rows under the SAME {uid}_{sid}_{i} keys as
                    # the retry. Purge what we can and mark failure — the
                    # ReplayBuffer fail-loud never samples failure prompts,
                    # so any late write lands on an unsampleable group.
                    logger.error(
                        f"prompt {uid}: session(s) still unwinding after 120s "
                        "grace — skipping the retry, marking failure"
                    )
                    self._purge_prompt_rows(uid, partition_id)
                    break
                self._purge_prompt_rows(uid, partition_id)
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})

    def _purge_prompt_rows(self, uid: str, partition_id: str) -> None:
        """Drop every trajectory row this prompt's sessions have written so a
        retry starts from a clean slate (and a final failure leaves nothing
        sampleable). Bookkept keys are the primary source; ``tq.kv_list()``
        (the same interface ReplayBuffer syncs from) backstops rows whose
        bookkeeping was lost to a cancellation landing inside the batch put.
        Prompt-row keys are ``{uid}_{session_id}_{index}`` — the split-prefix
        match mirrors ReplayBuffer._drop_stale_finished's assumption that uids
        contain no underscore."""
        rows = set(self._written_rows.pop((partition_id, uid), []))
        try:
            listing = tq.kv_list() or {}
            for key, tag in (listing.get(partition_id) or {}).items():
                if tag.get("is_prompt", False) or key == uid:
                    continue
                if key.split("_")[0] == uid:
                    rows.add(key)
        except Exception:
            logger.exception("kv_list backstop failed during purge; proceeding with bookkept keys only")
        if rows:
            tq.kv_clear(partition_id=partition_id, keys=sorted(rows))
            logger.warning(
                f"Purged {len(rows)} residual trajectory row(s) of prompt {uid} "
                f"(partition={partition_id}) before retry/failure"
            )

    async def _agent_loop_postprocess(
        self, output: AgentLoopOutput | list[AgentLoopOutput], validate, **kwargs
    ) -> None:
        """Put agent loop outputs into TransferQueue."""
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            logger.warning(f"Empty output for prompt {uid}_{session_id}")
            return

        await self._compute_score(outputs, kwargs=kwargs)

        final_output = outputs[-1]

        if final_output.reward_score is not None:
            for output in outputs[:-1]:
                output.reward_score = final_output.reward_score
                output.extra_fields["reward_extra_info"] = final_output.extra_fields["reward_extra_info"]

        # NOTE: agent loop may has multiple outputs, put each output into TransferQueue.
        # key format: {uid}_{session_id}_{index}
        # - uid: raw prompt uid from dataset
        # - session_id: session id for rollout.n sampling
        # - index: index of agent loop output
        keys, fields, tags = [], [], []
        for i, output in enumerate(outputs):
            prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(output.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
            ).squeeze(0)

            keys.append(f"{uid}_{session_id}_{i}")
            field = output.as_dict()
            field.update(kwargs)
            # do not store raw image/video
            field.pop("multi_modal_data", None)
            # trace-opd PGS: expand the sparse steering ledger into dense
            # per-token behaviour-correction weights riding verl's native
            # rollout_is_weights channel (consumed by the vanilla policy loss
            # OUTSIDE the PPO clip). Only steered trajectories carry the
            # ledger; everything else ships no field and trains untouched.
            extra = field.get("extra_fields") or {}
            # trace-opd v4 (self-researcher): research rows form their own
            # GRPO group — suffix the uid so advantage grouping (grpo_loo by
            # uid) never mixes them with the task's executor rows. λ_r and
            # warmup are folded into reward_score upstream (linear in the
            # LOO-mean advantage), so this suffix is the ONLY trainer-side
            # change the mechanism needs.
            if extra.get("sample_role") == "research":
                field["uid"] = f"{kwargs['uid']}:research"
            # 审阅 IF-09:判定用「PGS 协议在场」(steer_count 恒由 rollout 侧
            # finalize 写入,空账也在),而非「本轨迹有 steer 记录」——批内
            # schema 必须恒一致,空账展开为全 1 权重。
            wbeh_mode = str(extra.get("steer_w_beh_mode", "per_token"))
            if "steer_count" in extra and wbeh_mode != "off":
                # Hydra ALWAYS materialises algorithm.rollout_correction as a
                # mapping (sub-keys default to null) — `is not None` would trip
                # on every default config. The native feature only writes
                # rollout_is_weights when rollout_is/rollout_rs are set
                # (rollout_corr_helper contract), so that is the exclusivity
                # test.
                corr = self.config.algorithm.get("rollout_correction", None)
                if corr is not None and (
                    corr.get("rollout_is") or corr.get("rollout_rs")
                ):
                    raise RuntimeError(
                        "steering W_beh and algorithm.rollout_correction both write "
                        "rollout_is_weights — enable at most one"
                    )
                from verl.trainer.ppo.steering_weights import expand_behavior_weights

                dense = expand_behavior_weights(
                    positions=extra.get("steer_positions") or [],
                    log_pi=extra.get("steer_log_pi") or [],
                    log_b=extra.get("steer_log_b") or [],
                    response_len=int(field["response_mask"].shape[-1]),
                    mode=wbeh_mode,
                    log_ratio_cap=extra.get("steer_log_ratio_cap"),
                    rejected=extra.get("steer_rejected") or [],
                    shaped_gamma=float(extra.get("steer_w_beh_gamma", 0.1)),
                )
                field["rollout_is_weights"] = torch.from_numpy(dense)
            # TODO: uniform response_mask and loss_mask
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["position_ids"] = position_ids
            field["multi_modal_inputs"] = multi_modal_inputs
            fields.append(field)
            prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
            tags.append(
                {
                    "status": "success",
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "seq_len": prompt_len + response_len,
                    # These tags are used for off-policy staleness control, if a trajectory
                    # spans too many global steps, we need to filter it out.
                    # global_steps: which global steps this sample is from dataloader
                    "global_steps": kwargs["global_steps"],
                    # min_global_steps: start generation model weights version of this trajectory
                    "min_global_steps": field["extra_fields"].get("min_global_steps"),
                    # max_global_steps: end generation model weights version of this trajectory
                    "max_global_steps": field["extra_fields"].get("max_global_steps"),
                    # trace-opd PGS v3.3: per-row arm label for the asymmetric
                    # advantage (0=bare, 1=steered; absent e.g. on research
                    # rows -> None, the estimator treats it as "unknown" and
                    # falls back to plain LOO within the row's own uid group).
                    "steer_arm": field["extra_fields"].get("steer_arm"),
                }
            )

        put_partition = "train" if not validate else "val"
        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id=put_partition,
        )
        # trace-opd (2026-08-10): record the written keys for the
        # purge-on-retry path. Synchronous append right after the awaited put:
        # a cancellation can only land on an await point, so put-then-record
        # is atomic; a cancel inside the put itself is covered by the
        # kv_list() backstop in _purge_prompt_rows.
        self._written_rows.setdefault((put_partition, uid), []).extend(keys)


class AgentLoopManagerTQ(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = AgentLoopWorkerTQ
        super().__init__(*args, **kwargs)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    def generate_sequences(self, prompts: TensorDict) -> None:
        """
        Dispatch input batch to agent loop workers without blocking. Workers should put agent loop outputs
        into TransferQueue once an agent loop finished.

        Args:
            prompts (TensorDict): Input batch from train or validation dataset.
        """
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=False)
            ]
        )
