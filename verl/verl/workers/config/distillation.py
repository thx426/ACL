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

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig
from verl.utils.config import omega_conf_to_dataclass

from .rollout import RolloutConfig

__all__ = ["DistillationLossConfig", "DistillationTeacherModelConfig", "DistillationConfig"]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class DistillationLossConfig(BaseConfig):
    """Configuration for distillation loss settings.

    loss_mode (str):
        Distillation loss function to use.
    topk (int, optional):
        Number of top tokens to consider for top-k distillation losses.
    use_task_rewards (bool):
        Whether to include task rewards alongside distillation loss.
    distillation_loss_coef (float):
        Coefficient for distillation loss when combined with task rewards.
    loss_max_clamp (float, optional):
        Maximum value to clamp distillation loss. If None, no clamping is applied.
    log_prob_min_clamp (float, optional):
        Minimum value to clamp log probabilities for stability, e.g., log q - log p where p or q are
        very close to zero. If None, no clamping is applied.
    use_policy_gradient (bool):
        Whether to incorporate distillation loss as a reward, as done
        by https://thinkingmachines.ai/blog/on-policy-distillation/. Recommended to use loss_mode=k1.
        Otherwise, distillation loss is directly backpropagated as a supervised loss,
        as in https://arxiv.org/abs/2306.13649. Recommended to use loss_mode=k3 or forward_kl_topk.
    policy_loss_mode (str):
        Name of the policy loss to use when use_policy_gradient is true.
    clip_ratio (float):
        PPO clipping ratio for policy loss.
    clip_ratio_low (float):
        Lower bound for PPO clipping ratio.
    clip_ratio_high (float):
        Upper bound for PPO clipping ratio.
    loss_settings (DistillationLossSettings, optional):
        Runtime-populated settings based on loss_mode. Not set by user.
    gate_beta (float):
        Sigmoid sharpness of the confidence gate, consumed ONLY by
        ``loss_mode='reverse_kl_gated'`` (the SDAR/SEED gated logp-shift).
        5.0 is both papers' default (SDAR ``sdar_gate_beta``, SEED
        ``opd_gate_beta``); inert under every other loss_mode.
    opd_loss_coef (float):
        trace-opd mixed objective: weight of the distillation term. 1.0 (default)
        is the historical behaviour. Note that with the other arms off this is a
        plain multiplier on the loss, i.e. on the effective learning rate — the
        pure-OPD baseline was established at 1.0. 0 drops the term entirely,
        which is the arm that asks whether OPD still contributes at all.
    rl_loss_coef (float):
        trace-opd mixed objective: weight of the task-reward policy-gradient term.
        0 disables it (default) and keeps the pure-distillation behaviour.
    sft_loss_coef (float):
        trace-opd mixed objective: weight of the cross-entropy term over expert
        tokens marked by the ``sft_mask`` field. 0 disables it (default).
    """

    loss_mode: str = "k3"
    topk: Optional[int] = 128
    use_task_rewards: bool = True
    distillation_loss_coef: float = 1.0
    loss_max_clamp: Optional[float] = 10.0
    log_prob_min_clamp: Optional[float] = -10.0

    # Chunked top-K log-probs (opt-in, avoids [B, T, V] log_softmax buffer
    # at long context). Only consumed by ``loss_mode='forward_kl_topk'``.
    # Default ``False`` to preserve short-context performance (chunked path
    # has ~6x time overhead at N=14K, V=152K). Set ``True`` when hitting OOM
    # at long context (>=64K tokens, V=152K) where the baseline path OOMs.
    use_chunked_topk: bool = False
    # Tokens per chunk along (B*T) when ``use_chunked_topk=True``. Larger
    # chunks reduce kernel-launch overhead but increase per-chunk memory;
    # smaller chunks reduce per-chunk memory but increase kernel-launch
    # overhead (saved-tensor total stays constant either way).
    chunked_topk_chunk_size: int = 4096

    use_policy_gradient: bool = True
    policy_loss_mode: str = "vanilla"
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2

    # Confidence-gate sharpness for loss_mode='reverse_kl_gated' (SDAR/SEED
    # form); inert everywhere else. Both source repos default to 5.0.
    gate_beta: float = 5.0

    # trace-opd mixed objective (docs/mixed-objective-design.md):
    #     opd_loss_coef * OPD  +  rl_loss_coef * RL  +  sft_loss_coef * SFT
    # RL is the task-reward policy gradient over the same GRPO advantages
    # upstream's use_task_rewards path uses; SFT is a token-level cross-entropy
    # over the expert tokens marked by ``sft_mask`` (rollouts produced WITH the
    # privileged document).
    #
    # All three are aggregated the same way — a per-MICRO-BATCH token mean, which
    # is what the pure-OPD path has always done (only ppo_loss ever fills
    # global_batch_info, and pure distillation never calls it). That convention
    # accumulates to N x mean over the N micro-batches of a gradient-accumulation
    # step where a global mean accumulates to 1 x mean, so mixing the two
    # conventions across terms costs a factor of N (~60 at this run's shapes).
    # Hence the RL term does not go through ppo_loss; see task_reward_pg.
    #
    # RL and SFT default to 0.0 and OPD to 1.0, which is the pure-distillation
    # path unchanged: nothing extra activates, and the rollout ships no extra field.
    opd_loss_coef: float = 1.0
    rl_loss_coef: float = 0.0
    sft_loss_coef: float = 0.0

    # trace-opd teacher grounding (docs/teacher-grounding-design.md): reverse
    # distillation on bare SUCCESS trajectories — rows with sample_role == "tg"
    # carry the doc-prefixed sequence as their TRAINING prompt (the forward is
    # the doc-conditioned teacher q) and the bare policy's renormalised top-k
    # as their label (shipped through the teacher_ids/teacher_logprobs channel,
    # scored under the BARE prompt via teacher_prefix_ids). The per-token math
    # is byte-identical to forward_kl_topk — sum p_label*(log p_label - log q) —
    # so the logits processor needs no new mode; the term aggregates the SAME
    # per-token matrix under ``tg_mask`` with its own batch-global sequence
    # count (seq-mean-token-mean). 0.0 = arm off, byte-identical.
    #     ... + tg_loss_coef * TG
    tg_loss_coef: float = 0.0

    # trace-opd self-research module (docs/self-research-design.md): the research
    # ROLE's own RL term, orthogonal to the executor-side mix above (those three
    # optimise "execute better"; this one optimises "research better"; the only
    # coupling is the data flow — the winning document takes the teacher seat).
    # Rows with extra_fields.sample_role == "research" ship a ``research_mask``
    # (their researcher-generated tokens; their response_mask is cleared at TQ
    # put time, so they leave every executor-side term and denominator) and a
    # ``research_advantage`` (group-relative doc-utility reward, filled in by
    # the trainer's advantage stage). The loss is a clipped PG over those
    # tokens normalised by the batch-GLOBAL research-token count, so its value
    # does not depend on how rows are packed into micro-batches.
    #     ... + research_loss_coef * RESEARCH   (0.0 = channel off, byte-identical)
    research_loss_coef: float = 0.0
    # joint: research rows share the executor backward — ONE optimizer step,
    #        perfectly on-policy (every gradient at the sampling weights).
    # separate: research rows are split out and trained in a SECOND update
    #        within the same global step (UCOB/SEED form; one step off-policy,
    #        held by the PPO clip). Kept as a switchable ablation/fallback.
    research_update_mode: str = "joint"
    # Hard clamp on the group-relative research advantage (UCOB ships +/-2.0).
    research_adv_clip: float = 2.0
    # Research rows contribute no gradient before this many global steps (UCOB
    # warms its writer for ~30); document production and teacher selection run
    # from step one regardless — warmup gates the gradient, not the mechanism.
    research_warmup_steps: int = 0
    # trace-opd intervention-OPD (its design 12): minimum number of rows with
    # any distillation signal (non-zero response_mask) per training batch.
    # Below the floor the WHOLE actor update is skipped — a batch whose loss
    # is exactly 0 would still move the weights through AdamW's decoupled
    # weight decay and advance the LR scheduler. 0 disables the floor.
    min_signal_rows: int = 0

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    # Store distillation loss settings for computing the specified loss_mode
    # Not set by user, populated at runtime
    loss_settings: Optional[dict] = None

    def __post_init__(self):
        self._mutable_fields.add("loss_settings")
        from verl.trainer.distillation.losses import DistillationLossSettings, get_distillation_loss_settings

        self.loss_settings: DistillationLossSettings = get_distillation_loss_settings(self.loss_mode)

        if self.policy_loss_mode != "vanilla":
            raise NotImplementedError(
                f"Only vanilla policy loss is currently supported when use_policy_gradient is True, "
                f"but got {self.policy_loss_mode}."
            )

        if self.use_policy_gradient and self.loss_settings.use_topk:
            print(
                f"WARNING: {self.loss_mode} is most effective as a supervised distillation loss "
                "(use_policy_gradient=False). With policy gradient, the update uses only the sampled"
                " token's logprob ∇logπ(a), so the top-k distributional signal (how non-sampled logits "
                "should move) is largely unused."
            )

        if not self.use_policy_gradient and self.loss_mode == "k1":
            raise ValueError(
                "Directly backpropagating k1 loss is incorrect since gradient of k1 loss"
                " wrt model weights does not depend on teacher log probabilities."
            )

        # trace-opd mixed objective: a negative or non-finite coefficient flips or
        # blows up the corresponding term silently, so reject it at config time.
        for name in ("opd_loss_coef", "rl_loss_coef", "sft_loss_coef", "tg_loss_coef"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0 (0 disables the term), got {value}")

        # trace-opd teacher grounding: the tg rows ride the teacher_ids /
        # teacher_logprobs channel and their per-token loss is computed by the
        # forward_kl_topk logits-processor formula (which IS the tg soft-CE when
        # the label is the renormalised bare top-k). Any other loss_mode would
        # silently apply the wrong math to tg rows.
        if float(self.tg_loss_coef) > 0.0 and self.loss_mode != "forward_kl_topk":
            raise ValueError(
                f"tg_loss_coef > 0 requires loss_mode='forward_kl_topk' (the tg rows reuse its "
                f"per-token formula as their soft-CE), got loss_mode={self.loss_mode!r}"
            )

        gate_beta = float(self.gate_beta)
        if not math.isfinite(gate_beta) or gate_beta <= 0.0:
            # beta -> 0 degenerates the gate to a flat 0.5 (a plain half-strength
            # logp shift) and a negative beta INVERTS it — strengthening exactly
            # the teacher-rejected tokens. Neither is ever what an experiment means.
            raise ValueError(f"gate_beta must be finite and > 0, got {gate_beta}")

        if not (self.opd_loss_coef or self.rl_loss_coef or self.sft_loss_coef):
            raise ValueError(
                "opd_loss_coef, rl_loss_coef and sft_loss_coef are all zero — there is no "
                "objective left to optimise"
            )

        if self.rl_loss_coef > 0.0 and self.use_task_rewards:
            raise ValueError(
                "rl_loss_coef and use_task_rewards both scale the same task-reward policy-gradient "
                "term. Use rl_loss_coef (explicit weight, trace-opd mixed objective) or "
                "use_task_rewards (upstream, implicit weight 1), not both."
            )

        # trace-opd self-research module: reject silently-wrong shapes at config time.
        research_coef = float(self.research_loss_coef)
        if not math.isfinite(research_coef) or research_coef < 0.0:
            raise ValueError(
                f"research_loss_coef must be finite and >= 0 (0 disables the channel), got {research_coef}"
            )
        if self.research_update_mode not in ("joint", "separate"):
            raise ValueError(
                f"research_update_mode must be 'joint' or 'separate', got {self.research_update_mode!r}"
            )
        if research_coef > 0.0:
            adv_clip = float(self.research_adv_clip)
            if not math.isfinite(adv_clip) or adv_clip <= 0.0:
                raise ValueError(f"research_adv_clip must be finite and > 0, got {adv_clip}")
            if int(self.research_warmup_steps) < 0:
                raise ValueError(f"research_warmup_steps must be >= 0, got {self.research_warmup_steps}")
        if int(self.min_signal_rows) < 0:
            raise ValueError(f"min_signal_rows must be >= 0 (0 disables), got {self.min_signal_rows}")


@dataclass
class DistillationTeacherModelConfig(BaseConfig):
    """Configuration for on-policy distillation teacher.

    key (str, optional):
        Identifier to route examples to the teacher model in multi-teacher setting.
    model_path (str, optional):
        Model path for the teacher model. Can be a local path or a Hugging Face model
    inference (RolloutConfig):
        Rollout configuration for the teacher model inference during distillation.
    num_replicas (int):
        Number of inference replicas of this teacher to launch. Each replica occupies
        `per_replica_world_size` GPUs (= inference.data_parallel_size *
        inference.tensor_model_parallel_size * inference.pipeline_model_parallel_size),
        so the teacher's total GPU footprint is
        `num_replicas * per_replica_world_size`.
    """

    _mutable_fields = BaseConfig._mutable_fields | {"num_replicas", "key"}

    key: Optional[str] = None
    model_path: Optional[str] = None
    inference: RolloutConfig = field(default_factory=RolloutConfig)
    num_replicas: Optional[int] = 0

    @property
    def per_replica_world_size(self) -> int:
        return (
            self.inference.tensor_model_parallel_size
            * self.inference.data_parallel_size
            * self.inference.pipeline_model_parallel_size
        )

    @property
    def world_size(self) -> int:
        return self.num_replicas * self.per_replica_world_size

    def check_configured(self):
        if self.model_path is None:
            raise ValueError("model_path must be specified for distillation teacher model config.")
        if self.key is None:
            raise ValueError("key must be specified for distillation teacher model config.")
        if self.num_replicas is None:
            raise ValueError("num_replicas must be specified for distillation teacher model config.")

    def validate_and_prepare_for_distillation(self, use_topk: bool, topk: Optional[int]) -> None:
        # Prompt + Response from student are fed into teacher as context
        max_model_len = self.inference.max_model_len
        student_prompt_length = self.inference.prompt_length
        student_response_length = self.inference.response_length
        required_context_len = student_prompt_length + student_response_length + 1
        if max_model_len is not None and required_context_len > max_model_len:
            raise ValueError(
                "Distillation teacher inference requires room for the student prompt, the full student "
                f"response, and one generated token, but got {student_prompt_length=}, "
                f"{student_response_length=}, {required_context_len=}, {max_model_len=}."
            )
        self.inference.prompt_length = self.inference.prompt_length + self.inference.response_length
        self.inference.response_length = 1
        self._validate_topk_logprobs(use_topk=use_topk, topk=topk)

    def _validate_topk_logprobs(self, use_topk: bool, topk: Optional[int]) -> None:
        if not use_topk:
            return
        if topk is None:
            raise ValueError("topk must be specified when use_topk is True.")

        engine_name = self.inference.name
        engine_kwargs = self.inference.engine_kwargs
        match engine_name:
            case "vllm":
                vllm_engine_kwargs = dict(engine_kwargs.get("vllm", {}))
                max_logprobs = vllm_engine_kwargs.get("max_logprobs")
                if max_logprobs is None:
                    vllm_engine_kwargs["max_logprobs"] = topk
                    max_logprobs = topk
                if max_logprobs < topk:
                    raise ValueError(
                        f"VLLM max_logprobs ({max_logprobs}) must be >= distillation_loss topk "
                        f"({topk}) to enable distillation loss computation."
                    )
                engine_kwargs["vllm"] = vllm_engine_kwargs
            case "sglang":
                # SGLang's top_logprobs_num is a per-request parameter, so there is no
                # engine-boot cap to align (unlike vLLM's max_logprobs). The async
                # server translates sampling_params["prompt_logprobs"] into
                # return_logprob + logprob_start_len=0 + top_logprobs_num at call time.
                pass
            case _:
                raise NotImplementedError(
                    f"DistillationTeacherModelConfig does not support inference engine {engine_name}"
                )


@dataclass
class DistillationConfig(BaseConfig):
    """Configuration for on-policy distillation.

    enabled (bool):
        Whether on-policy distillation is enabled.
    n_gpus_per_node (int):
        Number of GPUs per node in the teacher resource pool.
    nnodes (int):
        Number of nodes in the teacher resource pool.
    teacher_models (dict[str, TeacherModelConfig]):
        Configurations for teacher models used for multi-teacher distillation.
    teacher_key (str):
        Key to route examples to the appropriate teacher model in multi-teacher setups. Should correspond to a field in
        the data proto, e.g., data_source.
    distillation_loss (DistillationLossConfig):
    Configuration for distillation loss settings.

    NOTE: The `teacher_model` entry is in the `teacher_models` dict by default.
    Since it is popped when other teacher entries are added, using `teacher_model` as
    one of several keys silently drops it. For example, the following CLI overrides result
    in ONLY `teacher_model2` being used:

    ```bash
    distillation.teacher_models.teacher_model.key=openai/gsm8k
    distillation.teacher_models.teacher_model.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    Instead, give the first teacher a different name:

    ```bash
    +distillation.teacher_models.teacher_model1.key=openai/gsm8k
    +distillation.teacher_models.teacher_model1.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    """

    _mutable_fields = BaseConfig._mutable_fields | {"teacher_models", "n_gpus_per_node", "nnodes"}

    enabled: bool = False
    # H3 pilot self-teacher mode: teacher log-probs are computed by the ROLLOUT
    # engine itself (current policy weights + a privileged prompt prefix from
    # the agent loop). No dedicated teacher pool, servers, or manager are
    # created; teacher_models/pool sizes are ignored.
    self_teacher: bool = False
    n_gpus_per_node: int = 0
    nnodes: int = 0
    teacher_models: dict[str, DistillationTeacherModelConfig] = field(default_factory=dict)
    teacher_key: str = "data_source"
    distillation_loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)

    def __post_init__(self):
        if not self.enabled:
            return
        if self.self_teacher:
            # Rollout engine doubles as the teacher: skip teacher-model
            # resolution and pool-size validation entirely.
            return

        self.teacher_models = self._resolve_teacher_models()
        teacher_world_size_sum = 0
        for teacher_model in self.teacher_models.values():
            teacher_model.validate_and_prepare_for_distillation(
                use_topk=self.distillation_loss.loss_settings.use_topk,
                topk=self.distillation_loss.topk,
            )
            teacher_world_size_sum += teacher_model.world_size
        total_pool_size = self.n_gpus_per_node * self.nnodes
        if teacher_world_size_sum != total_pool_size:
            raise ValueError(
                f"Sum of teacher (num_replicas * per_replica_world_size) ({teacher_world_size_sum}) must match "
                f"the distillation resource pool size "
                f"({self.n_gpus_per_node=} * {self.nnodes=} = {total_pool_size})."
            )

    def _resolve_teacher_models(self) -> dict[str, DistillationTeacherModelConfig]:
        assert "teacher_model" in self.teacher_models
        if len(self.teacher_models) == 1:
            # Single teacher occupies the entire teacher resource pool.
            teacher_model = self.teacher_models["teacher_model"]
            inference = teacher_model.inference
            per_replica = (
                inference.tensor_model_parallel_size
                * inference.data_parallel_size
                * inference.pipeline_model_parallel_size
            )
            pool_size = self.n_gpus_per_node * self.nnodes
            if pool_size % per_replica != 0:
                raise ValueError(
                    f"Single teacher's per_replica_world_size ({per_replica}) must divide the distillation "
                    f"resource pool size ({self.n_gpus_per_node=} * {self.nnodes=} = {pool_size})."
                )
            teacher_model.num_replicas = pool_size // per_replica
            teacher_model.key = "default"
        else:
            # Multiple teachers: remove default single teacher config
            self.teacher_models.pop("teacher_model")

        # Teacher models dict is keyed by teacher_key instead of YAML entry name
        teacher_models = {}
        for teacher_config in self.teacher_models.values():
            teacher_config = omega_conf_to_dataclass(teacher_config, dataclass_type=DistillationTeacherModelConfig)
            teacher_config.check_configured()
            if teacher_config.key in teacher_models:
                raise ValueError(f"Duplicate teacher key {teacher_config.key} found in teacher models.")
            teacher_models[teacher_config.key] = teacher_config
        return teacher_models
