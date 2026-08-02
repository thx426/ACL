"""TraceAgentLoop — the whole per-sample mechanism inside one verl AgentLoop
(docs/design.md §2, §6). For every training sample (= one task):

  1. RESEARCH (skipped on validation): a researcher session studies THIS task
     by running real experiment episodes on the rollout engine — i.e. on the
     policy weights being trained this very step — and finalizes an
     experience document. The document is a transient product of this step:
     archived for audit/SFT, never read back later.
  2. STUDENT ROLLOUT: one bare on-policy episode through the same episode
     runner (same contract, same engine, same weights).
  3. TEACHER PREFIX: extra_fields["teacher_prompt_ids"] carries the
     scaffold + "## Task guidance" + document first-turn prompt; the verl
     self-teacher patch scores the SAME response tokens under it.

Failure discipline (review P0-3): systemic misconfiguration dies at startup
(_shared_state raises, and the launcher pre-flights the researcher endpoint);
every PER-SAMPLE research step — assembly included — degrades to doc=None
(teacher = bare prefix ⇒ per-token KL ≈ 0, a neutral sample) and is counted.
Research must never turn a trainable sample into a TQ failure.

Observability: every sample (train and val) dumps its full student trajectory
under <run_dir>/rollouts/step_<N>[_val]/, and every research session archives
its conversation, experiment episodes (with narrations), document, and meta
under <run_dir>/research/ — together with verl checkpoints bound to the same
run_dir this makes a run inspectable and resumable from the latest step.

Config via environment variables (set by the launcher):
  TRACE_CONFIG          path to configs/trace.yaml   (required)
  TRACE_RUN_DIR         run root (archives + dumps)  (required; server-absolute)
  TRACE_AGENT_WORKERS   number of agent-loop worker processes; per-process
                        concurrency caps are configured GLOBALLY in trace.yaml
                        and divided by this here (defaults to
                        train.agent_num_workers)

Registered as agent_name "trace_opd" via rollout.agent.agent_loop_config_path.
Requires the trace-opd verl branch (self-teacher patches + the
validate/global_step pass-through in _run_agent_loop).
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)

from ..common.aio import shared_task
from ..common.config import Config, load_config
from ..common.hashing import response_token_hash
from ..common.llm import make_client
from ..common.tape import TapeWriter
from ..core.episode import EpisodeResult, run_episode
from ..core.expert import build_expert_sample
from ..core.template_delta import TemplateDelta, render_ids
from ..core.types import OUTCOME_ENV_ERROR, inject_guidance
from ..docstore.archive import ResearchArchive
from ..envs.alfworld.adapter import AlfworldEnvFactory
from ..research.prompts import build_researcher_system
from ..research.session import ResearchSession, SessionResult
from ..research.toolproto import make_tool_protocol
from ..research.tools import ToolRouter
from .backends import (
    EngineGenerate,
    EngineResearcherTokenChat,
    engine_narrate_fn,
    engine_researcher_chat,
)
from .curve import build_curve_metrics
from .research_group import select_best_doc

logger = logging.getLogger(__file__)

_STATE_LOCK = threading.Lock()
_STATE: dict[str, Any] = {}


def _as_int(value: Any) -> int | None:
    """Normalize step-like values: V1/TQ may hand us a tensordict
    NonTensorData wrapper (review P0-2) — unwrap defensively, never trust the
    transport type."""
    value = getattr(value, "data", value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _per_worker(cap: int, workers: int) -> int:
    """trace.yaml concurrency caps are GLOBAL; each of the N agent-loop
    worker processes gets its share (review P1-1)."""
    return max(1, math.ceil(cap / max(1, workers)))


def _shared_state(tokenizer) -> dict[str, Any]:
    """Process-wide singletons: config, template delta, env factory, external
    clients (shared connection pools/semaphores), archive. Raises on systemic
    misconfiguration — startup is the right time to die."""
    with _STATE_LOCK:
        if "config" in _STATE:
            return _STATE
        config_path = os.environ["TRACE_CONFIG"]
        run_dir = Path(os.environ["TRACE_RUN_DIR"])
        config: Config = load_config(config_path)
        from ..core import protocol

        protocol.configure_episode_caps(
            config.episode.max_env_steps, config.episode.max_assistant_turns
        )
        workers = int(os.environ.get("TRACE_AGENT_WORKERS", config.train.agent_num_workers))

        # Divide the global caps across worker processes.
        config.alfworld.spawn_concurrency = _per_worker(
            config.alfworld.spawn_concurrency, workers
        )

        pid = os.getpid()  # tape files are per-process: no cross-process writers
        tape_dir = run_dir / "tape" if config.run.tape else None
        researcher_client = None
        if config.research.enabled and config.research.backend == "external":
            role = config.role("researcher")
            role.max_concurrency = _per_worker(role.max_concurrency, workers)
            researcher_client = make_client(
                "researcher",
                role,
                tape=TapeWriter(tape_dir / f"researcher.{pid}.jsonl") if tape_dir else None,
            )
        narrator_client = None
        if config.narration.backend == "external":
            role = config.role("narrator_external")
            role.max_concurrency = _per_worker(role.max_concurrency, workers)
            narrator_client = make_client(
                "narrator",
                role,
                tape=TapeWriter(tape_dir / f"narrator.{pid}.jsonl") if tape_dir else None,
            )

        _STATE.update(
            {
                "config": config,
                "run_dir": run_dir,
                "delta": TemplateDelta(tokenizer),
                "factory": AlfworldEnvFactory(config.alfworld),
                "archive": ResearchArchive(run_dir),
                "researcher_client": researcher_client,
                "narrator_client": narrator_client,
                "executor_model_name": Path(config.train.model_path).name
                or "the policy under training",
                # rollout_n > 1: the n copies of a task in one step share ONE
                # research session (keyed by (step, task_id); see
                # common.aio.shared_task for the invariants).
                "research_tasks": {},
                # Self-research channel: per-(step, task) group registry — the
                # N research replicas register (doc, p) here and the bare
                # replicas wait for the full group to pick the teacher.
                # Same single-event-loop invariant as research_tasks.
                "research_groups": {},
                # Mixed objective: hands out the replica index within a
                # (step, task) so the trailing copies can be expert rollouts.
                # Same invariant as research_tasks — all copies of a task land
                # in one worker process and one event loop.
                "replica_counters": {},
            }
        )
        return _STATE


class TraceAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        state = _shared_state(self.tokenizer)
        self.cfg: Config = state["config"]
        self.run_dir: Path = state["run_dir"]
        self.delta: TemplateDelta = state["delta"]
        self.factory: AlfworldEnvFactory = state["factory"]
        self.archive: ResearchArchive = state["archive"]
        self.researcher_client = state["researcher_client"]
        self.narrator_client = state["narrator_client"]
        self.executor_model_name: str = state["executor_model_name"]
        self.research_tasks: dict = state["research_tasks"]
        self.research_groups: dict = state["research_groups"]
        self.replica_counters: dict = state["replica_counters"]

    def _next_replica_index(self, global_step: int | None, task_id: str) -> int:
        """Index of this copy within the (step, task) rollout group.

        No lock and no await between read and write: all copies of a task run as
        coroutines on one event loop (the same invariant ``shared_task`` relies
        on), so the increment is atomic with respect to them.
        """
        key = (global_step, task_id)
        index = self.replica_counters.get(key, 0)
        self.replica_counters[key] = index + 1
        return index

    # ------------------------------------------------------------- research

    def _narrate_fn(self, route: str):
        backend = self.cfg.narration.backend
        if backend == "none":
            return None
        if backend == "engine":
            return engine_narrate_fn(self.server_manager, self.tokenizer, request_id=route)
        client = self.narrator_client  # backend == "external", validated at load

        async def narrate(messages, temperature, max_tokens):
            result = await client.chat(messages, temperature=temperature, max_tokens=max_tokens)
            return result.text

        return narrate

    def _researcher_chat(self, route: str, task_id: str):
        rcfg = self.cfg.research
        if rcfg.backend == "external":
            client = self.researcher_client

            async def chat(messages, tools):
                # route_key pins the whole multi-turn session to one 27B DP
                # engine (X-data-parallel-rank; no-op unless roles.researcher
                # dp_size > 0). Task-id only — no step — so re-researching the
                # same task lands on the same engine across steps.
                return await client.chat(
                    messages, tools=tools, max_tokens=rcfg.researcher_max_tokens,
                    route_key=task_id,
                )

            return chat
        role = self.cfg.roles.get("researcher")  # backend == "engine"
        return engine_researcher_chat(
            self.server_manager,
            self.tokenizer,
            temperature=role.temperature if role else 0.7,
            max_tokens=rcfg.researcher_max_tokens,
            request_id=route,
        )

    async def _research(
        self, adapter, policy_version: int | None, route: str
    ) -> tuple[str | None, dict[str, Any]]:
        """One research session; returns (doc | None, metric fields).
        The WHOLE per-sample path — assembly included — degrades on failure;
        it never raises (review P0-3)."""
        rcfg = self.cfg.research
        router: ToolRouter | None = None
        session: ResearchSession | None = None
        result: SessionResult | None = None
        error: str | None = None
        try:
            router = ToolRouter(
                adapter=adapter,
                research=rcfg,
                narration=self.cfg.narration,
                # Per-EPISODE routing key (2026-07-24, user ruling): task-level
                # pinning starved one replica during long-tail tasks (measured:
                # 2 GPUs idle while 2 ground the tail). A fresh key per factory
                # call lets verl least-loaded spread episodes; turns WITHIN an
                # episode still share the key (its growing-prefix KV survives).
                # Cost: k copies re-prefill the shared doc+scaffold once each
                # (~1-2k tok) — seconds vs minutes of tail idle.
                generate_factory=lambda: EngineGenerate(
                    self.server_manager, self.tokenizer,
                    request_id=f"{route}:r{uuid4().hex[:6]}",
                ),
                tokenizer=self.tokenizer,
                delta=self.delta,
                episode_response_budget=self.rollout_config.response_length,
                narrate_fn=self._narrate_fn(route),
            )
            session = ResearchSession(
                researcher_chat=self._researcher_chat(route, adapter.task_id),
                toolproto=make_tool_protocol(rcfg.tool_protocol),
                router=router,
                system_prompt=build_researcher_system(
                    task_brief=adapter.task_brief(),
                    executor_model=self.executor_model_name,
                    n_exp=rcfg.n_exp,
                    doc_max_words=rcfg.doc_max_words,
                    gt_text=adapter.gt_text() if rcfg.gt == "on" else None,
                ),
                max_turns=rcfg.max_turns,
                wrapup_buffer=rcfg.wrapup_buffer,
            )
            # session.run() itself degrades mid-session errors to a partial
            # result; wait_for adds the hard wall.
            result = await asyncio.wait_for(session.run(), timeout=rcfg.session_timeout_s)
            error = result.error
        except asyncio.TimeoutError:
            error = f"research session exceeded {rcfg.session_timeout_s:.0f}s"
            if session is not None:
                result = session.snapshot(error=error)  # partial record survives
        except Exception as err:  # noqa: BLE001 — assembly failure: degrade
            error = f"{type(err).__name__}: {err}"
        if error:
            logger.warning("research degraded for %s: %s", adapter.task_id, error)

        archive_path: str | None = None
        records = list(router.state.episodes.values()) if router is not None else []
        try:
            path = await asyncio.to_thread(
                self.archive.save,
                task_id=adapter.task_id,
                result=result,
                records=records,
                policy_version=policy_version,
                extra_meta={
                    "gt": rcfg.gt,
                    "gt_injected": bool(rcfg.gt == "on" and adapter.gt_text()),
                    "narration_backend": self.cfg.narration.backend,
                    # SFT provenance: which model produced this session's
                    # trajectory, over which tool transport.
                    "researcher_model": (
                        self.researcher_client.model
                        if rcfg.backend == "external" and self.researcher_client
                        else "policy(engine)"
                    ),
                    "tool_protocol": rcfg.tool_protocol,
                    "error": error,
                },
            )
            archive_path = str(path.relative_to(self.run_dir))
        except Exception:  # noqa: BLE001
            logger.exception("archive write failed for %s", adapter.task_id)

        doc = result.doc if result is not None else None
        att = result.last_exp_attempts if result is not None else 0
        fields = {
            "research_error": int(error is not None),
            "research_experiments": result.experiments_used if result else 0,
            "research_episodes": result.episodes_consumed if result else 0,
            "research_turns": result.turns_used if result else 0,
            "research_final_success": (
                result.last_exp_successes / att if result is not None and att else 0.0
            ),
            "doc_words": len(doc.split()) if doc else 0,
            "doc_missing": int(not doc),
        }
        # One greppable line per session in the trainer log (the per-step
        # document-quality process signal).
        logger.info(
            "research %s: exp=%s eps=%s final=%s/%s doc_words=%s%s",
            adapter.task_id,
            fields["research_experiments"],
            fields["research_episodes"],
            result.last_exp_successes if result else 0,
            att,
            fields["doc_words"],
            " ERROR" if error else "",
        )
        return doc, {**fields, "research_archive": archive_path or ""}

    # -------------------------------------------------------------- teacher

    def _teacher_prompt_ids(self, adapter, doc: str, student_prompt_ids, initial_obs: str):
        """Privileged first-turn prompt, or None when it cannot be used.
        Uses the ADAPTER scaffold (same source as the student prompt — review
        P2-4) and enforces the engine context-length invariant up front."""
        from ..core import protocol

        teacher_prompt_ids = render_ids(
            self.tokenizer,
            [
                {
                    "role": "system",
                    "content": inject_guidance(adapter.executor_scaffold(), doc),
                },
                {"role": "user", "content": protocol.truncate_obs(initial_obs)},
            ],
            True,
        )
        max_len = self.cfg.train.rollout_max_model_len
        if len(teacher_prompt_ids) + self.rollout_config.response_length + 8 > max_len:
            logger.warning(
                "doc for %s overflows teacher context (%d prompt tokens) — bare teacher",
                adapter.task_id,
                len(teacher_prompt_ids),
            )
            return None
        # Privileged and bare prefixes must end in the same generation context.
        n = len(self.delta.gen_prefix) + 2
        assert teacher_prompt_ids[-n:] == student_prompt_ids[-n:], (
            "teacher/bare prompts diverge near the generation prefix"
        )
        return teacher_prompt_ids

    # ----------------------------------------------------------------- dump

    def _dump_rollout(self, *, step: int | None, validate: bool, payload: dict[str, Any]) -> None:
        step_dir = f"step_{step:05d}" if step is not None else "step_unknown"
        if validate:
            step_dir += "_val"
        path = (
            self.run_dir / "rollouts" / step_dir
            / f"{payload['task_id'].replace('/', '__')}_{uuid4().hex[:8]}.json"
        )
        from ..common.jsonl import write_json

        write_json(path, payload)

    # ----------------------------------------------------------------- main

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        metrics = AgentLoopMetrics()
        validate = bool(kwargs.get("validate", False))
        global_step = _as_int(kwargs.get("global_step"))
        extra_info = dict(kwargs["extra_info"])
        adapter = self.factory.adapter_from_item(extra_info)
        response_length = self.rollout_config.response_length
        # Base routing namespace, per (step, task). 2026-07-24 rebalance (user
        # ruling): engine calls are now routed per-EPISODE — student episodes
        # and research episodes append a unique suffix (":s…"/":r…") so verl's
        # least-loaded selection can spread long-tail work across replicas
        # (task-level pinning measurably idled half the GPUs during tails).
        # An episode's own turns still share one key → its growing-prefix KV
        # survives. Narration keeps the task key (one-shot small calls).
        route = f"task:{global_step}:{adapter.task_id}"

        doc: str | None = None
        research_fields: dict[str, Any] = {}
        # 2026-07-25 (user-approved): OVERLAP research with the student rollout.
        # The student episode is BARE — `_student_rollout` takes no document; only
        # the teacher scoring below needs `doc`. Awaiting research *before* the
        # rollout serialized the two and left ~half the batch blocked: measured
        # concurrency 21 vs a structural ceiling of ~112 (4 tasks still in research
        # blocked their 4x8=32 bare rollouts). Kicking research off as a task and
        # awaiting it only where `doc` is first needed keeps identical semantics
        # (policy weights are frozen within a step; shared_task still de-dupes one
        # session per (step, task)) while letting both phases run concurrently.
        research_fut = None
        # Self-research channel (docs/self-research-design.md): with the arm on,
        # documents come from the N-replica research GROUP (argmax by measured
        # doc utility), not from one shared session — so the shared task is not
        # started at all. Off: byte-identical single-session path below.
        research_rl_arm = (
            not validate
            and self.cfg.research_rl.weight > 0
            and self.cfg.research.enabled
            and self.cfg.research.backend == "engine"
        )
        if self.cfg.research.enabled and not validate and not research_rl_arm:
            # Shared across the rollout_n copies of this task in this step:
            # exactly one research session runs (and archives) per (step,
            # task); every copy distills against the same document.
            research_fut = asyncio.ensure_future(
                shared_task(
                    self.research_tasks,
                    (global_step, adapter.task_id),
                    lambda: self._research(adapter, global_step, route),
                )
            )

        # Mixed objective (docs/mixed-objective-design.md): the trailing copies
        # of a task roll out WITH the document and feed the SFT arm instead of
        # the OPD/RL arms. They need the document before they can start, so they
        # await research first; every other copy keeps the overlap below.
        # With the arm off this short-circuits on the weight — no replica
        # counter, no extra rollout, and (below) no sample_role stamp, which is
        # what keeps the trainer from allocating an sft_mask per row.
        sft_arm = (
            not validate
            and self.cfg.train.mix_sft_weight > 0
            and self.cfg.train.mix_sft_rollouts > 0
            # The expert rollout needs a document; either source will do — the
            # shared session (research_fut) or the research group (argmax).
            and (research_fut is not None or research_rl_arm)
        )
        # Replica layout per task: [0, rollout_n) bare | [+mix_sft_rollouts)
        # expert | [+research_rl.n_docs) research. With both arms off no index
        # is drawn at all (byte-identical to the historical path).
        n_bare = self.cfg.train.rollout_n
        n_expert = self.cfg.train.mix_sft_rollouts if sft_arm else 0
        replica_index = (
            self._next_replica_index(global_step, adapter.task_id)
            if (sft_arm or research_rl_arm)
            else None
        )
        is_expert = (
            sft_arm
            and replica_index is not None
            and n_bare <= replica_index < n_bare + n_expert
        )
        is_research = (
            research_rl_arm
            and replica_index is not None
            and replica_index >= n_bare + n_expert
        )
        if is_research:
            return await self._research_replica(
                adapter, global_step, route, sampling_params, response_length
            )
        expert_fields: dict[str, Any] = {}
        research_awaited = False

        async def await_research() -> None:
            # Only the *blocking* remainder is charged here; the part that ran
            # under the rollout is free. (Pre-overlap this metric was the full
            # research wall time — expect it to drop sharply.)
            nonlocal doc, research_fields, research_awaited
            t_block0 = time.time()
            if research_rl_arm:
                doc, research_fields = await self._await_research_group(
                    global_step, adapter.task_id
                )
            else:
                doc, research_fields = await research_fut
            metrics.tool_calls += time.time() - t_block0
            research_awaited = True

        expert_missing = ""
        if is_expert:
            await await_research()
            if not doc:
                # Research failed, so there is nothing to imitate. The slot stays
                # an expert slot: turning it into an extra bare rollout would
                # hand the batch more on-policy data exactly on the tasks where
                # research fails — which correlates with task difficulty, so the
                # SFT arms would quietly train on a different, easier-skewed
                # sample than the arms they are compared against. It contributes
                # nothing instead, and says so on a curve.
                expert_missing = "no_doc"

        t0 = time.time()
        result = None
        if is_expert and not expert_missing:
            result = await self._expert_rollout(
                adapter, doc, sampling_params, response_length, route
            )
            if not result.tokens.response_ids:
                # An environment fault before the first generation. An empty
                # response is a shape the trainer cannot take (as_dict writes
                # rm_scores[-1]), so this slot needs SOME tokens — but for the
                # same reason as above it must not become a bare training
                # sample. It is re-rolled bare and then masked out entirely.
                logger.warning(
                    "expert episode for %s produced no tokens (%s); the slot contributes nothing",
                    adapter.task_id,
                    result.trajectory.error,
                )
                expert_missing = "env_error"
                result = None
        if result is None:
            result = await self._student_rollout(adapter, sampling_params, response_length, route)
        metrics.generate_sequences += time.time() - t0

        if (research_fut is not None or research_rl_arm) and not research_awaited:
            await await_research()

        traj = result.trajectory
        env_steps = sum(1 for s in traj.steps if s.acted)
        tokens = result.tokens
        if is_expert and expert_missing:
            # The slot exists but has no supervision to offer: zero mask keeps it
            # out of SFT, and sample_role="sft" keeps it out of the on-policy
            # terms, the GRPO group, and every behaviour metric.
            tokens = replace(tokens, response_mask=[0] * len(tokens.response_ids))
            expert_fields = {
                "sample_role": "sft", "sft_role": 1, "sft_won": 0,
                "sft_accepted": 0, "sft_dropped_ref": 0,
                f"sft_missing_{expert_missing}": 1,
            }
        elif is_expert:
            tokens, expert_fields = build_expert_sample(
                adapter=adapter,
                tokenizer=self.tokenizer,
                result=result,
                max_model_len=self.cfg.train.rollout_max_model_len,
                prefix=self.cfg.train.mix_sft_prefix,
            )

        output = AgentLoopOutput(
            prompt_ids=tokens.prompt_ids,
            response_ids=tokens.response_ids,
            response_mask=tokens.response_mask,
            reward_score=1.0 if traj.won else 0.0,
            num_turns=2 * result.assistant_turns + 1,
            metrics=metrics,
            extra_fields={
                "task_id": adapter.task_id,
                "won": traj.won,
                # Observation packs are self-contained (docs/microscope-design.md):
                # turn structure rides the row instead of a cross-file join.
                "turn_spans": [list(x) for x in result.tokens.turn_spans],
                "outcome": traj.outcome,
                "env_steps": env_steps,
                "invalid_format": result.invalid_format,
                "truncated": result.truncated,
                **result.gen_extra,
                **research_fields,
                **expert_fields,
            },
        )

        # Teacher grounding(docs/teacher-grounding-design.md):won=1 的 bare 行
        # 按用户裁定退出 OPD——tg 臂开时**转换**为 tg 行(训练 prompt 换成 doc
        # 前缀版 = q 的前向;打分前缀换成裸 prompt = p̄ 的标签;response 原样,
        # v2.2 已锁的同构性质保证换前缀合法),tg 臂关而 opd_lost_only 开时以
        # 零 mask 占位行存在(T1' 归因臂;verl 合约要求每 replica 一行,物理
        # 上不能不 put)。expert/research 行与 val 不经此路。
        tg_arm = (
            not validate
            and self.cfg.train.mix_tg_weight > 0
            and self.cfg.research.enabled
        )
        lost_only = self.cfg.train.opd_lost_only and not validate
        tg_converted = False
        if traj.won and not is_expert and (tg_arm or lost_only):
            # 标签质量(设计 §动力学:枢纽变量 = won 标签的 thinking 健康度)。
            # 口径与 scripts/won_traj_evolution.py 的 thought_chars_per_turn
            # 逐字节同源:Σ len(steps[].thought) / len(steps)。在线曲线发
            # chars/turns 两个和,cpt = 两条曲线相除(= 全部标签行的 turn 加权
            # per-turn 均值;预注册锐判据的在线版,基线 315→1986 ×6)。
            thought_chars = sum(len(s.thought or "") for s in traj.steps)
            thought_turns = len(traj.steps)
            gate = self.cfg.train.mix_tg_health_gate
            unhealthy = (
                gate is not None
                and thought_chars / max(thought_turns, 1) > gate
            )
            tg_prompt_ids = None
            if tg_arm and doc and not unhealthy:
                tg_prompt_ids = self._teacher_prompt_ids(
                    adapter, doc, result.tokens.prompt_ids, traj.initial_obs
                )
            if tg_prompt_ids is not None:
                # 完整 tg 行:doc 前缀训练前向 + 裸前缀打分标签。
                output.prompt_ids = list(tg_prompt_ids)
                output.extra_fields["teacher_prompt_ids"] = list(
                    result.tokens.prompt_ids
                )
                output.extra_fields["sample_role"] = "tg"
                output.extra_fields["tg_role"] = 1
                # 只有真正成为标签的行才计入标签质量曲线(被门挡掉/降级的
                # 占位行不是标签)。cpt 键与离线基准同 estimand(codex
                # impl-review P1-5):离线 won_traj_evolution 是逐轨迹
                # chars/turns 再平均(trajectory-weighted),chars÷turns 两曲线
                # 相除则是 turn-weighted,轨迹长短差异大时二者分离——判读用
                # mean(cpt)/mean(tg_role) 恢复标签行的 trajectory mean,直接
                # 对齐 284–315 健康基线;chars/turns 两键保留作总量视图。
                output.extra_fields["tg_label_thought_chars"] = thought_chars
                output.extra_fields["tg_label_thought_turns"] = thought_turns
                output.extra_fields["tg_label_thought_cpt"] = (
                    thought_chars / max(thought_turns, 1)
                )
            else:
                # 无 doc / 前缀溢出 / 健康门拦截 / tg 臂关:won 行按
                # opd_lost_only 语义零贡献占位(role="tg" + 全零 mask ⇒ TQ 侧
                # tg_mask 亦零,打分侧判 tg_skip,损失两侧都看不见它)。
                output.response_mask = [0] * len(output.response_ids)
                output.extra_fields["sample_role"] = "tg"
                reason = (
                    "unhealthy" if (tg_arm and doc and unhealthy) else
                    "overflow" if (tg_arm and doc) else
                    ("nodoc" if tg_arm else "lostonly")
                )
                output.extra_fields[f"tg_dropped_{reason}"] = 1
            tg_converted = True

        if sft_arm:
            # Stamped on every row of the batch, and ONLY when the SFT arm is
            # on: the trainer keys the whole expert-sample machinery off this
            # field's presence, so with the arm off it allocates and ships
            # nothing. Expert rows already carry "sft" from expert_fields.
            output.extra_fields.setdefault("sample_role", "bare")
        if tg_arm or lost_only:
            # 同一纪律:臂开时每行携带 stamp,TQ 侧据此统一分配 tg_mask,
            # batch 布局在有无 tg 行时保持一致;关断时零字段零传输。
            output.extra_fields.setdefault("sample_role", "bare")
        if research_rl_arm:
            # Same discipline for the research channel: the stamp on EVERY row
            # is what makes the TQ side allocate research_mask uniformly; off,
            # nothing ships. Research rows return earlier with their own stamp.
            output.extra_fields.setdefault("sample_role", "bare")
            output.extra_fields["research_arm"] = 1

        if not validate and not is_expert and not tg_converted:
            # Teacher scoring joins the task's sticky replica too (verl pops
            # this key and uses it as the routing id for the scoring request —
            # the 8 copies then share one prefill of the doc prefix).
            output.extra_fields["teacher_routing_key"] = route
        if doc and not validate and not is_expert and not tg_converted:
            teacher_prompt_ids = self._teacher_prompt_ids(
                adapter, doc, result.tokens.prompt_ids, traj.initial_obs
            )
            if teacher_prompt_ids is not None:
                output.extra_fields["teacher_prompt_ids"] = teacher_prompt_ids
            else:
                output.extra_fields["doc_missing"] = 1  # overflow degraded to bare

        # Curve channel: numeric per-sample scalars -> swanlab (train:
        # rollout_extra/<key>/mean via verl f8f26d5; val: val-aux/...). After
        # the teacher step so doc_missing reflects overflow degradation.
        output.extra_fields["reward_extra_info"] = build_curve_metrics(
            validate=validate,
            research_enabled=self.cfg.research.enabled,
            mixed_sft=self.cfg.train.mix_sft_weight > 0 and self.cfg.train.mix_sft_rollouts > 0,
            mixed_tg=self.cfg.train.mix_tg_weight > 0 or self.cfg.train.opd_lost_only,
            research_rl=self.cfg.research_rl.weight > 0,
            fields=output.extra_fields,
        )

        try:
            await asyncio.to_thread(
                self._dump_rollout,
                step=global_step,
                validate=validate,
                payload={
                    "task_id": adapter.task_id,
                    "split": extra_info.get("split", ""),
                    "validate": validate,
                    "global_step": global_step,
                    "outcome": traj.outcome,
                    "won": traj.won,
                    "env_steps": env_steps,
                    "assistant_turns": result.assistant_turns,
                    "invalid_format": result.invalid_format,
                    "truncated": result.truncated,
                    # Mixed objective: "bare" (OPD/RL arms) or "sft" (an expert
                    # rollout run WITH the document). Analysis and the viz layer
                    # must be able to separate the two populations — an expert
                    # trajectory's ~96% success would otherwise inflate every
                    # behaviour curve read off the dumps.
                    "sample_role": output.extra_fields.get("sample_role", "bare"),
                    "sft_accepted": output.extra_fields.get("sft_accepted", 0),
                    "doc_words": research_fields.get("doc_words", 0),
                    "research": {
                        k: v for k, v in research_fields.items() if k != "doc_words"
                    },
                    "trajectory": traj.to_dict(),
                    # Distillation-microscope join keys (analysis-design.md §2):
                    # the training-side sampler records the same rolling hash
                    # over response ids; turn spans bucket per-token KL by turn.
                    "response_token_len": len(result.tokens.response_ids),
                    "response_token_hash": response_token_hash(
                        result.tokens.response_ids
                    ),
                    "turn_token_spans": result.tokens.turn_spans,
                    "ts": time.time(),
                },
            )
        except Exception:  # noqa: BLE001 — dumps must never fail a sample
            logger.exception("rollout dump failed for %s", adapter.task_id)
        return output

    # ------------------------------------------------- self-research channel

    def _purge_stale_step_state(self, global_step) -> None:
        """codex review P1-6: research_groups and replica_counters grow one
        entry per (step, task) forever — groups even hold N document texts.
        Steps run as barriers (a new step's replicas only spawn after the
        previous batch was consumed), so entries older than the previous step
        are dead. Keep one step of slack for the V1 pipeline's tail."""
        if global_step is None:
            return
        cutoff = int(global_step) - 1
        for registry in (self.research_groups, self.replica_counters):
            stale = [
                k for k in registry
                if isinstance(k, tuple) and isinstance(k[0], int) and k[0] < cutoff
            ]
            for k in stale:
                del registry[k]

    def _research_group(self, key) -> dict[str, Any]:
        group = self.research_groups.get(key)
        if group is None:
            self._purge_stale_step_state(key[0])
            group = {
                "results": [],
                "event": asyncio.Event(),
                "n": self.cfg.research_rl.n_docs,
            }
            self.research_groups[key] = group
        return group

    def _register_research_result(self, global_step, task_id, **entry) -> None:
        """Called from a finally block: EVERY research replica registers, no
        matter how it ended — an unregistered replica would leave the bare
        replicas waiting on the group event forever."""
        group = self._research_group((global_step, task_id))
        group["results"].append(entry)
        if len(group["results"]) >= group["n"]:
            group["event"].set()

    async def _await_research_group(
        self, global_step, task_id
    ) -> tuple[str | None, dict[str, Any]]:
        """Teacher selection over the research group: argmax measured doc
        utility p_i, ties to the shorter document; p*=0 admits no teacher
        (PG-5(3) input filtering, realised endogenously — a document that never
        helped downstream never reaches the distillation prefix)."""
        rcfg = self.cfg.research
        group = self._research_group((global_step, task_id))
        # Sessions run concurrently, so the group wall-clock is one session
        # plus its eval rollouts — the shared-session timeout plus slack.
        timeout = rcfg.session_timeout_s + 900.0
        try:
            await asyncio.wait_for(group["event"].wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "research group %s/%s incomplete after %.0fs: %d/%d registered — "
                "selecting over what arrived",
                global_step, task_id, timeout, len(group["results"]), group["n"],
            )
        # codex review P1-9: freeze the teacher decision on FIRST resolution.
        # Multiple bare/expert awaiters resolve concurrently; without a cache a
        # late-arriving registration between two timeouts could hand the same
        # task's replicas DIFFERENT documents. Same-event-loop invariant makes
        # the check-then-set below atomic (no await between them).
        cached = group.get("resolution")
        if cached is not None:
            doc, fields = cached
            return doc, dict(fields)
        results = list(group["results"])
        best = select_best_doc(results)
        doc = best["doc"] if best else None
        ps = [float(r.get("p", 0.0)) for r in results]
        # The historical research_* keys keep their curves alive under the
        # group regime, re-based on the WINNING document (the one training
        # actually distills from); research_final_success upgrades from "last
        # in-session experiment" to the measured K-rollout success of that
        # document — strictly closer to the PG-7 signal it always proxied.
        fields = {
            "research_error": sum(1 for r in results if r.get("error")),
            "research_experiments": float(best["experiments"]) if best else 0.0,
            "research_episodes": float(sum(r.get("episodes", 0) for r in results)),
            "research_turns": float(best["turns"]) if best else 0.0,
            "research_final_success": float(best["p"]) if best else 0.0,
            "doc_words": len(doc.split()) if doc else 0,
            "doc_missing": int(not doc),
            "research_rl_group_all_fail": int(bool(ps) and max(ps) <= 0.0),
            "research_rl_group_all_pass": int(bool(ps) and min(ps) >= 1.0),
            "research_rl_best_p": max(ps) if ps else 0.0,
            "research_archive": "",
        }
        # (group finalize rate is readable as research_rl_finalized /
        # research_rl_role from the curves; no extra field is shipped here)
        group["resolution"] = (doc, dict(fields))
        return doc, dict(fields)

    async def _research_replica(
        self, adapter, global_step, route, sampling_params, response_length
    ) -> AgentLoopOutput:
        """One research-role replica (docs/self-research-design.md 4):
        an independent researcher session ON THE ENGINE in token mode -> its
        document -> eval_k doc-conditioned rollouts -> p_i -> the research
        training sample. Registration into the group is unconditional
        (finally), so a dead session costs its own reward, never the batch."""
        metrics = AgentLoopMetrics()
        rcfg, rrl = self.cfg.research, self.cfg.research_rl
        rep_route = f"{route}:rr{uuid4().hex[:6]}"
        role = self.cfg.roles.get("researcher")
        chat = EngineResearcherTokenChat(
            self.server_manager,
            self.tokenizer,
            self.delta,
            temperature=role.temperature if role else 0.7,
            max_tokens=rcfg.researcher_max_tokens,
            max_sample_tokens=rrl.max_sample_tokens,
            request_id=rep_route,
        )
        router: ToolRouter | None = None
        session: ResearchSession | None = None
        result: SessionResult | None = None
        error: str | None = None
        doc: str | None = None
        wins = 0
        p = 0.0
        eval_summaries: list[dict[str, Any]] = []
        try:
            try:
                router = ToolRouter(
                    adapter=adapter,
                    research=rcfg,
                    narration=self.cfg.narration,
                    # Per-EPISODE routing keys, same rationale as _research().
                    generate_factory=lambda: EngineGenerate(
                        self.server_manager, self.tokenizer,
                        request_id=f"{rep_route}:r{uuid4().hex[:6]}",
                    ),
                    tokenizer=self.tokenizer,
                    delta=self.delta,
                    episode_response_budget=self.rollout_config.response_length,
                    narrate_fn=self._narrate_fn(route),
                )
                session = ResearchSession(
                    researcher_chat=chat,
                    toolproto=make_tool_protocol(rcfg.tool_protocol),
                    router=router,
                    system_prompt=build_researcher_system(
                        task_brief=adapter.task_brief(),
                        executor_model=self.executor_model_name,
                        n_exp=rcfg.n_exp,
                        doc_max_words=rcfg.doc_max_words,
                        gt_text=adapter.gt_text() if rcfg.gt == "on" else None,
                    ),
                    max_turns=rcfg.max_turns,
                    wrapup_buffer=rcfg.wrapup_buffer,
                )
                t0 = time.time()
                result = await asyncio.wait_for(
                    session.run(), timeout=rcfg.session_timeout_s
                )
                metrics.tool_calls += time.time() - t0
                error = result.error
            except asyncio.TimeoutError:
                error = f"research session exceeded {rcfg.session_timeout_s:.0f}s"
                if session is not None:
                    result = session.snapshot(error=error)
            except Exception as err:  # noqa: BLE001 — degrade, never raise
                error = f"{type(err).__name__}: {err}"
            if error:
                logger.warning(
                    "research replica degraded for %s: %s", adapter.task_id, error
                )
            doc = result.doc if result is not None else None

            # eval_k doc-conditioned rollouts: the document's measured utility.
            # An unfinalized session evaluates nothing and scores p=0 —
            # finalizing IS part of the researched skill (design 3.2-3).
            if doc:
                t0 = time.time()
                eval_results = await asyncio.gather(
                    *[
                        self._expert_rollout(
                            adapter, doc, sampling_params, response_length, rep_route
                        )
                        for _ in range(rrl.eval_k)
                    ],
                    return_exceptions=True,
                )
                metrics.generate_sequences += time.time() - t0
                for er in eval_results:
                    if isinstance(er, BaseException):
                        eval_summaries.append({"won": False, "error": str(er)})
                        continue
                    won = bool(er.trajectory.won)
                    wins += int(won)
                    eval_summaries.append(
                        {
                            "won": won,
                            "outcome": er.trajectory.outcome,
                            "env_steps": sum(1 for st in er.trajectory.steps if st.acted),
                        }
                    )
                p = wins / float(rrl.eval_k)
        finally:
            self._register_research_result(
                global_step,
                adapter.task_id,
                doc=doc,
                p=p,
                finalized=bool(doc),
                error=error,
                experiments=result.experiments_used if result else 0,
                episodes=result.episodes_consumed if result else 0,
                turns=result.turns_used if result else 0,
            )

        # Archive the session like the shared path does (same directory
        # contract; consumers key off researcher_model/tool_protocol).
        archive_path = ""
        records = list(router.state.episodes.values()) if router is not None else []
        try:
            path = await asyncio.to_thread(
                self.archive.save,
                task_id=adapter.task_id,
                result=result,
                records=records,
                policy_version=global_step,
                extra_meta={
                    "gt": rcfg.gt,
                    "gt_injected": bool(rcfg.gt == "on" and adapter.gt_text()),
                    "narration_backend": self.cfg.narration.backend,
                    "researcher_model": "policy(engine)",
                    "tool_protocol": rcfg.tool_protocol,
                    "error": error,
                    "research_rl": {
                        "p": p,
                        "wins": wins,
                        "eval_k": rrl.eval_k,
                        "evals": eval_summaries,
                    },
                },
            )
            archive_path = str(path.relative_to(self.run_dir))
        except Exception:  # noqa: BLE001
            logger.exception("research replica archive failed for %s", adapter.task_id)

        prompt_ids, response_ids, response_mask = chat.token_sample()
        dropped = ""
        if not response_ids:
            # The session died before its first generation. The trainer cannot
            # take an empty response (as_dict writes rm_scores[-1]), so the
            # slot borrows a bare rollout's tokens and masks them out entirely:
            # it still carries p into the group statistics, and nothing trains.
            placeholder = await self._student_rollout(
                adapter, sampling_params, response_length, rep_route
            )
            prompt_ids = placeholder.tokens.prompt_ids
            response_ids = placeholder.tokens.response_ids
            response_mask = [0] * len(response_ids)
            dropped = "empty"
        elif len(prompt_ids) + len(response_ids) > rrl.max_sample_tokens:
            # Defensive: the chat budget should make this unreachable.
            response_mask = [0] * len(response_ids)
            dropped = "overlen"

        turns_used = result.turns_used if result is not None else 0
        research_turn_spans: list[list[int]] = []
        span_start = None
        for pos, bit in enumerate(response_mask):
            if bit and span_start is None:
                span_start = pos
            elif not bit and span_start is not None:
                research_turn_spans.append([span_start, pos])
                span_start = None
        if span_start is not None:
            research_turn_spans.append([span_start, len(response_mask)])
        fields: dict[str, Any] = {
            "task_id": adapter.task_id,
            "sample_role": "research",
            "research_arm": 1,
            # Behaviour fields exist for dump/curve homogeneity; the trainer
            # excludes research rows from every on-policy behaviour metric.
            "won": False,
            "outcome": "research",
            "env_steps": 0,
            "invalid_format": 0,
            "truncated": int(bool(dropped)),
            "research_error": int(error is not None),
            "doc_words": len(doc.split()) if doc else 0,
            "doc_missing": int(not doc),
            "research_experiments": result.experiments_used if result else 0,
            "research_episodes": result.episodes_consumed if result else 0,
            "research_turns": turns_used,
            "research_final_success": p,
            "research_archive": archive_path,
            "research_rl_role": 1,
            "research_rl_reward": p,
            "research_rl_finalized": int(bool(doc)),
            "research_rl_traj_dropped": int(bool(dropped)),
            "turn_spans": research_turn_spans,
        }
        logger.info(
            "research replica %s: doc=%s p=%.3f (%d/%d) tokens=%d%s%s",
            adapter.task_id,
            "yes" if doc else "no",
            p,
            wins,
            rrl.eval_k,
            len(prompt_ids) + len(response_ids),
            f" dropped={dropped}" if dropped else "",
            " ERROR" if error else "",
        )
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            reward_score=p,
            num_turns=2 * turns_used + 1,
            metrics=metrics,
            extra_fields=fields,
        )
        output.extra_fields["reward_extra_info"] = build_curve_metrics(
            validate=False,
            research_enabled=True,
            mixed_sft=self.cfg.train.mix_sft_weight > 0
            and self.cfg.train.mix_sft_rollouts > 0,
            # codex tg-review P0-3: the curve key set must be batch-uniform —
            # verl takes the key list from the FIRST row and indexes every
            # other row with it, so a research row missing the tg_* keys
            # either KeyErrors the batch (ordinary first row) or silently
            # drops the tg curves (research first row).
            mixed_tg=self.cfg.train.mix_tg_weight > 0 or self.cfg.train.opd_lost_only,
            research_rl=True,
            fields=output.extra_fields,
        )
        try:
            await asyncio.to_thread(
                self._dump_rollout,
                step=global_step,
                validate=False,
                payload={
                    "task_id": adapter.task_id,
                    "sample_role": "research",
                    "global_step": global_step,
                    "p": p,
                    "wins": wins,
                    "eval_k": rrl.eval_k,
                    "evals": eval_summaries,
                    "doc_words": fields["doc_words"],
                    "doc_missing": fields["doc_missing"],
                    "error": error,
                    "dropped": dropped,
                    "turns_used": turns_used,
                    "response_token_len": len(response_ids),
                    "response_token_hash": response_token_hash(response_ids),
                    "research_archive": archive_path,
                    "ts": time.time(),
                },
            )
        except Exception:  # noqa: BLE001 — dumps must never fail a sample
            logger.exception("research replica dump failed for %s", adapter.task_id)
        return output

    async def _expert_rollout(
        self,
        adapter,
        doc: str,
        sampling_params: dict[str, Any],
        response_length: int,
        route: str,
    ) -> EpisodeResult:
        """One rollout WITH the document in context — the SFT arm's data source.

        Same episode runner, same protocol, same sampling as the bare student;
        the only difference is the guidance slot. Success rate on these is also
        the PG-7 document-quality signal (doc-injected rollout success).
        """
        epi_route = f"{route}:e{uuid4().hex[:6]}"
        return await run_episode(
            adapter=adapter,
            generate=EngineGenerate(self.server_manager, self.tokenizer, request_id=epi_route),
            tokenizer=self.tokenizer,
            delta=self.delta,
            guidance=doc,
            episode_id="expert",
            response_budget=response_length,
            sampling_params=sampling_params,
        )

    async def _student_rollout(
        self, adapter, sampling_params: dict[str, Any], response_length: int, route: str
    ) -> EpisodeResult:
        # Per-EPISODE routing key (2026-07-24, user ruling): the rollout_n
        # copies of a task must NOT share one replica — long-tail tasks starved
        # the other replica (measured idle). Unique suffix per episode; the
        # retry below reuses it so a retried episode stays on its replica.
        epi_route = f"{route}:s{uuid4().hex[:6]}"
        result = await run_episode(
            adapter=adapter,
            generate=EngineGenerate(self.server_manager, self.tokenizer, request_id=epi_route),
            tokenizer=self.tokenizer,
            delta=self.delta,
            guidance="",
            episode_id="student",
            response_budget=response_length,
            sampling_params=sampling_params,
        )
        if result.trajectory.outcome == OUTCOME_ENV_ERROR and not result.tokens.response_ids:
            logger.warning(
                "student episode infra error for %s (%s); retrying once",
                adapter.task_id,
                result.trajectory.error,
            )
            result = await run_episode(
                adapter=adapter,
                generate=EngineGenerate(self.server_manager, self.tokenizer, request_id=epi_route),
                tokenizer=self.tokenizer,
                delta=self.delta,
                guidance="",
                episode_id="student-retry",
                response_budget=response_length,
                sampling_params=sampling_params,
            )
        if result.trajectory.outcome == OUTCOME_ENV_ERROR and not result.tokens.response_ids:
            raise RuntimeError(
                f"student episode failed on infrastructure twice for {adapter.task_id}: "
                f"{result.trajectory.error}"
            )
        return result
