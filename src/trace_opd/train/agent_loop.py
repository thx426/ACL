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
from ..core.template_delta import TemplateDelta, render_ids
from ..core.types import OUTCOME_ENV_ERROR, inject_guidance
from ..docstore.archive import ResearchArchive
from ..envs.alfworld.adapter import AlfworldEnvFactory
from ..research.prompts import build_researcher_system
from ..research.session import ResearchSession, SessionResult
from ..research.toolproto import make_tool_protocol
from ..research.tools import ToolRouter
from .backends import EngineGenerate, engine_narrate_fn, engine_researcher_chat
from .curve import build_curve_metrics

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
        t_research0 = 0.0
        if self.cfg.research.enabled and not validate:
            t_research0 = time.time()
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

        t0 = time.time()
        result = await self._student_rollout(adapter, sampling_params, response_length, route)
        metrics.generate_sequences += time.time() - t0

        if research_fut is not None:
            # Only the *blocking* remainder is charged here; the part that ran
            # under the rollout is free. (Pre-overlap this metric was the full
            # research wall time — expect it to drop sharply.)
            t_block0 = time.time()
            doc, research_fields = await research_fut
            metrics.tool_calls += time.time() - t_block0
            del t_research0

        traj = result.trajectory
        env_steps = sum(1 for s in traj.steps if s.acted)
        output = AgentLoopOutput(
            prompt_ids=result.tokens.prompt_ids,
            response_ids=result.tokens.response_ids,
            response_mask=result.tokens.response_mask,
            reward_score=1.0 if traj.won else 0.0,
            num_turns=2 * result.assistant_turns + 1,
            metrics=metrics,
            extra_fields={
                "task_id": adapter.task_id,
                "won": traj.won,
                "outcome": traj.outcome,
                "env_steps": env_steps,
                "invalid_format": result.invalid_format,
                "truncated": result.truncated,
                **result.gen_extra,
                **research_fields,
            },
        )

        if not validate:
            # Teacher scoring joins the task's sticky replica too (verl pops
            # this key and uses it as the routing id for the scoring request —
            # the 8 copies then share one prefill of the doc prefix).
            output.extra_fields["teacher_routing_key"] = route
        if doc and not validate:
            teacher_prompt_ids = self._teacher_prompt_ids(
                adapter, doc, result.tokens.prompt_ids, traj.initial_obs
            )
            if teacher_prompt_ids is not None:
                output.extra_fields["teacher_prompt_ids"] = teacher_prompt_ids
                # K3 block-refresh scoring (2026-07-24, user-approved, W=4):
                # doc-as-prefix influence decays ~50x by turn 7 (measured), so
                # ship block boundaries (response offsets after every W-th
                # assistant turn) + the tokenized full doc; the verl scorer
                # inserts ONE fresh copy per block. Block 0 stays byte-
                # identical to the historical teacher (doc adjacent anyway).
                W = self.cfg.train.distill_block_turns
                spans = result.tokens.turn_spans
                if W > 0 and len(spans) > W:
                    doc_block_ids = self.tokenizer.encode(
                        "\n\n## Task guidance\n" + doc + "\n", add_special_tokens=False
                    )
                    fits = (
                        len(teacher_prompt_ids)
                        + len(result.tokens.response_ids)
                        + len(doc_block_ids)
                        + 8
                        <= self.cfg.train.rollout_max_model_len
                    )
                    if fits:
                        output.extra_fields["teacher_block_starts"] = [
                            spans[m][1] for m in range(W - 1, len(spans) - 1, W)
                        ]
                        output.extra_fields["teacher_doc_ids"] = doc_block_ids
                    # 不合长时静默退回单次打分(等价旧机制,非错误)
            else:
                output.extra_fields["doc_missing"] = 1  # overflow degraded to bare

        # Curve channel: numeric per-sample scalars -> swanlab (train:
        # rollout_extra/<key>/mean via verl f8f26d5; val: val-aux/...). After
        # the teacher step so doc_missing reflects overflow degradation.
        output.extra_fields["reward_extra_info"] = build_curve_metrics(
            validate=validate,
            research_enabled=self.cfg.research.enabled,
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
