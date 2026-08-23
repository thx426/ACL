"""Researcher tool layer: budget accounting, episode launch, narration
pipeline, raw fetch, finalize soft-checks.

Division-of-labor discipline: every guard here is program logic over program
truth (counters, stored trajectories, enumerable keywords). Nothing parses
the structure of model-generated text; all refusals are SOFT (informative
messages, never session-killing errors).

Experiment episodes run through the ONE episode runner (core/episode.py)
against the injected GenerateFn — in training that is the verl rollout
engine, so the researcher experiments on exactly the policy being trained.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..common.config import NarrationCfg, ResearchCfg
from ..common.llm import ToolCall
from ..core.adapter import TaskAdapter
from ..core.episode import EpisodeResult, GenerateFn, run_episode
from ..core.template_delta import TemplateDelta
from ..core.types import (
    OUTCOME_ENV_ERROR,
    OUTCOME_SUCCESS,
    EvalResult,
    FactCard,
    Trajectory,
    counts_as_attempt,
    guidance_hash,
)
from ..narrate.factcard import build_fact_card
from ..narrate.narrator import NarrateFn, NarrationResult, narrate_episode
from ..narrate.preprocess import render_for_narrator, render_raw
from .versioning import attach_episode_policy_range

DOC_BLACKLIST = re.compile(
    r"\b(distill\w*|teacher|student|OPD|reward\w*|fine-?tun\w*|training)\b",
    re.IGNORECASE,
)
VERBATIM_WINDOW = 4  # consecutive actions counting as a pasted script
# 零正例会话里构成"把解法说成已确立"的断言措辞(2026-08-10 裁定;动词原形
# verify/confirm 放过 —— "could not verify" 一类否定语境降误伤)。
_ZERO_SUCCESS_CLAIM = re.compile(r"\b(verified|confirmed)\b", re.IGNORECASE)


@dataclass
class EpisodeRecord:
    trajectory: Trajectory
    fact_card: FactCard
    narration: NarrationResult | None  # None in narration backend "none"
    mechanical_log: str | None = None  # set when narration is absent/failed


@dataclass
class SessionState:
    experiments_used: int = 0
    episodes_consumed: int = 0
    fetch_used: int = 0
    episode_counter: int = 0
    episodes: dict[str, EpisodeRecord] = field(default_factory=dict)
    guidance_log: list[dict[str, Any]] = field(default_factory=list)
    last_exp_successes: int = 0
    last_exp_attempts: int = 0
    # 会话累计(2026-08-10 零正例裁定):文档可信度的机器判据 —— 零正例
    # (exp_successes_total == 0)的会话不允许交出"verified solution",
    # 下游按此加权/审计,不依赖模型自报。
    exp_successes_total: int = 0
    exp_attempts_total: int = 0
    finalized: bool = False
    doc: str | None = None
    self_assessment: str = ""
    flags: list[str] = field(default_factory=list)
    revision_requested: bool = False
    no_finalize: bool = False

    def next_episode_id(self) -> str:
        self.episode_counter += 1
        return f"ep{self.episode_counter:03d}"


class ToolRouter:
    """Binds one researcher session to its adapter, backends and state."""

    def __init__(
        self,
        *,
        adapter: TaskAdapter,
        research: ResearchCfg,
        narration: NarrationCfg,
        generate_factory: Callable[[], GenerateFn],
        tokenizer,
        delta: TemplateDelta,
        episode_response_budget: int,
        narrate_fn: NarrateFn | None,
    ) -> None:
        self.adapter = adapter
        self.cfg = research
        self.narration_cfg = narration
        self._generate_factory = generate_factory
        self._tokenizer = tokenizer
        self._delta = delta
        self._episode_budget = episode_response_budget
        self._narrate_fn = narrate_fn
        self.state = SessionState()
        self._seen_guidance_hashes: set[str] = set()

    # ------------------------------------------------------------ dispatch

    async def dispatch(self, call: ToolCall) -> str:
        try:
            if not call.arguments and call.raw_arguments.strip() not in ("", "{}", "null"):
                # The model DID pass arguments but they failed JSON parsing
                # (e.g. XML-parser fragments). Execute nothing, spend nothing.
                return (
                    f"TOOL ARGUMENTS ERROR: the arguments for '{call.name}' were "
                    "not valid JSON, so the call was NOT executed and no budget "
                    "was spent. Re-issue the call with well-formed arguments "
                    "(keep quotes and newlines inside string values escaped)."
                )
            if call.name == "run_experiment":
                return await self._run_experiment(call.arguments)
            if call.name == "fetch_raw":
                return self._fetch_raw(call.arguments)
            if call.name == "finalize":
                return self._finalize(call.arguments)
            # 角色纠正而非单纯报错:实测(2026-08-11,终端域 9B)调错工具
            # 几乎总是同一种——模型把自己当成 executor 去调环境工具(`bash`)。
            # 只说"未知工具"不纠正身份,模型下一轮还会再犯(每会话 1-2 个废轮)。
            return (
                f"Unknown tool '{call.name}' — you are the RESEARCHER, not the "
                "executor. You have no environment access of your own: you "
                "cannot run commands, read files, or edit anything directly. "
                "To make something happen in the environment, dispatch the "
                "executor with run_experiment(guidance=...) and read what comes "
                "back. Available tools: run_experiment, fetch_raw, finalize."
            )
        except Exception as err:  # noqa: BLE001 — tool faults must not kill the session
            return f"TOOL ERROR ({call.name}): {type(err).__name__}: {err}"

    # ------------------------------------------------------- run_experiment

    async def _run_experiment(self, args: dict[str, Any]) -> str:
        st = self.state
        if st.finalized:
            return "The session is already finalized; no further experiments."
        cfg = self.cfg
        if st.experiments_used >= cfg.n_exp:
            return (
                f"Experiment budget exhausted ({cfg.n_exp}/{cfg.n_exp} used). "
                "Please consolidate your observations and submit your experience "
                "document via finalize."
            )

        guidance = str(args.get("guidance") or "")
        k_req = args.get("k", cfg.k_default)
        try:
            k = int(k_req)
        except (TypeError, ValueError):
            k = cfg.k_default
        k_clamped = max(1, min(k, cfg.k_max))
        clamp_note = (
            f" (k adjusted from {k} to {k_clamped}; allowed range 1..{cfg.k_max})"
            if k_clamped != k
            else ""
        )

        st.experiments_used += 1
        exp_idx = st.experiments_used
        ghash = guidance_hash(guidance)
        st.guidance_log.append(
            {
                "ts": time.time(),
                "experiment_idx": exp_idx,
                "guidance_hash": ghash,
                "guidance": guidance,
                "episodes_before": st.episodes_consumed,
                "k": k_clamped,
                "new_version": ghash not in self._seen_guidance_hashes,
            }
        )
        self._seen_guidance_hashes.add(ghash)

        episode_ids = [st.next_episode_id() for _ in range(k_clamped)]
        results: list[EpisodeResult] = await asyncio.gather(
            *[
                run_episode(
                    adapter=self.adapter,
                    generate=self._generate_factory(),  # fresh request identity per episode
                    tokenizer=self._tokenizer,
                    delta=self._delta,
                    guidance=guidance,
                    episode_id=eid,
                    sample_idx=i,
                    response_budget=self._episode_budget,
                    sampling_params={"temperature": cfg.episode_temperature},
                )
                for i, eid in enumerate(episode_ids)
            ]
        )

        # 2026-07-25 (user-approved): narrate the k episodes CONCURRENTLY.
        # `_make_record` awaits `narrate_episode`, i.e. one local-policy LLM call
        # per episode. Doing that in a serial `for` loop made each experiment
        # round cost ~4x the narration latency while the k parallel episodes had
        # already finished — measured duty cycle 32% (per-task in-flight episodes
        # 1.2 vs k=4), which capped research-phase concurrency at ~19 even though
        # the launch peak reached 189. Narrations are independent (each reads only
        # its own trajectory) and `gather` preserves order, so this is a pure
        # scheduling change with identical results.
        records: list[EpisodeRecord] = list(
            await asyncio.gather(
                *[self._make_record_with_version(res) for res in results]
            )
        )
        for res, record in zip(results, records, strict=True):
            st.episodes[res.trajectory.episode_id] = record
            if counts_as_attempt(res.trajectory.outcome):
                st.episodes_consumed += 1

        env_errors = [r for r in records if r.trajectory.outcome == OUTCOME_ENV_ERROR]
        refunded = False
        if len(env_errors) == len(records):
            st.experiments_used -= 1
            refunded = True

        return self._render_experiment_result(
            exp_idx=exp_idx,
            ghash=ghash,
            records=records,
            clamp_note=clamp_note,
            refunded=refunded,
        )

    async def _make_record_with_version(self, result: EpisodeResult) -> EpisodeRecord:
        """把引擎返回的真实权重版本固化进trajectory审计证据。"""
        attach_episode_policy_range(result.trajectory, result.gen_extra)
        return await self._make_record(result.trajectory)

    async def _make_record(self, traj: Trajectory) -> EpisodeRecord:
        card = build_fact_card(traj)
        narration: NarrationResult | None = None
        mechanical: str | None = None
        if self._narrate_fn is None:
            # narration backend "none": the researcher reads the mechanical log
            mechanical = render_for_narrator(traj)
        else:
            narration = await narrate_episode(
                traj,
                self._narrate_fn,
                max_words=self.narration_cfg.max_words,
                temperature=self.narration_cfg.temperature,
                max_tokens=self.narration_cfg.max_tokens,
                domain_notes=self.adapter.domain_notes(),
            )
            if narration.text is None and narration.skip is None:
                # Narrator failed → soft landing on the mechanical log.
                mechanical = render_for_narrator(traj)
        return EpisodeRecord(
            trajectory=traj, fact_card=card, narration=narration, mechanical_log=mechanical
        )

    def _render_experiment_result(
        self,
        *,
        exp_idx: int,
        ghash: str,
        records: list[EpisodeRecord],
        clamp_note: str,
        refunded: bool,
    ) -> str:
        cfg = self.cfg
        st = self.state
        lines: list[str] = []
        remaining = cfg.n_exp - st.experiments_used
        lines.append(
            f"[Experiment {exp_idx}/{cfg.n_exp}] guidance={ghash} "
            f"k={len(records)}{clamp_note} | remaining experiments: {remaining}"
        )
        successes = 0
        attempts = 0
        for rec in records:
            traj = rec.trajectory
            lines.append(f"\n--- {traj.episode_id} ---")
            lines.append(rec.fact_card.render())
            ev = EvalResult(outcome=traj.outcome, won=traj.won, steps=traj.step_count)
            lines.append(ev.render())
            if counts_as_attempt(traj.outcome):
                attempts += 1
                successes += int(traj.outcome == OUTCOME_SUCCESS)
            if rec.narration is not None and rec.narration.text:
                lines.append("NARRATION:")
                lines.append(rec.narration.text)
                if rec.narration.conflict:
                    lines.append(
                        "!! CONFLICT FLAG: the narration's stated result contradicts "
                        "the program-verified outcome above. Trust the fact card; "
                        "use fetch_raw to inspect the raw log."
                    )
            elif rec.narration is not None and rec.narration.skip is not None:
                lines.append(rec.narration.skip.render())
            elif rec.mechanical_log is not None:
                label = (
                    "MECHANICAL LOG (no narrator configured):"
                    if self._narrate_fn is None
                    else "MECHANICAL LOG (narration unavailable for this episode):"
                )
                lines.append(label)
                lines.append(rec.mechanical_log)
        if attempts:
            st.last_exp_successes = successes
            st.last_exp_attempts = attempts
        st.exp_successes_total += successes
        st.exp_attempts_total += attempts
        summary = f"\nSummary: {successes}/{attempts} SUCCESS"
        if len(records) != attempts:
            summary += f" ({len(records) - attempts} episode(s) hit environment errors — not counted)"
        if refunded:
            summary += (
                ". All episodes failed on environment infrastructure; this "
                "experiment was refunded and does not count against your budget."
            )
        lines.append(summary)
        lines.append(f"Total episodes consumed: {st.episodes_consumed}.")
        return "\n".join(lines)

    # ------------------------------------------------------------ fetch_raw

    def _fetch_raw(self, args: dict[str, Any]) -> str:
        st = self.state
        cfg = self.cfg
        if st.fetch_used >= cfg.fetch_raw_limit:
            return (
                f"fetch_raw limit reached ({cfg.fetch_raw_limit} calls). "
                "Work from the narrations and fact cards you already have."
            )
        episode_id = str(args.get("episode_id") or "").strip()
        record = st.episodes.get(episode_id)
        if record is None:
            known = ", ".join(sorted(st.episodes)) or "(none yet)"
            return f"Unknown episode_id '{episode_id}'. Available episodes: {known}."
        step_range = args.get("step_range")
        rng: tuple[int, int] | None = None
        if isinstance(step_range, (list, tuple)) and len(step_range) == 2:
            try:
                rng = (int(step_range[0]), int(step_range[1]))
            except (TypeError, ValueError):
                rng = None
        st.fetch_used += 1
        text, effective = render_raw(record.trajectory, rng, cfg.fetch_raw_max_chars)
        note = ""
        if rng is not None and effective != rng:
            note = f"(step_range adjusted to t{effective[0]}..t{effective[1]})\n"
        return (
            f"{note}{text}\n\n{record.fact_card.render()}\n"
            f"(fetch_raw calls used: {st.fetch_used}/{cfg.fetch_raw_limit})"
        )

    # ------------------------------------------------------------- finalize

    def _finalize(self, args: dict[str, Any]) -> str:
        st = self.state
        if st.finalized:
            return "Already finalized."
        doc = str(args.get("experience_doc") or "").strip()
        self_assessment = str(args.get("self_assessment") or "").strip()
        if not doc:
            return "finalize requires a non-empty experience_doc."

        violations = self._check_doc(doc)
        if violations and not st.revision_requested:
            st.revision_requested = True
            listing = "\n".join(f"- {v}" for v in violations)
            return (
                "Before accepting, please revise the document once — the "
                "following issues were detected:\n"
                f"{listing}\n"
                "Submit the revised document via finalize. (If you resubmit "
                "unchanged, it will be accepted with these flags recorded.)"
            )
        st.finalized = True
        st.doc = doc
        st.self_assessment = self_assessment
        st.flags = violations
        note = " (flags recorded)" if violations else ""
        return f"Experience document accepted{note}. The session is complete."

    def _check_doc(self, doc: str) -> list[str]:
        violations: list[str] = []
        words = len(doc.split())
        limit = self.cfg.doc_max_words
        if words > limit:
            violations.append(f"document is {words} words (limit {limit}); please condense")
        hits = sorted({m.group(0).lower() for m in DOC_BLACKLIST.finditer(doc)})
        if hits:
            violations.append(
                "remove training/tooling vocabulary from the document: " + ", ".join(hits)
            )
        if self._verbatim_script_hit(doc):
            violations.append(
                f"document contains a pasted sequence of {VERBATIM_WINDOW}+ consecutive "
                "episode actions; express the approach as situation-level knowledge instead"
            )
        # 2026-08-10 用户裁定(零正例):一次成功都没有的会话不得交出
        # "verified" 解法 —— 程序化 enforce,不只靠 prompt 期望(SSB 实测
        # 194-19:零正例被旧硬格式逼出带 3 处 "verified" 的编造解法,
        # steered 0/20)。判据用机制侧累计 exp_successes_total,非模型自报;
        # 零实验直接 finalize 的会话同样零证据,一并适用。
        if self.state.exp_successes_total == 0:
            claim_hits = sorted({
                m.group(0).lower()
                for m in _ZERO_SUCCESS_CLAIM.finditer(doc)
            })
            if "# Understanding & Approach".lower() in doc.lower():
                claim_hits.append("'# Understanding & Approach' section")
            if claim_hits:
                violations.append(
                    "zero of this session's runs succeeded, but the document "
                    "presents a solution as established ("
                    + ", ".join(claim_hits)
                    + ") — rewrite in the zero-success layout ('# Current "
                    "Understanding & Ruled-Out Directions', '# Failure Modes "
                    "& Lessons', '# Facts & Reusable Notes') and state "
                    "observations without verified/confirmed claims"
                )
        return violations

    def _verbatim_script_hit(self, doc: str) -> bool:
        doc_lines = [
            re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", line).strip().lower().rstrip(".")
            for line in doc.splitlines()
        ]
        doc_lines = [l for l in doc_lines if l]
        if len(doc_lines) < VERBATIM_WINDOW:
            return False
        for rec in self.state.episodes.values():
            actions = [a for a in rec.trajectory.acted_actions() if a]
            for w in range(len(actions) - VERBATIM_WINDOW + 1):
                window = [a.lower() for a in actions[w : w + VERBATIM_WINDOW]]
                for d in range(len(doc_lines) - VERBATIM_WINDOW + 1):
                    if doc_lines[d : d + VERBATIM_WINDOW] == window:
                        return True
        return False
