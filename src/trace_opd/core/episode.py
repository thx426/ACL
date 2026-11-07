"""The ONE episode runner shared by research experiments, the training
student rollout, and the standalone evaluator (docs/design.md §4).

Drives: a GenerateFn  ×  an EnvHandle  →  append-only token stream
(+ response mask) and a structured Trajectory. The three call sites differ
only in the GenerateFn backend (verl rollout engine / external vLLM server),
the guidance text, the sampling temperature, and the response-token budget —
the contract (system prompt, tag parsing, limits, format-failure breaker,
stream construction) is this single implementation, so the behavior the
researcher observes, the behavior that gets trained, and the behavior that
gets evaluated are the same distribution by construction.

GenerateFn contract:
    async fn(prompt_ids: list[int], sampling_params: dict) -> GenResult
  - engine backend: GenResult.token_ids are the SAMPLED ids verbatim (the
    trained sequence must be byte-identical to what the policy sampled);
  - external backend: the completions API returns text with the stop token
    stripped; token_ids are the re-tokenization of that text
    (semantics-preserving; TemplateDelta.after_generation then appends the
    full glue including the close token).
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__file__)

from . import protocol
from .adapter import TaskAdapter
from .template_delta import TemplateDelta, render_ids
from .types import (
    DEGENERATE_PREFIX,
    OUTCOME_ENV_ERROR,
    OUTCOME_FAILED,
    OUTCOME_SUCCESS,
    Step,
    Trajectory,
    guidance_hash,
    inject_guidance,
)


@dataclass
class GenResult:
    token_ids: list[int]
    text: str
    extra: dict[str, Any] = field(default_factory=dict)


GenerateFn = Callable[[list[int], dict[str, Any]], Awaitable[GenResult]]


@dataclass
class EpisodeTokens:
    """Append-only training-layout stream (prompt | response with mask).

    turn_spans: [start, end) of every assistant generation segment within
    response_ids — the offline turn-bucketing key of the distillation
    microscope (analysis-design.md §2). Spans cover exactly the mask=1
    positions, clamped to the response budget.
    """

    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    turn_spans: list[list[int]] = field(default_factory=list)


@dataclass
class EpisodeResult:
    trajectory: Trajectory
    tokens: EpisodeTokens
    assistant_turns: int
    invalid_format: int
    truncated: bool
    gen_extra: dict[str, Any]  # first/last generation extras (e.g. policy version)
    env_steps_by_turn: list[int] = field(default_factory=list)

    @property
    def won(self) -> bool:
        return self.trajectory.won


async def run_episode(
    *,
    adapter: TaskAdapter,
    generate: GenerateFn,
    tokenizer,
    delta: TemplateDelta,
    guidance: str = "",
    episode_id: str = "ep",
    sample_idx: int = 0,
    response_budget: int,
    sampling_params: dict[str, Any],
    resume: tuple[EpisodeResult, int] | None = None,
) -> EpisodeResult:
    """Run one multi-turn episode. Environment faults yield outcome=ENV_ERROR
    (never raise) — the caller decides whether to refund (research), retry
    (training), or record (evaluation)."""
    ghash = guidance_hash(guidance)
    t0 = time.time()
    # The adapter is the scaffold source (non-ALFWorld adapters stay correct);
    # inject_guidance is a no-op for empty guidance.
    system_prompt = inject_guidance(adapter.executor_scaffold(), guidance)

    traj = Trajectory(
        episode_id=episode_id,
        task_id=adapter.task_id,
        guidance_hash=ghash,
        sample_idx=sample_idx,
        initial_obs="",
    )
    empty_tokens = EpisodeTokens(prompt_ids=[], response_ids=[], response_mask=[])

    # Boot with a leak guard: a session-timeout cancellation arriving while
    # the engine subprocess is booting must not orphan a live worker — the
    # to_thread keeps running, so hand the eventual handle to a closer.
    boot = asyncio.ensure_future(asyncio.to_thread(adapter.open_episode))
    try:
        env = await asyncio.shield(boot)
    except asyncio.CancelledError:
        boot.add_done_callback(_close_orphan_env)
        raise
    except Exception as err:  # noqa: BLE001 — env infra fault, not a task failure
        traj.outcome = OUTCOME_ENV_ERROR
        traj.error = f"open_episode failed: {type(err).__name__}: {err}"
        return EpisodeResult(traj, empty_tokens, 0, 0, False, {})

    stream: list[int] = []
    mask: list[int] = []
    turn_spans: list[list[int]] = []
    prompt_ids: list[int] = []
    assistant_turns = 0
    invalid_format = 0
    consecutive_bad = 0
    truncated = False
    won = False
    gen_extra: dict[str, Any] = {}
    env_steps_by_turn: list[int] = []
    # 轮末停止形态哨兵(2026-08-08 Qwen3.5 EOS 事故固化):clean=停在我们
    # 下发的轮末符;capped=满额截断;other=引擎用了停止集之外的符号停下
    # ——engine 后端(生产 rollout,停止 token 保留在输出)上 other 升高
    # 即 EOS 家族回归指纹(事故批次 82.7% 的轮停在 <|endoftext|>,该口径
    # 下全记 other);external 风格后端(completions API 剥停符再重
    # tokenize)天然落 other,无判别力。三计数随 traj.meta 进 dump。
    stop_clean = stop_capped = stop_other = 0

    try:
        initial_obs = env.initial_obs
        traj.initial_obs = initial_obs
        # Determinism invariant: the live engine must reproduce the dataset's
        # initial observation. A mismatch means environment drift — the
        # researcher would study one scene while the student runs another.
        # Fail loud as an infrastructure fault.
        expected_obs = adapter.initial_obs
        # 非确定环境(WebArena:任务指令静态、页面观测动态)经可选 hook
        # 自定义校验;无 hook 时保持逐字比较(确定性引擎的漂移检测)。
        validator = getattr(adapter, "validate_initial_obs", None)
        if validator is not None:
            if not validator(initial_obs):
                traj.outcome = OUTCOME_ENV_ERROR
                traj.error = ("initial_obs rejected by adapter validator "
                              "(environment drift?)")
                return EpisodeResult(traj, empty_tokens, 0, 0, False, {})
        elif expected_obs and initial_obs.strip() != expected_obs.strip():
            traj.outcome = OUTCOME_ENV_ERROR
            traj.error = "initial_obs mismatch vs dataset (environment drift?)"
            return EpisodeResult(traj, empty_tokens, 0, 0, False, {})
        prompt_ids = render_ids(
            tokenizer,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": protocol.truncate_obs(initial_obs)},
            ],
            True,
            tools=adapter.tools(),
        )

        env_steps = 0
        if resume is not None:
            source, before_step = resume
            prefix_turns = before_step - 1
            if (not getattr(adapter, "supports_exact_replay", False)
                    or source.trajectory.task_id != adapter.task_id
                    or not 0 <= prefix_turns < len(source.tokens.turn_spans)
                    or len(source.env_steps_by_turn) != len(source.trajectory.steps)):
                raise ValueError("exact replay is unsupported or prefix evidence is incomplete")
            if initial_obs != source.trajectory.initial_obs:
                raise ValueError("replay_drift: initial observation mismatch")
            for step in source.trajectory.steps[:prefix_turns]:
                if step.acted:
                    live, ended = await asyncio.to_thread(adapter.replay_action, env, step.action_norm)
                    def env_text(value):
                        return protocol.strip_tool_response(value).split("\n(note: ", 1)[0].strip()
                    if ended or env_text(live) != env_text(step.observation):
                        raise ValueError("replay_drift: prefix observation mismatch or early termination")
            cut = source.tokens.turn_spans[prefix_turns][0]
            stream = list(source.tokens.response_ids[:cut])
            # Local continuations are evaluation-only. Prefix tokens are never trained.
            mask = [0] * len(stream)
            assistant_turns = prefix_turns
            env_steps = source.env_steps_by_turn[prefix_turns - 1] if prefix_turns else 0
            for step in reversed(source.trajectory.steps[:prefix_turns]):
                if step.acted:
                    break
                consecutive_bad += 1
            traj.meta["resume"] = {"episode_id": source.trajectory.episode_id,
                                   "before_step": before_step, "prefix_tokens": cut,
                                   "env_steps_used": env_steps, "exact_replay": True}
        done = False
        while (
            env_steps < protocol.MAX_ENV_STEPS
            and assistant_turns < protocol.MAX_ASSISTANT_TURNS
        ):
            budget = response_budget - len(stream) - 64
            if budget < 32:
                truncated = True
                break
            turn_max = min(protocol.GEN_MAX_TOKENS_PER_TURN, budget)
            out = await generate(
                prompt_ids + stream,
                {
                    **sampling_params,
                    "max_tokens": turn_max,
                },
            )
            if out.extra.get("stop_reason") == "aborted":
                # Engine abort (e.g. weight-sync preemption) is an
                # infrastructure fault: research refunds it, training retries.
                traj.outcome = OUTCOME_ENV_ERROR
                traj.error = "generation aborted by the engine"
                truncated = True
                break
            gen_ids = list(out.token_ids)
            if not gen_ids:
                truncated = True
                break
            assistant_turns += 1
            if len(gen_ids) >= turn_max:
                stop_capped += 1
            elif gen_ids[-1] in delta.stop_ids:
                stop_clean += 1
            else:
                stop_other += 1
            turn_spans.append([len(stream), len(stream) + len(gen_ids)])
            stream += gen_ids
            mask += [1] * len(gen_ids)
            for key in ("min_global_steps",):
                if key in out.extra:
                    gen_extra.setdefault(key, out.extra[key])
            for key in ("max_global_steps",):
                if key in out.extra:
                    gen_extra[key] = out.extra[key]

            text = out.text
            # 交互协议整体归 adapter(react 抽象):动作语法、执行、评估
            # 反馈都是任务实现的事;core 只消费 ActionOutcome。
            outcome = await asyncio.to_thread(adapter.react, env, text)
            traj.steps.append(
                Step(
                    t=len(traj.steps) + 1,
                    thought=outcome.thought,
                    action_raw=outcome.action_raw,
                    action_norm=outcome.action_norm,
                    observation=outcome.feedback,
                    valid=outcome.valid,
                    acted=outcome.acted,
                )
            )
            env_steps_by_turn.append(env_steps + int(outcome.acted and outcome.consumed_env_step))
            if not outcome.acted:
                invalid_format += 1
                consecutive_bad += 1
                if consecutive_bad >= protocol.MAX_CONSECUTIVE_FORMAT_FAILURES:
                    truncated = True
                    traj.outcome = DEGENERATE_PREFIX + "format"
                    traj.error = "no parseable action repeatedly"
                    break
                obs_text = outcome.feedback
                done = False
            else:
                consecutive_bad = 0
                if outcome.consumed_env_step:
                    env_steps += 1
                obs_text = protocol.truncate_obs(outcome.feedback)
                done = outcome.done
                if done:
                    traj.score = outcome.score
                    won = outcome.score >= 1.0
            if done or env_steps >= protocol.MAX_ENV_STEPS:
                break

            glue = delta.after_generation(gen_ids)
            obs_ids = tokenizer.encode(obs_text, add_special_tokens=False)
            structure = glue + obs_ids + delta.post_obs
            # Reserve enough for the NEXT turn's budget check (64 slack + 32
            # minimum generation) — never leave a dangling mask=0 generation
            # prefix that no generation can follow.
            if len(stream) + len(structure) + 96 > response_budget:
                truncated = True
                break
            stream += structure
            mask += [0] * len(structure)

        if (
            not done
            and assistant_turns >= protocol.MAX_ASSISTANT_TURNS
            and env_steps < protocol.MAX_ENV_STEPS
        ):
            # The assistant-turn guard cap fired (format-failure turns inflated
            # the count) — that is a truncation, not a normal budget exhaustion.
            truncated = True

        traj.won = bool(won)
        if traj.outcome != OUTCOME_ENV_ERROR and not traj.outcome.startswith(
            DEGENERATE_PREFIX
        ):
            traj.outcome = OUTCOME_SUCCESS if won else OUTCOME_FAILED
        return EpisodeResult(
            trajectory=traj,
            tokens=EpisodeTokens(
                prompt_ids=prompt_ids,
                response_ids=stream[:response_budget],
                response_mask=mask[:response_budget],
                turn_spans=[
                    [s, min(e, response_budget)]
                    for s, e in turn_spans
                    if s < response_budget
                ],
            ),
            assistant_turns=assistant_turns,
            invalid_format=invalid_format,
            truncated=truncated,
            gen_extra=gen_extra,
            env_steps_by_turn=env_steps_by_turn,
        )
    except Exception as err:  # noqa: BLE001 — engine/transport fault mid-episode
        traj.outcome = OUTCOME_ENV_ERROR
        traj.error = f"{type(err).__name__}: {err}"
        return EpisodeResult(
            trajectory=traj,
            tokens=EpisodeTokens(
                prompt_ids=prompt_ids,
                response_ids=stream[:response_budget],
                response_mask=mask[:response_budget],
                turn_spans=[
                    [s, min(e, response_budget)]
                    for s, e in turn_spans
                    if s < response_budget
                ],
            ),
            assistant_turns=assistant_turns,
            invalid_format=invalid_format,
            truncated=truncated,
            gen_extra=gen_extra,
            env_steps_by_turn=env_steps_by_turn,
        )
    finally:
        traj.meta.setdefault("wall_s", round(time.time() - t0, 2))
        traj.meta.setdefault("guidance_chars", len(guidance or ""))
        traj.meta.setdefault("turns_stop_clean", stop_clean)
        traj.meta.setdefault("turns_len_capped", stop_capped)
        traj.meta.setdefault("turns_stop_other", stop_other)
        # Close must survive cancellation: shield keeps the close thread
        # running even when this task is already being cancelled.
        try:
            await asyncio.shield(asyncio.to_thread(env.close))
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass


def _close_orphan_env(boot: "asyncio.Future") -> None:
    """Done-callback for a cancelled boot: close the handle off-loop."""
    if boot.cancelled() or boot.exception() is not None:
        return
    env = boot.result()
    threading.Thread(target=lambda: _quiet_close(env), daemon=True).start()


def _quiet_close(env) -> None:
    try:
        env.close()
    except Exception:  # noqa: BLE001
        logger.debug("orphan env close failed", exc_info=True)
