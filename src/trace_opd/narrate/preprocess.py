"""Input-side mechanical preprocessing for narration and raw views.

Program-side, semantics-free transforms only (division-of-labor principle):
fold obvious repetition, bound per-item length with visible truncation
markers, number the steps. The narrator LLM receives a clean, bounded log;
what is IMPORTANT stays its judgment, not ours.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..core.types import Trajectory


@dataclass
class RenderOpts:
    obs_max_chars: int = 400
    thought_max_chars: int = 300
    fold_min: int = 3  # fold runs of >= fold_min identical (action, observation)


RAW_OPTS = RenderOpts(obs_max_chars=1200, thought_max_chars=800, fold_min=10**9)


def truncate_middle(text: str, max_chars: int) -> str:
    text = "" if text is None else str(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    omitted = len(text) - head - tail
    return text[:head] + f" [...{omitted} chars truncated...] " + text[-tail:]


def _step_lines(traj: Trajectory, opts: RenderOpts, include_thought: bool) -> list[str]:
    lines: list[str] = []
    steps = traj.steps
    i = 0
    while i < len(steps):
        s = steps[i]
        # fold runs of identical (action_norm, observation) acted steps
        j = i
        while (
            j + 1 < len(steps)
            and steps[j + 1].acted
            and s.acted
            and steps[j + 1].action_norm == s.action_norm
            and steps[j + 1].observation == s.observation
        ):
            j += 1
        run_len = j - i + 1
        if s.acted:
            desc = f"[t{s.t}] action: {s.action_norm} -> {truncate_middle(s.observation, opts.obs_max_chars)}"
        else:
            desc = f"[t{s.t}] (reply contained no executable action) -> {truncate_middle(s.observation, opts.obs_max_chars)}"
        if run_len >= opts.fold_min:
            desc = (
                f"[t{s.t}-t{steps[j].t}] action repeated {run_len}x: {s.action_norm} "
                f"-> {truncate_middle(s.observation, opts.obs_max_chars)}"
            )
            if include_thought and s.thought:
                desc += f"\n    thought(first of run): {truncate_middle(s.thought, opts.thought_max_chars)}"
            lines.append(desc)
            i = j + 1
            continue
        if include_thought and s.thought:
            desc = (
                f"[t{s.t}] thought: {truncate_middle(s.thought, opts.thought_max_chars)}\n"
                f"    action: {s.action_norm if s.acted else '(none)'} -> "
                f"{truncate_middle(s.observation, opts.obs_max_chars)}"
            )
        lines.append(desc)
        i += 1
    return lines


def render_for_narrator(traj: Trajectory, opts: RenderOpts | None = None) -> str:
    """Numbered, folded, bounded log — the narrator's whole input."""
    opts = opts or RenderOpts()
    parts = [f"INITIAL OBSERVATION:\n{truncate_middle(traj.initial_obs, 1200)}", ""]
    parts.extend(_step_lines(traj, opts, include_thought=True))
    parts.append("")
    parts.append(f"FINAL RESULT (engine-verified): {traj.outcome}, steps={traj.step_count}")
    return "\n".join(parts)


def render_raw(
    traj: Trajectory,
    step_range: tuple[int, int] | None,
    char_budget: int,
) -> tuple[str, tuple[int, int]]:
    """fetch_raw view: unfolded, generous per-item bounds, hard char budget.

    Returns (text, effective_range). Range is clamped, never an error."""
    n = traj.step_count
    if n == 0:
        return "(empty trajectory: no steps were recorded)", (0, 0)
    lo, hi = (1, n) if step_range is None else step_range
    lo = max(1, min(int(lo), n))
    hi = max(lo, min(int(hi), n))
    sub = Trajectory(
        episode_id=traj.episode_id,
        task_id=traj.task_id,
        guidance_hash=traj.guidance_hash,
        sample_idx=traj.sample_idx,
        initial_obs=traj.initial_obs,
        steps=traj.steps[lo - 1 : hi],
        outcome=traj.outcome,
        won=traj.won,
    )
    lines = _step_lines(sub, RAW_OPTS, include_thought=True)
    header = f"RAW LOG of {traj.episode_id}, steps t{lo}..t{hi} (of {n} total):"
    body = "\n".join([header, *lines])
    if len(body) > char_budget:
        body = body[:char_budget] + (
            f"\n[output cut at {char_budget} chars — request a narrower step_range]"
        )
    return body, (lo, hi)
