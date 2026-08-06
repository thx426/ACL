"""The narrator: ONE global semantic narration per episode (LLM as the judge
of salience), guarded by program-side assertions, fact-card juxtaposition,
and a keyword-level outcome-conflict check.

Backend-agnostic: narration is a single-turn chat completion, expressed
against a NarrateFn so the default engine backend (the policy itself narrates
its own trajectories — user decision) and the external backend (a strong
model) share this code. The safety net NEVER parses narration structure; it
only checks the one enumerable, program-verifiable claim (success vs failure
wording) and raises a conflict flag for the researcher when the narration's
ending contradicts the engine verdict."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..common.tape import TAPE_CONTEXT
from ..core.types import OUTCOME_SUCCESS, Trajectory
from .assertions import NarrationSkip, classify
from .preprocess import RenderOpts, render_for_narrator

# (messages, temperature, max_tokens) -> completion text. Implementations:
# engine (train/backends.py: render + rollout-engine generate + decode) and
# external (common/llm.py client.chat wrapper). Must raise on failure.
NarrateFn = Callable[[list[dict[str, Any]], float, int], Awaitable[str]]

NARRATOR_SYSTEM = """\
You are a trajectory narrator. You receive one episode of an agent
attempting a task in an interactive environment, as a numbered list of
steps (the agent's thought, its action, the environment's response).

Write a faithful natural-language account of what happened, so that a
researcher who never saw the raw log can fully understand the attempt.

Rules:
- Cover the whole episode in order. Organize around what actually
  happened: key decisions, turning points, errors and the environment's
  responses to them. Routine uneventful steps may be summarized together;
  never silently drop events that changed the course of the attempt.
- Quote error or failure messages verbatim when short. Keep entity names
  exactly as they appear (e.g. "drawer 4", not "a drawer").
- Repeated behavior: state it as repetition with counts; do not omit it.
- Reconstruct, never comment: no advice, no praise or blame, no guesses
  about what the agent "should" have done. Judgment belongs to the reader.
- Do not invent anything not present in the log. If the log is unclear
  somewhere, say so plainly.
- Length in proportion to the trajectory; at most ~{max_words} words.
- End with one sentence stating the final result (success or failure,
  and the step count)."""


@dataclass
class NarrationResult:
    text: str | None
    skip: NarrationSkip | None = None
    conflict: bool = False
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.text is not None


async def narrate_episode(
    traj: Trajectory,
    narrate: NarrateFn,
    *,
    max_words: int = 600,
    temperature: float = 0.3,
    domain_notes: str = "",
    opts: RenderOpts | None = None,
) -> NarrationResult:
    skip = classify(traj)
    if skip is not None:
        return NarrationResult(text=None, skip=skip)

    system = NARRATOR_SYSTEM.format(max_words=max_words)
    if domain_notes.strip():
        system += f"\n\nEnvironment-specific notes on key events:\n{domain_notes.strip()}"
    log_text = render_for_narrator(traj, opts)
    token = TAPE_CONTEXT.set(f"narrate:{traj.task_id}:{traj.episode_id}")
    try:
        text = await narrate(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": log_text},
            ],
            temperature,
            # budget = narration itself (~1.3 tok/word) + headroom for models
            # that reason before writing
            max(1000, int(max_words * 3)),
        )
    except Exception as err:  # noqa: BLE001 — soft landing on the mechanical log
        return NarrationResult(text=None, error=f"{type(err).__name__}: {err}")
    finally:
        TAPE_CONTEXT.reset(token)

    text = (text or "").strip()
    if not text:
        return NarrationResult(text=None, error="narrator returned empty text")
    return NarrationResult(text=text, conflict=_outcome_conflict(text, traj))


_SUCCESS_PAT = re.compile(
    r"\b(succe(ss|ssful|ssfully|eded)|task (was )?complet(ed|e)|goal (was )?(reached|achieved|satisfied)|episode ended in success)\b",
    re.IGNORECASE,
)
_FAIL_PAT = re.compile(
    r"\b(fail(ed|ure)?|did not (complete|succeed|reach|achieve)|was not (completed|achieved|reached)|unsuccessful|without (succeeding|success))\b",
    re.IGNORECASE,
)


def _outcome_conflict(narration: str, traj: Trajectory) -> bool:
    """Conservative: only flag a CLEAR contradiction in the narration's
    ending (final ~2 sentences)."""
    tail = " ".join(narration.strip().split()[-60:])
    says_success = bool(_SUCCESS_PAT.search(tail))
    says_failure = bool(_FAIL_PAT.search(tail))
    if says_success == says_failure:  # ambiguous or silent → no flag
        return False
    truly_success = traj.outcome == OUTCOME_SUCCESS
    return says_success != truly_success
