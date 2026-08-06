"""Pre-narration assertions: refuse to narrate what is not a narratable run.

Program-side classification BEFORE any narrator LLM call. An empty or
collapsed trajectory gets a structured verdict instead of a narration —
narrating nothing invites hallucination (hard prior-project lesson).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..core.types import OUTCOME_ENV_ERROR, Trajectory, is_degenerate


@dataclass
class NarrationSkip:
    kind: str    # env_error | degenerate | empty | all_unparseable
    detail: str

    def render(self) -> str:
        return f"NARRATION SKIPPED ({self.kind}): {self.detail}"


def classify(traj: Trajectory) -> NarrationSkip | None:
    """Return a skip verdict, or None when the trajectory is narratable."""
    if traj.outcome == OUTCOME_ENV_ERROR:
        return NarrationSkip(
            "env_error",
            f"environment/infrastructure fault, not a task attempt: {traj.error or 'unknown'}",
        )
    if is_degenerate(traj.outcome):
        return NarrationSkip(
            "degenerate",
            f"{traj.outcome} — {traj.error or 'executor output collapsed'}",
        )
    if traj.step_count == 0:
        return NarrationSkip("empty", "no steps were recorded")
    if all(not s.acted for s in traj.steps):
        return NarrationSkip(
            "all_unparseable",
            f"none of the {traj.step_count} replies contained a parseable <action> block",
        )
    return None
