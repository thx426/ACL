"""Fact card: program-held truth placed NEXT TO every narration."""
from __future__ import annotations

from ..core.types import FactCard, Trajectory
from .preprocess import truncate_middle


def build_fact_card(traj: Trajectory) -> FactCard:
    invalid = sum(1 for s in traj.steps if s.acted and not s.valid)
    not_acted = sum(1 for s in traj.steps if not s.acted)

    best_action, best_run = "", 0
    run_action, run_len = None, 0
    for s in traj.steps:
        key = s.action_norm if s.acted else None
        if key is not None and key == run_action:
            run_len += 1
        else:
            run_action, run_len = key, 1 if key is not None else 0
        if key is not None and run_len > best_run:
            best_action, best_run = key, run_len

    final_obs = traj.steps[-1].observation if traj.steps else "(none)"
    return FactCard(
        outcome=traj.outcome,
        steps=traj.step_count,
        invalid_count=invalid,
        not_acted_count=not_acted,
        longest_repeat_action=best_action,
        longest_repeat_count=best_run,
        final_obs_excerpt=truncate_middle(final_obs.strip().replace("\n", " "), 120),
    )
