"""Trace-backed error evidence; a failed terminal outcome is not a diagnosis.

Semantic categories require a trusted adapter hook, not a researcher assertion.
The built-in loop detector reports observable repetition, not hidden causality.
"""
from __future__ import annotations

import hashlib
import json

from ..core.types import OUTCOME_ENV_ERROR, Trajectory

ERROR_TYPES = ("illegal_action", "missing_precondition", "wrong_choice", "repeated_loop")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def step_error(adapter, trajectory: Trajectory, index: int):
    """Return (category, reason, verifier) or None. index is zero-based.

    Optional adapter.diagnose_error(trajectory, index) returns None or
    {error_type, reason}. It must be a deterministic environment-specific
    check. It receives only the observed prefix, never future steps/labels.
    """
    step = trajectory.steps[index]
    hook = getattr(adapter, "diagnose_error", None)
    if hook is not None:
        prefix = Trajectory.from_dict(trajectory.to_dict())
        prefix.steps = prefix.steps[:index + 1]
        prefix.won, prefix.score, prefix.outcome = False, 0.0, "FAILED"
        evidence = hook(prefix, index)
        if evidence is not None:
            kind, reason = evidence["error_type"], evidence["reason"]
            if kind not in ERROR_TYPES or not isinstance(reason, str) or not reason.strip():
                raise ValueError("invalid adapter error diagnosis")
            return kind, reason, f"{type(adapter).__name__}.diagnose_error"
    if not step.acted or not step.valid:
        return ("illegal_action", "Unparseable action or adapter-reported invalid/no-op action.",
                "step.acted/step.valid")
    return None


def error_record(adapter, trajectory: Trajectory, *, error_id: str,
                 claimed_step=None, claimed_type=None, researcher_note="") -> dict:
    """Select the FIRST supported step; keep unsupported claims as uncertain."""
    found = None
    if trajectory.outcome != OUTCOME_ENV_ERROR:
        for index in range(len(trajectory.steps)):
            evidence = step_error(adapter, trajectory, index)
            if evidence:
                found = (index, evidence)
                break
    status = "verified" if found else "uncertain"
    index = found[0] if found else None
    if not found and claimed_step is None:
        # Repeated observations can hide real state changes (polling, gathering,
        # retries). Preserve the candidate pattern without inventing a diagnosis.
        for i in range(2, len(trajectory.steps)):
            step = trajectory.steps[i]
            if all(s.acted and s.action_norm == step.action_norm
                   and s.observation == step.observation for s in trajectory.steps[i - 2:i + 1]):
                index, claimed_type = i, "repeated_loop"
                break
    if claimed_step is not None:
        if type(claimed_step) is not int or not 1 <= claimed_step <= len(trajectory.steps):
            raise ValueError("step must reference a recorded 1-based turn")
        if claimed_type not in ERROR_TYPES:
            raise ValueError("unknown error_type")
        if not found or claimed_step != index + 1 or claimed_type != found[1][0]:
            status, index = "uncertain", claimed_step - 1
    step = trajectory.steps[index] if index is not None else None
    kind, reason, verifier = found[1] if status == "verified" else (
        claimed_type, "No independent evidence confirms this as the first verifiable error.", None)
    return {
        "error_id": error_id, "task_id": trajectory.task_id,
        "episode_id": trajectory.episode_id, "trajectory_sha256": digest(trajectory.to_dict()),
        "skill_hash": trajectory.guidance_hash, "task_requirements": adapter.task_brief(),
        "step": index + 1 if index is not None else None,
        "status": status, "error_type": kind, "reason": reason, "verifier": verifier,
        "observation_before": (trajectory.initial_obs if index == 0 else
                               trajectory.steps[index - 1].observation) if index is not None else None,
        "action": step.action_raw if step else None,
        "action_normalized": step.action_norm if step else None,
        "environment_feedback": step.observation if step else None,
        "researcher_note": researcher_note,
        "first_supported_step": found[0] + 1 if found else None,
    }


def has_error(adapter, trajectory, kind):
    return any((e := step_error(adapter, trajectory, i)) is not None and e[0] == kind
               for i in range(len(trajectory.steps)))
