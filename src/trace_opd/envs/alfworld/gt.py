"""Expert-plan extraction for the GT switch (host side).

Two sources, tried in order (reliability validated at smoke time; if both
fail on real data, report — never fabricate a plan):
  1. ALFRED annotations: <task_dir>/traj_data.json → plan.high_pddl discrete
     actions ("GotoLocation(fridge)", "PickupObject(apple)", ...).
  2. Engine walkthrough: the worker's ``__GT__`` request (policy_commands),
     which yields playable text commands when the engine can produce them.
"""
from __future__ import annotations

import json
from pathlib import Path


def expert_plan_from_traj_data(data_root: str, gamefile_rel: str) -> str | None:
    traj_path = Path(data_root) / Path(gamefile_rel).parent / "traj_data.json"
    if not traj_path.exists():
        return None
    try:
        data = json.loads(traj_path.read_text(encoding="utf-8"))
        # Defensive against legal-but-odd annotations ("plan": null etc.) —
        # any shape surprise must fall through to the walkthrough fallback,
        # never abort the dataset build.
        plan = data.get("plan") if isinstance(data, dict) else None
        steps = plan.get("high_pddl") if isinstance(plan, dict) else None
        if not isinstance(steps, list):
            return None
        lines: list[str] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            da = step.get("discrete_action")
            if not isinstance(da, dict):
                continue
            action = str(da.get("action", "")).strip()
            if not action or action.lower() in {"noop", "end"}:
                continue
            raw_args = da.get("args")
            args = [str(a) for a in raw_args if str(a).strip()] if isinstance(raw_args, list) else []
            lines.append(f"{len(lines) + 1}. {action}({', '.join(args)})")
        if not lines:
            return None
        return "High-level expert plan (ALFRED annotation):\n" + "\n".join(lines)
    except (json.JSONDecodeError, OSError, TypeError, ValueError, AttributeError):
        return None


def render_walkthrough(plan: list[str] | None) -> str | None:
    if not plan:
        return None
    lines = [f"{i + 1}. {cmd}" for i, cmd in enumerate(plan)]
    return "Engine walkthrough (playable commands):\n" + "\n".join(lines)
