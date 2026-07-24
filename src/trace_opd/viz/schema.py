"""Artifact readers + environment-agnostic trajectory analysis.

Source of truth for the on-disk layout: train/agent_loop.py::_dump_rollout
and docstore/archive.py (mirrored in docs/viz-design.md §2). Everything here
must stay generic: only structured protocol fields (valid/acted/won/outcome),
never environment text like "Nothing happens".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------ reading


def read_json(path: Path) -> dict[str, Any] | None:
    """A half-written or corrupt file must never break a scan — skip it."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def parse_step_dir(name: str) -> tuple[int, bool] | None:
    """rollouts/step_00009[_val] -> (9, is_val)."""
    if not name.startswith("step_"):
        return None
    core = name[len("step_"):]
    validate = core.endswith("_val")
    if validate:
        core = core[: -len("_val")]
    try:
        return int(core), validate
    except ValueError:
        return None


# ------------------------------------------------ generic trajectory analysis


@dataclass
class TrajStats:
    n_steps: int
    n_invalid: int      # acted env no-ops (valid=False, acted=True)
    n_noact: int        # format failures (acted=False)
    max_repeat: int     # longest consecutive run of one action_norm (acted only)
    distinct_actions: int


def traj_stats(traj: dict[str, Any]) -> TrajStats:
    steps = traj.get("steps") or []
    n_invalid = n_noact = 0
    max_repeat = repeat = 0
    prev = None
    distinct: set[str] = set()
    for s in steps:
        if not s.get("acted", True):
            n_noact += 1
            continue
        if not s.get("valid", False):
            n_invalid += 1
        act = s.get("action_norm", "")
        distinct.add(act)
        repeat = repeat + 1 if act == prev else 1
        max_repeat = max(max_repeat, repeat)
        prev = act
    return TrajStats(
        n_steps=len(steps),
        n_invalid=n_invalid,
        n_noact=n_noact,
        max_repeat=max_repeat,
        distinct_actions=len(distinct),
    )


def generic_step_tags(step: dict[str, Any], prev_action: str | None) -> list[str]:
    """Per-step render tags derivable from protocol fields alone."""
    tags: list[str] = []
    if not step.get("acted", True):
        tags.append("no_action")
    elif not step.get("valid", False):
        tags.append("invalid")
    if prev_action is not None and step.get("acted", True) and step.get("action_norm") == prev_action:
        tags.append("repeat")
    return tags


# ------------------------------------------------------------ research layer


def research_meta(session_dir: Path) -> dict[str, Any] | None:
    return read_json(session_dir / "meta.json")


def parse_research_dirname(name: str) -> int | None:
    """<ts>_v<step>[-k] -> step (None when unparseable / 'v?')."""
    if "_v" not in name:
        return None
    ver = name.rsplit("_v", 1)[1].split("-", 1)[0]
    try:
        return int(ver)
    except ValueError:
        return None
