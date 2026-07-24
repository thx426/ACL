"""ALFWorld observation plugin, v1.

Failure-stage taxonomy follows CD-006's hand analysis of trace_opd_v1
(docs/experiments/cd006-run-observations.md: never go-to / moved-but-no-take /
took-but-no-finish, plus the "exhaustive search" pattern). v1 is our
re-implementation of that criterion; alignment against CD-006's offline
script is requested via the collab changelog — bump `version` on any
criterion change so cached labels reclassify.
"""
from __future__ import annotations

from typing import Any

FAMILIES = (
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
)

# Verbs that reach/acquire; used only inside this plugin, never by the kernel.
_MOVE = "go to "
_TAKE = ("take ",)
_SEARCH_OPEN = ("open ",)
_EXHAUSTIVE_MIN_DESTS = 5


def _acted_valid(traj: dict[str, Any]):
    for s in traj.get("steps") or []:
        if s.get("acted", True) and s.get("valid", False):
            yield s.get("action_norm", "")


class AlfworldPlugin:
    name = "alfworld"
    version = "1"

    def matches(self, task_id: str) -> bool:
        return any(seg.startswith(FAMILIES) for seg in task_id.split("/"))

    def task_family(self, task_id: str) -> str:
        for seg in task_id.split("/"):
            for fam in FAMILIES:
                if seg.startswith(fam):
                    return fam
        return task_id.split("/", 1)[0]

    def classify_failure(self, traj: dict[str, Any]) -> str | None:
        moved = took = False
        for action in _acted_valid(traj):
            if action.startswith(_MOVE):
                moved = True
            elif action.startswith(_TAKE):
                took = True
        if took:
            return "took_no_finish"
        if moved:
            return "moved_no_take"
        return "never_moved"

    def step_tags(self, step: dict[str, Any]) -> list[str]:
        if not step.get("acted", True):
            return []
        action = step.get("action_norm", "")
        if action.startswith(_MOVE):
            return ["move"]
        if action.startswith(_TAKE):
            return ["take"]
        return []

    def traj_tags(self, traj: dict[str, Any]) -> list[str]:
        dests = {a[len(_MOVE):] for a in _acted_valid(traj) if a.startswith(_MOVE)}
        took = any(a.startswith(_TAKE) for a in _acted_valid(traj))
        if len(dests) >= _EXHAUSTIVE_MIN_DESTS and not took and not traj.get("won"):
            return ["exhaustive_search"]
        return []


from .plugins import register  # noqa: E402

register(AlfworldPlugin())
