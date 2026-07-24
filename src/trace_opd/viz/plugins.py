"""Observation plugins — the ONLY place environment semantics may live.

The kernel indexes/renders any environment through protocol fields alone;
a plugin adds domain interpretation (failure-stage classifier, task-family
parsing, domain step tags). No plugin matching a task is a supported state:
every kernel feature still works, failures just stay "unclassified".

Classifier outputs are cached in the index together with (name, version);
bumping a plugin's version triggers reclassification on the next scan.
"""
from __future__ import annotations

from typing import Any, Protocol

UNCLASSIFIED = "unclassified"


class ObservationPlugin(Protocol):
    name: str
    version: str

    def matches(self, task_id: str) -> bool: ...

    def task_family(self, task_id: str) -> str: ...

    def classify_failure(self, traj: dict[str, Any]) -> str | None:
        """Failure-stage label for a lost trajectory (won ones are never asked)."""
        ...

    def step_tags(self, step: dict[str, Any]) -> list[str]: ...

    def traj_tags(self, traj: dict[str, Any]) -> list[str]:
        """Whole-trajectory pattern tags (e.g. exhaustive search)."""
        ...


_REGISTRY: list[ObservationPlugin] = []


def register(plugin: ObservationPlugin) -> None:
    _REGISTRY.append(plugin)


def plugin_for(task_id: str) -> ObservationPlugin | None:
    for p in _REGISTRY:
        if p.matches(task_id):
            return p
    return None


def task_family(task_id: str) -> str:
    p = plugin_for(task_id)
    if p is not None:
        return p.task_family(task_id)
    # Degraded heuristic: first path segment names the task group well enough
    # to group curves by; a real plugin refines this.
    return task_id.split("/", 1)[0] if task_id else ""


def classifier_version(task_id: str) -> str:
    p = plugin_for(task_id)
    return f"{p.name}@{p.version}" if p is not None else "none@0"


def classify(task_id: str, traj: dict[str, Any], won: bool) -> str:
    if won:
        return "success"
    p = plugin_for(task_id)
    label = p.classify_failure(traj) if p is not None else None
    return label or UNCLASSIFIED


# Built-in plugins self-register on import.
from . import plugins_alfworld  # noqa: E402,F401
