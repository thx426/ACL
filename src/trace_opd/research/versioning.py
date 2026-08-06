"""动态 researcher 与 student 的 rollout policy 版本证据。"""
from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping
from typing import Any

from ..core.types import OUTCOME_ENV_ERROR


def _version_int(value: Any) -> int | None:
    value = getattr(value, "data", value)
    if isinstance(value, bool):
        return None
    try:
        return int(operator.index(value))
    except (TypeError, ValueError):
        return None


def attach_episode_policy_range(trajectory: Any, gen_extra: Mapping[str, Any]) -> None:
    """把引擎返回的版本区间固化进结构化 trajectory 证据。"""
    for key in ("min_global_steps", "max_global_steps"):
        if key not in gen_extra:
            continue
        raw = getattr(gen_extra[key], "data", gen_extra[key])
        parsed = _version_int(raw)
        trajectory.meta[key] = parsed if parsed is not None else repr(raw)


def research_policy_version(records: Iterable[Any]) -> int | None:
    """提取一个 research session 唯一且未跨越的真实 policy 版本。"""
    versions: set[int] = set()
    for record in records:
        trajectory = record.trajectory
        lo_raw = trajectory.meta.get("min_global_steps")
        hi_raw = trajectory.meta.get("max_global_steps")
        if lo_raw is None and hi_raw is None:
            if trajectory.outcome == OUTCOME_ENV_ERROR:
                continue
            raise RuntimeError(
                f"research episode {trajectory.episode_id} has no rollout policy version"
            )
        lo, hi = _version_int(lo_raw), _version_int(hi_raw)
        if lo is None or hi is None:
            raise RuntimeError(
                f"research episode {trajectory.episode_id} has malformed policy version "
                f"range {lo_raw!r}..{hi_raw!r}"
            )
        if lo != hi:
            raise RuntimeError(
                f"research episode {trajectory.episode_id} crossed policy versions "
                f"{lo}..{hi}"
            )
        versions.add(lo)
    if len(versions) > 1:
        raise RuntimeError(f"research session crossed policy versions {sorted(versions)}")
    return next(iter(versions)) if versions else None


def assert_rollout_policy_version(
    gen_extra: Mapping[str, Any], expected: int, *, label: str
) -> None:
    """拒绝一个 rollout 缺版本、跨版本或不匹配冻结 research policy。"""
    lo = _version_int(gen_extra.get("min_global_steps"))
    hi = _version_int(gen_extra.get("max_global_steps"))
    if lo is None or hi is None:
        raise RuntimeError(f"{label} has no policy version evidence")
    if lo != hi:
        raise RuntimeError(f"{label} crossed policy versions {lo}..{hi}")
    if lo != expected:
        raise RuntimeError(
            f"research/{label} policy mismatch: research={expected}, {label}={lo}"
        )
