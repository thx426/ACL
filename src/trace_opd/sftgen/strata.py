"""Benchmark-specific task stratification for SFT sampling.

The pipeline body is benchmark-agnostic; what varies per benchmark is (a) the
env adapter (envs/<name>/) and (b) how tasks group into behavioural strata so
a fixed-size sample covers the behaviour space instead of whatever the parquet
ordering happens to front-load. Adding a benchmark = one stratum function here
plus its adapter package; the pipeline itself does not change.
"""
from __future__ import annotations

import random
from typing import Any, Callable

StratumFn = Callable[[dict[str, Any]], str]


def alfworld_stratum(item: dict[str, Any]) -> str:
    """ALFWorld task ids lead with the task type
    (``pick_and_place_simple-Box-None-Sofa-205/...``) — the six types are the
    natural behaviour strata (heat/cool/clean/look/two-object differ far more
    across than within)."""
    task_id = str(item.get("task_id", ""))
    head = task_id.split("-", 1)[0].strip()
    return head or "unknown"


STRATUM_FNS: dict[str, StratumFn] = {
    "alfworld": alfworld_stratum,
}


def stratified_sample(
    items: list[dict[str, Any]],
    n_tasks: int,
    seed: int,
    stratum_fn: StratumFn,
) -> list[dict[str, Any]]:
    """Deterministic stratified sample: shuffle within each stratum with a
    stratum-salted seed, then round-robin across strata until n_tasks are
    drawn. Equalises coverage; a stratum that runs dry simply drops out of the
    rotation. n_tasks >= population returns everything (stable order)."""
    by_stratum: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_stratum.setdefault(stratum_fn(item), []).append(item)
    for name, bucket in by_stratum.items():
        bucket.sort(key=lambda it: str(it.get("task_id", "")))
        random.Random(f"{seed}:{name}").shuffle(bucket)

    picked: list[dict[str, Any]] = []
    strata = sorted(by_stratum)
    cursors = {name: 0 for name in strata}
    while len(picked) < min(n_tasks, len(items)):
        progressed = False
        for name in strata:
            if len(picked) >= n_tasks:
                break
            bucket, cursor = by_stratum[name], cursors[name]
            if cursor < len(bucket):
                picked.append(bucket[cursor])
                cursors[name] = cursor + 1
                progressed = True
        if not progressed:
            break
    return picked
