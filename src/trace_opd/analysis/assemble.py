"""Assemble real-run data into figure inputs (the glue between the data layer
and figures.py). Every function reads only artifacts the pipeline already
produces: rollout dumps, collect outputs, probe sweeps/npz caches.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .figures import TOKEN_TYPES
from .schema import ActionFlow, ForkStats, LedgerPoint, ProbePoint, TokenRecord, TurnProfile, read_jsonl

# Action-group vocabulary for the flow/fork views. Environment-flavored by
# nature (ALFWorld verbs here) — adjust per environment; the assembly itself
# is generic over any {group: [first-words]} map.
ALFWORLD_ACTION_GROUPS: dict[str, list[str]] = {
    "navigate": ["go"],
    "acquire": ["take"],
    "place": ["move", "put"],
    "process": ["heat", "cool", "clean", "use", "toggle", "slice"],
    "inspect": ["examine", "look"],
    "open/close": ["open", "close"],
}


def action_group_map(tokenizer, groups: dict[str, list[str]]) -> dict[int, str]:
    """First-token id -> group, covering both leading-space and bare variants."""
    mapping: dict[int, str] = {}
    for group, words in groups.items():
        for word in words:
            for variant in (word, " " + word):
                ids = tokenizer.encode(variant, add_special_tokens=False)
                if ids:
                    mapping.setdefault(ids[0], group)
    return mapping


# ------------------------------------------------------------------ Fig A


def ledger_points(
    run_dir: Path,
    analysis_dir: Path | None = None,
    sweeps_path: Path | None = None,
) -> list[LedgerPoint]:
    """score/priv from dumps, G from the training-side microscope (top-k
    estimate), G_frozen from probe sweeps (exact, val-point cadence; NaN
    elsewhere — the figure breaks the line there)."""
    from ..viz.schema import parse_step_dir, read_json

    per_step: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"won": [], "priv": []}
    )
    for step_dir in sorted((run_dir / "rollouts").glob("step_*")):
        parsed = parse_step_dir(step_dir.name)
        if parsed is None or parsed[1]:  # train steps only
            continue
        for file in step_dir.glob("*.json"):
            d = read_json(file)
            if not d:
                continue
            acc = per_step[parsed[0]]
            acc["won"].append(float(bool(d.get("won"))))
            acc["priv"].append(float((d.get("research") or {}).get(
                "research_final_success") or 0.0))

    g_by_step: dict[int, float] = {}
    analysis_dir = analysis_dir or run_dir / "analysis"
    profile_path = analysis_dir / "turn_profiles.jsonl"
    if profile_path.exists():
        acc: dict[int, list[tuple[float, int]]] = defaultdict(list)
        for row in read_jsonl(profile_path, TurnProfile):
            acc[row.step].append((row.g_mean, row.n_tokens))
        g_by_step = {
            step: sum(g * n for g, n in vals) / max(1, sum(n for _, n in vals))
            for step, vals in acc.items()
        }
    frozen_by_step: dict[int, float] = {}
    if sweeps_path and sweeps_path.exists():
        for line in sweeps_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                frozen_by_step[int(d["step"])] = float(d.get("d_frozen", math.nan))

    mean = lambda xs: sum(xs) / len(xs) if xs else math.nan
    return [
        LedgerPoint(
            step=step,
            score=round(mean(acc["won"]), 5),
            priv_success=round(mean(acc["priv"]), 5),
            g=round(g_by_step.get(step, math.nan), 5),
            g_frozen=round(frozen_by_step.get(step, math.nan), 5),
        )
        for step, acc in sorted(per_step.items())
    ]


# ------------------------------------------------------- Fig E / Fig G


def _group_probs(logprobs, group_map: dict[int, str], groups: list[str]) -> dict[str, float]:
    import numpy as np

    probs = np.exp(logprobs)
    out = {g: 0.0 for g in groups + ["other"]}
    grouped = 0.0
    for token_id, group in group_map.items():
        if token_id < len(probs):
            out[group] += float(probs[token_id])
            grouped += float(probs[token_id])
    out["other"] = max(0.0, float(probs.sum()) - grouped)
    return out


def _sweep_steps(state_dir: Path) -> list[int]:
    return sorted(
        int(p.stem[len("bare_step"):]) for p in state_dir.glob("bare_step*.npz")
    )


def action_flow_from_sweeps(
    state_dir: Path, group_map: dict[int, str], groups: list[str] | None = None
) -> ActionFlow:
    from .probe import load_bare

    groups = groups or sorted(set(group_map.values()))
    steps = _sweep_steps(state_dir)
    student: dict[str, list[float]] = {g: [] for g in groups + ["other"]}
    uplift: dict[str, list[float]] = {g: [] for g in groups + ["other"]}
    for step in steps:
        bare = load_bare(state_dir / f"bare_step{step:05d}.npz")
        teacher_path = state_dir / f"teacher_step{step:05d}.npz"
        teacher = load_bare(teacher_path) if teacher_path.exists() else {}
        s_acc: dict[str, list[float]] = defaultdict(list)
        u_acc: dict[str, list[float]] = defaultdict(list)
        for pid, lp in bare.items():
            sp = _group_probs(lp, group_map, groups)
            for g, v in sp.items():
                s_acc[g].append(v)
            if pid in teacher:
                tp = _group_probs(teacher[pid], group_map, groups)
                for g in tp:
                    u_acc[g].append(tp[g] - sp[g])
        for g in student:
            student[g].append(sum(s_acc[g]) / max(1, len(s_acc[g])))
            uplift[g].append(sum(u_acc[g]) / max(1, len(u_acc[g])))
    return ActionFlow(steps=steps, student_probs=student, doc_uplift=uplift)


def fork_stats_from_sweeps(
    state_dir: Path,
    group_map: dict[int, str],
    groups: list[str] | None = None,
    fork_below: float = 0.25,
    lock_above: float = 0.65,
) -> ForkStats:
    import numpy as np

    from .probe import load_bare

    groups = groups or sorted(set(group_map.values()))
    steps = _sweep_steps(state_dir)
    fork_rate, lock_rate = [], []
    fork_uplift: dict[str, list[float]] = {g: [] for g in groups + ["other"]}
    for step in steps:
        bare = load_bare(state_dir / f"bare_step{step:05d}.npz")
        teacher_path = state_dir / f"teacher_step{step:05d}.npz"
        teacher = load_bare(teacher_path) if teacher_path.exists() else {}
        top1 = {pid: float(np.exp(lp).max()) for pid, lp in bare.items()}
        n = max(1, len(top1))
        fork_ids = {pid for pid, p in top1.items() if p < fork_below}
        fork_rate.append(len(fork_ids) / n)
        lock_rate.append(sum(1 for p in top1.values() if p > lock_above) / n)
        u_acc: dict[str, list[float]] = defaultdict(list)
        for pid in fork_ids & set(teacher):
            sp = _group_probs(bare[pid], group_map, groups)
            tp = _group_probs(teacher[pid], group_map, groups)
            for g in tp:
                u_acc[g].append(tp[g] - sp[g])
        for g in fork_uplift:
            fork_uplift[g].append(sum(u_acc[g]) / max(1, len(u_acc[g])))
    return ForkStats(steps=steps, fork_rate=fork_rate, lock_rate=lock_rate,
                     fork_uplift=fork_uplift)


# ------------------------------------------------------------------ Fig D


def type_shares(records: list[TokenRecord]) -> tuple[list[int], dict[str, list[float]]]:
    """Share of KL mass per token type per step (annotated records only)."""
    acc: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in records:
        if r.token_type:
            acc[r.step][r.token_type] += max(0.0, r.kl)
    steps = sorted(acc)
    shares: dict[str, list[float]] = {t: [] for t in TOKEN_TYPES}
    for step in steps:
        total = sum(acc[step].values()) or 1.0
        for t in TOKEN_TYPES:
            shares[t].append(acc[step].get(t, 0.0) / total)
    return steps, shares


def probe_points(sweeps_path: Path) -> list[ProbePoint]:
    out = []
    for line in sweeps_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            d["ci"] = {k: tuple(v) for k, v in (d.get("ci") or {}).items()}
            out.append(ProbePoint(**d))
    return sorted(out, key=lambda p: p.step)
