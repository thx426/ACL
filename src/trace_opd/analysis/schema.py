"""Data contract of the distillation microscope (analysis-design.md §2).

These rows are produced by three sources — the training-side sampler (verl
patch, P-a), the offline turn-span join, and the probe tool (P-b) — and
consumed by `figures`. JSONL on disk, one dataclass per line.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, TypeVar

T = TypeVar("T")

# ------------------------------------------------------- training-side rows


@dataclass
class TurnProfile:
    """Per-(trajectory, turn) distillation aggregate — dumped for EVERY
    trajectory (cheap); the depth-terrain figure averages over traj."""

    step: int
    traj_id: str
    turn: int
    g_mean: float          # mean per-token KL(teacher || student), nats
    g_gated: float         # same but only tokens with student entropy > h0
    overlap: float         # mean top-k overlap ratio (Prune-OPD metric)
    h_teacher: float
    h_student: float
    n_tokens: int


@dataclass
class TokenRecord:
    """Token-level sample (a few trajectories per step)."""

    step: int
    traj_id: str
    t: int                 # position in response
    turn: int
    token_id: int
    kl: float
    logratio: float        # log p_T(y_t) - log p_S(y_t) at the sampled token
    h_t: float
    h_s: float
    top1_t: int
    top1_s: int
    in_overlap: bool
    token_type: str = ""   # offline annotation: action-verb / action-arg /
                           # reasoning-connective / format


# ------------------------------------------------------------- probe rows


@dataclass
class ProbePoint:
    """One probe sweep at training step k (evalm-side tool, exact full-vocab)."""

    step: int
    d_init: float          # KL(pi_k || pi_0)
    d_gain: float          # KL(pi_k + doc_k || pi_k)
    d_frozen: float        # KL(pi_k + doc_fix || pi_k)
    align: float = float("nan")         # cos(delta_doc, delta_train)
    align_robust: float = float("nan")  # uplift-set overlap (robust twin)
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)


# ------------------------------------------------------------ figure inputs


@dataclass
class LedgerPoint:
    step: int
    score: float           # bare sampled success
    priv_success: float    # doc-privileged success (research final)
    g: float               # residual privileged info, moving doc
    g_frozen: float        # frozen-doc reference


@dataclass
class ActionFlow:
    """Decision-point action-group dynamics on the probe set."""

    steps: list[int]
    student_probs: dict[str, list[float]]   # group -> prob per step (sums ~1)
    doc_uplift: dict[str, list[float]]      # group -> (teacher - student) prob


@dataclass
class FingerprintCase:
    """One decision point: does the training move match the doc's push?"""

    labels: list[str]          # candidate actions
    delta_doc: list[float]
    delta_train: list[float]
    context: str = ""          # short description for the caption


@dataclass
class ForkStats:
    steps: list[int]
    fork_rate: list[float]     # share of positions with teacher top-1 < 0.25
    lock_rate: list[float]     # share with top-1 > 0.65
    fork_uplift: dict[str, list[float]]  # action group doc-uplift at forks


# ------------------------------------------------------------------- jsonl


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path, cls: type[T]) -> list[T]:
    names = {f.name for f in fields(cls)}
    out: list[T] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(cls(**{k: v for k, v in d.items() if k in names}))
    return out
