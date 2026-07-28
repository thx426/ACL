"""Reasoning/action segment parsing over token pieces + per-segment aggregation.

The protocol contract (`<reasoning>...</reasoning><action>...</action>` per
assistant turn) is a TEXT-level structure; the packs store token ids. This
module maps tag regions to token-index spans by scanning the concatenated
piece stream (tags survive BPE splitting arbitrarily — the scan is on
characters, so how the tokenizer cut "<action>" is irrelevant), then
aggregates any per-token track by segment kind.

Pure functions over (pieces, row, tracks) — the tokenizer stays in server.py.
Token-level boundaries are approximate by at most one token where a piece
straddles a tag boundary; the observation protocol treats segment stats as
±1-token accurate, which is far below the effect sizes under study.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

TAGS = ("reasoning", "action")


def segment_spans(pieces: list[str], tags: tuple[str, ...] = TAGS) -> dict[str, Any]:
    """Map ``<tag>...</tag>`` regions to half-open token-index spans.

    Returns ``{tag: [[a, b], ...], "structure": [[a, b], ...],
    "closings": {tag: [tok_idx, ...]}}`` where ``closings[tag]`` holds the
    first token of each closing marker — the position whose realized
    probability the length-inflation observation (protocol item 3) reads.
    An unclosed open tag (generation truncated mid-segment) runs to the end
    and contributes no closing.
    """
    bounds = [0]
    for p in pieces:
        bounds.append(bounds[-1] + len(p))
    text = "".join(pieces)
    starts = np.asarray(bounds[:-1])
    ends = np.asarray(bounds[1:])

    def tok_range(c0: int, c1: int) -> tuple[int, int] | None:
        if c1 <= c0:
            return None
        a = int(np.searchsorted(ends, c0, side="right"))
        b = int(np.searchsorted(starts, c1, side="left"))
        return (a, b) if b > a else None

    out: dict[str, Any] = {tag: [] for tag in tags}
    out["structure"] = []
    out["closings"] = {tag: [] for tag in tags}
    for tag in tags:
        for m in re.finditer(rf"<{tag}>(.*?)(</{tag}>|$)", text, re.DOTALL):
            inner = tok_range(m.start(1), m.end(1))
            if inner:
                out[tag].append([inner[0], inner[1]])
            open_r = tok_range(m.start(0), m.start(1))
            if open_r:
                out["structure"].append([open_r[0], open_r[1]])
            if m.group(2):
                close_r = tok_range(m.start(2), m.end(2))
                if close_r:
                    out["structure"].append([close_r[0], close_r[1]])
                    out["closings"][tag].append(close_r[0])
    return out


def segment_mask(seg: dict[str, Any], kind: str, length: int) -> np.ndarray:
    sel = np.zeros(length, dtype=bool)
    for a, b in seg.get(kind, ()):
        sel[a:b] = True
    return sel


def aggregate_segments(
    row: dict[str, Any],
    tr: dict[str, np.ndarray],
    seg: dict[str, Any],
    clamp: float = -10.0,
    kinds: tuple[str, ...] = TAGS,
) -> dict[str, Any]:
    """Per-segment token-mean of the core tracks plus the KL budget share.

    ``kl_share`` is this segment's masked forward-KL SUM over the row's total
    masked sum — the budget-decomposition quantity of protocol item 1.
    ``clamp_hit_rate`` is the mean fraction of teacher-support entries whose
    student log-prob sits at/below the training clamp — where the loss shows
    KL but the gradient is dead (protocol item 6).
    """
    R = len(row["response_ids"])
    m_base = row["response_mask"].astype(bool) & row["covered"].astype(bool)
    kl = tr["kl_forward"]
    fin = np.isfinite(kl)
    denom_m = m_base & fin
    total_kl = float(kl[denom_m].sum()) if denom_m.any() else 0.0
    clamp_hits = (row["student_support_lps"] <= clamp + 1e-6).mean(-1)

    out: dict[str, Any] = {}
    for kind in kinds:
        m = m_base & segment_mask(seg, kind, R)
        entry: dict[str, Any] = {"tokens": int(m.sum())}
        if m.any():
            mk = m & fin
            entry["kl_mean"] = float(kl[mk].mean()) if mk.any() else None
            entry["kl_share"] = (
                float(kl[mk].sum() / total_kl) if mk.any() and total_kl > 0 else None
            )
            for name in ("student_entropy", "teacher_entropy", "teacher_mass",
                         "token_loss"):
                v = tr[name][m]
                v = v[np.isfinite(v)]
                entry[name] = float(v.mean()) if len(v) else None
            entry["clamp_hit_rate"] = float(clamp_hits[m].mean())
        out[kind] = entry
    return out


def closing_token_stats(
    row: dict[str, Any],
    tr: dict[str, np.ndarray],
    seg: dict[str, Any],
    kind: str = "reasoning",
) -> list[dict[str, Any]]:
    """Both sides' realized log-prob at each closing-marker first token.

    teacher_lp is None when the realized closing token fell outside the
    teacher's top-K — which for this observation is itself the loudest
    possible signal (the teacher pushes AWAY from closing the segment).
    """
    out = []
    for idx in seg.get("closings", {}).get(kind, ()):
        if idx >= len(row["response_ids"]):
            continue
        if not (row["response_mask"][idx] and row["covered"][idx]):
            continue
        t = tr["realized_teacher_lp"][idx]
        out.append(
            {
                "pos": int(idx),
                "teacher_lp": float(t) if np.isfinite(t) else None,
                "student_lp": float(row["realized_student_lp"][idx]),
            }
        )
    return out
