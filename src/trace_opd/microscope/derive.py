"""Derived quantities over observation packs — the analysis layer.

Everything here is a pure function of the pack's raw quartet; NOTHING is read
from stored scalars except ``token_loss``, which is the training-side truth
anchor the ``audit`` function re-derives against. New analysis views belong
here: add a function, rebuild the cache, and every historical pack gains the
view — the packs themselves never change.

Formula parity: ``forward_kl`` reproduces the vendored
``compute_forward_kl_topk`` arithmetic (clamp both sides to
``log_prob_min_clamp``, sum p_T·(lp_T − lp_S) over the teacher support) and
``reverse_kl`` reproduces ``compute_reverse_kl_topk`` (member KL + tail
bucket off the RAW masses). Both are pinned against the vendored functions by
tests/test_microscope.py — if training arithmetic changes, the tests force
this file to follow.
"""
from __future__ import annotations

from typing import Any

import numpy as np

LOG_PROB_MIN_CLAMP = -10.0  # default only; every formula takes the PACK value


def _clamp(lp: np.ndarray, floor: float) -> np.ndarray:
    return np.maximum(lp, floor)


def forward_kl(row: dict[str, Any], clamp: float = LOG_PROB_MIN_CLAMP) -> np.ndarray:
    t = _clamp(row["teacher_lps"], clamp)
    s = _clamp(row["student_support_lps"], clamp)
    return (np.exp(t) * (t - s)).sum(-1)


def reverse_kl(row: dict[str, Any], clamp: float = LOG_PROB_MIN_CLAMP) -> np.ndarray:
    """Vendored compute_reverse_kl_topk, exactly: masses off the RAW
    log-probs, member KL on the clamped ones, and — codex review P1-15 —
    the tail floor applied to BOTH tail logs (the old derive clamped only
    the student side; a 0.99999-mass teacher against a 0.5-mass student put
    it ~0.756 off the training value). tail_floor follows the vendored
    None -> -80 convention."""
    tail_floor = clamp if clamp is not None else -80.0
    t_raw, s_raw = row["teacher_lps"], row["student_support_lps"]
    s_mass = np.clip(np.exp(s_raw).sum(-1), 0.0, 1.0 - 1e-6)
    t_mass = np.clip(np.exp(t_raw).sum(-1), 0.0, 1.0 - 1e-6)
    t, s = _clamp(t_raw, clamp), _clamp(s_raw, clamp)
    member = (np.exp(s) * (s - t)).sum(-1)
    s_tail = np.maximum(1.0 - s_mass, 1e-6)
    t_tail = np.maximum(1.0 - t_mass, 1e-6)
    log_s_tail = np.maximum(np.log(s_tail), tail_floor)
    log_t_tail = np.maximum(np.log(t_tail), tail_floor)
    tail = s_tail * (log_s_tail - log_t_tail)
    return member + tail


def realized_teacher_lp(row: dict[str, Any]) -> np.ndarray:
    """Teacher log-prob of the sampled token — NaN where the token fell
    outside the teacher's top-K (that absence is itself a signal)."""
    hit = row["teacher_ids"] == row["response_ids"][:, None]
    any_hit = hit.any(-1)
    picked = np.where(hit, row["teacher_lps"], 0.0).sum(-1)
    return np.where(any_hit, picked, np.nan)


def realized_ranks(row: dict[str, Any]) -> dict[str, np.ndarray]:
    """Rank (0-based) of the sampled token in each side's top-K; -1 = outside."""
    out = {}
    for side, ids_key in (("student", "student_ids"), ("teacher", "teacher_ids")):
        hit = row[ids_key] == row["response_ids"][:, None]
        rank = np.where(hit.any(-1), hit.argmax(-1), -1)
        out[f"{side}_rank"] = rank
    return out


def tracks(row: dict[str, Any], clamp: float = LOG_PROB_MIN_CLAMP) -> dict[str, np.ndarray]:
    """The per-token scalar tracks every view renders. All derived."""
    t_lp, s_own, s_sup = row["teacher_lps"], row["student_lps"], row["student_support_lps"]
    t_p = np.exp(_clamp(t_lp, clamp))
    s_own_p = np.exp(_clamp(s_own, clamp))
    overlap = (row["teacher_ids"][:, :, None] == row["student_ids"][:, None, :]).any(-1)
    r_t = realized_teacher_lp(row)
    out = {
        "kl_forward": forward_kl(row, clamp),
        "kl_reverse": reverse_kl(row, clamp),
        "token_loss": row["token_loss"],
        "teacher_mass": np.exp(t_lp).sum(-1),
        "student_mass_on_support": np.exp(s_sup).sum(-1),
        "student_mass_own": np.exp(s_own).sum(-1),
        "teacher_entropy": -(t_p * _clamp(t_lp, clamp)).sum(-1),
        "student_entropy": -(s_own_p * _clamp(s_own, clamp)).sum(-1),
        "overlap_count": overlap.sum(-1),
        "realized_student_lp": row["realized_student_lp"],
        "realized_teacher_lp": r_t,
        "gap": r_t - row["realized_student_lp"],
        "mask": row["response_mask"].astype(np.float32),
        "covered": row["covered"].astype(np.float32),
    }
    out.update({k: v.astype(np.float32) for k, v in realized_ranks(row).items()})
    return out


def divergence_label(row: dict[str, Any], pos: int) -> str:
    """Coarse per-position diagnosis for the inspector."""
    tr = tracks(row)
    if not row["covered"][pos] or not row["response_mask"][pos]:
        return "unmasked"
    t_ent, s_ent = tr["teacher_entropy"][pos], tr["student_entropy"][pos]
    outflow = 1.0 - tr["student_mass_own"][pos]
    if tr["kl_forward"][pos] < 0.05:
        return "agreement"
    if t_ent > 3.0:
        return "teacher-abstains"
    if outflow > 0.3:
        return "mass-outflow"
    t_arg = row["teacher_ids"][pos, 0]
    s_arg = row["student_ids"][pos, 0]
    if t_arg != s_arg and s_ent < 1.0 and t_ent < 1.0:
        return "verdict-conflict"
    return "student-unconverged"


def repeat_positions(row: dict[str, Any], n: int = 4, min_repeats: int = 2) -> np.ndarray:
    """Start positions of n-grams that already occurred earlier in the
    response — the token-level face of the thought-repetition signal."""
    ids = row["response_ids"]
    seen: dict[tuple, int] = {}
    hits = np.zeros(len(ids), dtype=bool)
    for i in range(len(ids) - n + 1):
        key = tuple(ids[i : i + n])
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= min_repeats:
            hits[i] = True
    return hits


def sample_summary(row: dict[str, Any]) -> dict[str, Any]:
    """Listing-page scalars, computed WITHOUT the full tracks() sweep
    (CD-006 req 3-Z: the old path burned 500+ s per step computing 14
    per-token tracks — one of them an O(L·K²) broadcast — to keep four
    means). Key set and semantics are unchanged, except kl_mean/kl_max now
    read the TRAINED objective (the token_loss truth anchor, i.e. reverse
    values under reverse_kl_topk) instead of always re-deriving forward KL —
    strictly more honest for a listing of what trained."""
    m = row["response_mask"].astype(bool) & row["covered"].astype(bool)
    loss = row["token_loss"]
    lm = m & np.isfinite(loss)
    kl_mean = float(loss[lm].mean()) if lm.any() else 0.0
    kl_max = float(loss[lm].max()) if lm.any() else 0.0

    r_t = realized_teacher_lp(row)  # O(L·K)
    gap = r_t - row["realized_student_lp"]
    gm = m & np.isfinite(gap)
    gap_mean = float(gap[gm].mean()) if gm.any() else 0.0

    t_clamped = _clamp(row["teacher_lps"], LOG_PROB_MIN_CLAMP)
    t_ent = -(np.exp(t_clamped) * t_clamped).sum(-1)  # O(L·K)
    teacher_entropy_mean = float(t_ent[m].mean()) if m.any() else 0.0

    overlap_mean = 0.0
    idx = np.nonzero(m)[0]
    if len(idx):
        t_ids, s_ids = row["teacher_ids"], row["student_ids"]
        total = 0
        for i in idx:  # per-position isin: O(L·K log K), not O(L·K²)
            total += int(np.isin(t_ids[i], s_ids[i]).sum())
        overlap_mean = total / float(len(idx))

    return {
        "resp_hash": row["resp_hash"],
        "step": row["step"],
        "task_id": row["task_id"],
        "sample_role": row["sample_role"],
        "won": row["won"],
        "reward": row["reward"],
        "resp_len": int(len(row["response_ids"])),
        "masked_tokens": int(m.sum()),
        "kl_mean": kl_mean,
        "kl_max": kl_max,
        "gap_mean": gap_mean,
        "overlap_mean": overlap_mean,
        "teacher_entropy_mean": teacher_entropy_mean,
        "turns": len(row["turn_spans"]),
    }


def audit_row(
    row: dict[str, Any], atol: float = 0.05, loss_mode: str = "forward_kl_topk",
    clamp: float = LOG_PROB_MIN_CLAMP, max_exceed_frac: float = 0.005,
) -> dict[str, Any]:
    """Re-derive the TRAINED objective (per the pack's recorded loss_mode)
    from the quartet and compare against the training-side truth anchor
    (fp16 both ways → tolerance ~5e-2).

    Verdict is on the EXCEED FRACTION, not the per-row max (CD-006 req 3-V):
    the reverse tail term ``s_tail·(log s_tail − log t_tail)`` loses 4-5
    significant digits when teacher mass ≈ 1−1e-4 survives an fp16 round
    trip — a handful of such tokens in a 3000-token row is a storage-precision
    artefact, not a data fault (measured: 0.1% of tokens failed 89% of rows).
    A row passes when at most ``max_exceed_frac`` of its checked tokens
    exceed ``atol``; the exceed count/fraction are reported so a reader never
    mistakes 3/3000 for a broken row."""
    m = row["response_mask"] & row["covered"] & np.isfinite(row["token_loss"])
    if not m.any():
        # codex review P1-16: absent data is a FAILURE, not a pass — a gate
        # reporting OK on zero checked tokens hides every upstream loss.
        return {"checked": 0, "max_err": 0.0, "n_exceed": 0, "exceed_frac": 0.0,
                "ok": False}
    if loss_mode not in ("forward_kl_topk", "reverse_kl_topk"):
        return {"checked": 0, "max_err": float("inf"), "n_exceed": 0,
                "exceed_frac": 1.0, "ok": False}
    formula = reverse_kl if loss_mode == "reverse_kl_topk" else forward_kl
    rederived = formula(row, clamp)[m]
    truth = row["token_loss"][m]
    err = np.abs(rederived - truth)
    n_exceed = int((err > atol).sum())
    frac = n_exceed / float(len(err))
    return {
        "checked": int(m.sum()), "max_err": float(err.max()),
        "n_exceed": n_exceed, "exceed_frac": round(frac, 6),
        "ok": bool(frac <= max_exceed_frac),
    }
