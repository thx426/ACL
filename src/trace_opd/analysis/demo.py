"""Synthetic demo data + renderer for the six figures.

Purpose: pin the visual form and the data contract BEFORE the real data layer
(P-a/P-b) lands, and serve as the figure regression fixture. Every output is
stamped "synthetic demo" — repo rule: real figures only from real runs.

Usage: python -m trace_opd.analysis.demo --out <dir>
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from . import figures
from .schema import ActionFlow, FingerprintCase, ForkStats, LedgerPoint, ProbePoint, TokenRecord, TurnProfile
from .style import mark_synthetic, save

STEPS = list(range(0, 221, 10))  # val-point cadence over one epoch
_rng = random.Random(7)


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def ledger_points() -> list[LedgerPoint]:
    out = []
    for k in STEPS:
        breakthrough = _sig((k - 90) / 18)
        out.append(LedgerPoint(
            step=k,
            score=0.01 + 0.42 * breakthrough + _rng.uniform(-0.02, 0.02),
            priv_success=0.84 + _rng.uniform(-0.03, 0.03),
            g=max(0.05, 1.15 - 0.55 * breakthrough
                  + 0.08 * math.sin(k / 9) + _rng.uniform(-0.03, 0.03)),
            g_frozen=max(0.05, 0.34 + 0.85 * math.exp(-k / 70)
                         + _rng.uniform(-0.02, 0.02)),
        ))
    return out


def probe_arms() -> dict[str, list[ProbePoint]]:
    main, frozen_arm = [], []
    for k in STEPS:
        d_init = 0.85 * (1 - math.exp(-k / 90)) + _rng.uniform(-0.01, 0.01)
        g_frozen = max(0.05, 0.34 + 0.85 * math.exp(-k / 70))
        align = 0.12 + 0.45 * _sig((k - 80) / 25) + _rng.uniform(-0.05, 0.05)
        main.append(ProbePoint(
            step=k, d_init=d_init, d_gain=g_frozen * 1.1, d_frozen=g_frozen,
            align=align, align_robust=align * 0.8 + 0.08,
            ci={"align": (align - 0.07, align + 0.07)},
        ))
        # Frozen-teacher (E1) arm: consumes its static doc, then drifts —
        # the curve hooks back up (exhaustion signature).
        drift = max(0.0, (k - 120) / 100)
        frozen_arm.append(ProbePoint(
            step=k,
            d_init=1.0 * (1 - math.exp(-k / 80)) + drift * 0.3,
            d_gain=0.0,
            d_frozen=max(0.05, 0.30 + 0.9 * math.exp(-k / 45)) + drift * 0.55,
        ))
    return {"dynamic docs (ours)": main, "frozen teacher (E1)": frozen_arm}


def turn_profiles() -> list[TurnProfile]:
    out = []
    for k in STEPS:
        consumed = _sig((k - 70) / 30)
        front = 4 + 22 * consumed  # consumption wavefront moves deeper
        for turn in range(1, 31):
            base = 1.0 * math.exp(-turn / 16)          # front-loaded doc info
            eaten = 0.75 * consumed * _sig((front - turn) / 3)
            g = max(0.02, base * (1 - eaten) + _rng.uniform(-0.02, 0.02))
            for traj in ("a", "b"):
                out.append(TurnProfile(
                    step=k, traj_id=traj, turn=turn, g_mean=g,
                    g_gated=g * 1.25, overlap=0.8 + 0.1 * consumed,
                    h_teacher=1.1 - 0.3 * consumed,
                    h_student=1.4 - 0.5 * consumed, n_tokens=90,
                ))
    return out


def microscope_track() -> list[TokenRecord]:
    out = []
    t = 0
    for turn in range(1, 7):
        for i in range(20):
            is_action = i in (14, 15, 16)
            kind = ("action-verb" if i == 14 else "action-arg") if is_action else (
                "format" if i in (0, 19) else "reasoning-connective")
            ratio = (_rng.uniform(0.6, 1.6) if is_action and turn <= 3
                     else _rng.uniform(-0.15, 0.25))
            out.append(TokenRecord(
                step=90, traj_id="demo", t=t, turn=turn, token_id=1000 + t,
                kl=abs(ratio) * 0.5, logratio=ratio, h_t=1.0, h_s=1.3,
                top1_t=1, top1_s=1 if not is_action else 2,
                in_overlap=not is_action, token_type=kind,
            ))
            t += 1
    return out


def type_shares() -> dict[str, list[float]]:
    shares: dict[str, list[float]] = {t: [] for t in figures.TOKEN_TYPES}
    for k in STEPS:
        w = _sig((k - 80) / 30)
        raw = {
            "action-verb": 0.30 - 0.12 * w,
            "action-arg": 0.22 + 0.16 * w,
            "reasoning-connective": 0.38 - 0.06 * w,
            "format": 0.10 + 0.02 * w,
        }
        z = sum(raw.values())
        for t in shares:
            shares[t].append(raw[t] / z)
    return shares


def action_flow() -> ActionFlow:
    student: dict[str, list[float]] = {g: [] for g in ("move", "acquire", "inspect", "other")}
    uplift: dict[str, list[float]] = {g: [] for g in student}
    for k in STEPS:
        w = _sig((k - 85) / 25)
        raw = {"move": 0.14 + 0.34 * w, "acquire": 0.16 + 0.1 * w,
               "inspect": 0.55 - 0.42 * w, "other": 0.15 - 0.02 * w}
        z = sum(raw.values())
        for g in student:
            student[g].append(raw[g] / z)
        uplift["move"].append(0.30 * (1 - w) + 0.02)
        uplift["acquire"].append(0.05 + 0.13 * w * (1 - w) * 4 * 0.4)
        uplift["inspect"].append(-0.26 * (1 - w) - 0.03)
        uplift["other"].append(0.01)
    return ActionFlow(steps=STEPS, student_probs=student, doc_uplift=uplift)


def fingerprint_case() -> FingerprintCase:
    return FingerprintCase(
        labels=["go to countertop 1", "open drawer 2", "examine shelf 1",
                "take candle 1", "look"],
        delta_doc=[0.32, -0.18, -0.09, 0.05, -0.10],
        delta_train=[0.25, -0.12, -0.06, 0.08, -0.08],
        context='doc: "skip all drawers — the candle is on countertop 1"',
    )


def fork_stats() -> ForkStats:
    return ForkStats(
        steps=STEPS,
        fork_rate=[0.23 - 0.07 * _sig((k - 90) / 30) for k in STEPS],
        lock_rate=[0.48 + 0.13 * _sig((k - 90) / 30) for k in STEPS],
        fork_uplift={
            "move": [0.24 * (1 - _sig((k - 85) / 25)) + 0.03 for k in STEPS],
            "inspect": [-0.2 * (1 - _sig((k - 85) / 25)) - 0.02 for k in STEPS],
            "self-correction": [0.06 + 0.02 * _sig((k - 85) / 25) for k in STEPS],
        },
    )


def render_all(out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    made = []
    jobs = [
        ("fig_a_ledger", figures.fig_ledger(ledger_points(), events={90: "breakthrough"})),
        ("fig_b_trajectory", figures.fig_trajectory(probe_arms())),
        ("fig_c_terrain", figures.fig_depth_terrain(turn_profiles())),
        ("fig_d_microscope", figures.fig_token_microscope(
            microscope_track(), STEPS, type_shares())),
        ("fig_e_action_flow", figures.fig_action_flow(action_flow())),
        ("fig_f_fingerprint", figures.fig_fingerprint(
            probe_arms()["dynamic docs (ours)"], fingerprint_case())),
        ("fig_g_fork", figures.fig_fork(fork_stats())),
    ]
    for name, fig in jobs:
        mark_synthetic(fig)
        made += save(fig, out_dir / name)
    return made


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    for path in render_all(args.out):
        print(path)
    return 0


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    raise SystemExit(main())
