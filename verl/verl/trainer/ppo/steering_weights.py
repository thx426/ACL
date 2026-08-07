# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""trace-opd PGS: expand the sparse steering ledger into dense per-token
behaviour-correction weights (W_beh), shipped as ``rollout_is_weights``.

The rollout side samples from the tail-preserving kernel b (NOT pi), so the
policy-gradient estimator needs the importance correction W_t = pi(y_t)/b(y_t)
per steered token. It rides verl's native ``rollout_is_weights`` channel: the
vanilla policy loss multiplies it OUTSIDE the PPO dual-clip (the separated
correction form — the clip keeps acting on the pi_theta/pi_old ratio alone).

Guarantees inherited from the sampling side (trace_opd steering package):
  * per-token |log W| <= 2*eta*c  (kernel density-ratio bound)
  * per-trajectory |sum log W| <= b_max  (D_inf budget, conservative charging)

Modes (ablation switch, spec §2.6; PRECISE definitions per review max-IF-14 —
the short names are configuration labels, not literature-faithful claims):
  per_token     W_t = pi/b at steered positions, 1 elsewhere (v3.0 default).
  per_decision  PDIS prefix product: W_t = prod_{s<=t} pi/b — the product up
                to position t applied to position t's score term ONLY. This
                is per-decision importance sampling, NOT full-trajectory IS
                (that would apply the whole-trajectory ratio to every term).
                Bounded by e^{±b_max}; heavy-tailed — ablation arm only.
  vtrace        min(1, pi/b) per token — the rho-bar=1 pointwise truncation
                borrowed from V-trace. NOT the full V-trace estimator (no
                value targets, no c-bar trace recursion); do not cite
                V-trace bias/contraction results for runs using it.
  shaped_reject v3.3 M3 (docs/pgs-v33-mixed-policy-design.md): rejection
                positions (teacher actually replaced the draft) get
                f(pi) = pi/(pi+gamma) — LUFFY's policy-shaping weight, which
                RAISES the gradient share of low-probability taught tokens
                instead of discounting them (the faithful pi/b weight
                suppresses exactly the positions where the teacher carries
                information — the s3 absorption bottleneck). Accepted
                positions stay at 1. This is a gradient-shaping weight, NOT
                an unbiased IS correction — do not cite unbiasedness for
                runs using it.
  ("off" never reaches this function: the caller skips the whole
  rollout_is_weights channel, MENTOR form — ratio denominator stays pi_old.)
"""
from __future__ import annotations

import numpy as np

__all__ = ["expand_behavior_weights", "WBEH_MODES"]

WBEH_MODES = ("per_token", "per_decision", "vtrace", "shaped_reject")


def expand_behavior_weights(
    *,
    positions: list[int],
    log_pi: list[float],
    log_b: list[float],
    response_len: int,
    mode: str = "per_token",
    log_ratio_cap: float | None = None,
    rejected: list[int] | None = None,
    shaped_gamma: float = 0.1,
) -> np.ndarray:
    """Sparse ledger rows -> dense per-token weights (float32, response_len).

    Args:
        positions: steered positions (0-based, strictly increasing, within
            the response coordinate frame the ledger recorded).
        log_pi / log_b: the ledger's per-position log pi(y) / log b(y).
        response_len: dense length (positions beyond it are dropped — the
            trainer's response window may truncate the tail).
        mode: one of WBEH_MODES.
        log_ratio_cap: optional hard bound on |log(pi/b)| per token; a ledger
            row exceeding it raises (the sampling-side kernel guarantees
            2*eta*c — a violation means corrupted accounting, fail loud).
    """
    if mode not in WBEH_MODES:
        raise ValueError(f"unknown w_beh mode {mode!r}, expected one of {WBEH_MODES}")
    if not (len(positions) == len(log_pi) == len(log_b)):
        raise ValueError("positions/log_pi/log_b must be equal length")
    weights = np.ones(response_len, dtype=np.float32)
    if not positions:
        return weights

    pos = np.asarray(positions, dtype=np.int64)
    if np.any(pos[1:] <= pos[:-1]):
        raise ValueError("ledger positions must be strictly increasing")
    if pos[0] < 0:
        raise ValueError("negative ledger position")
    log_ratio = np.asarray(log_pi, dtype=np.float64) - np.asarray(log_b, dtype=np.float64)
    if log_ratio_cap is not None and np.any(np.abs(log_ratio) > log_ratio_cap + 1e-9):
        raise ValueError(
            f"|log(pi/b)| exceeds the kernel bound {log_ratio_cap:.4f} — corrupted ledger"
        )

    keep = pos < response_len
    pos, log_ratio = pos[keep], log_ratio[keep]
    if mode == "per_token":
        weights[pos] = np.exp(log_ratio).astype(np.float32)
    elif mode == "vtrace":
        weights[pos] = np.minimum(1.0, np.exp(log_ratio)).astype(np.float32)
    elif mode == "shaped_reject":
        if rejected is None or len(rejected) != len(positions):
            raise ValueError(
                "shaped_reject needs a rejected flag per ledger row "
                f"(got {0 if rejected is None else len(rejected)} for "
                f"{len(positions)} rows) — ledger accounting incomplete"
            )
        if shaped_gamma <= 0:
            raise ValueError(f"shaped_gamma must be > 0, got {shaped_gamma}")
        rej = np.asarray(rejected, dtype=bool)[keep]
        # log_pi rows record the EXECUTED token's log pi — on a rejection row
        # that is the resampled (taught) token, exactly the pi that LUFFY's
        # f(pi) = pi/(pi+gamma) shaping wants.
        p = np.exp(np.asarray(log_pi, dtype=np.float64)[keep][rej])
        weights[pos[rej]] = (p / (p + shaped_gamma)).astype(np.float32)
    else:  # per_decision: prefix product carried forward to every later token
        running = np.zeros(response_len, dtype=np.float64)
        np.add.at(running, pos, log_ratio)
        weights = np.exp(np.cumsum(running)).astype(np.float32)
    return weights
