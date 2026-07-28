"""Observation-pack reader: npz packs → per-sample records.

A pack directory looks like ``obs/step_00012/rank{r}.npz``; each file holds
this rank's sequence segments of every row it trained (under sequence
parallel a row's response window is split across ranks). The reader
reassembles rows into full response-window arrays and exposes a small,
allocation-light API the derive/server layers build on.

Pure numpy + stdlib — runs on any machine the packs are copied to.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import numpy as np

QUARTET = (
    "teacher_ids", "teacher_lps", "student_ids", "student_lps",
    "student_support_lps",
)


def list_steps(obs_dir: Path) -> list[int]:
    return sorted(
        int(p.name.split("_")[1])
        for p in Path(obs_dir).glob("step_*")
        if p.is_dir() and any(p.glob("rank*.npz"))
    )


class StepPack:
    """All rows of one training step, reassembled across ranks."""

    def __init__(self, obs_dir: Path, step: int) -> None:
        self.step = int(step)
        self.dir = Path(obs_dir) / f"step_{step:05d}"
        self._rows: dict[int, dict[str, Any]] = {}
        self.loss_mode = "forward_kl_topk"
        self.log_prob_min_clamp = -10.0
        self._load()

    def _load(self) -> None:
        for path in sorted(self.dir.glob("rank*.npz")):
            with np.load(path) as z:
                if "loss_mode" in z:
                    self.loss_mode = str(z["loss_mode"][0])
                    self.log_prob_min_clamp = float(z["log_prob_min_clamp"][0])
                n = len(z["row_hash"])
                seg_off = z["seg_offsets"]
                resp_off = z["resp_offsets"]
                spans_off = z["turn_spans_offsets"]
                for i in range(n):
                    # codex review P1-14: the rolling hash alone is not a
                    # primary key — key on (hash, length) and REFUSE a merge
                    # whose token stream disagrees (a silent cross-sample
                    # merge is the one corruption a microscope must never do).
                    resp = z["response_ids"][resp_off[i] : resp_off[i + 1]]
                    h = (int(z["row_hash"][i]), int(len(resp)))
                    row = self._rows.get(h)
                    if row is not None and not np.array_equal(
                        row["response_ids"], resp.astype(np.int64)
                    ):
                        raise RuntimeError(
                            f"observation pack row-key collision at step "
                            f"{self.step}: hash {h[0]} len {h[1]} maps to two "
                            "different token streams — needs a stronger UID"
                        )
                    if row is None:
                        R = len(resp)
                        spans = z["turn_spans"][spans_off[i] : spans_off[i + 1]]
                        row = {
                            "resp_hash": h[0],
                            "step": self.step,
                            "task_id": str(z["row_task"][i]),
                            "sample_role": str(z["row_role"][i]),
                            "won": int(z["row_won"][i]),
                            "reward": float(z["row_reward"][i]),
                            "response_ids": resp.astype(np.int64),
                            "response_mask": z["response_mask"][resp_off[i] : resp_off[i + 1]].astype(bool),
                            "realized_student_lp": z["realized_student_lp"][
                                resp_off[i] : resp_off[i + 1]
                            ].astype(np.float32),
                            "turn_spans": spans.reshape(-1, 2).tolist(),
                            "covered": np.zeros(R, dtype=bool),
                        }
                        K = z["teacher_ids"].shape[-1] if z["teacher_ids"].ndim == 2 else 0
                        for key in QUARTET:
                            dt = np.int64 if key.endswith("ids") else np.float32
                            width = z[key].shape[-1]
                            fill = 0 if key.endswith("ids") else np.nan
                            row[key] = np.full((R, width), fill, dtype=dt)
                        row["token_loss"] = np.full(R, np.nan, dtype=np.float32)
                        self._rows[h] = row
                    a, b = int(z["seg_start"][i]), int(z["seg_start"][i]) + int(z["seg_len"][i])
                    lo, hi = int(seg_off[i]), int(seg_off[i + 1])
                    for key in QUARTET:
                        row[key][a:b] = z[key][lo:hi]
                    row["token_loss"][a:b] = z["token_loss"][lo:hi]
                    row["covered"][a:b] = True

    def hashes(self) -> list[int]:
        return sorted(h for h, _ in self._rows)

    def rows(self) -> list[dict[str, Any]]:
        return [self._rows[k] for k in sorted(self._rows)]

    def row(self, resp_hash: int) -> dict[str, Any]:
        matches = [k for k in self._rows if k[0] == int(resp_hash)]
        if not matches:
            raise KeyError(resp_hash)
        if len(matches) > 1:
            raise RuntimeError(
                f"hash {resp_hash} matches {len(matches)} rows of different "
                "lengths — address rows via rows()"
            )
        return self._rows[matches[0]]


@functools.lru_cache(maxsize=8)
def load_step(obs_dir: str, step: int) -> StepPack:
    return StepPack(Path(obs_dir), step)
