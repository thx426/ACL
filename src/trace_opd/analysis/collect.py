"""Offline join: training-side microscope samples × rollout dumps
→ TurnProfile / TokenRecord rows + position-band depth profile.

The training-side sampler (verl 6f3d089) knows tokens but not turns; the
rollout dump knows turns but not per-token KL. The polynomial content hash
over response token ids (common/hashing.py) is the join key.

Usage: python -m trace_opd.analysis.collect --run-dir <run> [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..viz.plugins import task_family
from ..viz.schema import parse_step_dir, read_json
from .schema import TokenRecord, TurnProfile, write_jsonl

logger = logging.getLogger(__name__)

# v1 gating criterion for g_gated: the student still leaves noticeable mass
# outside the teacher's top-k (unsettled or diverging position). The exact
# student-entropy gate comes from the probe tool; see analysis-design.md §1-6.
GATE_S_MASS_BELOW = 0.9
POSITION_BAND = 512


def index_dumps(run_dir: Path) -> dict[int, dict[str, Any]]:
    """hash -> dump info for every rollout that carries the microscope keys."""
    index: dict[int, dict[str, Any]] = {}
    collisions = 0
    for step_dir in sorted((run_dir / "rollouts").glob("step_*")):
        if parse_step_dir(step_dir.name) is None:
            continue
        for file in step_dir.glob("*.json"):
            d = read_json(file)
            if not d or "response_token_hash" not in d:
                continue
            h = int(d["response_token_hash"])
            if h in index:
                collisions += 1
                continue
            index[h] = {
                "step": int(d.get("global_step") or 0),
                "validate": bool(d.get("validate")),
                "task_id": str(d.get("task_id", "")),
                "spans": [tuple(s) for s in d.get("turn_token_spans") or []],
                "resp_len": int(d.get("response_token_len") or 0),
                "file": f"rollouts/{step_dir.name}/{file.name}",
            }
    if collisions:
        logger.warning("%d hash collisions in dump index (rows kept first-wins)", collisions)
    return index


def _turn_of(spans: list[tuple[int, int]], t: int) -> int | None:
    for turn, (start, end) in enumerate(spans, start=1):
        if start <= t < end:
            return turn
    return None


def collect(run_dir: Path, microscope_dir: Path, out_dir: Path) -> dict[str, int]:
    index = index_dumps(run_dir)
    turn_rows: list[TurnProfile] = []
    token_rows: list[TokenRecord] = []
    bands: dict[tuple[int, int], list[float]] = defaultdict(list)
    joined = orphans = mask_mismatch = 0

    for shard in sorted(microscope_dir.glob("rank*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            info = index.get(int(row["resp_hash"]))
            if info is None or info["resp_len"] != row["resp_len"]:
                orphans += 1
                continue
            joined += 1
            step, spans = info["step"], info["spans"]
            traj_id = info["file"]
            kl, mask = row["kl"], row["mask"]
            s_mass, t_ent = row.get("s_mass"), row.get("t_ent")
            overlap = row.get("overlap")
            ids = row["ids"]

            # Integrity self-check: spans must cover exactly mask==1 positions.
            in_span = [False] * len(mask)
            for start, end in spans:
                for t in range(start, min(end, len(mask))):
                    in_span[t] = True
            if any(bool(m) != s for m, s in zip(mask, in_span)):
                mask_mismatch += 1

            per_turn: dict[int, dict[str, list[float]]] = defaultdict(
                lambda: {"kl": [], "gated": [], "ov": [], "ent": []}
            )
            for t, m in enumerate(mask):
                if not m:
                    continue
                turn = _turn_of(spans, t)
                if turn is None:
                    continue
                acc = per_turn[turn]
                acc["kl"].append(kl[t])
                if s_mass is not None and s_mass[t] < GATE_S_MASS_BELOW:
                    acc["gated"].append(kl[t])
                if overlap is not None:
                    acc["ov"].append(overlap[t])
                if t_ent is not None:
                    acc["ent"].append(t_ent[t])
                bands[(step, t // POSITION_BAND)].append(kl[t])
                token_rows.append(TokenRecord(
                    step=step, traj_id=traj_id, t=t, turn=turn,
                    token_id=int(ids[t]), kl=float(kl[t]),
                    h_t=float(t_ent[t]) if t_ent is not None else -1.0,
                    in_overlap=bool(overlap[t]) if overlap is not None else False,
                ))
            mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
            for turn, acc in sorted(per_turn.items()):
                turn_rows.append(TurnProfile(
                    step=step, traj_id=traj_id, turn=turn,
                    g_mean=round(mean(acc["kl"]), 5),
                    g_gated=round(mean(acc["gated"]), 5),
                    overlap=round(mean(acc["ov"]), 5),
                    h_teacher=round(mean(acc["ent"]), 5),
                    h_student=-1.0,
                    n_tokens=len(acc["kl"]),
                ))

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "turn_profiles.jsonl", turn_rows)
    write_jsonl(out_dir / "token_records.jsonl", token_rows)
    band_rows = [
        {"step": step, "band": band, "band_tokens": f"{band * POSITION_BAND}-{(band + 1) * POSITION_BAND}",
         "g_mean": round(sum(v) / len(v), 5), "n": len(v)}
        for (step, band), v in sorted(bands.items())
    ]
    (out_dir / "position_bands.json").write_text(
        json.dumps(band_rows, indent=1), encoding="utf-8"
    )
    stats = {
        "joined": joined, "orphans": orphans, "mask_mismatch": mask_mismatch,
        "turn_rows": len(turn_rows), "token_rows": len(token_rows),
    }
    if mask_mismatch:
        logger.warning("mask/span mismatch on %d trajectories — check episode "
                       "turn-span recording vs verl response layout", mask_mismatch)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--microscope-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    micro = args.microscope_dir or args.run_dir / "microscope"
    out = args.out or args.run_dir / "analysis"
    stats = collect(args.run_dir, micro, out)
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
