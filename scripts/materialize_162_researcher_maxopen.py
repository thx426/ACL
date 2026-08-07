#!/usr/bin/env python3
"""从完整dynamic标定run生成max-open正式配置，并落P80审计记录。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace_opd.common.config import load_config, objective_fingerprint  # noqa: E402

CALIB_NAME = "pgs_s2_sgl_1p7b_162_researcher_maxopen_calib_p80"
FORMAL_NAME = "pgs_s2_sgl_1p7b_162_researcher_maxopen"
EXPECTED_TRAJECTORIES = 128
EXPECTED_TASKS = 16
TARGET_QUANTILE = 0.80


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_step(run: Path) -> tuple[Path, list[tuple[Path, dict]]]:
    candidates = sorted(
        path for path in (run / "rollouts").glob("step_[0-9][0-9][0-9][0-9][0-9]")
        if path.is_dir()
    )
    if candidates != [run / "rollouts" / "step_00001"]:
        raise SystemExit(
            "FATAL: calibration must contain exactly one train step_00001; got "
            + ", ".join(path.name for path in candidates)
        )
    step = candidates[0]
    rows: list[tuple[Path, dict]] = []
    for path in sorted(step.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"FATAL: invalid rollout {path}: {exc}") from exc
        rows.append((path, row))
    if len(rows) != EXPECTED_TRAJECTORIES:
        raise SystemExit(
            f"FATAL: calibration needs {EXPECTED_TRAJECTORIES} valid rollouts, got {len(rows)}"
        )
    return step, rows


def _audit_dynamic(run: Path, rows: list[tuple[Path, dict]]) -> dict:
    tasks: dict[str, list[dict]] = {}
    archives: set[str] = set()
    d_total = 0
    hist: list[int] | None = None
    hist_max: float | None = None
    for path, row in rows:
        task = str(row.get("task_id") or "")
        research = row.get("research") or {}
        steering = row.get("steering") or {}
        if row.get("validate"):
            raise SystemExit(f"FATAL: calibration train row marked validate: {path}")
        if not task:
            raise SystemExit(f"FATAL: missing task_id: {path}")
        if research.get("research_error") or research.get("research_doc_missing"):
            raise SystemExit(f"FATAL: degraded researcher row: {path}")
        archive = str(research.get("research_archive") or "")
        if not archive.startswith("research/"):
            raise SystemExit(f"FATAL: missing dynamic archive: {path}")
        archive_dir = run / archive
        doc = archive_dir / "doc.md"
        meta_path = archive_dir / "meta.json"
        if not doc.is_file() or not doc.read_text(encoding="utf-8").strip():
            raise SystemExit(f"FATAL: empty dynamic document: {doc}")
        if not meta_path.is_file():
            raise SystemExit(f"FATAL: missing archive meta: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("global_step") != 1 or meta.get("policy_version") != 0:
            raise SystemExit(
                f"FATAL: calibration lineage must be global_step=1/policy_version=0: {meta_path}"
            )
        if meta.get("error") or meta.get("session_error") or not meta.get("finalized"):
            raise SystemExit(f"FATAL: failed/unfinalized research archive: {meta_path}")
        policy = row.get("policy_version") or {}
        if policy.get("min_global_steps") != 0 or policy.get("max_global_steps") != 0:
            raise SystemExit(f"FATAL: student policy lineage is not 0..0: {path}")
        if research.get("research_policy_version") != 0:
            raise SystemExit(f"FATAL: rollout research_policy_version is not 0: {path}")
        if steering.get("steer_count") != 0 or steering.get("steer_reject_count") != 0:
            raise SystemExit(f"FATAL: calibration changed behavior: {path}")
        if steering.get("steer_n_base_open") != 0 or steering.get("steer_n_probe_flipped") != 0:
            raise SystemExit(f"FATAL: calibration gate/probe was not fully closed: {path}")
        for key in (
            "steer_n_budget_blocked",
            "steer_n_pacing_wait",
            "steer_n_bypassed",
            "steer_turns_capped",
        ):
            if steering.get(key, 0) != 0:
                raise SystemExit(f"FATAL: calibration non-binding invariant {key} != 0: {path}")
        assistant_tokens = sum(int(b) - int(a) for a, b in row.get("turn_token_spans", []))
        d_count = int(steering.get("steer_d_count", 0))
        if d_count != assistant_tokens:
            raise SystemExit(
                f"FATAL: D coverage {d_count} != assistant tokens {assistant_tokens}: {path}"
            )
        row_hist = steering.get("steer_d_hist")
        row_bins = int(steering.get("steer_d_hist_bins", len(row_hist or [])))
        row_max = float(steering.get("steer_d_hist_max", 0.0))
        if not row_hist or len(row_hist) != row_bins:
            raise SystemExit(f"FATAL: missing/malformed D histogram: {path}")
        if hist is None:
            hist = [0] * row_bins
            hist_max = row_max
        elif len(hist) != row_bins or not math.isclose(float(hist_max), row_max):
            raise SystemExit(f"FATAL: mixed D histogram schemas: {path}")
        for index, value in enumerate(row_hist):
            hist[index] += int(value)
        d_total += d_count
        tasks.setdefault(task, []).append(row)
        archives.add(archive)

    if len(tasks) != EXPECTED_TASKS or any(len(group) != 8 for group in tasks.values()):
        raise SystemExit(
            f"FATAL: expected {EXPECTED_TASKS} tasks x 8 students, got "
            f"{len(tasks)} tasks and sizes {sorted(len(group) for group in tasks.values())}"
        )
    if len(archives) != EXPECTED_TASKS:
        raise SystemExit(f"FATAL: expected {EXPECTED_TASKS} shared archives, got {len(archives)}")
    for task, group in tasks.items():
        task_archives = {(row.get("research") or {}).get("research_archive") for row in group}
        if len(task_archives) != 1:
            raise SystemExit(f"FATAL: n=8 students do not share one archive for {task}")
    if hist is None or hist_max is None or d_total <= 0 or sum(hist) != d_total:
        raise SystemExit("FATAL: invalid aggregate D histogram")
    return {
        "tasks": len(tasks),
        "trajectories": len(rows),
        "archives": len(archives),
        "d_observations": d_total,
        "hist": hist,
        "hist_max": hist_max,
    }


def _hist_quantile(hist: list[int], hist_max: float, q: float) -> float:
    total = sum(hist)
    target = q * total
    acc = 0
    for index, count in enumerate(hist):
        acc += count
        if acc >= target:
            if index == len(hist) - 1:
                raise SystemExit("FATAL: target P80 falls in D histogram overflow bucket")
            lower = index * hist_max / len(hist)
            fraction = 1.0 - (acc - target) / count if count else 0.0
            return lower + fraction * hist_max / len(hist)
    raise SystemExit("FATAL: unable to compute P80")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-run", required=True, type=Path)
    parser.add_argument("--calibration-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists() or args.audit_output.exists():
        raise SystemExit("FATAL: output or audit path already exists; refusing overwrite")
    cfg = load_config(args.calibration_config)
    if cfg.run.name != CALIB_NAME:
        raise SystemExit(f"FATAL: unexpected calibration config run.name {cfg.run.name!r}")
    if not (
        cfg.steering.theta_d == 1_000_000_000.0
        and not cfg.steering.gate_adaptive
        and cfg.steering.epsilon == 0.0
        and cfg.steering.b_max == 81920.0
        and not cfg.steering.budget_pacing
        and cfg.steering.r_max == 32769
    ):
        raise SystemExit("FATAL: calibration config does not satisfy the zero-intervention contract")

    _, rows = _load_step(args.calibration_run)
    audit = _audit_dynamic(args.calibration_run, rows)
    theta = _hist_quantile(audit.pop("hist"), float(audit["hist_max"]), TARGET_QUANTILE)
    if not math.isfinite(theta) or theta <= 0:
        raise SystemExit(f"FATAL: invalid P80 theta {theta!r}")

    data = yaml.safe_load(args.calibration_config.read_text(encoding="utf-8"))
    data["run"]["name"] = FORMAL_NAME
    data["train"]["experiment_name"] = FORMAL_NAME
    data["steering"]["theta_d"] = round(theta, 8)
    data["steering"]["gate_adaptive"] = True
    data["steering"]["gate_rate_bounds"] = [0.10, 0.30]
    header = (
        "# MATERIALIZED FORMAL CONFIG — generated only after complete dynamic P80 calibration.\n"
        f"# calibration run: {args.calibration_run}\n"
        f"# calibration config sha256: {_sha256(args.calibration_config)}\n"
        f"# calibrated P80 theta_d: {theta:.8f}\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        header + yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    formal = load_config(args.output)
    audit.update(
        {
            "calibration_run": str(args.calibration_run),
            "calibration_config": str(args.calibration_config),
            "calibration_config_sha256": _sha256(args.calibration_config),
            "formal_config": str(args.output),
            "formal_config_sha256": _sha256(args.output),
            "target_quantile": TARGET_QUANTILE,
            "theta_d": formal.steering.theta_d,
            "objective_fingerprint": objective_fingerprint(formal),
        }
    )
    args.audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
