#!/usr/bin/env python3
"""Audit one complete max-open dynamic-researcher training step."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

EXPECTED_TASKS = 16
EXPECTED_TRAJECTORIES = 128
METRIC_RE = re.compile(r"step:1\s+-\s+(.*)")


def _metric(line: str, key: str) -> float:
    match = re.search(rf"(?:^|\s-\s){re.escape(key)}:([^\s]+)", line)
    if not match:
        raise SystemExit(f"FATAL: step-1 metric missing: {key}")
    return float(match.group(1))


def _credential_values() -> set[str]:
    values = [sys.stdin.readline().rstrip("\n"), sys.stdin.readline().rstrip("\n")]
    if any(not value for value in values):
        raise SystemExit("FATAL: secret audit requires two non-empty credential lines")
    return {value for value in values}


def _scan_secrets(paths: list[Path], secrets: set[str]) -> int:
    needles = [value.encode() for value in secrets]
    overlap = max(len(needle) for needle in needles) - 1
    scanned = 0
    for root in paths:
        members = root.rglob("*") if root.is_dir() else [root]
        for path in members:
            if not path.is_file() or "checkpoints" in path.parts:
                continue
            try:
                carry = b""
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        payload = carry + chunk
                        for needle in needles:
                            if needle in payload:
                                raise SystemExit(f"FATAL: credential value leaked to {path}")
                        carry = payload[-overlap:] if overlap > 0 else b""
            except OSError as exc:
                raise SystemExit(f"FATAL: cannot scan {path}: {exc}") from exc
            scanned += 1
    return scanned


def _load_rows(run: Path) -> list[tuple[Path, dict]]:
    step = run / "rollouts/step_00001"
    paths = sorted(step.glob("*.json"))
    if len(paths) != EXPECTED_TRAJECTORIES:
        raise SystemExit(
            f"FATAL: step_00001 needs {EXPECTED_TRAJECTORIES} rollouts, got {len(paths)}"
        )
    rows = []
    for path in paths:
        try:
            rows.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"FATAL: invalid rollout {path}: {exc}") from exc
    return rows


def _audit_rows(run: Path, rows: list[tuple[Path, dict]]) -> dict:
    groups: dict[str, list[dict]] = {}
    archives: set[str] = set()
    totals = {
        "assistant_tokens": 0,
        "d_observations": 0,
        "base_open": 0,
        "tilts": 0,
        "rejects": 0,
        "budget_blocked": 0,
        "pacing_wait": 0,
        "bypassed": 0,
        "turns_capped": 0,
        "probe_flipped": 0,
    }
    for path, row in rows:
        if row.get("validate"):
            raise SystemExit(f"FATAL: train row marked validate: {path}")
        task = str(row.get("task_id") or "")
        research = row.get("research") or {}
        steering = row.get("steering") or {}
        if not task or research.get("research_error") or research.get("research_doc_missing"):
            raise SystemExit(f"FATAL: missing task or degraded researcher: {path}")
        archive = str(research.get("research_archive") or "")
        archive_dir = run / archive
        doc_path = archive_dir / "doc.md"
        meta_path = archive_dir / "meta.json"
        if not archive.startswith("research/") or not doc_path.is_file() or not meta_path.is_file():
            raise SystemExit(f"FATAL: incomplete dynamic archive for {path}")
        if not doc_path.read_text(encoding="utf-8").strip():
            raise SystemExit(f"FATAL: empty dynamic document for {path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("global_step") != 1
            or meta.get("policy_version") != 0
            or meta.get("error")
            or meta.get("session_error")
            or not meta.get("finalized")
        ):
            raise SystemExit(f"FATAL: bad archive lineage/status: {meta_path}")
        policy = row.get("policy_version") or {}
        if policy != {"min_global_steps": 0, "max_global_steps": 0}:
            raise SystemExit(f"FATAL: student policy lineage is not exact 0..0: {path}")
        if research.get("research_policy_version") != 0:
            raise SystemExit(f"FATAL: research policy lineage is not version 0: {path}")
        assistant = sum(int(b) - int(a) for a, b in row.get("turn_token_spans", []))
        d_count = int(steering.get("steer_d_count", 0))
        if d_count != assistant:
            raise SystemExit(
                f"FATAL: D coverage {d_count} != assistant tokens {assistant}: {path}"
            )
        fields = {
            "budget_blocked": "steer_n_budget_blocked",
            "pacing_wait": "steer_n_pacing_wait",
            "bypassed": "steer_n_bypassed",
            "turns_capped": "steer_turns_capped",
            "probe_flipped": "steer_n_probe_flipped",
        }
        for target, source in fields.items():
            value = int(steering.get(source, 0))
            totals[target] += value
            if value:
                raise SystemExit(f"FATAL: max-open invariant {source}={value}: {path}")
        totals["assistant_tokens"] += assistant
        totals["d_observations"] += d_count
        totals["base_open"] += int(steering.get("steer_n_base_open", 0))
        totals["tilts"] += int(steering.get("steer_count", 0))
        totals["rejects"] += int(steering.get("steer_reject_count", 0))
        groups.setdefault(task, []).append(row)
        archives.add(archive)

    sizes = sorted(len(group) for group in groups.values())
    if len(groups) != EXPECTED_TASKS or sizes != [8] * EXPECTED_TASKS:
        raise SystemExit(f"FATAL: expected 16 tasks x 8 students, got {len(groups)} tasks / {sizes}")
    if len(archives) != EXPECTED_TASKS:
        raise SystemExit(f"FATAL: expected 16 archives, got {len(archives)}")
    for task, group in groups.items():
        if len({(row["research"])["research_archive"] for row in group}) != 1:
            raise SystemExit(f"FATAL: task group does not share one archive: {task}")
    if totals["base_open"] <= 0 or totals["tilts"] <= 0:
        raise SystemExit("FATAL: formal/smoke treatment performed no intervention")
    totals.update(tasks=len(groups), trajectories=len(rows), archives=len(archives))
    totals["base_open_rate"] = totals["base_open"] / totals["d_observations"]
    totals["reject_per_tilt"] = totals["rejects"] / totals["tilts"]
    return totals


def _audit_config(run: Path) -> dict:
    path = run / "effective_config.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    st = data["steering"]
    required = {
        "doc_source": "researcher",
        "epsilon": 0.0,
        "gate_target_rate": 0.20,
        "b_max": 81920.0,
        "budget_pacing": False,
        "r_max": 32769,
        "gate_adaptive": True,
    }
    for key, value in required.items():
        if st.get(key) != value:
            raise SystemExit(f"FATAL: effective steering.{key}={st.get(key)!r}, expected {value!r}")
    theta = float(st.get("theta_d", 0))
    if not (0 < theta < 1_000_000_000):
        raise SystemExit(f"FATAL: invalid materialized theta_d {theta}")
    return {"theta_d": theta, "b_max": st["b_max"], "r_max": st["r_max"]}


def _audit_updates(log: Path) -> dict:
    text = log.read_text(errors="replace")
    lines = [line for line in text.splitlines() if "step:1 -" in line]
    if len(lines) != 1:
        raise SystemExit(f"FATAL: expected exactly one completed step-1 metrics line, got {len(lines)}")
    line = lines[0]
    metrics = {
        "global_step": _metric(line, "training/global_step"),
        "grad_norm": _metric(line, "actor/grad_norm"),
        "w_off1": _metric(line, "actor/rollout_is_weight_off1_frac"),
        "update_actor_s": _metric(line, "timing_s/update_actor"),
        "update_weights_s": _metric(line, "timing_s/update_weights"),
    }
    if metrics["global_step"] != 1 or any(
        metrics[key] <= 0 for key in ("grad_norm", "w_off1", "update_actor_s", "update_weights_s")
    ):
        raise SystemExit(f"FATAL: actor/SGLang update evidence is incomplete: {metrics}")
    forbidden = (
        "OutOfMemoryError",
        "CUDA out of memory",
        "EngineDeadError",
        "RayTaskError",
        "steering verify aborted",
        "physically unscorable",
        "misaligned",
    )
    hits = [needle for needle in forbidden if needle in text]
    if hits:
        raise SystemExit(f"FATAL: fatal/update-path signatures in log: {hits}")
    return metrics


def _audit_checkpoint(run: Path) -> dict:
    marker = run / "checkpoints/latest_checkpointed_iteration.txt"
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != "1":
        raise SystemExit("FATAL: smoke checkpoint marker is not complete step 1")
    checkpoint = run / "checkpoints/global_step_1"
    data_pt = checkpoint / "data.pt"
    if not data_pt.is_file() or data_pt.stat().st_size == 0:
        raise SystemExit("FATAL: smoke checkpoint data.pt is missing/empty")
    files = [path for path in checkpoint.rglob("*") if path.is_file()]
    if not files:
        raise SystemExit("FATAL: smoke checkpoint contains no files")
    return {"checkpoint_files": len(files), "checkpoint_bytes": sum(path.stat().st_size for path in files)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check-checkpoint", action="store_true")
    parser.add_argument("--scan-root", action="append", type=Path, default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"FATAL: audit output exists; refusing overwrite: {args.output}")
    secrets = _credential_values()
    rows = _load_rows(args.run)
    result = {
        "run": str(args.run),
        "config": _audit_config(args.run),
        "rollouts": _audit_rows(args.run, rows),
        "updates": _audit_updates(args.log),
    }
    if args.check_checkpoint:
        result["checkpoint"] = _audit_checkpoint(args.run)
    scan_roots = [args.run, args.log, *args.scan_root]
    result["secret_scan_files"] = _scan_secrets(scan_roots, secrets)
    result["status"] = "PASS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
