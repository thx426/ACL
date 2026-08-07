#!/usr/bin/env python3
"""低频只读监控 PGS 长跑，输出产物、进程、GPU、磁盘与 fatal 摘要。"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

FATAL = re.compile(
    r"Traceback \(most recent call last\)|"
    r"OutOfMemoryError|CUDA out of memory|EngineDeadError|RayTaskError|"
    r"PytorchStreamWriter[^\n]*failed|file write failed|unexpected pos|"
    r"DataLoader worker[^\n]*(?:killed by signal|is killed by signal)|"
    r"NCCL WARN|NCCL ERROR|ncclUnhandledCudaError|unhandled system error|"
    r"Watchdog caught collective operation timeout|"
    r"ProcessGroupNCCL[^\n]*(?:Error|timed out)"
)
STEP_RE = re.compile(r"^step_(\d{5})(_val)?$")
TZ = dt.timezone(dt.timedelta(hours=8))


def stamp() -> str:
    return dt.datetime.now(TZ).isoformat(timespec="seconds")


def nonempty_json_count(path: Path) -> tuple[int, float]:
    count = 0
    newest = 0.0
    if not path.is_dir():
        return count, newest
    for item in path.iterdir():
        if item.is_file() and item.suffix == ".json":
            stat = item.stat()
            if stat.st_size > 0:
                count += 1
                newest = max(newest, stat.st_mtime)
    return count, newest


def latest_rollouts(run_dir: Path) -> tuple[tuple[int, int, float], tuple[int, int, float]]:
    trains: list[tuple[int, Path]] = []
    vals: list[tuple[int, Path]] = []
    rollouts = run_dir / "rollouts"
    if rollouts.is_dir():
        for path in rollouts.iterdir():
            match = STEP_RE.match(path.name)
            if not path.is_dir() or not match:
                continue
            target = vals if match.group(2) else trains
            target.append((int(match.group(1)), path))

    def summarize(items: list[tuple[int, Path]]) -> tuple[int, int, float]:
        if not items:
            return -1, 0, 0.0
        step, path = max(items)
        count, newest = nonempty_json_count(path)
        return step, count, newest

    return summarize(trains), summarize(vals)


def checkpoint_state(run_dir: Path) -> tuple[int, int, float]:
    marker = run_dir / "checkpoints" / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        return -1, 0, 0.0
    try:
        step = int(marker.read_text().strip())
    except (OSError, ValueError):
        return -1, 0, 0.0
    checkpoint = marker.parent / f"global_step_{step}"
    files = sum(item.is_file() for item in checkpoint.rglob("*")) if checkpoint.is_dir() else 0
    return step, files, marker.stat().st_mtime


def process_alive(pid: int, required_text: str) -> bool:
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.is_file():
        return False
    try:
        command = cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return required_text in command


def gpu_summary() -> str:
    command = [
        "timeout",
        "20s",
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        return "nvidia-smi-unavailable"
    return ";".join(line.strip().replace(", ", "/") for line in result.stdout.splitlines())


def memory_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    return -1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--train-log", required=True, type=Path)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--pid-cmd-substr", required=True)
    parser.add_argument("--interval", type=int, default=600)
    parser.add_argument("--stale-seconds", type=int, default=1800)
    parser.add_argument("--target-steps", type=int, default=358)
    args = parser.parse_args()

    offset = 0
    seen_fatal: set[str] = set()
    last_artifact_time = time.time()
    previous_state: tuple[int, ...] | None = None
    stale_reported = False

    while True:
        train, val = latest_rollouts(args.run_dir)
        checkpoint = checkpoint_state(args.run_dir)
        state = (train[0], train[1], val[0], val[1], checkpoint[0], checkpoint[1])
        newest = max(train[2], val[2], checkpoint[2])
        if state != previous_state:
            previous_state = state
            last_artifact_time = max(time.time(), newest)
            stale_reported = False
        elif newest:
            last_artifact_time = max(last_artifact_time, newest)

        new_text = ""
        if args.train_log.is_file():
            size = args.train_log.stat().st_size
            if size < offset:
                offset = 0
            with args.train_log.open("rb") as stream:
                stream.seek(offset)
                new_text = stream.read().decode(errors="replace")
                offset = stream.tell()
        fatals = sorted(set(FATAL.findall(new_text)) - seen_fatal)
        seen_fatal.update(fatals)

        alive = process_alive(args.pid, args.pid_cmd_substr)
        stale = int(time.time() - last_artifact_time)
        disk_free = shutil.disk_usage(args.run_dir).free / 1024**4
        print(
            f"{stamp()} alive={alive} train={train[0]}:{train[1]} "
            f"val={val[0]}:{val[1]} ckpt={checkpoint[0]}:{checkpoint[1]} "
            f"stale_s={stale} mem_avail_gib={memory_available_gib():.1f} "
            f"disk_free_tib={disk_free:.2f} gpu={gpu_summary()}",
            flush=True,
        )
        if fatals:
            print(f"{stamp()} ALERT_FATAL {' | '.join(fatals)}", flush=True)
        if stale > args.stale_seconds and not stale_reported:
            print(f"{stamp()} ALERT_STALE no_artifact_progress={stale}s", flush=True)
            stale_reported = True
        if not alive:
            status = "TARGET_REACHED" if train[0] >= args.target_steps and train[1] >= 128 else "EARLY_EXIT"
            print(f"{stamp()} ALERT_PROCESS_EXIT status={status} pid={args.pid}", flush=True)
            return 0 if status == "TARGET_REACHED" and not seen_fatal else 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
