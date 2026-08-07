"""AppWorld dataset builder — 官方划分 → verl parquet(须在装有 appworld
的服务器环境运行:每任务开一次 worker 捕获 instruction/supervisor/难度)。

划分(旧引入 2026-07-04 谈判定稿,继承):
  train.parquet  = official train(90)
  valid.parquet  = dev(57)+ test_normal(168),`data_source=
                   appworld/{dev,test_normal}` 双桶曲线
  test_challenge(417)**永不触碰**(终局评测另行裁定)。

运行:python -m trace_opd.envs.appworld.build_dataset \
        --appworld-root $APPWORLD_ROOT --worker-python <py311> --out <dir> \
        [--boot-concurrency 8]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ...core import protocol
from .adapter import _WORKER_SCRIPT, SYSTEM_PROMPT, initial_obs_of

_SPLIT_FILES = {"train": "train.txt", "dev": "dev.txt", "test_normal": "test_normal.txt"}


def _task_ids(root: Path, split: str) -> list[str]:
    path = root / "data" / "datasets" / _SPLIT_FILES[split]
    ids = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not ids:
        raise SystemExit(f"FATAL: empty split file {path}")
    return ids


def _capture(args, task_id: str) -> dict:
    """开 worker 拿 ready 事件(instruction/supervisor/metadata)即关。"""
    proc = subprocess.Popen(
        [args.worker_python, _WORKER_SCRIPT, task_id, f"build_{task_id}",
         "minimal", "1", "0"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True,
        env={**__import__("os").environ, "APPWORLD_ROOT": args.appworld_root},
    )
    try:
        line = proc.stdout.readline()
        ev = json.loads(line)
        if ev.get("event") != "ready":
            raise SystemExit(f"FATAL: capture failed for {task_id}: {ev}")
        return ev
    finally:
        try:
            proc.stdin.write("__CLOSE__\n")
            proc.stdin.flush()
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--appworld-root", required=True)
    ap.add_argument("--worker-python", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot-concurrency", type=int, default=8)
    args = ap.parse_args()
    root = Path(args.appworld_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gate = threading.Semaphore(args.boot_concurrency)

    def one(split: str, tid: str) -> dict:
        with gate:
            ev = _capture(args, tid)
        item = {
            "task_id": tid,
            "instruction": str(ev["instruction"]),
            "supervisor": {str(k): str(v) for k, v in (ev.get("supervisor") or {}).items()},
            "difficulty": (ev.get("metadata") or {}).get("difficulty", ""),
            "split": split,
        }
        item["initial_obs"] = initial_obs_of(item)
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": protocol.truncate_obs(item["initial_obs"])},
            ],
            "data_source": "appworld" if split == "train" else f"appworld/{split}",
            "agent_name": "trace_opd",
            "reward_model": {"style": "env", "ground_truth": ""},
            "extra_info": item,
        }

    with ThreadPoolExecutor(max_workers=args.boot_concurrency) as pool:
        train_rows = list(pool.map(lambda t: one("train", t), _task_ids(root, "train")))
        valid_rows = list(pool.map(lambda t: one("dev", t), _task_ids(root, "dev")))
        valid_rows += list(pool.map(
            lambda t: one("test_normal", t), _task_ids(root, "test_normal")))

    for name, rows in (("train", train_rows), ("valid", valid_rows)):
        pd.DataFrame(rows).to_parquet(out / f"{name}.parquet")
        diffs: dict[str, int] = {}
        for r in rows:
            d = str(r["extra_info"]["difficulty"])
            diffs[d] = diffs.get(d, 0) + 1
        print(f"[appworld] {name}: {len(rows)} tasks -> {out / (name + '.parquet')}"
              f" | difficulty: {diffs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
