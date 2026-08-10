#!/usr/bin/env python
"""ScienceWorld 启动门禁探针(供 experiments/*/run.sh 调用)。

两个子命令,都只做**行为断言**而非存在性检查:

  rows <data_dir>   ETO 口径行数身份:train 1483 / seen 194 / unseen 211。
                    只看目录名或文件存在会漏掉"用错口径重建"的情况。
  smoke             真引擎冒烟:起一个 JVM,走**生产 worker.py 的 line 协议**,
                    确认 ready 事件与一次真实 step。
                    `import scienceworld` 成功不是可用性判据 —— A800 系统无 java,
                    JVM 起不来只会在 episode 装配期才炸,那时已经烧掉了排队时间。

输出契约:成功打印 `OK <详情>`,失败打印 `BAD <原因>`;调用方按首词判定。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# ETO 口径(裁定 2026-08-08):train 1483 / dev(seen) 194 / test(unseen) 211
EXPECT = {"train": 1483, "valid_seen": 194, "valid_unseen": 211}


def cmd_rows(data_dir: str) -> int:
    try:
        import pandas as pd
    except ImportError:
        print("BAD pandas 不可导入")
        return 1
    d = Path(data_dir)
    if not d.is_dir():
        print(f"BAD 数据目录不存在: {data_dir}")
        return 1
    bad, got = [], {}
    for split, want in EXPECT.items():
        p = d / f"{split}.parquet"
        if not p.is_file():
            bad.append(f"{split}.parquet 缺失")
            continue
        try:
            n = len(pd.read_parquet(p))
        except Exception as exc:  # noqa: BLE001 — 门禁要吃下一切并如实报告
            bad.append(f"{split} 读取失败 {type(exc).__name__}")
            continue
        got[split] = n
        if n != want:
            bad.append(f"{split} {n} 行(ETO 口径应为 {want})")
    if bad:
        print("BAD " + " | ".join(bad))
        return 1
    print("OK " + " / ".join(f"{k} {v}" for k, v in got.items()))
    return 0


def cmd_smoke() -> int:
    worker_py = os.environ.get("SCIWORLD_WORKER_PYTHON", "")
    if not worker_py or not os.access(worker_py, os.X_OK):
        print(f"BAD SCIWORLD_WORKER_PYTHON 不可执行: {worker_py or '<未设>'}")
        return 1
    worker = (Path(__file__).resolve().parent.parent
              / "src" / "trace_opd" / "envs" / "scienceworld" / "worker.py")
    if not worker.is_file():
        print(f"BAD worker.py 不存在: {worker}")
        return 1
    try:
        proc = subprocess.run(
            [worker_py, str(worker), "task-1-boil", "0", "easy", "100"],
            input="look around\n__CLOSE__\n",
            capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        print("BAD 真引擎冒烟超时 180s(JVM 起不来?)")
        return 1
    events, step_ok = [], False
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        events.append(ev.get("event"))
        if ev.get("event") == "fatal":
            print(f"BAD worker 报 fatal: {str(ev.get('error'))[:120]}")
            return 1
        if ev.get("event") == "step" and ev.get("obs"):
            step_ok = True
    if "ready" not in events:
        tail = (proc.stderr or "").strip().splitlines()[-2:]
        print(f"BAD 未收到 ready 事件(exit={proc.returncode}) {' '.join(tail)[:120]}")
        return 1
    if not step_ok:
        print(f"BAD 收到 ready 但 step 无观察输出 events={events}")
        return 1
    print(f"OK ready+step 均正常 events={'/'.join(e for e in events if e)}")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print("BAD 用法: sw_probe.py rows <data_dir> | sw_probe.py smoke")
        return 2
    what = sys.argv[1]
    if what == "rows":
        if len(sys.argv) < 3:
            print("BAD rows 需要 <data_dir>")
            return 2
        return cmd_rows(sys.argv[2])
    if what == "smoke":
        return cmd_smoke()
    print(f"BAD 未知子命令 {what}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
