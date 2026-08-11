"""Terminal dataset builder — TMax-15K(ai2 tmax)→ 训练 parquet。

输入(本地文件,不走网络;资产已在 experiments/bench_env/data/tmax/):
  --src        tmax15k.parquet(14,601 行:messages/ground_truth/env_config)
  --task-data  解包后的 per-task 资产根(<task_id>/{instruction.md,setup.sh,
               tests/test.sh};tar 用 tar -xzf tmax-task-data.tar.gz)

忠实性断言(任一失败即 FATAL——数据版本漂移必须可见):
  * messages 恒 [system, user] 两条;system 与 adapter.SYSTEM_PROMPT 逐字节相等;
  * ground_truth == env_config.task_id 且全局唯一;
  * 每个 task_id 在 task-data 下三件资产齐全;
  * user_prompt **逐字嵌入**(实测 max 6,476 字符——core 的 truncate_obs(4000)
    会截 53% 任务的说明书,任务表面损毁,故绝不套用)。

切分:TMax-15K 无官方切分(主终点 TB2.0 在库外);--val-count 从语料按
seed 抽出 in-training val 池(默认 128——每 val step 都要开等量云沙箱,
池子大小=真金白银;terminal-bench 89 任务另走 run_eval,不进本 parquet)。

运行:
  python -m trace_opd.envs.terminal.build_dataset \
    --src .../tmax15k.parquet --task-data .../task-data --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

from .adapter import SYSTEM_PROMPT


def _row(task_id: str, user_prompt: str, split: str) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "data_source": "terminal",
        "agent_name": "trace_opd",
        "reward_model": {"style": "env", "ground_truth": ""},
        "extra_info": {
            "task_id": task_id,
            "task_type": "terminal",
            "split": split,
            "user_prompt": user_prompt,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="tmax15k.parquet 路径")
    ap.add_argument("--task-data", required=True, help="解包的 per-task 资产根")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-count", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    task_data = Path(args.task_data)

    src = pd.read_parquet(args.src, columns=["messages", "ground_truth", "env_config"])
    rows: list[tuple[str, str]] = []
    missing_assets: list[str] = []
    for rec in src.itertuples(index=False):
        msgs = list(rec.messages)
        if len(msgs) != 2 or msgs[0]["role"] != "system" or msgs[1]["role"] != "user":
            raise SystemExit(f"FATAL: unexpected messages shape for {rec.ground_truth}")
        if msgs[0]["content"] != SYSTEM_PROMPT:
            raise SystemExit(
                "FATAL: system prompt drifted from adapter.SYSTEM_PROMPT — "
                f"got head={msgs[0]['content'][:80]!r}"
            )
        task_id = str(rec.ground_truth)
        if task_id != rec.env_config["task_id"]:
            raise SystemExit(f"FATAL: ground_truth != env_config.task_id ({task_id})")
        d = task_data / task_id
        if not ((d / "setup.sh").is_file() and (d / "tests" / "test.sh").is_file()
                and (d / "instruction.md").is_file()):
            missing_assets.append(task_id)
            continue
        rows.append((task_id, str(msgs[1]["content"])))

    ids = [t for t, _ in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("FATAL: duplicate task_id in source parquet")
    if missing_assets:
        raise SystemExit(
            f"FATAL: {len(missing_assets)} tasks missing assets under "
            f"{task_data} (first: {missing_assets[:5]}) — re-unpack the tarball"
        )

    rng = random.Random(args.seed)
    val_ids = set(rng.sample(ids, args.val_count)) if args.val_count else set()
    train_rows = [_row(t, u, "train") for t, u in rows if t not in val_ids]
    val_rows = [_row(t, u, "valid") for t, u in rows if t in val_ids]

    pd.DataFrame(train_rows).to_parquet(out / "train.parquet")
    pd.DataFrame(val_rows).to_parquet(out / "valid.parquet")
    src_sha = hashlib.sha1(Path(args.src).read_bytes()).hexdigest()[:16]
    (out / "manifest.json").write_text(json.dumps({
        "src": str(args.src), "src_sha1_16": src_sha,
        "task_data": str(task_data),
        "counts": {"train": len(train_rows), "valid": len(val_rows)},
        "val_seed": args.seed, "val_count": args.val_count,
        "note": "val 池是语料内抽样(TMax-15K 无官方切分);"
                "主终点 terminal-bench 2.0 走 run_eval,不在本 parquet",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[terminal] train={len(train_rows)} valid={len(val_rows)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
