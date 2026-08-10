#!/usr/bin/env python
"""从既有 SSB parquet 中剔除 reward 恒零的任务,产出过滤后的数据集。

与 `envs/spreadsheetbench/build_dataset.py --（不加）--keep-unreliable` 殊途同归,
差别只在**输入**:

  · build_dataset  : 从 dataset.json 重新渲染每行(含 initial_obs)。适合从零构建。
  · 本脚本(过滤)   : 只按 task_id 删行,其余字段逐字节原样保留。

已有实验(A800 的 9B GRPO 对照臂)正跑在既有 parquet 上,而 initial_obs 是
openpyxl 渲染进 parquet 的 —— 重新渲染一旦与当时环境有丝毫差异,两臂的 prompt
就不同源、对照被破坏。故凡是"已有 parquet 且要保持可比"的场合一律用本脚本;
从零构建新数据集才走 build_dataset。两者共用同一份清单
(envs/spreadsheetbench/unreliable_tasks.json),不存在第二套真源。

用法:
  python scripts/ssb_filter_parquet.py --in <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from trace_opd.envs.spreadsheetbench.build_dataset import (  # noqa: E402
    load_unreliable)


def task_ids(df: pd.DataFrame) -> list[str]:
    out = []
    for x in df["extra_info"]:
        v = x if isinstance(x, dict) else json.loads(x)
        out.append(str(v.get("task_id")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "valid"])
    args = ap.parse_args()

    bad, doc = load_unreliable()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    print(f"[filter] 清单 {len(bad)} 个不可靠任务 "
          f"({doc['判定环境']['libreoffice'][:24]}… / {doc['决定'][:24]}…)")

    counts, removed_all = {}, {}
    for split in args.splits:
        p = src / f"{split}.parquet"
        if not p.is_file():
            print(f"FATAL: 缺 {p}", file=sys.stderr)
            return 2
        df = pd.read_parquet(p)
        ids = task_ids(df)
        keep = [i for i, t in enumerate(ids) if t not in bad]
        removed = sorted({t for t in ids if t in bad})
        out_df = df.iloc[keep].reset_index(drop=True)
        out_df.to_parquet(dst / f"{split}.parquet")
        counts[split] = {"before": len(df), "after": len(out_df),
                         "removed": len(df) - len(out_df)}
        removed_all[split] = removed
        print(f"[filter] {split}: {len(df)} → {len(out_df)} "
              f"(剔除 {len(df) - len(out_df)}) -> {dst / f'{split}.parquet'}")

    manifest = {
        "source": str(src),
        "method": ("按 task_id 从既有 parquet 删行;其余字段(含 initial_obs)"
                   "逐字节原样保留 ⇒ 与源数据集是严格子集关系,prompt 同源"),
        "counts": counts,
        "removed_task_ids": removed_all,
        "unreliable_list": {
            "source": "envs/spreadsheetbench/unreliable_tasks.json",
            "n": len(bad),
            "决定": doc["决定"],
            "判定环境": doc["判定环境"],
        },
    }
    (dst / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[filter] manifest -> {dst / 'split_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
