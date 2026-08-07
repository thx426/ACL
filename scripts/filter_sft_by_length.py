#!/usr/bin/env python3
"""按真实 token 长度过滤 SFT parquet(阶段C 前置)。

为什么必须过滤而不是截断:verl 的 `data.truncation=error` 是 fail-loud 设计——
MultiTurnSFTDataset 逐轮 apply_chat_template 再拼接,assistant-only loss mask
与 token 位置严格对齐;截断会让 mask 与内容错位,训出来的东西无法解释。所以
超长样本只能整条剔除。

实测(162 v4,800 会话):整会话 token 中位 29737、p95 73225、max 130411 ——
绝大部分 token 是工具返回(ALFWorld 观测+叙述),assistant 只占中位 5318。
可行上限由 logits 显存(vocab 151936)支配,须实测,不可推算。

用法:
  python scripts/filter_sft_by_length.py --in <parquet> --model <hf> \
      --max-length 49152 --out <parquet> [--stats-only]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def token_len(tok, messages: list[dict]) -> int:
    """与 MultiTurnSFTDataset 同口径:逐轮编码后相加。

    注意 apply_chat_template(tokenize=True) 返回 BatchEncoding(字典),
    len() 数的是键数而不是 token 数 —— 这里显式取 input_ids 并解包批维。
    """
    total = 0
    for m in messages:
        enc = tok.apply_chat_template([m], tokenize=True, add_generation_prompt=False)
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        total += len(ids)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-length", type=int, required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--messages-key", default="messages")
    a = ap.parse_args()

    df = pd.read_parquet(a.src)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    lens = [token_len(tok, list(r)) for r in df[a.messages_key]]
    df = df.assign(_tok_len=lens)

    srt = sorted(lens)
    n = len(srt)
    def q(p):
        return srt[min(int(p * n), n - 1)]
    print(f"  样本 {n} 条 | token 中位 {q(.5)} p90 {q(.9)} p95 {q(.95)} max {srt[-1]}")
    for cap in (16384, 24576, 32768, 40960, 49152, 65536, 81920, 131072):
        keep = sum(1 for x in lens if x <= cap)
        print(f"    max_length={cap:>7}: 保留 {keep:>4}/{n} ({100*keep/n:>5.1f}%)")

    kept = df[df._tok_len <= a.max_length].drop(columns=["_tok_len"])
    fam = None
    for c in ("family", "task_family"):
        if c in kept.columns:
            fam = kept[c].value_counts().to_dict()
            break
    print(f"\n  选定 max_length={a.max_length} ⇒ 保留 {len(kept)}/{n} "
          f"({100*len(kept)/n:.1f}%)")
    if fam:
        print(f"  family 分布: {fam}")
    if len(kept) < 300:
        print(f"  ⚠ 保留样本 {len(kept)} < 300,SFT 数据量偏薄", file=sys.stderr)

    if a.stats_only:
        return 0
    out = a.out or a.src.replace(".parquet", f".le{a.max_length}.parquet")
    kept.to_parquet(out, index=False)
    print(f"  已写出 {out}")
    Path(out + ".filter.json").write_text(json.dumps({
        "source": a.src, "max_length": a.max_length,
        "total": n, "kept": len(kept), "family": fam,
        "token_p50": q(.5), "token_p95": q(.95), "token_max": srt[-1],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
