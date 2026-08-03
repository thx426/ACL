#!/usr/bin/env python3
"""GT-doc 消融实验(docs/gt-doc-ablation.md)的两项地基预检。

预检 1 — 普查(离线,只读 parquet):
  gt_text 覆盖率、形态构成(walkthrough 可执行 / high_pddl 高层 / 其他)、
  render_gt_as_doc 渲染成功率、doc 长度分布。walkthrough 覆盖 <95% ⇒ 数据
  须重建(build_datasets 的 walkthrough 通道),不许带着高层计划形态起跑。

预检 2 — 重放验证(--replay N,需环境,在训练机上跑):
  抽 N 个渲染成功的任务,起真实环境逐条执行 walkthrough 命令,报成功率与
  失败样本。GT 解的正确性是整个消融的地基:重放不 ~100% 赢,消融读数无效。

用法:
  python3 scripts/gt_doc_precheck.py --train-file <train.parquet>
  python3 scripts/gt_doc_precheck.py --train-file <...> --replay 30 [--seed 7]
"""
import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from trace_opd.envs.alfworld.gt import (  # noqa: E402
    _WALKTHROUGH_HEADER,
    render_gt_as_doc,
)


def _iter_items(train_file: str):
    import pandas as pd

    df = pd.read_parquet(train_file)
    col = "extra_info" if "extra_info" in df.columns else None
    if col is None:
        raise SystemExit(f"no extra_info column in {train_file} (columns: {list(df.columns)})")
    for _, row in df.iterrows():
        info = row[col]
        if isinstance(info, str):
            info = json.loads(info)
        yield dict(info)


def census(train_file: str):
    total = 0
    empty, walkthrough, high_pddl, other = 0, 0, 0, 0
    rendered, doc_words = 0, []
    items = []
    for info in _iter_items(train_file):
        total += 1
        items.append(info)
        text = str(info.get("gt_text") or "").strip()
        if not text:
            empty += 1
            continue
        if text.startswith(_WALKTHROUGH_HEADER):
            walkthrough += 1
        elif text.startswith("High-level expert plan"):
            high_pddl += 1
        else:
            other += 1
        doc = render_gt_as_doc(text)
        if doc:
            rendered += 1
            doc_words.append(len(doc.split()))
    doc_words.sort()

    def pct(q):
        return doc_words[int(q * (len(doc_words) - 1))] if doc_words else 0

    print(f"rows                    {total}")
    print(f"gt_text empty           {empty}  ({empty / max(total,1):.1%})")
    print(f"walkthrough (playable)  {walkthrough}  ({walkthrough / max(total,1):.1%})")
    print(f"high_pddl (abstract)    {high_pddl}  ({high_pddl / max(total,1):.1%})")
    print(f"other form              {other}")
    print(f"render_gt_as_doc ok     {rendered}  ({rendered / max(total,1):.1%})")
    if doc_words:
        print(f"doc words p50/p90/max   {pct(0.5)}/{pct(0.9)}/{doc_words[-1]}")
    verdict = "PASS" if rendered / max(total, 1) >= 0.95 else "FAIL (<95% — rebuild data with walkthrough GT)"
    print(f"precheck-1 verdict      {verdict}")
    return items


def replay(items, n, seed, config_path):
    """预检 2:真实环境逐条重放 walkthrough,报成功率。"""
    from trace_opd.common.config import load_config
    from trace_opd.envs.alfworld.adapter import AlfworldEnvFactory

    cfg = load_config(config_path)
    cand = [i for i in items if render_gt_as_doc(i.get("gt_text"))]
    random.Random(seed).shuffle(cand)
    cand = cand[:n]
    factory = AlfworldEnvFactory(cfg.alfworld)
    won_n, fail = 0, []
    for info in cand:
        adapter = factory.adapter_from_item(info)
        steps = [
            line.split(". ", 1)[1]
            for line in str(info["gt_text"]).strip().splitlines()[1:]
            if ". " in line
        ]
        handle = adapter.open_episode()
        won = False
        try:
            for cmd in steps:
                _obs, done, won = handle.step(adapter.normalize_action(cmd))
                if done:
                    break
        finally:
            handle.close()
        if won:
            won_n += 1
        else:
            fail.append(info.get("task_id", "?"))
        print(f"  {'WON ' if won else 'LOST'}  {info.get('task_id', '?')}")
    print(f"replay: {won_n}/{len(cand)} won")
    if fail:
        print("failed tasks:")
        for t in fail:
            print(f"  {t}")
    print(f"precheck-2 verdict      {'PASS' if won_n == len(cand) else 'FAIL (GT must replay ~100%)'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--replay", type=int, default=0,
                    help="重放验证的抽样任务数(0=只做普查;需要环境)")
    ap.add_argument("--config", default=None,
                    help="trace 配置(--replay 时必需:alfworld 环境参数来源)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    items = census(args.train_file)
    if args.replay:
        if not args.config:
            raise SystemExit("--replay requires --config (alfworld env params)")
        replay(items, args.replay, args.seed, args.config)


if __name__ == "__main__":
    main()
