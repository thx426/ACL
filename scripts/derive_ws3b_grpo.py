#!/usr/bin/env python3
"""从 webshop_3b_pgs.yaml 程序化派生纯 GRPO 基线臂。

纪律(与 derive_alf3b_grpo.py 同):**逐锚点替换,锚点非唯一即中止** ——
手抄配置会悄悄漂移 train 段,而基线臂的全部价值在于"除机制外逐键相同"。
差异恰好等于机制本身:adv_estimator / steering / researcher-doc 三项。

用法: python3 scripts/derive_ws3b_grpo.py [--check]
  --check 只校验既有产物是否仍与源一致(不写文件),供回归用。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "configs/classic/webshop_3b_pgs.yaml"
DST = ROOT / "configs/classic/webshop_3b_grpo.yaml"

HEADER_OLD = """# WebShop × Qwen2.5-3B-Instruct — PGS 机制臂(2026-08-16 用户裁定模型档 3B)
#"""
HEADER_NEW = """# WebShop × Qwen2.5-3B-Instruct — **纯 GRPO 基线臂**(2026-08-16)
#
# 由 webshop_3b_pgs.yaml 程序化派生(scripts/derive_ws3b_grpo.py,锚点唯一
# 即中止):train 段的 batch/n/lr/caps/model/数据/eval/存盘**逐键不动**,
# 与机制臂的差异恰好等于机制本身 —— adv_estimator / steering /
# researcher-doc 三项。对照对象:classic_ws3b_pgs。
#"""

REPLACEMENTS: list[tuple[str, str]] = [
    (HEADER_OLD, HEADER_NEW),
    ("  name: classic_ws3b_pgs", "  name: classic_ws3b_grpo"),
    ("  experiment_name: classic_ws3b_pgs",
     "  experiment_name: classic_ws3b_grpo"),
    ("  adv_estimator: pgs_asym_grpo",
     "  adv_estimator: grpo        # 基线臂:纯 GRPO 底座"),
]


def _cut_block(text: str, start_marker: str, end_marker: str, new: str) -> str:
    """把 [start_marker, end_marker) 之间整段换成 new。两个 marker 都必须唯一。"""
    for m in (start_marker, end_marker):
        if text.count(m) != 1:
            sys.exit(f"FATAL: 锚点命中 {text.count(m)} 次(需恰好 1): {m!r}")
    i = text.index(start_marker)
    j = text.index(end_marker)
    if j <= i:
        sys.exit("FATAL: 锚点顺序颠倒")
    return text[:i] + new + text[j:]


def build() -> str:
    text = SRC.read_text(encoding="utf-8")
    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            sys.exit(f"FATAL: 锚点命中 {text.count(old)} 次(需恰好 1): {old!r}")
        text = text.replace(old, new)

    # steering 整段 → 关闭(基线臂零特权信号)
    text = _cut_block(
        text,
        "# ---- PGS 机制(采样端特权",
        "# ---- 在线研究员",
        "# ---- 基线臂:无采样端干预、无特权 advantage ----\nsteering:\n"
        "  enabled: false\n\n",
    )
    # researcher / narration / roles 三段 → 关闭
    i = text.index("# ---- 在线研究员")
    text = text[:i] + (
        "# ---- 基线臂:无 researcher / 无 narration / 无外部角色 ----\n"
        "research:\n  enabled: false\n"
    )
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    out = build()
    if a.check:
        if not DST.exists():
            sys.exit(f"FATAL: {DST} 不存在")
        cur = DST.read_text(encoding="utf-8")
        if cur != out:
            sys.exit("FATAL: 产物与源已漂移 —— 重跑本脚本重新派生")
        print(f"[check] {DST.name} 与源一致 ✓")
        return 0
    DST.write_text(out, encoding="utf-8")
    print(f"[ok] {DST}  ({len(out.splitlines())} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
