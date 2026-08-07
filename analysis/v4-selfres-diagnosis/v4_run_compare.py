#!/usr/bin/env python3
"""两个 run 的 research 行对比,控制"部分数据的幸存者偏差"。

用法: v4_run_compare.py <runA_rollout_dir> <runB_rollout_dir> [N]
N 给定时,两侧都按 ts 排序只取前 N 条 —— 这样"先完成的偏短"这个偏差在
两侧同向,差异才可归因到配置/代码而非采样进度。
另外做一次 task_id 配对比较(只用两侧都有的任务),进一步去掉任务难度差异。
"""
import glob
import json
import statistics as st
import sys
from collections import defaultdict


def load(d):
    out = []
    for f in sorted(glob.glob(d + "/*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if r.get("sample_role") == "research":
            out.append(r)
    out.sort(key=lambda r: r.get("ts") or 0)
    return out


def rf(r, k):
    return (r.get("research") or {}).get(k)


def summarize(rows, label):
    if not rows:
        print(f"  {label}: 无数据")
        return
    n = len(rows)
    err = sum(1 for r in rows if rf(r, "research_error"))
    nf = sum(1 for r in rows if rf(r, "research_no_finalize"))
    ov = sum(1 for r in rows if rf(r, "research_overlong"))
    pos = sum(1 for r in rows if (rf(r, "research_r_raw") or 0) > 0)
    L = [r.get("response_token_len") or 0 for r in rows]
    T = [rf(r, "research_turns") or 0 for r in rows]
    E = [rf(r, "research_experiments") or 0 for r in rows]
    print(f"  {label:<22} n={n:>3}  error {100*err/n:>5.1f}%  no_fin {100*nf/n:>5.1f}%  "
          f"overlong {100*ov/n:>5.1f}%  r_raw>0 {100*pos/n:>5.1f}%")
    print(f"  {'':22}      respLen中位 {st.median(L):>7.0f}  turns中位 {st.median(T):>4.1f}  "
          f"exps中位 {st.median(E):>4.1f}")


A, B = load(sys.argv[1]), load(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else min(len(A), len(B))

print(f"runA {len(A)} 条 / runB {len(B)} 条;同偏窗口 N={N}\n")
print("=== ① 全量(若一侧未跑完则不可比,仅列出) ===")
summarize(A, "runA 全量")
summarize(B, "runB 全量")

print(f"\n=== ② 同偏窗口:两侧各取按 ts 排序的前 {N} 条 ===")
summarize(A[:N], f"runA 前{N}")
summarize(B[:N], f"runB 前{N}")

print("\n=== ③ task_id 配对(只用两侧都有的任务) ===")
ta = defaultdict(list)
tb = defaultdict(list)
for r in A:
    ta[r.get("task_id")].append(r)
for r in B:
    tb[r.get("task_id")].append(r)
common = sorted(set(ta) & set(tb))
print(f"  共同任务 {len(common)} 个 (runA {len(ta)} / runB {len(tb)})")
if common:
    pa = [r for t in common for r in ta[t]]
    pb = [r for t in common for r in tb[t]]
    summarize(pa, "runA 配对")
    summarize(pb, "runB 配对")
    print("\n  逐任务 finalize 成功率 (runA → runB):")
    for t in common:
        fa = sum(1 for r in ta[t] if not rf(r, "research_no_finalize"))
        fb = sum(1 for r in tb[t] if not rf(r, "research_no_finalize"))
        short = t.split("/")[-1][:44] if "/" in t else t[:44]
        print(f"    {short:<46} {fa}/{len(ta[t])} → {fb}/{len(tb[t])}")
