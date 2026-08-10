#!/usr/bin/env python3
"""researcher 在降预算(n_exp 16→8, k 4→2)后的效果评估。

三问:
  1. 预算实际消耗了多少(验证降预算真的生效,而非模型本来就没用满)
  2. 实验过程中 episode 成功率有没有随 guidance 迭代上升(= 对任务的提升)
  3. 产出文档的质量(长度/结构/具体性)

用法: researcher_quality.py <research_root> [标签]
"""
import glob
import json
import os
import re
import statistics as st
import sys
from collections import Counter

ROOT = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(ROOT)

#: 第三个参数 = 只看某个 global_step 的档案(跨步混合会被后期更强的 policy 稀释,
#: 与 step1 对照必须同步对齐)
ONLY_STEP = int(sys.argv[3]) if len(sys.argv) > 3 else None

dirs = sorted(glob.glob(os.path.join(ROOT, "*", "*")))
if ONLY_STEP is not None:
    kept = []
    for d in dirs:
        mp = os.path.join(d, "meta.json")
        try:
            if json.load(open(mp)).get("global_step") == ONLY_STEP:
                kept.append(d)
        except Exception:
            pass
    dirs = kept
    TAG += f" @step{ONLY_STEP}"
print(f"=== {TAG}: {len(dirs)} 份档案 ===\n")

metas, docs, ep_curves = [], [], []
for d in dirs:
    mp, dp, ep = (os.path.join(d, x) for x in ("meta.json", "doc.md", "episodes.jsonl"))
    m = {}
    if os.path.exists(mp):
        try:
            m = json.load(open(mp))
        except Exception:
            pass
    metas.append(m)
    docs.append(open(dp, encoding="utf-8").read() if os.path.exists(dp) else "")
    # episodes.jsonl: 逐 episode 的成败,按实验序分组看趋势
    rows = []
    if os.path.exists(ep):
        for line in open(ep, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    ep_curves.append(rows)

if metas and metas[0]:
    print("meta.json 顶层键:", sorted(metas[0].keys())[:16], "\n")


def mget(m, *names, default=None):
    for n in names:
        if n in m:
            return m[n]
        for k, v in m.items():
            if isinstance(v, dict) and n in v:
                return v[n]
    return default


print("=== 1. 预算实际消耗 ===")
exps = [mget(m, "experiments_used", "n_experiments", "experiments") for m in metas]
epis = [mget(m, "episodes_consumed", "n_episodes", "episodes") for m in metas]
exps = [x for x in exps if isinstance(x, (int, float))]
epis = [x for x in epis if isinstance(x, (int, float))]
if exps:
    print(f"  实验次数: 中位 {st.median(exps):.1f}  均值 {st.mean(exps):.1f}  "
          f"max {max(exps)}  分布 {dict(sorted(Counter(exps).items()))}")
if epis:
    print(f"  episode 数: 中位 {st.median(epis):.1f}  均值 {st.mean(epis):.1f}  "
          f"max {max(epis)}  合计 {sum(epis)}")
if exps and epis:
    print(f"  ⇒ episode/实验 = {sum(epis)/max(sum(exps),1):.2f}  (k_default=2 时应≈2)")

print("\n=== 2. 实验过程中的任务成功率(= 对任务的提升) ===")
# episodes.jsonl 的成败在 trajectory.won;won 只在轨迹层
allw, first_half, last_half = [], [], []
per_exp_curve = []          # 每份档案:按实验序的成功率
for rows in ep_curves:
    ws = []
    for r in rows:
        tr = r.get("trajectory") or {}
        w = tr.get("won")
        if w is None:
            w = r.get("won")
        if w is not None:
            ws.append(bool(w))
    if not ws:
        continue
    allw.extend(ws)
    h = len(ws) // 2
    if h:
        first_half.append(sum(ws[:h]) / h)
        last_half.append(sum(ws[h:]) / len(ws[h:]))
    per_exp_curve.append(ws)
if allw:
    print(f"  全部实验 episode: {sum(allw)}/{len(allw)} = {100*sum(allw)/len(allw):.1f}% 成功")
if first_half:
    d = st.mean(last_half) - st.mean(first_half)
    print(f"  会话内前半 {100*st.mean(first_half):.1f}%  →  后半 {100*st.mean(last_half):.1f}%"
          f"  ({100*d:+.1f}pp {'✓ guidance 在迭代改进' if d > 0 else '未见改进'})")

# meta 里的权威字段:最后一次实验的 attempts/successes
la = [mget(m, "last_exp_attempts") for m in metas]
ls_ = [mget(m, "last_exp_successes") for m in metas]
pair = [(a, s) for a, s in zip(la, ls_) if isinstance(a, (int, float)) and a]
if pair:
    print(f"  最后一次实验(文档定稿依据): "
          f"{sum(s for _, s in pair)}/{sum(a for a, _ in pair)} = "
          f"{100*sum(s for _,s in pair)/max(sum(a for a,_ in pair),1):.1f}% 成功")

# guidance_log:逐次实验的成功数,看迭代曲线
buckets = {}
for m in metas:
    gl = mget(m, "guidance_log") or []
    for i, g in enumerate(gl if isinstance(gl, list) else []):
        if not isinstance(g, dict):
            continue
        a = g.get("attempts") or g.get("k")
        s = g.get("successes")
        if isinstance(a, (int, float)) and isinstance(s, (int, float)) and a:
            buckets.setdefault(i + 1, [0, 0])
            buckets[i + 1][0] += s
            buckets[i + 1][1] += a
if buckets:
    print("  逐次实验成功率(跨全部档案汇总):")
    for i in sorted(buckets):
        s, a = buckets[i]
        print(f"    第{i}次: {s:>3}/{a:>3} = {100*s/max(a,1):>5.1f}%")

print("\n=== 3. 文档质量 ===")
wl = [len(d.split()) for d in docs if d]
cl = [len(d) for d in docs if d]
if wl:
    print(f"  词数: 中位 {st.median(wl):.0f}  均值 {st.mean(wl):.0f}  min {min(wl)}  max {max(wl)}")
    print(f"  空文档: {sum(1 for d in docs if not d.strip())}/{len(docs)}")
    # 具体性:是否含真实动作词/受体名/episode 引用
    act = sum(1 for d in docs if re.search(r"\b(go to|take|put|open|close|heat|cool|clean|move)\b", d))
    rec = sum(1 for d in docs if re.search(r"\b\w+ \d\b", d))          # "sidetable 2" 这类实例名
    epi = sum(1 for d in docs if re.search(r"episode|attempt|trial", d, re.I))
    hdr = sum(1 for d in docs if d.count("#") >= 2)
    print(f"  含真实动作词: {act}/{len(docs)}   含实例名(如 'drawer 1'): {rec}/{len(docs)}")
    print(f"  引用实验证据: {epi}/{len(docs)}   有 markdown 结构: {hdr}/{len(docs)}")
