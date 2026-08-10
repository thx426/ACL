#!/usr/bin/env python3
"""9B 单步为何慢:把墙钟拆到可归因的量上,不停在"模型大"。

关键分解:单步墙钟 ≈ researcher 阶段 + rollout 阶段;
rollout 阶段 ≈ (总生成 token 数) / (吞吐) + 环境交互 + teacher verify。
模型大小只影响"吞吐"这一项(9B/4B 参数比 2.25×),
若实测慢 ~7×,差额必来自 **生成 token 数** 或 **串行度**。
"""
import glob, json, os, sys, statistics as S
from collections import Counter

d = sys.argv[1] if len(sys.argv) > 1 else "."
step = sys.argv[2] if len(sys.argv) > 2 else "00001"
fs = sorted(glob.glob(os.path.join(d, "rollouts", f"step_{step}", "*.json")))
rows = [json.load(open(f)) for f in fs]
print(f"样本 {len(rows)} 行\n")

# ---------- 1. 生成量 ----------
print("═" * 64)
print("① 生成量(决定 rollout 墙钟的第一因子)")
L = [r.get("response_token_len") or 0 for r in rows]
tr = [int(bool(r.get("truncated"))) for r in rows]
es = [r.get("env_steps") or 0 for r in rows]
at = [r.get("assistant_turns") or 0 for r in rows]
L.sort()
print(f"  response_token_len  中位 {int(S.median(L))}  P90 {L[int(len(L)*0.9)]}  max {max(L)}")
print(f"  截断率 {sum(tr)/len(tr)*100:.1f}%")
print(f"  env_steps    中位 {int(S.median(es))}  max {max(es)}")
print(f"  assistant_turns 中位 {int(S.median(at))}  max {max(at)}")
tot_tok = sum(L)
print(f"  本步 executor 总生成 ≈ {tot_tok:,} token ({len(rows)} 行)")
per_turn = [l / a for l, a in zip(L, at) if a]
if per_turn:
    print(f"  每轮平均 token = {S.median(per_turn):.0f}   (max_tokens_per_turn=2048)")

# ---------- 2. researcher 量 ----------
print("═" * 64)
print("② researcher 阶段的量(它与 executor 串行)")
rex, rep, rtu, rdw = [], [], [], []
for r in rows:
    rs = r.get("research") or {}
    if rs.get("research_experiments"): rex.append(rs["research_experiments"])
    if rs.get("research_episodes"):    rep.append(rs["research_episodes"])
    if rs.get("research_turns"):       rtu.append(rs["research_turns"])
    if rs.get("research_doc_words"):   rdw.append(rs["research_doc_words"])
if rex:
    print(f"  每任务 experiments 中位 {S.median(rex):.0f}  episodes 中位 {S.median(rep):.0f}")
    print(f"  researcher 会话轮数 中位 {S.median(rtu):.0f}")
    print(f"  文档词数 中位 {S.median(rdw):.0f}")
    n_task = len(set(r.get("task_id") for r in rows))
    print(f"  本步任务数 {n_task} -> researcher 侧 episodes 合计 ≈ {n_task*S.median(rep):.0f}")
    print("  注意:这些 episode 也要真跑 ALFWorld 环境,且用的是 35B 外部端点")

# ---------- 3. teacher verify 量 ----------
print("═" * 64)
print("③ teacher verify 量(steered 臂独有,与生成同步串行)")
cand, base, ell = [], [], []
for r in rows:
    st = r.get("steering") or {}
    if st.get("steer_arm") != 1:
        continue
    cand.append(st.get("steer_n_candidates") or 0)
    base.append(st.get("steer_n_base_open") or 0)
    ell.append(st.get("steer_ell_count") or 0)
if cand:
    print(f"  每条 steered 轨迹的候选位 中位 {int(S.median(cand)):,}")
    print(f"  其中开门(需 verify)位 中位 {int(S.median(base)):,}")
    print(f"  ell 观测 token 中位 {int(S.median(ell)):,}")
    print(f"  -> steered 臂合计 verify 位 ≈ {sum(base):,}")
    print("  每个开门位都要 teacher 前向一次(hook.py:_verify_rows),这是纯额外算力")

# ---------- 4. 重复率(长度是否来自空转) ----------
print("═" * 64)
print("④ 长度是不是有效的:thought 逐字重复率")
print("   (thought-repeat-metric: 赢轨迹恒为 0,输轨迹 26–70%)")
rep_rates = []
for r in rows[:60]:
    steps = ((r.get("trajectory") or {}).get("steps")) or []
    ths = []
    for s in steps:
        t = s.get("thought") or s.get("reasoning") or ""
        if isinstance(t, str) and t.strip():
            ths.append(t.strip())
    if len(ths) >= 4:
        c = Counter(ths)
        dup = sum(v - 1 for v in c.values() if v > 1)
        rep_rates.append(dup / len(ths))
if rep_rates:
    rep_rates.sort()
    print(f"  样本 {len(rep_rates)} 条;重复率 中位 {S.median(rep_rates)*100:.1f}%  "
          f"P90 {rep_rates[int(len(rep_rates)*0.9)]*100:.1f}%  max {max(rep_rates)*100:.1f}%")
    hi = sum(1 for x in rep_rates if x > 0.55)
    print(f"  重复率 >55% 的轨迹: {hi}/{len(rep_rates)} = {hi/len(rep_rates)*100:.0f}%")
    print("  -> 若高,说明长度大量来自原地打转,而非真在解题")
else:
    print("  (轨迹里取不到 thought 字段,跳过)")

# ---------- 5. 归因 ----------
print("═" * 64)
print("⑤ 与 4B 线对照(4B v3.3 单步实测 14.7 分钟)")
print(f"  9B respLen 中位 {int(S.median(L)):,}  vs  4B s3 线 respLen 2075(s35)→5942(s48)")
ratio = S.median(L) / 5942
print(f"  -> 生成量比 ≈ {ratio:.1f}×(取 4B 的高值 5942 作保守分母)")
print(f"  -> 参数比 2.25×,若吞吐 ∝ 参数,合计预期 ≈ {ratio*2.25:.1f}× 的 rollout 时间")
print("  另有两项 4B 没有的固定开销:")
print("    · mamba 层使 SGLang 禁用 overlap schedule(日志实证)")
print("    · 首次 triton JIT 编译 seq4096 档 +13s(只在首步)")
