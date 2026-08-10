#!/usr/bin/env python3
"""9B 效果分析:分臂 × family,以及文档与结果的关系。

口径提醒:
- 能力读数只认 won_bare_rate(bare 臂);δ gap 是教师增益,不是 policy 变强。
- 单步 rollout 判不了改善(MDE 大、任务不重复),这里只做**当步结构**描述,
  趋势要等多步 + val。
"""
import glob, json, os, re, sys, statistics as S
from collections import defaultdict, Counter

run = sys.argv[1]
steps = sys.argv[2:] or None
if not steps:
    steps = [os.path.basename(d).split("_")[1]
             for d in sorted(glob.glob(os.path.join(run, "rollouts", "step_[0-9]*")))]

def family(task_id):
    m = re.search(r"/([a-z_]+)-", str(task_id) or "")
    if m: return m.group(1)
    m = re.search(r"__([a-z_]+)-", str(task_id) or "")
    return m.group(1) if m else "?"

for st in steps:
    d = os.path.join(run, "rollouts", f"step_{int(st):05d}")
    fs = sorted(glob.glob(os.path.join(d, "*.json")))
    if not fs: continue
    rows = [json.load(open(f)) for f in fs]
    print("=" * 74)
    print(f"step {int(st)}   n={len(rows)}")

    arms = defaultdict(list)
    for r in rows:
        a = (r.get("steering") or {}).get("steer_arm")
        if a is not None: arms[int(a)].append(r)

    print("\n── 分臂总览 ──")
    for a, nm in ((0, "bare  "), (1, "steer ")):
        v = arms.get(a) or []
        if not v: continue
        w = [int(bool(x.get("won"))) for x in v]
        L = [x.get("response_token_len") or 0 for x in v]
        at = [x.get("assistant_turns") or 0 for x in v]
        es = [x.get("env_steps") or 0 for x in v]
        ivf = [int(x.get("invalid_format") or 0) for x in v]
        tr = [int(bool(x.get("truncated"))) for x in v]
        oc = Counter(x.get("outcome") for x in v)
        print(f"  {nm} n={len(v):>3}  won {sum(w)/len(w)*100:5.1f}%  "
              f"respLen中位 {int(S.median(L)):>5}  turns中位 {int(S.median(at)):>3}  "
              f"env_steps中位 {int(S.median(es)):>3}")
        print(f"          invalid_format 均值 {sum(ivf)/len(ivf):.2f}  截断 {sum(tr)}/{len(tr)}  "
              f"outcome {dict(oc)}")
    if 0 in arms and 1 in arms:
        b = sum(int(bool(x.get("won"))) for x in arms[0]) / len(arms[0])
        s = sum(int(bool(x.get("won"))) for x in arms[1]) / len(arms[1])
        print(f"\n  能力读数 won_bare = {b*100:.1f}%   教师增益 δ = {(s-b)*100:+.1f}pp")
        print("  (δ 大只说明文档在起作用,不等于 policy 变强;policy 看 won_bare 随步数与 val)")

    print("\n── 按 family 分臂(留意组内差异) ──")
    fam = defaultdict(lambda: {0: [], 1: []})
    for a in (0, 1):
        for x in arms.get(a) or []:
            fam[family(x.get("task_id"))][a].append(int(bool(x.get("won"))))
    print(f"  {'family':<28} {'n_b':>4} {'bare':>7} {'n_s':>4} {'steer':>7} {'δ':>7}")
    for k in sorted(fam):
        b, s = fam[k][0], fam[k][1]
        if not b or not s: continue
        bb, ss = sum(b)/len(b), sum(s)/len(s)
        print(f"  {k:<28} {len(b):>4} {bb*100:6.1f}% {len(s):>4} {ss*100:6.1f}% {(ss-bb)*100:+6.1f}pp")

    print("\n── researcher 与结果 ──")
    docw, fs_, ex = [], [], []
    for r in rows:
        rs = r.get("research") or {}
        if rs.get("research_doc_words"): docw.append(rs["research_doc_words"])
        if rs.get("research_final_success") is not None: fs_.append(rs["research_final_success"])
        if rs.get("research_experiments"): ex.append(rs["research_experiments"])
    if docw:
        print(f"  文档词数 中位 {int(S.median(docw))}  范围 [{min(docw)}, {max(docw)}]")
        print(f"  researcher 末次实验成功率 均值 {sum(fs_)/len(fs_)*100:.1f}%  experiments 均值 {sum(ex)/len(ex):.1f}")
    # 文档词数 vs steered 胜率(按任务聚合)
    per_task = defaultdict(lambda: {"doc": None, "b": [], "s": []})
    for r in rows:
        t = r.get("task_id"); a = (r.get("steering") or {}).get("steer_arm")
        rs = r.get("research") or {}
        if rs.get("research_doc_words"): per_task[t]["doc"] = rs["research_doc_words"]
        if a == 0: per_task[t]["b"].append(int(bool(r.get("won"))))
        elif a == 1: per_task[t]["s"].append(int(bool(r.get("won"))))
    pts = [(v["doc"], sum(v["s"])/len(v["s"]) - sum(v["b"])/len(v["b"]))
           for v in per_task.values() if v["doc"] and v["b"] and v["s"]]
    if len(pts) >= 4:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        mx, my = S.mean(xs), S.mean(ys)
        num = sum((x-mx)*(y-my) for x, y in pts)
        den = (sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys)) ** 0.5
        print(f"  文档词数 vs 任务级 δ 的相关: r={num/den if den else 0:+.3f}  (n={len(pts)} 任务)")
        print("  (样本极小,只作描述,不作推断)")
