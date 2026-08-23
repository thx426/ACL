#!/usr/bin/env python3
"""双臂 val 对照(avg@3 权威口径:每题 3 次,按 split 分桶,只算跑满 3 次的题)。"""
import collections, glob, json, os

RUNS = {
    "PGS(我们的机制)": "/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM/runs/classic_alf3b_pgs",
    "SEED(对照臂)": "/fs/fast/u2021000902/wushang/workspace/TRACE-OPD/trace_opd/runs/classic_alf3b_seed",
}
res = {}
for lbl, run in RUNS.items():
    for d in sorted(glob.glob(os.path.join(run, "rollouts", "step_*_val"))):
        recs = []
        for f in glob.glob(os.path.join(d, "*.json")):
            if os.path.getsize(f):
                try:
                    recs.append(json.load(open(f)))
                except Exception:
                    pass
        if not recs:
            continue
        step = int(os.path.basename(d).split("_")[1])
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in recs:
            by[r.get("split") or "?"][r.get("task_id")].append(r)
        row = {}
        for sp in sorted(by):
            full = {t: v for t, v in by[sp].items() if len(v) == 3}
            if not full:
                continue
            wins = [sum(1 for x in v if x.get("won")) for v in full.values()]
            row[sp] = (100 * sum(wins) / (3 * len(full)),
                       100 * sum(1 for w in wins if w >= 1) / len(full),
                       len(full))
        res.setdefault(step, {})[lbl] = row

print(f"{'step':<6}{'arm':<18}{'seen mean@3':>13}{'unseen mean@3':>15}"
      f"{'seen best@3':>13}{'unseen best@3':>15}")
print("-" * 80)
for step in sorted(res):
    for lbl in RUNS:
        r = res[step].get(lbl)
        if not r:
            continue
        s = r.get("valid_seen") or (0, 0, 0)
        u = r.get("valid_unseen") or (0, 0, 0)
        print(f"{step:<6}{lbl:<18}{s[0]:>12.2f}%{u[0]:>14.2f}%{s[1]:>12.2f}%{u[1]:>14.2f}%")
    print()
print("对照锚点(classic-bench-baselines):GRPO 主锚 ALFWorld-seen 77.6")
