#!/usr/bin/env python3
"""从 s3 dump 的 pi_top16 标定学生截断熵分布(top-16 口径,与 v3.3
ENTROPY_TOPK 一致)。用法: python3 ent_calib.py <run_dir> <step> [...]"""
import glob, json, math, os, sys

def main():
    run, steps = sys.argv[1], [int(x) for x in sys.argv[2:]]
    ents = []
    for s in steps:
        files = sorted(glob.glob(os.path.join(run, "rollouts", f"step_{s:05d}", "*.json")))
        for fp in files:
            try:
                d = json.load(open(fp))
            except Exception:
                continue
            st = d.get("steering", {})
            if st.get("steer_arm") != 1:
                continue
            for top in st.get("steer_pi_top16") or []:
                if not top:
                    continue
                h = 0.0
                for _tid, lp in top:
                    h -= math.exp(lp) * lp
                ents.append(h)
    ents.sort()
    n = len(ents)
    def q(p): return ents[min(n - 1, int(p * n))] if n else None
    print(json.dumps({
        "n": n,
        "P50": q(.50), "P75": q(.75), "P90": q(.90),
        "P95": q(.95), "P99": q(.99), "mean": sum(ents)/n if n else None,
    }, indent=1))

main()
