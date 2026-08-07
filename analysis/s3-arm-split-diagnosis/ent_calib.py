#!/usr/bin/env python3
"""从 dump 的 pi_top16 标定学生截断熵分布(top-16 口径,与 v3.3
ENTROPY_TOPK 一致)。用法: python3 ent_calib.py <run_dir> <step> [...]

⚠ 采样面警告(审阅 L29):数据源是 ledger 记录位(η>0 行)。maxopen/
always_on run(记录≈全部生成位)给出无偏分位;从 adaptive(D 门选中的
~10% 位)或 entropy 档(w≥w_min 位)的 dump 复标时,样本对门控信号
条件化,分位有选择偏差——跨档复标请优先用 steer_ent_hist 全量直方图
(v3.3 起随账落盘,覆盖含 bypass 的全部执行位)。"""
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
