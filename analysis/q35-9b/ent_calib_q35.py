#!/usr/bin/env python3
"""9B 熵门标定:从 steer_ent_hist 全量直方图定 entropy_theta_init / entropy_floor。

为什么不用 ent_calib.py 的 steer_pi_top16 路径:那条路的样本是 ledger 记录位
(η>0),在 entropy 档下已对门控信号条件化,分位有选择偏差。steer_ent_hist 是
**每个生成位置一次 observe_entropy(含 bypass 段)**,是无偏的全量分布。

核心判据不是"P95 是多少",而是**当前 theta 下的门死区率** ——
[[opd-base-model-floor]] 的教训:1.7B 上 gate_mean=0.486 看着完全正常,
但 95.5% 位置门是关的,机制实际没跑。gate_mean 会骗人,死区率不会。

用法: python3 ent_calib_q35.py <run_dir> [step ...]
"""
import glob, json, os, sys

ENT_HIST_BINS = 1024
ENT_HIST_MAX = 8.0
BIN_W = ENT_HIST_MAX / ENT_HIST_BINS

# 参照系
REF_4B_P95 = 1.276        # 4B 实测,当前 theta_init=1.28 的来源
REF_1P7B_P95 = 0.043      # 1.7B 实测,熵门在此结构性失效
CUR_THETA = 1.28
CUR_FLOOR = 0.6


def load_hists(run: str, steps):
    hist = [0] * ENT_HIST_BINS
    n_files = n_used = 0
    if not steps:
        dirs = sorted(glob.glob(os.path.join(run, "rollouts", "step_*")))
        steps = [int(os.path.basename(d).split("_")[1]) for d in dirs]
    for s in steps:
        for fp in sorted(glob.glob(os.path.join(run, "rollouts", f"step_{s:05d}", "*.json"))):
            n_files += 1
            try:
                d = json.load(open(fp))
            except Exception:
                continue
            h = (d.get("steering") or {}).get("steer_ent_hist")
            if not h or len(h) != ENT_HIST_BINS:
                continue
            n_used += 1
            for i, c in enumerate(h):
                hist[i] += c
    return hist, steps, n_files, n_used


def quantile(hist, q):
    tot = sum(hist)
    if not tot:
        return None
    target, acc = q * tot, 0
    for i, c in enumerate(hist):
        acc += c
        if acc >= target:
            return (i + 0.5) * BIN_W
    return ENT_HIST_MAX


def frac_below(hist, thr):
    tot = sum(hist)
    if not tot:
        return None
    b = min(int(thr / BIN_W), ENT_HIST_BINS - 1)
    return sum(hist[:b]) / tot


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    run = sys.argv[1]
    steps = [int(x) for x in sys.argv[2:]]
    hist, steps, n_files, n_used = load_hists(run, steps)
    tot = sum(hist)
    print(f"run   = {run}")
    print(f"steps = {steps}")
    print(f"文件 {n_files} 个,含 steer_ent_hist 的 {n_used} 个;累计观测位 {tot:,}")
    if not tot:
        print("\nXX 没有熵观测数据 —— 检查 step 目录或 steering 是否真的开着")
        return 1

    print("\n════ 9B 学生截断熵分布(top-16 口径,全生成位无偏) ════")
    qs = {}
    for q in (0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        v = quantile(hist, q)
        qs[q] = v
        print(f"  P{int(q*100):<3} = {v:.4f}")
    mean = sum((i + 0.5) * BIN_W * c for i, c in enumerate(hist)) / tot
    print(f"  mean = {mean:.4f}")

    print("\n════ 与参照系对比 ════")
    p95 = qs[0.95]
    print(f"  9B   P95 = {p95:.4f}")
    print(f"  4B   P95 = {REF_4B_P95:.4f}   (当前 theta_init={CUR_THETA} 的来源)")
    print(f"  1.7B P95 = {REF_1P7B_P95:.4f}  (此值下熵门结构性失效)")
    print(f"  9B/4B 比 = {p95/REF_4B_P95:.2f}×")

    print("\n════ 关键判据:当前 4B 占位值下的门死区率 ════")
    dead_theta = frac_below(hist, CUR_THETA)
    dead_floor = frac_below(hist, CUR_FLOOR)
    print(f"  熵 < theta({CUR_THETA})  的位置占比 = {dead_theta*100:.1f}%")
    print(f"  熵 < floor({CUR_FLOOR}) 的位置占比 = {dead_floor*100:.1f}%   ← floor 也救不回来的部分")
    if dead_floor > 0.90:
        verdict = "XX 结构性失效(同 1.7B 形态):floor 之下超九成,机制实际不跑"
    elif dead_theta > 0.90:
        verdict = "!! theta 明显偏高:九成以上位置门关闭,需下调"
    elif dead_theta < 0.30:
        verdict = "!! theta 明显偏低:门几乎常开,失去选择性"
    else:
        verdict = "OK 4B 占位值在 9B 上可用(死区率在合理区间)"
    print(f"  判定: {verdict}")

    print("\n════ 建议值(按 v3.3 口径:theta_init=P95、floor 按分布形状) ════")
    print(f"  entropy_theta_init: {p95:.2f}      (= 9B 的 P95)")
    sug_floor = round(qs[0.50], 2)
    print(f"  entropy_floor:      {sug_floor:.2f}      (= P50;floor 是自适应下界,")
    print(f"                                取中位保证半数位置有机会开门)")
    print(f"\n  改这两个值会变更 objective_fingerprint —— 需用户裁定后另立正式配置。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
