#!/usr/bin/env python3
"""θ_D 标定 —— 聚合标定 run 的 D 观测直方图,输出分位数与建议 theta_d。

标定运行形态(规范 §2.3 的目标语境预扫描):
  steering.enabled=true + steering.theta_d=1e9(门恒关哨兵)
  + steering.epsilon=0(探针也关——标定 run 必须零干预)
  + steering.gate_rate_bounds=[0.0, 1.0](豁免开门率告警——恒关是标定的本意)
  ⇒ 行为 = 纯 bare rollout(η 恒 0、W_beh 恒 1),但 verify 照跑、
    D 逐候选位置照算,直方图随 rollout dump 落盘。

跑几个 step 后:
  python scripts/calibrate_theta_d.py --rollouts <run_dir>/rollouts [--q 0.85]

输出建议值写入正式 run 的 steering.theta_d(冻结语义:标定一次、全程不动;
探针与 robust_quantile_update 属于后续版本的慢更新通道)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 与 ledger 的直方图口径一致(单一真源:steering.ledger;上界随账落盘)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trace_opd.steering.ledger import D_HIST_BINS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="<run_dir>/rollouts")
    ap.add_argument("--q", type=float, default=0.85, help="目标分位数(规范 P85)")
    args = ap.parse_args()

    hist = [0] * D_HIST_BINS
    n_traj = n_files = 0
    total = 0
    d_mean_sum = 0.0
    hist_max = None
    for f in sorted(Path(args.rollouts).glob("step_*/**/*.json")):
        n_files += 1
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        st = payload.get("steering") or {}
        h = st.get("steer_d_hist")
        if not h:
            continue
        # 审阅 IF-12:坐标系(上界=c²)随账落盘;不同坐标系的账不可混合聚合
        hm = float(st.get("steer_d_hist_max", 6.25))
        if hist_max is None:
            hist_max = hm
        elif abs(hm - hist_max) > 1e-9:
            raise SystemExit(
                f"FATAL: mixed histogram coordinate systems ({hist_max} vs {hm}) "
                f"in {f} — calibrate one run at a time"
            )
        n_traj += 1
        c = int(st.get("steer_d_count", sum(h)))
        total += c
        d_mean_sum += float(st.get("steer_d_mean", 0.0)) * c
        for i, v in enumerate(h[:D_HIST_BINS]):
            hist[i] += int(v)

    if total == 0:
        print(f"no D observations in {args.rollouts} ({n_files} files scanned) — "
              "is the run steering-enabled?")
        return 1

    if not (0.0 < args.q < 1.0):
        raise SystemExit(f"FATAL: --q must be in (0,1), got {args.q}")

    def quantile(q: float) -> tuple[float, bool]:
        """返回 (值, 是否落在末桶)。末桶同时含合法尾区与溢出,值不可信。"""
        target = q * total
        acc = 0
        for i, v in enumerate(hist):
            acc += v
            if acc >= target:
                lo = i * hist_max / D_HIST_BINS
                frac = 1.0 - (acc - target) / v if v else 0.0
                return lo + frac * (hist_max / D_HIST_BINS), i == D_HIST_BINS - 1
        return hist_max, True

    print(f"trajectories: {n_traj}   D observations: {total}   hist_max: {hist_max}")
    print(f"D mean: {d_mean_sum / total:.4f}")
    for q in (0.5, 0.75, args.q, 0.95, 0.99):
        val, ovf = quantile(q)
        print(f"P{int(q * 100):02d}: {val:.4f}" + ("  (in overflow bucket — floor only)" if ovf else ""))
    val, ovf = quantile(args.q)
    if ovf:
        # 审阅 IF-12:目标分位落进末桶时桶内插值没有辨识依据——拒绝给出
        # 建议值,而不是输出一个会被误用的 floor。
        print(f"\nFATAL: target P{int(args.q*100)} falls in the overflow bucket "
              f"(> {hist_max * (D_HIST_BINS - 1) / D_HIST_BINS:.3f}) — no suggested "
              "theta_d. Rerun calibration with a larger c (hist range = c^2).")
        return 2
    print(f"\nsuggested steering.theta_d: {val:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
