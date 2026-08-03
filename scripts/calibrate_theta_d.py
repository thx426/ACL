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

# 与 ledger 的直方图口径一致(单一真源:steering.ledger)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trace_opd.steering.ledger import D_HIST_BINS, D_HIST_MAX  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="<run_dir>/rollouts")
    ap.add_argument("--q", type=float, default=0.85, help="目标分位数(规范 P85)")
    args = ap.parse_args()

    hist = [0] * D_HIST_BINS
    n_traj = n_files = 0
    total = 0
    d_mean_sum = 0.0
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

    def quantile(q: float) -> float:
        target = q * total
        acc = 0
        for i, v in enumerate(hist):
            acc += v
            if acc >= target:
                # 桶内线性插值
                lo = i * D_HIST_MAX / D_HIST_BINS
                frac = 1.0 - (acc - target) / v if v else 0.0
                return lo + frac * (D_HIST_MAX / D_HIST_BINS)
        return D_HIST_MAX

    print(f"trajectories: {n_traj}   D observations: {total}")
    print(f"D mean: {d_mean_sum / total:.4f}")
    for q in (0.5, 0.75, args.q, 0.95, 0.99):
        print(f"P{int(q * 100):02d}: {quantile(q):.4f}")
    tail = hist[-1] / total
    if tail > 0.02:
        print(f"WARNING: {tail:.1%} of D mass in the overflow bucket "
              f"(> {D_HIST_MAX}) — quantiles above are floor estimates")
    print(f"\nsuggested steering.theta_d: {quantile(args.q):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
