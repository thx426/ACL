#!/usr/bin/env python3
"""A800 s3 (4B raw / n=16 分臂 / maxopen) 逐步深度分析。

回答两个问题:
  ① steered 明显优于 bare,为何能力不进 policy 自身(bare 一直平)?
  ② 中途为何出现性能衰退 + 输出长度显著变长?
"""
import json
import re
import sys

KEYS = [
    "actor/entropy", "actor/grad_norm", "actor/pg_loss",
    "actor/rollout_is_weight_off1_frac", "actor/rollout_is_weight_mean",
    "critic/advantages/mean", "critic/advantages/max", "critic/score/mean",
    "rollout_extra/won/mean", "rollout_extra/won_bare_rate/mean",
    "rollout_extra/won_steered_rate/mean", "rollout_extra/steer_arm_gap/mean",
    "rollout_extra/steer_d_mean/mean", "rollout_extra/steer_budget_spent/mean",
    "rollout_extra/steer_count/mean", "rollout_extra/steer_reject_count/mean",
    "rollout_extra/steer_n_candidates/mean", "rollout_extra/steer_n_pt_window_miss/mean",
    "rollout_extra/truncated/mean", "rollout_extra/invalid_format/mean",
    "rollout_extra/env_steps/mean",
    "response_length/mean", "response_length/max", "response_length/clip_ratio",
]
NUM = r"(-?[0-9.]+(?:[eE][-+]?[0-9]+)?)"


def parse(log):
    rows = []
    with open(log, errors="ignore") as fh:
        for raw in fh:
            for line in raw.replace("\r", "\n").split("\n"):
                if "actor/entropy" not in line:
                    continue
                m = re.search(r"step:(\d+)", line)
                if not m:
                    continue
                d = {"step": int(m.group(1))}
                for k in KEYS:
                    mm = re.search(re.escape(k) + ":" + NUM, line)
                    d[k] = float(mm.group(1)) if mm else None
                rows.append(d)
    return rows


def main():
    rows = parse(sys.argv[1])
    g = lambda r, k: r.get(k)
    hdr = ("step", "bare%", "steer%", "gap", "熵", "gradN", "W_mean", "off1%",
           "len均", "len最大", "clip", "D均", "倾斜", "拒绝", "花费", "窗外", "轮数")
    print("{:>4} {:>6} {:>7} {:>6} {:>7} {:>7} {:>8} {:>6} {:>7} {:>8} {:>6} {:>7} {:>7} {:>6} {:>7} {:>7} {:>6}".format(*hdr))
    for r in rows:
        def f(k, fmt="{:.3f}", scale=1.0):
            v = g(r, k)
            return fmt.format(v * scale) if v is not None else "-"
        print("{:>4} {:>6} {:>7} {:>6} {:>7} {:>7} {:>8} {:>6} {:>7} {:>8} {:>6} {:>7} {:>7} {:>6} {:>7} {:>7} {:>6}".format(
            r["step"],
            f("rollout_extra/won_bare_rate/mean", "{:.1f}", 100),
            f("rollout_extra/won_steered_rate/mean", "{:.1f}", 100),
            f("rollout_extra/steer_arm_gap/mean", "{:+.1f}", 100),
            f("actor/entropy", "{:.4f}"),
            f("actor/grad_norm", "{:.2f}"),
            f("actor/rollout_is_weight_mean", "{:.5f}"),
            f("actor/rollout_is_weight_off1_frac", "{:.1f}", 100),
            f("response_length/mean", "{:.0f}"),
            f("response_length/max", "{:.0f}"),
            f("response_length/clip_ratio", "{:.3f}"),
            f("rollout_extra/steer_d_mean/mean", "{:.4f}"),
            f("rollout_extra/steer_count/mean", "{:.0f}"),
            f("rollout_extra/steer_reject_count/mean", "{:.1f}"),
            f("rollout_extra/steer_budget_spent/mean", "{:.0f}"),
            f("rollout_extra/steer_n_pt_window_miss/mean", "{:.0f}"),
            f("rollout_extra/env_steps/mean", "{:.1f}"),
        ))


if __name__ == "__main__":
    main()
