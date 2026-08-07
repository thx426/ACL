#!/usr/bin/env python3
"""逐 rollout 分析:验证「教师纠错位置的梯度贡献被 IS 权重压制」假设。

核心检验:speculative 拒绝位(教师真正改写 token 的地方)携带教师的全部新信息,
但那里 π 小 / b 大 ⇒ W_beh = π/b 小 ⇒ 梯度贡献被压。若成立,则"教师帮得越多的
位置越不被学习",能力自然传不进 policy。

用法: s3_rollout.py <run_dir> <step> [<step> ...]
"""
import glob
import json
import math
import os
import statistics as st
import sys


def load(run, step):
    d = os.path.join(run, "rollouts", f"step_{step:05d}")
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            rows.append(json.load(open(f)))
        except Exception:
            pass
    return rows


def arm(r):
    s = r.get("steering") or {}
    a = s.get("steer_arm")
    return int(a) if a is not None else None


def main():
    run = sys.argv[1]
    for step in [int(x) for x in sys.argv[2:]]:
        rows = load(run, step)
        if not rows:
            print(f"step {step}: 无数据")
            continue
        bare = [r for r in rows if arm(r) == 0]
        steer = [r for r in rows if arm(r) == 1]
        print(f"\n===== step {step}: {len(rows)} 行 (bare {len(bare)} / steered {len(steer)}) =====")

        for name, grp in (("bare", bare), ("steered", steer)):
            if not grp:
                continue
            w = sum(1 for r in grp if r.get("won"))
            ln = [r.get("response_token_len", 0) for r in grp]
            tn = [r.get("assistant_turns", 0) for r in grp]
            wl = [r.get("response_token_len", 0) for r in grp if r.get("won")]
            ll = [r.get("response_token_len", 0) for r in grp if not r.get("won")]
            print(f"  {name:<8} won {w}/{len(grp)}={100*w/len(grp):.1f}%  "
                  f"len中位 {st.median(ln):.0f}  轮 {st.median(tn):.0f}  "
                  f"赢len中位 {st.median(wl) if wl else 0:.0f}  输len中位 {st.median(ll) if ll else 0:.0f}")

        # === 核心:steered 臂的 per-token 倾向权重 ===
        # steer_propensity = b(draft)/π(draft) 的记账(接受位);拒绝位 final≠draft。
        acc_lp, acc_lb, rej_lp, rej_lb = [], [], [], []
        rej_rank = []
        for r in steer:
            s = r.get("steering") or {}
            lp = s.get("steer_log_pi") or []
            lb = s.get("steer_log_b") or []
            rj = s.get("steer_rejected") or []
            rk = s.get("steer_final_pi_rank") or []
            n = min(len(lp), len(lb), len(rj))
            for i in range(n):
                if rj[i]:
                    rej_lp.append(lp[i]); rej_lb.append(lb[i])
                    if i < len(rk) and rk[i] is not None and rk[i] >= 0:
                        rej_rank.append(rk[i])
                else:
                    acc_lp.append(lp[i]); acc_lb.append(lb[i])

        def wstats(lp, lb, tag):
            if not lp:
                print(f"  {tag}: 无数据")
                return
            # W_beh(per-token) = π/b  ⇒ log W = log π − log b
            logw = [a - b for a, b in zip(lp, lb)]
            w = [math.exp(max(min(x, 20), -20)) for x in logw]
            w.sort()
            q = lambda p: w[min(int(p * len(w)), len(w) - 1)]
            print(f"  {tag}: n={len(w):>7}  W中位 {q(.5):.4f}  p10 {q(.10):.4f}  "
                  f"p90 {q(.90):.4f}  均值 {st.mean(w):.4f}  "
                  f"W<0.5占比 {100*sum(1 for x in w if x < 0.5)/len(w):.1f}%")

        wstats(acc_lp, acc_lb, "接受位 W")
        wstats(rej_lp, rej_lb, "拒绝位 W(教师真正改写处)")
        if rej_rank:
            rej_rank.sort()
            qq = lambda p: rej_rank[min(int(p * len(rej_rank)), len(rej_rank) - 1)]
            print(f"  拒绝位重采 token 在 π 中的 rank: 中位 {qq(.5)}  p90 {qq(.9)}  "
                  f"max {rej_rank[-1]}  (rank 越大=π 越不可能自发产生)")


if __name__ == "__main__":
    main()
