#!/usr/bin/env python3
"""口径 A(混合组)下的分臂 advantage 结构。

GRPO-LOO: A_i = r_i − mean(r_{-i})。混合组里 steered 赢得多 ⇒ 基线被抬高 ⇒
bare 行系统性拿负 advantage。若成立:policy 在自己的 on-policy 样本上持续
挨负梯度,而正梯度主要落在 off-policy 的 steered 行(且其教师纠错位被 IS 压)。
"""
import glob
import json
import os
import statistics as st
import sys


def main():
    run = sys.argv[1]
    print(f"{'step':>5} {'bareW%':>7} {'steerW%':>8} {'A_bare':>8} {'A_steer':>8} "
          f"{'A_bare赢':>9} {'A_bare输':>9} {'退化组':>7} {'组内全输':>9} {'组内全赢':>9}")
    for step in [int(x) for x in sys.argv[2:]]:
        d = os.path.join(run, "rollouts", f"step_{step:05d}")
        rows = []
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                rows.append(json.load(open(f)))
            except Exception:
                pass
        if not rows:
            continue
        bytask = {}
        for r in rows:
            s = r.get("steering") or {}
            a = s.get("steer_arm")
            if a is None:
                continue
            bytask.setdefault(r.get("task_id"), []).append((int(a), bool(r.get("won"))))

        ab, as_, abw, abl = [], [], [], []
        degen = alllose = allwin = 0
        for t, g in bytask.items():
            n = len(g)
            wins = sum(1 for _, w in g if w)
            if wins == 0:
                alllose += 1
            if wins == n:
                allwin += 1
            if wins in (0, n):
                degen += 1
            for arm, won in g:
                # LOO 基线:去掉自己
                base = (wins - int(won)) / (n - 1) if n > 1 else 0.0
                adv = int(won) - base
                (ab if arm == 0 else as_).append(adv)
                if arm == 0:
                    (abw if won else abl).append(adv)
        bw = sum(1 for r in rows if (r.get("steering") or {}).get("steer_arm") == 0 and r.get("won"))
        bn = sum(1 for r in rows if (r.get("steering") or {}).get("steer_arm") == 0)
        sw = sum(1 for r in rows if (r.get("steering") or {}).get("steer_arm") == 1 and r.get("won"))
        sn = sum(1 for r in rows if (r.get("steering") or {}).get("steer_arm") == 1)
        f = lambda xs: f"{st.mean(xs):+.4f}" if xs else "-"
        print(f"{step:>5} {100*bw/max(bn,1):>7.1f} {100*sw/max(sn,1):>8.1f} "
              f"{f(ab):>8} {f(as_):>8} {f(abw):>9} {f(abl):>9} "
              f"{degen:>7} {alllose:>9} {allwin:>9}")


if __name__ == "__main__":
    main()
