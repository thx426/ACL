#!/usr/bin/env python3
"""长度膨胀的精确分解:死锁率 × 死锁轮长度 × 正常轮长度。

死锁轮定义:该轮的 (thought, action) 与前一轮逐字相同 —— 环境状态没变、
模型输出没变,纯粹在原地打转。
"""
import glob
import json
import os
import statistics as st
import sys


def arm(r):
    s = r.get("steering") or {}
    a = s.get("steer_arm")
    return int(a) if a is not None else None


def decompose(rows, tag):
    if not rows:
        return
    for label, grp in (("赢", [r for r in rows if r.get("won")]),
                       ("输", [r for r in rows if not r.get("won")])):
        if not grp:
            continue
        dead_frac, dead_len, live_len, tot_len, dead_share = [], [], [], [], []
        for r in grp:
            steps = ((r.get("trajectory") or {}).get("steps")) or []
            spans = r.get("turn_token_spans") or []
            n = min(len(steps), len(spans))
            if n < 2:
                continue
            sig = [((s.get("thought") or "").strip(), (s.get("action_norm") or "").strip())
                   for s in steps[:n]]
            tok = [b - a for a, b in spans[:n]]
            dead = [i for i in range(1, n) if sig[i] == sig[i - 1]]
            live = [i for i in range(n) if i not in dead]
            dead_frac.append(len(dead) / n)
            if dead:
                dead_len.append(st.mean([tok[i] for i in dead]))
                dead_share.append(sum(tok[i] for i in dead) / max(sum(tok), 1))
            if live:
                live_len.append(st.mean([tok[i] for i in live]))
            tot_len.append(sum(tok))
        if not tot_len:
            continue
        print(f"    {tag}/{label}: n={len(tot_len):>3}  总len中位 {st.median(tot_len):>6.0f}  "
              f"死锁轮占比 {100*st.mean(dead_frac):>5.1f}%  "
              f"死锁轮len {st.mean(dead_len) if dead_len else 0:>6.0f}  "
              f"正常轮len {st.mean(live_len) if live_len else 0:>6.0f}  "
              f"死锁贡献token {100*st.mean(dead_share) if dead_share else 0:>5.1f}%")


def main():
    run = sys.argv[1]
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
        print(f"\n===== step {step} =====")
        decompose([r for r in rows if arm(r) == 0], "bare")
        decompose([r for r in rows if arm(r) == 1], "steer")


if __name__ == "__main__":
    main()
