#!/usr/bin/env python3
"""抽真实轨迹文本:看"长篇复读"到底长什么样。"""
import glob
import json
import os
import sys


def arm(r):
    s = r.get("steering") or {}
    a = s.get("steer_arm")
    return int(a) if a is not None else None


def rep(items):
    xs = [x.strip() for x in items if (x or "").strip()]
    return (1.0 - len(set(xs)) / len(xs)) if len(xs) >= 2 else 0.0


def main():
    run, step = sys.argv[1], int(sys.argv[2])
    want_arm = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    d = os.path.join(run, "rollouts", f"step_{step:05d}")
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            rows.append(json.load(open(f)))
        except Exception:
            pass
    lost = [r for r in rows if not r.get("won") and arm(r) == want_arm]
    lost.sort(key=lambda r: -r.get("response_token_len", 0))
    r = lost[0]
    steps = ((r.get("trajectory") or {}).get("steps")) or []
    th = [s.get("thought") or "" for s in steps]
    print(f"# step {step} arm={want_arm} 最长失败轨迹")
    print(f"# task: {r.get('task_id')}")
    print(f"# token {r.get('response_token_len')}  轮 {len(steps)}  "
          f"thought重复 {100*rep(th):.1f}%  有效动作 "
          f"{100*sum(1 for s in steps if s.get('valid'))/max(len(steps),1):.1f}%")
    print(f"# 初始观测: {((r.get('trajectory') or {}).get('initial_obs') or '')[:200]}")
    for i in (0, 1, 2, len(steps)//2, len(steps)-2, len(steps)-1):
        if i >= len(steps) or i < 0:
            continue
        s = steps[i]
        t = s.get("thought") or ""
        print(f"\n───── 轮 {i+1}/{len(steps)} (thought {len(t)} 字符, "
              f"valid={s.get('valid')}, acted={s.get('acted')}) ─────")
        print(t[:1400])
        print(f"  ▶ action: {s.get('action_norm')!r}")
        print(f"  ◀ obs   : {(s.get('observation') or '')[:160]!r}")


if __name__ == "__main__":
    main()
