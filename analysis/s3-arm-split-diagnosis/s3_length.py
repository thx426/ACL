#!/usr/bin/env python3
"""长度膨胀的本质:逐轨迹、逐轮拆解。

查五件事:
  1. thought 逐字重复率(历史最强判别指标:赢轨迹恒 0 / 输轨迹 26-70%)
  2. 逐轮 token 增长形态(是均匀变长,还是后段发散)
  3. thought 长度 vs action 长度分别的增长
  4. 有效动作率、动作多样性
  5. bare 与 steered 分开看(排除是不是只有被干预的臂在膨胀)
"""
import glob
import json
import os
import statistics as st
import sys
from collections import Counter


def arm(r):
    s = r.get("steering") or {}
    a = s.get("steer_arm")
    return int(a) if a is not None else None


def rep_rate(items):
    """逐字重复率 = 1 - 去重数/总数。"""
    xs = [x.strip() for x in items if (x or "").strip()]
    return (1.0 - len(set(xs)) / len(xs)) if len(xs) >= 2 else 0.0


def analyse(rows, tag):
    if not rows:
        return
    won = [r for r in rows if r.get("won")]
    lost = [r for r in rows if not r.get("won")]
    print(f"  --- {tag}: {len(rows)} 条 (赢 {len(won)} / 输 {len(lost)}) ---")
    for name, grp in (("赢", won), ("输", lost)):
        if not grp:
            continue
        rt, ra, tl, al, va, nt, seg = [], [], [], [], [], [], []
        for r in grp:
            steps = ((r.get("trajectory") or {}).get("steps")) or []
            if not steps:
                continue
            th = [s.get("thought") or "" for s in steps]
            ac = [s.get("action_norm") or "" for s in steps]
            rt.append(rep_rate(th))
            ra.append(rep_rate(ac))
            tl.append(st.mean([len(x) for x in th]) if th else 0)
            al.append(st.mean([len(x) for x in ac]) if ac else 0)
            va.append(sum(1 for s in steps if s.get("valid")) / len(steps))
            nt.append(len(steps))
            # 逐轮长度:前 1/3 vs 后 1/3(看是否后段发散)
            k = max(1, len(th) // 3)
            seg.append((st.mean([len(x) for x in th[:k]]),
                        st.mean([len(x) for x in th[-k:]])))
        if not rt:
            continue
        head = st.mean([a for a, _ in seg])
        tail = st.mean([b for _, b in seg])
        print(f"    {name}: n={len(rt):>3}  thought重复 {100*st.mean(rt):>5.1f}%  "
              f"action重复 {100*st.mean(ra):>5.1f}%  有效动作 {100*st.mean(va):>5.1f}%  "
              f"轮数 {st.mean(nt):>4.1f}")
        print(f"        thought均长 {st.mean(tl):>6.0f} 字符  action均长 {st.mean(al):>5.1f}  "
              f"前1/3轮 {head:>6.0f} → 后1/3轮 {tail:>6.0f} ({tail/max(head,1):.2f}×)")


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
            print(f"step {step}: 无数据")
            continue
        print(f"\n===== step {step} ({len(rows)} 行) =====")
        analyse([r for r in rows if arm(r) == 0], "bare")
        analyse([r for r in rows if arm(r) == 1], "steered")
        # 动作动词分布(看是否退化成少数几个无效动作)
        verbs = Counter()
        for r in rows:
            if r.get("won"):
                continue
            for s in ((r.get("trajectory") or {}).get("steps")) or []:
                a = (s.get("action_norm") or "").strip()
                if a:
                    verbs[a.split()[0]] += 1
        tot = sum(verbs.values()) or 1
        top = ", ".join(f"{k} {100*v/tot:.0f}%" for k, v in verbs.most_common(6))
        print(f"  失败轨迹动作动词分布: {top}")


if __name__ == "__main__":
    main()
