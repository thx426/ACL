#!/usr/bin/env python3
"""9B step-1 验收:一次跑完四组读数,避免逐项来回。

  ① researcher 真的产文档了吗(不是"调用成功",是产出非空且被注入)
  ② 分臂结构对不对(8 bare + 8 steered,won_bare_rate 只按 bare 算)
  ③ 熵门在 4B 占位值下是否可用(判据=门死区率,不是 gate_mean)
  ④ SP=4 是否真生效(靠 steer 覆盖与长度分布交叉印证)

用法: python3 step1_accept.py <run_dir> [step]
"""
import glob, json, os, sys
from collections import Counter

ENT_BINS, ENT_MAX = 1024, 8.0
BW = ENT_MAX / ENT_BINS
CUR_THETA, CUR_FLOOR = 1.28, 0.6


def q(hist, p):
    tot = sum(hist)
    if not tot:
        return None
    t, a = p * tot, 0
    for i, c in enumerate(hist):
        a += c
        if a >= t:
            return (i + 0.5) * BW
    return ENT_MAX


def frac_below(hist, thr):
    tot = sum(hist)
    return sum(hist[:min(int(thr / BW), ENT_BINS - 1)]) / tot if tot else None


def main():
    run = sys.argv[1]
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    d = os.path.join(run, "rollouts", f"step_{step:05d}")
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    print(f"run={run}  step={step}  文件 {len(files)} 个")
    if not files:
        print("XX 无产物"); return 1

    rows = []
    for fp in files:
        try:
            rows.append(json.load(open(fp)))
        except Exception:
            pass
    print(f"可解析 {len(rows)} 条\n")

    # ---------- ① researcher ----------
    print("═" * 62)
    print("① researcher 文档产出")
    docs, doc_lens, roles = [], [], Counter()
    for r in rows:
        roles[r.get("role", "?")] += 1
        doc = (r.get("doc") or r.get("guidance") or
               (r.get("steering") or {}).get("doc") or "")
        if isinstance(doc, str) and doc.strip():
            docs.append(doc)
            doc_lens.append(len(doc.split()))
    print(f"  行角色分布: {dict(roles)}")
    print(f"  带非空文档的行: {len(docs)}/{len(rows)}")
    if doc_lens:
        doc_lens.sort()
        print(f"  文档词数 中位 {doc_lens[len(doc_lens)//2]}  "
              f"min {doc_lens[0]}  max {doc_lens[-1]}")
        uniq = len(set(docs))
        print(f"  去重后 {uniq} 份不同文档 "
              f"({'✓ 每任务一份' if uniq > 1 else '!! 只有一份,检查是否共享/占位'})")
    else:
        print("  !! 没有任何非空文档 —— researcher 链路要查")

    # ---------- ② 分臂 ----------
    print("═" * 62)
    print("② 分臂结构与能力读数")
    arm = Counter()
    won_by_arm = {0: [], 1: []}
    for r in rows:
        st = r.get("steering") or {}
        a = st.get("steer_arm")
        if a is None:
            continue
        arm[a] += 1
        w = r.get("won")
        if w is None:
            w = (r.get("reward_extra_info") or {}).get("won")
        if w is not None:
            won_by_arm[int(a)].append(int(bool(w)))
    print(f"  steer_arm 分布: bare={arm.get(0,0)}  steered={arm.get(1,0)}"
          f"   (期望各 8×16=128)")
    for a, name in ((0, "bare  "), (1, "steer ")):
        v = won_by_arm[a]
        if v:
            print(f"  won_{name}rate = {sum(v)/len(v)*100:5.1f}%  (n={len(v)})")
    if won_by_arm[0] and won_by_arm[1]:
        g = sum(won_by_arm[1])/len(won_by_arm[1]) - sum(won_by_arm[0])/len(won_by_arm[0])
        print(f"  δ gap = {g*100:+.1f}pp   ← 教师增益,不是能力读数")
    print("  ⚠ 能力读数只认 won_bare_rate;δ gap 大不等于 policy 变强")

    # ---------- ③ 熵门 ----------
    print("═" * 62)
    print("③ 熵门标定(判据=死区率,不是 gate_mean)")
    hist = [0] * ENT_BINS
    n_h = 0
    for r in rows:
        h = (r.get("steering") or {}).get("steer_ent_hist")
        if h and len(h) == ENT_BINS:
            n_h += 1
            for i, c in enumerate(h):
                hist[i] += c
    tot = sum(hist)
    print(f"  有直方图的行 {n_h};累计观测位 {tot:,}")
    if tot:
        for p in (0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
            print(f"    P{int(p*100):<3} = {q(hist,p):.4f}")
        dt, df = frac_below(hist, CUR_THETA), frac_below(hist, CUR_FLOOR)
        print(f"\n  当前 theta={CUR_THETA}: 熵低于它的位置 {dt*100:.1f}%")
        print(f"  当前 floor={CUR_FLOOR}: 熵低于它的位置 {df*100:.1f}%  ← floor 也救不回")
        p95 = q(hist, 0.95)
        print(f"\n  参照: 4B P95=1.276(theta 来源) / 1.7B P95=0.043(此处熵门失效)")
        print(f"        9B P95={p95:.4f}  比 4B {p95/1.276:.2f}×")
        if df > 0.90:
            print("  判定: XX 结构性失效(同 1.7B 形态)——必须重标 theta")
        elif dt > 0.90:
            print("  判定: !! theta 偏高,九成位置门关闭")
        elif dt < 0.30:
            print("  判定: !! theta 偏低,门几乎常开")
        else:
            print("  判定: OK 4B 占位值可用")
        print(f"  建议 entropy_theta_init = {p95:.2f} (9B P95)")
    else:
        print("  !! 无熵直方图 —— steering 可能没真正执行")

    # ---------- ④ 干预覆盖 ----------
    print("═" * 62)
    print("④ 干预覆盖与长度")
    cov, dcount, lens = [], [], []
    for r in rows:
        st = r.get("steering") or {}
        if st.get("steer_arm") != 1:
            continue
        n = st.get("steer_n_tilted") or st.get("steer_n_accepted") or 0
        tot_pos = st.get("steer_ent_count") or 0
        if tot_pos:
            cov.append(n / tot_pos)
        if st.get("steer_d_count") is not None:
            dcount.append(st["steer_d_count"])
        L = r.get("response_length") or len(r.get("response_ids") or [])
        if L:
            lens.append(L)
    if cov:
        cov.sort()
        print(f"  steered 行干预覆盖率 中位 {cov[len(cov)//2]*100:.2f}%  "
              f"max {cov[-1]*100:.2f}%")
        print("  (1.7B 上该值只有 1.45% 时,机制实际没跑)")
    if lens:
        lens.sort()
        print(f"  response_length 中位 {lens[len(lens)//2]}  P90 {lens[int(len(lens)*0.9)]}  max {lens[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
