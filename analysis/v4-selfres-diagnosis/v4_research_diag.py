#!/usr/bin/env python3
"""v4 research 行的因果链诊断:长度 → overlong → no_finalize → r_raw=0。

要回答四个问题:
  1. valid_continuations 是不是双峰(要么 0 要么满分)?
  2. overlong / no_finalize / error 三者的交叉表是什么?
  3. r_raw>0 的行与 r_raw=0 的行,在长度/轮数上差多少?
  4. 若把 research_max_sample_tokens 从 24576 提到各档,能救回多少行?
"""
import collections
import glob
import json
import statistics as st
import sys

RD = sys.argv[1]
rows = [json.load(open(f)) for f in sorted(glob.glob(RD + "/*.json"))]
res = [r for r in rows if r.get("sample_role") == "research"]


def rf(r, k):
    return (r.get("research") or {}).get(k)


print(f"research 行 {len(res)} 条\n")

print("=== 1. valid_continuations 分布 ===")
vc = collections.Counter(rf(r, "research_valid_continuations") for r in res)
for k in sorted(vc, key=lambda x: (x is None, x)):
    print(f"  {k}: {vc[k]} 行 ({100*vc[k]/len(res):.0f}%)")

print("\n=== 2. overlong × no_finalize × error 交叉表 ===")
cross = collections.Counter(
    (bool(rf(r, "research_overlong")), bool(rf(r, "research_no_finalize")),
     bool(rf(r, "research_error"))) for r in res)
print("  (overlong, no_finalize, error) -> 行数")
for k, v in sorted(cross.items(), key=lambda kv: -kv[1]):
    print(f"  {k} -> {v:>3}  ({100*v/len(res):.0f}%)")

print("\n=== 3. r_raw>0 vs r_raw=0 的对比 ===")
pos = [r for r in res if (rf(r, "research_r_raw") or 0) > 0]
zero = [r for r in res if not (rf(r, "research_r_raw") or 0) > 0]
for label, grp in (("r_raw>0", pos), ("r_raw=0", zero)):
    if not grp:
        continue
    L = [r.get("response_token_len") or 0 for r in grp]
    T = [rf(r, "research_turns") or 0 for r in grp]
    E = [rf(r, "research_experiments") or 0 for r in grp]
    W = [rf(r, "research_doc_words") or 0 for r in grp]
    print(f"  {label:<9} n={len(grp):>3}  respLen中位 {st.median(L):>7.0f}  "
          f"turns中位 {st.median(T):>4.1f}  exps中位 {st.median(E):>4.1f}  "
          f"doc_words中位 {st.median(W):>5.0f}  overlong {100*sum(1 for r in grp if rf(r,'research_overlong'))/len(grp):>3.0f}%")

print("\n=== 4. 提高 research_max_sample_tokens 的救回量(按 respLen 反推) ===")
L = sorted(r.get("response_token_len") or 0 for r in res)
for cap in (24576, 32768, 40960, 49152, 65536):
    keep = [x for x in L if x <= cap]
    # overlong 行是 **mask 全 0**(零梯度),不是截断 —— 训练 token 只来自未超限的行
    tok = sum(keep)
    print(f"  cap={cap:>6}: 未超限 {len(keep):>3}/{len(L)} ({100*len(keep)/len(L):>3.0f}%)  "
          f"该 step research 有效训练 token ≈ {tok/1e6:.2f}M  "
          f"(相对 cap=24576 的 {tok/max(sum(x for x in L if x<=24576),1):.1f}×)")
print(f"  respLen 分位: p10={L[len(L)//10]:.0f} p50={L[len(L)//2]:.0f} "
      f"p90={L[9*len(L)//10]:.0f} max={L[-1]:.0f}")

print("\n=== 5. 若只看未 overlong 的行,飞轮读数是什么 ===")
good = [r for r in res if not rf(r, "research_overlong")]
if good:
    rr = [rf(r, "research_r_raw") or 0 for r in good]
    vcg = [rf(r, "research_valid_continuations") or 0 for r in good]
    nf = sum(1 for r in good if rf(r, "research_no_finalize"))
    print(f"  未 overlong 的 {len(good)} 行: r_raw 非零 "
          f"{sum(1 for x in rr if x>0)}/{len(rr)} ({100*sum(1 for x in rr if x>0)/len(rr):.0f}%)  "
          f"中位 {st.median(rr):.3f}  valid_conts 中位 {st.median(vcg):.0f}  "
          f"no_finalize {nf}/{len(good)} ({100*nf/len(good):.0f}%)")
