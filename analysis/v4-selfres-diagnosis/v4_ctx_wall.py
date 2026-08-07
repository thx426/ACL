#!/usr/bin/env python3
"""验证 research 会话是不是撞在 40960 的 context 墙上。

判据:把每条 research 行的 messages 累计 token 量逐轮还原,看它在第几轮
越过 40960;若 error/no_finalize 的行恰好都越过而成功的行都没越过,
则 context 墙就是病因,而不是"模型不会做研究"。
"""
import glob
import json
import statistics as st
import sys

RD = sys.argv[1]
CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 40960

rows = [json.load(open(f)) for f in sorted(glob.glob(RD + "/*.json"))]
res = [r for r in rows if r.get("sample_role") == "research"]


def rf(r, k):
    return (r.get("research") or {}).get(k)


# 粗略 token 估计:英文约 4 字符/token(只用于相对比较,不做精确断言)
def approx_tok(msgs):
    return sum(len(m.get("content") or "") for m in msgs) / 4.0


print(f"research 行 {len(res)} 条,context 上限按 {CTX} 判\n")

buckets = {}
for r in res:
    msgs = r.get("messages") or []
    est = approx_tok(msgs)
    real = r.get("response_token_len") or 0
    key = (bool(rf(r, "research_error")), bool(rf(r, "research_no_finalize")))
    buckets.setdefault(key, []).append((est, real, len(msgs), rf(r, "research_turns")))

print(f"{'(error,no_finalize)':<22} {'n':>4} {'msgs估tok中位':>14} {'respLen中位':>12} "
      f"{'msg数中位':>10} {'turns中位':>10}")
for k in sorted(buckets):
    v = buckets[k]
    print(f"{str(k):<22} {len(v):>4} {st.median([x[0] for x in v]):>14.0f} "
          f"{st.median([x[1] for x in v]):>12.0f} {st.median([x[2] for x in v]):>10.0f} "
          f"{st.median([x[3] or 0 for x in v]):>10.1f}")

print(f"\n=== respLen 相对 context 上限 {CTX} ===")
over = [r for r in res if (r.get("response_token_len") or 0) > CTX]
under = [r for r in res if (r.get("response_token_len") or 0) <= CTX]
for label, grp in (("respLen > ctx", over), ("respLen <= ctx", under)):
    if not grp:
        continue
    e = sum(1 for r in grp if rf(r, "research_error"))
    nf = sum(1 for r in grp if rf(r, "research_no_finalize"))
    rz = sum(1 for r in grp if (rf(r, "research_r_raw") or 0) > 0)
    print(f"  {label:<16} n={len(grp):>3}  error {100*e/len(grp):>3.0f}%  "
          f"no_finalize {100*nf/len(grp):>3.0f}%  r_raw>0 {100*rz/len(grp):>3.0f}%")

# 更强的判据:接近但未超 ctx 的行 vs 远低于 ctx 的行
print(f"\n=== 按 respLen 分档看 finalize 成功率(能否看出 {CTX} 处的断崖) ===")
edges = [0, 8192, 16384, 24576, 32768, 36864, CTX, 49152, 10**9]
for lo, hi in zip(edges, edges[1:]):
    grp = [r for r in res if lo < (r.get("response_token_len") or 0) <= hi]
    if not grp:
        continue
    nf = sum(1 for r in grp if rf(r, "research_no_finalize"))
    rz = sum(1 for r in grp if (rf(r, "research_r_raw") or 0) > 0)
    print(f"  ({lo:>6},{min(hi,99999):>6}]  n={len(grp):>3}  "
          f"finalize成功 {100*(1-nf/len(grp)):>5.1f}%   r_raw>0 {100*rz/len(grp):>5.1f}%")
