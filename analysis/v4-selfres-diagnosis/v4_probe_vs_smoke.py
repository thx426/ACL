#!/usr/bin/env python3
"""辨析:同一个 SFT ckpt,为什么 M0 探针 finalize 75% 而 RL 冒烟只有 29%?

两个候选解释:
  (A) ctx 墙 —— 冒烟里的会话更长,撞 40960 被截断;
  (B) 协议没学会 —— tool_ok 下界 0.668 < 0.8,本来就不稳。
若 (A) 为主,两边的**长度分布**应显著不同,且冒烟里短会话的成功率应回到探针水平。
若 (B) 为主,两边同长度档的成功率应接近。

用法: v4_probe_vs_smoke.py <probe_step_dir> <smoke_rollout_dir>
"""
import glob
import json
import statistics as st
import sys


def probe_rows(d):
    out = []
    for f in sorted(glob.glob(d + "/*/*.json")):
        try:
            out.append(json.load(open(f)))
        except Exception:
            pass
    return out


def smoke_research(d):
    out = []
    for f in sorted(glob.glob(d + "/*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if r.get("sample_role") == "research":
            out.append(r)
    return out


def approx_tok(msgs):
    """粗略 token 估计,只用于两侧同口径的相对比较。"""
    return sum(len(m.get("content") or "") for m in (msgs or [])) / 4.0


pr = probe_rows(sys.argv[1])
sm = smoke_research(sys.argv[2])
print(f"探针 {len(pr)} 会话 / 冒烟 research {len(sm)} 会话\n")

print("=== 探针会话的字段 ===")
if pr:
    print("  顶层键:", sorted(pr[0].keys())[:14])

# 探针侧:找 finalize / turns / 长度
p_ok, p_tok, p_turn = [], [], []
for r in pr:
    msgs = r.get("messages") or r.get("transcript") or []
    p_tok.append(approx_tok(msgs))
    p_turn.append(len([m for m in msgs if m.get("role") == "assistant"]))
    fin = r.get("finalized")
    if fin is None:
        fin = bool(r.get("doc") or r.get("document") or "")
    p_ok.append(bool(fin))

s_ok, s_tok, s_turn = [], [], []
for r in sm:
    rr = r.get("research") or {}
    s_ok.append(not rr.get("research_no_finalize"))
    s_tok.append(r.get("response_token_len") or 0)
    s_turn.append(rr.get("research_turns") or 0)

print("\n=== 两侧总览 ===")
for label, ok, tok, turn in (("探针", p_ok, p_tok, p_turn), ("冒烟", s_ok, s_tok, s_turn)):
    if not ok:
        continue
    print(f"  {label}: finalize {100*sum(ok)/len(ok):>5.1f}%  "
          f"长度中位 {st.median(tok):>7.0f}  长度p90 {sorted(tok)[int(.9*len(tok))-1]:>7.0f}  "
          f"轮数中位 {st.median(turn):>4.1f}")

print("\n=== 冒烟侧按长度分档(看短会话能否回到探针水平) ===")
edges = [0, 16384, 24576, 32768, 36864, 40960, 10**9]
for lo, hi in zip(edges, edges[1:]):
    g = [(o, t) for o, t in zip(s_ok, s_tok) if lo < t <= hi]
    if not g:
        continue
    print(f"  ({lo:>6},{min(hi,99999):>6}]  n={len(g):>3}  finalize {100*sum(o for o,_ in g)/len(g):>5.1f}%")

print("\n=== 探针侧按长度分档(同口径,看有没有同样的断崖) ===")
for lo, hi in zip(edges, edges[1:]):
    g = [(o, t) for o, t in zip(p_ok, p_tok) if lo < t <= hi]
    if not g:
        continue
    print(f"  ({lo:>6},{min(hi,99999):>6}]  n={len(g):>3}  finalize {100*sum(o for o,_ in g)/len(g):>5.1f}%")
