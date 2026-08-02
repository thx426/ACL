#!/usr/bin/env python3
"""Lost-region OPD-KL thought/action span decomposition.

Design §动力学(teacher-grounding-design.md)预注册机理判据:tg 治教师的良性
循环第二环——教师在 LOST 前缀上给学生的 per-token 目标是否变健康。读数 =
lost(won=0)bare 行的 OPD per-token KL 按 thought/action 段分解:
  * thought 段 per-token KL 均值 vs action 段 per-token KL 均值;
  * thought 段 KL 总量份额(并报 token 份额对照,排除"份额涨只是 token 变多");
T1 预测:thought 段份额随训练下降/持平;基线预测:随叙述通胀上升。

数据源:TRACE_OBS_DIR 观察包(verl observation.py 契约)——token_loss 是训练
时 per-token KL 真值锚(pre-outer-clamp),turn_spans 是 response 坐标下的
assistant 生成段。SP 分段按 (row_hash, seg_start) 重组。thought/action 分界
用逐 token decode 的字符定位(`</reasoning>` 闭合含 tag 之前 = thought;
tag 均为 ASCII,单 token decode 的多字节碎片误差不影响 tag 定位)。

用法:
  python3 scripts/opd_kl_span_decomp.py --obs-dir <run>/obs \
      --tokenizer /path/to/Qwen3-4B-Instruct-2507 [--json-out out.json]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

CLOSE_TAG = "</reasoning>"
ACTION_TAG = "<action>"


def load_step(step_dir: Path):
    """Merge all rank npz of one step -> {row_hash: row dict with full-length
    token_loss (NaN where no rank covered it), turn_spans, won, role}."""
    rows = {}
    for npz_path in sorted(step_dir.glob("rank*.npz")):
        z = np.load(npz_path, allow_pickle=False)
        n = len(z["row_hash"])
        for i in range(n):
            h = int(z["row_hash"][i])
            resp_len = int(z["resp_len"][i])
            row = rows.get(h)
            if row is None:
                r0, r1 = z["resp_offsets"][i], z["resp_offsets"][i + 1]
                s0, s1 = z["turn_spans_offsets"][i], z["turn_spans_offsets"][i + 1]
                spans_flat = z["turn_spans"][s0:s1]
                row = rows[h] = {
                    "token_loss": np.full(resp_len, np.nan, dtype=np.float64),
                    "response_ids": z["response_ids"][r0:r1],
                    "turn_spans": [
                        (int(spans_flat[j]), int(spans_flat[j + 1]))
                        for j in range(0, len(spans_flat), 2)
                    ],
                    "won": int(z["row_won"][i]),
                    "role": str(z["row_role"][i]),
                }
            g0, g1 = z["seg_offsets"][i], z["seg_offsets"][i + 1]
            seg = z["token_loss"][g0:g1].astype(np.float64)
            a = int(z["seg_start"][i])
            row["token_loss"][a : a + len(seg)] = seg
    return rows


def classify_span_tokens(span_ids, decode_one):
    """One assistant span -> per-token category ('thought' | 'action').
    Boundary = end of the first CLOSE_TAG (inclusive) if present, else start
    of ACTION_TAG, else the whole span counts as thought (malformed turn)."""
    pieces = [decode_one(int(t)) for t in span_ids]
    lens = [len(p) for p in pieces]
    text = "".join(pieces)
    close = text.find(CLOSE_TAG)
    if close >= 0:
        boundary = close + len(CLOSE_TAG)
    else:
        act = text.find(ACTION_TAG)
        boundary = act if act >= 0 else len(text)
    cats, c = [], 0
    for ln in lens:
        cats.append("thought" if c < boundary else "action")
        c += ln
    return cats


def decompose_row(row, decode_one):
    """-> (thought_kls, action_kls) lists of finite per-token KL values."""
    tl = row["token_loss"]
    ids = row["response_ids"]
    thought, action = [], []
    for a, b in row["turn_spans"]:
        b = min(b, len(ids))
        if b <= a:
            continue
        cats = classify_span_tokens(ids[a:b], decode_one)
        for off, cat in enumerate(cats):
            v = tl[a + off]
            if math.isfinite(v):
                (thought if cat == "thought" else action).append(float(v))
    return thought, action


def step_summary(rows, decode_one):
    """Aggregate LOST bare rows of one step."""
    th_all, ac_all, n_rows = [], [], 0
    for row in rows.values():
        if row["won"] or row["role"] != "bare":
            continue
        th, ac = decompose_row(row, decode_one)
        if th or ac:
            n_rows += 1
            th_all.extend(th)
            ac_all.extend(ac)
    if not n_rows:
        return None
    th_sum, ac_sum = sum(th_all), sum(ac_all)
    total = th_sum + ac_sum
    n_th, n_ac = len(th_all), len(ac_all)
    return {
        "lost_rows": n_rows,
        "thought_tokens": n_th,
        "action_tokens": n_ac,
        "thought_kl_per_token": th_sum / max(n_th, 1),
        "action_kl_per_token": ac_sum / max(n_ac, 1),
        "thought_kl_share": th_sum / total if total > 0 else float("nan"),
        "thought_token_share": n_th / max(n_th + n_ac, 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--tokenizer", required=True,
                    help="HF tokenizer path (decode-only use, CPU)")
    ap.add_argument("--steps", default=None,
                    help="comma list to restrict, e.g. 2,4,6")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    cache: dict[int, str] = {}

    def decode_one(tid: int) -> str:
        if tid not in cache:
            cache[tid] = tok.decode([tid], skip_special_tokens=False)
        return cache[tid]

    obs = Path(args.obs_dir)
    step_dirs = sorted(
        (d for d in obs.iterdir() if d.is_dir() and d.name.startswith("step")),
        key=lambda d: int(d.name.replace("step", "") or 0),
    )
    if not step_dirs:
        raise SystemExit(f"no step*/ dirs under {obs} — was TRACE_OBS_DIR on for this run?")

    out = {}
    hdr = (f"{'step':>5} {'rows':>5} {'th_kl/tok':>10} {'ac_kl/tok':>10} "
           f"{'th_share':>9} {'th_tok%':>8}")
    print(hdr)
    print("-" * len(hdr))
    for d in step_dirs:
        step = int(d.name.replace("step", "") or 0)
        if args.steps and str(step) not in args.steps.split(","):
            continue
        summ = step_summary(load_step(d), decode_one)
        if summ is None:
            print(f"{step:>5}  (no lost bare rows)")
            continue
        out[step] = summ
        print(f"{step:>5} {summ['lost_rows']:>5} "
              f"{summ['thought_kl_per_token']:>10.4f} "
              f"{summ['action_kl_per_token']:>10.4f} "
              f"{summ['thought_kl_share']:>9.3f} "
              f"{summ['thought_token_share']:>8.3f}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=1))
        print(f"json -> {args.json_out}")


if __name__ == "__main__":
    main()
