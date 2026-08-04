#!/usr/bin/env python3
"""PGS 单轨迹检视器 —— 一条 rollout dump → HTML(steer 位置内联标注)。

    python scripts/pgs_traj.py <dump.json> [--tokenizer <hf-path>] [--out t.html]

有 tokenizer:按 turn_token_spans 重建各 turn 的 token 级文本,steer token
用 <mark> 高亮(hover 显示 draft→final、η/D/W、accept/reject、预算余额、
探针标记);无 tokenizer:每 turn 下方给 steer 行表(token 显 id)。
用途:机制正确性的人工逐例抽查(重采是否语义合理、坐标是否对齐)。
"""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


def _steer_rows(st):
    keys = ["steer_positions", "steer_log_pi", "steer_log_b", "steer_eta",
            "steer_d", "steer_base_open", "steer_probe", "steer_rejected",
            "steer_cap_active", "steer_draft_token", "steer_final_token",
            "steer_propensity"]
    arrays = {k: st.get(k) or [] for k in keys}
    n = len(arrays["steer_positions"])
    rows = []
    for i in range(n):
        r = {k.replace("steer_", ""): (arrays[k][i] if i < len(arrays[k]) else None)
             for k in keys}
        r["w"] = math.exp(r["log_pi"] - r["log_b"]) if r["log_b"] is not None else None
        rows.append(r)
    return rows


def _row_tip(r, tok=None):
    d = tok.decode([r["draft_token"]]) if tok and r["draft_token"] is not None and r["draft_token"] >= 0 else r["draft_token"]
    f = tok.decode([r["final_token"]]) if tok and r["final_token"] is not None and r["final_token"] >= 0 else r["final_token"]
    bits = [
        f"pos={r['positions']}",
        f"draft={d!r}→final={f!r}" if r["rejected"] else f"token={f!r}",
        "REJECTED" if r["rejected"] else "accepted",
        f"η={r['eta']:.3f} D={r['d']:.3f} W={r['w']:.3f}" if r["eta"] is not None else "",
        "probe-flip" if r["probe"] else "",
        "cap" if r["cap_active"] else "",
    ]
    return " | ".join(str(b) for b in bits if b)


def render(payload, tok=None) -> str:
    st = payload.get("steering") or {}
    rows = _steer_rows(st)
    by_pos = {r["positions"]: r for r in rows}
    spans = payload.get("turn_token_spans") or []
    steps = (payload.get("trajectory") or {}).get("steps") or []

    head = (
        f"<h1>{html.escape(str(payload.get('task_id')))}</h1>"
        f"<p>step={payload.get('global_step')} won=<b>{payload.get('won')}</b> "
        f"outcome={payload.get('outcome')} env_steps={payload.get('env_steps')} "
        f"truncated={payload.get('truncated')}</p>"
        f"<p>steering: 倾斜 {st.get('steer_count', 0)} 次 / 拒绝 {st.get('steer_reject_count', 0)} 次 / "
        f"预算 {st.get('steer_budget_spent', 0):.2f} / 候选 {st.get('steer_n_candidates', 0)} / "
        f"开门 {st.get('steer_n_base_open', 0)} / 探针翻转 {st.get('steer_n_probe_flipped', 0)} / "
        f"cap 轮 {st.get('steer_turns_capped', 0)} / doc_missing={st.get('doc_missing')}</p>"
    )

    blocks = []
    # response token 流重建(有 tokenizer 时内联高亮)
    resp_hash_note = ""
    for t_idx, step in enumerate(steps):
        span = spans[t_idx] if t_idx < len(spans) else None
        seg_rows = [r for r in rows if span and span[0] <= r["positions"] < span[1]]
        thought = html.escape(step.get("thought", ""))
        action = html.escape(step.get("action_norm") or step.get("action_raw") or "")
        obs = html.escape((step.get("observation") or "")[:600])
        marks = ""
        if seg_rows:
            trs = "".join(
                f"<tr class='{'rej' if r['rejected'] else ''}'>"
                f"<td>{r['positions']}</td><td>{_tok_txt(tok, r['draft_token'])}</td>"
                f"<td>{_tok_txt(tok, r['final_token'])}</td>"
                f"<td>{'✗拒' if r['rejected'] else '✓收'}</td>"
                f"<td>{r['eta']:.2f}</td><td>{r['d']:.2f}</td>"
                f"<td>{r['w']:.2f}</td>"
                f"<td>{'probe' if r['probe'] else ''}{' cap' if r['cap_active'] else ''}</td></tr>"
                for r in seg_rows
            )
            marks = ("<table><tr><th>pos</th><th>draft</th><th>final</th><th></th>"
                     f"<th>η</th><th>D</th><th>W</th><th>标记</th></tr>{trs}</table>")
        blocks.append(
            f"<div class='turn'><h3>turn {t_idx + 1}"
            f"{'  ·  ' + str(len(seg_rows)) + ' steer 行' if seg_rows else ''}</h3>"
            f"<div class='th'>{thought}</div>"
            f"<div class='ac'>▶ {action}</div>"
            f"<div class='ob'>{obs}</div>{marks}</div>"
        )

    style = """<style>body{font-family:system-ui;margin:24px;max-width:980px}
.turn{border:1px solid #ddd;border-radius:8px;padding:10px 14px;margin:10px 0}
.th{white-space:pre-wrap;color:#333}.ac{font-weight:600;color:#0a58ca;margin:6px 0}
.ob{color:#666;font-size:13px;white-space:pre-wrap}
table{border-collapse:collapse;font-size:12px;margin-top:8px}
td,th{border:1px solid #ccc;padding:2px 7px}tr.rej{background:#ffe9e9}</style>"""
    return f"<!doctype html><meta charset='utf-8'><title>traj</title>{style}{head}{''.join(blocks)}"


def _tok_txt(tok, tid):
    if tid is None or tid < 0:
        return "-"
    if tok is None:
        return str(tid)
    return f"<code>{html.escape(tok.decode([tid]))}</code><small>({tid})</small>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    payload = json.loads(Path(args.dump).read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else Path(args.dump).with_suffix(".html")
    out.write_text(render(payload, tok), encoding="utf-8")
    print(f"traj view: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
