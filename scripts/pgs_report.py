#!/usr/bin/env python3
"""PGS 机制健康报表 —— 聚合 rollout dumps → 自包含单页 HTML。

    python scripts/pgs_report.py --run-dir <run> [--out report.html]
           [--max-steps N] [--stride K] [--tokenizer <hf-path>]

四区结构(docs/pgs-observability-design.md):
  L1 正确性(不变量核验/门率带/数值前提) L2 干预行为(位置/内容/强度)
  L3 效果信号(探针 ITT/won-lost 分层/doc 对照) L4 崩塌监控(长度/截断/重复率)
matplotlib 可用时出图(base64 内联),否则降级表格。
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trace_opd.steering.ledger import (  # noqa: E402
    D_HIST_BINS,
    ELL_HIST_BINS,
    ELL_HIST_RANGE,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:  # noqa: BLE001
    HAS_MPL = False


# ---------------------------------------------------------------- 数据装载

def load_run(run_dir: Path, max_steps: int | None, stride: int):
    """遍历 rollouts/step_*/(train)聚合;返回 per-step 与全局累积结构。"""
    rollouts = run_dir / "rollouts"
    steps = sorted(
        d for d in rollouts.glob("step_*")
        if d.is_dir() and not d.name.endswith("_val")
    )
    if stride > 1:
        steps = steps[::stride]
    if max_steps:
        steps = steps[-max_steps:]

    per_step: list[dict] = []
    glob_acc = {
        "d_hist": [0] * D_HIST_BINS, "d_hist_max": None,
        "ell_hist": [0] * ELL_HIST_BINS,
        "pair_counter": Counter(),           # (draft,final) 重采对
        "eta_values": [], "reject_positions_rel": [], "steer_positions_rel": [],
        "exhaust_positions_rel": [],
        "probe_flip_off_won": [], "probe_noflip_won": [],   # base_open 组
        "probe_flip_on_won": [], "probe_noflip_closed_won": [],
        "violations": [],
        "won_steer": [], "lost_steer": [], "won_budget": [], "lost_budget": [],
        "doc_won": [], "nodoc_won": [],
        "cap": None,   # (eta_max*2c, b_max) 从首条轨迹的 log_ratio_cap 推
    }

    for sd in steps:
        agg = defaultdict(float)
        n = 0
        rep_rates = []
        for f in sd.glob("*.json"):
            try:
                p = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            st = p.get("steering") or {}
            n += 1
            won = bool(p.get("won"))
            agg["won"] += won
            agg["truncated"] += bool(p.get("truncated"))
            agg["resp_len"] += p.get("response_token_len", 0)
            agg["env_steps"] += p.get("env_steps", 0)
            rep_rates.append(_thought_repeat_rate(p))
            if not st or "steer_n_candidates" not in st:
                continue
            agg["n_steering_traj"] += 1
            agg["candidates"] += st.get("steer_n_candidates", 0)
            agg["base_open"] += st.get("steer_n_base_open", 0)
            agg["probe_flipped"] += st.get("steer_n_probe_flipped", 0)
            agg["budget_blocked"] += st.get("steer_n_budget_blocked", 0)
            agg["bypassed"] += st.get("steer_n_bypassed", 0)
            agg["window_miss"] += st.get("steer_n_pt_window_miss", 0)
            agg["steer_count"] += st.get("steer_count", 0)
            agg["reject_count"] += st.get("steer_reject_count", 0)
            agg["budget_spent"] += st.get("steer_budget_spent", 0.0)
            agg["turns_capped"] += st.get("steer_turns_capped", 0)
            agg["doc_missing"] += st.get("doc_missing", 0)
            agg["ell_count"] += st.get("steer_ell_count", 0)
            _acc_traj(glob_acc, p, st)
        if n:
            per_step.append({
                "step": int(sd.name.split("_")[1]),
                "n": n,
                "won": agg["won"] / n,
                "truncated": agg["truncated"] / n,
                "resp_len": agg["resp_len"] / n,
                "repeat_rate": sum(rep_rates) / len(rep_rates) if rep_rates else 0.0,
                "gate_rate": (agg["base_open"] / agg["candidates"]) if agg["candidates"] else 0.0,
                "reject_rate": (agg["reject_count"] / agg["steer_count"]) if agg["steer_count"] else 0.0,
                "steer_per_traj": (agg["steer_count"] / agg["n_steering_traj"]) if agg["n_steering_traj"] else 0.0,
                "budget_per_traj": (agg["budget_spent"] / agg["n_steering_traj"]) if agg["n_steering_traj"] else 0.0,
                "window_miss_rate": (agg["window_miss"] / agg["candidates"]) if agg["candidates"] else 0.0,
                "doc_missing_rate": (agg["doc_missing"] / agg["n_steering_traj"]) if agg["n_steering_traj"] else 0.0,
                "cap_rate": (agg["turns_capped"] / agg["n_steering_traj"]) if agg["n_steering_traj"] else 0.0,
            })
    return per_step, glob_acc, steps


def _thought_repeat_rate(payload) -> float:
    """轨迹内相邻 thought 的逐字重复率(最强崩塌判别指标的既有口径)。"""
    steps = (payload.get("trajectory") or {}).get("steps") or []
    thoughts = [s.get("thought", "") for s in steps if s.get("thought")]
    if len(thoughts) < 2:
        return 0.0
    dup = sum(1 for a, b in zip(thoughts, thoughts[1:]) if a.strip() == b.strip())
    return dup / (len(thoughts) - 1)


def _acc_traj(g, payload, st):
    resp_len = max(1, int(payload.get("response_token_len", 1)))
    won = bool(payload.get("won"))
    # 直方图
    if st.get("steer_d_hist"):
        if g["d_hist_max"] is None:
            g["d_hist_max"] = st.get("steer_d_hist_max", 6.25)
        for i, v in enumerate(st["steer_d_hist"][:D_HIST_BINS]):
            g["d_hist"][i] += int(v)
    if st.get("steer_ell_hist"):
        for i, v in enumerate(st["steer_ell_hist"][:ELL_HIST_BINS]):
            g["ell_hist"][i] += int(v)
    # doc 对照
    (g["nodoc_won"] if st.get("doc_missing") else g["doc_won"]).append(won)
    # won-lost 分层
    (g["won_steer"] if won else g["lost_steer"]).append(st.get("steer_count", 0))
    (g["won_budget"] if won else g["lost_budget"]).append(st.get("steer_budget_spent", 0.0))

    pos = st.get("steer_positions") or []
    if not pos:
        # 探针对照(无翻转、门也没参与)不计入 probe 组
        return
    etas = st.get("steer_eta") or []
    rejected = st.get("steer_rejected") or []
    log_pi = st.get("steer_log_pi") or []
    log_b = st.get("steer_log_b") or []
    dts = st.get("steer_draft_token") or []
    fts = st.get("steer_final_token") or []
    probes = st.get("steer_probe") or []
    opens = st.get("steer_base_open") or []
    budget_after = None
    cap = st.get("steer_log_ratio_cap")
    if cap is not None and g["cap"] is None:
        g["cap"] = float(cap)

    # 不变量核验
    for i, p_ in enumerate(pos):
        lr = abs(float(log_pi[i]) - float(log_b[i]))
        if cap is not None and lr > float(cap) + 1e-6:
            g["violations"].append(
                f"{payload.get('task_id')}@{p_}: |logπ−logb|={lr:.3f} > cap {cap}")
        if etas[i] > 0:
            g["eta_values"].append(float(etas[i]))
            g["steer_positions_rel"].append(p_ / resp_len)
        if rejected[i]:
            g["reject_positions_rel"].append(p_ / resp_len)
            if i < len(dts) and dts[i] >= 0:
                g["pair_counter"][(int(dts[i]), int(fts[i]))] += 1
        if i < len(dts) and rejected[i] and dts[i] == fts[i]:
            g["violations"].append(f"{payload.get('task_id')}@{p_}: rejected but draft==final")
        if (not rejected[i]) and i < len(dts) and dts[i] >= 0 and dts[i] != fts[i]:
            g["violations"].append(f"{payload.get('task_id')}@{p_}: accepted but draft!=final")

    # 探针 ITT 素材(轨迹级:是否含翻转位)
    has_flip_off = any(probes[i] and opens[i] for i in range(len(pos)))
    has_flip_on = any(probes[i] and not opens[i] for i in range(len(pos)))
    if has_flip_off:
        g["probe_flip_off_won"].append(won)
    elif any(opens):
        g["probe_noflip_won"].append(won)
    if has_flip_on:
        g["probe_flip_on_won"].append(won)


# ---------------------------------------------------------------- 渲染

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img(fig) -> str:
    return f'<img src="data:image/png;base64,{_fig_to_b64(fig)}" style="max-width:100%">'


def _line(per_step, key, title, band=None):
    if not HAS_MPL or not per_step:
        rows = "".join(f"<tr><td>{r['step']}</td><td>{r[key]:.4f}</td></tr>" for r in per_step)
        return f"<h4>{title}</h4><table><tr><th>step</th><th>{key}</th></tr>{rows}</table>"
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    xs = [r["step"] for r in per_step]
    ax.plot(xs, [r[key] for r in per_step], marker=".", lw=1)
    if band:
        ax.axhspan(band[0], band[1], alpha=0.12, color="green")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    return _img(fig)


def _hist(values_or_hist, title, bins=None, edges=None):
    if not HAS_MPL:
        return f"<h4>{title}</h4><p>(matplotlib 不可用,略)</p>"
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    if edges is not None:
        ax.bar(edges, values_or_hist, width=(edges[1] - edges[0]) * 0.95)
    else:
        ax.hist(values_or_hist, bins=bins or 40)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    return _img(fig)


def _rate_ci(xs):
    if not xs:
        return "n=0"
    p = sum(xs) / len(xs)
    se = math.sqrt(max(p * (1 - p), 1e-9) / len(xs))
    return f"{p:.1%} ± {1.96 * se:.1%} (n={len(xs)})"


def render(run_dir: Path, per_step, g, tokenizer=None) -> str:
    latest = per_step[-1] if per_step else {}
    gate_ok = per_step and all(0.05 <= r["gate_rate"] <= 0.25 for r in per_step[-5:])
    lights = [
        ("不变量违例", len(g["violations"]), len(g["violations"]) == 0),
        ("门率在带内(近5步)", f"{latest.get('gate_rate', 0):.1%}", bool(gate_ok)),
        ("窗外代入率(最新)", f"{latest.get('window_miss_rate', 0):.2%}",
         latest.get("window_miss_rate", 0) < 0.05),
        ("覆盖 step", f"{per_step[0]['step']}–{per_step[-1]['step']}" if per_step else "-", True),
    ]
    head = "".join(
        f'<span class="light {"ok" if ok else "bad"}">{k}: {v}</span>'
        for k, v, ok in lights
    )

    # L2 重采对表
    pair_rows = ""
    for (d, f), cnt in g["pair_counter"].most_common(25):
        dt = html.escape(tokenizer.decode([d])) if tokenizer else str(d)
        ft = html.escape(tokenizer.decode([f])) if tokenizer else str(f)
        pair_rows += f"<tr><td><code>{dt}</code></td><td><code>{ft}</code></td><td>{cnt}</td></tr>"

    ell_edges = [(i + 0.5 - ELL_HIST_BINS / 2) / (ELL_HIST_BINS / 2) * ELL_HIST_RANGE for i in range(ELL_HIST_BINS)]
    d_max = g["d_hist_max"] or 6.25
    d_edges = [(i + 0.5) / D_HIST_BINS * d_max for i in range(D_HIST_BINS)]

    viol_html = "".join(f"<li>{html.escape(v)}</li>" for v in g["violations"][:50])
    won_ws = sum(g["won_steer"]) / len(g["won_steer"]) if g["won_steer"] else 0
    lost_ws = sum(g["lost_steer"]) / len(g["lost_steer"]) if g["lost_steer"] else 0

    parts = [
        "<!doctype html><meta charset='utf-8'><title>PGS report</title>",
        "<style>body{font-family:system-ui;margin:24px;max-width:1100px}",
        ".light{padding:4px 10px;border-radius:6px;margin-right:8px;font-size:13px}",
        ".ok{background:#d4efd4}.bad{background:#f6cfcf}",
        "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:3px 8px}",
        "h2{border-bottom:2px solid #444;margin-top:36px}",
        ".grid{display:flex;flex-wrap:wrap;gap:14px}</style>",
        f"<h1>PGS 机制健康报表</h1><p>run: <code>{run_dir}</code></p>",
        f"<p>{head}</p>",
        "<h2>L1 正确性</h2><div class='grid'>",
        _line(per_step, "gate_rate", "gate open rate (band=[5%,25%])", band=(0.05, 0.25)),
        _line(per_step, "window_miss_rate", "p_T window-miss rate"),
        "</div>",
        f"<p>不变量违例 {len(g['violations'])} 条:</p><ul>{viol_html or '<li>无</li>'}</ul>",
        "<h2>L2 干预行为</h2><div class='grid'>",
        _hist(g["steer_positions_rel"], "steer positions (relative)", bins=30),
        _hist(g["reject_positions_rel"], "reject positions (relative)", bins=30),
        _hist(g["eta_values"], "eta distribution (full=0.5)", bins=25),
        _hist(g["d_hist"], f"D distribution (0-{d_max:.2f})", edges=d_edges),
        _hist(g["ell_hist"], "raw ell distribution (+lift / -suppress)", edges=ell_edges),
        _line(per_step, "reject_rate", "reject rate (~TV(b,pi))"),
        _line(per_step, "steer_per_traj", "steered positions per traj"),
        _line(per_step, "budget_per_traj", "budget spent per traj"),
        "</div>",
        f"<h3>top 重采对(draft → final)</h3><table><tr><th>policy 想说</th><th>教师改成</th><th>次数</th></tr>{pair_rows or '<tr><td colspan=3>无拒绝</td></tr>'}</table>",
        "<h2>L3 效果信号</h2>",
        f"<p><b>探针 ITT(轨迹级粗读数,不作判决)</b>:base_open 被翻关的轨迹 won={_rate_ci(g['probe_flip_off_won'])};"
        f"未翻转对照 won={_rate_ci(g['probe_noflip_won'])};翻开(门关被开)won={_rate_ci(g['probe_flip_on_won'])}</p>",
        f"<p><b>won-lost 分层</b>:won 轨迹平均倾斜 {won_ws:.2f} 次 / lost {lost_ws:.2f} 次;"
        f"doc 对照:有 doc won={_rate_ci(g['doc_won'])} vs 缺 doc won={_rate_ci(g['nodoc_won'])}</p>",
        "<div class='grid'>",
        _line(per_step, "won", "train rollout won rate (collapse watch only)"),
        _line(per_step, "doc_missing_rate", "doc-missing fraction"),
        "</div>",
        "<h2>L4 崩塌监控</h2><div class='grid'>",
        _line(per_step, "resp_len", "mean response length (tokens)"),
        _line(per_step, "truncated", "truncation rate"),
        _line(per_step, "repeat_rate", "thought verbatim repeat rate (alert >0.55)"),
        "</div>",
        "<h2>单轨迹检视</h2><p>抽查命令:<code>python scripts/pgs_traj.py "
        "&lt;run&gt;/rollouts/step_XXXXX/&lt;task&gt;.json --tokenizer &lt;model-path&gt;</code></p>",
    ]
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--tokenizer", default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    per_step, g, steps = load_run(run_dir, args.max_steps, args.stride)
    if not per_step:
        print(f"no train rollout dumps under {run_dir}/rollouts")
        return 1
    out = Path(args.out) if args.out else run_dir / "pgs_report.html"
    out.write_text(render(run_dir, per_step, g, tok), encoding="utf-8")
    print(f"report: {out}  (steps={len(per_step)}, violations={len(g['violations'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
