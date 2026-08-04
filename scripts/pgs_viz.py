#!/usr/bin/env python3
"""PGS 逐 token 可视化系统 —— step → 轨迹 → token 三级导航(离线消费 dump)。

    python scripts/pgs_viz.py --run-dir <run> [--tokenizer <hf-path>]
           [--out-dir <run>/viz] [--steps 1,2,5] [--max-trajs N]

产出(全静态,零服务,可 scp 回本机直接打开):
  viz/index.html            run 级:step 导航表 + 机制曲线(门率/θ_D/拒绝率/预算)
  viz/step_XXXXX.html       step 级:轨迹列表(won/steer/拒绝/预算,可排序)
  viz/traj/<name>.html      轨迹级:逐 token 三态着色 + 点击下钻
                            (D vs θ_D 标尺、ℓ̃、π/p_T top-16 对比、倾斜后
                             分布重算、预算余额、draft→final)

着色语义(新 schema;旧 dump 自动降级为 turn 表格模式):
  灰       结构段(glue/obs,mask=0)或非候选位置
  蓝底     候选-未开门(hover 显 D/θ)
  绿底     开门-接受(采样分布被倾斜,token 未变)
  红底     开门-拒绝重采(内联 draft→final)
  虚线框   探针翻转位;⛔ 前缀 = cap/预算旁路段起点(段级,无逐位置记录)
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:  # noqa: BLE001
    HAS_MPL = False

CSS = """
body{font-family:system-ui;margin:22px;max-width:1180px;line-height:1.45}
table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:3px 9px}
th{background:#f3f3f3;cursor:pointer}
a{text-decoration:none;color:#0a58ca}
.tok{white-space:pre-wrap;word-break:break-all;font-family:ui-monospace,monospace;font-size:13px}
.tk{border-radius:2px}
.cand{background:#e3edf9}
.acc{background:#cdeccd;font-weight:600}
.rej{background:#f6c9c9;font-weight:600}
.probe{outline:1.5px dashed #b26b00}
.struct{color:#999}
.tk:hover{outline:1.5px solid #333}
#panel{position:fixed;right:16px;top:16px;width:400px;max-height:92vh;overflow:auto;
 background:#fff;border:2px solid #444;border-radius:10px;padding:12px 14px;display:none;
 box-shadow:0 4px 18px rgba(0,0,0,.25);font-size:13px}
#panel h4{margin:4px 0}
.bar{height:11px;background:#7aa7d9;display:inline-block;vertical-align:middle}
.bar.t{background:#d9917a}
.bar.b{background:#69b569}
.scale{position:relative;height:16px;background:linear-gradient(90deg,#eee,#ddd);border-radius:4px}
.scale .mark{position:absolute;top:-2px;width:2px;height:20px;background:#333}
.scale .theta{background:#c33}
.lgd{font-size:12px;color:#555;margin:8px 0}
.turnhdr{margin:14px 0 4px;font-weight:700;color:#333;border-top:1px solid #ddd;padding-top:8px}
.kv{color:#555}
"""

JS = """
function showPanel(el){
  const d = JSON.parse(el.dataset.info);
  const p = document.getElementById('panel');
  let h = '<h4>pos ' + d.pos + (d.state ? ' · ' + d.state : '') + '</h4>';
  if(d.D !== undefined){
    const t = d.theta, D = d.D, mx = Math.max(D, t) * 1.4 + 1e-9;
    h += '<div class="kv">D = ' + D.toFixed(4) + '  vs  θ_D = ' + (t===null?'?':t.toFixed(4)) + '</div>';
    if(t !== null){
      h += '<div class="scale"><div class="mark" style="left:' + (100*D/mx) + '%"></div>' +
           '<div class="mark theta" style="left:' + (100*t/mx) + '%"></div></div>' +
           '<div class="lgd">黑=本位置 D,红=θ_D(超过红线即开门)</div>';
    }
  }
  if(d.draft !== undefined){
    h += '<div class="kv">' + (d.rejected ? '拒绝重采: <b>' + esc(d.draft_s) + '</b> → <b>' + esc(d.final_s) + '</b>'
                                          : '接受: <b>' + esc(d.final_s) + '</b>') + '</div>';
    h += '<div class="kv">η=' + d.eta.toFixed(3) + '  W=π/b=' + d.W.toFixed(3) +
         '  预算余额=' + (d.budget===null?'?':d.budget.toFixed(2)) + (d.probe?'  <b>[探针翻转]</b>':'') + '</div>';
  }
  if(d.top && d.top.length){
    h += '<h4>π / p_T / 倾斜后 b(top-16)</h4>';
    h += '<div class="lgd">蓝=π 绿=b(重算) 橙=p_T;标注 ℓ̃=clip(logp_T−logπ,±c)</div>';
    // 倾斜后 b 重算:b ∝ π·exp(η·ℓ̃) on shown set(展示近似,S 头部)
    let Z = 0, rows = [];
    for(const r of d.top){
      const pi = Math.exp(r.pi), pt = (r.pt===null? null : Math.exp(r.pt));
      const ell = (r.pt===null? 0 : Math.max(-d.c, Math.min(d.c, r.pt - r.pi)));
      const bw = pi * Math.exp(d.eta * ell);
      Z += bw; rows.push({t:r.t, pi:pi, pt:pt, b:bw, ell:ell});
    }
    const piZ = rows.reduce((a,r)=>a+r.pi,0);
    for(const r of rows){ r.b = r.b / Z * piZ; }   // 保 shown 集质量(展示口径)
    const m = Math.max(...rows.map(r=>Math.max(r.pi, r.b, r.pt||0)));
    for(const r of rows){
      h += '<div style="margin:2px 0"><code>' + esc(r.t) + '</code> ' +
        '<span class="lgd">ℓ̃=' + r.ell.toFixed(2) + '</span><br>' +
        '<span class="bar" style="width:' + (170*r.pi/m) + 'px"></span> ' +
        '<span class="bar b" style="width:' + (170*r.b/m) + 'px"></span> ' +
        (r.pt===null ? '<span class="lgd">p_T 窗外</span>'
                     : '<span class="bar t" style="width:' + (170*r.pt/m) + 'px"></span>') +
        '</div>';
    }
  }
  h += '<div style="margin-top:8px"><a href="#" onclick="document.getElementById(\\'panel\\').style.display=\\'none\\';return false">关闭</a></div>';
  p.innerHTML = h; p.style.display = 'block';
}
function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;'); }
document.addEventListener('keydown', e => { if(e.key==='Escape') document.getElementById('panel').style.display='none'; });
"""


# ------------------------------------------------------------------ helpers

def tok_texts(tok, ids):
    """逐 token 显示文本(BBPE 标记转义;decode 语义近似,viz 够用)。"""
    toks = tok.convert_ids_to_tokens(list(ids))
    out = []
    for t in toks:
        t = t.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ")
        out.append(t)
    return out


def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def line_png(xs, ys, title, band=None):
    if not HAS_MPL:
        return ""
    fig, ax = plt.subplots(figsize=(5.6, 2.2))
    ax.plot(xs, ys, marker=".", lw=1)
    if band:
        ax.axhspan(band[0], band[1], alpha=0.12, color="green")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    return f'<img src="data:image/png;base64,{fig_b64(fig)}">'


# ------------------------------------------------------------------ 轨迹页

def render_traj(payload, tok, cfg) -> str:
    st = payload.get("steering") or {}
    resp_ids = payload.get("response_token_ids")
    spans = payload.get("turn_token_spans") or []
    theta = st.get("steer_theta_d")
    c_val = float(cfg.get("c", 2.5))
    eta_max = float(cfg.get("eta_max", 0.5))

    # 稀疏行按 position 索引
    rows = {}
    pos_arr = st.get("steer_positions") or []
    for i, p in enumerate(pos_arr):
        def g(key, default=None):
            a = st.get(key) or []
            return a[i] if i < len(a) else default
        rows[int(p)] = {
            "eta": g("steer_eta", 0.0), "d": g("steer_d", 0.0),
            "rejected": bool(g("steer_rejected", 0)), "probe": bool(g("steer_probe", 0)),
            "open": bool(g("steer_base_open", 0)),
            "log_pi": g("steer_log_pi", 0.0), "log_b": g("steer_log_b", 0.0),
            "budget": g("steer_budget_after"), "draft": g("steer_draft_token", -1),
            "final": g("steer_final_token", -1),
            "pi16": g("steer_pi_top16") or [], "pt16": g("steer_pt_top16") or [],
        }
    cand = dict(zip(st.get("steer_cand_positions") or [], st.get("steer_d_all") or []))

    head = (
        f"<h2>{html.escape(str(payload.get('task_id')))}</h2>"
        f"<p class='kv'>step={payload.get('global_step')} won=<b>{payload.get('won')}</b> "
        f"outcome={payload.get('outcome')} 轮数={payload.get('assistant_turns')} "
        f"θ_D={theta if theta is not None else '?'} | 倾斜 {st.get('steer_count', 0)} 次 · "
        f"拒绝 {st.get('steer_reject_count', 0)} · 预算 {st.get('steer_budget_spent', 0):.2f}"
        f"/{cfg.get('b_max', '?')} · 候选 {st.get('steer_n_candidates', 0)} · "
        f"开门率 {st.get('steer_gate_open_rate', 0):.1%} · 旁路 {st.get('steer_n_bypassed', 0)}</p>"
        "<p class='lgd'>着色:<span class='tk cand'>候选未开门</span> "
        "<span class='tk acc'>开门-接受</span> <span class='tk rej'>开门-拒绝重采</span> "
        "<span class='tk probe'>探针翻转</span> <span class='struct'>结构段/非候选</span>"
        " —— 点击彩色 token 看下钻面板(Esc 关闭)</p>"
    )

    if resp_ids and tok is not None:
        texts = tok_texts(tok, resp_ids)
        turn_of = {}
        for t_idx, (a, b) in enumerate(spans):
            for p in range(a, min(b, len(resp_ids))):
                turn_of[p] = t_idx
        parts = []
        cur_turn = None
        for p, txt in enumerate(texts):
            t_idx = turn_of.get(p)
            if t_idx is not None and t_idx != cur_turn:
                cur_turn = t_idx
                parts.append(f"</div><div class='turnhdr'>turn {t_idx + 1}</div><div class='tok'>")
            e = html.escape(txt)
            r = rows.get(p)
            if r is not None:
                cls = "rej" if r["rejected"] else ("acc" if r["eta"] > 0 else "cand")
                if r["probe"]:
                    cls += " probe"
                info = {
                    "pos": p, "state": "拒绝重采" if r["rejected"] else ("倾斜-接受" if r["eta"] > 0 else "门参与(η=0)"),
                    "D": r["d"], "theta": theta, "eta": r["eta"], "c": c_val,
                    "W": float(__import__("math").exp(r["log_pi"] - r["log_b"])),
                    "budget": r["budget"], "probe": r["probe"],
                    "rejected": r["rejected"], "draft": r["draft"],
                    "draft_s": (tok_texts(tok, [r["draft"]])[0] if r["draft"] >= 0 else "?"),
                    "final_s": (tok_texts(tok, [r["final"]])[0] if r["final"] >= 0 else "?"),
                    "top": _merge_top(r["pi16"], r["pt16"]),
                }
                inline = ""
                if r["rejected"] and r["draft"] >= 0:
                    inline = f"<s class='lgd'>{html.escape(tok_texts(tok, [r['draft']])[0])}</s>→"
                parts.append(
                    f"<span class='tk {cls}' onclick='showPanel(this)' "
                    f"data-info='{html.escape(json.dumps(info, ensure_ascii=False))}'>{inline}{e}</span>"
                )
            elif p in cand:
                info = {"pos": p, "state": "候选-未开门", "D": cand[p], "theta": theta, "c": c_val}
                parts.append(
                    f"<span class='tk cand' onclick='showPanel(this)' "
                    f"data-info='{html.escape(json.dumps(info))}'>{e}</span>"
                )
            elif t_idx is None:
                parts.append(f"<span class='struct'>{e}</span>")
            else:
                parts.append(e)
        body = "<div class='tok'>" + "".join(parts) + "</div>"
    else:
        # 降级:无 token ids / 无 tokenizer —— turn 文本 + steer 行表
        steps = (payload.get("trajectory") or {}).get("steps") or []
        blocks = []
        for t_idx, step in enumerate(steps):
            span = spans[t_idx] if t_idx < len(spans) else None
            seg = [(p, r) for p, r in sorted(rows.items()) if span and span[0] <= p < span[1]]
            trs = "".join(
                f"<tr><td>{p}</td><td>{r['draft']}</td><td>{r['final']}</td>"
                f"<td>{'✗' if r['rejected'] else '✓'}</td><td>{r['eta']:.2f}</td>"
                f"<td>{r['d']:.3f}</td></tr>" for p, r in seg)
            tbl = (f"<table><tr><th>pos</th><th>draft</th><th>final</th><th></th><th>η</th><th>D</th></tr>{trs}</table>"
                   if seg else "")
            blocks.append(
                f"<div class='turnhdr'>turn {t_idx + 1}</div>"
                f"<div class='tok'>{html.escape(step.get('thought', ''))}</div>"
                f"<div><b>▶ {html.escape(step.get('action_norm') or '')}</b></div>{tbl}")
        body = ("<p class='lgd'>(旧 schema dump 或未给 --tokenizer:降级为 turn 表格模式)</p>"
                + "".join(blocks))

    return (f"<!doctype html><meta charset='utf-8'><title>traj</title><style>{CSS}</style>"
            f"{head}{body}<div id='panel'></div><script>{JS}</script>")


def _merge_top(pi16, pt16):
    pt_map = {int(t): float(lp) for t, lp in (pt16 or [])}
    out = []
    for t, lp in (pi16 or []):
        out.append({"t": str(int(t)), "pi": float(lp), "pt": pt_map.get(int(t))})
    return out


# ------------------------------------------------------------------ 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--steps", default="", help="只渲染指定 step(逗号分隔);默认全部")
    ap.add_argument("--max-trajs", type=int, default=None, help="每 step 最多渲染的轨迹页数")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out_dir) if args.out_dir else run_dir / "viz"
    (out / "traj").mkdir(parents=True, exist_ok=True)
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # 配置(下钻重算要 c/eta_max/b_max)
    cfg = {}
    eff = run_dir / "effective_config.yaml"
    if eff.exists():
        try:
            import yaml

            cfg = (yaml.safe_load(eff.read_text(encoding="utf-8")) or {}).get("steering") or {}
        except Exception:  # noqa: BLE001
            pass

    want = {int(x) for x in args.steps.split(",") if x.strip()} if args.steps else None
    step_dirs = sorted(
        d for d in (run_dir / "rollouts").glob("step_*")
        if d.is_dir() and not d.name.endswith("_val")
    )

    per_step = []
    step_pages = []
    for sd in step_dirs:
        step_no = int(sd.name.split("_")[1])
        if want and step_no not in want:
            continue
        trajs = []
        agg = defaultdict(float)
        n = 0
        for f in sorted(sd.glob("*.json")):
            try:
                p = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            st = p.get("steering") or {}
            n += 1
            agg["won"] += bool(p.get("won"))
            agg["open"] += st.get("steer_n_base_open", 0)
            agg["cand"] += st.get("steer_n_candidates", 0)
            agg["steer"] += st.get("steer_count", 0)
            agg["rej"] += st.get("steer_reject_count", 0)
            agg["budget"] += st.get("steer_budget_spent", 0.0)
            agg["theta"] += st.get("steer_theta_d", 0.0)
            trajs.append((f, p, st))
        if not n:
            continue
        # 轨迹排序:先拒绝数降序(最有信息量),再 won
        trajs.sort(key=lambda x: (-(x[2].get("steer_reject_count", 0)), -(x[2].get("steer_count", 0))))
        rendered = trajs[: args.max_trajs] if args.max_trajs else trajs
        rows_html = []
        for f, p, st in trajs:
            name = f.stem + ".html"
            is_r = any(ff.stem == f.stem for ff, _, _ in rendered)
            if is_r:
                (out / "traj" / name).write_text(render_traj(p, tok, cfg), encoding="utf-8")
            link = f"<a href='traj/{name}'>{html.escape(p.get('task_id', '')[:70])}</a>" if is_r \
                else html.escape(p.get("task_id", "")[:70])
            rows_html.append(
                f"<tr><td>{link}</td><td>{'✓' if p.get('won') else ''}</td>"
                f"<td>{st.get('steer_count', 0)}</td><td>{st.get('steer_reject_count', 0)}</td>"
                f"<td>{st.get('steer_budget_spent', 0):.1f}</td>"
                f"<td>{st.get('steer_gate_open_rate', 0):.1%}</td>"
                f"<td>{st.get('steer_n_bypassed', 0)}</td><td>{p.get('outcome', '')}</td></tr>"
            )
        page = (f"<!doctype html><meta charset='utf-8'><title>step {step_no}</title><style>{CSS}</style>"
                f"<h2><a href='index.html'>←</a> step {step_no}({n} 轨迹,won {agg['won']/n:.1%})</h2>"
                "<table><tr><th>任务(点击看逐 token)</th><th>won</th><th>倾斜</th><th>拒绝</th>"
                "<th>预算</th><th>门率</th><th>旁路</th><th>outcome</th></tr>"
                + "".join(rows_html) + "</table>")
        (out / f"step_{step_no:05d}.html").write_text(page, encoding="utf-8")
        step_pages.append((step_no, n, agg))
        per_step.append({
            "step": step_no,
            "won": agg["won"] / n,
            "gate": (agg["open"] / agg["cand"]) if agg["cand"] else 0.0,
            "rej_rate": (agg["rej"] / agg["steer"]) if agg["steer"] else 0.0,
            "budget": agg["budget"] / n,
            "theta": agg["theta"] / n,
        })

    xs = [r["step"] for r in per_step]
    charts = "".join([
        line_png(xs, [r["gate"] for r in per_step], "gate open rate", band=(0.05, 0.25)),
        line_png(xs, [r["theta"] for r in per_step], "theta_D (EMA trajectory)"),
        line_png(xs, [r["rej_rate"] for r in per_step], "reject rate"),
        line_png(xs, [r["budget"] for r in per_step], "budget spent / traj"),
    ])
    nav = "".join(
        f"<tr><td><a href='step_{s:05d}.html'>step {s}</a></td><td>{n}</td>"
        f"<td>{a['won']/n:.1%}</td><td>{(a['open']/a['cand']) if a['cand'] else 0:.1%}</td>"
        f"<td>{a['steer']/n:.1f}</td><td>{a['rej']/n:.2f}</td><td>{a['budget']/n:.1f}</td></tr>"
        for s, n, a in step_pages
    )
    idx = (f"<!doctype html><meta charset='utf-8'><title>PGS viz</title><style>{CSS}</style>"
           f"<h1>PGS 逐 token 可视化 · {html.escape(str(run_dir))}</h1>"
           f"<div>{charts}</div>"
           "<table><tr><th>step</th><th>轨迹</th><th>won</th><th>门率</th>"
           "<th>倾斜/轨迹</th><th>拒绝/轨迹</th><th>预算/轨迹</th></tr>"
           + nav + "</table>"
           "<p class='lgd'>深度体检另见 pgs_report.py;本系统管定位与逐例下钻。</p>")
    (out / "index.html").write_text(idx, encoding="utf-8")
    print(f"viz: {out}/index.html  (steps={len(step_pages)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
