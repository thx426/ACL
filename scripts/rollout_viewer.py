#!/usr/bin/env python3
"""executor rollout 查看器 —— 看任务执行轨迹的每一步与模型原始输出。

与既有工具的分工:
  * pgs_traj.py          单条 dump → 静态 HTML,标注 steering 的 token 级明细
  * researcher_viewer.py 读 sessions 目录,看 researcher 会话与产出文档
  * 本工具               读 run 的 rollouts/ 目录,浏览 **executor 轨迹**:
                         逐步 thought/action/observation + 模型**原始输出**
                         (FC 协议下即 <tool_call><function=...> 形态)

两种视图并存的理由:解析后视图看"agent 在想什么、做了什么";原始视图看
"模型到底吐了什么字节" —— 协议切换(text → Function Call)后,格式失败率、
调用形态是否规范只能在原始视图里看清。

用法:
  python3 rollout_viewer.py --run /path/to/runs/<name> [--tokenizer <hf-path>] \
                            --port 8802 --host 127.0.0.1
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

_TOK = None


def load_tokenizer(path: str | None):
    global _TOK
    if not path:
        return None
    try:
        from transformers import AutoTokenizer

        _TOK = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001 — 无 tokenizer 只是降级,不该中断服务
        print(f"[viewer] tokenizer 不可用({exc});原始视图将不可用")
    return _TOK


def family_of(task_id: str) -> str:
    """ALFWorld 的 task family 藏在 task_id 的中段(split/family-Obj-.../trial_x)。"""
    m = re.search(r"/([a-z_]+)-", task_id or "")
    return m.group(1) if m else "?"


def list_steps(run_dir: str) -> list[str]:
    root = os.path.join(run_dir, "rollouts")
    if not os.path.isdir(root):
        return []
    return sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )


def load_rows(run_dir: str, step: str) -> list[dict]:
    rows: list[dict] = []
    for fp in sorted(glob.glob(os.path.join(run_dir, "rollouts", step, "*.json"))):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        for r in d if isinstance(d, list) else (d.get("rows") or [d]):
            if isinstance(r, dict):
                r["_file"] = os.path.basename(fp)
                rows.append(r)
    return rows


def is_research(r: dict) -> bool:
    return str(r.get("sample_role")) == "research"


def arm_of(r: dict) -> str:
    """臂标签。steer_arm 落在 steering 子字典里(顶层同名键恒为 None,
    2026-08-09 实测 —— 早期版本读顶层导致臂筛选恒空)。"""
    if is_research(r):
        return "research"
    a = (r.get("steering") or {}).get("steer_arm")
    return {0: "bare", 1: "steered"}.get(a, "?")


def rmetric(r: dict, key: str):
    return (r.get("research") or {}).get(key)


def raw_turns(row: dict) -> list[str]:
    """按 turn_token_spans 把 response_token_ids 切回每轮的模型原始输出。"""
    if _TOK is None:
        return []
    ids = row.get("response_token_ids") or []
    spans = row.get("turn_token_spans") or []
    out = []
    for s in spans:
        try:
            a, b = int(s[0]), int(s[1])
        except Exception:
            continue
        out.append(_TOK.decode(ids[a:b], skip_special_tokens=False))
    return out


def stats(rows: list[dict]) -> dict:
    rows = [r for r in rows if not is_research(r)] or rows
    n = len(rows) or 1
    won = sum(1 for r in rows if r.get("won"))
    trunc = sum(1 for r in rows if r.get("truncated"))
    inval = sum(1 for r in rows if (r.get("invalid_format") or 0) > 0)
    steps = sorted(int(r.get("env_steps") or 0) for r in rows)
    turns = sorted(int(r.get("assistant_turns") or 0) for r in rows)
    outc: dict[str, int] = {}
    for r in rows:
        outc[str(r.get("outcome"))] = outc.get(str(r.get("outcome")), 0) + 1
    med = lambda v: v[len(v) // 2] if v else 0  # noqa: E731
    return {
        "n": len(rows),
        "won": won,
        "won_rate": won / n * 100,
        "truncated": trunc,
        "invalid_any": inval,
        "env_steps_med": med(steps),
        "env_steps_max": steps[-1] if steps else 0,
        "turns_med": med(turns),
        "turns_max": turns[-1] if turns else 0,
        "outcomes": outc,
    }


CSS = """
:root{--bg:#fbfaf8;--fg:#1c1a17;--mut:#6b6560;--line:#e3ded6;--card:#fff;
--ok:#1a7f4b;--bad:#b3261e;--warn:#8a6d00;--accent:#2d5f8a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif}
header{position:sticky;top:0;z-index:9;background:var(--card);border-bottom:1px solid var(--line);
padding:10px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
header b{font-size:15px}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
select,input{font:inherit;padding:3px 6px;border:1px solid var(--line);border-radius:5px;background:#fff}
.wrap{padding:14px 16px;max-width:1180px;margin:0 auto}
.stats{display:flex;gap:18px;flex-wrap:wrap;background:var(--card);border:1px solid var(--line);
border-radius:8px;padding:10px 14px;margin-bottom:14px}
.stat b{display:block;font-size:19px;font-variant-numeric:tabular-nums}
.stat span{color:var(--mut);font-size:12px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:8px}
th,td{padding:6px 10px;text-align:left;border-bottom:1px solid var(--line);font-size:13px}
th{background:#f5f2ec;font-weight:600;position:sticky;top:52px}
tr:last-child td{border-bottom:none}
td.num{font-variant-numeric:tabular-nums;text-align:right}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad)}.warn{color:var(--warn)}
.step{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:10px 0;overflow:hidden}
.step>.hd{background:#f5f2ec;padding:5px 12px;font-size:12px;color:var(--mut);
display:flex;gap:12px;align-items:center}
.step .body{padding:10px 12px}
.lbl{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);margin:8px 0 3px}
.thought{white-space:pre-wrap;color:#332f2a}
.act{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#eef4f9;
border-left:3px solid var(--accent);padding:5px 9px;border-radius:0 5px 5px 0;white-space:pre-wrap}
.obs{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#f3f1ec;
padding:5px 9px;border-radius:5px;white-space:pre-wrap;color:#4a453f}
pre.code{margin:0;padding:9px 11px;background:#f7f5f0;border:1px solid var(--line);
border-left:3px solid var(--accent);border-radius:0 6px 6px 0;overflow-x:auto;
font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap}
pre.raw{margin:0;padding:9px 11px;background:#241f1b;color:#e6ddd2;border-radius:6px;
overflow-x:auto;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap}
pre.raw .tc{color:#7fd18a}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;font-weight:600}
.b-ok{background:#dff3e6;color:var(--ok)}.b-bad{background:#fbe3e1;color:var(--bad)}
.b-warn{background:#faf0cf;color:var(--warn)}
.b-res{background:#e7e0f5;color:#5b3fa0}
.muted{color:var(--mut)}
"""


def h(s) -> str:
    return html.escape(str(s if s is not None else ""))


def page(body: str, title: str) -> bytes:
    return (
        f"<!doctype html><meta charset=utf-8><title>{h(title)}</title>"
        f"<style>{CSS}</style>{body}"
    ).encode()


class Handler(BaseHTTPRequestHandler):
    run_dir = ""
    run_name = ""

    def log_message(self, *a):  # 静音
        pass

    def _send(self, payload: bytes, ctype="text/html; charset=utf-8"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)
        steps = list_steps(self.run_dir)
        if not steps:
            self._send(page("<div class=wrap>run 目录下还没有 rollouts/</div>", "empty"))
            return
        train_steps = [d for d in steps if not d.endswith("_val")]
        default_step = train_steps[-1] if train_steps else steps[-1]
        step = (q.get("step") or [default_step])[0]
        if step not in steps:
            step = steps[0]
        rows = load_rows(self.run_dir, step)

        if u.path == "/traj":
            self._send(self._traj(rows, int((q.get("i") or ["0"])[0]), step, steps))
            return
        self._send(self._index(rows, step, steps, q))

    # ---------------------------------------------------------------- 列表页
    def _index(self, rows, step, steps, q) -> bytes:
        fam = (q.get("family") or [""])[0]
        out = (q.get("outcome") or [""])[0]
        arm = (q.get("arm") or [""])[0]
        view = [r for r in rows]
        if fam:
            view = [r for r in view if family_of(r.get("task_id", "")) == fam]
        if out:
            view = [r for r in view if str(r.get("outcome")) == out]
        if arm:
            view = [r for r in view if arm_of(r) == arm]

        s = stats(view)
        fams = sorted({family_of(r.get("task_id", "")) for r in rows})
        outs = sorted({str(r.get("outcome")) for r in rows})

        sel = lambda name, cur, opts: (  # noqa: E731
            f"<select name={name} onchange='this.form.submit()'>"
            + "".join(
                f"<option value='{h(o)}'{' selected' if o == cur else ''}>{h(o or '全部')}</option>"
                for o in [""] + list(opts)
            )
            + "</select>"
        )
        n_res = sum(1 for r in rows if is_research(r))
        n_exe = len(rows) - n_res
        jump = (f" <a href='/?step={h(step)}&arm=research'>[只看 research]</a>"
                if n_res else " <span class=muted>(本步无 research 行)</span>")
        head = (
            f"<header><b>{h(self.run_name)}</b>"
            f"<span class=muted>executor {n_exe} · research {n_res}{jump}</span>"
            f"<form style='display:flex;gap:10px;align-items:center' method=get>"
            f"步骤 {sel('step', step, steps)} "
            f"family {sel('family', fam, fams)} "
            f"结局 {sel('outcome', out, outs)} "
            f"角色/臂 {sel('arm', arm, ['bare', 'steered', 'research'])}"
            f"</form></header>"
        )
        res_rows = [r for r in view if is_research(r)]
        if arm == "research" and res_rows:
            import statistics as _S

            g = lambda k: [x for x in (rmetric(r, k) for r in res_rows)  # noqa: E731
                           if isinstance(x, (int, float))]
            rr, dw, ex = g("research_r_raw"), g("research_doc_words"), g("research_experiments")
            nofin = g("research_no_finalize")
            pairs = (
                ("research 行", len(res_rows)),
                ("★ r_raw 中位", f"{_S.median(rr):.2f}" if rr else "-"),
                ("r_raw>0", f"{sum(1 for x in rr if x > 0)}/{len(rr)}" if rr else "-"),
                ("未产出文档", f"{sum(nofin)}/{len(nofin)}" if nofin else "-"),
                ("文档词数 中位", f"{_S.median(dw):.0f}" if dw else "-"),
                ("实验次数 中位", f"{_S.median(ex):.0f}" if ex else "-"),
            )
        else:
            pairs = (
                ("轨迹数", s["n"]),
                ("成功", f"{s['won']} ({s['won_rate']:.1f}%)"),
                ("截断", s["truncated"]),
                ("有格式失败", s["invalid_any"]),
                ("env_steps 中位/max", f"{s['env_steps_med']}/{s['env_steps_max']}"),
                ("轮数 中位/max", f"{s['turns_med']}/{s['turns_max']}"),
            )
        cards = "".join(
            f"<div class=stat><b>{v}</b><span>{k}</span></div>" for k, v in pairs
        )
        outline = " · ".join(f"{h(k)} {v}" for k, v in sorted(s["outcomes"].items()))
        ARMC = {"bare": "b-warn", "steered": "b-ok", "research": "b-res"}
        trs = []
        for r in view:
            a = arm_of(r)
            badge = f"<span class='badge {ARMC.get(a,'')}'>{h(a)}</span>"
            if is_research(r):
                rr = rmetric(r, "research_r_raw")
                col3 = ("<span class=ok>r_raw " + f"{rr:.2f}</span>") if isinstance(rr, (int, float)) else "-"
                c4 = h(rmetric(r, "research_experiments"))
                c5 = h(rmetric(r, "research_turns"))
                c6 = h(rmetric(r, "research_doc_words"))
                c7 = "未出文档" if rmetric(r, "research_no_finalize") else ""
            else:
                w = r.get("won")
                col3 = (f"<span class='{'ok' if w else 'bad'}'>{'✓' if w else '✗'} "
                        f"{h(r.get('outcome'))}</span>")
                c4 = h(r.get("env_steps"))
                c5 = h(r.get("assistant_turns"))
                c6 = h(r.get("invalid_format"))
                c7 = "截断" if r.get("truncated") else ""
            trs.append(
                f"<tr><td><a href='/traj?i={rows.index(r)}&step={h(step)}'>"
                f"{h((r.get('task_id') or '')[:62])}</a></td>"
                f"<td>{badge}</td>"
                f"<td>{h(family_of(r.get('task_id','')))}</td>"
                f"<td>{col3}</td>"
                f"<td class=num>{c4}</td><td class=num>{c5}</td><td class=num>{c6}</td>"
                f"<td class=num>{h(r.get('response_token_len'))}</td>"
                f"<td>{c7}</td></tr>"
            )
        body = (
            head
            + "<div class=wrap>"
            + f"<div class=stats>{cards}</div>"
            + f"<p class=muted>结局分布:{outline}</p>"
            + "<table><tr><th>task</th><th>角色</th><th>family</th>"
            "<th>结局 / r_raw</th><th>env步 / 实验</th><th>轮数</th>"
            "<th>格式失败 / doc词</th><th>resp token</th><th></th></tr>"
            + "".join(trs)
            + "</table></div>"
        )
        return page(body, f"{self.run_name} · {step}")

    # ---------------------------------------------------------------- 详情页
    def _traj(self, rows, idx, step, steps) -> bytes:
        if not (0 <= idx < len(rows)):
            return page("<div class=wrap>索引越界</div>", "oob")
        r = rows[idx]
        if is_research(r):
            return self._research(r, idx, len(rows), step)
        tr = r.get("trajectory") or {}
        st = tr.get("steps") or []
        raws = raw_turns(r)

        won = r.get("won")
        hd = (
            f"<header><a href='/?step={h(step)}'>&larr; 列表</a>"
            f"<b>{h((r.get('task_id') or '')[:80])}</b>"
            f"<span class='badge {'b-ok' if won else 'b-bad'}'>{h(r.get('outcome'))}</span>"
            f"<span class=muted>env步 {h(r.get('env_steps'))} · 轮 {h(r.get('assistant_turns'))}"
            f" · 格式失败 {h(r.get('invalid_format'))}"
            f" · resp {h(r.get('response_token_len'))} tok</span></header>"
        )
        parts = []
        init = tr.get("initial_obs")
        if init:
            parts.append(
                f"<div class=step><div class=hd>初始观察</div>"
                f"<div class=body><div class=obs>{h(init)}</div></div></div>"
            )
        for i, s in enumerate(st):
            acted, valid = s.get("acted"), s.get("valid")
            flag = ""
            if not acted:
                flag = "<span class='badge b-bad'>未产生动作</span>"
            elif valid is False or valid == "":
                flag = "<span class='badge b-warn'>动作无效</span>"
            raw = raws[i] if i < len(raws) else None
            raw_block = ""
            if raw:
                marked = h(raw)
                for tag in ("&lt;tool_call&gt;", "&lt;/tool_call&gt;",
                            "&lt;function=", "&lt;parameter=", "&lt;/function&gt;",
                            "&lt;/parameter&gt;"):
                    marked = marked.replace(tag, f"<span class=tc>{tag}</span>")
                raw_block = (
                    f"<div class=lbl>模型原始输出(FC 形态)</div><pre class=raw>{marked}</pre>"
                )
            # SSB:action_raw 是 JSON 工具调用,把 run_python 的 code 拆出来做代码块
            act_raw = str(s.get("action_raw") or "")
            act_block = f"<div class=lbl>action</div><div class=act>{h(act_raw)}</div>"
            if act_raw.lstrip().startswith("{"):
                try:
                    call = json.loads(act_raw)
                    nm = call.get("name", "?")
                    args = call.get("arguments") or {}
                    code = args.get("code")
                    if code:
                        act_block = (
                            f"<div class=lbl>action · {h(nm)}(code)</div>"
                            f"<pre class=code>{h(code)}</pre>"
                        )
                    else:
                        rest = {k: v for k, v in args.items()}
                        act_block = (
                            f"<div class=lbl>action</div><div class=act>{h(nm)}("
                            f"{h(json.dumps(rest, ensure_ascii=False))})</div>"
                        )
                except Exception:
                    pass
            parts.append(
                f"<div class=step><div class=hd>step {i+1} {flag}</div><div class=body>"
                f"<div class=lbl>thought</div><div class=thought>{h(s.get('thought'))}</div>"
                f"{act_block}"
                f"<div class=lbl>observation</div><div class=obs>{h(s.get('observation'))}</div>"
                f"{raw_block}</div></div>"
            )
        nav = (
            f"<p><a href='/traj?i={max(0, idx-1)}&step={h(step)}'>← 上一条</a> · "
            f"<a href='/traj?i={min(len(rows)-1, idx+1)}&step={h(step)}'>下一条 →</a> "
            f"<span class=muted>({idx+1}/{len(rows)})</span></p>"
        )
        note = "" if _TOK else "<p class=muted>未提供 --tokenizer,原始输出视图不可用</p>"
        return page(hd + "<div class=wrap>" + nav + note + "".join(parts) + nav + "</div>",
                    f"traj {idx}")


    # ------------------------------------------------------ research 详情页
    def _research(self, r, idx, total, step) -> bytes:
        m = r.get("research") or {}
        rr = m.get("research_r_raw")
        hd = (
            f"<header><a href='/?step={h(step)}&arm=research'>&larr; 列表</a>"
            f"<b>{h((r.get('task_id') or '')[:70])}</b>"
            f"<span class='badge b-res'>research · session {h(r.get('session_id'))}</span>"
            f"<span class=muted>"
            f"r_raw {rr if rr is None else format(rr, '.2f')} · "
            f"实验 {h(m.get('research_experiments'))} · "
            f"轮 {h(m.get('research_turns'))} · "
            f"episode {h(m.get('research_episodes'))} · "
            f"doc {h(m.get('research_doc_words'))} 词</span></header>"
        )
        parts = []

        # ① 产出的文档(最关心的东西,放最前)
        doc = r.get("doc") or ""
        parts.append(
            "<div class=step><div class=hd>① 产出的经验文档"
            + (f" <span class='badge b-bad'>未产出</span>" if not doc else "")
            + "</div><div class=body><div class=thought>"
            + (h(doc) or "<span class=muted>(空)</span>")
            + "</div></div></div>"
        )

        # ② 续写评估:文档到底有没有用
        cont = r.get("continuations") or []
        if cont:
            won = sum(1 for c in cont if c.get("won"))
            rowsh = "".join(
                f"<tr><td class=num>{i+1}</td>"
                f"<td class='{'ok' if c.get('won') else 'bad'}'>{'✓' if c.get('won') else '✗'}</td>"
                f"<td class=num>{h(c.get('env_steps'))}</td>"
                f"<td class=num>{h(c.get('gen_tokens'))}</td>"
                f"<td>{'截断' if c.get('truncated') else ''}</td>"
                f"<td class=muted>{h(c.get('error') if c.get('error') not in (None,'None') else '')}</td></tr>"
                for i, c in enumerate(cont)
            )
            parts.append(
                f"<div class=step><div class=hd>② 基于该文档的续写评估 —— "
                f"成功 {won}/{len(cont)} = {won/len(cont)*100:.0f}%"
                f"(即 reward r_raw 的来源)</div><div class=body>"
                f"<table><tr><th>#</th><th>结果</th><th>env步</th><th>生成token</th>"
                f"<th></th><th>error</th></tr>{rowsh}</table></div></div>"
            )

        # ③ researcher 会话全过程
        for i, msg in enumerate(r.get("messages") or []):
            role = str(msg.get("role"))
            cls = {"system": "obs", "user": "obs", "assistant": "thought"}.get(role, "obs")
            content = str(msg.get("content") or "")
            tc = msg.get("tool_calls")
            tcblock = ""
            if tc:
                tcblock = ("<div class=lbl>tool_calls</div><div class=act>"
                           + h(json.dumps(tc, ensure_ascii=False, indent=1)[:1500]) + "</div>")
            parts.append(
                f"<div class=step><div class=hd>③ 会话 msg {i} · {h(role)}</div>"
                f"<div class=body><div class={cls}>{h(content[:6000])}</div>{tcblock}</div></div>"
            )

        nav = (
            f"<p><a href='/traj?i={max(0, idx-1)}&step={h(step)}'>← 上一条</a> · "
            f"<a href='/traj?i={min(total-1, idx+1)}&step={h(step)}'>下一条 →</a> "
            f"<span class=muted>({idx+1}/{total})</span></p>"
        )
        return page(hd + "<div class=wrap>" + nav + "".join(parts) + nav + "</div>",
                    f"research {idx}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run 目录(其下应有 rollouts/)")
    ap.add_argument("--tokenizer", default="", help="HF 模型目录;给了才能看模型原始输出")
    ap.add_argument("--port", type=int, default=8802)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    load_tokenizer(a.tokenizer or None)
    Handler.run_dir = os.path.abspath(a.run)
    Handler.run_name = os.path.basename(Handler.run_dir.rstrip("/"))
    steps = list_steps(Handler.run_dir)
    print(f"[viewer] run={Handler.run_name} 可见步骤: {steps or '(暂无)'}")
    print(f"[viewer] http://{a.host}:{a.port}/")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
