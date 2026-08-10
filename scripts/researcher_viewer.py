#!/usr/bin/env python3
"""researcher 会话查看器 —— 看 sftgen_researcher sample 产出的会话过程与最终文档。

与 pgs_microscope 的区别:后者读 run 目录的 rollout json(逐 token 的 steering
明细);本工具读 sessions 目录的会话 json(schema: messages / doc / finalized /
experiments_used / episodes_consumed / turns_used / family / error),用于人工
审阅 researcher 的**推理过程**与**产出文档**,并支持多个模型目录并排对比。

用法:
  python3 researcher_viewer.py --sessions 9B=/path/sessions_9b [4B=/path/sessions_4b] \
                               --port 8801 --host 127.0.0.1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

CMD = re.compile(r"`(go to|take|move|use|open|close|clean|heat|cool|examine|look|inventory)\b[^`]*`", re.I)
PROB = re.compile(r"\b(most (frequently|reliable|common)|rarely|occasionally|usually|typically)\b", re.I)

SOURCES: dict[str, str] = {}


def scan(root: str) -> list[dict]:
    """扫描 sessions 目录 → 轻量索引(不载入 messages,列表要快)。"""
    out = []
    for fp in sorted(glob.glob(os.path.join(root, "*", "s*.json"))):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        doc = d.get("doc") or ""
        out.append({
            "path": os.path.relpath(fp, root),
            "task_id": d.get("task_id") or os.path.basename(os.path.dirname(fp)),
            "family": d.get("family") or "?",
            "session_idx": d.get("session_idx"),
            "finalized": bool(d.get("finalized")),
            "error": d.get("error"),
            "experiments_used": d.get("experiments_used"),
            "episodes_consumed": d.get("episodes_consumed"),
            "turns_used": d.get("turns_used"),
            "doc_words": len(doc.split()),
            "n_cmd": len(CMD.findall(doc)),
            "has_prob": bool(PROB.search(doc)),
            "n_msgs": len(d.get("messages") or []),
        })
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 静音
        pass

    def _send(self, body: bytes, ctype="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            self._send(PAGE.encode(), "text/html")
        elif u.path == "/api/sources":
            self._send(json.dumps(sorted(SOURCES)).encode())
        elif u.path == "/api/index":
            src = (q.get("src") or [""])[0]
            self._send(json.dumps(scan(SOURCES.get(src, ""))).encode())
        elif u.path == "/api/session":
            src = (q.get("src") or [""])[0]
            rel = (q.get("path") or [""])[0]
            root = SOURCES.get(src, "")
            fp = os.path.normpath(os.path.join(root, rel))
            if not fp.startswith(os.path.normpath(root)):   # 路径穿越防护
                self._send(b'{"error":"bad path"}'); return
            try:
                self._send(json.dumps(json.load(open(fp))).encode())
            except Exception as e:
                self._send(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404); self.end_headers()


PAGE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>researcher 会话查看器</title>
<style>
:root{--bg:#f7f8fa;--surface:#fff;--bd:#e4e7ec;--bd2:#d0d5dd;--tx:#111827;--tx2:#4b5563;--tx3:#9ca3af;
 --pri:#4338ca;--pri-bg:#eef2ff;--ok:#047857;--ok-bg:#ecfdf5;--bad:#b91c1c;--bad-bg:#fef2f2;
 --warn:#b45309;--warn-bg:#fffbeb;--code:#f3f4f6;}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0b0e14;--surface:#141821;--bd:#232936;
 --bd2:#2f3746;--tx:#e5e7eb;--tx2:#9ca3af;--tx3:#6b7280;--pri:#a5b4fc;--pri-bg:#1e1b4b;--ok:#6ee7b7;
 --ok-bg:#052e26;--bad:#fca5a5;--bad-bg:#3f1212;--warn:#fcd34d;--warn-bg:#3b2a06;--code:#1b202b;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",
 "PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;display:flex;height:100vh;overflow:hidden}
#side{width:390px;min-width:390px;border-right:1px solid var(--bd);background:var(--surface);
 display:flex;flex-direction:column}
#head{padding:12px 14px;border-bottom:1px solid var(--bd);display:flex;flex-direction:column;gap:8px}
h1{margin:0;font-size:15px;font-weight:650;letter-spacing:-.01em}
#srcbar{display:flex;gap:6px;flex-wrap:wrap}
.srcbtn{padding:5px 12px;border:1px solid var(--bd2);border-radius:7px;background:transparent;color:var(--tx2);
 cursor:pointer;font-size:13px;font-weight:600}
.srcbtn.on{background:var(--pri-bg);border-color:var(--pri);color:var(--pri)}
#stats{font-size:12px;color:var(--tx2);font-variant-numeric:tabular-nums}
#list{flex:1;overflow-y:auto;padding:6px}
.fam{padding:8px 10px 4px;font-size:11px;font-weight:700;color:var(--tx3);text-transform:uppercase;letter-spacing:.06em}
.it{padding:8px 10px;border-radius:8px;cursor:pointer;border:1px solid transparent}
.it:hover{background:var(--bg)}
.it.on{background:var(--pri-bg);border-color:var(--pri)}
.it .t{font-size:12.5px;font-weight:560;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.it .m{font-size:11px;color:var(--tx2);margin-top:3px;display:flex;gap:8px;font-variant-numeric:tabular-nums}
.badge{display:inline-block;padding:1px 6px;border-radius:5px;font-size:10.5px;font-weight:650}
.b-ok{background:var(--ok-bg);color:var(--ok)}.b-bad{background:var(--bad-bg);color:var(--bad)}
.b-warn{background:var(--warn-bg);color:var(--warn)}
#main{flex:1;overflow-y:auto;padding:20px 26px}
#meta{display:flex;gap:18px;flex-wrap:wrap;padding:12px 16px;background:var(--surface);border:1px solid var(--bd);
 border-radius:10px;margin-bottom:16px;font-variant-numeric:tabular-nums}
#meta div{display:flex;flex-direction:column;gap:2px}
#meta .k{font-size:11px;color:var(--tx3);text-transform:uppercase;letter-spacing:.05em}
#meta .v{font-size:16px;font-weight:640}
h2{font-size:13px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:.06em;
 margin:22px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--bd)}
.msg{border:1px solid var(--bd);border-radius:10px;margin-bottom:10px;overflow:hidden;background:var(--surface)}
.mh{padding:7px 12px;font-size:11.5px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;
 cursor:pointer;display:flex;justify-content:space-between;align-items:center}
.r-system .mh{background:var(--warn-bg);color:var(--warn)}
.r-user .mh{background:var(--bg);color:var(--tx2)}
.r-assistant .mh{background:var(--pri-bg);color:var(--pri)}
.mb{padding:12px 14px;white-space:pre-wrap;word-break:break-word;font-size:13px;
 font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;border-top:1px solid var(--bd)}
.mb.hid{display:none}
#doc{background:var(--surface);border:1px solid var(--bd);border-radius:10px;padding:18px 22px;
 white-space:pre-wrap;word-break:break-word}
#doc code,.mb code{background:var(--code);padding:1px 5px;border-radius:4px;
 font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.92em}
.empty{color:var(--tx3);padding:60px 0;text-align:center}
</style></head><body>
<div id="side">
  <div id="head"><h1>researcher 会话查看器</h1><div id="srcbar"></div><div id="stats"></div></div>
  <div id="list"></div>
</div>
<div id="main"><div class="empty">从左侧选择一个会话</div></div>
<script>
let SRC='', IDX=[], CUR=null;
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const mdCode=s=>esc(s).replace(/`([^`\n]+)`/g,'<code>$1</code>');

async function boot(){
  const srcs=await (await fetch('/api/sources')).json();
  document.getElementById('srcbar').innerHTML=srcs.map(s=>
    `<button class="srcbtn" data-s="${esc(s)}">${esc(s)}</button>`).join('');
  document.querySelectorAll('.srcbtn').forEach(b=>b.onclick=()=>pick(b.dataset.s));
  if(srcs.length) pick(srcs[0]);
}
async function pick(s){
  SRC=s;
  document.querySelectorAll('.srcbtn').forEach(b=>b.classList.toggle('on',b.dataset.s===s));
  IDX=await (await fetch('/api/index?src='+encodeURIComponent(s))).json();
  const fin=IDX.filter(x=>x.finalized).length, err=IDX.filter(x=>x.error).length;
  const w=IDX.filter(x=>x.doc_words>0).map(x=>x.doc_words).sort((a,b)=>a-b);
  const med=w.length?w[Math.floor(w.length/2)]:0;
  document.getElementById('stats').textContent=
    `${IDX.length} 会话 · finalize ${IDX.length?(fin/IDX.length*100).toFixed(0):0}% · error ${err} · 文档词数中位 ${med}`;
  render();
}
function render(){
  const by={};
  IDX.forEach((x,i)=>{(by[x.family]=by[x.family]||[]).push([x,i])});
  document.getElementById('list').innerHTML=Object.keys(by).sort().map(f=>
    `<div class="fam">${esc(f)} · ${by[f].length}</div>`+by[f].map(([x,i])=>{
      const b=x.error?'<span class="badge b-bad">error</span>'
            :x.finalized?'<span class="badge b-ok">finalized</span>'
            :'<span class="badge b-warn">no-final</span>';
      const short=x.task_id.replace(/^train__/,'').slice(0,42);
      return `<div class="it" data-i="${i}"><div class="t">${esc(short)} · s${x.session_idx??'?'}</div>
        <div class="m">${b}<span>${x.doc_words}词</span><span>${x.n_cmd}命令</span>
        <span>${x.n_msgs}轮</span>${x.has_prob?'<span class="badge b-ok">概率表述</span>':''}</div></div>`;
    }).join('')).join('');
  document.querySelectorAll('.it').forEach(e=>e.onclick=()=>open_(+e.dataset.i,e));
}
async function open_(i,el){
  document.querySelectorAll('.it').forEach(e=>e.classList.remove('on'));
  el.classList.add('on');
  const x=IDX[i];
  const d=await (await fetch(`/api/session?src=${encodeURIComponent(SRC)}&path=${encodeURIComponent(x.path)}`)).json();
  CUR=d;
  const cell=(k,v)=>`<div><span class="k">${k}</span><span class="v">${v??'—'}</span></div>`;
  const msgs=(d.messages||[]).map((m,j)=>{
    const c=typeof m.content==='string'?m.content:JSON.stringify(m.content,null,2);
    const hid=m.role==='system'?' hid':'';
    return `<div class="msg r-${esc(m.role)}"><div class="mh" onclick="this.nextElementSibling.classList.toggle('hid')">
      <span>${esc(m.role)} · #${j+1}</span><span>${c.length} 字</span></div>
      <div class="mb${hid}">${mdCode(c)}</div></div>`;
  }).join('');
  document.getElementById('main').innerHTML=
    `<div id="meta">
      ${cell('family',esc(d.family))}${cell('session',d.session_idx)}
      ${cell('finalized',d.finalized?'✓':'✗')}${cell('experiments',d.experiments_used)}
      ${cell('episodes',d.episodes_consumed)}${cell('turns',d.turns_used)}
      ${cell('doc 词数',(d.doc||'').split(/\s+/).filter(Boolean).length)}
      ${cell('error',d.error?esc(String(d.error)).slice(0,40):'—')}
    </div>
    <h2>最终文档</h2><div id="doc">${d.doc?mdCode(d.doc):'<span class="empty">无文档</span>'}</div>
    <h2>会话过程 · ${(d.messages||[]).length} 轮（点标题展开/折叠，system 默认折叠）</h2>${msgs}`;
  document.getElementById('main').scrollTop=0;
}
boot();
</script></body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True,
                    help="标签=路径,可多个,例如 9B=/path/sessions_9b 4B=/path/sessions_4b")
    ap.add_argument("--port", type=int, default=8801)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    for spec in a.sessions:
        label, _, path = spec.partition("=")
        SOURCES[label] = os.path.abspath(path or label)
    for k, v in SOURCES.items():
        n = len(glob.glob(os.path.join(v, "*", "s*.json")))
        print(f"[viewer] {k}: {v}  ({n} 会话)")
    print(f"[viewer] http://{a.host}:{a.port}")
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()
