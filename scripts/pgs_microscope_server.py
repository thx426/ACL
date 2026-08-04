#!/usr/bin/env python3
"""PGS 显微镜实时后端 —— 驻训练机,实时读 run 目录;本地经 ssh -L 端口转发访问。

    python scripts/pgs_microscope_server.py --run-dir <run> --tokenizer <hf-path>
           [--port 8791] [--host 127.0.0.1]

本地访问(笔记本上):
    ssh -N -L 8791:127.0.0.1:8791 -p <port> <user>@<host> &
    open http://localhost:8791

API(前端 pgs_microscope_app.html 自动探测,探测失败即退回静态 data/*.js 模式):
    GET /                     前端应用
    GET /api/run              run 聚合:step 机制曲线 + val 曲线(实时,含进行中 step)
    GET /api/step/<dir>       该 step 的轨迹列表元数据(轻量,不含逐 token)
    GET /api/traj/<dir>/<i>   单条轨迹逐 token 全量(与静态导出同 schema,i=sorted 文件序)

设计:零第三方依赖(标准库 http.server);聚合结果按 (目录, 文件数) 缓存,
完成态 step 只算一次,运行中 step 文件数变化才重算;tokenizer 常驻 + 单 token
解码缓存 ⇒ 单轨迹接口 ~50ms。只读文件系统,不碰训练进程;绑 127.0.0.1。
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pgs_microscope as M  # noqa: E402  复用 Dec / build_traj / r

RUN: Path = None
DEC: "M.Dec" = None
DOCS: dict = {}
APP = Path(__file__).with_name("pgs_microscope_app.html")

_lock = threading.Lock()
_cache: dict[str, tuple[int, object]] = {}   # key -> (n_files, payload)


def _files(d: Path) -> list[str]:
    return sorted(glob.glob(str(d / "*.json")))


def _cached(key: str, d: Path, builder):
    fs = _files(d)
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] == len(fs):
            return hit[1]
    out = builder(fs)
    with _lock:
        _cache[key] = (len(fs), out)
    return out


def _meta_of(d: dict) -> dict:
    s = d.get("steering") or {}
    return dict(
        task=d.get("task_id", "?"),
        won=bool(d.get("won")),
        outcome=d.get("outcome", ""),
        split=d.get("split", ""),
        len=d.get("response_token_len", 0),
        m=dict(
            cand=s.get("steer_n_candidates", 0),
            open=s.get("steer_n_base_open", 0),
            tilt=s.get("steer_count", 0),
            rej=s.get("steer_reject_count", 0),
            spent=M.r(s.get("steer_budget_spent", 0), 2),
            bypass=s.get("steer_n_bypassed", 0),
        ),
        theta=M.r(s.get("steer_theta_d", 0.0)),
    )


def api_run() -> dict:
    roll = RUN / "rollouts"
    steps, vals = [], []
    if roll.is_dir():
        for d in sorted(roll.iterdir()):
            if not d.name.startswith("step_"):
                continue
            if d.name.endswith("_val"):
                def vb(fs, name=d.name):
                    tot = dict(n=0, won=0, seen_n=0, seen_w=0, un_n=0, un_w=0)
                    for f in fs:
                        j = json.load(open(f))
                        tot["n"] += 1
                        tot["won"] += bool(j.get("won"))
                        if "unseen" in (j.get("split") or ""):
                            tot["un_n"] += 1
                            tot["un_w"] += bool(j.get("won"))
                        else:
                            tot["seen_n"] += 1
                            tot["seen_w"] += bool(j.get("won"))
                    sid = int(name.split("_")[1])
                    return dict(
                        id=sid, n=tot["n"],
                        won=M.r(tot["won"] / max(tot["n"], 1), 4),
                        seen=M.r(tot["seen_w"] / max(tot["seen_n"], 1), 4),
                        unseen=M.r(tot["un_w"] / max(tot["un_n"], 1), 4),
                    )
                vals.append(_cached("val:" + d.name, d, vb))
            else:
                def sb(fs, name=d.name):
                    agg = dict(won=0, cand=0, open=0, tilt=0.0, rej=0.0, spent=0.0, theta=0.0, bypass=0.0)
                    for f in fs:
                        t = _meta_of(json.load(open(f)))
                        agg["won"] += t["won"]
                        agg["cand"] += t["m"]["cand"]
                        agg["open"] += t["m"]["open"]
                        agg["tilt"] += t["m"]["tilt"]
                        agg["rej"] += t["m"]["rej"]
                        agg["spent"] += t["m"]["spent"]
                        agg["theta"] += t["theta"]
                        agg["bypass"] += t["m"]["bypass"]
                    n = max(len(fs), 1)
                    return dict(
                        id=int(name.split("_")[1]), dir=name, n=len(fs),
                        won=M.r(agg["won"] / n, 4),
                        gate=M.r(agg["open"] / max(agg["cand"], 1), 4),
                        tilt=M.r(agg["tilt"] / n, 2), rej=M.r(agg["rej"] / n, 2),
                        rejrate=M.r(agg["rej"] / max(agg["open"], 1), 4),
                        spent=M.r(agg["spent"] / n, 2), theta=M.r(agg["theta"] / n, 4),
                        bypass=M.r(agg["bypass"] / n, 1),
                    )
                steps.append(_cached("step:" + d.name, d, sb))
    # 进行中标注:最后一个 train step 若文件数 < batch(以历史最大为准)
    full = max((s["n"] for s in steps), default=0)
    for s in steps:
        s["partial"] = bool(full and s["n"] < full)
    return dict(name=RUN.name, live=True, hasDocs=bool(DOCS), ts=time.time(), steps=steps, vals=sorted(vals, key=lambda v: v["id"]))


def api_step(dirname: str) -> dict:
    d = RUN / "rollouts" / dirname
    if not d.is_dir():
        return {"error": "no such step"}

    def lb(fs):
        return {"trajs": [_meta_of(json.load(open(f))) for f in fs]}
    return _cached("list:" + dirname, d, lb)


def api_traj(dirname: str, idx: int) -> dict:
    fs = _files(RUN / "rollouts" / dirname)
    if not (0 <= idx < len(fs)):
        return {"error": "index out of range"}
    return M.build_traj(json.load(open(fs[idx])), DEC, DOCS)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 静默访问日志
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        p = self.path.split("?")[0].strip("/")
        try:
            if p in ("", "index.html"):
                return self._send(200, APP.read_bytes(), "text/html; charset=utf-8")
            if p == "api/run":
                return self._json(api_run())
            if p.startswith("api/step/"):
                return self._json(api_step(p.split("/")[2]))
            if p.startswith("api/traj/"):
                _, _, d, i = p.split("/")
                return self._json(api_traj(d, int(i)))
            return self._send(404, b"not found", "text/plain")
        except BrokenPipeError:
            pass
        except Exception as e:  # noqa: BLE001  单请求失败不倒服务
            self._send(500, f"error: {e}".encode(), "text/plain; charset=utf-8")


def main() -> int:
    global RUN, DEC, DOCS
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--doc-table", default=None, help="task_id→特权文档 jsonl(s1 臂)")
    a = ap.parse_args()
    RUN = Path(a.run_dir)
    print(f"[server] loading tokenizer {a.tokenizer} …", flush=True)
    DEC = M.Dec(M.load_tok(a.tokenizer))
    DOCS = M.load_docs(a.doc_table)
    print(f"[server] doc table: {len(DOCS)} 条", flush=True)
    srv = ThreadingHTTPServer((a.host, a.port), H)
    print(f"[server] run={RUN}  http://{a.host}:{a.port}  (本地: ssh -N -L {a.port}:127.0.0.1:{a.port} …)", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
