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

设计:零第三方依赖(标准库 http.server);聚合结果按 JSON 文件名/大小/mtime
指纹缓存，同一缓存 key 以 single-flight 合并并发冷请求；完成态 step 只算一次，
运行中 step 变化才重算。tokenizer 常驻 + 单 token 解码缓存 ⇒ 热缓存单轨迹接口
约50ms。只读文件系统,不碰训练进程;绑 127.0.0.1。
"""
from __future__ import annotations

import argparse
import glob
import json
import re
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
CONFIG: dict = {}
APP = Path(__file__).with_name("pgs_microscope_app.html")
_STEP_DIR = re.compile(r"^step_[0-9]+(?:_val)?$")

_lock = threading.Lock()
_cache: dict[str, tuple[tuple, object]] = {}
_inflight: dict[str, threading.Event] = {}
_dump_cache: dict[str, tuple[int, dict]] = {}
_DUMP_CACHE_MAX = 6
_meta_cache: dict[str, tuple[tuple, dict]] = {}
_META_CACHE_MAX = 20000


def _files(d: Path) -> list[str]:
    return sorted(glob.glob(str(d / "*.json")))


def _fingerprint(fs: list[str]) -> tuple:
    rows = []
    for name in fs:
        try:
            stat = Path(name).stat()
            rows.append((name, stat.st_size, stat.st_mtime_ns))
        except FileNotFoundError:
            rows.append((name, -1, -1))
    return tuple(rows)


def _cached(key: str, d: Path, builder):
    fs = _files(d)
    fingerprint = _fingerprint(fs)
    while True:
        with _lock:
            hit = _cache.get(key)
            if hit and hit[0] == fingerprint:
                return hit[1]
            event = _inflight.get(key)
            if event is None:
                event = threading.Event()
                _inflight[key] = event
                break
        event.wait()

    try:
        out = builder(fs)
    except BaseException:
        with _lock:
            _inflight.pop(key, None)
            event.set()
        raise
    with _lock:
        _cache[key] = (fingerprint, out)
        _inflight.pop(key, None)
        event.set()
    return out


def _meta_cached(path: str) -> dict:
    """单文件元数据 memo(按 mtime+size 失效)。

    一条 rollout 写完就不再变,进行中的 step 只是追加新文件 —— 但整步聚合的
    指纹会因此改变,旧实现于是把全部文件重解析一遍。maxopen 下单文件约 2.9MB、
    一步 256 条,冷解析实测 7.5s,而盯在跑的 step 每次刷新都要重付。按文件 memo
    后,刷新只解析新到的那几条。
    """
    try:
        st = Path(path).stat()
        key = (st.st_mtime_ns, st.st_size)
    except OSError:
        key = (-1, -1)
    with _lock:
        hit = _meta_cache.get(path)
        if hit and hit[0] == key:
            return hit[1]
    with open(path) as fh:
        meta = _meta_of(json.load(fh))
    with _lock:
        _meta_cache[path] = (key, meta)
        while len(_meta_cache) > _META_CACHE_MAX:
            _meta_cache.pop(next(iter(_meta_cache)))
    return meta


def _meta_of(d: dict) -> dict:
    s = d.get("steering") or {}
    arm = s.get("steer_arm")
    # v3.3 吸收进度(dump 已整读,增量 O(拒绝数)):拒绝位上 π 的
    # draft−final logp 差 + final=π-top1(rank 0-based==0)计数。
    ab_dlogp, ab_top1, ab_miss16 = [], 0, 0
    tops = s.get("steer_pi_top16") or []
    rejs = s.get("steer_rejected") or []
    for i, isrej in enumerate(rejs):
        if not isrej or i >= len(tops) or not tops[i]:
            continue
        dr = s["steer_draft_token"][i]
        fi = s["steer_final_token"][i]
        lut = {int(t): float(lp) for t, lp in tops[i]}
        if dr in lut and fi in lut:
            ab_dlogp.append(M.r(lut[dr] - lut[fi], 3))
        else:
            ab_miss16 += 1
        if int(tops[i][0][0]) == fi:
            ab_top1 += 1
    fork = M.fork_hits(d.get("trajectory") or {})
    return dict(
        task=d.get("task_id", "?"),
        won=bool(d.get("won")),
        outcome=d.get("outcome", ""),
        split=d.get("split", ""),
        # 分臂采样(v3.1):None=非分臂 run(历史全臂);0=bare 臂;1=steered 臂
        arm=int(arm) if arm is not None else None,
        # v4 行类的**权威**判据(落盘顶层字段)。arm 只能间接推断且在 research
        # 行上取到 None,2026-08-09 的分母污染即源于此,不要退回按 arm 猜。
        role=d.get("sample_role"),
        research=d.get("research") or {},
        conts=d.get("continuations") or [],
        len=d.get("response_token_len", 0),
        m=dict(
            cand=s.get("steer_n_candidates", 0),
            open=s.get("steer_n_base_open", 0),
            tilt=s.get("steer_count", 0),
            rej=s.get("steer_reject_count", 0),
            spent=M.r(s.get("steer_budget_spent", 0), 2),
            bypass=s.get("steer_n_bypassed", 0),
            gbypass=s.get("steer_n_gate_bypassed", 0),
        ),
        theta=M.r(s.get("steer_theta_d", 0.0)),
        enttheta=M.r(s.get("steer_entropy_theta", 0.0), 3),
        entmean=M.r(s.get("steer_entropy_mean", 0.0), 3),
        wmean=M.r(s.get("steer_intensity_mean", 0.0), 3),
        absorb=dict(dlogp=ab_dlogp, top1=ab_top1, miss16=ab_miss16,
                    nrej=int(s.get("steer_reject_count", 0))),
        fork=fork,
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
                        j = _meta_cached(f)
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
                    # 聚合口径唯一定义在 M.build_step_entry(导出器同源);
                    # _meta_of 的轻量 dict 覆盖它消费的全部键。
                    metas = [_meta_cached(f) for f in fs]
                    return M.build_step_entry(
                        int(name.split("_")[1]), name, metas, CONFIG
                    )
                steps.append(_cached("step:" + d.name, d, sb))
    # 完成态必须由实收batch合同判定。旧逻辑用历史最大n推断full，首个partial
    # step会把自己的64条当成“历史最大”而误标完成。只有旧run缺配置时才回退历史值。
    expected_train = int(CONFIG.get("expectedTrainTrajs") or 0)
    full = expected_train or max((s["n"] for s in steps), default=0)
    for s in steps:
        s["partial"] = bool(full and s["n"] < full)
        s["expected"] = full
    vals = sorted(vals, key=lambda v: v["id"])
    vfull = max((v["n"] for v in vals), default=0)
    for v in vals:
        v["partial"] = bool(vfull and v["n"] < vfull)
    return dict(
        name=RUN.name,
        live=True,
        hasDocs=bool(DOCS),
        docs={"staticCount": len(DOCS), "dynamicArchives": (RUN / "research").is_dir()},
        config=CONFIG,
        ts=time.time(),
        steps=steps,
        vals=vals,
    )


def _step_dir(dirname: str) -> Path | None:
    if not _STEP_DIR.fullmatch(dirname):
        return None
    path = RUN / "rollouts" / dirname
    return path if path.is_dir() else None


def api_step(dirname: str) -> dict:
    d = _step_dir(dirname)
    if d is None:
        return {"error": "no such step"}

    def lb(fs):
        return {"trajs": [_meta_cached(f) for f in fs]}
    return _cached("list:" + dirname, d, lb)


def _traj_file(dirname: str, idx: int) -> str | None:
    d = _step_dir(dirname)
    if d is None:
        return None
    fs = _files(d)
    return fs[idx] if 0 <= idx < len(fs) else None


def _load_dump(path: str) -> dict:
    """原始 dump 的小 LRU。maxopen 下单文件可达 8MB,解析成本高;抽屉按需取
    分布时会反复命中同一条轨迹,不缓存则每次重新读盘+反序列化。"""
    with _lock:
        hit = _dump_cache.get(path)
    try:
        mtime = Path(path).stat().st_mtime_ns
    except OSError:
        mtime = -1
    if hit and hit[0] == mtime:
        return hit[1]
    with open(path) as fh:
        raw = json.load(fh)
    with _lock:
        _dump_cache[path] = (mtime, raw)
        while len(_dump_cache) > _DUMP_CACHE_MAX:
            _dump_cache.pop(next(iter(_dump_cache)))
    return raw


def api_traj(dirname: str, idx: int) -> dict:
    path = _traj_file(dirname, idx)
    if path is None:
        return {"error": "no such step or index out of range"}
    # 实时后端走精简档:省掉占 payload 95% 的 top16 分布,抽屉再按需取。
    return M.build_traj(
        _load_dump(path),
        DEC,
        DOCS,
        run=RUN,
        bmax=CONFIG["bmax"],
        with_dists=False,
    )


def api_rec(dirname: str, idx: int, ref: int) -> dict:
    path = _traj_file(dirname, idx)
    if path is None:
        return {"error": "no such step or index out of range"}
    return M.build_rec_dists(_load_dump(path), DEC, ref)


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
            if p == "data/run.js":
                return self._send(200, b"", "application/javascript; charset=utf-8")
            if p == "favicon.ico":
                return self._send(204, b"", "image/x-icon")
            if p == "api/run":
                return self._json(api_run())
            if p.startswith("api/step/"):
                return self._json(api_step(p.split("/")[2]))
            if p.startswith("api/traj/"):
                _, _, d, i = p.split("/")
                return self._json(api_traj(d, int(i)))
            if p.startswith("api/rec/"):
                _, _, d, i, ref = p.split("/")
                return self._json(api_rec(d, int(i), int(ref)))
            return self._send(404, b"not found", "text/plain")
        except BrokenPipeError:
            pass
        except Exception as e:  # noqa: BLE001  单请求失败不倒服务
            self._send(500, f"error: {e}".encode(), "text/plain; charset=utf-8")


def main() -> int:
    global RUN, DEC, DOCS, CONFIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--doc-table", default=None, help="task_id→特权文档 jsonl(s1 臂)")
    ap.add_argument("--fork-lexicon", default=None, help="fork 词表 json 覆盖(与导出器同格式)")
    a = ap.parse_args()
    RUN = Path(a.run_dir)
    CONFIG = M.load_run_config(RUN)
    if a.fork_lexicon:
        M.set_fork_lexicon(json.load(open(a.fork_lexicon)))
    print(f"[server] loading tokenizer {a.tokenizer} …", flush=True)
    DEC = M.Dec(M.load_tok(a.tokenizer))
    DOCS = M.load_docs(a.doc_table)
    print(f"[server] doc table: {len(DOCS)} 条", flush=True)
    print(
        f"[server] budget B∞={CONFIG['bmax']} ({CONFIG['bmaxSource']})"
        + (f" WARNING: {CONFIG['bmaxWarning']}" if CONFIG["bmaxWarning"] else ""),
        flush=True,
    )
    srv = ThreadingHTTPServer((a.host, a.port), H)
    print(f"[server] run={RUN}  http://{a.host}:{a.port}  (本地: ssh -N -L {a.port}:127.0.0.1:{a.port} …)", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
