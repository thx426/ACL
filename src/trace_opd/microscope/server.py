"""Distillation-microscope workbench server — stdlib http + numpy only.

Serves the observation packs behind a small JSON API and a single-page
front-end (static/). All derived quantities are computed on demand from the
raw packs (derive.py) and memoised in process — the caches ARE disposable,
restart recomputes them, and a new analysis view is a code change plus a
restart, never a resample.

USABILITY CONTRACT (user ruling 2026-07-29): the workbench must never leave
the user staring at a silent page. Concretely:
  * the detail view is SPLIT into three endpoints — /api/sample (skeleton,
    instant once the pack is resident), /api/sample_tokens (batch-decoded
    text), /api/sample_tracks (the heavy per-token derivation, cached per
    row) — so the front-end renders progressively instead of all-or-nothing;
  * pack loading is single-flight (a per-step lock) and OBSERVABLE:
    /api/status reports what is being decompressed and for how long, and the
    front-end shows it;
  * /api/position reuses the cached tracks — clicking a token is instant
    after the first tracks fetch;
  * requests slower than 1 s are logged server-side with their duration.

Run on the training server, browse through an SSH tunnel:

  PYTHONPATH=src python3 -m trace_opd.microscope serve \\
      --obs-dir /abs/run/obs --tokenizer <model_path> [--port 8643]
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import derive, segments
from .reader import StepPack, list_steps

_STATIC = Path(__file__).parent / "static"
_TOKENIZER = None
_OBS_DIR: str = ""
_MEMO: dict = {}

# ------------------------------------------------- pack + tracks residency
_PACKS: "OrderedDict[int, StepPack]" = OrderedDict()
_PACK_CAP = 8
_LOAD_LOCKS: dict[int, threading.Lock] = {}
_LOADING: dict[int, float] = {}  # step -> load start ts (for /api/status)
_STATE_LOCK = threading.Lock()

_TRACKS: "OrderedDict[tuple, dict]" = OrderedDict()
_TRACKS_CAP = 64
_TRACKS_LOCK = threading.Lock()
_SAMPLES_LOCKS: dict[int, threading.Lock] = {}
# Background preload progress (user ruling 2026-07-29: the listing must warm
# itself — nobody should have to CLICK a step to trigger its first, slow
# computation). Reported via /api/status, shown in the front-end meta line.
_WARM = {"active": False, "total": 0, "done": 0, "current": None}

_METRICS = (
    "kl_forward", "kl_reverse", "token_loss", "gap", "teacher_entropy",
    "student_entropy", "overlap_count", "teacher_mass", "student_mass_own",
)


def _pack(step: int) -> StepPack:
    """Resident pack with single-flight loading and status registration —
    two concurrent requests for a cold step must not decompress it twice,
    and the user must be able to SEE that a 300 MB decompress is running."""
    with _STATE_LOCK:
        if step in _PACKS:
            _PACKS.move_to_end(step)
            return _PACKS[step]
        lock = _LOAD_LOCKS.setdefault(step, threading.Lock())
    with lock:
        with _STATE_LOCK:
            if step in _PACKS:
                _PACKS.move_to_end(step)
                return _PACKS[step]
            _LOADING[step] = time.time()
        try:
            pack = StepPack(Path(_OBS_DIR), step)
        finally:
            with _STATE_LOCK:
                _LOADING.pop(step, None)
        with _STATE_LOCK:
            _PACKS[step] = pack
            while len(_PACKS) > _PACK_CAP:
                _PACKS.popitem(last=False)
        return pack


def _tracks_of(step: int, row: dict) -> dict:
    """Per-row derived tracks: memory LRU -> persistent npz cache -> compute.
    The heavy derivation happens ONCE per row EVER (survives restarts); every
    colouring switch / minimap / position click after that is a lookup."""
    key = (step, row["resp_hash"], len(row["response_ids"]))
    with _TRACKS_LOCK:
        if key in _TRACKS:
            _TRACKS.move_to_end(key)
            return _TRACKS[key]
    sig = _pack_sig(step)
    cache = _cache_dir(step) / f"tracks_{key[1]}_{key[2]}.npz"
    tr = None
    if cache.exists():
        try:
            with np.load(cache) as z:
                if str(z["sig"][0]) == sig:
                    tr = {k: z[k] for k in z.files if k != "sig"}
        except Exception:  # noqa: BLE001 — cache corruption -> recompute
            tr = None
    if tr is None:
        pack = _pack(step)
        tr = derive.tracks(row, pack.log_prob_min_clamp)
        try:
            np.savez_compressed(
                cache, sig=np.array([sig]),
                **{k: v.astype(np.float32) for k, v in tr.items()},
            )
        except OSError:
            pass
    with _TRACKS_LOCK:
        _TRACKS[key] = tr
        while len(_TRACKS) > _TRACKS_CAP:
            _TRACKS.popitem(last=False)
    return tr


def _pack_sig(step: int) -> str:
    """Fingerprint of the step's SOURCE npz files — persistent derived caches
    carry it and are rebuilt whenever the truth changes. The packs stay the
    single source; every cache is disposable."""
    import hashlib

    parts = []
    for f in sorted((Path(_OBS_DIR) / f"step_{step:05d}").glob("rank*.npz")):
        st = f.stat()
        parts.append(f"{f.name}:{st.st_size}:{st.st_mtime_ns}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _cache_dir(step: int) -> Path:
    d = Path(_OBS_DIR) / f"step_{step:05d}" / "_cache"
    d.mkdir(exist_ok=True)
    return d


@lru_cache(maxsize=1)
def _step_sizes_mb() -> dict:
    sizes = {}
    for step in list_steps(Path(_OBS_DIR)):
        total = sum(
            p.stat().st_size
            for p in (Path(_OBS_DIR) / f"step_{step:05d}").glob("rank*.npz")
        )
        sizes[str(step)] = round(total / 1e6, 1)
    return sizes


def _tok_text(token_id: int) -> str:
    if _TOKENIZER is None:
        return f"<{token_id}>"
    piece = _TOKENIZER.convert_ids_to_tokens([int(token_id)])[0]
    return piece.replace("Ġ", " ").replace("Ċ", "\\n").replace("▁", " ")


def _tok_texts(ids) -> list[str]:
    """Batch display pieces — ONE tokenizer call for the whole row (the old
    per-token loop made 4000+ single-id calls per detail view)."""
    if _TOKENIZER is None:
        return [f"<{int(t)}>" for t in ids]
    toks = _TOKENIZER.convert_ids_to_tokens([int(t) for t in ids])
    return [t.replace("Ġ", " ").replace("Ċ", "\\n").replace("▁", " ") for t in toks]


def _tok_pieces(ids) -> list[str]:
    """Batch piece stream with REAL whitespace — the segment scanner's input
    (unlike _tok_texts, which escapes newlines for display)."""
    toks = _TOKENIZER.convert_ids_to_tokens([int(t) for t in ids])
    return [t.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ") for t in toks]


def _clean(value):
    if isinstance(value, np.ndarray):
        return [None if not np.isfinite(v) else round(float(v), 5) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else round(float(value), 5)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _summary_row(row: dict) -> dict:
    """Listing row: derive's summary plus what the skeleton fast path needs.
    resp_hash is serialised as a STRING — it is a 64-bit hash, and JSON
    numbers above 2^53 lose precision the moment a browser parses them
    (CD-006 req 3-AD: every hash round-tripped through JS came back wrong,
    so the detail view could never open)."""
    return {
        **derive.sample_summary(row),
        "resp_hash": str(row["resp_hash"]),
        "turn_spans": [list(x) for x in row["turn_spans"]],
    }


def _samples_payload_cached(step: int):
    """The summary payload if it is ALREADY available (memo or valid disk
    cache) — never triggers a pack decompress. The skeleton fast path."""
    key = ("samples", step)
    if key in _MEMO:
        return _MEMO[key]
    cache = _cache_dir(step) / "summary.json"
    if cache.exists():
        try:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            if payload.get("sig") == _pack_sig(step) and payload.get("v") == 2:
                _MEMO[key] = payload
                return payload
        except Exception:  # noqa: BLE001
            pass
    return None


def _samples(step: int) -> list[dict]:
    """Listing rows, with a PERSISTENT summary cache: once computed, the
    samples page opens in milliseconds — across restarts — without touching
    the 300 MB pack at all."""
    payload = _samples_payload_cached(step)
    if payload is not None:
        return payload["rows"]
    lock = _SAMPLES_LOCKS.setdefault(step, threading.Lock())
    with lock:  # single-flight: a double-fired listing must not compute twice
        payload = _samples_payload_cached(step)
        if payload is not None:
            return payload["rows"]
        pack = _pack(step)
        payload = {
            "sig": _pack_sig(step), "v": 2, "loss_mode": pack.loss_mode,
            "rows": [_summary_row(row) for row in pack.rows()],
        }
        try:
            (_cache_dir(step) / "summary.json").write_text(
                json.dumps(payload), encoding="utf-8")
        except OSError:
            pass  # read-only obs dir: in-memory memo still holds
        _MEMO[("samples", step)] = payload
        return payload["rows"]


# ----------------------------------------------- detail view, split three ways

def _sample_skeleton(step: int, resp_hash: int) -> dict:
    """The detail view's FRAME. Fast path (CD-006 req 3-AC): if the step's
    summary cache exists, the skeleton is served from it WITHOUT touching the
    pack — instant even on a cold 350 MB step the listing has visited.
    audit/segments ride the first tracks chunk instead (they need the pack
    anyway)."""
    payload = _samples_payload_cached(step)
    if payload is not None:
        want = str(resp_hash)
        for r in payload["rows"]:
            if r["resp_hash"] == want:
                return {
                    "summary": r,
                    "loss_mode": payload.get("loss_mode", ""),
                    "turn_spans": r.get("turn_spans", []),
                }
    pack = _pack(step)
    row = pack.row(resp_hash)
    summary = _summary_row(row)
    return {
        "summary": summary,
        "loss_mode": pack.loss_mode,
        "turn_spans": summary["turn_spans"],
    }


def _window(n: int, q_lo, q_hi) -> tuple[int, int]:
    lo = max(0, int(q_lo)) if q_lo is not None else 0
    hi = min(n, int(q_hi)) if q_hi is not None else n
    return lo, max(lo, hi)


def _sample_tokens(step: int, resp_hash: int, q_lo=None, q_hi=None) -> dict:
    """Windowed token text — the front-end streams [lo,hi) chunks and renders
    each as it lands, so a 30k-token row appears progressively."""
    pack = _pack(step)
    row = pack.row(resp_hash)
    lo, hi = _window(len(row["response_ids"]), q_lo, q_hi)
    ids = row["response_ids"][lo:hi]
    return {
        "lo": lo, "hi": hi, "n": int(len(row["response_ids"])),
        "tokens": [
            {"id": int(t), "text": s} for t, s in zip(ids, _tok_texts(ids))
        ],
    }


def _sample_tracks(step: int, resp_hash: int, q_lo=None, q_hi=None) -> dict:
    """Windowed track values (full-row derivation is cached persistently; the
    window only slices the response)."""
    pack = _pack(step)
    row = pack.row(resp_hash)
    tr = _tracks_of(step, row)
    lo, hi = _window(len(row["response_ids"]), q_lo, q_hi)
    repeats = derive.repeat_positions(row)
    out = {
        "lo": lo, "hi": hi, "n": int(len(row["response_ids"])),
        "tracks": {k: _clean(v[lo:hi]) for k, v in tr.items()},
        "repeat_mask": [int(x) for x in repeats[lo:hi].tolist()],
    }
    if lo == 0:  # frame extras ride the first chunk (they need the pack anyway)
        out["audit"] = derive.audit_row(
            row, loss_mode=pack.loss_mode, clamp=pack.log_prob_min_clamp)
        out["segments"] = (
            segments.segment_spans(_tok_pieces(row["response_ids"]))
            if _TOKENIZER is not None else None
        )
    return out


def _position_payload(step: int, resp_hash: int, pos: int) -> dict:
    pack = _pack(step)
    row = pack.row(resp_hash)
    tr = _tracks_of(step, row)  # cached — a click costs one dict lookup

    def side(ids_key: str, lps_key: str) -> list[dict]:
        return [
            {"id": int(i), "text": _tok_text(i), "lp": _clean(float(lp)),
             "p": _clean(float(np.exp(max(lp, -30.0))))}
            for i, lp in zip(row[ids_key][pos], row[lps_key][pos])
        ]

    if not (0 <= pos < len(row["response_ids"])):
        raise KeyError(f"pos {pos} out of range 0..{len(row['response_ids']) - 1}")
    realized_id = int(row["response_ids"][pos])
    lo, hi = max(0, pos - 30), min(len(row["response_ids"]), pos + 31)
    return {
        "pos": pos,
        "label": derive.divergence_label(row, pos, tr=tr),
        "teacher": side("teacher_ids", "teacher_lps"),
        "student": side("student_ids", "student_lps"),
        "student_on_support": _clean(row["student_support_lps"][pos]),
        "realized": {
            "id": realized_id,
            "text": _tok_text(realized_id),
            "student_lp": _clean(float(row["realized_student_lp"][pos])),
            "teacher_lp": _clean(float(tr["realized_teacher_lp"][pos])),
            "student_rank": int(tr["student_rank"][pos]),
            "teacher_rank": int(tr["teacher_rank"][pos]),
        },
        "scalars": {k: _clean(float(tr[k][pos])) for k in _METRICS},
        "context": {
            "start": lo,
            "tokens": _tok_texts(row["response_ids"][lo:hi]),
        },
        "covered": bool(row["covered"][pos]),
        "masked": bool(row["response_mask"][pos]),
    }


def _leaderboard(metric: str, step_lo: int, step_hi: int, mode: str, limit: int) -> list[dict]:
    key = ("lb", metric, step_lo, step_hi, mode)
    if key in _MEMO:
        return _MEMO[key][:limit]
    entries: list[dict] = []
    for step in list_steps(Path(_OBS_DIR)):
        if not (step_lo <= step <= step_hi):
            continue
        pack = _pack(step)
        for row in pack.rows():
            tr = derive.tracks(row, pack.log_prob_min_clamp)
            m = row["response_mask"] & row["covered"]
            if mode == "verdict_conflict":
                cand = m & (row["teacher_ids"][:, 0] != row["student_ids"][:, 0]) \
                    & (tr["teacher_entropy"] < 1.0) & (tr["student_entropy"] < 1.0)
                values = np.where(cand, tr["kl_forward"], -np.inf)
            elif mode == "repeat":
                cand = m & derive.repeat_positions(row)
                values = np.where(cand, tr["kl_forward"], -np.inf)
            elif mode == "teacher_abstains":
                cand = m & (tr["teacher_entropy"] > 3.0)
                values = np.where(cand, tr["teacher_entropy"], -np.inf)
            else:
                values = np.where(m, np.nan_to_num(tr[metric], nan=-np.inf), -np.inf)
            if not np.isfinite(values).any():
                continue
            top = np.argsort(values)[::-1][:8]
            for pos in top:
                if not np.isfinite(values[pos]):
                    continue
                snippet = "".join(
                    _tok_text(t) for t in row["response_ids"][max(0, pos - 6) : pos + 7]
                )
                entries.append(
                    {
                        "step": step, "hash": str(row["resp_hash"]), "task": row["task_id"],
                        "won": row["won"], "pos": int(pos),
                        "value": _clean(float(values[pos])), "snippet": snippet[:120],
                    }
                )
    entries.sort(key=lambda e: -(e["value"] if e["value"] is not None else -1e30))
    _MEMO[key] = entries
    return entries[:limit]


def _evolution(task: str, metric: str) -> dict:
    key = ("evo", task, metric)
    if key in _MEMO:
        return _MEMO[key]
    acc: dict[tuple, list] = {}
    reps: dict[tuple, list] = {}
    for step in list_steps(Path(_OBS_DIR)):
        pack = _pack(step)
        for row in pack.rows():
            if row["task_id"] != task or row["sample_role"] != "bare":
                continue
            tr = derive.tracks(row, pack.log_prob_min_clamp)
            for turn_index, (a, b) in enumerate(row["turn_spans"]):
                m = row["response_mask"][a:b] & row["covered"][a:b]
                vals = tr[metric][a:b][m]
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    cell_key = (step, turn_index)
                    acc.setdefault(cell_key, []).append(float(vals.mean()))
                    reps.setdefault(cell_key, []).append(row["resp_hash"])
    # codex review P1-18: rollout_n replicas per task are the NORM — the cell
    # is the replica MEAN with n attached, never a last-writer-wins sample.
    cells = [
        {"step": step, "turn": turn, "value": _clean(float(np.mean(vals))),
         "n": len(vals), "hash": str(reps[(step, turn)][0])}
        for (step, turn), vals in sorted(acc.items())
    ]
    out = {"task": task, "metric": metric, "cells": cells}
    _MEMO[key] = out
    return out


def _summarize_group(items: list[dict]) -> dict:
    """Row-equal-weight means (never token-pooled: a single inflating row
    must not dominate the step statistic through sheer length)."""
    out: dict = {"n": len(items)}
    if not items:
        return out
    out["resp_len_mean"] = float(np.mean([it["resp_len"] for it in items]))
    out["repeat_rate_mean"] = float(np.mean([it["repeat_rate"] for it in items]))
    for kind in segments.TAGS:
        agg: dict = {}
        toks = [it["agg"][kind]["tokens"] for it in items]
        agg["tokens_mean"] = float(np.mean(toks))
        for metric in ("kl_mean", "kl_share", "student_entropy", "teacher_entropy",
                       "teacher_mass", "token_loss", "clamp_hit_rate"):
            vals = [
                it["agg"][kind][metric]
                for it in items
                if it["agg"][kind].get(metric) is not None
            ]
            agg[metric] = float(np.mean(vals)) if vals else None
        out[kind] = agg
    closings = [c for it in items for c in it["closing"]]
    t_lps = [c["teacher_lp"] for c in closings if c["teacher_lp"] is not None]
    s_lps = [c["student_lp"] for c in closings]
    out["closing"] = {
        "n": len(closings),
        "teacher_outside_topk": len(closings) - len(t_lps),
        "teacher_lp_mean": float(np.mean(t_lps)) if t_lps else None,
        "student_lp_mean": float(np.mean(s_lps)) if s_lps else None,
    }
    return out


def _segment_report(step_lo: int, step_hi: int, role: str) -> dict:
    """Protocol items 1/3/4/6/7 in one sweep: per-step, won/lost-stratified,
    per-segment KL budget, entropies, clamp/mass artefacts, closing-token
    probabilities, lengths and repeat rates."""
    key = ("segrep", step_lo, step_hi, role)
    if key in _MEMO:
        return _MEMO[key]
    if _TOKENIZER is None:
        raise KeyError("segment_report requires --tokenizer (tag scan is textual)")
    records = []
    for step in list_steps(Path(_OBS_DIR)):
        if not (step_lo <= step <= step_hi):
            continue
        pack = _pack(step)
        groups: dict[str, list] = {"won": [], "lost": []}
        for row in pack.rows():
            if role and row["sample_role"] != role:
                continue
            seg = segments.segment_spans(_tok_pieces(row["response_ids"]))
            tr = derive.tracks(row, pack.log_prob_min_clamp)
            m = row["response_mask"].astype(bool) & row["covered"].astype(bool)
            rep = derive.repeat_positions(row)
            groups["won" if row["won"] else "lost"].append(
                {
                    "resp_len": int(len(row["response_ids"])),
                    "repeat_rate": float(rep[m].mean()) if m.any() else 0.0,
                    "agg": segments.aggregate_segments(
                        row, tr, seg, clamp=pack.log_prob_min_clamp
                    ),
                    "closing": segments.closing_token_stats(row, tr, seg),
                }
            )
        records.append(
            {
                "step": step,
                "won": _summarize_group(groups["won"]),
                "lost": _summarize_group(groups["lost"]),
                "all": _summarize_group(groups["won"] + groups["lost"]),
            }
        )
    out = {"role": role, "step_lo": step_lo, "step_hi": step_hi, "records": records}
    _MEMO[key] = out
    return out


def _tasks() -> list[str]:
    key = ("tasks",)
    if key in _MEMO:
        return _MEMO[key]
    tasks = set()
    for step in list_steps(Path(_OBS_DIR)):
        for path in (Path(_OBS_DIR) / f"step_{step:05d}").glob("rank*.npz"):
            with np.load(path) as z:  # single-key read: cheap even at scale
                tasks.update(str(t) for t in z["row_task"])
    out = sorted(tasks)
    _MEMO[key] = out
    return out


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.1 keep-alive: the front-end polls /api/status every second while
    # anything is loading — under HTTP/1.0 every poll opened a NEW connection,
    # and with the default listen backlog of 5 a busy backend started
    # REJECTING fresh connections (the skeleton request died at the network
    # layer while long requests sat in flight). Persistent connections plus a
    # deep backlog (see _Server) remove that whole failure class.
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # quiet (slow requests are logged in do_GET)
        pass

    def _json(self, payload, code=200):
        try:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client aborted (view switch) — routine, not an error

    def do_GET(self):  # noqa: N802
        t0 = time.time()
        try:
            self._route()
        finally:
            dt = time.time() - t0
            if dt > 1.0:  # slow requests must be visible to the operator
                print(f"[microscope] {self.path} took {dt:.1f}s", flush=True)

    def _route(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/meta":
                # Deliberately CHEAP (no task scan): the front-end must boot
                # instantly even on a cold 5 GB obs dir.
                return self._json(
                    {
                        "obs_dir": _OBS_DIR,
                        "steps": list_steps(Path(_OBS_DIR)),
                        "metrics": list(_METRICS),
                        "tokenizer": _TOKENIZER is not None,
                        "sizes_mb": _step_sizes_mb(),
                    }
                )
            if url.path == "/api/status":
                with _STATE_LOCK:
                    loading = {
                        str(s): round(time.time() - ts, 1)
                        for s, ts in _LOADING.items()
                    }
                    loaded = sorted(_PACKS)
                    warm = dict(_WARM)
                return self._json({"loading": loading, "loaded": loaded, "warm": warm})
            if url.path == "/api/tasks":
                return self._json(_tasks())
            if url.path == "/api/samples":
                return self._json(_samples(int(q["step"])))
            if url.path == "/api/sample":
                return self._json(_sample_skeleton(int(q["step"]), int(q["hash"])))
            if url.path == "/api/sample_tokens":
                return self._json(_sample_tokens(
                    int(q["step"]), int(q["hash"]), q.get("lo"), q.get("hi")))
            if url.path == "/api/sample_tracks":
                return self._json(_sample_tracks(
                    int(q["step"]), int(q["hash"]), q.get("lo"), q.get("hi")))
            if url.path == "/api/position":
                return self._json(
                    _position_payload(int(q["step"]), int(q["hash"]), int(q["pos"]))
                )
            if url.path == "/api/leaderboard":
                steps = list_steps(Path(_OBS_DIR)) or [0]
                return self._json(
                    _leaderboard(
                        q.get("metric", "kl_forward"),
                        int(q.get("step_lo", steps[0])),
                        int(q.get("step_hi", steps[-1])),
                        q.get("mode", "metric"),
                        int(q.get("limit", 100)),
                    )
                )
            if url.path == "/api/evolution":
                return self._json(_evolution(q["task"], q.get("metric", "kl_forward")))
            if url.path == "/api/segment_report":
                steps = list_steps(Path(_OBS_DIR)) or [0]
                return self._json(
                    _segment_report(
                        int(q.get("step_lo", steps[0])),
                        int(q.get("step_hi", steps[-1])),
                        q.get("role", "bare"),
                    )
                )
            # static
            rel = "index.html" if url.path in ("/", "") else url.path.lstrip("/")
            rel = re.sub(r"[^A-Za-z0-9._/-]", "", rel)
            path = (_STATIC / rel).resolve()
            if _STATIC.resolve() in path.parents or path == (_STATIC / "index.html").resolve():
                if path.exists():
                    ctype = {
                        ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
                    }.get(path.suffix, "application/octet-stream")
                    body = path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", f"{ctype}; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
            self._json({"error": "not found"}, 404)
        except KeyError as err:
            self._json({"error": f"missing/unknown: {err}"}, 400)
        except Exception as err:  # noqa: BLE001
            self._json({"error": f"{type(err).__name__}: {err}"}, 500)


def _preload(workers: int) -> None:
    """Warm every step's summary cache in the background, NEWEST FIRST (the
    steps a user opens first), and keep the newest packs resident so the
    first detail click skips the decompress too. User requests that land on
    a step mid-warm share the same single-flight computation — never a
    duplicate. Failures skip the step and keep warming."""
    from concurrent.futures import ThreadPoolExecutor

    steps = sorted(list_steps(Path(_OBS_DIR)), reverse=True)
    with _STATE_LOCK:
        _WARM.update(active=True, total=len(steps), done=0, current=None)

    def warm(step: int) -> None:
        with _STATE_LOCK:
            _WARM["current"] = step
        try:
            _samples(step)  # builds summary.json if missing (may decompress)
            if steps.index(step) < _PACK_CAP:
                _pack(step)  # newest packs stay resident for detail clicks
        except Exception as err:  # noqa: BLE001 — warm what can be warmed
            print(f"[microscope] preload step {step} failed: {err}", flush=True)
        finally:
            with _STATE_LOCK:
                _WARM["done"] += 1

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(warm, steps))
    with _STATE_LOCK:
        _WARM.update(active=False, current=None)
    print(f"[microscope] preload complete: {len(steps)} steps warm", flush=True)


def serve(obs_dir: str, tokenizer_path: str | None, host: str, port: int,
          preload_workers: int = 2) -> None:
    global _OBS_DIR, _TOKENIZER
    _OBS_DIR = str(obs_dir)
    if tokenizer_path:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path)
    steps = list_steps(Path(obs_dir))
    print(f"[microscope] obs={obs_dir} steps={len(steps)} "
          f"tokenizer={'yes' if _TOKENIZER else 'NO (ids only)'}")
    print(f"[microscope] http://{host}:{port}  (ssh -L {port}:127.0.0.1:{port} <server>)")
    if preload_workers > 0 and steps:
        threading.Thread(target=_preload, args=(preload_workers,), daemon=True).start()
        print(f"[microscope] background preload started "
              f"({preload_workers} workers, newest step first)", flush=True)

    class _Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 128  # never bounce connections while decompressing

    _Server((host, port), Handler).serve_forever()
