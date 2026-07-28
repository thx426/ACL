"""Distillation-microscope workbench server — stdlib http + numpy only.

Serves the observation packs behind a small JSON API and a single-page
front-end (static/). All derived quantities are computed on demand from the
raw packs (derive.py); cross-step aggregates (leaderboard, evolution) are
memoised in process — the cache IS disposable, restart recomputes it, and a
new analysis view is a code change plus a restart, never a resample.

Run on the training server, browse through an SSH tunnel:

  PYTHONPATH=src python3 -m trace_opd.microscope serve \\
      --obs-dir /abs/run/obs --tokenizer <model_path> [--port 8643]
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import derive
from .reader import list_steps, load_step

_STATIC = Path(__file__).parent / "static"
_TOKENIZER = None
_OBS_DIR: str = ""
_MEMO: dict = {}

_METRICS = (
    "kl_forward", "kl_reverse", "token_loss", "gap", "teacher_entropy",
    "student_entropy", "overlap_count", "teacher_mass", "student_mass_own",
)


def _tok_text(token_id: int) -> str:
    if _TOKENIZER is None:
        return f"<{token_id}>"
    piece = _TOKENIZER.convert_ids_to_tokens([int(token_id)])[0]
    return piece.replace("Ġ", " ").replace("Ċ", "\\n").replace("▁", " ")


def _clean(value):
    if isinstance(value, np.ndarray):
        return [None if not np.isfinite(v) else round(float(v), 5) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else round(float(value), 5)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


@lru_cache(maxsize=64)
def _samples(step: int) -> list[dict]:
    pack = load_step(_OBS_DIR, step)
    return [derive.sample_summary(row) for row in pack.rows()]


def _sample_payload(step: int, resp_hash: int) -> dict:
    pack = load_step(_OBS_DIR, step)
    row = pack.row(resp_hash)
    tr = derive.tracks(row, pack.log_prob_min_clamp)
    repeats = derive.repeat_positions(row)
    tokens = [{"id": int(t), "text": _tok_text(t)} for t in row["response_ids"]]
    return {
        "summary": derive.sample_summary(row),
        "loss_mode": pack.loss_mode,
        "turn_spans": row["turn_spans"],
        "tokens": tokens,
        "tracks": {k: _clean(v) for k, v in tr.items()},
        "repeat_mask": [int(x) for x in repeats.tolist()],
        "audit": derive.audit_row(row, loss_mode=pack.loss_mode, clamp=pack.log_prob_min_clamp),
    }


def _position_payload(step: int, resp_hash: int, pos: int) -> dict:
    pack = load_step(_OBS_DIR, step)
    row = pack.row(resp_hash)
    tr = derive.tracks(row, pack.log_prob_min_clamp)

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
        "label": derive.divergence_label(row, pos),
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
            "tokens": [_tok_text(t) for t in row["response_ids"][lo:hi]],
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
        pack = load_step(_OBS_DIR, step)
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
                        "step": step, "hash": row["resp_hash"], "task": row["task_id"],
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
        pack = load_step(_OBS_DIR, step)
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
         "n": len(vals), "hash": reps[(step, turn)][0]}
        for (step, turn), vals in sorted(acc.items())
    ]
    out = {"task": task, "metric": metric, "cells": cells}
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
    def log_message(self, *args):  # quiet
        pass

    def _json(self, payload, code=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/meta":
                return self._json(
                    {
                        "obs_dir": _OBS_DIR,
                        "steps": list_steps(Path(_OBS_DIR)),
                        "metrics": list(_METRICS),
                        "tokenizer": _TOKENIZER is not None,
                        "tasks": _tasks(),
                    }
                )
            if url.path == "/api/samples":
                return self._json(_samples(int(q["step"])))
            if url.path == "/api/sample":
                return self._json(_sample_payload(int(q["step"]), int(q["hash"])))
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


def serve(obs_dir: str, tokenizer_path: str | None, host: str, port: int) -> None:
    global _OBS_DIR, _TOKENIZER
    _OBS_DIR = str(obs_dir)
    if tokenizer_path:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path)
    steps = list_steps(Path(obs_dir))
    print(f"[microscope] obs={obs_dir} steps={len(steps)} "
          f"tokenizer={'yes' if _TOKENIZER else 'NO (ids only)'}")
    print(f"[microscope] http://{host}:{port}  (ssh -L {port}:127.0.0.1:{port} <server>)")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
