"""Zero-dependency HTTP server (stdlib only — deploys onto the training host
without touching any venv).

Read-only by construction: every artifact access resolves inside runs_root
(no traversal), the only writes go to the index DB under --data-dir. Binds
127.0.0.1 by default; remote access is `ssh -L`, matching the repo's
loopback-only service discipline.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .index import Index
from .queries import Queries

logger = logging.getLogger(__name__)
_STATIC = Path(__file__).parent / "static"
_MIME = {".html": "text/html", ".js": "text/javascript", ".css": "text/css",
         ".svg": "image/svg+xml"}


def _safe_rel(root: Path, rel: str) -> Path | None:
    """Resolve a run-relative artifact path, refusing anything outside root."""
    if not rel or rel.startswith(("/", "\\")):
        return None
    path = (root / rel).resolve()
    return path if path.is_relative_to(root.resolve()) else None


class VizHandler(BaseHTTPRequestHandler):
    queries: Queries  # injected via make_server

    # ------------------------------------------------------------- plumbing
    def log_message(self, fmt, *args):  # keep the terminal quiet
        logger.debug("%s %s", self.address_string(), fmt % args)

    def _json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # Live-updating data — heuristic browser caching serves stale steps.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, message: str) -> None:
        self._json({"error": message}, code)

    def _static(self, name: str) -> None:
        path = (_STATIC / name).resolve()
        if not path.is_relative_to(_STATIC.resolve()) or not path.is_file():
            self._error(404, "not found")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _MIME.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # --------------------------------------------------------------- routes
    def do_GET(self) -> None:  # noqa: N802 (stdlib API)
        try:
            url = urlparse(self.path)
            params = {k: v[0] for k, v in parse_qs(url.query).items()}
            parts = [unquote(p) for p in url.path.split("/") if p]
            self._route(parts, params)
        except BrokenPipeError:
            pass
        except Exception as exc:  # noqa: BLE001 — a bad request must not kill the server
            logger.exception("request failed: %s", self.path)
            self._error(500, str(exc))

    def _route(self, parts: list[str], params: dict[str, str]) -> None:
        if not parts:
            return self._static("index.html")
        if parts[0] == "static" and len(parts) == 2:
            return self._static(parts[1])
        if parts[0] != "api":
            return self._error(404, "not found")
        if parts[1:] == ["runs"]:
            return self._json({"runs": self.queries.runs()})
        if len(parts) >= 4 and parts[1] == "run":
            run = parts[2]
            if "/" in run or ".." in run:
                return self._error(400, "bad run")
            return self._run_route(run, parts[3:], params)
        return self._error(404, "not found")

    def _run_route(self, run: str, tail: list[str], params: dict[str, str]) -> None:
        q = self.queries
        if tail == ["overview"]:
            return self._json(q.overview(run))
        if tail == ["curves"]:
            keys = [k for k in params.get("keys", "score").split(",") if k]
            mode = params.get("mode", "train")
            try:
                return self._json(q.curves(run, keys, mode))
            except KeyError as exc:
                return self._error(400, str(exc))
        if len(tail) == 3 and tail[0] == "step" and tail[2] == "rollouts":
            try:
                step = int(tail[1])
            except ValueError:
                return self._error(400, "bad step")
            validate = params.get("validate") == "1"
            return self._json(q.step_rollouts(run, step, validate, params))
        if tail == ["rollout"]:
            rel = params.get("file", "")
            if _safe_rel(q.index.runs_root / run, rel) is None:
                return self._error(400, "bad path")
            detail = q.rollout_detail(run, rel)
            return self._json(detail) if detail else self._error(404, "no such rollout")
        if tail == ["research"]:
            rel = params.get("dir", "")
            if _safe_rel(q.index.runs_root / run, rel) is None:
                return self._error(400, "bad path")
            detail = q.research_detail(run, rel)
            return self._json(detail) if detail else self._error(404, "no such session")
        if tail == ["task"]:
            task_id = params.get("task_id", "")
            if not task_id:
                return self._error(400, "task_id required")
            return self._json(q.task_history(run, task_id))
        if tail == ["microscope"]:
            return self._json(q.microscope_overview(run))
        if tail == ["microscope_track"]:
            try:
                step = int(params.get("step", ""))
            except ValueError:
                return self._error(400, "bad step")
            traj = params.get("traj", "")
            if _safe_rel(q.index.runs_root / run, traj) is None:
                return self._error(400, "bad path")
            return self._json(q.microscope_track(run, step, traj))
        return self._error(404, "not found")


def make_server(host: str, port: int, queries: Queries) -> ThreadingHTTPServer:
    handler = type("BoundVizHandler", (VizHandler,), {"queries": queries})
    return ThreadingHTTPServer((host, port), handler)


def start_rescan_thread(index: Index, interval: float) -> threading.Thread:
    def loop() -> None:
        while True:
            time.sleep(interval)
            try:
                index.scan()
            except Exception:  # noqa: BLE001 — scanning must never die silently
                logger.exception("rescan failed")

    t = threading.Thread(target=loop, daemon=True, name="viz-rescan")
    t.start()
    return t
