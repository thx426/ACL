"""Worker-side runtime helpers — stdlib-only, alien-interpreter safe.

Copied from experience_internalization_claude (read-only asset source,
audited 2026-07-20); unchanged except this note. Loaded BY FILE PATH by
worker.py — the worker runs under the environment's own interpreter where
this package is not importable.
"""
from __future__ import annotations

import json
import os
import sys

CLOSE_SENTINEL = "__CLOSE__"
DEFAULT_MESSAGE_CAP = 65536


def harden(pdeathsig: bool = True, max_rss_gb: float = 0.0) -> None:
    """Best-effort worker self-hardening. Never raises."""
    if pdeathsig and sys.platform.startswith("linux"):
        try:
            import ctypes
            import signal

            PR_SET_PDEATHSIG = 1
            libc = ctypes.CDLL(None, use_errno=True)
            libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
        except Exception:  # noqa: BLE001 — hardening is optional
            pass
    if max_rss_gb and max_rss_gb > 0:
        try:
            import resource

            limit = int(max_rss_gb * (1024**3))
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except Exception:  # noqa: BLE001
            pass


def protocol_channel():
    """Duplicate the real stdout for protocol lines, then fd1 -> fd2 so
    stray engine output cannot corrupt the line protocol."""
    proto = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return proto


def truncate_middle(text: str, max_chars: int = 6000) -> str:
    text = "" if text is None else str(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    omitted = len(text) - head - tail
    return text[:head] + f"\n... [{omitted} chars truncated] ...\n" + text[-tail:]


def emit(proto, payload: dict) -> None:
    proto.write(json.dumps(payload, ensure_ascii=False) + "\n")
    proto.flush()


def iter_stdin_lines():
    """Yield request lines until CLOSE_SENTINEL or stdin EOF (parent died)."""
    for line in sys.stdin:
        line = line.rstrip("\n")
        if line == CLOSE_SENTINEL:
            return
        yield line
