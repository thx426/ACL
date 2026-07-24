"""Interaction tape: full-fidelity append-only record of every external LLM
call (request messages/tools, raw response, usage, latency, retries, errors).

Twice battle-proven in the pilots (30-second diagnosis of empty narrations;
verbatim evidence of a tool-parser incident). Nothing ever reads the tape
back at runtime — it exists for human inspection and as the future SFT
cold-start data source (docs/design.md §8, §11).

TAPE_CONTEXT carries a free-form attribution label through async call chains
(e.g. "research:<task>:turn03", "narrate:<task>:ep002")."""
from __future__ import annotations

import contextvars
import time
from pathlib import Path
from typing import Any

from .jsonl import append_jsonl

TAPE_CONTEXT: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_tape_context", default=""
)


class TapeWriter:
    def __init__(self, path: str | Path) -> None:
        self._path = str(path)

    def record(self, entry: dict[str, Any]) -> None:
        """Append one call record. Never raises — the tape must not break a run."""
        try:
            append_jsonl(
                self._path,
                {"ts": time.time(), "ctx": TAPE_CONTEXT.get(), **entry},
            )
        except Exception:  # noqa: BLE001
            pass
