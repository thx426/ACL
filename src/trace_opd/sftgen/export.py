"""Export sampled researcher sessions to verl multiturn-SFT parquet.

Reads the pipeline's per-task archives, converts every conversation to the
TEXT protocol (the wire format the policy trains and runs research under —
sftgen/convert.py, deterministic), applies the quality filter, and writes a
parquet with a ``messages`` column — the schema verl's MultiTurnSFTDataset
consumes (messages_key="messages"; assistant turns carry the loss).

Filtering happens HERE, not at sampling time: the archives keep everything,
so a different filter is a re-export, not a re-sample.

Default filter: finalized, no session error. Optional: --min-gain (keep only
sessions whose document beat the bare baseline by at least this much on the
initial policy) and --drop-multi-call (drop sessions where any assistant turn
carries more than one tool call — the runtime executes only the first, so
single-call behaviour is the cleaner thing to teach).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..research.prompts import TOOL_SCHEMAS
from .convert import messages_to_text_protocol


def iter_sessions(archive_dir: Path):
    for meta_path in sorted(archive_dir.glob("tasks/*/*/meta.json")):
        session_dir = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        session = json.loads(
            (session_dir / "session.json").read_text(encoding="utf-8")
        )
        yield session_dir, meta, session


def _multi_call_turns(messages: list[dict[str, Any]]) -> int:
    return sum(
        1
        for m in messages
        if m.get("role") == "assistant" and len(m.get("tool_calls") or []) > 1
    )


def export(
    archive_dir: Path,
    out_path: Path,
    *,
    min_gain: float | None = None,
    drop_multi_call: bool = False,
) -> dict[str, Any]:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    stats = {
        "sessions": 0,
        "kept": 0,
        "dropped_not_finalized": 0,
        "dropped_error": 0,
        "dropped_gain": 0,
        "dropped_multi_call": 0,
    }
    for session_dir, meta, session in iter_sessions(archive_dir):
        stats["sessions"] += 1
        if not meta.get("finalized"):
            stats["dropped_not_finalized"] += 1
            continue
        if meta.get("error"):
            stats["dropped_error"] += 1
            continue
        if min_gain is not None and float(meta.get("gain", 0.0)) < min_gain:
            stats["dropped_gain"] += 1
            continue
        messages = session["messages"]
        if drop_multi_call and _multi_call_turns(messages):
            stats["dropped_multi_call"] += 1
            continue
        text_messages = messages_to_text_protocol(
            messages, TOOL_SCHEMAS, session.get("tool_protocol", "fncall")
        )
        rows.append(
            {
                "messages": text_messages,
                "task_id": meta["task_id"],
                "gain": float(meta.get("gain", 0.0)),
                "p_doc": float(meta.get("p_doc", 0.0)),
                "doc_words": int(meta.get("doc_words", 0)),
                "researcher_model": meta.get("researcher_model", ""),
                "source_dir": str(session_dir),
            }
        )
        stats["kept"] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    manifest = {
        "archive_dir": str(archive_dir),
        "out_path": str(out_path),
        "min_gain": min_gain,
        "drop_multi_call": drop_multi_call,
        **stats,
    }
    out_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"[export] {json.dumps(manifest, ensure_ascii=False)}", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--out", required=True, help="output parquet path")
    parser.add_argument("--min-gain", type=float, default=None)
    parser.add_argument("--drop-multi-call", action="store_true")
    args = parser.parse_args()
    export(
        Path(args.archive_dir),
        Path(args.out),
        min_gain=args.min_gain,
        drop_multi_call=args.drop_multi_call,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
