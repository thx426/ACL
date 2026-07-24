"""Research archive — write-only (docs/design.md §8).

Documents have NO cross-step lifecycle: each research session's outputs are
archived for auditability, evidence trails, and the future researcher-SFT
data source, and are never read back into a later session or training step.

Layout:  <root>/research/<task_id_safe>/<ts>_<policy_ver>/
             session.jsonl   researcher conversation, one message per line
             doc.md          the experience document (absent when none)
             episodes.jsonl  structured Trajectory of every experiment episode
             meta.json       counters, outcome, flags, policy version, gt switch
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..common.jsonl import append_jsonl, write_json
from ..research.session import SessionResult
from ..research.tools import EpisodeRecord


def _safe(task_id: str) -> str:
    return task_id.replace("/", "__").replace(" ", "_")


def _episode_row(rec: EpisodeRecord) -> dict[str, Any]:
    """One archive line per experiment episode: structured trajectory PLUS the
    narration the researcher actually read (full observability of the loop)."""
    row: dict[str, Any] = {
        "trajectory": rec.trajectory.to_dict(),
        "fact_card": rec.fact_card.render(),
        "narration": None,
        "narration_skip": None,
        "narration_conflict": False,
    }
    if rec.narration is not None:
        row["narration"] = rec.narration.text
        row["narration_conflict"] = rec.narration.conflict
        if rec.narration.skip is not None:
            row["narration_skip"] = rec.narration.skip.kind
    elif rec.mechanical_log is not None:
        row["narration_skip"] = "mechanical_log"
    return row


class ResearchArchive:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root) / "research"

    def save(
        self,
        *,
        task_id: str,
        result: SessionResult | None,
        records: list[EpisodeRecord],
        policy_version: Any = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        ver = "v?" if policy_version is None else f"v{policy_version}"
        out = self._root / _safe(task_id) / f"{stamp}_{ver}"
        # A task can recur within one wall-clock second across epochs/attempts;
        # suffix until unique rather than overwrite.
        suffix = 0
        final = out
        while final.exists():
            suffix += 1
            final = Path(f"{out}-{suffix}")
        final.mkdir(parents=True, exist_ok=True)

        if result is not None:
            for msg in result.messages:
                append_jsonl(final / "session.jsonl", msg)
            if result.doc:
                (final / "doc.md").write_text(result.doc, encoding="utf-8")
        for rec in records:
            append_jsonl(final / "episodes.jsonl", _episode_row(rec))

        meta: dict[str, Any] = {
            "task_id": task_id,
            "ts": time.time(),
            "policy_version": policy_version,
        }
        if result is not None:
            meta.update(
                {
                    "finalized": result.finalized,
                    "no_finalize": result.no_finalize,
                    "turns_used": result.turns_used,
                    "experiments_used": result.experiments_used,
                    "episodes_consumed": result.episodes_consumed,
                    "doc_words": len(result.doc.split()) if result.doc else 0,
                    "self_assessment": result.self_assessment,
                    "flags": result.flags,
                    "last_exp_successes": result.last_exp_successes,
                    "last_exp_attempts": result.last_exp_attempts,
                    "guidance_log": result.guidance_log,
                    "session_error": result.error,
                    "researcher_prompt_tokens": result.prompt_tokens,
                    "researcher_completion_tokens": result.completion_tokens,
                }
            )
        meta.update(extra_meta or {})
        write_json(final / "meta.json", meta)
        return final
