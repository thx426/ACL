"""Research-group pure logic (no verl imports — unit-testable directly).

The agent loop owns the async orchestration; the decisions that define the
mechanism live here.
"""
from __future__ import annotations

from typing import Any


def select_best_doc(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Teacher admission over a research group (docs/self-research-design.md
    3.2-4): argmax measured utility p, ties to the SHORTER document, and p<=0
    admits nothing — a document that never helped downstream never becomes the
    distillation teacher (PG-5(3) input filtering, endogenous form)."""
    best: dict[str, Any] | None = None
    for r in results:
        if r.get("doc") and float(r.get("p", 0.0)) > 0.0:
            better = (
                best is None
                or r["p"] > best["p"]
                or (r["p"] == best["p"] and len(r["doc"]) < len(best["doc"]))
            )
            if better:
                best = r
    return best
