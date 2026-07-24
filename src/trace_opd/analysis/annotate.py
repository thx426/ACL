"""Token-type annotator: fill TokenRecord.token_type / token_text
(analysis-design.md view A typology, v1).

Blocks are located by searching the protocol tag token sequences INSIDE the
real sampled ids — no text reconstruction, so no tokenization drift. Rules:
tag tokens -> format; inside <action>...</action> the first token is
action-verb, the rest action-arg; inside <reasoning> -> reasoning-connective
(v1 keeps the whole reasoning body under one label); anything outside a
block (glue/whitespace) -> format.

Usage: python -m trace_opd.analysis.annotate --records <token_records.jsonl>
           --tokenizer <hf-path> [--out <path>]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .schema import TokenRecord, read_jsonl, write_jsonl

TAGS = ("<reasoning>", "</reasoning>", "<action>", "</action>")


def _find(seq: list[int], sub: list[int], start: int = 0) -> int:
    n, m = len(seq), len(sub)
    if not m:
        return -1
    for i in range(start, n - m + 1):
        if seq[i:i + m] == sub:
            return i
    return -1


def _tag_ids(tokenizer) -> dict[str, list[int]]:
    return {tag: tokenizer.encode(tag, add_special_tokens=False) for tag in TAGS}


def _label_span(ids: list[int], tags: dict[str, list[int]]) -> list[str]:
    """Labels for one contiguous assistant-generation span."""
    labels = ["format"] * len(ids)

    def mark_block(open_tag: str, close_tag: str, body: str, head: str | None = None):
        pos = 0
        while True:
            o = _find(ids, tags[open_tag], pos)
            if o < 0:
                return
            body_start = o + len(tags[open_tag])
            c = _find(ids, tags[close_tag], body_start)
            body_end = c if c >= 0 else len(ids)
            for i in range(body_start, body_end):
                labels[i] = body
            if head is not None and body_start < body_end:
                labels[body_start] = head
            pos = (c + len(tags[close_tag])) if c >= 0 else len(ids)
            if c < 0:
                return

    mark_block("<reasoning>", "</reasoning>", "reasoning-connective")
    mark_block("<action>", "</action>", "action-arg", head="action-verb")
    return labels


def annotate(records: list[TokenRecord], tokenizer) -> list[TokenRecord]:
    tags = _tag_ids(tokenizer)
    by_span: dict[tuple[str, int, int], list[TokenRecord]] = defaultdict(list)
    for r in records:
        by_span[(r.traj_id, r.step, r.turn)].append(r)
    for group in by_span.values():
        group.sort(key=lambda r: r.t)
        ids = [r.token_id for r in group]
        labels = _label_span(ids, tags)
        for r, label in zip(group, labels):
            r.token_type = label
            r.token_text = tokenizer.decode([r.token_id])
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True, type=Path)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: annotate in place")
    args = ap.parse_args()
    from transformers import AutoTokenizer  # server-side only

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    records = annotate(read_jsonl(args.records, TokenRecord), tokenizer)
    out = args.out or args.records
    write_jsonl(out, records)
    print(f"annotated {len(records)} records -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
