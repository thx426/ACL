"""Per-sample scalars for the swanlab curve channel (user request 2026-07-23:
richer training-time observability of the mechanism).

The dict returned here goes into ``extra_fields["reward_extra_info"]``; verl
turns every numeric key into a curve — ``rollout_extra/<key>/mean`` per train
step (verl patch f8f26d5) and ``val-aux/<data_source>/<key>/mean@1`` per val.

HOMOGENEITY CONTRACT: verl expands the batch using the FIRST sample's key set
and indexes every sample with it, so all samples of one generate_sequences
call must produce the same keys. The three shapes below (train+research /
train without research / val) are each internally homogeneous because
``validate`` and ``research_enabled`` are batch-wide constants.

Key curve reading (mechanism health): ``research_final_success`` (doc-guided
experiment success) vs ``critic/score/mean`` (bare on-policy success, free
from the rollout's own won signal — no extra eval pass) — the gap IS the
document's privileged gain; score rising while the gap narrows is the
internalization signature.
"""
from __future__ import annotations

from typing import Any

_ALWAYS = ("env_steps", "invalid_format", "truncated")
_RESEARCH = (
    "research_final_success",
    "doc_missing",
    "research_error",
    "doc_words",
    "research_experiments",
    "research_episodes",
)
# Mixed objective, SFT arm (docs/mixed-objective-design.md).
_MIXED_SFT = ("sft_role", "sft_won", "sft_accepted", "sft_dropped_ref")


def build_curve_metrics(
    *,
    validate: bool,
    research_enabled: bool,
    fields: dict[str, Any],
    mixed_sft: bool = False,
) -> dict[str, float]:
    """Assemble the per-sample curve scalars from the sample's extra_fields.

    Call AFTER the teacher-prefix step so ``doc_missing`` reflects the final
    state (a doc overflowing the teacher context degrades to bare and flips
    doc_missing to 1 in extra_fields after research reported 0).

    ``mixed_sft`` is a batch-wide constant (the config's SFT weight), so adding
    its keys keeps the homogeneity contract. They are per-BATCH means over all
    replicas, expert and bare alike — read them as ratios: ``sft_won/sft_role``
    is the document-guided success rate (the PG-7 document-quality signal) and
    ``sft_accepted/sft_role`` is what actually reaches the SFT loss.
    """
    metrics = {key: float(fields[key]) for key in _ALWAYS}
    if not validate and research_enabled:
        for key in _RESEARCH:
            metrics[key] = float(fields.get(key, 0) or 0)
    if not validate and mixed_sft:
        for key in _MIXED_SFT:
            metrics[key] = float(fields.get(key, 0) or 0)
    return metrics
