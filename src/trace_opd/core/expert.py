"""Expert samples for the SFT arm of the mixed objective.

A document-guided rollout becomes a training sample only after it is re-addressed
to the bare prompt and passes three filters. Kept here — free of any verl import
— so the accept/reject rules can be tested without a training environment.

See docs/mixed-objective-design.md §3.
"""
from __future__ import annotations

import logging
from typing import Any

from .adapter import TaskAdapter
from .episode import EpisodeResult, EpisodeTokens, replay_bare_tokens
from .types import mentions_guidance

logger = logging.getLogger(__file__)


def build_expert_sample(
    *,
    adapter: TaskAdapter,
    tokenizer,
    result: EpisodeResult,
    max_model_len: int,
) -> tuple[EpisodeTokens, dict[str, Any]]:
    """Turn a document-guided episode into the SFT arm's sample.

    A rejected sample still comes back as a well-formed row, just with an
    all-zero response mask, so the loss reads exactly nothing from it. That is
    much safer than returning an empty response: several verl paths index the
    last response token unconditionally (``rm_scores[-1] = reward``), and an
    all-zero mask is a shape the trainer already survives — the agent-loop hang
    guard produces it whenever it cancels a prompt.

    Three rejection reasons, each counted onto a curve:
      * the guided rollout did not solve the task — imitating a failure teaches
        failure, and expert data is only worth having while it is expert;
      * the trajectory's own text refers to the guidance document, which the
        bare policy never sees (see ``mentions_guidance``);
      * the replay does not fit the sequence budget.
    """
    if not result.tokens.response_ids:
        # Caller contract: an empty response must never reach the trainer — verl
        # writes rm_scores[-1] unconditionally and would die on the index. The
        # agent loop falls back to a bare rollout instead.
        raise ValueError("build_expert_sample: episode produced no response tokens")

    trajectory = result.trajectory
    fields: dict[str, Any] = {
        "sample_role": "sft",
        "sft_role": 1,
        "sft_won": 1 if trajectory.won else 0,
        "sft_accepted": 0,
        "sft_dropped_ref": 0,
    }
    tokens = replay_bare_tokens(adapter=adapter, tokenizer=tokenizer, result=result)

    rejected = not trajectory.won
    if not rejected:
        text = "\n".join(f"{step.thought}\n{step.action_raw}" for step in trajectory.steps)
        if mentions_guidance(text):
            fields["sft_dropped_ref"] = 1
            rejected = True
    if not rejected:
        total = len(tokens.prompt_ids) + len(tokens.response_ids)
        if total + 8 > max_model_len:
            logger.warning(
                "expert sample for %s overflows the sequence budget (%d tokens) — dropped",
                adapter.task_id,
                total,
            )
            rejected = True

    if rejected:
        tokens.response_mask = [0] * len(tokens.response_ids)
    else:
        fields["sft_accepted"] = 1
    return tokens, fields
