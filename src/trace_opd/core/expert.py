"""Expert samples for the SFT arm of the mixed objective.

A document-guided rollout becomes a training sample only after it is re-addressed
to the bare prompt and passes three filters. Kept here — free of any verl import
— so the accept/reject rules can be tested without a training environment.

See docs/mixed-objective-design.md §3.
"""
from __future__ import annotations

import logging
from dataclasses import replace
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
    prefix: str = "doc",
) -> tuple[EpisodeTokens, dict[str, Any]]:
    """Turn a document-guided episode into the SFT arm's sample.

    ``prefix`` decides what the SFT term optimises, and the two answers are not
    variations on one idea — they are different objectives:

    ``doc`` (default) keeps the prompt the episode was generated under, so the
    term optimises ``policy(·|bare + document)``, which IS the OPD teacher. The
    division of labour becomes "SFT strengthens the teacher, OPD carries it
    across", aimed straight at the measured failure: the teacher losing its grip
    on the document once it is dropped into the student's erroneous prefix.

    ``bare`` re-addresses the trajectory to the document-free prompt, so the term
    teaches the bare policy directly. That overlaps with what OPD already does,
    and it leaks: the expert knew where objects were because it had read the
    document, so under a bare prompt those statements have no basis. Measured on
    the first A2 run, 23.6% of accepted samples were teaching the model to
    assert an object's location at step 0 with nothing observed — and no expert
    trajectory demonstrates recovering from a wrong guess, because an expert
    with the document never makes one. Kept as a control arm.

    A rejected sample still comes back as a well-formed row, just with an
    all-zero response mask, so the loss reads exactly nothing from it. That is
    much safer than returning an empty response: several verl paths index the
    last response token unconditionally (``rm_scores[-1] = reward``), and an
    all-zero mask is a shape the trainer already survives — the agent-loop hang
    guard produces it whenever it cancels a prompt.

    Rejection reasons, each counted onto a curve: the rollout did not solve the
    task; the sample does not fit the sequence budget; and — under ``bare`` only
    — the trajectory refers to the document in its own words.
    """
    if prefix not in ("doc", "bare"):
        raise ValueError(f"build_expert_sample: prefix must be 'doc' or 'bare', got {prefix!r}")
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
    if prefix == "doc":
        # Nothing to rewrite: the episode already carries the prompt this term
        # optimises. (The same identity that makes the bare replay exact also
        # makes not replaying exact — the response stream never depended on
        # what was in the system message.)
        tokens = replace(
            result.tokens,
            prompt_ids=list(result.tokens.prompt_ids),
            response_ids=list(result.tokens.response_ids),
            response_mask=list(result.tokens.response_mask),
            turn_spans=[list(span) for span in result.tokens.turn_spans],
        )
    else:
        tokens = replay_bare_tokens(adapter=adapter, tokenizer=tokenizer, result=result)

    rejected = not trajectory.won
    if not rejected and prefix == "bare":
        # Only meaningful without the document: with it in the prompt, referring
        # to it is a correct thing for the trajectory to do, and filtering those
        # samples would drop the good ones.
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
