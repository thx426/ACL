"""Engine-backend adapter: the verl rollout engine (current policy weights,
auto-synced every step) exposed as core.episode.GenerateFn — the student
rollout's唯一生成通道(PGS:SPS 钩子在 episode 层复用同一通道做
draft/verify,见 docs/pgs-impl-design.md §2)。
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from ..core.episode import GenResult


class EngineGenerate:
    """The request_id here is a ROUTING KEY only: verl's LLMServerClient uses
    it for sticky replica selection and sends a fresh uuid to vLLM per call
    (llm_server.py), so any number of concurrent requests may share one key.

    Pass a per-(step, task) key so EVERYTHING about one task — all rollout_n
    student episodes (each multi-turn), every research experiment episode —
    lands on the same TP2 replica and hits its prefix cache (user issue
    2026-07-23: no task affinity ⇒ shared prefixes re-prefilled per replica).
    Default (None) keeps the old per-episode identity."""

    def __init__(self, server_manager, tokenizer, request_id: str | None = None) -> None:
        self._server = server_manager
        self._tokenizer = tokenizer
        self._request_id = request_id or uuid4().hex

    async def __call__(self, prompt_ids: list[int], sampling_params: dict[str, Any]) -> GenResult:
        out = await self._server.generate(
            request_id=self._request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )
        ids = list(out.token_ids)
        extra: dict[str, Any] = {}
        out_extra = out.extra_fields or {}
        for key in ("min_global_steps", "max_global_steps"):
            if key in out_extra:
                extra[key] = out_extra[key]
        # verl aborts (e.g. weight sync preemption) return empty ids with
        # stop_reason="aborted" — the episode runner classifies that as an
        # infrastructure fault, not a task outcome.
        stop_reason = getattr(out, "stop_reason", None)
        if stop_reason is not None:
            extra["stop_reason"] = stop_reason
        return GenResult(
            token_ids=ids,
            text=self._tokenizer.decode(ids, skip_special_tokens=True),
            extra=extra,
        )
