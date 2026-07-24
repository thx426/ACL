"""Engine-backend adapters: the verl rollout engine (current policy weights,
auto-synced every step) exposed through the mechanism's three function
surfaces (docs/design.md §3).

  EngineGenerate      -> core.episode.GenerateFn   (executor episodes, student rollout)
  engine_narrate_fn   -> narrate.narrator.NarrateFn (policy narrates its own trajectories)
  engine_researcher_chat -> research.session.ResearcherChatFn (policy-as-researcher, text protocol)

External counterparts live in common/llm.py; mechanism code holds only the
function surface and never knows which backend it got.
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from ..common.llm import ChatResult
from ..core.episode import GenResult
from ..core.template_delta import render_ids


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


def engine_narrate_fn(server_manager, tokenizer, request_id: str | None = None):
    """NarrateFn on the rollout engine: render → generate → decode.
    With a task routing key, narrations join the task's sticky replica (the
    narrator system prefix and the narrated trajectory both live there)."""

    async def narrate(messages: list[dict[str, Any]], temperature: float, max_tokens: int) -> str:
        prompt_ids = render_ids(tokenizer, messages, True)
        out = await server_manager.generate(
            request_id=request_id or uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params={"temperature": temperature, "max_tokens": max_tokens},
        )
        return tokenizer.decode(list(out.token_ids), skip_special_tokens=True)

    return narrate


def engine_researcher_chat(
    server_manager, tokenizer, *, temperature: float, max_tokens: int,
    request_id: str | None = None,
):
    """ResearcherChatFn on the rollout engine (policy-as-researcher).

    Only valid with the ``text`` tool protocol: the engine surface has no
    function-calling parser, so tool documentation is already folded into the
    system prompt and tool calls ride the completion text (toolproto.py).
    Interface implemented for the self-evolve form; not exercised in round 1.
    """

    async def chat(
        messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> ChatResult:
        assert not tools, (
            "engine researcher backend requires research.tool_protocol=text "
            "(function-calling tools cannot be served by the rollout engine)"
        )
        prompt_ids = render_ids(tokenizer, messages, True)
        out = await server_manager.generate(
            request_id=request_id or uuid4().hex,  # multi-turn session: sticky
            prompt_ids=prompt_ids,
            sampling_params={"temperature": temperature, "max_tokens": max_tokens},
        )
        text = tokenizer.decode(list(out.token_ids), skip_special_tokens=True)
        return ChatResult(content=text)

    return chat
