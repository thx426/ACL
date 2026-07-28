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

class ResearchBudgetExceeded(RuntimeError):
    """The research session's token stream hit its hard budget. The session
    loop catches any exception into a partial SessionResult, so raising here
    ends the session cleanly: an already-finalized doc survives, an unfinished
    one degrades to doc=None -> p=0 (docs/self-research-design.md 3.2)."""


class EngineResearcherTokenChat:
    """ResearcherChatFn on the rollout engine that ALSO builds the training
    sample (trace-opd self-research, docs/self-research-design.md 4.1).

    Same append-only discipline as core.episode: the researcher's generated
    ids join the stream verbatim (mask=1) — training tokens ARE the engine's
    sampled tokens, never a re-render (BPE roundtrips do not commute; UCOB's
    re-tokenized replay is the cautionary tale) — and everything the session
    injects between generations (tool results, wrap-up reminders, malformed
    hints) is rendered as ONE merged user turn through the TemplateDelta glue
    (mask=0). The archive keeps the messages un-merged; only the token stream
    coalesces consecutive user turns, which is template-equivalent content.

    The instance is single-session: the agent loop constructs one per research
    replica and reads ``token_sample()`` after ``session.run()`` returns.
    """

    def __init__(
        self, server_manager, tokenizer, delta, *, temperature: float,
        max_tokens: int, max_sample_tokens: int, request_id: str,
    ) -> None:
        self._server = server_manager
        self._tokenizer = tokenizer
        self._delta = delta
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_sample_tokens = max_sample_tokens
        self._request_id = request_id
        self.prompt_ids: list[int] = []
        self.stream: list[int] = []
        self.mask: list[int] = []
        self._n_seen = 0
        self._last_gen_ids: list[int] = []
        self._last_gen_text: str | None = None

    async def __call__(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> ChatResult:
        assert not tools, (
            "engine researcher token backend requires research.tool_protocol=text "
            "(function-calling tools cannot be served by the rollout engine)"
        )
        if not self.prompt_ids:
            # First call: [system, kickoff] becomes the sample's prompt.
            self.prompt_ids = render_ids(self._tokenizer, messages, True)
        else:
            new = messages[self._n_seen :]
            if not new or new[0].get("role") != "assistant":
                raise RuntimeError(
                    "token researcher stream out of step: expected the previous "
                    f"assistant turn at index {self._n_seen}, got "
                    f"{[m.get('role') for m in new]}"
                )
            if (new[0].get("content") or "") != (self._last_gen_text or ""):
                # canonicalize_assistant must be the identity under the text
                # protocol; anything else forks the archive from the sample.
                raise RuntimeError(
                    "token researcher stream out of step: the archived assistant "
                    "turn differs from the engine generation (canonicalize is "
                    "expected to be the identity under the text protocol)"
                )
            user_texts = []
            for message in new[1:]:
                if message.get("role") != "user":
                    raise RuntimeError(
                        "token researcher stream out of step: non-user message "
                        f"{message.get('role')!r} between generations"
                    )
                user_texts.append(str(message.get("content") or ""))
            merged = "\n\n".join(user_texts)
            glue = self._delta.after_generation(self._last_gen_ids)
            user_ids = self._tokenizer.encode(merged, add_special_tokens=False)
            structure = glue + user_ids + self._delta.post_obs
            self.stream += structure
            self.mask += [0] * len(structure)

        budget = self._max_sample_tokens - len(self.prompt_ids) - len(self.stream)
        if budget < 64:
            raise ResearchBudgetExceeded(
                f"research token stream at {len(self.prompt_ids) + len(self.stream)} "
                f"tokens exceeds max_sample_tokens={self._max_sample_tokens}"
            )
        out = await self._server.generate(
            request_id=self._request_id,  # multi-turn session: sticky replica
            prompt_ids=self.prompt_ids + self.stream,
            sampling_params={
                "temperature": self._temperature,
                "max_tokens": min(self._max_tokens, budget),
            },
        )
        gen_ids = list(out.token_ids)
        if getattr(out, "stop_reason", None) == "aborted" or not gen_ids:
            raise ResearchBudgetExceeded(
                "engine aborted or returned no tokens mid research session"
            )
        text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
        self.stream += gen_ids
        self.mask += [1] * len(gen_ids)
        self._last_gen_ids = gen_ids
        self._last_gen_text = text
        self._n_seen = len(messages)
        return ChatResult(
            content=text,
            prompt_tokens=len(self.prompt_ids) + len(self.stream) - len(gen_ids),
            completion_tokens=len(gen_ids),
        )

    def token_sample(self) -> tuple[list[int], list[int], list[int]]:
        """(prompt_ids, response_ids, response_mask) — the training sample."""
        return list(self.prompt_ids), list(self.stream), list(self.mask)

