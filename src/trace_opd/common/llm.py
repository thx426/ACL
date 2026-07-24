"""External-backend LLM client: one protocol, an OpenAI-compatible
implementation (docs/design.md §3).

This is the ``external`` role backend (researcher on a strong model, optional
external narrator). The ``engine`` backend (verl rollout engine) implements
the same ChatFn surface in train/backends.py — mechanism code never knows
which one it holds.

Defensive discipline inherited from the pilots (all client-side, verified in
live incidents — do NOT rely on server-side fixes):
  * tool-call arguments that fail JSON parsing are surfaced as arguments={}
    with raw_arguments preserved, so the tool layer can refuse execution
    without spending budget (defense 2 lives in research/tools.py);
  * canonicalizing arguments before history re-entry is the session's job
    (defense 1, research/session.py);
  * every completed call reports usage and lands on the tape in full.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .tape import TapeWriter


def dp_rank_for(route_key: str, dp_size: int) -> int:
    """Deterministic DP-rank for a routing key (client-side sticky routing to
    one vLLM data-parallel engine via the X-data-parallel-rank header).

    crc32, NOT hash(): builtin hash is salted per process (PYTHONHASHSEED),
    and the agent-loop workers are separate processes — the same task must map
    to the same rank from every worker or the affinity is lost."""
    return zlib.crc32(route_key.encode("utf-8")) % dp_size


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""


@dataclass
class ChatResult:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    raw_message: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.content or ""


class ChatClient(Protocol):
    role: str

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult: ...


class LLMError(RuntimeError):
    pass


class OpenAICompatClient:
    """Minimal, robust OpenAI-compatible /chat/completions client."""

    RETRIABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        *,
        role: str,
        endpoint: str,
        model: str,
        api_key: str = "",
        max_concurrency: int = 4,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout_s: float = 180.0,
        max_retries: int = 5,
        tape: TapeWriter | None = None,
        extra_body: dict[str, Any] | None = None,
        dp_size: int = 0,
    ) -> None:
        if not endpoint:
            raise LLMError(f"role {role!r}: empty endpoint (env var not set?)")
        self.role = role
        self.model = model
        self._url = endpoint.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(timeout=timeout_s, headers=headers)
        self._sem = asyncio.Semaphore(max_concurrency)
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._tape = tape
        self._extra_body = dict(extra_body or {})
        self._dp_size = dp_size

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        route_key: str | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self._temperature if temperature is None else temperature,
            "max_tokens": self._max_tokens if max_tokens is None else max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        for key, value in self._extra_body.items():  # additive only, never overrides
            payload.setdefault(key, value)

        # DP sticky routing: pin every request of one session/task to a fixed
        # data-parallel engine so its prefix cache actually accumulates
        # (vLLM honors X-data-parallel-rank ahead of least-loaded balancing;
        # verified on 0.19.1 by CD-006, feedback 2026-07-23).
        headers: dict[str, str] | None = None
        dp_rank: int | None = None
        if route_key is not None and self._dp_size > 0:
            dp_rank = dp_rank_for(route_key, self._dp_size)
            headers = {"X-data-parallel-rank": str(dp_rank)}

        t0 = time.monotonic()
        last_err: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                async with self._sem:
                    resp = await self._client.post(self._url, json=payload, headers=headers)
                if resp.status_code in self.RETRIABLE_STATUS:
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                resp.raise_for_status()
                result = self._parse(resp.json())
                self._tape_call(payload, attempt + 1, t0, result=result, dp_rank=dp_rank)
                return result
            except (httpx.TransportError, LLMError) as err:  # retriable
                last_err = err
                if attempt >= self._max_retries:
                    break
                delay = min(30.0, (2.0**attempt) * (0.5 + random.random()))
                await asyncio.sleep(delay)
            except httpx.HTTPStatusError as err:  # non-retriable 4xx
                self._tape_call(
                    payload, attempt + 1, t0,
                    error=f"HTTP {err.response.status_code}: {err.response.text[:500]}",
                    dp_rank=dp_rank,
                )
                raise LLMError(
                    f"role {self.role}: {err.response.status_code} "
                    f"{err.response.text[:500]}"
                ) from err
        self._tape_call(payload, self._max_retries + 1, t0, error=str(last_err), dp_rank=dp_rank)
        raise LLMError(f"role {self.role}: exhausted retries — {last_err}")

    def _tape_call(
        self,
        payload: dict[str, Any],
        attempts: int,
        t0: float,
        *,
        result: ChatResult | None = None,
        error: str | None = None,
        dp_rank: int | None = None,
    ) -> None:
        if self._tape is None:
            return
        self._tape.record(
            {
                "role": self.role,
                "model": self.model,
                "attempts": attempts,
                "dp_rank": dp_rank,
                "latency_s": round(time.monotonic() - t0, 3),
                "request": {
                    "temperature": payload.get("temperature"),
                    "max_tokens": payload.get("max_tokens"),
                    "messages": payload["messages"],
                    **({"tools": payload["tools"]} if "tools" in payload else {}),
                },
                "response": None if result is None else result.raw_message,
                "finish_reason": "" if result is None else result.finish_reason,
                "usage": None
                if result is None
                else {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                },
                "error": error,
            }
        )

    def _parse(self, data: dict[str, Any]) -> ChatResult:
        try:
            choice = data["choices"][0]
            message = choice.get("message") or {}
        except (KeyError, IndexError) as err:
            raise LLMError(f"malformed completion response: {str(data)[:500]}") from err
        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                if not isinstance(args, dict):
                    args = {}
            except (json.JSONDecodeError, TypeError, ValueError):
                # Malformed OR non-mapping arguments from a compat server must
                # soft-land as {} (router refuses without spending), never
                # kill the session.
                args = {}
            tool_calls.append(
                ToolCall(
                    id=str(tc.get("id", f"call_{len(tool_calls)}")),
                    name=str(fn.get("name", "")),
                    arguments=args,
                    raw_arguments=raw_args if isinstance(raw_args, str) else json.dumps(raw_args),
                )
            )
        usage = data.get("usage") or {}
        return ChatResult(
            content=message.get("content"),
            tool_calls=tool_calls,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            finish_reason=str(choice.get("finish_reason", "")),
            raw_message=message,
        )


def make_client(role: str, model_cfg, tape: TapeWriter | None = None) -> OpenAICompatClient:
    """Build a client from a RoleCfg (common/config.py)."""
    return OpenAICompatClient(
        role=role,
        endpoint=model_cfg.endpoint,
        model=model_cfg.model,
        api_key=model_cfg.api_key,
        max_concurrency=model_cfg.max_concurrency,
        temperature=model_cfg.temperature,
        max_tokens=model_cfg.max_tokens,
        timeout_s=model_cfg.timeout_s,
        tape=tape,
        extra_body=model_cfg.extra_body,
        dp_size=getattr(model_cfg, "dp_size", 0),
    )
