"""fncall → text-protocol conversion for researcher SFT samples.

The sampling model speaks whatever its serving stack supports best — the
STANDARD OpenAI function-calling surface (user ruling 2026-07-28: the sampler
must not be Qwen-specific; a future non-Qwen sampler plugs in unchanged). The
POLICY trains and runs research under the TEXT protocol (the Qwen-native
``<tool_call>`` wire format, research.tool_protocol=text). This module is the
deterministic bridge: archived fncall messages become the exact text-protocol
conversation the policy would have had, so the SFT distribution matches the
self-evolve runtime byte-for-byte on the wire format.

Correctness contract (pinned by tests):
  * the rebuilt system prompt equals ``TextProtocol.prepare(system, tools)`` —
    the same folding the runtime performs;
  * every converted assistant turn re-parses through ``TextProtocol.extract``
    into the same (name, arguments) calls that were archived;
  * tool results become ``<tool_response>`` user turns exactly as
    ``TextProtocol.tool_result_message`` emits them.

Sessions archived under the text protocol pass through unchanged (identity).
"""
from __future__ import annotations

import json
from typing import Any

from ..research.toolproto import TextProtocol


def _assistant_to_text(message: dict[str, Any]) -> str:
    """Render an fncall assistant message onto the text wire.

    Analysis text first (the session's WRITE BEFORE YOU ACT discipline), then
    one ``<tool_call>`` block per archived call — faithful conversion, no
    behaviour editing: a multi-call message stays multi-call (the runtime
    executes the first and says so; export filters may drop such sessions).
    """
    parts: list[str] = []
    content = str(message.get("content") or "").rstrip()
    if content:
        parts.append(content)
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        raw_arguments = function.get("arguments")
        # canonicalize_assistant stores json.dumps of the parsed arguments, so
        # this loads cleanly for every archived session; anything else is a
        # corrupt archive and must fail loud, not ship garbage into SFT.
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        if not isinstance(arguments, dict):
            raise ValueError(
                f"archived tool call has non-dict arguments: {raw_arguments!r}"
            )
        payload = json.dumps(
            {"name": name, "arguments": arguments}, ensure_ascii=False
        )
        parts.append(f"<tool_call>\n{payload}\n</tool_call>")
    return "\n".join(parts)


def fncall_messages_to_text(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Convert an archived fncall conversation to its text-protocol twin.

    ``tools`` are the JSON schemas the session was prepared with
    (research.tools.TOOL_SCHEMAS) — the text system prompt folds them in.
    """
    proto = TextProtocol()
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            prepared, api_tools = proto.prepare(str(message.get("content") or ""), tools)
            assert api_tools is None
            converted.append({"role": "system", "content": prepared})
        elif role == "assistant":
            converted.append(
                {"role": "assistant", "content": _assistant_to_text(message)}
            )
        elif role == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": f"<tool_response>\n{message.get('content') or ''}\n</tool_response>",
                }
            )
        elif role == "user":
            converted.append({"role": "user", "content": str(message.get("content") or "")})
        else:
            raise ValueError(f"unexpected role in archived session: {role!r}")
    return converted


def messages_to_text_protocol(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_protocol: str,
) -> list[dict[str, Any]]:
    """Protocol-dispatching entry: text sessions pass through as-is (their
    system prompt already carries the folded tool docs), fncall sessions are
    converted."""
    if tool_protocol == "text":
        return [dict(m) for m in messages]
    if tool_protocol == "fncall":
        return fncall_messages_to_text(messages, tools)
    raise ValueError(f"unknown tool protocol {tool_protocol!r}")
