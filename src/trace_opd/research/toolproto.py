"""ToolProtocol — how researcher tool use is expressed on the wire.

Decouples the research session from the tool-call transport so the two
researcher forms plug into one session implementation (docs/design.md §3):

  * ``fncall`` — OpenAI function calling (vLLM ``--tool-call-parser
    qwen3_xml`` for the strong-model researcher). Extraction is the client's
    parse; canonicalization (defense against parser-emitted broken JSON in
    history) happens in canonicalize_assistant.
  * ``text``  — plain-text tool protocol for policy-as-researcher and future
    researcher RL: tool documentation is appended to the system prompt, the
    model emits ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``
    (the Qwen-native wire format), results return as ``<tool_response>``
    user messages. No server-side tool parser involved.

Both yield the same ToolCall objects; malformed arguments surface as
arguments={} with raw_arguments preserved (the router refuses execution
without spending budget).
"""
from __future__ import annotations

import json
import re
from typing import Any, Protocol

from ..common.llm import ChatResult, ToolCall


class ToolProtocol(Protocol):
    def prepare(
        self, system_prompt: str, tools: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """Return (system_prompt, tools_for_api). text mode folds the tool
        documentation into the system prompt and passes tools=None."""
        ...

    def extract(self, result: ChatResult) -> list[ToolCall]: ...

    def canonicalize_assistant(self, result: ChatResult) -> dict[str, Any]:
        """The assistant message as it re-enters history."""
        ...

    def tool_result_message(self, call: ToolCall, content: str) -> dict[str, Any]: ...

    def malformed_hint(self, result: ChatResult) -> str | None:
        """A corrective message when the reply LOOKS like a tool call but no
        call could be extracted (None when the reply is plain thinking)."""
        ...


# ------------------------------------------------------------------ fncall


class FncallProtocol:
    name = "fncall"

    def prepare(self, system_prompt, tools):
        return system_prompt, tools

    def extract(self, result: ChatResult) -> list[ToolCall]:
        return list(result.tool_calls)

    def canonicalize_assistant(self, result: ChatResult) -> dict[str, Any]:
        msg = dict(result.raw_message) or {"role": "assistant", "content": result.content}
        msg.setdefault("role", "assistant")
        # Canonicalize tool-call arguments before the message re-enters
        # history: the server-side XML tool parser can emit non-JSON argument
        # strings (observed live: a stray "parameter>" fragment), and replaying
        # such a string 400s every later request when the chat template
        # re-parses history. Always store json.dumps of what WE parsed —
        # identical to what gets dispatched. (The tape keeps the original.)
        if result.tool_calls and msg.get("tool_calls"):
            for raw_tc, parsed in zip(msg["tool_calls"], result.tool_calls):
                if isinstance(raw_tc, dict) and isinstance(raw_tc.get("function"), dict):
                    raw_tc["function"]["arguments"] = json.dumps(
                        parsed.arguments, ensure_ascii=False
                    )
        return msg

    def tool_result_message(self, call: ToolCall, content: str) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": call.id, "content": content}

    def malformed_hint(self, result: ChatResult) -> str | None:
        return None  # the server-side parser owns malformed-call handling


# -------------------------------------------------------------------- text

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

_TEXT_TOOLS_TEMPLATE = """

=== TOOL CALL FORMAT ===
To use an instrument, end your message with exactly one tool call block:
<tool_call>
{{"name": "<tool_name>", "arguments": {{...}}}}
</tool_call>
The arguments object must be valid JSON matching the tool's parameters.
Available tools (JSON schemas):
{tool_docs}
Write your analysis in plain text BEFORE the tool call block. One tool call
per message. The result arrives in the next message inside
<tool_response>...</tool_response>."""


class TextProtocol:
    name = "text"

    def prepare(self, system_prompt, tools):
        docs = "\n".join(
            json.dumps(t["function"], ensure_ascii=False) for t in tools
        )
        return system_prompt + _TEXT_TOOLS_TEMPLATE.format(tool_docs=docs), None

    def extract(self, result: ChatResult) -> list[ToolCall]:
        matches = _TOOL_CALL_RE.findall(result.text or "")
        calls: list[ToolCall] = []
        for i, raw in enumerate(matches):
            name, args, parsed_ok = "", {}, False
            try:
                obj = json.loads(raw)
                name = str(obj.get("name", ""))
                args = obj.get("arguments", {})
                if not isinstance(args, dict):
                    args = {}
                else:
                    parsed_ok = True
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass
            # raw_arguments semantics must match fncall: on a successful parse
            # it is the canonical dump of the ARGUMENTS (so a legal empty {}
            # is not mistaken for a parse failure by the router); only a
            # failed parse carries the offending raw text.
            calls.append(
                ToolCall(
                    id=f"text_{i}",
                    name=name,
                    arguments=args,
                    raw_arguments=json.dumps(args, ensure_ascii=False) if parsed_ok else raw,
                )
            )
        return calls

    def canonicalize_assistant(self, result: ChatResult) -> dict[str, Any]:
        return {"role": "assistant", "content": result.text}

    def tool_result_message(self, call: ToolCall, content: str) -> dict[str, Any]:
        return {"role": "user", "content": f"<tool_response>\n{content}\n</tool_response>"}

    def malformed_hint(self, result: ChatResult) -> str | None:
        text = result.text or ""
        if "<tool_call>" in text and not _TOOL_CALL_RE.search(text):
            return (
                "Your message contains a <tool_call> marker but no well-formed "
                "call could be parsed. Emit exactly:\n<tool_call>\n"
                '{"name": "<tool_name>", "arguments": {...}}\n</tool_call>\n'
                "with the JSON object on its own lines and the closing tag present."
            )
        return None


def make_tool_protocol(name: str) -> ToolProtocol:
    if name == "fncall":
        return FncallProtocol()
    if name == "text":
        return TextProtocol()
    raise ValueError(f"unknown tool protocol {name!r} (expected fncall|text)")
