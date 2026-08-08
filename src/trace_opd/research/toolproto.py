"""ToolProtocol — how researcher tool use is expressed on the wire.

Decouples the research session from the tool-call transport so the two
researcher forms plug into one session implementation (docs/design.md §3).
Both are the SAME OpenAI-format Function Call protocol(需求 §2/§6,用户
裁定 2026-08-08)— they differ only in where parsing happens:

  * ``fncall`` — server-parsed FC for the external strong-model researcher:
    tools go to the chat.completions API, the serving engine's tool parser
    (Qwen3.5 上必须 ``--tool-call-parser qwen3_coder``,见
    core.protocol.SERVER_TOOL_CALL_PARSER 的绑定纪律) returns structured
    tool_calls. Canonicalization (defense against parser-emitted broken
    JSON in history) happens in canonicalize_assistant.
  * ``text``  — client-parsed FC for the token-in-token-out channel
    (policy-as-researcher / research RL,§6.2「训练期转纯 text」的落地):
    tools are handed to the chat template for native rendering (never
    hand-written into the system prompt), the model's raw generated tokens
    stay in the training stream, extraction is the family-dispatched
    text-level parser (core.protocol.parse_tool_calls, golden 对拍服务端
    parser), and tool results return as ``tool_response_wrap`` user turns
    (byte-identical to the template's own role=tool rendering —
    fc_stream_equivalence_probe 守护).

Both yield the same ToolCall objects; a reply that looks like a call but
parses to nothing gets a corrective hint and spends no budget.
"""
from __future__ import annotations

import json
from typing import Any, Protocol

from ..common.llm import ChatResult, ToolCall
from ..core import protocol as core_protocol


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

#: 会话行为规约(协议行为,非 wire 格式——调用的具体形态由 chat template
#: 原生教学,此处绝不复述,避免与模板家族的格式教学冲突)。
_TOOL_USE_RULES = """

=== TOOL USE ===
Your instruments are declared as native tools. Write your analysis in plain
text before any tool call. Make exactly ONE tool call per message; its
result arrives in the next message inside <tool_response>...</tool_response>."""


class TextProtocol:
    """Client-parsed FC(token-in-token-out 通道;§6.2 用户裁定的落地)。

    历史形态(schema JSON 塞 system prompt + 手写 hermes 块,要求模型把整篇
    doc 逐字符转义进 JSON)在 ≤9B 上实测 45.2% 工具调用失败,且在 Qwen3.5
    上是分布外形态,已废弃(2026-08-08)。现形态:tools 交模板渲染、
    家族分派解析、schema 类型还原(k:int 等)。"""

    name = "text"

    def __init__(self) -> None:
        self._schemas: list[dict[str, Any]] | None = None

    def prepare(self, system_prompt, tools):
        self._schemas = tools
        return system_prompt + _TOOL_USE_RULES, tools

    def extract(self, result: ChatResult) -> list[ToolCall]:
        parsed = core_protocol.parse_tool_calls(result.text or "", tools=self._schemas)
        if not parsed and result.tool_calls:
            # 部署契约防呆:text 协议(经 API 时,如 sftgen 的外部采样)要求
            # 端点**不启用** tool-call parser——parser 会把调用文本剥出
            # content 塞进结构化 tool_calls,原始文本形态(训练/SFT 数据的
            # 本体)即告丢失。fail loud,勿静默兜底去读结构化表示。
            raise RuntimeError(
                "text protocol got structured tool_calls with no parseable "
                "call text — the serving endpoint has a tool-call parser "
                "enabled; disable it for the text/token channel"
            )
        return [
            ToolCall(
                id=f"text_{i}",
                name=c["name"],
                arguments=c["arguments"],
                raw_arguments=json.dumps(c["arguments"], ensure_ascii=False),
            )
            for i, c in enumerate(parsed)
        ]

    def canonicalize_assistant(self, result: ChatResult) -> dict[str, Any]:
        return {"role": "assistant", "content": result.text}

    def tool_result_message(self, call: ToolCall, content: str) -> dict[str, Any]:
        return {
            "role": "user",
            "content": core_protocol.tool_response_wrap(content),
        }

    def malformed_hint(self, result: ChatResult) -> str | None:
        text = result.text or ""
        if "<tool_call>" in text and not self.extract(result):
            d = core_protocol.fc_dialect_configured()
            if d == "qwen35_xml":
                return (
                    "Your message contains a <tool_call> marker but no "
                    "well-formed call could be parsed. Emit exactly one block "
                    "of the form:\n<tool_call>\n<function=tool-name>\n"
                    "<parameter=param-name>\nvalue\n</parameter>\n</function>\n"
                    "</tool_call>\nwith every tag closed."
                )
            return (
                "Your message contains a <tool_call> marker but no well-formed "
                "call could be parsed. Emit exactly:\n<tool_call>\n"
                '{"name": "<tool_name>", "arguments": {...}}\n</tool_call>\n'
                "with the JSON object on its own lines and the closing tag "
                "present. If argument values contain quotes or newlines, "
                "escape them as valid JSON."
            )
        return None


def make_tool_protocol(name: str) -> ToolProtocol:
    if name == "fncall":
        return FncallProtocol()
    if name == "text":
        return TextProtocol()
    raise ValueError(f"unknown tool protocol {name!r} (expected fncall|text)")
