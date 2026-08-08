"""M2:TextProtocol 升级为客户端解析 FC(§6.2 用户裁定 2026-08-08)。

关键行为:tools 交模板渲染(不再把 schema JSON 塞 system prompt)、extract
走家族分派(qwen35_xml/hermes)+ schema 类型还原、tool 结果用公共包裹件、
malformed 提醒随 dialect 措辞。
"""
import json

import pytest

from trace_opd.core import protocol as core_protocol
from trace_opd.research.toolproto import TextProtocol, make_tool_protocol

SCHEMAS = [{
    "type": "function",
    "function": {
        "name": "run_experiment",
        "description": "d",
        "parameters": {
            "type": "object",
            "properties": {"guidance": {"type": "string"},
                           "k": {"type": "integer"}},
            "required": ["guidance"],
        },
    },
}]


@pytest.fixture(autouse=True)
def _reset_dialect():
    yield
    core_protocol.configure_fc_dialect("hermes")


class _Result:
    def __init__(self, text, tool_calls=()):
        self.text = text
        self.content = text
        self.tool_calls = list(tool_calls)


def test_prepare_hands_tools_to_template_not_prompt():
    proto = TextProtocol()
    prompt, tools = proto.prepare("SYS.", SCHEMAS)
    assert tools is SCHEMAS                      # 交模板渲染(分布内形态)
    assert "run_experiment" not in prompt        # schema 不再手写进 system prompt
    assert "ONE tool call per message" in prompt  # 会话行为规约仍在
    assert "TOOL CALL FORMAT" not in prompt      # 旧格式教学段已废弃


def test_extract_qwen35_xml_with_schema_typing():
    core_protocol.configure_fc_dialect("qwen35_xml")
    proto = TextProtocol()
    proto.prepare("SYS.", SCHEMAS)
    calls = proto.extract(_Result(
        "Analysis.\n<tool_call>\n<function=run_experiment>\n"
        "<parameter=guidance>\ntry the fridge first\n</parameter>\n"
        "<parameter=k>\n2\n</parameter>\n</function>\n</tool_call>"
    ))
    assert len(calls) == 1
    assert calls[0].name == "run_experiment"
    assert calls[0].arguments == {"guidance": "try the fridge first", "k": 2}
    assert json.loads(calls[0].raw_arguments) == calls[0].arguments


def test_extract_hermes_default_backcompat():
    proto = TextProtocol()
    proto.prepare("SYS.", SCHEMAS)
    calls = proto.extract(_Result(
        '<tool_call>\n{"name": "run_experiment", "arguments": '
        '{"guidance": "g", "k": 1}}\n</tool_call>'
    ))
    assert calls and calls[0].arguments == {"guidance": "g", "k": 1}


def test_tool_result_uses_shared_wrap():
    proto = TextProtocol()
    msg = proto.tool_result_message(None, "RESULT BODY")
    assert msg == {"role": "user",
                   "content": core_protocol.tool_response_wrap("RESULT BODY")}


def test_malformed_hint_follows_dialect():
    proto = TextProtocol()
    proto.prepare("SYS.", SCHEMAS)
    broken = _Result("<tool_call>garbage")
    assert '{"name"' in proto.malformed_hint(broken)
    core_protocol.configure_fc_dialect("qwen35_xml")
    hint = proto.malformed_hint(broken)
    assert "<function=tool-name>" in hint and "<parameter=param-name>" in hint
    # 干净文本(无标记)不触发提醒
    assert proto.malformed_hint(_Result("plain thinking")) is None


def test_extract_fails_loud_when_endpoint_parser_stripped_calls():
    """部署契约:text 协议端点不得启用 tool-call parser——结构化 tool_calls
    非空而文本无调用 = 调用文本被剥,原始形态丢失,必须 fail loud。"""
    proto = TextProtocol()
    proto.prepare("SYS.", SCHEMAS)
    stripped = _Result("only reasoning text", tool_calls=[object()])
    with pytest.raises(RuntimeError, match="tool-call parser"):
        proto.extract(stripped)


def test_make_tool_protocol_returns_fresh_instances():
    # prepare 存 schema 于实例(extract 的类型还原依赖它)——实例必须每
    # session 新建,不得共享
    a, b = make_tool_protocol("text"), make_tool_protocol("text")
    assert a is not b
