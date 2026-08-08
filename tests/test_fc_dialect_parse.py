"""FC 协议家族化(M0):qwen35_xml 文本层解析器与 sglang 服务端 parser 的
逐语义等价 + dialect 探测/配置/分派件。

golden fixture(tests/fixtures/fc_qwen35_xml_golden.json)由 A800 上真实
sglang 0.5.10 Qwen3CoderDetector.detect_and_parse 生成(2026-08-08,
scratchpad gen_fc_golden.py)——我方解析器的唯一语义真源。任何解析行为
调整必须重新生成 golden,不得手改 fixture。
"""
import json
from pathlib import Path

import pytest

from trace_opd.core import protocol
from trace_opd.core.template_delta import fc_dialect, fc_roundtrip_probe

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "fc_qwen35_xml_golden.json").read_text()
)


@pytest.fixture(autouse=True)
def _reset_dialect():
    yield
    protocol.configure_fc_dialect("hermes")


# ---- golden 对拍(逐 case 全等) --------------------------------------

@pytest.mark.parametrize("case", GOLDEN["cases"], ids=[c["case"] for c in GOLDEN["cases"]])
def test_qwen35_xml_matches_sglang_golden(case):
    calls, normal = protocol._parse_qwen35_xml(case["text"], GOLDEN["tools"])
    golden_calls = [
        {"name": c["name"], "arguments": json.loads(c["parameters"])}
        for c in case["calls"]
    ]
    assert calls == golden_calls
    assert normal == case["normal_text"]


def test_qwen35_without_schema_keeps_strings():
    # sglang 语义:tools 缺席 ⇒ 全部参数原样 string(不猜类型)
    text = ("<tool_call>\n<function=run_experiment>\n<parameter=k>\n4\n"
            "</parameter>\n</function>\n</tool_call>")
    calls = protocol.parse_tool_calls(text, dialect="qwen35_xml")
    assert calls == [{"name": "run_experiment", "arguments": {"k": "4"}}]


# ---- dialect 配置与分派 -----------------------------------------------

def test_default_dialect_is_hermes_and_configurable():
    assert protocol.fc_dialect_configured() == "hermes"
    hermes_text = '<tool_call>{"name": "submit", "arguments": {}}</tool_call>'
    assert protocol.parse_tool_calls(hermes_text) == [{"name": "submit", "arguments": {}}]
    protocol.configure_fc_dialect("qwen35_xml")
    assert protocol.parse_tool_calls(hermes_text) == []  # hermes JSON 不是 XML 形态
    xml_text = ("<tool_call>\n<function=submit>\n</function>\n</tool_call>")
    assert protocol.parse_tool_calls(xml_text) == [{"name": "submit", "arguments": {}}]


def test_unknown_dialect_rejected_and_none_raises():
    with pytest.raises(ValueError):
        protocol.configure_fc_dialect("qwen4_yaml")
    with pytest.raises(RuntimeError):
        protocol.parse_tool_calls("anything", dialect="none")


def test_strip_tool_calls_dialect_semantics():
    xml = ("Reasoning first.\n<tool_call>\n<function=submit>\n</function>\n"
           "</tool_call>\ntrailing junk")
    # qwen35_xml:首块前文本;块后丢弃(与服务端 normal_text 一致)
    assert protocol.strip_tool_calls(xml, dialect="qwen35_xml") == "Reasoning first."
    # hermes:删块留余(历史语义;xml 块不匹配 hermes 正则故全保留)
    hermes = 'Thought.\n<tool_call>{"name": "submit", "arguments": {}}</tool_call>\ntail'
    assert protocol.strip_tool_calls(hermes, dialect="hermes") == "Thought.\n\ntail"


def test_fc_format_reminder_follows_dialect():
    assert '{"name"' in protocol.fc_format_reminder("a", "b", dialect="hermes")
    xml_reminder = protocol.fc_format_reminder("a", "b", dialect="qwen35_xml")
    # 示例必须用真实工具名(占位符会被模型照抄成 unknown-tool 循环)
    assert "<function=a>" in xml_reminder and "<parameter=param-name>" in xml_reminder
    protocol.configure_fc_dialect("qwen35_xml")
    assert "<function=a>" in protocol.fc_format_reminder("a")


# ---- fc_dialect 行为探测 + roundtrip 门 -------------------------------

class _FakeTemplateTok:
    """逐轮渲染的假模板:system Tools 块含一个示例调用(复刻 Qwen3.5 实测
    形态——roundtrip 必须差分出 assistant 段,整段解析会把示例当首个调用),
    assistant 的 tool_calls 按 style 渲染。"""

    def __init__(self, style):
        self.style = style

    def _call_block(self, name, arguments):
        if self.style == "qwen35_xml":
            args = "".join(
                f"<parameter={k}>\n{v}\n</parameter>\n"
                for k, v in arguments.items()
            )
            return f"<tool_call>\n<function={name}>\n{args}</function>\n</tool_call>"
        if self.style == "hermes":
            return ("<tool_call>\n"
                    + json.dumps({"name": name, "arguments": arguments})
                    + "\n</tool_call>")
        return "(no tool rendering)"

    def apply_chat_template(self, msgs, tools=None, tokenize=False,
                            add_generation_prompt=False, **kw):
        if self.style == "reject":
            raise TypeError("template does not accept tools")
        parts = []
        if tools is not None and self.style in ("qwen35_xml", "hermes"):
            example = self._call_block("example_function_name", {"p1": "value_1"})
            parts.append(f"<|im_start|>system\n# Tools\n{example}<|im_end|>\n")
        for m in msgs:
            if m["role"] == "assistant" and m.get("tool_calls"):
                fn = m["tool_calls"][0]["function"]
                block = self._call_block(fn["name"], fn["arguments"])
                parts.append(f"<|im_start|>assistant\n{block}<|im_end|>\n")
            else:
                parts.append(f"<|im_start|>{m['role']}\n{m.get('content', '')}<|im_end|>\n")
        return "".join(parts)


@pytest.mark.parametrize("style,expect", [
    ("qwen35_xml", "qwen35_xml"),
    ("hermes", "hermes"),
    ("drop", "none"),
    ("reject", "none"),
])
def test_fc_dialect_probe(style, expect):
    assert fc_dialect(_FakeTemplateTok(style)) == expect


@pytest.mark.parametrize("style", ["qwen35_xml", "hermes"])
def test_fc_roundtrip_probe_passes_matching_family(style):
    fc_roundtrip_probe(_FakeTemplateTok(style), style)


def test_fc_roundtrip_probe_fails_cross_family():
    # 模板渲染 XML、按 hermes 解析 ⇒ 必须 raise(两端形态不同族即断)
    with pytest.raises(RuntimeError):
        fc_roundtrip_probe(_FakeTemplateTok("qwen35_xml"), "hermes")
