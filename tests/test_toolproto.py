import json

from trace_opd.common.llm import ChatResult, ToolCall
from trace_opd.research.prompts import TOOL_SCHEMAS
from trace_opd.research.toolproto import FncallProtocol, TextProtocol, make_tool_protocol


def test_factory():
    assert isinstance(make_tool_protocol("fncall"), FncallProtocol)
    assert isinstance(make_tool_protocol("text"), TextProtocol)
    try:
        make_tool_protocol("nope")
        assert False
    except ValueError:
        pass


def test_fncall_prepare_passthrough():
    proto = FncallProtocol()
    system, tools = proto.prepare("SYS", TOOL_SCHEMAS)
    assert system == "SYS" and tools is TOOL_SCHEMAS


def test_fncall_canonicalize_rewrites_broken_arguments():
    proto = FncallProtocol()
    raw_msg = {
        "role": "assistant",
        "content": "analysis",
        "tool_calls": [
            {"id": "c1", "function": {"name": "run_experiment",
                                      "arguments": '{"k": 4\nparameter>}'}}
        ],
    }
    parsed = [ToolCall(id="c1", name="run_experiment", arguments={},
                       raw_arguments='{"k": 4\nparameter>}')]
    result = ChatResult(content="analysis", tool_calls=parsed, raw_message=raw_msg)
    msg = proto.canonicalize_assistant(result)
    assert msg["tool_calls"][0]["function"]["arguments"] == "{}"


def test_fncall_tool_result_message():
    proto = FncallProtocol()
    msg = proto.tool_result_message(ToolCall(id="c9", name="x", arguments={}), "OUT")
    assert msg == {"role": "tool", "tool_call_id": "c9", "content": "OUT"}


def test_text_prepare_folds_tool_docs():
    proto = TextProtocol()
    system, tools = proto.prepare("SYS", TOOL_SCHEMAS)
    assert tools is None
    assert system.startswith("SYS")
    assert "<tool_call>" in system
    assert "run_experiment" in system and "finalize" in system


def test_text_extract_good_and_bad_json():
    proto = TextProtocol()
    good = json.dumps({"name": "fetch_raw", "arguments": {"episode_id": "ep001"}})
    text = f"analysis first\n<tool_call>\n{good}\n</tool_call>"
    calls = proto.extract(ChatResult(content=text))
    assert len(calls) == 1
    assert calls[0].name == "fetch_raw"
    assert calls[0].arguments == {"episode_id": "ep001"}

    bad = "<tool_call>{not json}</tool_call>"
    calls = proto.extract(ChatResult(content=bad))
    assert len(calls) == 1
    assert calls[0].arguments == {} and calls[0].raw_arguments


def test_text_extract_none_when_absent():
    proto = TextProtocol()
    assert proto.extract(ChatResult(content="just thinking aloud")) == []


def test_text_tool_result_wrapped():
    proto = TextProtocol()
    msg = proto.tool_result_message(ToolCall(id="t0", name="x", arguments={}), "OUT")
    assert msg["role"] == "user"
    assert "<tool_response>" in msg["content"] and "OUT" in msg["content"]
