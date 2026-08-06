"""Review-driven research-session edge cases (P1-7 partial, P2-5 single call)."""
import json

import pytest

from fakes import FakeTokenizer, ScriptedAdapter, ScriptedGenerate, reply
from test_research import (
    GOOD_DOC,
    ScriptedChat,
    assistant_with_call,
    flat_narration,
    make_router,
    tc,
)

from trace_opd.common.llm import ChatResult, ToolCall
from trace_opd.research.session import ResearchSession
from trace_opd.research.toolproto import FncallProtocol, TextProtocol

pytestmark = pytest.mark.asyncio


async def test_mid_session_failure_returns_partial_record():
    """An HTTP failure after real work must keep the conversation, guidance
    log and true counters (review P1-7)."""
    router = make_router()

    calls = {"n": 0}

    async def flaky_chat(messages, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            return assistant_with_call(
                "Baseline.", tc("run_experiment", {"guidance": "", "k": 2})
            )
        raise RuntimeError("connection reset by peer")

    session = ResearchSession(
        researcher_chat=flaky_chat,
        toolproto=FncallProtocol(),
        router=router,
        system_prompt="SYS",
        max_turns=10,
    )
    result = await session.run()  # must NOT raise
    assert result.error and "connection reset" in result.error
    assert result.doc is None and result.no_finalize
    assert result.experiments_used == 1 and result.episodes_consumed == 2
    assert result.guidance_log  # the guidance actually issued survives
    roles = [m["role"] for m in result.messages]
    assert roles == ["system", "user", "assistant", "tool"]  # full partial record


async def test_snapshot_available_after_external_timeout():
    """The timeout owner (agent loop) recovers the partial state via
    session.snapshot() even though run() was cancelled."""
    router = make_router()
    session = ResearchSession(
        researcher_chat=ScriptedChat([ChatResult(content="thinking")] * 50),
        toolproto=FncallProtocol(),
        router=router,
        system_prompt="SYS",
        max_turns=3,
    )
    await session.run()
    snap = session.snapshot(error="deadline")
    assert snap.error == "deadline"
    assert snap.turns_used == 3
    assert len(snap.messages) == len(session.messages)


async def test_single_call_per_message_enforced():
    """Two calls in one assistant turn: only the first executes; the model is
    told the rest were ignored (review P2-5)."""
    router = make_router()
    two_calls = ChatResult(
        content="double spend",
        tool_calls=[
            tc("run_experiment", {"guidance": "a"}, "c1"),
            tc("run_experiment", {"guidance": "b"}, "c2"),
        ],
        raw_message={"role": "assistant", "content": "double spend"},
    )
    finalize = assistant_with_call(
        "done", tc("finalize", {"experience_doc": GOOD_DOC}, "c3")
    )
    session = ResearchSession(
        researcher_chat=ScriptedChat([two_calls, finalize]),
        toolproto=FncallProtocol(),
        router=router,
        system_prompt="SYS",
        max_turns=10,
    )
    result = await session.run()
    assert result.experiments_used == 1  # second call never ran
    tool_msgs = [m for m in result.messages if m["role"] == "tool"]
    assert "IGNORED" in tool_msgs[0]["content"]
    assert result.finalized


async def test_text_protocol_malformed_framing_gets_hint():
    router = make_router()
    broken = ChatResult(content='I will call now\n<tool_call>\n{"name": "run_experiment"')
    finalize_json = json.dumps(
        {"name": "finalize", "arguments": {"experience_doc": GOOD_DOC}}
    )
    good = ChatResult(content=f"<tool_call>\n{finalize_json}\n</tool_call>")
    session = ResearchSession(
        researcher_chat=ScriptedChat([broken, good]),
        toolproto=TextProtocol(),
        router=router,
        system_prompt="SYS",
        max_turns=10,
    )
    result = await session.run()
    assert result.finalized
    hints = [
        m for m in result.messages
        if m["role"] == "user" and "no well-formed" in str(m.get("content", ""))
    ]
    assert len(hints) == 1


async def test_text_protocol_legal_empty_arguments_not_refused():
    """arguments:{} parsed OK must not be mistaken for a JSON failure."""
    proto = TextProtocol()
    raw = json.dumps({"name": "fetch_raw", "arguments": {}})
    calls = proto.extract(ChatResult(content=f"<tool_call>{raw}</tool_call>"))
    assert calls[0].arguments == {}
    assert calls[0].raw_arguments == "{}"  # canonical dump, not the raw call JSON
    router = make_router()
    out = await router.dispatch(calls[0])
    assert "TOOL ARGUMENTS ERROR" not in out  # reaches the tool (which asks for episode_id)


async def test_session_accumulates_researcher_usage():
    """Per-call usage sums into the SessionResult (SFT filtering signal) and
    survives a mid-session failure via the partial snapshot."""
    router = make_router()

    calls = {"n": 0}

    async def chat(messages, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            r = assistant_with_call(
                "Baseline.", tc("run_experiment", {"guidance": "", "k": 2})
            )
            r.prompt_tokens, r.completion_tokens = 1000, 200
            return r
        raise RuntimeError("gone")

    session = ResearchSession(
        researcher_chat=chat,
        toolproto=FncallProtocol(),
        router=router,
        system_prompt="SYS",
        max_turns=10,
    )
    result = await session.run()
    assert result.error and result.prompt_tokens == 1000
    assert result.completion_tokens == 200
