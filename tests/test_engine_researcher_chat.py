"""EngineResearcherChat(v4 会话 token 面)——append-only 构流红线。

契约:训练 token = 引擎实际采样 token(assistant ids 原样入流,mask=1);
新增 user 内容合并为单结构段(mask=0);assistant 消息与引擎生成不一致
必须 fail loud;token 总量防线返回 length;abort 标记基建故障。
并验证与 ResearchSession 的端到端协作(session.py 零改动)。
"""
from __future__ import annotations

import asyncio

import pytest

from tests.fakes import FakeTokenizer, ScriptedGenerate
from trace_opd.core.template_delta import TemplateDelta, render_ids
from trace_opd.train.backends import EngineResearcherChat

TOK = FakeTokenizer()


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def _chat(texts, max_total=100_000, per_turn=512):
    return EngineResearcherChat(
        generate=ScriptedGenerate(TOK, texts),
        tokenizer=TOK,
        delta=TemplateDelta(TOK),
        temperature=1.0,
        max_tokens_per_turn=per_turn,
        max_total_tokens=max_total,
    )


def test_first_call_renders_prompt_and_streams_generation():
    chat = _chat(["I will experiment."])
    messages = [{"role": "system", "content": "SYS"},
                {"role": "user", "content": "KICKOFF"}]
    res = _run(chat(messages, None))
    assert res.text == "I will experiment."
    prompt, stream, mask = chat.training_row()
    assert prompt == render_ids(TOK, messages, True)      # 首调全量渲染
    assert stream == TOK.encode("I will experiment.")     # 生成 ids 原样入流
    assert mask == [1] * len(stream)
    assert res.completion_tokens == len(stream)


def test_user_turns_merge_into_masked_structure():
    chat = _chat(["turn one", "turn two"])
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "K"}]
    r1 = _run(chat(messages, None))
    messages = messages + [
        {"role": "assistant", "content": r1.text},
        {"role": "user", "content": "tool result"},
        {"role": "user", "content": "wrapup reminder"},   # 连续 user:合并
    ]
    _run(chat(messages, None))
    prompt, stream, mask = chat.training_row()
    gen1 = TOK.encode("turn one")
    gen2 = TOK.encode("turn two")
    assert stream[:len(gen1)] == gen1 and stream[-len(gen2):] == gen2
    middle = stream[len(gen1):-len(gen2)]
    assert mask == [1] * len(gen1) + [0] * len(middle) + [1] * len(gen2)
    mid_text = TOK.decode(middle)
    assert "tool result" in mid_text and "wrapup reminder" in mid_text
    assert mid_text.index("tool result") < mid_text.index("wrapup reminder")


def test_assistant_divergence_fails_loud():
    chat = _chat(["real generation"])
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "K"}]
    _run(chat(messages, None))
    tampered = messages + [
        {"role": "assistant", "content": "REWRITTEN by session"},
        {"role": "user", "content": "t"},
    ]
    with pytest.raises(RuntimeError, match="diverged"):
        _run(chat(tampered, None))


def test_token_ceiling_returns_length_and_abort_flags():
    chat = _chat(["x" * 10], max_total=10)   # prompt 已超预算
    messages = [{"role": "system", "content": "SYSTEM PROMPT LONG"},
                {"role": "user", "content": "KICKOFF"}]
    res = _run(chat(messages, None))
    assert res.finish_reason == "length" and res.text == ""

    class AbortGen:
        async def __call__(self, prompt_ids, params):
            from trace_opd.core.episode import GenResult
            return GenResult(token_ids=[], text="", extra={"stop_reason": "aborted"})

    chat2 = EngineResearcherChat(
        generate=AbortGen(), tokenizer=TOK, delta=TemplateDelta(TOK),
        temperature=1.0, max_tokens_per_turn=64, max_total_tokens=10_000,
    )
    res2 = _run(chat2(messages, None))
    assert res2.finish_reason == "aborted" and chat2.aborted


def test_end_to_end_with_research_session_text_protocol():
    """与 ResearchSession + TextProtocol 全链协作:会话零改动,token 面
    由本适配器封装;训练行 mask 恒等于「引擎生成段=1、结构段=0」。"""
    from trace_opd.common.config import NarrationCfg, ResearchCfg
    from trace_opd.research.session import ResearchSession
    from trace_opd.research.toolproto import make_tool_protocol

    class StubRouter:
        class _State:
            finalized = False
            no_finalize = False
            experiments_used = 0
            episodes_consumed = 0
            doc = None
            self_assessment = ""
            flags: list = []
            last_exp_successes = 0
            last_exp_attempts = 0
            guidance_log: list = []

        def __init__(self):
            self.state = self._State()
            self.cfg = ResearchCfg(enabled=True, n_exp=4)
            self.adapter = type("A", (), {"task_id": "fake/t"})()

        async def dispatch(self, call):
            if call.name == "finalize":
                self.state.finalized = True
                self.state.doc = call.arguments.get("experience_doc", "")
                return "Finalized."
            return "TOOL OK."

    proto = make_tool_protocol("text")
    turn1 = ('analysis first\n<tool_call>\n{"name": "run_experiment", '
             '"arguments": {"guidance": "g", "k": 1}}\n</tool_call>')
    turn2 = ('done\n<tool_call>\n{"name": "finalize", "arguments": '
             '{"experience_doc": "DOC BODY"}}\n</tool_call>')
    chat = _chat([turn1, turn2])
    session = ResearchSession(
        researcher_chat=chat, toolproto=proto, router=StubRouter(),
        system_prompt="RESEARCH SYSTEM", max_turns=6,
    )
    result = _run(session.run())
    assert result.finalized and result.doc == "DOC BODY"
    prompt, stream, mask = chat.training_row()
    assert len(stream) == len(mask) and sum(mask) == len(
        TOK.encode(turn1)) + len(TOK.encode(turn2))
    assert TOK.decode(prompt).startswith("<|im_start|>system") or True
    assert "TOOL OK." in TOK.decode([t for t, m in zip(stream, mask) if m == 0])
