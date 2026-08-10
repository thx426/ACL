import json

import pytest

from fakes import FakeTokenizer, ScriptedAdapter, ScriptedGenerate, reply

from trace_opd.common.config import NarrationCfg, ResearchCfg
from trace_opd.common.llm import ChatResult, ToolCall
from trace_opd.core.template_delta import TemplateDelta
from trace_opd.research.session import ResearchSession
from trace_opd.research.toolproto import FncallProtocol
from trace_opd.research.tools import ToolRouter

pytestmark = pytest.mark.asyncio

TOK = FakeTokenizer()
DELTA = TemplateDelta(TOK)

GOOD_DOC = """# Understanding & Approach
Go directly to the target and act on it.

# Failure Modes & Remedies
WHEN the room looks empty DO look again. (verified by experiment)

# Facts & Reusable Notes
The engine says 'Nothing happens.' for invalid commands."""


async def flat_narration(messages, temperature, max_tokens):
    return "The agent looked around once. The attempt failed after 1 steps."


def make_router(adapter=None, *, n_exp=12, narrate=flat_narration, action="look"):
    # action="do the thing" 命中 ScriptedAdapter 默认 win_on ⇒ 实验会成功
    # (零正例文档规则的测试杠杆)。
    cfg = ResearchCfg(n_exp=n_exp, k_default=2, k_max=4)
    return ToolRouter(
        adapter=adapter or ScriptedAdapter(),
        research=cfg,
        narration=NarrationCfg(backend="external"),
        generate_factory=lambda: ScriptedGenerate(TOK, [reply("try", action)]),
        tokenizer=TOK,
        delta=DELTA,
        episode_response_budget=100000,
        narrate_fn=narrate,
    )


def tc(name, args, cid="c1"):
    return ToolCall(id=cid, name=name, arguments=args,
                    raw_arguments=json.dumps(args))


class ScriptedChat:
    """ResearcherChatFn double: yields scripted ChatResults in order."""

    def __init__(self, results):
        self._results = list(results)
        self.seen_messages = []

    async def __call__(self, messages, tools):
        self.seen_messages.append([dict(m) for m in messages])
        return self._results.pop(0)


def assistant_with_call(text, call: ToolCall):
    return ChatResult(
        content=text,
        tool_calls=[call],
        raw_message={
            "role": "assistant",
            "content": text,
            "tool_calls": [
                {"id": call.id,
                 "function": {"name": call.name, "arguments": call.raw_arguments}}
            ],
        },
    )


# ------------------------------------------------------------------- router


async def test_run_experiment_and_budget_accounting():
    router = make_router()
    out = await router.dispatch(tc("run_experiment", {"guidance": "look around", "k": 2}))
    assert "[Experiment 1/12]" in out
    assert "FACT CARD" in out and "NARRATION:" in out
    assert "Summary: 0/2 SUCCESS" in out
    st = router.state
    assert st.experiments_used == 1 and st.episodes_consumed == 2
    assert st.last_exp_attempts == 2 and st.last_exp_successes == 0
    assert len(st.guidance_log) == 1 and st.guidance_log[0]["k"] == 2


async def test_k_clamped():
    router = make_router()
    out = await router.dispatch(tc("run_experiment", {"guidance": "", "k": 99}))
    assert "k adjusted from 99 to 4" in out


async def test_budget_exhaustion_soft_refusal():
    router = make_router(n_exp=1)
    await router.dispatch(tc("run_experiment", {"guidance": ""}))
    out = await router.dispatch(tc("run_experiment", {"guidance": ""}))
    assert "budget exhausted" in out
    assert router.state.experiments_used == 1


async def test_env_error_refund():
    router = make_router(adapter=ScriptedAdapter(fail_open=True))
    out = await router.dispatch(tc("run_experiment", {"guidance": ""}))
    assert "refunded" in out
    assert router.state.experiments_used == 0


async def test_bad_arguments_spend_nothing():
    router = make_router()
    call = ToolCall(id="c1", name="run_experiment", arguments={},
                    raw_arguments='{"k": 4\nparameter>}')
    out = await router.dispatch(call)
    assert "TOOL ARGUMENTS ERROR" in out
    assert router.state.experiments_used == 0


async def test_fetch_raw_flow():
    router = make_router()
    await router.dispatch(tc("run_experiment", {"guidance": ""}))
    out = await router.dispatch(tc("fetch_raw", {"episode_id": "ep001"}))
    assert "RAW LOG of ep001" in out and "fetch_raw calls used: 1/20" in out
    missing = await router.dispatch(tc("fetch_raw", {"episode_id": "nope"}))
    assert "Unknown episode_id" in missing


async def test_finalize_doc_checks_and_revision():
    router = make_router()
    bad_doc = GOOD_DOC + "\nThis helps the teacher during training."
    out = await router.dispatch(tc("finalize", {"experience_doc": bad_doc}))
    assert "revise" in out and not router.state.finalized
    out = await router.dispatch(tc("finalize", {"experience_doc": bad_doc}))
    assert router.state.finalized and router.state.flags
    assert "flags recorded" in out


async def test_finalize_clean_doc():
    router = make_router()
    # GOOD_DOC 声称 verified —— 2026-08-10 零正例规则下须会话真有成功
    # 才合法;模拟一次成功后走原干净接受路径。
    router.state.exp_successes_total = 1
    router.state.exp_attempts_total = 2
    out = await router.dispatch(
        tc("finalize", {"experience_doc": GOOD_DOC, "self_assessment": "stable"})
    )
    assert "accepted" in out and router.state.finalized
    assert router.state.doc == GOOD_DOC and not router.state.flags


ZERO_SUCCESS_DOC = """# Current Understanding & Ruled-Out Directions
The task needs the target object; going left was ruled out (wall, 3 runs).

# Failure Modes & Lessons
WHEN the room looks empty DO look again before moving (untested next step).

# Facts & Reusable Notes
The engine says 'Nothing happens.' for invalid commands."""


async def test_finalize_zero_success_rejects_verified_claims():
    # 2026-08-10 用户裁定:零成功会话交 "verified" 解法 → 一次 revision
    # 指引;原样重交则带 flags 接受(现有软质检回路语义)。
    router = make_router()
    out = await router.dispatch(
        tc("finalize", {"experience_doc": GOOD_DOC, "self_assessment": "stable"})
    )
    assert "revise" in out and "zero" in out
    assert not router.state.finalized and router.state.revision_requested
    out2 = await router.dispatch(
        tc("finalize", {"experience_doc": GOOD_DOC, "self_assessment": "stable"})
    )
    assert "accepted" in out2 and router.state.finalized
    assert any("zero" in f for f in router.state.flags)


async def test_finalize_zero_success_lessons_doc_clean():
    # 合法出口:零成功 + 教训型文档(无 verified/confirmed 断言)干净接受。
    router = make_router()
    out = await router.dispatch(
        tc("finalize", {"experience_doc": ZERO_SUCCESS_DOC, "self_assessment": "nothing worked"})
    )
    assert "accepted" in out and router.state.finalized
    assert not router.state.flags


async def test_unknown_tool_soft():
    router = make_router()
    out = await router.dispatch(tc("teleport", {}))
    assert "Unknown tool" in out


# ------------------------------------------------------------------ session


async def test_full_session_flow():
    # executor 动作命中 win_on ⇒ 实验成功,GOOD_DOC 的 verified 断言在
    # 零正例规则下合法(会话真有成功)。
    router = make_router(action="do the thing")
    chat = ScriptedChat(
        [
            assistant_with_call(
                "Baseline first.", tc("run_experiment", {"guidance": "", "k": 2})
            ),
            assistant_with_call(
                "Enough evidence.",
                tc("finalize", {"experience_doc": GOOD_DOC, "self_assessment": "ok"}, "c2"),
            ),
        ]
    )
    session = ResearchSession(
        researcher_chat=chat,
        toolproto=FncallProtocol(),
        router=router,
        system_prompt="SYS",
        max_turns=10,
    )
    result = await session.run()
    assert result.finalized and result.doc == GOOD_DOC
    assert result.turns_used == 2
    assert result.experiments_used == 1 and result.episodes_consumed == 2
    # 会话累计成败透出(2026-08-10 零正例裁定的机器判据)
    assert result.exp_successes_total == 2 and result.exp_attempts_total == 2
    # message record: system, kickoff, assistant, tool, assistant, tool
    roles = [m["role"] for m in result.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert result.guidance_log and result.guidance_log[0]["experiment_idx"] == 1


async def test_session_no_finalize_at_turn_limit():
    router = make_router()
    chat = ScriptedChat([ChatResult(content="thinking...")] * 3)
    session = ResearchSession(
        researcher_chat=chat,
        toolproto=FncallProtocol(),
        router=router,
        system_prompt="SYS",
        max_turns=3,
        wrapup_buffer=1,
    )
    result = await session.run()
    assert result.no_finalize and not result.finalized and result.doc is None
    # wrap-up reminder was injected as a user message
    assert any(
        m["role"] == "user" and "nearing its turn limit" in str(m.get("content"))
        for m in result.messages
    )
