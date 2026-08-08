"""Adapter identity fields (review P1-5) and evaluator summary discipline
(review P1-9) — no engine, no server."""
import threading

from trace_opd.common.config import AlfworldCfg
from trace_opd.core.types import OUTCOME_ENV_ERROR, OUTCOME_FAILED, OUTCOME_SUCCESS
from trace_opd.envs.alfworld.adapter import AlfworldTaskAdapter
from trace_opd.evalm.run_eval import summarize


def _adapter(task_id: str, protocol: str = "text") -> AlfworldTaskAdapter:
    # 既有断言守护 text 协议路径(迁移前行为锁);FC 路径的断言显式传 "fncall"。
    return AlfworldTaskAdapter(
        {
            "task_id": task_id,
            "gamefile": "x/game.tw-pddl",
            "initial_obs": "You are in a kitchen. Your task is to: heat some egg.",
            "gt_text": "1. GotoLocation(fridge)",
        },
        AlfworldCfg(protocol=protocol, data_root="/d", worker_python="/p"),
        spawn_gate=threading.Semaphore(1),
    )


def test_task_type_skips_split_segment():
    a = _adapter("train/pick_heat_then_place_in_recep-Egg-None-Fridge-13/trial_T123")
    assert a.task_type == "pick_heat_then_place_in_recep"
    # ids without a split prefix still resolve
    b = _adapter("pick_and_place_simple-Vase-None-Shelf-1/trial_T9")
    assert b.task_type == "pick_and_place_simple"


def test_adapter_contract_surface_complete():
    """通用性契约:两个 TaskAdapter 实现都必须具备 v4 新增的三个成员
    (task_type / observation_indicates_invalid / format_reminder)。"""
    from fakes import ScriptedAdapter

    for impl in (_adapter("train/pick-Egg-1/trial_T1"), ScriptedAdapter()):
        assert isinstance(impl.task_type, str) and impl.task_type
        assert impl.observation_indicates_invalid("Nothing happens.") is True
        assert impl.observation_indicates_invalid("You open the fridge.") is False
        assert "<action>" in impl.format_reminder()
        assert impl.tools() is None    # 文本动作协议:无原生 FC 工具


def test_alfworld_format_reminder_byte_identical_to_pre_migration():
    """行为锁:提示词材料迁入 adapter 后,发给模型的格式提醒必须与迁移前
    core/protocol.FORMAT_REMINDER_OBS 逐字节一致(s4 行为零变化)。"""
    a = _adapter("train/pick-Egg-1/trial_T1")
    assert a.format_reminder() == (
        "Nothing happens. (Your reply had no <action>...</action> block — "
        "respond with <reasoning>...</reasoning> then <action>one command</action>.)"
    )


def test_task_brief_carries_real_type_and_scene():
    a = _adapter("train/look_at_obj_in_light-Book-None-DeskLamp-308/trial_T1")
    brief = a.task_brief()
    assert "Task type: look_at_obj_in_light" in brief
    assert "heat some egg" in brief
    assert a.gt_text().startswith("1. GotoLocation")


def _row(split, outcome, won, steps=10):
    return {"split": split, "outcome": outcome, "won": won, "env_steps": steps}


def test_summarize_excludes_env_errors_from_denominator():
    rows = [
        _row("valid_seen", OUTCOME_SUCCESS, True),
        _row("valid_seen", OUTCOME_FAILED, False),
        _row("valid_seen", OUTCOME_ENV_ERROR, False),
        _row("valid_unseen", OUTCOME_SUCCESS, True),
    ]
    summary, incomplete = summarize(rows, "bare")
    seen = summary["splits"]["valid_seen"]
    assert seen["total"] == 3 and seen["attempts"] == 2
    assert seen["success_rate"] == 0.5          # 1/2, NOT 1/3
    assert seen["env_errors"] == 1
    assert incomplete and summary["incomplete"]


def test_summarize_clean_run_is_complete():
    rows = [_row("valid_unseen", OUTCOME_FAILED, False)]
    summary, incomplete = summarize(rows, "bare")
    assert not incomplete
    assert summary["splits"]["valid_unseen"]["success_rate"] == 0.0


def test_alfworld_fncall_surface():
    """FC 协议面(2026-08-08 用户裁定的默认):单工具 execute、FC 提醒、
    scaffold 不再教 <reasoning>/<action>,而命令语法照旧在场。"""
    a = _adapter("train/pick-Egg-1/trial_T1", protocol="fncall")
    tools = a.tools()
    assert [t["function"]["name"] for t in tools] == ["execute"]
    assert "command" in tools[0]["function"]["parameters"]["properties"]
    scaffold = a.executor_scaffold()
    assert "<action>" not in scaffold and "execute" in scaffold
    assert "Available commands:" in scaffold
    rem = a.format_reminder()
    assert rem.startswith("Nothing happens. ") and "execute" in rem
    assert "via the execute(command) tool" in a.task_brief()


def test_alfworld_fncall_react_paths():
    """FC react:合法调用执行并以 tool 响应包裹观察;unknown tool/空命令
    不消耗 env step;无调用走格式提醒(裸文本)。"""
    from trace_opd.core import protocol as proto

    class _Env:
        def __init__(self):
            self.stepped = []
        def step(self, cmd):
            self.stepped.append(cmd)
            return "You arrive at desk 1.", False, False

    a = _adapter("train/pick-Egg-1/trial_T1", protocol="fncall")
    env = _Env()
    ok = a.react(env, 'Reasoning.\n<tool_call>{"name": "execute", "arguments": {"command": "Go to desk 1"}}</tool_call>')
    assert ok.acted and ok.consumed_env_step and env.stepped == ["go to desk 1"]
    assert ok.feedback == proto.tool_response_wrap("You arrive at desk 1.")
    assert ok.thought == "Reasoning."
    # unknown/空调用与 math/SSB 一致:发起调用即耗步(2026-08-08 审阅——
    # 不耗步 + acted=True 重置 breaker 会让坏调用零成本烧满 40 轮)
    bad_tool = a.react(env, '<tool_call>{"name": "teleport", "arguments": {}}</tool_call>')
    assert bad_tool.acted and bad_tool.consumed_env_step and "unknown tool" in bad_tool.feedback
    empty = a.react(env, '<tool_call>{"name": "execute", "arguments": {"command": "  "}}</tool_call>')
    assert empty.acted and empty.consumed_env_step and "empty command" in empty.feedback
    none = a.react(env, "no call at all")
    assert not none.acted and none.feedback.startswith("Nothing happens. ")
    assert len(env.stepped) == 1
