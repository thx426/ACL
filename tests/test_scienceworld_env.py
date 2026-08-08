"""ScienceWorld 引入(2026-08-08 用户五裁定):adapter 面单测(fake env,
无 JVM)。终局二元 reward / FC 单工具 / electrical 动作表附加 / ETO indices
守卫。真引擎端到端由本地冒烟脚本与 preflight probe 覆盖。"""
import json
import threading
from pathlib import Path

from trace_opd.common.config import SciworldCfg
from trace_opd.core import protocol as proto
from trace_opd.envs.scienceworld.adapter import (
    COMMAND_REFERENCE,
    INVALID_ACTION_OBS,
    SciworldTaskAdapter,
    _is_electrical,
)
from trace_opd.envs.scienceworld.build_dataset import EXPECTED, render_gold


def _adapter(task_name: str = "task-1-boil") -> SciworldTaskAdapter:
    return SciworldTaskAdapter(
        {
            "task_id": f"{task_name}/12",
            "task_name": task_name,
            "variation_idx": 12,
            "initial_obs": "Task: boil water.\n\nThis room is called the hallway.",
            "gt_text": "1. open door to kitchen\n2. go to kitchen",
        },
        SciworldCfg(worker_python="/p"),
        spawn_gate=threading.Semaphore(1),
    )


class _Env:
    def __init__(self, script):
        self.script = list(script)
        self.stepped = []
        self.last_score = 0

    def step(self, cmd):
        self.stepped.append(cmd)
        obs, done, won, score = self.script.pop(0)
        self.last_score = score
        return obs, done, won


def test_identity_and_surfaces():
    a = _adapter()
    assert a.task_type == "task-1-boil"
    assert a.tools()[0]["function"]["name"] == "execute"
    scaffold = a.executor_scaffold()
    assert "Available commands:" in scaffold and "focus on OBJ" in scaffold
    assert "connect OBJ" not in scaffold          # 非 electrical 无附加行
    assert "boil water" in a.task_brief() and "SCORED" in a.task_brief()
    assert a.gt_text().startswith("1. open door")
    assert a.observation_indicates_invalid(INVALID_ACTION_OBS)
    assert a.format_reminder().startswith(INVALID_ACTION_OBS)


def test_electrical_tasks_get_connect_rows():
    for name in ("task-2-power-component", "task-2a-test-conductivity-of-unknown-substances"):
        assert _is_electrical(name)
        assert "connect OBJ to OBJ" in _adapter(name).executor_scaffold()
    assert not _is_electrical("task-1-boil")
    assert "connect" not in COMMAND_REFERENCE


def test_react_binary_reward_and_step_semantics():
    a = _adapter()
    env = _Env([
        ("You teleport to the kitchen.", False, False, 0),
        ("The task is accomplished.", True, True, 100),
    ])
    ok = a.react(env, '<tool_call>{"name": "execute", "arguments": {"command": "Teleport to kitchen"}}</tool_call>')
    assert ok.acted and ok.consumed_env_step and env.stepped == ["teleport to kitchen"]
    assert ok.score == 0.0 and not ok.done
    assert ok.feedback == proto.tool_response_wrap("You teleport to the kitchen.")
    win = a.react(env, '<tool_call>{"name": "execute", "arguments": {"command": "focus on water"}}</tool_call>')
    # 终局二元(裁定):won → score=1.0;引擎部分分只在 handle.last_score 观察面
    assert win.done and win.score == 1.0
    # 失败终局(负分/超限):won=False → score=0.0
    env2 = _Env([("Task failed.", True, False, -100)])
    lose = a.react(env2, '<tool_call>{"name": "execute", "arguments": {"command": "focus on sink"}}</tool_call>')
    assert lose.done and lose.score == 0.0


def test_react_bad_calls_consume_steps():
    a = _adapter()
    env = _Env([("x", False, False, 0)])
    bad = a.react(env, '<tool_call>{"name": "submit", "arguments": {}}</tool_call>')
    assert bad.acted and bad.consumed_env_step and "unknown tool" in bad.feedback
    empty = a.react(env, '<tool_call>{"name": "execute", "arguments": {"command": " "}}</tool_call>')
    assert empty.acted and empty.consumed_env_step and "empty command" in empty.feedback
    none = a.react(env, "no call here")
    assert not none.acted and none.feedback.startswith(INVALID_ACTION_OBS)
    assert env.stepped == []  # 坏调用耗预算步但不碰引擎


def test_vendored_eto_indices_guard():
    data = json.loads(
        (Path("src/trace_opd/envs/scienceworld/data/eto_indices.json")).read_text()
    )
    for split, n in EXPECTED.items():
        assert len(data[split]) == n
        tasks = {p[0] for p in data[split]}
        assert len(tasks) == 24            # ETO 实际任务集(实测定案,非 30)
    assert "ETO" in data["_source"]


def test_render_gold():
    assert render_gold(["a", "b"]) == "1. a\n2. b"
    assert render_gold(None) == "" and render_gold([]) == ""
