"""AppWorld 环境:worker 协议 driver、fenced-code react、部分分与红线。

本地无 appworld 包(py3.11 服务器专属)——用 stub worker 讲同一 JSON
协议锁定 driver 行为;真引擎端到端归服务器冒烟(GO 清单)。
"""
from __future__ import annotations

import asyncio
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from fakes import FakeTokenizer, ScriptedGenerate
from trace_opd.common.config import AppworldCfg, load_config, objective_fingerprint
from trace_opd.core.episode import run_episode
from trace_opd.core.template_delta import TemplateDelta
from trace_opd.envs import registry
from trace_opd.envs.appworld.adapter import (
    FORMAT_REMINDER,
    SYSTEM_PROMPT,
    AppworldEnvFactory,
    AppworldTaskAdapter,
    initial_obs_of,
)

TOK = FakeTokenizer()
DELTA = TemplateDelta(TOK)

#: stub worker:讲真 worker 的 JSON lines 协议;execute 回显,代码含
#: complete_task ⇒ completed;evaluate 按环境变量控制 success/report。
_STUB = textwrap.dedent("""
    import json, os, sys
    print(json.dumps({
        "event": "ready",
        "instruction": "Find the spotify password.",
        "supervisor": {"first_name": "Alex", "last_name": "Kim",
                       "email": "alex@x.io", "phone_number": "555"},
        "metadata": {"difficulty": 2},
    }), flush=True)
    mode = os.environ.get("STUB_EVAL", "full")
    for line in sys.stdin:
        if line.strip() == "__CLOSE__":
            break
        req = json.loads(line)
        if req["op"] == "execute":
            code = req["code"]
            print(json.dumps({
                "event": "execute_result",
                "output": "OUT<" + code.strip().splitlines()[0][:60] + ">",
                "completed": "complete_task" in code,
            }), flush=True)
        elif req["op"] == "evaluate":
            if mode == "full":
                ev = {"success": True, "report": {"passes": [1, 2, 3], "failures": []}}
            else:
                ev = {"success": False, "report": {"passes": [1, 2], "failures": [3]}}
            print(json.dumps({"event": "evaluation", **ev}), flush=True)
""")

ITEM = {
    "task_id": "aw_test_1",
    "instruction": "Find the spotify password.",
    "supervisor": {"first_name": "Alex", "last_name": "Kim",
                   "email": "alex@x.io", "phone_number": "555"},
    "difficulty": 2,
    "split": "train",
}
ITEM["initial_obs"] = initial_obs_of(ITEM)


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


@pytest.fixture
def stub_cfg(tmp_path, monkeypatch):
    stub = tmp_path / "stub_worker.py"
    stub.write_text(_STUB)
    return AppworldCfg(
        appworld_root=str(tmp_path),
        worker_python=sys.executable,
        worker_script=str(stub),
        engine_slots=2,
        start_timeout_s=20,
        exec_timeout_s=20,
    )


def _factory(cfg) -> AppworldEnvFactory:
    return AppworldEnvFactory(cfg)


def _code(text: str, body: str) -> str:
    return f"{text}\n```python\n{body}\n```"


def _episode(cfg, texts):
    adapter = _factory(cfg).adapter_from_item(ITEM)
    return _run(run_episode(
        adapter=adapter,
        generate=ScriptedGenerate(TOK, texts),
        tokenizer=TOK,
        delta=DELTA,
        guidance="",
        episode_id="ep-aw",
        response_budget=200000,
        sampling_params={"temperature": 0.5},
    ))


def test_episode_full_success_via_stub(stub_cfg):
    result = _episode(stub_cfg, [
        _code("explore first", "print(apis.api_docs.show_app_descriptions())"),
        _code("done", "apis.supervisor.complete_task(answer='x')"),
    ])
    traj = result.trajectory
    assert traj.won and traj.score == 1.0
    assert traj.steps[0].observation.startswith("OUT<")
    # 红线:评估信息零反馈(completed 轮的反馈仍是 execute 原始输出)
    assert "success" not in traj.steps[1].observation.lower()
    assert "passes" not in traj.steps[1].observation


def test_episode_partial_credit(stub_cfg, monkeypatch):
    monkeypatch.setenv("STUB_EVAL", "partial")
    result = _episode(stub_cfg, [
        _code("go", "apis.supervisor.complete_task()"),
    ])
    traj = result.trajectory
    assert traj.won is False
    assert traj.score == pytest.approx(2 / 3)   # requirement 通过比例保真


def test_react_parsing_and_format_failure(stub_cfg):
    adapter = _factory(stub_cfg).adapter_from_item(ITEM)
    env = adapter.open_episode()
    try:
        out = adapter.react(env, "thinking without any code")
        assert out.acted is False and out.feedback == FORMAT_REMINDER
        # 最后一个块生效;块外文本进 thought
        two = "plan A\n```python\nprint(1)\n```\nrevised:\n```python\nprint(2)\n```"
        out = adapter.react(env, two)
        assert out.acted and out.action_norm == "print(2)"
        assert "plan A" in out.thought and "print(1)" not in out.action_norm
        fb, ended = adapter.replay_action(env, "print(2)")
        assert fb.startswith("OUT<") and not ended
    finally:
        env.close()


def test_slot_released_on_close(stub_cfg):
    adapter = _factory(stub_cfg).adapter_from_item(ITEM)
    envs = [adapter.open_episode() for _ in range(2)]   # slots=2 用满
    for e in envs:
        e.close()
    env = adapter.open_episode()                        # close 后可复用
    env.close()


def test_factory_fail_loud_and_registry(tmp_path):
    with pytest.raises(ValueError, match="appworld_root"):
        AppworldEnvFactory(AppworldCfg())
    cfg = SimpleNamespace(
        environment=SimpleNamespace(name="appworld"),
        appworld=AppworldCfg(appworld_root=str(tmp_path), worker_python=sys.executable),
    )
    assert isinstance(registry.make_env_factory(cfg), AppworldEnvFactory)
    # 契约面:fenced-code 协议(非 FC),难度桶 task_type
    adapter = AppworldTaskAdapter(ITEM, cfg.appworld, __import__("threading").Semaphore(1))
    assert adapter.tools() is None
    assert adapter.task_type == "difficulty_2"
    assert adapter.gt_text() is None
    assert "complete_task" in SYSTEM_PROMPT and "api_docs" in SYSTEM_PROMPT


@pytest.fixture
def deploy_env(monkeypatch, tmp_path):
    for k, v in {
        "TRACE_MODEL_PATH": str(tmp_path / "model"),
        "APPWORLD_TRAIN_FILE": str(tmp_path / "t.parquet"),
        "APPWORLD_VAL_FILE": str(tmp_path / "v.parquet"),
        "APPWORLD_ROOT": str(tmp_path / "d"),
        "APPWORLD_WORKER_PYTHON": sys.executable,
        "TRACE_RESEARCHER_API_KEY": "dummy-for-validation",
    }.items():
        monkeypatch.setenv(k, v)


def test_appworld_yaml_pair(deploy_env):
    root = Path(__file__).resolve().parents[1]
    selfres = load_config(root / "configs/experiments/a800-appworld-s1-selfres.yaml")
    extres = load_config(root / "configs/experiments/a800-appworld-s1-extres.yaml")
    assert selfres.environment.name == "appworld"
    assert selfres.episode.max_env_steps == 30
    assert selfres.appworld.engine_slots == 64
    assert selfres.research_rl.enabled and not extres.research_rl.enabled
    assert objective_fingerprint(selfres) != objective_fingerprint(extres)
