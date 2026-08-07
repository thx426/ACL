"""Math 环境:verifier 等价语义锁、FC react、端到端与信息红线。

verifier 走真 math-verify(本地已装,训练环境依赖在 GO 清单);episode
端到端跑真 subprocess(run_python)——TIR 形态在机制上成立的机器证明。
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from fakes import FakeTokenizer, ScriptedGenerate
from trace_opd.common.config import MathCfg, load_config, objective_fingerprint
from trace_opd.core.episode import run_episode
from trace_opd.core.template_delta import TemplateDelta
from trace_opd.envs import registry
from trace_opd.envs.math.adapter import (
    FORMAT_REMINDER,
    SYSTEM_PROMPT,
    TOOLS,
    MathEnvFactory,
    MathTaskAdapter,
    initial_obs_of,
)
from trace_opd.envs.math.build_dataset import _extract_answer, strip_dapo_template
from trace_opd.envs.math.verifier import answers_equivalent

TOK = FakeTokenizer()
DELTA = TemplateDelta(TOK)

ITEM = {
    "task_id": "dapo/test1",
    "task_type": "math",
    "split": "train",
    "problem": "Compute 4 * 13.",
    "ground_truth": "98765",   # 特意与题面无关的哨兵值(信息红线断言用)
}
ITEM["initial_obs"] = initial_obs_of(ITEM)


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def _adapter(item=None) -> MathTaskAdapter:
    return MathTaskAdapter(item or ITEM, MathCfg(), threading.Semaphore(4))


def _call(name: str, **arguments) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"reasoning here\n<tool_call>\n{payload}\n</tool_call>"


def _episode(item, texts):
    return _run(run_episode(
        adapter=_adapter(item),
        generate=ScriptedGenerate(TOK, texts),
        tokenizer=TOK,
        delta=DELTA,
        guidance="",
        episode_id="ep-math",
        response_budget=200000,
        sampling_params={"temperature": 0.5},
    ))


# ------------------------------------------------------------ verifier 语义锁

def test_answers_equivalent_semantics():
    assert answers_equivalent("204", "204")
    assert answers_equivalent("204", "\\boxed{204}")
    assert answers_equivalent("\\frac{1}{2}", "0.5")
    assert answers_equivalent("1/2", "\\frac{1}{2}")
    # latex 元组(实测坑:裸 parse 丢元素,封装用 $ 包裹组合)
    assert answers_equivalent("\\left( 3, \\frac{\\pi}{2} \\right)", "(3, \\pi/2)")
    assert not answers_equivalent("204", "205")
    assert not answers_equivalent("1/2", "1/3")
    assert not answers_equivalent("", "204")
    assert not answers_equivalent("204", "")


# ------------------------------------------------------------ 数据管线单元

def test_strip_dapo_template_anchors():
    content = (
        "Solve the following math problem step by step. The last line of "
        "your response should be of the form Answer: $Answer (without "
        "quotes) where $Answer is the answer to the problem.\n\n"
        "What is 2+2?\n\n"
        'Remember to put your answer on its own line after "Answer:".'
    )
    assert strip_dapo_template(content) == "What is 2+2?"
    with pytest.raises(SystemExit, match="anchors did not match"):
        strip_dapo_template("Some drifted prompt\nWhat is 2+2?")


def test_extract_answer_per_source():
    assert _extract_answer("aime24", {"solution": "\\boxed{204}"}) == "204"
    assert _extract_answer("aime26", {"answer": 277}) == "277"
    assert _extract_answer("amc23", {"answer": "27"}) == "27"
    assert _extract_answer("olympiad", {"final_answer": ["2"]}) == "2"
    assert _extract_answer("olympiad", {"final_answer": ["2", "3"]}) is None  # 多答案跳过
    assert _extract_answer("aime24", {"solution": "no box here"}) is None


# ------------------------------------------------------------ episode 端到端

def test_episode_tool_then_submit_wins():
    result = _episode(
        {**ITEM, "ground_truth": "52"},
        [_call("run_python", code="print(4 * 13)"), _call("submit", answer="52")],
    )
    traj = result.trajectory
    assert traj.won and traj.score == 1.0
    assert "52" in traj.steps[0].observation          # 工具真执行
    assert traj.steps[0].observation.startswith("<tool_response>")
    assert traj.steps[1].observation.startswith("<tool_response>\nSubmitted.")


def test_episode_direct_submit_and_wrong_answer():
    win = _episode({**ITEM, "ground_truth": "52"}, [_call("submit", answer="52")])
    assert win.trajectory.won and win.trajectory.step_count == 1
    lose = _episode({**ITEM, "ground_truth": "52"}, [_call("submit", answer="53")])
    assert lose.trajectory.won is False and lose.trajectory.score == 0.0
    # 信息红线:错误提交的反馈与胜利提交逐字节同形(零评分信息)
    assert lose.trajectory.steps[-1].observation == win.trajectory.steps[-1].observation


def test_episode_format_failure_and_empty_submit():
    result = _episode({**ITEM, "ground_truth": "52"}, [
        "just thinking, no call",
        _call("submit", answer=""),
        _call("submit", answer="52"),
    ])
    traj = result.trajectory
    assert traj.steps[0].acted is False
    assert traj.steps[0].observation == FORMAT_REMINDER
    assert traj.steps[1].acted and not traj.steps[1].valid
    assert "empty answer" in traj.steps[1].observation
    assert traj.won


def test_ground_truth_never_leaks():
    """哨兵 gt=98765 不得出现在 scaffold/brief/tools/任何反馈里。"""
    adapter = _adapter()
    assert "98765" not in adapter.executor_scaffold()
    assert "98765" not in adapter.task_brief()
    assert "98765" not in json.dumps(TOOLS)
    assert adapter.gt_text() is None
    result = _episode(ITEM, [_call("submit", answer="1")])
    assert all("98765" not in s.observation for s in result.trajectory.steps)


def test_replay_action_deterministic():
    adapter = _adapter()
    env = adapter.open_episode()
    try:
        norm = json.dumps(
            {"name": "run_python", "arguments": {"code": "print(sorted({'b','a'}))"}},
            sort_keys=True,
        )
        fb1, e1 = adapter.replay_action(env, norm)
        fb2, e2 = adapter.replay_action(env, norm)
        assert fb1 == fb2 and not e1 and not e2       # PYTHONHASHSEED=0 确定性
    finally:
        env.close()
    assert not Path(env.workdir).exists()


# ------------------------------------------------------------- 装配与配置

def test_registry_dispatch_math():
    cfg = SimpleNamespace(
        environment=SimpleNamespace(name="math"),
        math=MathCfg(),
    )
    assert isinstance(registry.make_env_factory(cfg), MathEnvFactory)
    # scaffold 零手写工具块(模板渲染)
    assert "<tools>" not in SYSTEM_PROMPT and "<tool_call>" not in SYSTEM_PROMPT
    assert [t["function"]["name"] for t in TOOLS] == ["run_python", "submit"]
    assert TOOLS[1]["function"]["parameters"]["required"] == ["answer"]


@pytest.fixture
def deploy_env(monkeypatch, tmp_path):
    for k, v in {
        "TRACE_MODEL_PATH": str(tmp_path / "model"),
        "MATH_TRAIN_FILE": str(tmp_path / "t.parquet"),
        "MATH_VAL_FILE": str(tmp_path / "v.parquet"),
        "TRACE_RESEARCHER_API_KEY": "dummy-for-validation",
    }.items():
        monkeypatch.setenv(k, v)


def test_math_yaml_pair_loads_and_fingerprints(deploy_env):
    root = Path(__file__).resolve().parents[1]
    selfres = load_config(root / "configs/experiments/a800-math-s1-selfres.yaml")
    extres = load_config(root / "configs/experiments/a800-math-s1-extres.yaml")
    assert selfres.environment.name == "math"
    assert selfres.episode.max_env_steps == 12
    assert selfres.episode.max_tokens_per_turn == 8192
    assert selfres.research_rl.enabled and not extres.research_rl.enabled
    assert objective_fingerprint(selfres) != objective_fingerprint(extres)
