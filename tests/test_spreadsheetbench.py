"""SpreadsheetBench 环境:评估语义锁、function-call react、端到端与隔离红线。

评估语义 = 旧 skills_evolve 移植(官方 evaluation.py 规则),用行为快照锁定
(round-2 量化/类型严格/空值等价/datetime/bool);episode 端到端跑真
subprocess(run_python)驱动真实机制循环 —— function-calling 协议在非
文本动作环境上成立的机器证明。
"""
from __future__ import annotations

import asyncio
import datetime
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import openpyxl
import pytest

from fakes import FakeTokenizer, ScriptedGenerate
from trace_opd.common.config import (
    SpreadsheetbenchCfg,
    load_config,
    objective_fingerprint,
)
from trace_opd.core.episode import run_episode
from trace_opd.core.template_delta import TemplateDelta
from trace_opd.envs import registry
from trace_opd.envs.spreadsheetbench.adapter import (
    FORMAT_REMINDER,
    SpreadsheetbenchEnvFactory,
    SpreadsheetbenchTaskAdapter,
    initial_obs_of,
)
from trace_opd.envs.spreadsheetbench.evaluator import (
    _compare_cell_value,
    evaluate,
    find_test_cases,
    preview_workbook,
)

TOK = FakeTokenizer()
DELTA = TemplateDelta(TOK)


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def _wb(path: Path, cells: dict[str, object]) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    for coord, v in cells.items():
        ws[coord] = v
    wb.save(path)
    wb.close()


def _make_task(tmp_path: Path, *, answers: list[dict[str, object]]) -> dict:
    """构造任务目录(N 个 case)+ 返回自足 extra_info item。
    input: A1=1, A2=2;答案单元格 B1 由 answers[i] 给出。"""
    task_dir = tmp_path / "spreadsheet" / "T1"
    task_dir.mkdir(parents=True)
    for i, ans in enumerate(answers, 1):
        _wb(task_dir / f"{i}_t1_input.xlsx", {"A1": 1, "A2": 2})
        _wb(task_dir / f"{i}_t1_answer.xlsx", {"A1": 1, "A2": 2, **ans})
    item = {
        "task_id": "T1",
        "instruction": "Sum A1:A2 into B1.",
        "instruction_type": "Cell-Level Manipulation",
        "answer_position": "B1",
        "spreadsheet_dir": "spreadsheet/T1",
        "split": "train",
        "n_cases": len(answers),
    }
    item["initial_obs"] = initial_obs_of(
        item, preview_workbook(str(task_dir / "1_t1_input.xlsx"))
    )
    return item


def _adapter(tmp_path: Path, item: dict) -> SpreadsheetbenchTaskAdapter:
    cfg = SpreadsheetbenchCfg(data_root=str(tmp_path))
    return SpreadsheetbenchTaskAdapter(item, cfg, threading.Semaphore(4))


def _call(name: str, **arguments) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"working on it\n<tool_call>\n{payload}\n</tool_call>"


SUM_CODE = (
    "import openpyxl\n"
    "wb = openpyxl.load_workbook('input.xlsx')\n"
    "ws = wb.active\n"
    "ws['B1'] = ws['A1'].value + ws['A2'].value\n"
    "wb.save('output.xlsx')\n"
    "print('saved')\n"
)


# ------------------------------------------------------- 评估语义锁(官方移植)

def test_compare_cell_value_official_semantics():
    assert _compare_cell_value(3.001, 3.0)            # round-2 量化
    assert not _compare_cell_value(3.01, 3.0)
    assert _compare_cell_value("3.00", 3)             # 数值字符串→float
    assert _compare_cell_value("", None)              # 空值等价
    assert not _compare_cell_value("3", "three")
    assert not _compare_cell_value(1, "x")            # 类型严格
    assert _compare_cell_value(True, 1)               # bool 提升
    assert _compare_cell_value(
        datetime.datetime(2024, 1, 2, 3), datetime.datetime(2024, 1, 2, 9)
    )                                                 # 序列日取整(同日等价)


def test_evaluate_and_case_discovery(tmp_path):
    item = _make_task(tmp_path, answers=[{"B1": 3}])
    task_dir = tmp_path / "spreadsheet" / "T1"
    cases = find_test_cases(str(task_dir))
    assert len(cases) == 1 and cases[0][0] == "1"
    pred = tmp_path / "pred.xlsx"
    _wb(pred, {"A1": 1, "A2": 2, "B1": 3})
    assert evaluate(str(pred), cases[0][2], item["answer_position"])["ok"]
    _wb(pred, {"A1": 1, "A2": 2, "B1": 4})
    verdict = evaluate(str(pred), cases[0][2], item["answer_position"])
    assert not verdict["ok"]
    assert "pred=4" in verdict["reason"]              # GT SAFETY:只报 pred
    assert "3" not in verdict["reason"].split("pred=")[1]  # 不泄 golden 值


# ------------------------------------------------------- episode 端到端(真执行)

def _episode(item, tmp_path, texts):
    adapter = _adapter(tmp_path, item)
    return _run(run_episode(
        adapter=adapter,
        generate=ScriptedGenerate(TOK, texts),
        tokenizer=TOK,
        delta=DELTA,
        guidance="",
        episode_id="ep-ssb",
        response_budget=200000,
        sampling_params={"temperature": 0.5},
    ))


def test_episode_win_via_function_calls(tmp_path):
    item = _make_task(tmp_path, answers=[{"B1": 3}])
    result = _episode(item, tmp_path, [
        _call("run_python", code=SUM_CODE),
        _call("submit"),
    ])
    traj = result.trajectory
    assert traj.won and traj.score == 1.0
    assert traj.steps[0].valid                       # exit 0
    assert "saved" in traj.steps[0].observation
    assert traj.steps[0].observation.startswith("<tool_response>")
    # 信息状态红线:submit 反馈零评估信息
    assert "Submitted." in traj.steps[1].observation
    assert "pass" not in traj.steps[1].observation.lower()
    # action_norm 可 json 解析(replay 契约)
    assert json.loads(traj.steps[0].action_norm)["name"] == "run_python"


def test_episode_partial_credit_multi_case(tmp_path):
    # case1 期望 B1=3(agent 能对),case2 期望 B1=99(同一 output 必挂)
    item = _make_task(tmp_path, answers=[{"B1": 3}, {"B1": 99}])
    result = _episode(item, tmp_path, [
        _call("run_python", code=SUM_CODE),
        _call("submit"),
    ])
    traj = result.trajectory
    assert traj.won is False
    assert traj.score == 0.5                          # soft 保真


def test_episode_format_failure_and_premature_submit(tmp_path):
    item = _make_task(tmp_path, answers=[{"B1": 3}])
    result = _episode(item, tmp_path, [
        "let me think, no call yet",                  # 格式失败
        _call("submit"),                              # output 未写:不终局
        _call("run_python", code=SUM_CODE),
        _call("submit"),
    ])
    traj = result.trajectory
    assert traj.steps[0].acted is False
    assert traj.steps[0].observation == FORMAT_REMINDER
    assert traj.steps[1].acted and not traj.steps[1].valid
    assert "not found" in traj.steps[1].observation
    assert traj.won and traj.score == 1.0
    assert result.invalid_format == 1


def test_replay_action_deterministic_and_isolated(tmp_path):
    item = _make_task(tmp_path, answers=[{"B1": 3}])
    adapter = _adapter(tmp_path, item)
    env = adapter.open_episode()
    try:
        # 隔离红线:沙箱里只有 input.xlsx,golden 不可达
        assert sorted(p.name for p in Path(env.workdir).iterdir()) == ["input.xlsx"]
        norm = json.dumps(
            {"name": "run_python", "arguments": {"code": "print(open('input.xlsx','rb').read(4).hex())"}},
            sort_keys=True,
        )
        fb1, ended1 = adapter.replay_action(env, norm)
        fb2, ended2 = adapter.replay_action(env, norm)
        assert fb1 == fb2 and not ended1 and not ended2   # 重放确定性
    finally:
        env.close()
    assert not Path(env.workdir).exists()                 # close 清理沙箱


# ------------------------------------------------------------- 装配与配置

def test_registry_dispatch_and_dataroot_required(tmp_path):
    cfg = SimpleNamespace(
        environment=SimpleNamespace(name="spreadsheetbench"),
        spreadsheetbench=SpreadsheetbenchCfg(data_root=str(tmp_path)),
    )
    factory = registry.make_env_factory(cfg)
    assert isinstance(factory, SpreadsheetbenchEnvFactory)
    bad = SimpleNamespace(
        environment=SimpleNamespace(name="spreadsheetbench"),
        spreadsheetbench=SpreadsheetbenchCfg(data_root=""),
    )
    with pytest.raises(ValueError, match="data_root"):
        registry.make_env_factory(bad)


@pytest.fixture
def deploy_env(monkeypatch, tmp_path):
    for k, v in {
        "TRACE_MODEL_PATH": str(tmp_path / "model"),
        "SSB_TRAIN_FILE": str(tmp_path / "t.parquet"),
        "SSB_VAL_FILE": str(tmp_path / "v.parquet"),
        "SSB_DATA": str(tmp_path / "d"),
        "TRACE_RESEARCHER_API_KEY": "dummy-for-validation",
    }.items():
        monkeypatch.setenv(k, v)


def test_ssb_yaml_pair_loads_and_fingerprints(deploy_env):
    root = Path(__file__).resolve().parents[1]
    selfres = load_config(root / "configs/experiments/a800-ssb-s1-selfres.yaml")
    extres = load_config(root / "configs/experiments/a800-ssb-s1-extres.yaml")
    assert selfres.environment.name == "spreadsheetbench"
    assert selfres.research_rl.enabled and not extres.research_rl.enabled
    assert extres.research.backend == "external"      # 外部 researcher 路保留
    fp_self, fp_ext = objective_fingerprint(selfres), objective_fingerprint(extres)
    assert fp_self != fp_ext                          # 双目标 vs 单目标
    assert "environment=spreadsheetbench" not in (fp_self + fp_ext)  # 哈希非明文
    # 换环境进指纹:同配置改回 alfworld 指纹必变
    selfres.environment.name = "alfworld"
    assert objective_fingerprint(selfres) != fp_self
