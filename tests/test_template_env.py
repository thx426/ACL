"""模板 benchmark 端到端:机制核心 × 任务实现分离性的机器证明。

用与 ALFWorld 零共享的玩具环境(envs/_template:独立动作文法、独立环境
习语、部分分评估)驱动真实机制——episode 运行器、v4 前缀续写、registry
装配全程零机制改动即为分离性成立;同时锁定:environment 默认值不扰动
objective_fingerprint(既有 run 可续)。
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from fakes import FakeTokenizer, ScriptedGenerate, reply
from trace_opd.common.config import load_config, objective_fingerprint
from trace_opd.core.episode import run_episode
from trace_opd.core.template_delta import TemplateDelta
from trace_opd.core.types import OUTCOME_SUCCESS
from trace_opd.envs import registry
from trace_opd.envs._template.adapter import (
    REJECT_OBS,
    PatchboardEnv,
    TemplateEnvFactory,
    TemplateTaskAdapter,
)
from trace_opd.envs._template.build_dataset import make_rows
from trace_opd.research.continuation import build_prefix_set, continue_from_prefix

TOK = FakeTokenizer()
DELTA = TemplateDelta(TOK)
ROWS = make_rows("train")
PASSTHROUGH = ROWS[0]["extra_info"]          # {"s1": "out1"}
CROSSLINK = ROWS[3]["extra_info"]            # {"s1": "out2", "s2": "out1"}


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


async def _episode(item, texts, budget=100000):
    adapter = TemplateTaskAdapter(item)
    result = await run_episode(
        adapter=adapter,
        generate=ScriptedGenerate(TOK, texts),
        tokenizer=TOK,
        delta=DELTA,
        guidance="",
        episode_id="ep-template",
        response_budget=budget,
        sampling_params={"temperature": 0.5},
    )
    return result


# ----------------------------------------------------------------- 契约完备

def test_contract_surface_complete():
    adapter = TemplateTaskAdapter(PASSTHROUGH)
    factory = TemplateEnvFactory()
    for name in (
        "task_id", "task_type", "initial_obs", "task_brief",
        "executor_scaffold", "open_episode", "gt_text", "domain_notes",
        "react", "replay_action", "normalize_action",
        "observation_indicates_invalid", "format_reminder", "tools",
    ):
        assert hasattr(adapter, name), f"TaskAdapter contract missing: {name}"
    assert hasattr(factory, "adapter_from_item")
    env = adapter.open_episode()
    assert hasattr(env, "initial_obs") and hasattr(env, "close")
    env.close()
    # dataset 行契约:extra_info 自足(工厂只见它)
    assert factory.adapter_from_item(PASSTHROUGH).task_id == PASSTHROUGH["task_id"]
    # GT 信息状态:存在且只经 gt_text 暴露
    assert adapter.gt_text().startswith("1. link")


def test_patchboard_partial_credit_and_reject():
    env = PatchboardEnv(dict(CROSSLINK["spec"]), CROSSLINK["initial_obs"])
    obs, done, _ = env.step("probe s1")
    assert not done and "signal" in obs           # 探索揭示配对信息
    obs, done, _ = env.step("link s1 out2")
    assert "cable now joins" in obs
    obs, done, score = env.step("run diagnostics")
    assert done and score == 0.5                  # 部分分:2 求 1 对
    env2 = PatchboardEnv(dict(CROSSLINK["spec"]), CROSSLINK["initial_obs"])
    obs, done, score = env2.step("open the pod bay doors")
    assert obs == REJECT_OBS and not done and score == 0.0


# --------------------------------------------------------- episode 端到端

def test_episode_win_on_template_env():
    texts = [
        reply("find the signal", "probe s1"),
        reply("route it", "link s1 out1"),
        reply("verify", "run diagnostics"),
    ]
    result = _run(_episode(PASSTHROUGH, texts))
    traj = result.trajectory
    assert traj.outcome == OUTCOME_SUCCESS and traj.won
    assert traj.score == 1.0
    assert traj.steps[0].valid                     # probe 是有效动作
    assert "signal" in traj.steps[0].observation
    # 分离性:全轨迹不含任何 ALFWorld 习语
    assert all("Nothing happens." not in s.observation for s in traj.steps)


def test_episode_partial_credit_score_fidelity():
    texts = [
        reply("route one", "link s1 out2"),
        reply("call it done", "run diagnostics"),
    ]
    result = _run(_episode(CROSSLINK, texts))
    traj = result.trajectory
    assert traj.won is False                       # GRPO 口径:score<1.0
    assert traj.score == 0.5                       # 部分分保真进轨迹
    assert traj.steps[-1].valid


def test_episode_format_failure_uses_template_reminder():
    texts = [
        "thinking aloud with no action block",
        reply("recover", "link s1 out1"),
        reply("verify", "run diagnostics"),
    ]
    result = _run(_episode(PASSTHROUGH, texts))
    traj = result.trajectory
    step0 = traj.steps[0]
    assert step0.acted is False
    assert step0.observation.startswith(REJECT_OBS)   # 本环境习语的提醒
    assert "Nothing happens." not in step0.observation
    assert traj.won and result.invalid_format == 1


# ------------------------------------------- v4 前缀续写端到端(重放+半轮)

def test_continuation_replay_and_win_on_template():
    item = PASSTHROUGH
    t0 = reply("scan", "probe s1")
    obs0 = "Port s1 carries signal alpha."          # env 对 probe s1 的真响应
    t1_half = "<reasoning>half thought"
    ids: list[int] = []
    mask: list[int] = []
    t0_ids = TOK.encode(t0)
    spans = [[0, len(t0_ids)]]
    ids += t0_ids
    mask += [1] * len(t0_ids)
    o_ids = TOK.encode(f"<obs>{obs0}</obs>")
    ids += o_ids
    mask += [0] * len(o_ids)
    t1_ids = TOK.encode(t1_half)
    spans.append([len(ids), len(ids) + len(t1_ids)])
    ids += t1_ids
    mask += [1] * len(t1_ids)
    row = {
        "won": False, "response_ids": ids, "response_mask": mask,
        "turn_spans": spans,
        "steps": [
            {"action_norm": "probe s1", "observation": obs0, "acted": True},
            {"action_norm": "", "observation": "", "acted": False},
        ],
    }
    (spec,) = build_prefix_set([row], [0.99], TOK)
    assert spec.replay_actions == ("probe s1",)     # 轮 0 进重放
    out = _run(continue_from_prefix(
        adapter=TemplateTaskAdapter(item),
        teacher_prefix_ids=TOK.encode("DOC:"),
        spec=spec,
        generate=ScriptedGenerate(TOK, [
            " continue</reasoning>\n<action>link s1 out1</action>",
            reply("finish", "run diagnostics"),
        ]),
        tokenizer=TOK,
        delta=DELTA,
        response_budget=8000,
        sampling_params={"temperature": 1.0},
    ))
    assert out.error is None, out.error             # 重放逐步校验通过
    assert out.won and out.score == 1.0


# ------------------------------------------------------------ registry 装配

def test_registry_dispatch_unknown_and_duplicate():
    name = "template-e2e"
    registry.register_environment(name, lambda config, workers: TemplateEnvFactory())
    try:
        cfg = SimpleNamespace(environment=SimpleNamespace(name=name))
        factory = registry.make_env_factory(cfg)
        assert isinstance(factory, TemplateEnvFactory)
        with pytest.raises(ValueError, match="already registered"):
            registry.register_environment(name, lambda config, workers: None)
    finally:
        registry._BUILDERS.pop(name, None)
    bad = SimpleNamespace(environment=SimpleNamespace(name="no-such-env"))
    with pytest.raises(ValueError, match="unknown environment"):
        registry.make_env_factory(bad)


# --------------------------------------------- 指纹:默认环境零扰动(续跑锁)

@pytest.fixture
def deploy_env(monkeypatch, tmp_path):
    for k, v in {
        "TRACE_MODEL_PATH": str(tmp_path / "model"),
        "TRACE_TRAIN_FILE": str(tmp_path / "t.parquet"),
        "TRACE_VAL_SEEN_FILE": str(tmp_path / "a.parquet"),
        "TRACE_VAL_UNSEEN_FILE": str(tmp_path / "b.parquet"),
        "ALFWORLD_DATA": str(tmp_path / "d"),
        "ALFWORLD_WORKER_PYTHON": str(tmp_path / "py"),
        "TRACE_RESEARCHER_API_KEY": "dummy-for-validation",
    }.items():
        monkeypatch.setenv(k, v)


def test_environment_default_keeps_fingerprint(deploy_env):
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/experiments/a800-pgs-s4-selfres.yaml")
    assert cfg.environment.name == "alfworld"       # 不写段 = 默认
    base = objective_fingerprint(cfg)
    cfg.environment.name = "other-bench"
    assert objective_fingerprint(cfg) != base       # 换环境 = 换优化目标
    cfg.environment.name = "alfworld"
    assert objective_fingerprint(cfg) == base       # 默认值零扰动(续跑锁)
