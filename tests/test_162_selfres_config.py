"""162/1.7B v4 self-researcher 配置合同。

红线:相对 A800 版**只允许机器与身份差异**——机制参数(research_rl 全段、
steering v3.2 全套、24 行布局)必须逐字一致,否则两机结果不可对照。
判据用 objective_fingerprint:同一 model_path 下两配置指纹必须相同。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace_opd.common.config import load_config, objective_fingerprint  # noqa: E402
from trace_opd.steering.arms import steered_arm_count  # noqa: E402

A800 = ROOT / "configs/experiments/a800-pgs-s4-selfres.yaml"
C162 = ROOT / "configs/experiments/162-pgs-s4-selfres-1p7b.yaml"


@pytest.fixture
def env(monkeypatch, tmp_path):
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


def test_mechanism_identical_to_a800(env):
    """同 model_path 下指纹必须逐字相同 = 机制零漂移。"""
    assert objective_fingerprint(load_config(C162)) == objective_fingerprint(load_config(A800))


def test_only_machine_and_identity_differ(env):
    a, c = load_config(A800), load_config(C162)
    # 机器/身份:允许不同
    assert c.train.n_gpus == 4 and a.train.n_gpus == 8
    assert c.run.name == "pgs_s4_selfres_1p7b_162"
    assert c.run.runs_root.startswith("/data4/")
    assert c.train.experiment_name == c.run.name
    # 小模型单卡推理优势:TP 必须 1;长序列反向仍需 SP=2
    assert c.train.rollout_tp == 1
    assert c.train.ulysses_sp == 2
    # 机制:必须一致
    assert c.research_rl.research_n == a.research_rl.research_n == 8
    assert c.research_rl.prefix_quantiles == a.research_rl.prefix_quantiles
    assert c.research_rl.continuation_samples == a.research_rl.continuation_samples == 4
    assert c.research_rl.research_rl_weight == a.research_rl.research_rl_weight
    assert c.steering.gate_mode == a.steering.gate_mode == "always_on"
    assert c.steering.b_max == a.steering.b_max == float("inf")
    assert c.steering.r_max == a.steering.r_max == 0
    assert c.steering.steered_fraction == a.steering.steered_fraction == 0.5
    assert c.research.backend == "engine" and c.research.tool_protocol == "text"


def test_24_row_layout_and_worker_invariant(env):
    c = load_config(C162)
    k = steered_arm_count(c.train.rollout_n, c.steering.steered_fraction)
    assert k == 8 and c.train.rollout_n - k == 8
    assert c.train.rollout_n + c.research_rl.research_n == 24
    # 同一任务的全部副本必须落在一个 worker(会话内状态在进程内)
    assert c.train.train_batch_size == c.train.agent_num_workers
    assert c.train.train_batch_size % c.train.agent_num_workers == 0
