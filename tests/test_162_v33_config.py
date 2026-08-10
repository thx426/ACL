"""162/1.7B v3.3 asym 配置合同。

红线:相对 A800/4B 版**只允许机器与基座身份差异**——v3.3 的三开关
(asym advantage / 熵门 G-E / W_beh off)、α 调度、ε 探针、研究预算
必须逐字一致,否则"4B vs 1.7B 的基座尺度效应"这个对照不成立。
判据用 objective_fingerprint:同一 model_path 下两配置指纹必须相同。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace_opd.common.config import load_config, objective_fingerprint  # noqa: E402

A800 = ROOT / "configs/experiments/a800-pgs-v33-4braw-asym.yaml"
C162 = ROOT / "configs/experiments/162-pgs-v33-1p7b-asym.yaml"


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


def test_only_machine_and_base_model_differ(env):
    a, c = load_config(A800), load_config(C162)
    # 机器/身份:允许不同
    assert a.train.n_gpus == 8 and c.train.n_gpus == 4
    assert c.run.name == "pgs_v33_1p7b_asym_162"
    assert c.train.experiment_name == c.run.name
    assert c.run.runs_root.startswith("/data4/")
    # v3.3 三开关:必须一致
    assert c.train.adv_estimator == a.train.adv_estimator == "pgs_asym_grpo"
    assert c.train.norm_adv_by_std is False, "asym 档强制显式 false"
    assert c.steering.adv_mode == a.steering.adv_mode == "asym"
    assert c.steering.gate_mode == a.steering.gate_mode == "entropy"
    assert c.steering.w_beh_mode == a.steering.w_beh_mode == "off"
    # α 有限期调度与探针
    assert c.steering.steered_adv_decay_steps == a.steering.steered_adv_decay_steps == 240
    assert c.steering.steered_adv_decay_mode == a.steering.steered_adv_decay_mode
    assert c.steering.epsilon == a.steering.epsilon == 0.05
    # 熵门参数
    assert c.steering.entropy_theta_init == a.steering.entropy_theta_init
    assert c.steering.entropy_floor == a.steering.entropy_floor
    assert c.steering.entropy_target_rate == a.steering.entropy_target_rate
    # 分臂与研究预算
    assert c.steering.steered_fraction == a.steering.steered_fraction == 0.5
    assert (c.research.n_exp, c.research.k_default, c.research.k_max) == (
        a.research.n_exp, a.research.k_default, a.research.k_max)


def test_external_researcher_not_self(env):
    """本臂用 external 35B 双端点,不是 v4 自体 researcher。"""
    c = load_config(C162)
    assert c.research.backend == "external"
    assert c.research.tool_protocol == "fncall"
    assert c.research_rl.enabled is False, "v3.3 主臂不开自体研究员"
    role = c.role("researcher")
    assert role.mode == "external_pool"
    assert len([e for e in role.endpoint.split(",") if e.strip()]) == 2, "双端点池"


def test_capacity_invariants_hold_on_4gpu(env):
    """4 卡 + sp=2 下,单序列容量与 rollout ctx 仍须放得下 prompt+response。"""
    c = load_config(C162)
    assert c.train.ulysses_sp == 2
    assert (c.train.max_token_len_per_gpu * c.train.ulysses_sp
            >= c.train.max_prompt_length + c.train.response_length)
    assert c.train.rollout_max_model_len >= (
        c.train.max_prompt_length + c.train.response_length)
    # 同一任务的全部副本须落在一个 worker(会话内状态在进程内)
    assert c.train.train_batch_size == c.train.agent_num_workers


def test_entropy_theta_init_is_a_4b_calibration(env):
    """⚠ 已知风险:entropy_theta_init=1.28 是在 **4B** 语境标定的
    (s3 dump step25/47/51 的 P95)。1.7B 的熵分布未必相同,起步几步的
    干预量可能偏高或偏低。不改是为了保住两机对照;靠 gate_adaptive 的
    EMA 自适应收敛,并以 entropy_floor 兜底。训练期需盯
    steer_entropy_theta 与 steer_n_base_open 的走向。"""
    c = load_config(C162)
    assert c.steering.gate_adaptive is True, "1.7B 语境靠自适应纠偏,不能关"
    assert c.steering.entropy_floor > 0, "必须有硬地板防阈值追熵坍缩"
