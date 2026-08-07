"""162/1.7B v4 self-researcher 配置合同。

红线:相对 A800 版只允许**机器与身份差异** + **2026-08-07 的长度裁定**——
其余机制参数(research_rl 其余项、steering v3.2 全套、24 行布局)必须逐字一致,
否则两机结果不可对照。

长度裁定单独列白名单而不是把红线整条废掉:指纹比较时先把 162 的长度参数
对齐回 A800 值,若仍相等即证明"除这一处外机制零漂移"。
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


def test_mechanism_identical_to_a800_modulo_length_ruling(env):
    """把长度参数对齐回 A800 后,指纹必须逐字相同 = 除长度外机制零漂移。"""
    a, c = load_config(A800), load_config(C162)
    c.research_rl.research_max_sample_tokens = a.research_rl.research_max_sample_tokens
    assert objective_fingerprint(c) == objective_fingerprint(a)


#: Qwen3-1.7B 的 max_position_embeddings。verl 的 async_sglang_server 硬校验
#: max_model_len <= 该值,超出即 ValueError。用户 2026-08-07 裁定的 256K 因此
#: 不可执行(要突破只能改模型 config 开 YaRN = 改基座,已升级用户裁定)。
QWEN3_1P7B_MAX_POS = 40960


def test_length_ruling_2026_08_07(env):
    """用户裁定 2026-08-07:上下文尽量大、researcher 单轮 8K、executor 单轮 4K。

    裁定中可执行的部分全部落到位;不可执行的部分(256K)锁住其物理原因,
    防止后来者把 rollout_max_model_len 改大后在启动时才发现起不来。
    """
    c = load_config(C162)
    assert c.episode.max_tokens_per_turn == 4096, "任务执行 agent 单轮输出 = 4K"
    assert c.research.researcher_max_tokens == 8192, "researcher 单轮输出 = 8K"
    assert c.role("researcher").max_tokens == 8192
    # 上下文顶到物理上限,不多不少
    assert c.train.rollout_max_model_len == QWEN3_1P7B_MAX_POS, (
        "超过 max_position_embeddings 会在 SGLang 起服时 ValueError"
    )
    # response 预算 = ctx − prompt,即在物理上限内顶满
    assert c.train.response_length == QWEN3_1P7B_MAX_POS - c.train.max_prompt_length
    assert c.eval.response_budget == c.train.response_length, "训评同构"
    # 本阈不再是绑定约束:等于 ctx 上限 ⇒ 只有撞 ctx 墙的行才会被 mask
    assert c.research_rl.research_max_sample_tokens == QWEN3_1P7B_MAX_POS, (
        "旧阈 24576 把 82% 的 research 行 mask 成零梯度(162 冒烟 step1 实测)"
    )
    assert (c.train.max_token_len_per_gpu * c.train.ulysses_sp
            >= c.train.max_prompt_length + c.train.response_length), (
        "单序列容量声明必须放得下 prompt+response,否则训练侧会 fail"
    )
    assert c.train.rollout_max_model_len >= (
        c.train.max_prompt_length + c.train.response_length
    ), "rollout ctx 必须 ≥ prompt+response"


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
