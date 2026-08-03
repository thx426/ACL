"""训练侧 W_beh 展开(verl/trainer/ppo/steering_weights.py)——稀疏账→稠密权重。

该模块在 vendored verl 里但零 verl 依赖(纯 numpy),直接按路径加载测试。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_MOD = (
    Path(__file__).resolve().parents[2]
    / "verl" / "verl" / "trainer" / "ppo" / "steering_weights.py"
)
_spec = importlib.util.spec_from_file_location("steering_weights", _MOD)
sw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sw)


def test_per_token_expansion():
    w = sw.expand_behavior_weights(
        positions=[2, 5], log_pi=[-1.0, -2.0], log_b=[-1.5, -1.6],
        response_len=8, mode="per_token",
    )
    assert w.dtype == np.float32 and len(w) == 8
    assert w[2] == pytest.approx(np.exp(0.5), rel=1e-6)
    assert w[5] == pytest.approx(np.exp(-0.4), rel=1e-6)
    mask = np.ones(8, bool); mask[[2, 5]] = False
    assert np.all(w[mask] == 1.0)


def test_vtrace_truncates_above_one():
    w = sw.expand_behavior_weights(
        positions=[0, 1], log_pi=[-1.0, -3.0], log_b=[-2.0, -2.0],
        response_len=3, mode="vtrace",
    )
    assert w[0] == 1.0                       # π/b = e^{1} → 截到 1
    assert w[1] == pytest.approx(np.exp(-1.0), rel=1e-6)
    assert w[2] == 1.0


def test_per_decision_prefix_product_carries_forward():
    w = sw.expand_behavior_weights(
        positions=[1, 3], log_pi=[-1.0, -1.0], log_b=[-1.5, -0.5],
        response_len=6, mode="per_decision",
    )
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(np.exp(0.5), rel=1e-6)
    assert w[2] == pytest.approx(np.exp(0.5), rel=1e-6)   # 前缀积延续
    assert w[3] == pytest.approx(np.exp(0.0), rel=1e-6)   # 0.5 + (−0.5)
    assert np.all(w[4:] == pytest.approx(1.0))


def test_empty_ledger_is_all_ones():
    w = sw.expand_behavior_weights(
        positions=[], log_pi=[], log_b=[], response_len=5, mode="per_token"
    )
    assert np.all(w == 1.0)


def test_tail_truncation_drops_out_of_window_rows():
    w = sw.expand_behavior_weights(
        positions=[1, 9], log_pi=[-1.0, -1.0], log_b=[-2.0, -2.0],
        response_len=4, mode="per_token",
    )
    assert w[1] == pytest.approx(np.e, rel=1e-6)
    assert len(w) == 4


def test_cap_violation_fails_loud():
    with pytest.raises(ValueError, match="corrupted ledger"):
        sw.expand_behavior_weights(
            positions=[0], log_pi=[0.0], log_b=[-5.0],
            response_len=2, mode="per_token", log_ratio_cap=2.5,
        )


def test_bad_mode_and_bad_positions():
    with pytest.raises(ValueError, match="unknown w_beh mode"):
        sw.expand_behavior_weights(positions=[], log_pi=[], log_b=[], response_len=1, mode="nope")
    with pytest.raises(ValueError, match="strictly increasing"):
        sw.expand_behavior_weights(
            positions=[3, 3], log_pi=[0, 0], log_b=[0, 0], response_len=5
        )
