"""分臂采样(v3.1)——臂分配纯函数、config 校验/指纹、聚合端分臂曲线。

臂分配的红线:fraction<1 时缺 session_id 必须 fail loud——静默回退会让
整组悄悄变回全 steered,正是分臂要修的观测污染。指纹的红线:fraction=1.0
(历史全臂)不得改变既有 run 的指纹;<1 必须换指纹(换优化目标)。
"""
from __future__ import annotations

import numpy as np
import pytest

from trace_opd.common.config import load_config, objective_fingerprint
from trace_opd.steering.arms import is_steered_arm, steered_arm_count

YAML = """
run: {name: t, runs_root: runs}
alfworld: {data_root: /d, worker_python: /p}
train: {model_path: /m, train_file: /t.parquet, val_files: [/v1.parquet]%s}
eval: {}
%s
"""

_STEER = "steering: {enabled: true, doc_table: /tmp/d.jsonl, theta_d: 0.8%s}"


def _load(tmp_path, train_extra="", steer_extra=""):
    path = tmp_path / f"c{abs(hash((train_extra, steer_extra)))}.yaml"
    path.write_text(YAML % (train_extra, _STEER % steer_extra))
    return load_config(path)


# ------------------------------------------------------------ 臂分配纯函数

def test_arm_count_rounding():
    assert steered_arm_count(8, 0.5) == 4
    assert steered_arm_count(8, 1.0) == 8
    assert steered_arm_count(8, 0.25) == 2
    assert steered_arm_count(3, 0.5) == 2      # round(1.5) → banker's 2
    assert steered_arm_count(8, 0.05) == 0     # 校验层负责拒绝,算术如实


def test_arm_assignment_is_prefix_deterministic():
    picks = [is_steered_arm(i, 8, 0.5) for i in range(8)]
    assert picks == [True] * 4 + [False] * 4


def test_full_fraction_keeps_legacy_behaviour_and_allows_missing_session():
    # fraction=1.0 = 历史全臂:臂身份无意义,V0 等不传 session_id 的路径合法
    assert is_steered_arm(None, 8, 1.0) is True
    assert all(is_steered_arm(i, 8, 1.0) for i in range(8))


def test_split_fraction_fails_loud_without_session_identity():
    with pytest.raises(RuntimeError, match="session_id"):
        is_steered_arm(None, 8, 0.5)
    with pytest.raises(RuntimeError, match="out of range"):
        is_steered_arm(8, 8, 0.5)
    with pytest.raises(RuntimeError, match="out of range"):
        is_steered_arm(-1, 8, 0.5)


# ------------------------------------------------------ config 校验与指纹

def test_config_validates_fraction_bounds(tmp_path):
    cfg = _load(tmp_path, steer_extra=", steered_fraction: 0.5")
    assert cfg.steering.steered_fraction == 0.5
    for bad in (", steered_fraction: 0.0", ", steered_fraction: 1.5",
                ", steered_fraction: -0.5"):
        with pytest.raises(ValueError):
            _load(tmp_path, steer_extra=bad)
    # 取整后 0 个 steered(等价关 steering)与"取整后整组 steered 却写 <1"
    # 都是歧义形态,启动即拒。
    with pytest.raises(ValueError, match="0 个 steered"):
        _load(tmp_path, steer_extra=", steered_fraction: 0.05")
    with pytest.raises(ValueError, match="整组 steered"):
        _load(tmp_path, ", rollout_n: 2", ", steered_fraction: 0.9")


def test_fingerprint_stable_at_full_fraction_and_moves_below(tmp_path):
    base = objective_fingerprint(_load(tmp_path))
    explicit_full = objective_fingerprint(
        _load(tmp_path, steer_extra=", steered_fraction: 1.0")
    )
    assert explicit_full == base            # 1.0 不进 material(既有 run 可续)
    split = objective_fingerprint(
        _load(tmp_path, steer_extra=", steered_fraction: 0.5")
    )
    assert split != base                    # 分臂 = 换优化目标 ⇒ 换指纹


# ------------------------------------------- 聚合端分臂曲线(vendored patch)

def _rollout_extra_metrics():
    """按路径抽取 compute_rollout_extra_metrics(模块顶层 import torch,
    本地测试环境无——ast 抽函数,先例 test_sglang_adapters)。"""
    import ast
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[2]
        / "verl" / "verl" / "trainer" / "ppo" / "metric_utils.py"
    )
    tree = ast.parse(src.read_text())
    ns: dict = {"np": np, "Any": object}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "compute_rollout_extra_metrics":
            mod = ast.Module(body=[node], type_ignores=[])
            exec(compile(mod, str(src), "exec"), ns)  # noqa: S102 — 本仓源码
            return ns["compute_rollout_extra_metrics"]
    raise AssertionError("compute_rollout_extra_metrics not found")


def test_per_arm_rates_and_gap():
    fn = _rollout_extra_metrics()
    m = fn({
        "won":       [1, 1, 0, 0, 1, 0, 0, 0],
        "steer_arm": [1, 1, 1, 1, 0, 0, 0, 0],
        "reward":    [9, 9, 9, 9, 9, 9, 9, 9],   # 必须仍被跳过
    })
    assert m["rollout_extra/won_steered_rate/mean"] == pytest.approx(0.5)
    assert m["rollout_extra/won_bare_rate/mean"] == pytest.approx(0.25)
    assert m["rollout_extra/steer_arm_gap/mean"] == pytest.approx(0.25)
    assert m["rollout_extra/won/mean"] == pytest.approx(3 / 8)
    assert not any("reward" in k for k in m)


def test_per_arm_rates_skip_empty_arm_no_nan():
    fn = _rollout_extra_metrics()
    m = fn({"won": [1, 0], "steer_arm": [1, 1]})   # fraction=1.0 批:无 bare
    assert "rollout_extra/won_steered_rate/mean" in m
    assert "rollout_extra/won_bare_rate/mean" not in m
    assert "rollout_extra/steer_arm_gap/mean" not in m
    assert all(np.isfinite(v) for v in m.values())
