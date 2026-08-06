"""maxopen 档(机制 v3.2,用户裁定 2026-08-06)——gate_mode: always_on
+ b_max=inf 一等公民化的契约测试。

红线:① always_on 的参数纪律显式化(ε=0/gate_adaptive=false 必须写,
不做静默忽略);② b_max=inf 与 pacing 组合 fail loud;③ 指纹条件追加
(adaptive 历史指纹逐字节不变);④ θ_D=−inf 的 SteeringGate 恒开且
零告警;⑤ 无限预算下 admissible/charge 永不设限、spent 照实累计。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from trace_opd.common.config import load_config, objective_fingerprint
from trace_opd.steering.budget import TrajectoryBudget
from trace_opd.steering.gate import SteeringGate

YAML = """
run: {name: t, runs_root: runs}
alfworld: {data_root: /d, worker_python: /p}
train: {model_path: /m, train_file: /t.parquet, val_files: [/v1.parquet]%s}
eval: {}
steering: {enabled: true, doc_table: /tmp/d.jsonl%s}
"""

_MAXOPEN = (
    ", gate_mode: always_on, epsilon: 0.0, gate_adaptive: false"
    ", b_max: .inf, budget_pacing: false"
)


def _load(tmp_path, train_extra="", steer_extra=""):
    path = tmp_path / f"c{abs(hash((train_extra, steer_extra)))}.yaml"
    path.write_text(YAML % (train_extra, steer_extra))
    return load_config(path)


# ------------------------------------------------------------- config 校验

def test_maxopen_config_loads_without_theta_calibration(tmp_path):
    cfg = _load(tmp_path, steer_extra=_MAXOPEN)   # theta_d 缺省 0:免标定
    assert cfg.steering.gate_mode == "always_on"
    assert math.isinf(cfg.steering.b_max)


def test_always_on_requires_explicit_probe_and_adaptive_off(tmp_path):
    with pytest.raises(ValueError, match="epsilon"):
        _load(tmp_path, steer_extra=(
            ", gate_mode: always_on, gate_adaptive: false, b_max: .inf"
            ", budget_pacing: false"))          # epsilon 默认 0.05 ⇒ 拒
    with pytest.raises(ValueError, match="gate_adaptive"):
        _load(tmp_path, steer_extra=(
            ", gate_mode: always_on, epsilon: 0.0, b_max: .inf"
            ", budget_pacing: false"))          # gate_adaptive 默认 true ⇒ 拒
    with pytest.raises(ValueError, match="gate_mode"):
        _load(tmp_path, steer_extra=", gate_mode: sometimes, theta_d: 0.8")


def test_infinite_budget_rejects_pacing_and_bad_infinities(tmp_path):
    with pytest.raises(ValueError, match="budget_pacing"):
        _load(tmp_path, steer_extra=(
            ", gate_mode: always_on, epsilon: 0.0, gate_adaptive: false"
            ", b_max: .inf, budget_pacing: true"))
    with pytest.raises(ValueError, match="b_max"):
        _load(tmp_path, steer_extra=", theta_d: 0.8, b_max: -.inf")
    with pytest.raises(ValueError, match="b_max"):
        _load(tmp_path, steer_extra=", theta_d: 0.8, b_max: .nan")
    # adaptive 模式仍要求 theta 标定(防呆不因 v3.2 松动)
    with pytest.raises(ValueError, match="theta_d"):
        _load(tmp_path, steer_extra=", b_max: .inf, budget_pacing: false")


def test_fingerprint_moves_with_gate_mode_only_when_maxopen(tmp_path):
    base = objective_fingerprint(_load(tmp_path, steer_extra=", theta_d: 0.8"))
    explicit = objective_fingerprint(
        _load(tmp_path, steer_extra=", theta_d: 0.8, gate_mode: adaptive")
    )
    assert explicit == base                    # adaptive 不进 material(可续跑)
    maxopen = objective_fingerprint(_load(tmp_path, steer_extra=_MAXOPEN))
    assert maxopen != base                     # 全开=换行为分布 ⇒ 换指纹


# --------------------------------------------------------- gate/budget 行为

def test_gate_at_minus_inf_is_always_open_and_never_alarms():
    g = SteeringGate(theta_d=float("-inf"), eta_max=0.5, epsilon=0.0,
                     rate_bounds=(0.0, 1.0))
    rng = np.random.default_rng(0)
    decisions = [g.decide(d, rng) for d in (0.0, 1e-9, 0.03, 6.25)]
    assert all(d.base_open and d.eta == 0.5 for d in decisions)
    assert all(not d.probe_flipped for d in decisions)
    assert all(d.propensity == 1.0 for d in decisions)
    g._decisions = 100; g._opens = 100         # 率=1 也不许告警(语义关闭)
    assert g.rate_alarm() is None


def test_infinite_budget_never_limits_and_still_accounts_spend():
    b = TrajectoryBudget(b_max=float("inf"), c=2.5)
    assert b.available == float("inf") and b.remaining == float("inf")
    assert b.admissible_eta(0.5) == 0.5
    for _ in range(10_000):                    # 远超任何有限预算的干预量
        b.charge(0.5)
    assert b.spent == pytest.approx(10_000 * 2.5)   # 观察量照实累计
    assert b.admissible_eta(0.5) == 0.5             # 永不缩
    with pytest.raises(ValueError, match="pacing"):
        TrajectoryBudget(b_max=float("inf"), c=2.5, pacing=True, t_max=4)


def test_r_max_zero_means_uncapped_rejections():
    # v3.2(用户裁定 2026-08-06):r_max=0 = 无上限——cap 永不激活,拒绝
    # 计数照记为观察量;≥1 保留历史 cap 行为;负值拒绝。
    from trace_opd.steering.budget import TurnRejectCap

    cap = TurnRejectCap(r_max=0)
    for _ in range(1_000):                     # 远超历史 r_max=4 的拒绝量
        assert not cap.cap_active
        cap.record_reject()                    # 永不因 cap_active 抛错
    assert cap.count == 1_000 and not cap.cap_active
    cap.new_turn()
    assert cap.count == 0
    legacy = TurnRejectCap(r_max=2)            # 历史行为不受 0 语义影响
    legacy.record_reject(); legacy.record_reject()
    assert legacy.cap_active
    with pytest.raises(ValueError, match="r_max"):
        TurnRejectCap(r_max=-1)


def test_config_accepts_r_max_zero_and_rejects_negative(tmp_path):
    cfg = _load(tmp_path, steer_extra=_MAXOPEN + ", r_max: 0")
    assert cfg.steering.r_max == 0
    with pytest.raises(ValueError, match="r_max"):
        _load(tmp_path, steer_extra=_MAXOPEN + ", r_max: -1")
