"""v3.3 机制行为锁(docs/pgs-v33-mixed-policy-design.md M1–M4)。

覆盖:IntensityGate 五档语义与旧门等价性、pgs_asym_grpo estimator 数值、
α cosine 调度、shaped_reject 权重、config 联动校验与指纹条件追加、
steered 臂停产判定。verl 行为级测试直接 import vendored 树(本地已具
torch/tensordict;CI 缺依赖时整段 skip 而非假绿)。
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trace_opd.common.config import load_config, objective_fingerprint
from trace_opd.steering.gate import GateThresholdTracker, IntensityGate, SteeringGate

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "verl") not in sys.path:
    sys.path.insert(0, str(_ROOT / "verl"))

torch = pytest.importorskip("torch")
from verl.trainer.ppo.core_algos import (  # noqa: E402
    compute_grpo_loo_outcome_advantage,
    compute_pgs_asym_outcome_advantage,
    pgs_steered_alpha,
)
from verl.trainer.ppo.steering_weights import expand_behavior_weights  # noqa: E402


# ------------------------------------------------------- A. IntensityGate

def test_adaptive_mode_matches_legacy_steering_gate_field_by_field():
    """等价适配红线:adaptive 档的 decide 产出必须与 v3.0 SteeringGate
    逐字段一致(同 θ/ε/η_max、同 rng 流)。"""
    for seed in range(5):
        old = SteeringGate(theta_d=0.5, eta_max=0.5, epsilon=0.1)
        new = IntensityGate(mode="adaptive", theta_d=0.5, eta_max=0.5, epsilon=0.1)
        r1, r2 = np.random.default_rng(seed), np.random.default_rng(seed)
        for d in (0.0, 0.3, 0.5, 0.9, 4.0):
            a = old.decide(d, r1)
            b = new.decide(entropy=0.0, d_value=d, rng=r2)
            assert (a.eta, a.base_open, a.probe_flipped, a.propensity) == (
                b.eta, b.base_open, b.probe_flipped, b.propensity)


def test_always_on_mode_full_intensity_no_probe():
    g = IntensityGate(mode="always_on", eta_max=0.5, epsilon=0.0)
    d = g.decide(entropy=0.0, d_value=0.0, rng=np.random.default_rng(0))
    assert d.eta == 0.5 and d.intensity == 1.0 and d.propensity == 1.0
    assert not g.needs_teacher


def test_entropy_mode_soft_ramp_and_bypass():
    g = IntensityGate(mode="entropy", eta_max=0.5, epsilon=0.0,
                      entropy_theta=2.0, w_min=0.05)
    assert not g.needs_teacher
    rng = np.random.default_rng(0)
    # 满强度:h >= H_q
    d = g.decide(entropy=3.0, d_value=0.0, rng=rng)
    assert d.eta == pytest.approx(0.5) and d.intensity == 1.0
    # 半强度:w = h/H_q
    d = g.decide(entropy=1.0, d_value=99.0, rng=rng)  # 熵档不读 D
    assert d.intensity == pytest.approx(0.5) and d.eta == pytest.approx(0.25)
    # w < w_min ⇒ 关门(无干预、propensity=0)
    d = g.decide(entropy=0.05, d_value=0.0, rng=rng)
    assert d.eta == 0.0 and not d.base_open and d.propensity == 0.0


def test_entropy_mode_probe_zeroes_intensity():
    g = IntensityGate(mode="entropy", eta_max=0.5, epsilon=0.4,
                      entropy_theta=1.0, w_min=0.05)
    seen_flip = seen_pass = False
    for seed in range(64):
        d = g.decide(entropy=5.0, d_value=0.0,
                     rng=np.random.default_rng(seed))
        if d.probe_flipped:
            seen_flip = True
            assert d.eta == 0.0 and d.intensity == 0.0
            assert d.propensity == pytest.approx(0.6)
        else:
            seen_pass = True
            assert d.eta == pytest.approx(0.5)
    assert seen_flip and seen_pass


def test_d_var_and_soft_or_formulas():
    gd = IntensityGate(mode="d_var", eta_max=0.5, theta_d=0.2, w_min=0.05)
    assert gd.needs_teacher
    assert gd.base_w(d_value=0.1) == pytest.approx(0.5)
    assert gd.base_w(d_value=1.0) == 1.0
    gs = IntensityGate(mode="soft_or", eta_max=0.5, theta_d=0.2,
                       entropy_theta=2.0, w_min=0.05)
    # w = 1 − (1−0.5)(1−0.5) = 0.75
    assert gs.base_w(entropy=1.0, d_value=0.1) == pytest.approx(0.75)
    assert gs.needs_teacher


def test_intensity_gate_calibration_guards():
    with pytest.raises(ValueError):
        IntensityGate(mode="d_var", eta_max=0.5, theta_d=0.0)
    with pytest.raises(ValueError):
        IntensityGate(mode="entropy", eta_max=0.5, entropy_theta=0.0)
    with pytest.raises(ValueError):
        IntensityGate(mode="nope", eta_max=0.5)


def test_tracker_floor_clamps_init_and_updates():
    tr = GateThresholdTracker(theta_init=0.2, target_rate=0.05, ema_beta=1.0,
                              d_hist_max=8.0, bins=64, min_obs=1, floor=1.5)
    assert tr.theta == 1.5          # 初值被地板抬起
    hist = [0] * 64
    hist[1] = 100                    # 全部观测在 ~0.19 ⇒ 分位远低于地板
    tr.maybe_step_update(1)
    tr.observe(hist, 100)
    tr.maybe_step_update(2)
    assert tr.theta == 1.5          # EMA 目标 < floor ⇒ 钳在地板


# ----------------------------------------------- B. pgs_asym_grpo estimator

def _adv(scores, arms, uids, alpha=1.0, cap=0.0):
    n = len(scores)
    rewards = torch.zeros(n, 4)
    rewards[:, -1] = torch.tensor(scores, dtype=torch.float32)
    mask = torch.ones(n, 4)
    adv, ret = compute_pgs_asym_outcome_advantage(
        token_level_rewards=rewards, response_mask=mask,
        index=np.array(uids, dtype=object),
        steer_arm=np.array(arms), alpha=alpha, steered_cap=cap)
    return adv[:, 0].tolist()


def test_asym_bare_loo_and_steered_positive_excess():
    # 一组:4 bare(1 胜)+ 2 steered(1 胜 1 负),α=0.5
    scores = [1, 0, 0, 0, 1, 0]
    arms = [0, 0, 0, 0, 1, 1]
    uids = ["t"] * 6
    a = _adv(scores, arms, uids, alpha=0.5)
    # bare LOO:胜行 1−0/3=1;负行 0−1/3
    assert a[0] == pytest.approx(1.0)
    assert a[1] == pytest.approx(-1 / 3)
    # steered:mean_bare=0.25;胜行 (1−0.25)×0.5;负行 clamp→0(不惩罚)
    assert a[4] == pytest.approx(0.75 * 0.5)
    assert a[5] == 0.0


def test_asym_all_bare_group_equals_grpo_loo():
    """research 组(uid 自成组、arm 全 0)必须与 grpo_loo no-std 逐值一致。"""
    scores = [1.0, 0.0, 0.0, 1.0, 1.0]
    rewards = torch.zeros(5, 3)
    rewards[:, -1] = torch.tensor(scores)
    mask = torch.ones(5, 3)
    uids = np.array(["r"] * 5, dtype=object)
    ref, _ = compute_grpo_loo_outcome_advantage(
        token_level_rewards=rewards, response_mask=mask, index=uids,
        config={"norm_adv_by_std_in_grpo": False})
    got, _ = compute_pgs_asym_outcome_advantage(
        token_level_rewards=rewards, response_mask=mask, index=uids,
        steer_arm=np.zeros(5, dtype=np.int64), alpha=1.0)
    assert torch.allclose(ref, got)


def test_asym_orphan_steered_group_degrades_to_zero():
    """部分失败组(timeout 取消慢 bare 后 steered 独存;审阅 C0):不 raise
    崩整步,也不 remap unknown(会绕过 α 调度)——整组显式零 advantage。"""
    a = _adv([1, 0], [1, 1], ["t", "t"], alpha=1.0)
    assert a == [0.0, 0.0]
    # 与正常组共存时互不影响
    a2 = _adv([1, 0, 1, 0], [1, 1, 0, 0], ["orphan", "orphan", "ok", "ok"])
    assert a2[0] == 0.0 and a2[1] == 0.0
    assert a2[2] == pytest.approx(1.0) and a2[3] == pytest.approx(-1.0)


def test_v33_config_rejects_nan_inf_on_new_float_fields(tmp_path, monkeypatch):
    """审阅 C5:新增浮点字段必须过 isfinite 纪律(IF-13)。"""
    for bad in (", steered_adv_weight: .nan", ", entropy_floor: .inf",
                ", w_beh_shaped_gamma: .nan"):
        with pytest.raises(ValueError, match="有限数"):
            _load(tmp_path, monkeypatch, steer_extra=bad)


def test_v33_tracker_step_update_sits_before_v4_fork():
    """审阅 C1/C2:maybe_step_update 必须在 run() 的 v3/v4 分流之前的公共
    位置(v4 executor 行走 _v4_coord 分支,原 elif 挂点永不经过 ⇒ θ_D/H_q
    静默冻结)。文本断言锁位置关系。"""
    src = (Path(__file__).resolve().parents[1]
           / "src/trace_opd/train/agent_loop.py").read_text(encoding="utf-8")
    run_body = src.split("async def run(", 1)[1]
    tick = run_body.index("maybe_step_update")
    v4_fork = run_body.index("_v4_role")
    assert tick < v4_fork, "tracker step update must precede the v4 role fork"
    assert run_body.count("maybe_step_update") == 2  # gate + ent,各恰一处


def test_asym_steered_cap_rescales():
    # 3 条 steered 全胜(bare 全败):RAW 超额和 3 > cap=1.5 ⇒ 先减半再乘 α
    # (MENTOR 顺序,审阅 L22——cap 若作用在 α 后总额,cosine 退火期会被
    # cap 钉平)。α=0.5 时:0.5(capped excess)×0.5 = 0.25;旧顺序会给 0.5。
    scores = [0, 0, 1, 1, 1]
    arms = [0, 0, 1, 1, 1]
    a = _adv(scores, arms, ["t"] * 5, alpha=1.0, cap=1.5)
    assert a[2] == pytest.approx(0.5)
    a2 = _adv(scores, arms, ["t"] * 5, alpha=0.5, cap=1.5)
    assert a2[2] == pytest.approx(0.25)


def test_asym_lone_unknown_inside_labeled_group_zeroed():
    # labeled 组内孤儿 unknown(丢标签)不得 raw-through 拿全批最大 advantage
    # (审阅 L2);纯 unknown 单行组保持 grpo_loo n==1 惯例(raw through)。
    a = _adv([1, 0, 1], [0, 0, -1], ["t", "t", "t"])
    assert a[2] == 0.0
    b = _adv([0.7], [-1], ["solo"])
    assert b[0] == pytest.approx(0.7)


def test_steered_alpha_schedule():
    assert pgs_steered_alpha(0, alpha0=1.0, decay_steps=100) == pytest.approx(1.0)
    assert pgs_steered_alpha(50, alpha0=1.0, decay_steps=100) == pytest.approx(0.5)
    assert pgs_steered_alpha(100, alpha0=1.0, decay_steps=100) == 0.0
    assert pgs_steered_alpha(10_000, alpha0=1.0, decay_steps=100) == 0.0
    assert pgs_steered_alpha(10_000, alpha0=0.7, decay_steps=0) == pytest.approx(0.7)


# ------------------------------------------------- C. shaped_reject weights

def test_shaped_reject_weights_only_rejections():
    dense = expand_behavior_weights(
        positions=[2, 5, 7], log_pi=[np.log(0.9), np.log(0.1), np.log(0.01)],
        log_b=[0.0, 0.0, 0.0], response_len=10, mode="shaped_reject",
        log_ratio_cap=None, rejected=[0, 1, 1], shaped_gamma=0.1)
    assert dense[2] == pytest.approx(1.0)          # 接受位不动
    assert dense[5] == pytest.approx(0.1 / 0.2)    # f(0.1)=0.5
    assert dense[7] == pytest.approx(0.01 / 0.11)  # 低概率教学位仍留 ~9% 梯度
    assert dense[0] == 1.0


def test_shaped_reject_requires_aligned_flags():
    with pytest.raises(ValueError, match="rejected"):
        expand_behavior_weights(
            positions=[1], log_pi=[0.0], log_b=[0.0], response_len=4,
            mode="shaped_reject", rejected=None)
    with pytest.raises(ValueError, match="unknown w_beh mode"):
        expand_behavior_weights(
            positions=[], log_pi=[], log_b=[], response_len=4, mode="off")


# --------------------------------------------------- D. config 联动与指纹

_YAML = """
run: {name: t, runs_root: runs}
alfworld: {data_root: /d, worker_python: /p}
train: {model_path: /m, train_file: /t.parquet, val_files: [/v1.parquet],
        agent_task_timeout_s: 0, rollout_n: 16%s}
eval: {}
research: {enabled: true, backend: external, tool_protocol: fncall, gt: "off", session_token_budget: 24576, max_prompt_tokens: 1024}
narration: {backend: engine, max_prompt_tokens: 6144}
roles:
  researcher:
    endpoint: "http://x/v1"
    model: m
    api_key_env: TEST_V33_KEY
    mode: external_pool
    max_concurrency: 8
    extra_body: {chat_template_kwargs: {enable_thinking: false}}
steering: {enabled: true, doc_source: researcher, theta_d: 0.8%s}
"""


def _load(tmp_path, monkeypatch, train_extra="", steer_extra=""):
    monkeypatch.setenv("TEST_V33_KEY", "k")
    p = tmp_path / f"v{abs(hash((train_extra, steer_extra)))}.yaml"
    p.write_text(_YAML % (train_extra, steer_extra))
    return load_config(p)


def test_asym_requires_estimator_and_arms(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="pgs_asym_grpo"):
        _load(tmp_path, monkeypatch,
              steer_extra=", adv_mode: asym, steered_fraction: 0.5")
    with pytest.raises(ValueError, match="adv_mode=asym"):
        _load(tmp_path, monkeypatch,
              train_extra=", adv_estimator: pgs_asym_grpo")
    with pytest.raises(ValueError, match="分臂"):
        _load(tmp_path, monkeypatch,
              train_extra=", adv_estimator: pgs_asym_grpo",
              steer_extra=", adv_mode: asym")
    cfg = _load(tmp_path, monkeypatch,
                train_extra=", adv_estimator: pgs_asym_grpo",
                steer_extra=", adv_mode: asym, steered_fraction: 0.5")
    assert cfg.steering.adv_mode == "asym"


def test_entropy_gate_requires_calibration(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="entropy_theta_init"):
        _load(tmp_path, monkeypatch, steer_extra=", gate_mode: entropy")
    with pytest.raises(ValueError, match="entropy_floor"):
        _load(tmp_path, monkeypatch,
              steer_extra=", gate_mode: entropy, entropy_theta_init: 1.2")
    cfg = _load(tmp_path, monkeypatch,
                steer_extra=(", gate_mode: entropy, entropy_theta_init: 1.2"
                             ", entropy_floor: 0.4"))
    assert cfg.steering.gate_mode == "entropy"


def test_fingerprint_legacy_defaults_unchanged_new_knobs_append(tmp_path, monkeypatch):
    base = _load(tmp_path, monkeypatch)
    fp0 = objective_fingerprint(base)
    # 新字段全部默认时,material 不得含任何 v3.3 条件段(指纹兼容既有 run)
    asym = _load(tmp_path, monkeypatch,
                 train_extra=", adv_estimator: pgs_asym_grpo",
                 steer_extra=(", adv_mode: asym, steered_fraction: 0.5"
                              ", steered_adv_decay_steps: 200"))
    ent = _load(tmp_path, monkeypatch,
                steer_extra=(", gate_mode: entropy, entropy_theta_init: 1.2"
                             ", entropy_floor: 0.4"))
    shaped = _load(tmp_path, monkeypatch,
                   steer_extra=", w_beh_mode: shaped_reject")
    fps = {fp0, objective_fingerprint(asym), objective_fingerprint(ent),
           objective_fingerprint(shaped)}
    assert len(fps) == 4       # 每个 v3.3 开关都改变优化目标指纹


# --------------------------------------------------- E. steered 臂停产

def test_steered_arm_retirement_gate():
    from trace_opd.steering.arms import steered_retired

    assert not steered_retired("asym", 100, 99)
    assert steered_retired("asym", 100, 100)
    assert steered_retired("asym", 100, 5000)
    assert not steered_retired("asym", 0, 5000)          # 不衰减 ⇒ 不停产
    assert not steered_retired("legacy_shared", 100, 5000)
    assert not steered_retired("asym", 100, None)        # step 未知 ⇒ 保守不停
