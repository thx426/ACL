"""steering 包测试 —— 规范 docs/pgs-final-spec.md 的机器可验证部分。

最重的一条是 test_sps_matches_analytic_kernel：小词表全模拟下，SPS
（draft~π → 接受/残差重采）的输出经验分布必须收敛到解析 b_S ——
命题 A（相对 b_S 的精确性）的实现级验证。
"""

from __future__ import annotations

import numpy as np
import pytest

from trace_opd.steering import (
    SteeringGate,
    SteeringKernel,
    SteeringLedger,
    TokenSteerRecord,
    TrajectoryBudget,
    TurnRejectCap,
    accept_or_resample,
    robust_quantile_update,
    truncated_increment,
)


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def _make_world(vocab=50, k=8, eta=0.5, c=2.5, seed=0):
    """随机全词表 π/p_T + 两侧 top-k 并集 S + 保尾核。返回全词表解析 b。"""
    rng = np.random.default_rng(seed)
    pi = _softmax(rng.normal(0, 1.5, vocab))
    pt = _softmax(rng.normal(0, 1.5, vocab))
    S = np.unique(np.concatenate([np.argsort(-pi)[:k], np.argsort(-pt)[:k]]))
    ell = truncated_increment(np.log(pt[S]), np.log(pi[S]), c)
    kern = SteeringKernel(token_ids=S, log_pi=np.log(pi[S]), ell_tilde=ell, eta=eta, c=c)
    # 全词表解析 b：S 内重分配、S 外恒 π
    b = pi.copy()
    M = pi[S].sum()
    A = float(np.sum(pi[S] * np.exp(eta * ell)))
    b[S] = M * pi[S] * np.exp(eta * ell) / A
    assert abs(b.sum() - 1.0) < 1e-12
    return pi, pt, S, kern, b


# --------------------------------------------------------------- kernel

def test_kernel_density_ratio_bound_and_mass():
    for seed in range(5):
        pi, _, S, kern, b = _make_world(seed=seed)
        # 全词表密度比界 e^{±2ηc}（S 外恒 1）
        ratio = b / pi
        assert np.all(ratio <= np.exp(2 * 0.5 * 2.5) + 1e-9)
        assert np.all(ratio >= np.exp(-2 * 0.5 * 2.5) - 1e-9)
        out = np.setdiff1d(np.arange(len(pi)), S)
        assert np.allclose(ratio[out], 1.0)
        # S 质量守恒
        assert abs(b[S].sum() - pi[S].sum()) < 1e-12


def test_kernel_eta_zero_is_identity():
    pi, pt, S, _, _ = _make_world()
    ell = truncated_increment(np.log(pt[S]), np.log(pi[S]), 2.5)
    kern = SteeringKernel(token_ids=S, log_pi=np.log(pi[S]), ell_tilde=ell, eta=0.0, c=2.5)
    for t in S:
        assert kern.accept_prob(int(t)) == 1.0
        assert abs(kern.log_ratio(int(t))) < 1e-12
    with pytest.raises(RuntimeError):
        kern.residual_distribution()   # b≡π ⇒ 残差为空，不允许触达


def test_kernel_bound_is_approachable():
    """极端构造逼近 2ηc 界（复审：界作为上确界是紧的）。"""
    pi = np.array([1e-9, 1.0 - 1e-9])
    ell = np.array([2.5, -2.5])       # 教师极力抬 token0、压 token1
    kern = SteeringKernel(
        token_ids=np.array([0, 1]), log_pi=np.log(pi), ell_tilde=ell, eta=0.5, c=2.5
    )
    # token0 的 log-ratio → η·(c−(−c)) = 2ηc = 2.5（分母质量几乎全在 −c 端）
    assert kern.log_ratio(0) > 2.5 - 1e-3
    assert kern.log_ratio(0) <= 2.5 + 1e-9


def test_truncation_rejects_bad_c():
    with pytest.raises(ValueError):
        truncated_increment(np.zeros(3), np.zeros(3), 0.0)


# --------------------------------------------------------------- SPS 统计等价

def test_sps_matches_analytic_kernel():
    """命题 A 实现验证：SPS 输出分布 ≡ 解析 b_S（TV < 3σ 统计界）。"""
    pi, _, S, kern, b = _make_world(vocab=50, k=8, eta=0.5, c=2.5, seed=3)
    rng = np.random.default_rng(7)
    n = 200_000
    drafts = rng.choice(len(pi), size=n, p=pi)          # draft ~ π（全词表）
    log_pi_lookup = {int(t): float(np.log(pi[t])) for t in S}
    counts = np.zeros(len(pi))
    n_rej = 0
    for y in drafts:
        out = accept_or_resample(
            kern, int(y), float(np.log(pi[y])), rng, log_pi_lookup=log_pi_lookup
        )
        counts[out.token_id] += 1
        n_rej += out.rejected
    emp = counts / n
    tv = 0.5 * float(np.abs(emp - b).sum())
    # 多项式采样的 TV 期望量级 ~ sqrt(V/(2πn))；50/2e5 ⇒ ~0.006，给 3 倍余量
    assert tv < 0.02, f"empirical vs analytic b: TV={tv:.4f}"
    # 拒绝率 = TV(b, π)（speculative 理论值）
    tv_bpi = 0.5 * float(np.abs(b - pi).sum())
    assert abs(n_rej / n - tv_bpi) < 0.01
    # 记账正确性：log_b 与解析值逐 token 一致（抽查）
    for t in S[:5]:
        got = kern.log_b(int(t), float(np.log(pi[t])))
        assert abs(got - float(np.log(b[t]))) < 1e-9


def test_sps_out_of_support_always_accepts():
    pi, _, S, kern, _ = _make_world()
    out_tok = int(np.setdiff1d(np.arange(len(pi)), S)[0])
    rng = np.random.default_rng(0)
    o = accept_or_resample(kern, out_tok, float(np.log(pi[out_tok])), rng)
    assert not o.rejected and o.token_id == out_tok
    assert abs(o.log_b - o.log_pi) < 1e-12          # 保尾：ratio=0


# --------------------------------------------------------------- gate

def test_gate_interface_is_causal_and_propensity_correct():
    g = SteeringGate(theta_d=1.0, eta_max=0.5, epsilon=0.1)
    rng = np.random.default_rng(0)
    n = 20_000
    opens_high = sum(g.decide(2.0, rng).eta > 0 for _ in range(n)) / n
    opens_low = sum(g.decide(0.1, rng).eta > 0 for _ in range(n)) / n
    assert abs(opens_high - 0.9) < 0.01     # base open, 10% 被探针翻关
    assert abs(opens_low - 0.1) < 0.01      # base closed, 10% 被探针翻开
    d = g.decide(2.0, rng)
    assert d.propensity == 0.9
    d2 = g.decide(0.1, rng)
    assert d2.propensity == 0.1


def test_gate_rate_alarm():
    g = SteeringGate(theta_d=10.0, eta_max=0.5, epsilon=0.0, rate_bounds=(0.05, 0.3))
    rng = np.random.default_rng(0)
    for _ in range(100):
        g.decide(0.1, rng)                  # 永远关门
    assert g.rate_alarm() is not None and "below floor" in g.rate_alarm()


def test_threshold_update_is_robust_quantile():
    theta = robust_quantile_update(1.0, np.array([0.0] * 85 + [10.0] * 15), q=0.85, ema=0.5)
    assert 0.5 < theta < 6.0                # EMA(1.0, P85≈10 邻域) —— 慢更新不跳变


# --------------------------------------------------------------- budget / cap

def test_budget_charges_and_exhausts():
    b = TrajectoryBudget(b_max=6.0, c=2.5)
    # δ = 2ηc = 2.5/次（η=0.5）⇒ 可扣 2 次整 + 1 次缩水
    assert b.admissible_eta(0.5) == 0.5
    b.charge(0.5)
    b.charge(0.5)
    assert abs(b.remaining - 1.0) < 1e-12
    eta3 = b.admissible_eta(0.5)
    assert abs(eta3 - 0.2) < 1e-12          # 1.0/(2·2.5)
    b.charge(eta3)
    assert b.remaining < 1e-12
    assert b.admissible_eta(0.5) == 0.0
    with pytest.raises(RuntimeError):
        b.charge(0.5)


def test_reject_cap_state_machine():
    cap = TurnRejectCap(r_max=2)
    cap.record_reject()
    assert not cap.cap_active
    cap.record_reject()
    assert cap.cap_active
    with pytest.raises(RuntimeError):
        cap.record_reject()
    cap.new_turn()
    assert not cap.cap_active


# --------------------------------------------------------------- ledger

def test_ledger_finalize_roundtrip():
    led = SteeringLedger()
    led.append(TokenSteerRecord(3, -1.0, -1.2, 0.5, 2.5, 0.8, True, False, 0.95, False, False))
    led.append(TokenSteerRecord(9, -2.0, -2.0, 0.0, 0.0, 0.1, False, False, 0.05, False, True))
    led.budget_spent = 2.5
    out = led.finalize()
    assert out["steer_positions"] == [3, 9]
    # steer_count = 实际倾斜位置数(η>0);记录行数另走 steer_record_rows
    assert out["steer_count"] == 1 and out["steer_record_rows"] == 2
    assert out["steer_budget_spent"] == 2.5
    with pytest.raises(ValueError):
        led.append(TokenSteerRecord(5, 0, 0, 0, 0, 0, False, False, 0.5, False, False))


def test_ledger_counters_and_hist_max():
    led = SteeringLedger(d_hist_max=9.0)   # c=3 ⇒ c²=9
    led.observe_d(8.5)                     # c=3 的合法值,不得落末桶外
    out = led.finalize()
    assert out["steer_d_hist_max"] == 9.0
    assert out["steer_n_candidates"] == 1
    # 计数器组齐全(率类审计的全量通道)
    for key in ("steer_n_base_open", "steer_n_probe_flipped",
                "steer_n_budget_blocked", "steer_n_bypassed",
                "steer_reject_count", "steer_record_rows"):
        assert key in out


def test_ledger_ell_hist_and_token_pairs():
    from trace_opd.steering.ledger import ELL_HIST_BINS

    led = SteeringLedger()
    led.observe_ell([-3.0, 0.5, 15.0])       # 含一个超范围值(入端桶)
    assert led.ell_count == 3
    assert sum(led.ell_hist) == 3
    assert led.ell_hist[-1] == 1              # +15 → 末桶
    led.append(TokenSteerRecord(2, -1.0, -1.5, 0.5, 2.5, 0.9, True, False, 0.95,
                                True, False, 7.5, 101, 202))
    out = led.finalize()
    assert out["steer_draft_token"] == [101]
    assert out["steer_final_token"] == [202]
    assert out["steer_ell_count"] == 3
    assert len(out["steer_ell_hist"]) == ELL_HIST_BINS
    # viz 数据面(2026-08-04 实验侧规格)
    led.observe_d(0.5, position=7)
    out2 = led.finalize()
    assert out2["steer_cand_positions"] == [7] and out2["steer_d_all"] == [0.5]
    assert 0.0 <= out2["steer_gate_open_rate"] <= 1.0
    assert isinstance(out2["steer_pi_top16"], list)


def test_adaptive_tracker_restores_target_rate_after_shift():
    """率控验收(变更请求验收标准):D 分布平移后,门率在若干 step 内回到
    q±2%;θ 轨迹平滑无跳变。"""
    from trace_opd.steering.gate import GateThresholdTracker

    rng = np.random.default_rng(0)
    bins, hmax, q = 2048, 6.25, 0.10

    def hist_of(samples):
        h = [0] * bins
        for v in samples:
            b = int(v / hmax * bins)
            h[min(max(b, 0), bins - 1)] += 1
        return h

    def open_rate(theta, samples):
        return float(np.mean(samples >= theta))

    tr = GateThresholdTracker(
        theta_init=0.09, target_rate=q, ema_beta=0.3, d_hist_max=hmax, bins=bins,
    )
    thetas = []
    # 阶段一:基准分布(lognormal 模拟陡 CDF),门率应稳在 q 附近
    for step in range(1, 9):
        samples = rng.lognormal(-3.2, 0.8, 6000)
        tr.observe(hist_of(samples), len(samples))
        tr.maybe_step_update(step)
        thetas.append(tr.theta)
    base_rate = open_rate(tr.theta, rng.lognormal(-3.2, 0.8, 40000))
    assert abs(base_rate - q) < 0.02, f"baseline rate {base_rate:.3f}"
    # 阶段二:分布整体平移(×2,CDF 平移远超 P85-P95 间距)⇒ 固定阈值下
    # 门率会飙;自适应应在几步内拉回
    for step in range(9, 25):
        samples = rng.lognormal(-3.2 + np.log(2), 0.8, 6000)
        tr.observe(hist_of(samples), len(samples))
        tr.maybe_step_update(step)
        thetas.append(tr.theta)
    shifted_rate = open_rate(tr.theta, rng.lognormal(-3.2 + np.log(2), 0.8, 40000))
    assert abs(shifted_rate - q) < 0.02, f"post-shift rate {shifted_rate:.3f}"
    # θ 平滑:相邻步变化受 β 界(无跳变)
    jumps = [abs(b - a) / max(a, 1e-9) for a, b in zip(thetas, thetas[1:])]
    assert max(jumps) < 0.5


def test_adaptive_tracker_step_semantics():
    from trace_opd.steering.gate import GateThresholdTracker

    tr = GateThresholdTracker(
        theta_init=1.0, target_rate=0.1, ema_beta=0.5, d_hist_max=4.0, bins=64,
    )
    h = [0] * 64
    h[8] = 1000                       # 全部质量在 ~0.53 附近
    tr.observe(h, 1000)
    tr.maybe_step_update(1)           # 首见 step:只登记,不更新
    assert tr.theta == 1.0
    tr.observe(h, 1000)
    tr.maybe_step_update(1)           # 同 step:不更新
    assert tr.theta == 1.0
    tr.maybe_step_update(2)           # step 前进:EMA 更新
    assert tr.theta < 1.0
    t2 = tr.theta
    tr.maybe_step_update(3)           # 无观测(不足 min_obs):跳过
    assert tr.theta == t2
