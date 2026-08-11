"""剂量通用化行为锁(2026-08-11 用户裁定;docs/notes/2026-08-11-mentor-dosage-comparison.md §4)。

覆盖:tracker 分位直标/自标定 warmup/ratchet 坍缩防线/ema 兼容/预注册率带
告警;warmup 哨兵下 IntensityGate 全 bypass 且零 rng 消费(与干预关闭路径
rng 流一致——恒等前提);rollout_health_check 脚本的判读逻辑。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from trace_opd.steering.gate import THETA_WARMUP, GateThresholdTracker, IntensityGate


def _tracker(**kw):
    base = dict(theta_init=0.0, target_rate=0.05, ema_beta=0.1,
                d_hist_max=8.0, bins=1024, min_obs=100,
                update_mode="direct", ratchet=True)
    base.update(kw)
    return GateThresholdTracker(**base)


def _hist(bins, d_hist_max, samples):
    h = [0] * bins
    for s in samples:
        b = int(s / d_hist_max * bins)
        h[min(max(b, 0), bins - 1)] += 1
    return h


def _feed_step(tr, samples, step):
    tr.observe(_hist(tr.bins, tr.d_hist_max, samples), len(samples))
    tr.maybe_step_update(step)


# ------------------------------------------------- tracker: direct/warmup

def test_warmup_sentinel_until_first_calibration():
    tr = _tracker()
    assert tr.theta == THETA_WARMUP
    _feed_step(tr, [0.2] * 200, step=1)          # 首次调用只记 step
    assert tr.theta == THETA_WARMUP
    _feed_step(tr, [0.2] * 190 + [3.0] * 10, step=2)   # step 前进 → 直标落地
    assert tr.theta < THETA_WARMUP
    # P95 落在 0.2 与 3.0 之间(5% 尾部在 3.0)
    assert 0.2 <= tr.theta <= 3.2


def test_direct_tracks_distribution_without_ema_lag():
    tr = _tracker(ratchet=False)
    _feed_step(tr, [1.0] * 200, 1)
    _feed_step(tr, [1.0] * 190 + [2.0] * 10, 2)
    theta_a = tr.theta
    _feed_step(tr, [4.0] * 190 + [6.0] * 10, 3)   # 分布整体上移 → θ 一步跟上
    assert tr.theta > 3.9, f"direct must jump, got {tr.theta} (ema would lag near {theta_a})"


def test_ratchet_floor_blocks_collapse_chasing():
    tr = _tracker(ratchet=True)
    _feed_step(tr, [1.0] * 200, 1)
    _feed_step(tr, [1.0] * 190 + [2.0] * 10, 2)
    first = tr.theta
    # 熵坍缩:分布掉到 0.1 —— θ 不许跟着降(干预自动淡出而非门变宽)
    _feed_step(tr, [0.1] * 200, 3)
    assert tr.theta == pytest.approx(first)
    # 分布上移仍可抬 θ(ratchet 只挡下行)
    _feed_step(tr, [5.0] * 200, 4)
    assert tr.theta > first


def test_ema_mode_requires_explicit_init_and_keeps_legacy_formula():
    with pytest.raises(ValueError, match="theta_init"):
        _tracker(update_mode="ema", theta_init=0.0)
    tr = _tracker(update_mode="ema", theta_init=1.0, ratchet=False)
    _feed_step(tr, [3.0] * 200, 1)
    _feed_step(tr, [3.0] * 200, 2)
    # EMA β=0.1:一步只走 10% 路程(直标会直接到 ~3)
    assert 1.0 < tr.theta < 1.5


def test_alarm_fires_on_pre_registered_band_violation(caplog):
    tr = _tracker(theta_init=4.0, alarm_bounds=(0.05, 0.25), alarm_w_min=0.05)
    _feed_step(tr, [1.0] * 200, 1)
    # 全部质量在 1.0 ≥ w_min·θ=0.2 ⇒ 旧 θ 生效期非零干预率 ≈ 1.0 > 0.25
    with caplog.at_level("WARNING"):
        _feed_step(tr, [1.0] * 200, 2)
    assert tr.last_info["alarm"] and "above" in tr.last_info["alarm"]
    assert any("open-rate" in r.message for r in caplog.records)


def test_alarm_suppressed_during_warmup():
    tr = _tracker(alarm_bounds=(0.05, 0.25), alarm_w_min=0.05)
    _feed_step(tr, [1.0] * 200, 1)
    _feed_step(tr, [1.0] * 200, 2)     # warmup 落地步:零干预是设计语义
    assert tr.last_info["alarm"] is None


# --------------------------------------- gate: warmup 哨兵 = 恒等零干预

def test_warmup_theta_full_bypass_and_zero_rng_draw():
    g = IntensityGate(mode="entropy", eta_max=0.5, epsilon=0.05,
                      entropy_theta=THETA_WARMUP, w_min=0.05)
    rng = np.random.default_rng(7)
    ref = np.random.default_rng(7)
    for h in (0.0, 0.3, 1.5, 7.9):
        d = g.decide(entropy=h, d_value=0.0, rng=rng)
        assert d.eta == 0.0 and d.intensity == 0.0 and not d.base_open
    # bypass 位不 draw:rng 流与未消费状态逐位一致(恒等前提)
    assert rng.bit_generator.state == ref.bit_generator.state


# ------------------------------------------------- health check 脚本

_spec = importlib.util.spec_from_file_location(
    "rollout_health_check",
    Path(__file__).resolve().parents[2] / "scripts" / "rollout_health_check.py",
)
health = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(health)

STOP = 999


def _row(turns, *, won=False, arm=None, mid_eos=False, trunc=False):
    ids, spans = [], []
    for t in range(turns):
        start = len(ids)
        body = [1, 2, 3]
        if mid_eos and t == 0:
            body = [1, STOP, 2, 3]      # 中段停止符(P0-1 指纹)
        ids += body + [STOP]
        spans.append([start, len(ids)])
    d = {"response_token_ids": ids, "turn_token_spans": spans,
         "assistant_turns": turns, "won": won, "truncated": trunc,
         "invalid_format": 0, "response_token_len": len(ids)}
    if arm is not None:
        d["steering"] = {"steer_arm": arm, "steer_entropy_theta": 1.5,
                         "steer_gate_open_rate": 0.08}
    return d


def _write_dump(dirpath, rows):
    dirpath.mkdir(parents=True)
    for i, r in enumerate(rows):
        (dirpath / f"r{i:03d}.json").write_text(json.dumps(r))


def test_health_check_detects_quick_submit_and_mid_eos(tmp_path):
    rows = (
        [_row(2, won=True, arm=0) for _ in range(6)]
        + [_row(10, arm=0) for _ in range(2)]
        + [_row(3, won=True, arm=1, mid_eos=True)]
        + [_row(8, arm=1)]
    )
    _write_dump(tmp_path / "d", rows)
    loaded, n_res, n_bad = health.load_rows(str(tmp_path / "d"))
    assert (len(loaded), n_res, n_bad) == (10, 0, 0)
    assert health.bootstrap_stop_id(loaded) == STOP
    s = health.summarize(loaded, STOP)
    assert s["quick_submit_rate"] == pytest.approx(0.6)
    assert s["mid_eos_rows"] == 1 and s["junk_tokens"] > 0
    arms = {k: health.summarize(v, STOP) for k, v in health.by_arm(loaded).items()}
    assert arms["arm0"]["n"] == 8 and arms["arm1"]["mid_eos_rows"] == 1
    assert arms["arm1"]["theta_mean"] == pytest.approx(1.5)


def test_health_check_warmup_theta_sentinel_split(tmp_path):
    rows = [_row(4, arm=1), _row(4, arm=1)]
    rows[0]["steering"]["steer_entropy_theta"] = -1.0    # warmup 哨兵落账口径
    _write_dump(tmp_path / "d", rows)
    loaded, _, _ = health.load_rows(str(tmp_path / "d"))
    s = health.summarize(loaded, STOP)
    assert s["theta_warmup_rows"] == 1
    assert s["theta_mean"] == pytest.approx(1.5)         # 哨兵不进均值
