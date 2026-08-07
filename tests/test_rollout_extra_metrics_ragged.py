"""`compute_rollout_extra_metrics` 对 v4 三段 batch 的契约。

背景(2026-08-07 162 v4 冒烟 step1 实测崩溃):
`reward_extra_infos_dict` 由 trainer 侧 **per-key append** 组装,任一 key 在
部分行缺席就产生"锯齿字典"。当时 research 行的 `won` 是 bool、被
`not isinstance(v, bool)` 数值过滤器吃掉 ⇒ `steer_arm` 384 个值而 `won` 只
256 个 ⇒ `won[keep]` 布尔索引长度不匹配 ⇒ IndexError 崩整步。

三条契约:
  1. 等长 + role_research ⇒ won_bare_rate 只由 executor 的 bare 行贡献,
     不被 research 行(won 恒 0)稀释;
  2. 锯齿字典 ⇒ **不崩**(宁可少一条曲线也不能崩整步),并打出 ragged 警告;
  3. 无 role_research 键(v3.1 两段 batch)⇒ 行为不变,向后兼容。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "verl/verl/trainer/ppo/metric_utils.py"


def _load():
    """按文件路径加载,绕过 verl/__init__ 的重依赖(本地无 tensordict)。"""
    spec = importlib.util.spec_from_file_location("_metric_utils_under_test", _SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    compute_rollout_extra_metrics = _load().compute_rollout_extra_metrics
except Exception as err:  # 环境缺依赖时跳过而非误报失败
    pytestmark = pytest.mark.skip(reason=f"metric_utils 不可加载: {err}")

    def compute_rollout_extra_metrics(*_a, **_k):  # type: ignore[misc]
        raise AssertionError("unreachable")

N_BARE = N_STEER = N_RESEARCH = 4
# executor: 4 bare(2 赢) + 4 steered(3 赢);research: 4 行 won 恒 0
ARM = [0] * N_BARE + [1] * N_STEER + [0] * N_RESEARCH
WON = [1, 1, 0, 0] + [1, 1, 1, 0] + [0] * N_RESEARCH
ROLE = [0] * (N_BARE + N_STEER) + [1] * N_RESEARCH


def test_research_rows_do_not_dilute_won_bare_rate():
    m = compute_rollout_extra_metrics(
        {"steer_arm": list(ARM), "won": list(WON), "role_research": list(ROLE)}
    )
    # 只看 executor 的 4 条 bare:2/4
    assert m["rollout_extra/won_bare_rate/mean"] == 0.5
    assert m["rollout_extra/won_steered_rate/mean"] == 0.75
    assert abs(m["rollout_extra/steer_arm_gap/mean"] - 0.25) < 1e-9
    # 若 research 行混进 bare 臂,分母会变成 8、won_bare 掉到 2/8=0.25
    assert m["rollout_extra/won_bare_rate/mean"] != 0.25


def test_ragged_dict_does_not_crash(capsys):
    """won 短 4 个(复现当时的 384/256):必须不崩,且把锯齿打出来。"""
    ragged = {
        "steer_arm": list(ARM),                      # 12
        "won": WON[: N_BARE + N_STEER],              # 8  <- 短缺
        "role_research": list(ROLE),                 # 12
    }
    m = compute_rollout_extra_metrics(ragged)        # 不抛异常即为通过
    assert "rollout_extra/won/mean" in m             # 逐键均值仍照常产出
    # 分臂三条曲线在长度不一致时宁可缺席,也不能算出错值
    assert "rollout_extra/won_bare_rate/mean" not in m
    out = capsys.readouterr().out
    assert "ragged" in out and "won" in out


def test_research_metrics_are_not_diluted_by_executor_zero_padding():
    """research_* 只在 research 行上取均值。

    executor 行带 agent_loop._V4_RESEARCH_ZERO 零占位(为批内 schema 恒齐),
    若算进分母,24 行布局下会把每条 research 曲线稀释精确 3×(384/128)。
    2026-08-07 实测:no_finalize 真值 65% 被上报成 0.216。
    """
    # 8 executor 行(零占位) + 4 research 行,其中 3 条 no_finalize
    d = {
        "role_research": [0] * 8 + [1] * 4,
        "research_no_finalize": [0] * 8 + [1, 1, 1, 0],
        "research_doc_words": [0] * 8 + [400, 400, 400, 400],
    }
    m = compute_rollout_extra_metrics(d)
    assert m["rollout_extra/research_no_finalize/mean"] == 0.75, "应为 3/4,不是 3/12"
    assert m["rollout_extra/research_doc_words/mean"] == 400.0, "应为 400,不是 133.3"


def test_non_research_keys_still_average_over_full_batch():
    """只有 research_ 前缀改口径;executor 侧指标仍按全 batch 平均。"""
    d = {
        "role_research": [0] * 8 + [1] * 4,
        "won": [1] * 4 + [0] * 4 + [0] * 4,     # 4/12 全 batch
        "research_turns": [0] * 8 + [10] * 4,
    }
    m = compute_rollout_extra_metrics(d)
    assert abs(m["rollout_extra/won/mean"] - 4 / 12) < 1e-9
    assert m["rollout_extra/research_turns/mean"] == 10.0


def test_backward_compatible_without_role_research():
    """v3.1 两段 batch(无 research 行):分臂读数与旧行为一致。"""
    m = compute_rollout_extra_metrics(
        {"steer_arm": ARM[: N_BARE + N_STEER], "won": WON[: N_BARE + N_STEER]}
    )
    assert m["rollout_extra/won_bare_rate/mean"] == 0.5
    assert m["rollout_extra/won_steered_rate/mean"] == 0.75


def test_v33_steered_only_keys_not_diluted_by_sentinels():
    """审阅 C6:软门观察量(H_q/w̄/熵均值)只在 steered 行有定义;bare 与
    research 行的 0.0 哨兵不得进分母(否则 8+8 分臂下 H_q 被精确减半,
    与未稀释的 entropy_floor 参考线同图对比会误读成"已贴地板")。"""
    mod = _load()
    info = {
        "steer_arm": [0, 0, 1, 1, 0],          # 最后一行是 research 占位
        "role_research": [0, 0, 0, 0, 1],
        "won": [0, 1, 1, 0, 0],
        "steer_entropy_theta": [0.0, 0.0, 1.28, 1.28, 0.0],
        "steer_intensity_mean": [0.0, 0.0, 0.5, 0.7, 0.0],
        "steer_entropy_mean": [0.0, 0.0, 0.4, 0.2, 0.0],
        "steer_count": [0, 0, 3, 5, 0],        # 非 steered-only 键:仍全批均值
    }
    m = mod.compute_rollout_extra_metrics(info)
    assert m["rollout_extra/steer_entropy_theta/mean"] == pytest.approx(1.28)
    assert m["rollout_extra/steer_intensity_mean/mean"] == pytest.approx(0.6)
    assert m["rollout_extra/steer_entropy_mean/mean"] == pytest.approx(0.3)
    assert m["rollout_extra/steer_count/mean"] == pytest.approx(8 / 5)
    # 无 steer_arm(历史非分臂)⇒ 回退全批均值,不崩
    m2 = mod.compute_rollout_extra_metrics(
        {"steer_entropy_theta": [1.0, 1.0], "won": [0, 1]}
    )
    assert m2["rollout_extra/steer_entropy_theta/mean"] == pytest.approx(1.0)
