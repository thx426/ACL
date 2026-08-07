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


def test_backward_compatible_without_role_research():
    """v3.1 两段 batch(无 research 行):分臂读数与旧行为一致。"""
    m = compute_rollout_extra_metrics(
        {"steer_arm": ARM[: N_BARE + N_STEER], "won": WON[: N_BARE + N_STEER]}
    )
    assert m["rollout_extra/won_bare_rate/mean"] == 0.5
    assert m["rollout_extra/won_steered_rate/mean"] == 0.75
