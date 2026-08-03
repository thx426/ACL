"""grpo_loo advantage(verl core_algos 注册)—— LOO baseline/scale 数值验证。

直接按路径加载 core_algos 会拖入 verl 依赖链;此处把函数体单独 exec 出来
(与 test_behavior_weights 同思路的轻依赖加载:只需要 torch/numpy/defaultdict)。
"""
from __future__ import annotations

import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_SRC = (
    Path(__file__).resolve().parents[2]
    / "verl" / "verl" / "trainer" / "ppo" / "core_algos.py"
).read_text(encoding="utf-8")
_i = _SRC.index("def compute_grpo_loo_outcome_advantage")
_j = _SRC.index("@register_adv_est", _i)
_ns = {"torch": torch, "np": np, "defaultdict": defaultdict, "Optional": Optional,
       "AlgoConfig": object, "tuple": tuple}
exec(_SRC[_i:_j], _ns)  # noqa: S102 — 受控源码片段
grpo_loo = _ns["compute_grpo_loo_outcome_advantage"]


def _run(scores, index, norm=True):
    n = len(scores)
    tlr = torch.zeros(n, 4)
    tlr[:, -1] = torch.tensor(scores, dtype=torch.float32)
    mask = torch.ones(n, 4)
    cfg = {"norm_adv_by_std_in_grpo": norm}

    class _Cfg(dict):
        pass

    adv, ret = grpo_loo(
        token_level_rewards=tlr, response_mask=mask,
        index=np.array(index, dtype=object), config=_Cfg(cfg),
    )
    return adv[:, 0]


def test_loo_baseline_excludes_self():
    # 组 [1,0,0,0]:样本0 的 LOO baseline=0 ⇒ centred=1;原 GRPO baseline=0.25
    adv = _run([1, 0, 0, 0], ["a"] * 4, norm=False)
    assert adv[0] == pytest.approx(1.0)
    assert adv[1] == pytest.approx(-1 / 3)


def test_lone_dissenter_does_not_explode():
    # 7胜1负:负样本剩余组同质 std_loo=0 —— full-std floor 兜住
    scores = [1] * 7 + [0]
    adv = _run(scores, ["g"] * 8, norm=True)
    assert abs(adv[7]) < 5.0, f"lone dissenter advantage exploded: {adv[7]}"
    assert adv[7] < 0 and adv[0] > 0


def test_all_same_group_is_zero():
    adv = _run([1, 1, 1, 1], ["g"] * 4)
    assert torch.all(adv.abs() < 1e-6)


def test_scale_uses_max_of_loo_and_full():
    # 异质剩余组:std_loo>0;断言有限且与手算一致
    scores = [3.0, 1.0, 0.0, 0.0]
    adv = _run(scores, ["g"] * 4, norm=True)
    r = torch.tensor(scores)
    mean_loo0 = r[1:].mean()
    std_loo0 = r[1:].std()
    std_full = r.std()
    expect0 = (r[0] - mean_loo0) / (max(std_loo0, std_full) + 1e-6)
    assert adv[0] == pytest.approx(float(expect0), rel=1e-4)


def test_singleton_group_passthrough():
    adv = _run([0.7], ["solo"], norm=True)
    assert adv[0] == pytest.approx(0.7)


def test_two_groups_are_independent():
    adv = _run([1, 0, 0, 1], ["a", "a", "b", "b"], norm=False)
    assert adv[0] == pytest.approx(1.0) and adv[1] == pytest.approx(-1.0)
    assert adv[2] == pytest.approx(-1.0) and adv[3] == pytest.approx(1.0)
