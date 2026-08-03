"""D 主门 + ε 探针（规范 §2.3）。

可测性红线在**接口层**锁死：`decide()` 只接受 D 值与独立随机数——签名里
不存在 draft token / 后缀的任何入口（NF04 的 σ(h_j, B_rem, 冻结统计, U_j)）。

阈值 Θ_D：冻结校准值起步，训练步之间 robust-EMA 慢更新（分位数而非均值，
防模板 token 主导——NF07）；触发率带预注册上下限，超限告警并降级。
propensity 只记录、不进训练权重（NF07：RL 换测度中相消）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["GateDecision", "SteeringGate"]


@dataclass(frozen=True)
class GateDecision:
    eta: float             # 本位置的 η（0 = 不干预）
    d_value: float         # D_j（诊断/事后因果分析）
    base_open: bool        # 主门（D ≥ Θ_D）的确定性决定
    probe_flipped: bool    # ε 探针是否翻转了主门决定
    propensity: float      # 实际执行「开」的分配概率（事后 ITT 用，不进权重）


class SteeringGate:
    """一条轨迹内可复用的门控器（阈值/参数冻结；随机数流独立）。"""

    def __init__(
        self,
        *,
        theta_d: float,
        eta_max: float,
        epsilon: float = 0.05,
        rate_bounds: tuple[float, float] = (0.02, 0.30),
    ) -> None:
        if not (0.0 <= epsilon < 0.5):
            raise ValueError(f"epsilon must be in [0, 0.5), got {epsilon}")
        if not (0.0 < eta_max < 1.0):
            raise ValueError(f"eta_max must be in (0,1), got {eta_max}")
        self.theta_d = float(theta_d)
        self.eta_max = float(eta_max)
        self.epsilon = float(epsilon)
        self.rate_bounds = rate_bounds
        self._decisions = 0
        self._opens = 0

    def decide(self, d_value: float, rng: np.random.Generator) -> GateDecision:
        """门决定。签名即红线：无 token、无后缀入口。"""
        base_open = bool(d_value >= self.theta_d)
        flipped = bool(rng.random() < self.epsilon)
        open_now = base_open ^ flipped
        # 实际「开」的分配概率（确定性主门 + 独立翻转）
        propensity = (1.0 - self.epsilon) if base_open else self.epsilon
        self._decisions += 1
        self._opens += int(open_now)
        return GateDecision(
            eta=self.eta_max if open_now else 0.0,
            d_value=float(d_value),
            base_open=base_open,
            probe_flipped=flipped,
            propensity=propensity,
        )

    # ------------------------------------------------------------ diagnostics

    @property
    def open_rate(self) -> float:
        return self._opens / self._decisions if self._decisions else 0.0

    def rate_alarm(self) -> str | None:
        """触发率超出预注册区间 ⇒ 返回告警文本（调用方决定降级）。"""
        if self._decisions < 50:
            return None
        lo, hi = self.rate_bounds
        r = self.open_rate
        if r < lo:
            return f"gate open-rate {r:.3f} below floor {lo} — Θ_D too high for this distribution"
        if r > hi:
            return f"gate open-rate {r:.3f} above ceiling {hi} — Θ_D too low / distribution shifted"
        return None


def robust_quantile_update(current_theta: float, d_samples: np.ndarray, q: float = 0.85, ema: float = 0.9) -> float:
    """训练步之间的 Θ_D 慢更新：robust-EMA 到本步 D 分布的 q 分位（NF07 修法）。

    只在训练步边界调用（冻结语义：一步之内 Θ_D 不变——因果性）。
    """
    if len(d_samples) == 0:
        return current_theta
    target = float(np.quantile(np.asarray(d_samples, dtype=np.float64), q))
    return ema * current_theta + (1.0 - ema) * target
