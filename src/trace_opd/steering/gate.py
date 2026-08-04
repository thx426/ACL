"""D 主门 + ε 探针（规范 §2.3）。

可测性红线在**接口层**锁死：`decide()` 只接受 D 值与独立随机数——签名里
不存在 draft token / 后缀的任何入口（NF04 的 σ(h_j, B_rem, 冻结统计, U_j)）。

阈值 Θ_D:**率控自适应**(2026-08-04 用户裁定,推翻此前 IF-11 的 v1 冻结
——标定实测 384 轨迹/84 万观测:D 分位跨 step 极稳但 CDF 门槛区间极陡,
P85 与 P95 只差 0.01,固定阈值下门率不可控;控制量从「值」改为「率」):

    θ_D(t+1) = (1−β)·θ_D(t) + β·P_{1−q}(第 t 步实测 D 分布)

q = gate_target_rate(默认 10%),β = gate_ema_beta(默认 0.1)。数据源 =
上一 step 已完成轨迹的 D 直方图(滞后一步;分布 step 间稳定 ⇒ 滞后无害;
因果性无损——判定仍不看当前 token,步内 θ 冻结)。worker 级独立追踪
(GateThresholdTracker;每 worker 每 step 数千观测足够估分位,worker 间
差异二阶)。标定 run 的角色从「一锤定音」降为「EMA 初值」。门率带告警
保留为 EMA 失控的兜底。θ_D 逐轨迹值进记账(steer_theta_d)。
propensity 只记录、不进训练权重(NF07:RL 换测度中相消)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["GateDecision", "GateThresholdTracker", "SteeringGate"]


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


class GateThresholdTracker:
    """worker 进程级的 θ_D 率控自适应追踪器(线程安全)。

    生命周期:agent-loop worker 进程单例。每条轨迹结束时 ``observe()`` 并入
    该轨迹的 D 直方图;检测到 global_step 前进时(``maybe_step_update``)
    先用**上一 step 累积的直方图**做一次 EMA 更新、再清空累积——同一 step
    内所有轨迹用同一个 θ_D(步内冻结,因果性;滞后一步,分布稳定则无害)。

    直方图分位数:线性桶内插值;累积观测不足 ``min_obs`` 时跳过更新
    (信号太少不动阈值,避免小样本抖动)。
    """

    def __init__(
        self,
        *,
        theta_init: float,
        target_rate: float,
        ema_beta: float,
        d_hist_max: float,
        bins: int,
        min_obs: int = 500,
    ) -> None:
        import threading

        if not (0.0 < target_rate < 1.0):
            raise ValueError(f"target_rate must be in (0,1), got {target_rate}")
        if not (0.0 < ema_beta <= 1.0):
            raise ValueError(f"ema_beta must be in (0,1], got {ema_beta}")
        self.theta = float(theta_init)
        self.target_rate = float(target_rate)
        self.ema_beta = float(ema_beta)
        self.d_hist_max = float(d_hist_max)
        self.bins = int(bins)
        self.min_obs = int(min_obs)
        self._acc = [0] * self.bins
        self._acc_count = 0
        self._last_step: int | None = None
        self._lock = threading.Lock()

    def observe(self, d_hist: list[int], d_count: int) -> None:
        with self._lock:
            for i, v in enumerate(d_hist[: self.bins]):
                self._acc[i] += int(v)
            self._acc_count += int(d_count)

    def maybe_step_update(self, global_step: int | None) -> None:
        """step 边界检测 + EMA 更新(worker 内每个 rollout 入口调一次)。"""
        if global_step is None:
            return
        with self._lock:
            if self._last_step is None:
                self._last_step = int(global_step)
                return
            if int(global_step) == self._last_step:
                return
            # step 前进:用累积直方图更新 θ,再清空
            if self._acc_count >= self.min_obs:
                target = self._hist_quantile(1.0 - self.target_rate)
                self.theta = (1.0 - self.ema_beta) * self.theta + self.ema_beta * target
            self._acc = [0] * self.bins
            self._acc_count = 0
            self._last_step = int(global_step)

    def _hist_quantile(self, q: float) -> float:
        target = q * self._acc_count
        acc = 0
        for i, v in enumerate(self._acc):
            acc += v
            if acc >= target:
                lo = i * self.d_hist_max / self.bins
                frac = 1.0 - (acc - target) / v if v else 0.0
                return lo + frac * (self.d_hist_max / self.bins)
        return self.d_hist_max
