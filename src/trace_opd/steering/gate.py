"""D 主门 + ε 探针（规范 §2.3）。

可测性红线在**接口层**锁死：`decide()` 只接受 D 值与独立随机数——签名里
不存在 draft token / 后缀的任何入口（NF04 的 σ(h_j, B_rem, 冻结统计, U_j)）。

阈值 Θ_D:**率控自适应**(2026-08-04 用户裁定,推翻此前 IF-11 的 v1 冻结)。
必要性论据(标定 v2 修正:v1 的「CDF 极陡/P85-P95 差 0.01」是 64 桶分辨率
伪影,2048 桶真值 P85=0.0203/P95=0.0721,静态可辨识性其实良好):**训练中
策略漂移使 D 分布整体移动**,冻结阈值必然过时;控制量从「值」改为「率」,
门率构造性稳定在目标 q:

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

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "THETA_WARMUP", "GateDecision", "GateThresholdTracker", "IntensityGate",
    "SteeringGate",
]


@dataclass(frozen=True)
class GateDecision:
    eta: float             # 本位置的 η（0 = 不干预;软强度档 = η_max·w）
    d_value: float         # D_j（诊断/事后因果分析）
    base_open: bool        # 主门确定性决定(软档:base_w ≥ w_min)
    probe_flipped: bool    # ε 探针是否翻转了主门决定(软档:是否被置零)
    propensity: float      # 实际执行「开」的分配概率（事后 ITT 用，不进权重）
    intensity: float = 1.0 # 软强度 w ∈[0,1](v3.3 M4;硬门档恒 1/0)


class SteeringGate:
    """一条轨迹内可复用的门控器（阈值/参数冻结；随机数流独立）。"""

    #: 跨 turn 可变状态注册表(Engine-Resident SPS 的 wire 往返按它驱动,
    #: steering/wire.py;其余属性是构造参数)。两个门类同名同义。
    _WIRE_MUTABLE = ("_decisions", "_opens")

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


class IntensityGate:
    """v3.3 M4 统一强度门(docs/pgs-v33-mixed-policy-design.md):门从
    「开/关 + 恒 η_max」重构为强度函数 w∈[0,1],η = η_max·w。五档:

      always_on  w ≡ 1(maxopen,v3.2 等价适配);
      adaptive   w ∈ {0,1} 硬门 D≥θ_D + ε 双向探针(v3.0 等价适配——
                 decide 产出与 SteeringGate 逐字段一致);
      entropy    G-E:w = min(1, h/H_q)(h=学生截断熵,draft 侧免费;
                 不需要教师信号 ⇒ needs_teacher=False,支持段级 bypass);
      d_var      G-D 软化:w = min(1, D/θ_D);
      soft_or    G-SOR:w = 1 − (1−w_E)(1−w_D)(两盲区互补)。

    软强度档(entropy/d_var/soft_or)的 ε 探针是**单向置零**(base_w>0 的
    位置以 ε 概率 w→0;连续强度下「翻开」无定义),propensity = 1−ε
    (base_w>0)/ 0(base_w=0)。硬门档保留 v3.0 双向翻转语义。
    阈值(θ_D / H_q)一条轨迹内冻结(因果性;自适应经 tracker 跨 step 慢更)。

    ⚠ 两个开门率口径勿混(2026-08-09 CD-006 曾据此误判"熵门未标定"):
    ``entropy_target_rate``(默认 0.05)标定的是 **w=1 满强度位置**的占比
    (H_q 追踪 H 分布的对应分位);而 base_open 的条件是 w0 ≥ w_min,即
    H ≥ w_min·H_q —— **非零干预率**是另一个(远松的)口径,观测 35–46%
    并非失标。段级 bypass 依赖的是后者(整段全部 w0 < w_min 才跳)。
    """

    #: 跨 turn 可变状态注册表(wire 往返;见 SteeringGate 同名注释)
    _WIRE_MUTABLE = ("_decisions", "_opens")

    NEEDS_TEACHER = {"adaptive": True, "always_on": False, "entropy": False,
                     "d_var": True, "soft_or": True}

    def __init__(
        self,
        *,
        mode: str,
        eta_max: float,
        epsilon: float = 0.0,
        theta_d: float = 0.0,
        entropy_theta: float = 0.0,
        w_min: float = 0.05,
        rate_bounds: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        if mode not in self.NEEDS_TEACHER:
            raise ValueError(f"unknown IntensityGate mode {mode!r}")
        if not (0.0 <= epsilon < 0.5):
            raise ValueError(f"epsilon must be in [0, 0.5), got {epsilon}")
        if not (0.0 < eta_max < 1.0):
            raise ValueError(f"eta_max must be in (0,1), got {eta_max}")
        if mode in ("adaptive", "d_var", "soft_or") and theta_d <= 0:
            raise ValueError(f"mode {mode!r} requires theta_d > 0, got {theta_d}")
        if mode in ("entropy", "soft_or") and entropy_theta <= 0:
            raise ValueError(
                f"mode {mode!r} requires entropy_theta > 0, got {entropy_theta}"
            )
        self.mode = mode
        self.eta_max = float(eta_max)
        self.epsilon = float(epsilon)
        self.theta_d = float(theta_d)
        self.entropy_theta = float(entropy_theta)
        self.w_min = float(w_min)
        self.rate_bounds = rate_bounds
        self._decisions = 0
        self._opens = 0

    @property
    def needs_teacher(self) -> bool:
        return self.NEEDS_TEACHER[self.mode]

    def base_w(self, *, entropy: float = 0.0, d_value: float = 0.0) -> float:
        """探针前的确定性强度(纯函数;entropy 档不读 d_value,可在
        verify 之前调用——段级 bypass 判定的依据)。"""
        if self.mode == "always_on":
            return 1.0
        if self.mode == "adaptive":
            return 1.0 if d_value >= self.theta_d else 0.0
        w_e = min(1.0, max(0.0, entropy) / self.entropy_theta) \
            if self.entropy_theta > 0 else 0.0
        if self.mode == "entropy":
            return w_e
        w_d = min(1.0, max(0.0, d_value) / self.theta_d)
        if self.mode == "d_var":
            return w_d
        return 1.0 - (1.0 - w_e) * (1.0 - w_d)   # soft_or

    def decide(
        self, *, entropy: float, d_value: float, rng: np.random.Generator
    ) -> GateDecision:
        """终判(含探针)。签名红线沿 SteeringGate:只接受观测信号与独立
        随机数,无 token / 后缀入口。"""
        w0 = self.base_w(entropy=entropy, d_value=d_value)
        if self.mode == "adaptive":
            # v3.0 硬门等价:双向翻转,propensity 按 base_open 分派。
            base_open = w0 >= 1.0
            flipped = bool(rng.random() < self.epsilon)
            open_now = base_open ^ flipped
            w = 1.0 if open_now else 0.0
            propensity = (1.0 - self.epsilon) if base_open else self.epsilon
        else:
            base_open = w0 >= self.w_min
            effective = w0 if base_open else 0.0
            flipped = bool(
                effective > 0.0 and self.epsilon > 0.0
                and rng.random() < self.epsilon
            )
            w = 0.0 if flipped else effective
            propensity = (1.0 - self.epsilon) if effective > 0.0 else 0.0
        self._decisions += 1
        self._opens += int(w > 0.0)
        return GateDecision(
            eta=self.eta_max * w,
            d_value=float(d_value),
            base_open=base_open,
            probe_flipped=flipped,
            propensity=propensity,
            intensity=w,
        )

    @property
    def open_rate(self) -> float:
        return self._opens / self._decisions if self._decisions else 0.0

    def rate_alarm(self) -> str | None:
        if self._decisions < 50:
            return None
        lo, hi = self.rate_bounds
        r = self.open_rate
        if r < lo:
            return f"intensity open-rate {r:.3f} below floor {lo}"
        if r > hi:
            return f"intensity open-rate {r:.3f} above ceiling {hi}"
        return None


def robust_quantile_update(current_theta: float, d_samples: np.ndarray, q: float = 0.85, ema: float = 0.9) -> float:
    """训练步之间的 Θ_D 慢更新：robust-EMA 到本步 D 分布的 q 分位（NF07 修法）。

    只在训练步边界调用（冻结语义：一步之内 Θ_D 不变——因果性）。
    """
    if len(d_samples) == 0:
        return current_theta
    target = float(np.quantile(np.asarray(d_samples, dtype=np.float64), q))
    return ema * current_theta + (1.0 - ema) * target


#: 自标定 warmup 哨兵阈值(``theta_init<=0`` 即「无先验,首个 step 边界从
#: 真实分布分位落地」)。取有限大数而非 inf:JSON/wire 可安全传输,且
#: IntensityGate 下 w = H/θ ≈ 1e-9 < w_min ⇒ 全 bypass、零 rng draw、零干预
#: ——warmup 期 steered 臂行为与 bare 逐位一致(rng 对齐:bypass 位不 draw),
#: G-E 段级 bypass 还会跳过教师 verify(零额外开销)。剂量控制通用化
#: (2026-08-11 用户裁定,MENTOR「首步 γ_p=999」同款先例):绝对 nats 阈值
#: 是 benchmark 锚定的(1.28 占位值事故),分位自标定是无量纲的。
THETA_WARMUP = 1.0e9


class GateThresholdTracker:
    """worker 进程级的 θ_D 率控自适应追踪器(线程安全)。

    生命周期:agent-loop worker 进程单例。每条轨迹结束时 ``observe()`` 并入
    该轨迹的 D 直方图;检测到 global_step 前进时(``maybe_step_update``)
    用**累积直方图**做一次更新——同一 step 内所有轨迹用同一个 θ
    (步内冻结,因果性;滞后一步,分布稳定则无害)。

    两种更新模式(2026-08-11 剂量通用化):

    * ``direct``:θ ← 分位直标(quantile_{1−rate};MENTOR 每步 set 同款,
      无 EMA 时延——EMA β=0.1 曾让错误起点 24 步收敛不到位,门率全程
      3× 超标)。``ratchet=True`` 时 θ 不低于首次标定值:策略熵坍缩时
      w=H/θ 全体下降、教师自动淡出,而不是阈值跟着坍缩把门变宽——
      取代绝对值硬地板(floor),后者是 benchmark 锚定的。
    * ``ema``:历史行为(robust-EMA 慢更新),复现旧 run 用;要求显式
      theta_init(>0)与 floor。

    直方图分位数:线性桶内插值。累积观测不足 ``min_obs`` 时**保留累积、
    继续跨 step 攒**,攒够才更新并清空——verl 多 worker 拓扑下每个进程
    只见全批的 1/N(162 s1 实测 16 worker × ~160 obs/step),逐 step 清空
    会让阈值永久冻结在初值(2026-08-05 s1 事故);ema 模式跨 k 步合并更新
    时 β 按 ``1-(1-β)^k`` 复利折算,时间常数与逐步更新等价。

    率告警(``alarm_bounds``):每次更新时用本步直方图估计旧 θ 生效期的
    非零干预率 P(H ≥ w_min·θ),超出预注册区间即 logger.warning + 写入
    ``last_info``(spec [5%,25%] 防线此前只接 adaptive 硬门,熵门不在环上
    是 2026-08-11 事故的防线缺口)。只告警不降级——自动降级是影响效果的
    干预,须另行裁定。
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
        floor: float = 0.0,
        update_mode: str = "ema",
        ratchet: bool = False,
        alarm_bounds: tuple[float, float] | None = None,
        alarm_w_min: float = 1.0,
    ) -> None:
        import threading

        if not (0.0 < target_rate < 1.0):
            raise ValueError(f"target_rate must be in (0,1), got {target_rate}")
        if not (0.0 < ema_beta <= 1.0):
            raise ValueError(f"ema_beta must be in (0,1], got {ema_beta}")
        if update_mode not in ("direct", "ema"):
            raise ValueError(f"update_mode must be direct|ema, got {update_mode!r}")
        if update_mode == "ema" and float(theta_init) <= 0.0:
            raise ValueError(
                "ema update_mode requires an explicit calibrated theta_init "
                "(>0) — auto warmup (theta_init<=0) is a direct-mode feature"
            )
        # v3.3 M4:硬地板(ema 模式用;防阈值追着分布坍缩跑,MENTOR
        # max(q80, 1.5) 先例)。direct 模式由 ratchet 承担该语义(自标定,
        # 无绝对刻度),config 校验拒绝 direct+floor 组合。
        self.floor = float(floor)
        self._warmup = float(theta_init) <= 0.0
        self.theta = (
            THETA_WARMUP if self._warmup else max(float(theta_init), self.floor)
        )
        self.update_mode = update_mode
        self.ratchet = bool(ratchet)
        self._ratchet_floor: float | None = None
        self.alarm_bounds = (
            (float(alarm_bounds[0]), float(alarm_bounds[1]))
            if alarm_bounds is not None else None
        )
        self.alarm_w_min = float(alarm_w_min)
        #: 最近一次 step 更新的观测快照(theta/分位/率估计/告警),供
        #: 调用方拉曲线;更新前为空 dict。
        self.last_info: dict = {}
        self.target_rate = float(target_rate)
        self.ema_beta = float(ema_beta)
        self.d_hist_max = float(d_hist_max)
        self.bins = int(bins)
        self.min_obs = int(min_obs)
        self._acc = [0] * self.bins
        self._acc_count = 0
        self._last_step: int | None = None
        self._last_update_step: int | None = None
        self._lock = threading.Lock()

    def observe(self, d_hist: list[int], d_count: int) -> None:
        with self._lock:
            for i, v in enumerate(d_hist[: self.bins]):
                self._acc[i] += int(v)
            self._acc_count += int(d_count)

    def maybe_step_update(self, global_step: int | None) -> None:
        """step 边界检测 + 阈值更新(worker 内每个 rollout 入口调一次)。"""
        if global_step is None:
            return
        with self._lock:
            if self._last_step is None:
                self._last_step = int(global_step)
                self._last_update_step = int(global_step)
                return
            if int(global_step) == self._last_step:
                return
            self._last_step = int(global_step)
            # step 前进:累积够 min_obs 才更新并清空;不够则保留继续攒
            # (多 worker 分片下单 step 观测不足,逐 step 清空会永久冻结)。
            if self._acc_count < self.min_obs:
                return
            target = self._hist_quantile(1.0 - self.target_rate)
            # 率告警:旧 θ 生效期的非零干预率估计(本步分布上 P(H≥w_min·θ);
            # warmup 哨兵期恒 0,天然不触上界,下界告警同样抑制——零干预
            # 是 warmup 的设计语义,不是失控)。
            rate_at_old = self._tail_mass(self.alarm_w_min * self.theta)
            if self.update_mode == "direct" or self._warmup:
                # 直标(MENTOR 每步 set 同款):EMA 从错误起点(占位值/
                # warmup 哨兵)出发要几十步才收敛,直标一步落位。
                new_theta = target
            else:
                k = self._last_step - (self._last_update_step or self._last_step)
                beta_eff = 1.0 - (1.0 - self.ema_beta) ** max(k, 1)
                new_theta = (1.0 - beta_eff) * self.theta + beta_eff * target
            if self.ratchet:
                if self._ratchet_floor is None:
                    self._ratchet_floor = new_theta
                new_theta = max(new_theta, self._ratchet_floor)
            self.theta = max(new_theta, self.floor)
            alarm: str | None = None
            if self.alarm_bounds is not None and not self._warmup:
                lo, hi = self.alarm_bounds
                if rate_at_old > hi:
                    alarm = (
                        f"steering open-rate {rate_at_old:.3f} above the "
                        f"pre-registered ceiling {hi} (theta was lagging the "
                        "distribution) — dosage overshoot, inspect before "
                        "trusting this step's steered rows"
                    )
                elif rate_at_old < lo:
                    alarm = (
                        f"steering open-rate {rate_at_old:.3f} below the "
                        f"pre-registered floor {lo} — intervention starved"
                    )
            self._warmup = False
            self.last_info = {
                "step": self._last_step,
                "theta": self.theta,
                "quantile_target": target,
                "open_rate_at_old_theta": rate_at_old,
                "alarm": alarm,
            }
            if alarm:
                logger.warning("gate tracker step %d: %s", self._last_step, alarm)
            self._acc = [0] * self.bins
            self._acc_count = 0
            self._last_update_step = self._last_step

    def _tail_mass(self, threshold: float) -> float:
        """累积直方图上 P(X ≥ threshold) 的桶级估计(线性桶内插值,与
        ``_hist_quantile`` 同坐标系;threshold 超直方图上界 ⇒ 0)。"""
        if self._acc_count <= 0:
            return 0.0
        if threshold <= 0.0:
            return 1.0
        width = self.d_hist_max / self.bins
        if threshold >= self.d_hist_max:
            return 0.0
        idx = int(threshold / width)
        frac_below = (threshold - idx * width) / width
        tail = self._acc[idx] * (1.0 - frac_below) + sum(self._acc[idx + 1:])
        return tail / self._acc_count

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
