"""D_∞ 预算余额制 + 每轮拒绝上限状态机（规范 §2.4 / §2.5）。

计费口径：δ = 2ηc —— 全词表密度比上界的保守扣费（复审自查点 C 修法：
直接扣上界，封掉 V′-max 与全词表 max 之间的缺口）。由此逐轨迹
|Σ log(π/b)| ≤ B_∞ 严格成立（命题 B 推论）。

R_max（每轮拒绝上限）是**行为状态**（NF04）：达到上限后该轮余下位置强制
η=0、b≡π 且如实记账——行为仍是良定义的自适应自回归核。
"""

from __future__ import annotations

__all__ = ["TrajectoryBudget", "TurnRejectCap"]


class TrajectoryBudget:
    """一条轨迹的 D_∞ 预算（在线余额制：先算后花，耗尽即纯 π）。

    pacing=True 时按解锁曲线族逐 turn 拨付额度(2026-08-05 用户裁定,
    docs/pgs-budget-pacing-design.md;修 s1 实测的头部坍缩——98.7% 干预
    堆在 turn 0 前 0.4%)：

        unlocked(t) = U0 + (B−U0) · φ(t/T_max),  φ(x) = 1−(1−x)^p

    p=1 匀速、p=2 解锁速率随轮次线性衰减(默认,预算重心前移但全程非零);
    花费只能动已解锁部分(token-bucket:没花的额度自动流向后段)。t 只依赖
    已实现的 turn 进度与环境硬配置 T_max,零任务先验,因果性完好;
    Σδ ≤ B_∞ 总约束不变(W_beh IS 界不受影响)。pacing=False 完全回退旧
    行为(t=0 全解锁)。
    """

    #: 跨 turn 可变状态注册表(单一真源)——Engine-Resident SPS 的 wire
    #: 往返按它驱动(steering/wire.py);其余属性是构造参数,两端各自从
    #: run 配置重建。新增可变状态字段必须同步登记,否则 engine 档静默丢态。
    _WIRE_MUTABLE = ("spent", "turn")

    def __init__(
        self,
        b_max: float,
        c: float,
        *,
        pacing: bool = False,
        u0: float = 0.0,
        profile_power: float = 2.0,
        t_max: int = 1,
    ) -> None:
        if b_max < 0 or c <= 0:
            raise ValueError(f"invalid budget b_max={b_max}, c={c}")
        if profile_power <= 0 or u0 < 0:
            raise ValueError(f"invalid pacing u0={u0}, profile_power={profile_power}")
        # b_max=inf(maxopen 档,机制 v3.2):无限预算是一等公民——available
        # 恒 inf、admissible_eta 不缩、charge 永不透支、spent 照实累计(观察量)。
        # 但解锁曲线在 B=inf 下无定义((B−U0)·φ 在 t=0 出 inf·0=NaN),
        # 组合必须在此 fail loud(config 校验之外的第二道防线)。
        import math

        if pacing and math.isinf(b_max):
            raise ValueError("budget_pacing is undefined with b_max=inf — disable pacing")
        self.b_max = float(b_max)
        self.c = float(c)
        self.spent = 0.0
        self.pacing = bool(pacing)
        self.u0 = min(float(u0), self.b_max)
        self.profile_power = float(profile_power)
        self.t_max = max(int(t_max), 1)
        self.turn = 0

    def advance_turn(self) -> None:
        """每个 assistant turn 结束时推进(hook 的 generate 调用边界)。"""
        self.turn += 1

    @property
    def unlocked(self) -> float:
        """当前 turn 的累计解锁额度(pacing 关闭时恒 = b_max)。"""
        if not self.pacing:
            return self.b_max
        x = min(self.turn / self.t_max, 1.0)
        phi = 1.0 - (1.0 - x) ** self.profile_power
        return min(self.b_max, self.u0 + (self.b_max - self.u0) * phi)

    @property
    def available(self) -> float:
        """本 turn 可花额度 = 已解锁 − 已花(花费判据一律用它)。"""
        return max(0.0, self.unlocked - self.spent)

    @property
    def remaining(self) -> float:
        """轨迹总余额(b_max − spent;区分"暂未解锁"与"总额耗尽"用)。"""
        return max(0.0, self.b_max - self.spent)

    def admissible_eta(self, eta_proposed: float) -> float:
        """可花额度允许的 η：δ(η)=2ηc ≤ available，不足则缩、为零归零。"""
        if eta_proposed <= 0.0:
            return 0.0
        cap = self.available / (2.0 * self.c)
        return float(min(eta_proposed, cap))

    def charge(self, eta: float) -> float:
        """按最终执行的 η 扣费；返回本次 δ。η=0 零费。"""
        delta = 2.0 * eta * self.c
        if delta > self.available + 1e-9:
            raise RuntimeError(
                f"budget overdraft: delta={delta:.4f} > available={self.available:.4f} "
                "— call admissible_eta before charging"
            )
        self.spent += delta
        return delta


class TurnRejectCap:
    """每轮拒绝计数与 cap-active 行为态（进记账；规范 §2.5）。

    r_max=0(机制 v3.2,用户裁定 2026-08-06)= **无上限**:cap 永不激活,
    干预完全按两分布差异进行,不受轮内拒绝数限制——分臂 bare 锚 + outcome
    reward 语境下最后一个行为限制装置退役。终止性不依赖 cap:SPS 拒绝后
    重采并**前进位置**,每位置至多判定一次,循环数自然 ≤ 轨迹长度;计数
    照记(观察量),尾部墙钟由监控而非机制截断兜底。r_max≥1 保留历史行为。
    """

    #: 跨 turn 可变状态注册表(wire 往返;r_max 是构造参数)
    _WIRE_MUTABLE = ("count",)

    def __init__(self, r_max: int = 4) -> None:
        if r_max < 0:
            raise ValueError(f"r_max must be >= 0 (0 = uncapped), got {r_max}")
        self.r_max = int(r_max)
        self.count = 0

    @property
    def cap_active(self) -> bool:
        """达到上限 ⇒ 该轮余下位置强制 η=0(r_max=0 时恒 False,无上限)。"""
        return self.r_max > 0 and self.count >= self.r_max

    def record_reject(self) -> None:
        if self.cap_active:
            raise RuntimeError("reject recorded while cap already active — hook must force eta=0")
        self.count += 1

    def new_turn(self) -> None:
        self.count = 0
