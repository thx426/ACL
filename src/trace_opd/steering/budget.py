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
    """一条轨迹的 D_∞ 预算（在线余额制：先算后花，耗尽即纯 π）。"""

    def __init__(self, b_max: float, c: float) -> None:
        if b_max < 0 or c <= 0:
            raise ValueError(f"invalid budget b_max={b_max}, c={c}")
        self.b_max = float(b_max)
        self.c = float(c)
        self.spent = 0.0

    @property
    def remaining(self) -> float:
        return max(0.0, self.b_max - self.spent)

    def admissible_eta(self, eta_proposed: float) -> float:
        """余额允许的 η：δ(η)=2ηc ≤ remaining，不足则缩到余额、耗尽归零。"""
        if eta_proposed <= 0.0:
            return 0.0
        cap = self.remaining / (2.0 * self.c)
        return float(min(eta_proposed, cap))

    def charge(self, eta: float) -> float:
        """按最终执行的 η 扣费；返回本次 δ。η=0 零费。"""
        delta = 2.0 * eta * self.c
        if delta > self.remaining + 1e-9:
            raise RuntimeError(
                f"budget overdraft: delta={delta:.4f} > remaining={self.remaining:.4f} "
                "— call admissible_eta before charging"
            )
        self.spent += delta
        return delta


class TurnRejectCap:
    """每轮拒绝计数与 cap-active 行为态（进记账；规范 §2.5）。"""

    def __init__(self, r_max: int = 4) -> None:
        if r_max < 1:
            raise ValueError(f"r_max must be >= 1, got {r_max}")
        self.r_max = int(r_max)
        self.count = 0

    @property
    def cap_active(self) -> bool:
        """达到上限 ⇒ 该轮余下位置强制 η=0（无歧义：恰 r_max 次后关闭）。"""
        return self.count >= self.r_max

    def record_reject(self) -> None:
        if self.cap_active:
            raise RuntimeError("reject recorded while cap already active — hook must force eta=0")
        self.count += 1

    def new_turn(self) -> None:
        self.count = 0
