"""逐 token steering 记账（规范 §4）→ rollout extra_fields。

记账自洽原则：log_b 必须是**实际行为分布**（保尾核）对最终执行 token 的
真值——训练端 W_beh 的无偏性只建立在这一条上。未 steer 位置 log_b == log_pi
（ratio 0），记账仍逐位置齐全（训练端不需要分支判断）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["TokenSteerRecord", "SteeringLedger"]


@dataclass(frozen=True)
class TokenSteerRecord:
    position: int          # response 内位置（与 turn_spans 同坐标）
    log_pi: float
    log_b: float
    eta: float
    delta: float           # 本位置预算扣费（=2ηc；η=0 ⇒ 0）
    d_value: float
    base_open: bool
    probe_flipped: bool
    propensity: float
    rejected: bool
    cap_active: bool


@dataclass
class SteeringLedger:
    """一条轨迹的记账。finalize() 输出进 extra_fields 的紧凑结构。"""

    records: list[TokenSteerRecord] = field(default_factory=list)
    budget_spent: float = 0.0
    turns_capped: int = 0

    def append(self, rec: TokenSteerRecord) -> None:
        if self.records and rec.position <= self.records[-1].position:
            raise ValueError("ledger positions must be strictly increasing")
        self.records.append(rec)

    def finalize(self) -> dict[str, Any]:
        """紧凑数组形式（extra_fields 可序列化；训练端按 position 对齐）。

        约定：只存 steer 相关位置的稀疏记录 + 全轨迹汇总；未记录位置训练端
        默认 log_b == log_pi_old（重算路径），W_beh = 1。
        """
        return {
            "steer_positions": [r.position for r in self.records],
            "steer_log_pi": [r.log_pi for r in self.records],
            "steer_log_b": [r.log_b for r in self.records],
            "steer_eta": [r.eta for r in self.records],
            "steer_delta": [r.delta for r in self.records],
            "steer_d": [r.d_value for r in self.records],
            "steer_base_open": [int(r.base_open) for r in self.records],
            "steer_probe": [int(r.probe_flipped) for r in self.records],
            "steer_propensity": [r.propensity for r in self.records],
            "steer_rejected": [int(r.rejected) for r in self.records],
            "steer_cap_active": [int(r.cap_active) for r in self.records],
            "steer_budget_spent": self.budget_spent,
            "steer_turns_capped": self.turns_capped,
            "steer_count": len(self.records),
        }
