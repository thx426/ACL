"""PGS steering 包 —— 保尾核倾斜采样的机制核心（规范 docs/pgs-final-spec.md）。

模块分工：
  kernel   保尾核 b_S 的构造与查询（数学核心；命题 A/B 的对象）
  sps      speculative 接受-拒绝-残差重采（单点判定）
  gate     D 主门 + ε 探针（可测性红线在接口层锁死）
  budget   D_∞ 预算余额制 + R_max 行为态
  ledger   逐 token 记账 → extra_fields（训练端 W_beh 的数据源）

损失中无教师、无蒸馏项、无 critic —— 教师只经由采样端的保尾核出现。
"""

from .budget import TrajectoryBudget, TurnRejectCap
from .gate import GateDecision, IntensityGate, SteeringGate, robust_quantile_update
from .kernel import SteeringKernel, truncated_increment
from .ledger import SteeringLedger, TokenSteerRecord
from .sps import SpsOutcome, accept_or_resample

__all__ = [
    "GateDecision",
    "SpsOutcome",
    "IntensityGate",
    "SteeringGate",
    "SteeringKernel",
    "SteeringLedger",
    "TokenSteerRecord",
    "TrajectoryBudget",
    "TurnRejectCap",
    "accept_or_resample",
    "robust_quantile_update",
    "truncated_increment",
]
