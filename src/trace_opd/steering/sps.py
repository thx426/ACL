"""SPS 单点执行：speculative 接受-拒绝-残差重采（规范 §2.5，命题 A 相对 b_S）。

恒等（对固定核）::

    P(输出=x) = π(x)·min(1, b/π) + P(拒绝)·residual(x) = b(x)

调用方（episode hook）逐位置从左到右应用；拒绝 ⇒ 丢弃 draft 后缀、以修正
前缀续 draft。本模块只管单点判定与记账值。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kernel import SteeringKernel

__all__ = ["SpsOutcome", "accept_or_resample"]


@dataclass(frozen=True)
class SpsOutcome:
    token_id: int          # 最终 token（= draft token 或残差重采结果）
    rejected: bool         # 该位置是否发生了拒绝-重采
    log_b: float           # log b(最终 token) —— 训练端 W_beh 的分母记账
    log_pi: float          # log π(最终 token) —— 同一 verify 路径的值
    accept_prob: float     # 本位置的接受概率（诊断）


def accept_or_resample(
    kernel: SteeringKernel,
    draft_token: int,
    draft_log_pi: float,
    rng: np.random.Generator,
    *,
    log_pi_lookup: dict[int, float] | None = None,
) -> SpsOutcome:
    """对一个 draft token 执行 speculative 判定。

    Args:
        kernel: 该位置的保尾核（在看到 draft token 前构造——可测性红线由
            episode hook 层保证，本函数不接触任何后缀信息）。
        draft_token: 学生 draft 采出的 token。
        draft_log_pi: verify 前向给出的 log π(draft_token)（数值统一路径）。
        rng: 独立随机数发生器。
        log_pi_lookup: S 内 token 的 log π 查表（残差重采出的 token 需要其
            log π 才能记账 log_b；缺省时用 kernel 自带的 S 上 log_pi）。

    Returns:
        SpsOutcome。行为分布严格为 b_S（kernel 固定 + 本判定 ⇒ 命题 A）。
    """
    p_acc = kernel.accept_prob(draft_token)
    if rng.random() < p_acc:
        return SpsOutcome(
            token_id=int(draft_token),
            rejected=False,
            log_b=kernel.log_b(draft_token, draft_log_pi),
            log_pi=float(draft_log_pi),
            accept_prob=p_acc,
        )

    # 拒绝：从残差 norm(max(0, b−π)) 重采（支撑 ⊆ S）
    ids, probs = kernel.residual_distribution()
    pick = int(ids[rng.choice(len(ids), p=probs)])
    if log_pi_lookup is not None and pick in log_pi_lookup:
        log_pi_pick = float(log_pi_lookup[pick])
    else:
        # S 内 token 的 log π 在核里就有
        idx = {int(t): i for i, t in enumerate(kernel.token_ids)}[pick]
        log_pi_pick = float(kernel.log_pi[idx])
    return SpsOutcome(
        token_id=pick,
        rejected=True,
        log_b=kernel.log_b(pick, log_pi_pick),
        log_pi=log_pi_pick,
        accept_prob=p_acc,
    )
