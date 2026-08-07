"""分臂采样(arm-split rollout,机制 v3.1;用户裁定 2026-08-06)。

每任务的 n 个 GRPO rollout 按 steered_fraction 切成两臂:
  * steered 臂:SPS 保尾核倾斜采样(带 W_beh 记账);
  * bare 臂:纯 π 采样(空账 ⇒ W≡1),同组参与 GRPO advantage。

动机(2026-08-06 探讨定案):全臂干预会 (a) 污染训练侧能力读数
(score 测的是 E_b[R] 非 E_π[R]);(b) 抬高组内成功率使 GRPO 组内方差
坍缩、policy 自然失败样本稀疏化——负梯度从"高频低幅"劣化为"低频高幅";
(c) per-token W_beh 不校正前缀测度,状态分布偏差随 B_∞ 线性放大。
分臂恢复 bare 逐步读数与原生负梯度,混合组内 baseline 同时把
"教师能成/你失败"的差距转成更强的对比信号(轨迹级 50% 探针)。

分配是确定性的:session_id ∈ [0, rollout_n) 由 verl TQ 层逐副本传入,
session_id < steered_arm_count 的副本走 steered 臂。确定性分配保证
逐 dump 可审计,且各副本先验同分布(session 顺序与任务内容无关)。

steered_fraction=1.0 为历史行为(全臂干预),此时臂身份无意义,
session_id 允许缺失(兼容 V0 等不传 session_id 的调用路径)。
"""
from __future__ import annotations

__all__ = ["steered_arm_count", "is_steered_arm", "role_of", "steered_retired"]


def steered_retired(adv_mode: str, decay_steps: int, global_step) -> bool:
    """v3.3 M2:α 衰减到期后 steered 臂停产的判定(纯函数;rollout 端与
    advantage 端各按同一 cosine 公式独立求值,此处只需到期布尔)。
    global_step 未知(None/非整数)时保守不停。"""
    if adv_mode != "asym" or int(decay_steps) <= 0:
        return False
    try:
        gs = int(global_step)
    except (TypeError, ValueError):
        return False
    return gs >= int(decay_steps)

#: v4 行角色常量(docs/pgs-self-researcher-design.md §0)
ROLE_BARE = "bare"
ROLE_STEERED = "steered"
ROLE_RESEARCH = "research"


def role_of(
    session_id: int | None,
    rollout_n: int,
    steered_fraction: float,
    research_n: int = 0,
) -> str:
    """v4 三段行布局:[0, bare) bare | [bare, rollout_n) steered |
    [rollout_n, rollout_n+research_n) research。

    research_n=0 时退化为 v3.1 两段(is_steered_arm 语义);research_n>0
    时缺 session_id / 越界一律 fail loud(行角色错配 = 训练数据静默错账)。
    """
    if research_n < 0:
        raise ValueError(f"research_n must be >= 0, got {research_n}")
    total = rollout_n + research_n
    if research_n == 0 and steered_fraction >= 1.0:
        return ROLE_STEERED if is_steered_arm(session_id, rollout_n, steered_fraction) else ROLE_BARE
    if session_id is None:
        raise RuntimeError(
            "multi-role rollout requires a session_id per replica — the caller "
            "did not forward it"
        )
    sid = int(session_id)
    if not (0 <= sid < total):
        raise RuntimeError(
            f"session_id {session_id} out of range [0, {total}) — role assignment undefined"
        )
    if sid >= rollout_n:
        return ROLE_RESEARCH
    return (
        ROLE_STEERED
        if is_steered_arm(sid, rollout_n, steered_fraction)
        else ROLE_BARE
    )


def steered_arm_count(rollout_n: int, steered_fraction: float) -> int:
    """steered 臂副本数 = round(n × fraction),并夹到 [0, n]。

    合法运行区间由 config 校验保证(count ≥ 1);本函数只做算术,
    好让校验层能引用同一个取整口径报错。
    """
    if rollout_n < 1:
        raise ValueError(f"rollout_n must be >= 1, got {rollout_n}")
    if not (0.0 < steered_fraction <= 1.0):
        raise ValueError(
            f"steered_fraction must be in (0, 1], got {steered_fraction}"
        )
    return max(0, min(rollout_n, round(rollout_n * steered_fraction)))


def is_steered_arm(
    session_id: int | None, rollout_n: int, steered_fraction: float
) -> bool:
    """本副本是否属于 steered 臂。

    fraction=1.0 恒 True(全臂,session_id 可为 None);fraction<1 时
    session_id 必须在 [0, rollout_n) —— 缺失/越界说明调用路径没有把
    副本身份传进来,静默回退会让整组悄悄变成全 steered(正是分臂要修
    的观测污染),必须 fail loud。
    """
    if steered_fraction >= 1.0:
        return True
    if session_id is None:
        raise RuntimeError(
            "arm-split steering (steered_fraction < 1) requires a session_id "
            "per rollout replica — the caller did not forward it"
        )
    if not (0 <= int(session_id) < rollout_n):
        raise RuntimeError(
            f"session_id {session_id} out of range [0, {rollout_n}) — "
            "arm assignment would be undefined"
        )
    return int(session_id) < steered_arm_count(rollout_n, steered_fraction)
