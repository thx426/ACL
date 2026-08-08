"""v4 前缀续写评估(docs/pgs-self-researcher-design.md §1.2-§1.4)。

reward 的测量装置:把「教师 = policy + 特权文档」放到学生 bare 轨迹的
前缀状态上续写至终局,r(doc) = 有效续写的成功比例——即教师价值函数在
学生状态分布上的蒙特卡洛均值(DAgger 式评估分布)。

三个机制承诺(§2 正确性论证的实现锚点):
  * 前缀按 **policy 生成 token(response_mask=1)的累计比例**切分,不对齐
    turn 边界——半轮承接是特性(教师接住写到一半的思路);
  * 续写为**纯 p_T 采样**(无 SPS/无倾斜/无 W 记账),与 PGS 倾斜方向源
    同一条件分布;续写轨迹只作 reward 测量,绝不进训练;
  * **预算继承**:教师只拿轨迹剩余的环境步/token 预算,与学生同规。

环境重放的正确性挂在 TaskAdapter 确定性契约上,并以逐步 observation
校验兜底(不一致 = 漂移,该次续写作废)。
"""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..core import protocol
from ..core.episode import GenResult
from ..core.template_delta import TemplateDelta

logger = logging.getLogger(__file__)

__all__ = [
    "PrefixSpec",
    "ContinuationOutcome",
    "build_prefix_set",
    "continue_from_prefix",
    "aggregate_reward",
]


# ------------------------------------------------------------- 前缀集(纯函数)

@dataclass(frozen=True)
class PrefixSpec:
    """一个续写起点的完整描述(从 bare 行的 token 账与轨迹结构导出)。"""

    traj_index: int                 # 前缀来源的 bare 行下标(记账/观察)
    quantile: float
    prefix_ids: tuple               # response 流前缀(token id,含结构 token)
    replay_actions: tuple           # 已完成轮中 acted=True 的 normalized action
    replay_observations: tuple      # 与 replay_actions 对齐的期望 observation
    env_steps_used: int             # = len(replay_actions)
    turns_used: int                 # 前缀覆盖的完整 assistant 轮数
    half_turn_text: str             # 半轮已生成文本(空串 = 完整轮边界起写)


def _cut_stream_pos(response_mask: list[int], quantile: float) -> int | None:
    """第 ceil(q·M) 个 mask=1 token 的流位置(M = policy token 总数)。"""
    total = sum(1 for m in response_mask if m)
    if total == 0:
        return None
    target = max(1, math.ceil(quantile * total))
    seen = 0
    for pos, m in enumerate(response_mask):
        if m:
            seen += 1
            if seen == target:
                return pos
    return None  # 数值上不可达(target<=total)


def _spec_from_row(
    row: dict[str, Any], traj_index: int, quantile: float, tokenizer
) -> PrefixSpec | None:
    """单条 bare 行 × 单分位 → PrefixSpec。

    轮定位:切点落在轮 j 的生成段 [s_j, e_j) 内。
      * pos < e_j−1:半轮——前缀止于切点(含),轮 j 的 action 未产生,
        重放轮 1..j−1;半轮文本 = 轮 j 段 [s_j, pos] 的解码。
      * pos == e_j−1(轮末)且存在下一轮:前缀延伸到下一轮生成段起点
        (含 glue+obs 结构 token),轮 j 计入重放,从完整轮边界起写。
      * pos == e_j−1 且轮 j 是最后一轮:整条轨迹已被覆盖,续写无意义——
        回退一个 policy token 强制半轮;单 token 轮无处可退 ⇒ 放弃该分位。
    """
    ids: list[int] = list(row["response_ids"])
    mask: list[int] = list(row["response_mask"])
    spans: list[list[int]] = [list(x) for x in row["turn_spans"]]
    steps: list[dict[str, Any]] = list(row["steps"])
    if not ids or not spans or len(spans) != len(steps):
        return None
    pos = _cut_stream_pos(mask, quantile)
    if pos is None:
        return None
    turn_j = next(
        (j for j, (s, e) in enumerate(spans) if s <= pos < e), None
    )
    if turn_j is None:
        return None  # 切点必在某生成段内(mask=1 只存在于 spans)——防御
    s_j, e_j = spans[turn_j]
    if pos == e_j - 1:
        if turn_j + 1 < len(spans):
            prefix_end = spans[turn_j + 1][0]     # 含轮 j 后的结构 token
            completed = turn_j + 1
            half_text = ""
        elif pos > s_j:
            pos -= 1                               # 最后一轮轮末:强制半轮
            completed = turn_j
            prefix_end = pos + 1
            half_text = tokenizer.decode(ids[s_j: pos + 1])
        else:
            return None                            # 单 token 尾轮,无处可退
    else:
        completed = turn_j
        prefix_end = pos + 1
        half_text = tokenizer.decode(ids[s_j: pos + 1])
    replay_actions: list[str] = []
    replay_obs: list[str] = []
    for st in steps[:completed]:
        if st.get("acted"):
            replay_actions.append(str(st["action_norm"]))
            replay_obs.append(str(st["observation"]))
    return PrefixSpec(
        traj_index=traj_index,
        quantile=quantile,
        prefix_ids=tuple(int(t) for t in ids[:prefix_end]),
        replay_actions=tuple(replay_actions),
        replay_observations=tuple(replay_obs),
        env_steps_used=len(replay_actions),
        turns_used=completed,
        half_turn_text=half_text,
    )


def build_prefix_set(
    bare_rows: list[dict[str, Any]], quantiles: list[float], tokenizer
) -> list[PrefixSpec]:
    """每任务一份前缀集(组内全部 doc 共享;§1.2)。

    bare_rows 行结构:{won, response_ids, response_mask, turn_spans,
    steps:[{action_norm, observation, acted}]}。
    选择规则(用户裁定):**失败优先、跨轨迹分散**——轨迹按失败在前排序
    (组内原序稳定),第 i 个分位取第 (i mod 可用数) 条轨迹;单条轨迹上
    某分位构造失败时按同序回退到下一条轨迹。
    """
    order = sorted(range(len(bare_rows)), key=lambda i: bool(bare_rows[i]["won"]))
    usable = [i for i in order if bare_rows[i]["response_ids"]]
    specs: list[PrefixSpec] = []
    for qi, q in enumerate(quantiles):
        spec = None
        for attempt in range(len(usable)):
            ti = usable[(qi + attempt) % len(usable)] if usable else None
            if ti is None:
                break
            spec = _spec_from_row(bare_rows[ti], ti, q, tokenizer)
            if spec is not None:
                break
        if spec is not None:
            specs.append(spec)
    return specs


# --------------------------------------------------- 重放 + 半轮续写(编排)

@dataclass
class ContinuationOutcome:
    won: bool = False
    score: float = 0.0              # 终态评估保真(won = score>=1.0 派生)
    error: str | None = None        # 非 None = 作废(重放漂移/基建故障)
    env_steps: int = 0              # 续写期新增环境步
    gen_tokens: int = 0
    truncated: bool = False


async def continue_from_prefix(
    *,
    adapter,
    teacher_prefix_ids: list[int],
    spec: PrefixSpec,
    generate: Callable[[list[int], dict[str, Any]], Awaitable[GenResult]],
    tokenizer,
    delta: TemplateDelta,
    response_budget: int,
    sampling_params: dict[str, Any],
) -> ContinuationOutcome:
    """单次续写:开新 episode → 重放已完成 action(逐步校验 obs)→ 从
    token 断点续写(教师承接半轮)→ 标准交互循环至终局。

    预算继承(§1.3):env 步 = MAX_ENV_STEPS − env_steps_used;token =
    response_budget − len(prefix_ids);assistant 轮 = MAX_ASSISTANT_TURNS −
    turns_used。纯 p_T 采样——调用方传入的 sampling_params 不含任何
    steering 面(本函数也绝不构造 SPS)。
    """
    out = ContinuationOutcome()
    try:
        env = await asyncio.to_thread(adapter.open_episode)
    except Exception as err:  # noqa: BLE001 — 基建故障,作废由调用方重试
        out.error = f"open_episode: {type(err).__name__}: {err}"
        return out
    def _env_obs(text: str) -> str:
        # 录制侧 observation 是回注进流的形态:FC 协议下带 tool_response
        # 包裹、多调用时还带 "(note: …)" 附加;重放侧 env 返回裸观察。
        # 环境确定性校验只比环境部分,两侧归一化(text 协议下 no-op)
        # ——不归一化则 FC 下首步必假 drift、续写 reward 全零
        # (2026-08-08 审阅 P0)。
        t = protocol.strip_tool_response(text)
        cut = t.find("\n(note: ")
        return (t[:cut] if cut >= 0 else t).strip()

    try:
        live0 = env.initial_obs
        expected0 = adapter.initial_obs
        if expected0 and live0.strip() != expected0.strip():
            out.error = "replay_drift: initial_obs mismatch"
            return out
        # ---- 环境重放(确定性契约 + 逐步校验兜底) ----
        for act, expected in zip(spec.replay_actions, spec.replay_observations):
            obs, ended = await asyncio.to_thread(adapter.replay_action, env, act)
            if ended:
                out.error = "replay_drift: episode ended during replay"
                return out
            if _env_obs(obs) != _env_obs(expected):
                out.error = "replay_drift: observation mismatch"
                return out
        # ---- 续写循环 ----
        stream: list[int] = list(spec.prefix_ids)
        half_text = spec.half_turn_text
        env_steps = 0
        turns = spec.turns_used
        consecutive_bad = 0
        done = False
        won = False
        while (
            env_steps + spec.env_steps_used < protocol.MAX_ENV_STEPS
            and turns < protocol.MAX_ASSISTANT_TURNS
        ):
            budget = response_budget - len(stream) - 64
            if budget < 32:
                out.truncated = True
                break
            gen = await generate(
                list(teacher_prefix_ids) + stream,
                {**sampling_params,
                 "max_tokens": min(protocol.GEN_MAX_TOKENS_PER_TURN, budget)},
            )
            if gen.extra.get("stop_reason") == "aborted":
                out.error = "generation aborted by the engine"
                return out
            gen_ids = list(gen.token_ids)
            if not gen_ids:
                out.truncated = True
                break
            turns += 1
            stream += gen_ids
            out.gen_tokens += len(gen_ids)
            # 半轮承接:动作解析作用于「半轮前缀文本 + 续写文本」拼接;
            # 交互协议整体归 adapter.react(与 episode 同一抽象)。
            full_text = half_text + gen.text
            half_text = ""
            outcome = await asyncio.to_thread(adapter.react, env, full_text)
            if not outcome.acted:
                consecutive_bad += 1
                if consecutive_bad >= protocol.MAX_CONSECUTIVE_FORMAT_FAILURES:
                    out.truncated = True
                    break
                obs_text = outcome.feedback
            else:
                consecutive_bad = 0
                if outcome.consumed_env_step:
                    env_steps += 1
                obs_text = protocol.truncate_obs(outcome.feedback)
                done = outcome.done
                if done:
                    out.score = outcome.score
                    won = outcome.score >= 1.0
            if done or env_steps + spec.env_steps_used >= protocol.MAX_ENV_STEPS:
                break
            structure = delta.after_generation(gen_ids) + tokenizer.encode(
                obs_text, add_special_tokens=False
            ) + delta.post_obs
            if len(stream) + len(structure) + 96 > response_budget:
                out.truncated = True
                break
            stream += structure
        out.won = bool(won)
        out.env_steps = env_steps
        return out
    except Exception as err:  # noqa: BLE001 — 引擎/传输故障,作废
        out.error = f"{type(err).__name__}: {err}"
        return out
    finally:
        try:
            await asyncio.shield(asyncio.to_thread(env.close))
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass


# ------------------------------------------------------------- reward 聚合

def aggregate_reward(
    outcomes: list[ContinuationOutcome], nominal_total: int
) -> tuple[float, int, bool]:
    """r = wins / 有效次数;有效 < 名义半数 ⇒ degraded(行降级:不进组、
    不参 argmax,但必须登记——argmax future 依赖每行登记)。§1.4。"""
    valid = [o for o in outcomes if o.error is None]
    wins = sum(1 for o in valid if o.won)
    degraded = len(valid) < math.ceil(nominal_total / 2)
    r = (wins / len(valid)) if valid else 0.0
    return r, len(valid), degraded
