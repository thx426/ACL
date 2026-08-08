"""v4 自体研究员编排(docs/pgs-self-researcher-design.md §0/§3)。

per-(step, task) 协调器 + research 行的完整执行(会话 → doc → 等前缀池 →
续写评估 → 登记 → 训练行数据)。并发红线(用户裁定):依赖用 shared
future 同步等待,其余全并发,机制侧零人为串行;**每行必须登记**(r 或
降级),否则 argmax future 悬挂。

agent_loop 只做薄封装(role 分流、AgentLoopOutput 组装、dump);本模块
不 import agent_loop(依赖注入,可用 fakes 全链单测)。
"""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..research.continuation import (
    ContinuationOutcome,
    aggregate_reward,
    build_prefix_set,
    continue_from_prefix,
)

logger = logging.getLogger(__file__)

__all__ = ["TaskCoord", "CoordRegistry", "ResearchRowResult", "run_research_row"]


class TaskCoord:
    """一个 (step, task) 的行间协调:bare 前缀池 future + research 登记 +
    argmax future。同任务全部行落同一 worker 同一 event loop(dedupe
    invariant),futures 无跨进程语义。"""

    def __init__(self, *, bare_expected: int, research_expected: int,
                 quantiles: list[float], tokenizer) -> None:
        self._bare_expected = int(bare_expected)
        self._research_expected = int(research_expected)
        self._quantiles = list(quantiles)
        self._tokenizer = tokenizer
        self._bare_rows: list[dict[str, Any]] = []
        self._registry: dict[int, tuple[str, float, bool]] = {}
        loop = asyncio.get_running_loop()   # 构造必在 worker event loop 内
        self.prefix_future: asyncio.Future = loop.create_future()
        self.argmax_future: asyncio.Future = loop.create_future()

    # ---- bare 侧 ----
    def register_bare(self, row: dict[str, Any]) -> None:
        """bare 行终态登记(won/response_ids/response_mask/turn_spans/steps)。
        第 bare_expected 条到齐时构造前缀集并 resolve。"""
        self._bare_rows.append(dict(row))
        if len(self._bare_rows) == self._bare_expected and not self.prefix_future.done():
            try:
                specs = build_prefix_set(self._bare_rows, self._quantiles, self._tokenizer)
                self.prefix_future.set_result(specs)
            except Exception as err:  # noqa: BLE001 — 前缀构造失败必须传播给等待者
                self.prefix_future.set_exception(err)

    # ---- research 侧 ----
    def register_research(self, session_id: int, doc: str, r: float,
                          degraded: bool) -> None:
        """每 research 行必须调用恰一次(含降级/空 doc);第 research_expected
        个登记触发 argmax(排除降级行;r*=0 准入门 ⇒ 无教师)。"""
        if session_id in self._registry:
            raise RuntimeError(f"research row {session_id} registered twice")
        self._registry[session_id] = (doc, float(r), bool(degraded))
        if len(self._registry) == self._research_expected and not self.argmax_future.done():
            eligible = [
                (doc_i, r_i) for doc_i, r_i, deg in self._registry.values()
                if not deg and doc_i
            ]
            if not eligible:
                self.argmax_future.set_result(("", 0.0))
                return
            best_r = max(r for _, r in eligible)
            if best_r <= 0.0:
                self.argmax_future.set_result(("", 0.0))     # 准入门:无教师
                return
            best = min((d for d, r in eligible if r == best_r), key=len)  # tie→短 doc
            self.argmax_future.set_result((best, best_r))


class CoordRegistry:
    """worker 级 (step, task) → TaskCoord;新 step 出现时丢弃旧 step 桶
    (doc 文本量大,长 run 必须回收;worker 内 step 单调)。"""

    def __init__(self) -> None:
        self._step: int | None = None
        self._coords: dict[str, TaskCoord] = {}
        self._lock = asyncio.Lock()

    async def get(self, step: int, task_id: str, factory: Callable[[], TaskCoord]) -> TaskCoord:
        async with self._lock:
            if self._step != step:
                self._step = step
                self._coords = {}
            if task_id not in self._coords:
                self._coords[task_id] = factory()
            return self._coords[task_id]


# ------------------------------------------------------------- research 行

@dataclass
class ResearchRowResult:
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    reward_score: float               # 已含 λ_r 与 warmup 缩放(预乘等价)
    fields: dict[str, Any] = field(default_factory=dict)   # research_* 记账
    session_result: Any = None        # SessionResult(归档用)
    outcomes: list[ContinuationOutcome] = field(default_factory=list)


async def run_research_row(
    *,
    session_id: int,
    global_step: int,
    coord: TaskCoord,
    run_session: Callable[[], Awaitable[Any]],     # → (SessionResult, chat)
    teacher_ids_for_doc: Callable[[str], list[int] | None],  # doc→教师前缀|None(overflow)
    make_generate: Callable[[str], Any],           # tag → GenerateFn(纯 p_T 续写通道)
    adapter,
    tokenizer,
    delta,
    response_budget: int,
    continuation_samples: int,
    continuation_temperature: float,
    research_rl_weight: float,
    research_warmup_steps: int,
    research_max_sample_tokens: int,
) -> ResearchRowResult:
    """research 行全流程(§0 尾段)。每条路径都保证 coord.register_research
    恰被调用一次(argmax future 依赖)。abort 类基建故障向上 raise(verl
    prompt failure,与 executor 行同语义)——raise 前不登记:同 prompt 的
    全部行会被一并取消,future 悬挂无消费者。"""
    session_result, chat = await run_session()
    if getattr(chat, "aborted", False):
        raise RuntimeError("researcher session aborted by the engine (weight sync?)")

    doc = (session_result.doc or "") if session_result is not None else ""
    fields: dict[str, Any] = {
        "research_no_finalize": int(not doc),
        "research_doc_words": len(doc.split()) if doc else 0,
        "research_turns": getattr(session_result, "turns_used", 0),
        "research_experiments": getattr(session_result, "experiments_used", 0),
        "research_episodes": getattr(session_result, "episodes_consumed", 0),
        "research_error": int(bool(getattr(session_result, "error", None))),
        "research_doc_overflow": 0,
        "research_valid_continuations": 0,
        "research_degraded": 0,
        "research_replay_drift": 0,
    }

    outcomes: list[ContinuationOutcome] = []
    degraded = False
    r = 0.0
    if doc:
        teacher_ids = teacher_ids_for_doc(doc)
        if teacher_ids is None:
            fields["research_doc_overflow"] = 1        # r=0:结构性长度压力
        else:
            specs = await coord.prefix_future
            nominal = len(specs) * continuation_samples

            async def _one(spec, k: int) -> ContinuationOutcome:
                gen = make_generate(f"cont:{session_id}:{spec.quantile}:{k}")
                out = await continue_from_prefix(
                    adapter=adapter, teacher_prefix_ids=teacher_ids, spec=spec,
                    generate=gen, tokenizer=tokenizer, delta=delta,
                    response_budget=response_budget,
                    sampling_params={"temperature": continuation_temperature},
                )
                if out.error is not None:              # ENV_ERROR 类重试一次
                    out = await continue_from_prefix(
                        adapter=adapter, teacher_prefix_ids=teacher_ids, spec=spec,
                        generate=gen, tokenizer=tokenizer, delta=delta,
                        response_budget=response_budget,
                        sampling_params={"temperature": continuation_temperature},
                    )
                return out

            outcomes = list(await asyncio.gather(*[
                _one(spec, k)
                for spec in specs for k in range(continuation_samples)
            ]))
            r, valid, degraded = aggregate_reward(outcomes, nominal)
            fields["research_valid_continuations"] = valid
            fields["research_degraded"] = int(degraded)
            fields["research_replay_drift"] = sum(
                1 for o in outcomes if o.error and "replay_drift" in o.error
            )
    coord.register_research(session_id, doc if not degraded else "", r, degraded)

    prompt_ids, stream, mask = chat.training_row()
    # 流口径(不含 prompt,与 session_token_budget/§7.4 联动阈同口径;
    # §7.5 口径统一——旧式含 prompt 会让同数值阈实际紧 ~1.5k)。
    overlong = len(stream) > research_max_sample_tokens
    if overlong:
        mask = [0] * len(mask)                         # 零梯度;r 留组内 baseline
    fields["research_overlong"] = int(overlong)
    fields["research_r_raw"] = r
    # policy 版本证据:research 行的训练 token 与 executor 行一样出自 rollout
    # 引擎,trainer 的 _compute_metrics 对**每一行**都 np.array(..., dtype=int),
    # 缺则取到 None 直接 TypeError 崩整步(2026-08-07 162 v4 冒烟实测)。
    version_fields = chat.policy_version_fields() if hasattr(chat, "policy_version_fields") else {}
    if not version_fields:
        raise RuntimeError(
            f"research row session={session_id} has no rollout policy version evidence "
            "(engine did not report min/max_global_steps)"
        )
    fields.update(version_fields)
    # λ_r 预乘 + warmup 置零:reward 的线性变换与 advantage 缩放等价
    # (组内去均值是线性算子),vendored 侧因此零改动(§1.6)。
    warmup = global_step is not None and global_step < research_warmup_steps
    scaled = 0.0 if warmup else research_rl_weight * r
    fields["research_warmup_active"] = int(bool(warmup))
    if not math.isfinite(scaled):
        raise RuntimeError(f"non-finite research reward {scaled!r}")
    return ResearchRowResult(
        prompt_ids=list(prompt_ids),
        response_ids=list(stream),
        response_mask=list(mask),
        reward_score=scaled,
        fields=fields,
        session_result=session_result,
        outcomes=outcomes,
    )
