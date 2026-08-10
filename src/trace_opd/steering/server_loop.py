"""Engine-Resident SPS 的 server 侧 turn 入口(P1,设计文档 §3.2)。

`run_steered_turn` 是 rollout server actor 上 `steered_generate` 的全部
业务逻辑:从线协议 payload 重建轨迹状态 → 用**进程内** draft/verify 原语
驱动同一个 `SpsGenerate` 循环 → 打包(tokens + 状态回写 + 本 turn 账本
增量)。刻意做成纯函数式(原语注入、不 import 任何引擎),因此:

  * 单测可用 fake 原语走完整客户端-服务器链,与单进程 SpsGenerate 对拍
    bit-equal(tests/steering/test_engine_executor.py);
  * server actor 侧只剩一层薄接线(async_sglang_server.steered_generate):
    把自己的 `generate()` 包成原语 + 调本函数。

分布恒等的保障来源:循环体/判定层与 python executor 是**同一份代码**,
本模块只负责状态搬运,不含任何机制逻辑。
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Awaitable, Callable

import numpy as np

from .budget import TrajectoryBudget, TurnRejectCap
from .gate import IntensityGate
from .hook import SpsGenerate, TrajectorySteerState
from .ledger import SteeringLedger
from .wire import ledger_to_wire, steer_state_apply_wire, steer_state_to_wire

__all__ = ["run_steered_turn", "token_output_to_prim"]


class _NullTok:
    """server 侧不需要文本:响应只带 token_ids,text 由持有真 tokenizer 的
    薄客户端重建(rollout server 以 skip_tokenizer_init 运行,别指望它有
    HF tokenizer)。"""

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return ""


def token_output_to_prim(out: Any) -> SimpleNamespace:
    """verl TokenOutput → SpsGenerate 原语消费面(.token_ids / .extra)。

    与训练进程侧 EngineGenerate 的适配**同一份字段清单**(backends.py):
    policy-version 区间、draft 的 log_probs 与 top-K 明细、verify 的
    prompt 打分行、stop_reason。任何一侧改字段名,对拍门先炸。
    """
    extra: dict[str, Any] = {}
    out_extra = dict(out.extra_fields or {})
    # 版本区间语义镜像 llm_server.py 的 manager 层(:273-275):server 的
    # generate() 只写 global_steps,min/max 由 manager setdefault——进程内
    # 原语绕过了 manager,这里必须补齐,否则 hook 的 _merge_policy_range
    # 取不到版本证据(research 行 fail-loud 的数据源)。
    if "global_steps" in out_extra:
        out_extra.setdefault("min_global_steps", out_extra["global_steps"])
        out_extra.setdefault("max_global_steps", out_extra["global_steps"])
    for key in ("min_global_steps", "max_global_steps"):
        if key in out_extra:
            extra[key] = out_extra[key]
    if getattr(out, "log_probs", None) is not None:
        extra["log_probs"] = list(out.log_probs)
    for key in ("response_topk_ids", "response_topk_logprobs", "prompt_ids", "prompt_logprobs"):
        if key in out_extra:
            extra[key] = out_extra[key]
    stop_reason = getattr(out, "stop_reason", None)
    if stop_reason is not None:
        extra["stop_reason"] = stop_reason
    return SimpleNamespace(token_ids=list(out.token_ids), text="", extra=extra)


def _rebuild_state(payload: dict[str, Any]) -> TrajectorySteerState:
    mech = payload["mech"]
    g = mech["gate"]
    b = mech["budget"]
    state = TrajectorySteerState(
        gate=IntensityGate(
            mode=g["mode"], eta_max=g["eta_max"], epsilon=g["epsilon"],
            theta_d=g["theta_d"], entropy_theta=g["entropy_theta"],
            w_min=g["w_min"], rate_bounds=tuple(g["rate_bounds"]),
        ),
        budget=TrajectoryBudget(
            b["b_max"], b["c"], pacing=b["pacing"], u0=b["u0"],
            profile_power=b["profile_power"], t_max=b["t_max"],
        ),
        reject_cap=TurnRejectCap(r_max=mech["r_max"]),
        # 本 turn 从空账记起(增量通道);坐标系(d_hist_max=c²)随快照走,
        # 客户端 merge 的 invariant 校验依赖两侧一致。
        ledger=SteeringLedger(d_hist_max=mech["d_hist_max"]),
        rng=np.random.default_rng(0),  # 占位:立刻被 wire 状态整体覆盖
    )
    steer_state_apply_wire(state, payload["state"])
    return state


async def run_steered_turn(
    payload: dict[str, Any],
    *,
    draft: Callable[[list[int], dict[str, Any]], Awaitable[Any]],
    verify: Callable[[list[int], dict[str, Any]], Awaitable[Any]],
) -> dict[str, Any]:
    """一个 assistant turn 的 server 侧执行。payload 契约见
    engine_client.EngineSteeredGenerate(构造方);响应:

        {"token_ids": [...], "extra": {...},           # GenResult 面(text 客户端重建)
         "state": <wire>, "ledger": <本 turn 增量 wire>}
    """
    mech = payload["mech"]
    state = _rebuild_state(payload)
    hook = SpsGenerate(
        draft=draft,
        verify=verify,
        teacher_prefix_ids=payload["teacher_prefix_ids"],
        student_prompt_ids=payload["student_prompt_ids"],
        state=state,
        tokenizer=_NullTok(),
        topk=mech["topk"],
        c=mech["c"],
        obs_topk=mech["obs_topk"],
        stop_token_ids=frozenset(int(t) for t in mech["stop_token_ids"]),
        draft_horizon=mech["draft_horizon"],
    )
    out = await hook(list(payload["prompt_ids"]), dict(payload["sampling_params"]))
    # aborted 首调透传路径返回原语对象(无 text 语义)——两种对象都只按
    # token_ids/extra 打包,与 GenResult 面一致。
    return {
        "token_ids": [int(t) for t in out.token_ids],
        "extra": dict(out.extra),
        "state": steer_state_to_wire(state),
        "ledger": ledger_to_wire(state.ledger),
    }
