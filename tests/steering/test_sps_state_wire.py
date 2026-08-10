"""状态线协议的正确性红线(Engine-Resident SPS P1 前置件)。

核心断言:「turn 间经 wire(含 JSON 序列化)往返 + 全新对象重建」与
「单进程同一 state 连续执行」在拒绝/预算/pacing/探针全开的世界里
**逐字节等价**——token 流、轨迹总账、budget/cap/gate 状态全同。
这锁死 P1 无状态 RPC 协议的核心正确性;server 侧从空 ledger 记增量、
客户端 merge 的通道也在此一并验证。
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

import numpy as np
import pytest

from trace_opd.steering import IntensityGate, TrajectoryBudget, TurnRejectCap
from trace_opd.steering.hook import SpsGenerate, TrajectorySteerState
from trace_opd.steering.ledger import SteeringLedger, TokenSteerRecord
from trace_opd.steering.wire import (
    ledger_from_wire,
    ledger_to_wire,
    merge_ledger,
    steer_state_apply_wire,
    steer_state_to_wire,
)

from test_sps_bounded_k import (
    EOS,
    PREFIX,
    PROMPT,
    _PositionalDraft,
    _PositionalVerify,
    _PositionalWorld,
)
from test_sps_hook import _Tok

# 拒绝+探针世界:两 turn 共 17 流位置,第 17 位 EOS
_SPEC = [{"y": 5, "alt": 6, "teacher": "objects", "entropy": "high"}] * 16 \
    + [{"y": EOS, "entropy": "high"}]


def _mk_state(seed: int) -> TrajectorySteerState:
    return TrajectorySteerState(
        gate=IntensityGate(mode="entropy", entropy_theta=1.0, eta_max=0.4,
                           epsilon=0.2, w_min=0.05),
        # pacing 开启:unlocked 依赖 turn 计数——turn 状态往返的最强测试面
        budget=TrajectoryBudget(b_max=30.0, c=2.5, pacing=True, u0=5.0,
                                profile_power=2.0, t_max=4),
        reject_cap=TurnRejectCap(r_max=6),
        rng=np.random.default_rng(seed),
    )


def _mk_hook(world: _PositionalWorld, state: TrajectorySteerState) -> SpsGenerate:
    return SpsGenerate(
        draft=_PositionalDraft(world), verify=_PositionalVerify(world),
        teacher_prefix_ids=PREFIX, student_prompt_ids=PROMPT,
        state=state, tokenizer=_Tok(), topk=8, c=2.5, obs_topk=16,
        stop_token_ids={EOS}, draft_horizon=3,
    )


def _turn(hook, prompt_ids, n):
    return asyncio.run(hook(list(prompt_ids), {"max_tokens": n, "temperature": 1.0}))


def test_wire_roundtrip_bitequal_with_single_process():
    world = _PositionalWorld(_SPEC)

    # 路径 A:单进程,同一 state 连续两 turn
    st_a = _mk_state(seed=7)
    hook_a = _mk_hook(world, st_a)
    a1 = _turn(hook_a, PROMPT, 8)
    a2 = _turn(hook_a, PROMPT + list(a1.token_ids), 16)

    # 路径 B:turn 间全量 wire 往返(含 JSON,模拟真实 RPC 载荷)+ 新对象重建
    st_b1 = _mk_state(seed=7)
    b1 = _turn(_mk_hook(world, st_b1), PROMPT, 8)
    wire_state = json.loads(json.dumps(steer_state_to_wire(st_b1)))
    wire_ledger = json.loads(json.dumps(ledger_to_wire(st_b1.ledger)))

    st_b2 = _mk_state(seed=999)          # seed 无关:状态整体被 wire 覆盖
    steer_state_apply_wire(st_b2, wire_state)
    b2 = _turn(_mk_hook(world, st_b2), PROMPT + list(b1.token_ids), 16)

    total = ledger_from_wire(wire_ledger)
    merge_ledger(total, st_b2.ledger)    # server 侧本 turn 增量汇入总账

    assert list(a1.token_ids) == list(b1.token_ids)
    assert list(a2.token_ids) == list(b2.token_ids)
    # 轨迹总账逐字段等价(records 含全部倾斜/拒绝/探针行)
    assert asdict(st_a.ledger) == asdict(total)
    assert st_a.ledger.records and st_a.ledger.records == total.records
    # 轨迹级状态等价
    assert st_a.budget.spent == st_b2.budget.spent
    assert st_a.budget.turn == st_b2.budget.turn
    assert st_a.reject_cap.count == st_b2.reject_cap.count
    assert st_a.gate._decisions == st_b2.gate._decisions
    assert st_a.gate._opens == st_b2.gate._opens
    # 世界确实触发了机制的各条路径(空转世界的等价是平凡的)
    assert any(r.rejected for r in total.records)
    assert st_a.budget.spent > 0


def test_ledger_wire_roundtrip_identity():
    world = _PositionalWorld(_SPEC)
    st = _mk_state(seed=3)
    _turn(_mk_hook(world, st), PROMPT, 12)
    back = ledger_from_wire(json.loads(json.dumps(ledger_to_wire(st.ledger))))
    assert asdict(back) == asdict(st.ledger)
    assert isinstance(back.records[0], TokenSteerRecord)


def test_merge_guard_fails_on_unregistered_field():
    """SteeringLedger 新增字段而 merge 清单未更新 ⇒ 必须炸(静默丢账防线)。"""
    total, delta = SteeringLedger(), SteeringLedger()
    merge_ledger(total, delta)           # 现字段集:干净通过

    from trace_opd.steering import wire as w
    stashed = w._MERGE_ADD
    w._MERGE_ADD = stashed - {"ent_sum"}  # 模拟"忘了登记"
    try:
        with pytest.raises(RuntimeError, match="unregistered"):
            merge_ledger(SteeringLedger(), SteeringLedger())
    finally:
        w._MERGE_ADD = stashed


def test_merge_rejects_config_mismatch_and_bad_positions():
    a, b = SteeringLedger(), SteeringLedger()
    b.d_hist_max = 9.0
    with pytest.raises(RuntimeError, match="d_hist_max"):
        merge_ledger(a, b)

    def _rec(pos):
        return TokenSteerRecord(
            position=pos, log_pi=-1.0, log_b=-1.0, eta=0.1, delta=0.5,
            d_value=0.0, base_open=True, probe_flipped=False, propensity=1.0,
            rejected=False, cap_active=False, budget_after=1.0, unlocked_at=1.0,
            draft_token=1, final_token=1,
        )

    total = SteeringLedger()
    total.append(_rec(10))
    late = SteeringLedger()
    late.records.append(_rec(4))         # 违反 stream 坐标单调:必须炸
    with pytest.raises(ValueError, match="strictly increasing"):
        merge_ledger(total, late)
