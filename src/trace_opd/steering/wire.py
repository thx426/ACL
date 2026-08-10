"""Engine-Resident SPS 的状态线协议(P1 前置件,设计文档 §3.2)。

P1 形态:server actor 无状态,每个 assistant turn 一次 RPC——轨迹级可变
状态(budget/cap/gate 计数/rng)随请求进、随响应回;账本在 server 侧从
**空 ledger** 起记本 turn(天然增量),响应带回后由客户端
``SteeringLedger.merge`` 汇入轨迹总账。

正确性红线:「turn 间经线协议往返」必须与「单进程连续执行」逐字节等价
(rng 用 bit_generator.state 恢复 ⇒ 判定 draw 流无缝续接)——由
tests/steering/test_sps_state_wire.py 的双路径 bit-equal 锁死。

所有 wire 结构均为纯 JSON 兼容类型(RPC 载荷即 dict)。
"""
from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any

import numpy as np

from .hook import TrajectorySteerState
from .ledger import SteeringLedger, TokenSteerRecord

__all__ = [
    "steer_state_to_wire",
    "steer_state_apply_wire",
    "ledger_to_wire",
    "ledger_from_wire",
    "merge_ledger",
]


def _wire_parts(state: TrajectorySteerState) -> list[tuple[str, Any]]:
    return [("budget", state.budget), ("cap", state.reject_cap), ("gate", state.gate)]


def steer_state_to_wire(state: TrajectorySteerState) -> dict[str, Any]:
    """轨迹级可变状态快照(不含 ledger——账本走增量通道)。

    字段清单的单一真源 = 各状态类自己的 ``_WIRE_MUTABLE`` 注册表(可变态
    定义在哪、清单就在哪;缺注册表的门类进 engine 档会在此 fail-loud,
    而不是静默丢态)。"""
    wire: dict[str, Any] = {
        # 协议版本(codex review P2-1):两端代码不同版时 fail-loud,
        # 而不是让缺键/改名伪装成 dataclass 默认值
        "v": 1,
        # numpy Generator 完整状态:恢复后 draw 流与原对象逐位一致
        "rng": state.rng.bit_generator.state,
    }
    for key, obj in _wire_parts(state):
        try:
            names = type(obj)._WIRE_MUTABLE
        except AttributeError as err:
            raise RuntimeError(
                f"{type(obj).__name__} has no _WIRE_MUTABLE registry — this "
                "state class cannot ride the engine-executor wire; register "
                "its cross-turn mutable fields (see TrajectoryBudget)"
            ) from err
        wire[key] = {name: getattr(obj, name) for name in names}
    return wire


def steer_state_apply_wire(state: TrajectorySteerState, wire: dict[str, Any]) -> None:
    """把线协议快照回写进(通常是新构造的)state。构造参数(b_max/c/
    r_max/门配置)不在 wire 里——两端各自从同一份 run 配置构造,漂移由
    对拍门兜底。快照键与注册表逐一对应,任何缺键即协议损坏,fail loud。"""
    if wire.get("v") != 1:
        raise RuntimeError(
            f"state wire protocol version mismatch: got {wire.get('v')!r}, "
            "expected 1 — the two sides run different code versions"
        )
    for key, obj in _wire_parts(state):
        part = wire[key]
        for name in type(obj)._WIRE_MUTABLE:
            setattr(obj, name, part[name])
    state.rng.bit_generator.state = wire["rng"]


def ledger_to_wire(ledger: SteeringLedger) -> dict[str, Any]:
    """全字段 dict 化(records 展开为 dict 列表);JSON 兼容。"""
    out = asdict(ledger)
    out["records"] = [asdict(r) for r in ledger.records]
    return out


def ledger_from_wire(wire: dict[str, Any]) -> SteeringLedger:
    data = dict(wire)
    records = [TokenSteerRecord(**r) for r in data.pop("records")]
    led = SteeringLedger(**data)
    led.records = records
    return led


# SteeringLedger.merge 的字段处置清单(fail-loud 守卫:新增字段没登记进
# 这里会在 merge 时炸——静默丢账是隐性事故,宁可编译期思考一次)。
_MERGE_EXTEND = {"records", "cand_positions", "d_all"}
_MERGE_ADD = {
    "turns_capped", "d_hist", "d_count", "d_sum",
    "n_candidates", "n_base_open", "n_probe_flipped", "n_budget_blocked",
    "n_bypassed", "n_pacing_wait", "n_gate_bypassed", "n_pt_window_miss",
    "n_draft_discarded",
    "ell_hist", "ell_count", "ent_hist", "ent_count", "ent_sum",
}
_MERGE_LATEST = {"budget_spent"}      # 全轨迹累计值,后段覆盖前段
_MERGE_INVARIANT = {"d_hist_max"}     # 配置同源,两段必须一致


def merge_ledger(total: SteeringLedger, delta: SteeringLedger) -> None:
    """把本 turn 的增量账(server 侧起账于轨迹累计)汇入轨迹总账。

    事务式(codex review P2-3):先做全部校验(零副作用),全过再纯累加
    提交——任何校验失败时 total 逐字节未动,调用方可安全放弃本 turn。"""
    # ---- 预检(零副作用)----
    known = _MERGE_EXTEND | _MERGE_ADD | _MERGE_LATEST | _MERGE_INVARIANT
    actual = {f.name for f in fields(SteeringLedger)}
    if actual != known:
        raise RuntimeError(
            "SteeringLedger fields changed without updating the merge policy: "
            f"unregistered={sorted(actual - known)}, stale={sorted(known - actual)}"
        )
    if delta.d_hist_max != total.d_hist_max:
        raise RuntimeError(
            f"d_hist_max mismatch: {total.d_hist_max} vs {delta.d_hist_max} — "
            "the two sides were built from different run configs"
        )
    # budget_spent 是轨迹累计(单调不减):delta 由 server 起账时写入轨迹
    # 累计(server_loop._rebuild_state,codex review P0-3),回退即协议损坏。
    if delta.budget_spent + 1e-9 < total.budget_spent:
        raise RuntimeError(
            f"budget_spent went backwards across the wire: total="
            f"{total.budget_spent} -> delta={delta.budget_spent} — the server "
            "turn ledger was not seeded with the trajectory total"
        )
    last = total.records[-1].position if total.records else -1
    for rec in delta.records:
        if rec.position <= last:
            raise ValueError("ledger positions must be strictly increasing")
        last = rec.position
    for name in _MERGE_ADD:
        a, b = getattr(total, name), getattr(delta, name)
        if isinstance(a, list) and len(a) != len(b):
            raise RuntimeError(f"histogram width mismatch on {name}: {len(a)} vs {len(b)}")
    # ---- 提交(纯累加,不失败)----
    total.records.extend(delta.records)
    for name in _MERGE_ADD:
        a, b = getattr(total, name), getattr(delta, name)
        if isinstance(a, list):
            setattr(total, name, [x + y for x, y in zip(a, b)])
        else:
            setattr(total, name, a + b)
    for name in _MERGE_EXTEND - {"records"}:
        getattr(total, name).extend(getattr(delta, name))
    total.budget_spent = delta.budget_spent
