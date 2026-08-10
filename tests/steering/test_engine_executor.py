"""Engine executor(P1)端到端协议对拍 + 原语适配 + 配置校验。

核心断言:`EngineSteeredGenerate`(薄客户端)→ fake server_manager(payload/
response 经 **pickle 往返**,模拟 Ray actor 调用的序列化边界)→
`run_steered_turn`(server 侧全部业务逻辑,fake 原语)——与单进程
`SpsGenerate` 在拒绝/pacing/探针全开世界里两 turn **逐字节等价**:token 流、
GenResult.extra、轨迹总账、budget/cap/gate 状态。客户端-服务器链上除了
engine 原语本身(P0b 单卡对拍),没有任何未被本测试覆盖的代码。
"""
from __future__ import annotations

import asyncio
import pickle
from dataclasses import asdict

import numpy as np
import pytest

from trace_opd.steering import IntensityGate, TrajectoryBudget, TurnRejectCap
from trace_opd.steering.engine_client import EngineSteeredGenerate
from trace_opd.steering.hook import SpsGenerate, TrajectorySteerState
from trace_opd.steering.ledger import SteeringLedger
from trace_opd.steering.server_loop import run_steered_turn, token_output_to_prim

from test_sps_bounded_k import (
    EOS,
    PREFIX,
    PROMPT,
    _PositionalDraft,
    _PositionalVerify,
    _PositionalWorld,
)
from test_sps_hook import _Tok

_SPEC = [{"y": 5, "alt": 6, "teacher": "objects", "entropy": "high"}] * 16 \
    + [{"y": EOS, "entropy": "high"}]


def _mk_state(seed: int) -> TrajectorySteerState:
    return TrajectorySteerState(
        gate=IntensityGate(mode="entropy", entropy_theta=1.0, eta_max=0.4,
                           epsilon=0.2, w_min=0.05),
        budget=TrajectoryBudget(b_max=30.0, c=2.5, pacing=True, u0=5.0,
                                profile_power=2.0, t_max=4),
        reject_cap=TurnRejectCap(r_max=6),
        ledger=SteeringLedger(d_hist_max=2.5 * 2.5),
        rng=np.random.default_rng(seed),
    )


class _FakeServerManager:
    """steered_generate 的 manager double:pickle 往返 payload 与响应
    (Ray 调用的序列化面),server 侧跑真 run_steered_turn + fake 原语。"""

    def __init__(self, world: _PositionalWorld):
        self._world = world
        self.turns = 0

    async def steered_generate(self, request_id, *, payload):  # noqa: ARG002
        self.turns += 1
        payload = pickle.loads(pickle.dumps(payload))
        resp = await run_steered_turn(
            payload,
            draft=_PositionalDraft(self._world),
            verify=_PositionalVerify(self._world),
        )
        return pickle.loads(pickle.dumps(resp))


def test_engine_executor_two_turns_bitequal_with_python_executor():
    world = _PositionalWorld(_SPEC)

    # 路径 A:python executor(单进程 SpsGenerate,draft_horizon=0 历史形态)
    st_a = _mk_state(seed=11)
    hook = SpsGenerate(
        draft=_PositionalDraft(world), verify=_PositionalVerify(world),
        teacher_prefix_ids=PREFIX, student_prompt_ids=PROMPT,
        state=st_a, tokenizer=_Tok(), topk=8, c=2.5, obs_topk=16,
        stop_token_ids={EOS},
    )
    a1 = asyncio.run(hook(list(PROMPT), {"max_tokens": 8, "temperature": 1.0}))
    a2 = asyncio.run(hook(list(PROMPT) + list(a1.token_ids),
                          {"max_tokens": 16, "temperature": 1.0}))

    # 路径 B:engine executor(薄客户端 + fake server,含 pickle 序列化边界;
    # draft_horizon=3 一并覆盖 bounded K 在 RPC 链上的恒等)
    st_b = _mk_state(seed=11)
    mgr = _FakeServerManager(world)
    client = EngineSteeredGenerate(
        server_manager=mgr, tokenizer=_Tok(), request_id="route-1",
        teacher_prefix_ids=PREFIX, student_prompt_ids=PROMPT, state=st_b,
        topk=8, c=2.5, obs_topk=16, stop_token_ids={EOS}, draft_horizon=3,
    )
    b1 = asyncio.run(client(list(PROMPT), {"max_tokens": 8, "temperature": 1.0}))
    b2 = asyncio.run(client(list(PROMPT) + list(b1.token_ids),
                            {"max_tokens": 16, "temperature": 1.0}))

    assert mgr.turns == 2                       # 每 turn 恰一次 RPC
    assert list(a1.token_ids) == list(b1.token_ids)
    assert list(a2.token_ids) == list(b2.token_ids)
    assert a1.text == b1.text and a2.text == b2.text
    assert a2.extra == b2.extra
    # 轨迹真源等价:总账逐字段、budget/cap/gate 状态、rng 续接
    assert asdict(st_a.ledger) == asdict(st_b.ledger)
    assert st_a.ledger.records and st_a.ledger.records == st_b.ledger.records
    assert st_a.budget.spent == st_b.budget.spent
    assert st_a.budget.turn == st_b.budget.turn == 2
    assert st_a.reject_cap.count == st_b.reject_cap.count
    assert st_a.gate._decisions == st_b.gate._decisions
    assert st_a.gate._opens == st_b.gate._opens
    assert any(r.rejected for r in st_b.ledger.records)   # 非平凡世界
    assert st_b.budget.spent > 0


def test_engine_executor_abort_passthrough():
    """draft 首调 aborted:episode 语义 = 基建故障透传(token 空、
    stop_reason=aborted),客户端与状态不被污染。"""
    world = _PositionalWorld(_SPEC)

    class _AbortDraft(_PositionalDraft):
        async def __call__(self, prompt_ids, params):
            out = await super().__call__(prompt_ids, params)
            out.token_ids = []
            out.extra["stop_reason"] = "aborted"
            return out

    class _AbortManager(_FakeServerManager):
        async def steered_generate(self, request_id, *, payload):  # noqa: ARG002
            payload = pickle.loads(pickle.dumps(payload))
            resp = await run_steered_turn(
                payload, draft=_AbortDraft(self._world),
                verify=_PositionalVerify(self._world),
            )
            return pickle.loads(pickle.dumps(resp))

    st = _mk_state(seed=1)
    client = EngineSteeredGenerate(
        server_manager=_AbortManager(world), tokenizer=_Tok(), request_id="r",
        teacher_prefix_ids=PREFIX, student_prompt_ids=PROMPT, state=st,
        topk=8, c=2.5, obs_topk=16, stop_token_ids={EOS}, draft_horizon=0,
    )
    out = asyncio.run(client(list(PROMPT), {"max_tokens": 8, "temperature": 1.0}))
    assert out.token_ids == [] and out.extra.get("stop_reason") == "aborted"
    assert st.ledger.records == []


# ------------------------------------------------- 原语适配(server 接线面)

class _FakeTokenOutput:
    def __init__(self, **kw):
        self.token_ids = kw.pop("token_ids")
        self.log_probs = kw.pop("log_probs", None)
        self.stop_reason = kw.pop("stop_reason", None)
        self.extra_fields = kw.pop("extra_fields", {})


def test_token_output_to_prim_mirrors_manager_version_defaults():
    # server 的 generate() 只写 global_steps;min/max 由 manager setdefault——
    # 进程内原语绕过 manager,适配层必须补齐(policy-version 证据数据源)。
    prim = token_output_to_prim(_FakeTokenOutput(
        token_ids=[1, 2], log_probs=[-0.1, -0.2], stop_reason="completed",
        extra_fields={"global_steps": 7,
                      "response_topk_ids": [[1]], "response_topk_logprobs": [[-0.1]]},
    ))
    assert prim.token_ids == [1, 2]
    assert prim.extra["min_global_steps"] == 7 and prim.extra["max_global_steps"] == 7
    assert prim.extra["log_probs"] == [-0.1, -0.2]
    assert prim.extra["response_topk_ids"] == [[1]]
    assert prim.extra["stop_reason"] == "completed"


def test_token_output_to_prim_respects_explicit_range():
    prim = token_output_to_prim(_FakeTokenOutput(
        token_ids=[], extra_fields={"global_steps": 9, "min_global_steps": 3,
                                    "max_global_steps": 9},
    ))
    assert prim.extra["min_global_steps"] == 3 and prim.extra["max_global_steps"] == 9


# ------------------------------------------------------------- config 校验

from test_budget_config import GOOD, _load  # noqa: E402


def _steering_line(**over):
    base = dict(enabled=True, doc_source="researcher", theta_d=0.8)
    base.update(over)
    items = ", ".join(
        f"{k}: {str(v).lower() if isinstance(v, bool) else v}" for k, v in base.items()
    )
    return f"steering: {{{items}}}\n"


def test_executor_config_validation(tmp_path, monkeypatch):
    import pytest as _pytest

    # python 档 + 非零 horizon:HTTP 红线,拒绝
    with _pytest.raises(ValueError, match="draft_horizon"):
        cfg = _load(tmp_path, monkeypatch, **GOOD,
                    extra=_steering_line(draft_horizon=64))
    # engine 档 + vllm 后端:P1 实现面,拒绝
    with _pytest.raises(ValueError, match="rollout_backend=sglang"):
        _load(tmp_path, monkeypatch, **GOOD, extra=_steering_line(executor="engine"))
    # engine 档 + sglang + bounded K:合法
    cfg = _load(tmp_path, monkeypatch, **GOOD,
                train=", rollout_backend: sglang",
                extra=_steering_line(executor="engine", draft_horizon=64))
    assert cfg.steering.executor == "engine" and cfg.steering.draft_horizon == 64
    # 未知档名:拒绝
    with _pytest.raises(ValueError, match="steering.executor"):
        _load(tmp_path, monkeypatch, **GOOD, extra=_steering_line(executor="cuda"))
