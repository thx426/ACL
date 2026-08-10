"""bounded K(draft_horizon)分布恒等的机器证明(Engine-Resident SPS P0a)。

设计文档 docs/plans/2026-08-10-engine-resident-sps-design.md §4:SPS 每个
位置的接受/重采只依赖(前缀,该位置 π/p_T 明细,rng draw),draft 分几段
不进入任何判定量 ⇒ 逐位分布恒等。本文件把它做成可执行断言:**按位置确定**
的 fake 世界(消除顺序 rng 采样的实现伪影——真实引擎每次请求独立采样,
分布层面天然满足)下,K=∞ 与任意 K 的输出 token 流与账本逐字节相等。

rng 对齐论证(测试的可行性依据,亦是恒等证明的一部分):
  * 判定 rng 只在两处消耗——gate.decide 的探针 draw(entropy 门有
    effective>0 短路,只在 base_open 位发生)与 accept_or_resample
    (只在 eta>0 位发生);
  * G-E 段级 bypass 只吞掉「整窗 w<w_min」的位置,这些位置在任何路径下
    都不产生 draw ⇒ 两路径 draw 序列按执行位置严格对齐,bit-equal 成立
    (含 ε>0 的探针档与 bypass 触发面不同的用例)。
"""
from __future__ import annotations

import asyncio
import math

import numpy as np
import pytest

from trace_opd.steering import IntensityGate, TrajectoryBudget, TurnRejectCap
from trace_opd.steering.hook import SpsGenerate, TrajectorySteerState

from test_sps_hook import _GenResult, _Tok

OBS = 16
EOS = 0
PREFIX = [101, 102, 103]      # teacher(doc)前缀
PROMPT = [7, 8, 9]            # 学生 prompt

# 明细行模板:低熵(top1≈0.999)与高熵(均匀 16)
_LOW = [math.log(0.999)] + [math.log(0.001 / (OBS - 1))] * (OBS - 1)
_HIGH = [math.log(1.0 / OBS)] * OBS


def _row(ids_head: list[int], lps: list[float]) -> tuple[list[int], list[float]]:
    ids = list(ids_head) + [i + 20 for i in range(len(ids_head), OBS)]
    return ids[:OBS], list(lps)[:OBS]


class _PositionalWorld:
    """流位置 pos → (draft_token, π 明细行, p_T 明细行)。

    - draft token 查表(位置确定,与请求分段无关);
    - π 行的 top1 = draft token(log_pi 取自行首);
    - p_T 行按 spec 表达异议或跟随。
    """

    def __init__(self, spec: list[dict]):
        self.spec = spec

    def n(self) -> int:
        return len(self.spec)

    def draft_token(self, pos: int) -> int:
        return int(self.spec[pos]["y"])

    def pi_row(self, pos: int):
        s = self.spec[pos]
        return _row([s["y"], s.get("alt", s["y"] + 1)],
                    _LOW if s.get("entropy") == "low" else _HIGH)

    def pt_row(self, pos: int):
        s = self.spec[pos]
        if s.get("teacher") == "objects":       # 强异议:偏好 alt,压 y
            head = [s.get("alt", s["y"] + 1), s["y"]]
            lps = [math.log(0.999)] + [math.log(0.001 / (OBS - 1))] * (OBS - 1)
            return _row(head, lps)
        return self.pi_row(pos)                 # 跟随:p_T = π


class _PositionalDraft:
    def __init__(self, world: _PositionalWorld):
        self.world = world
        self.calls: list[int] = []              # 每次请求的 max_tokens

    async def __call__(self, prompt_ids, params):
        n = int(params["max_tokens"])
        self.calls.append(n)
        base = len(prompt_ids) - len(PROMPT)    # 已执行流长度 = 起始流位置
        ids, lps, rows_i, rows_l = [], [], [], []
        for k in range(n):
            pos = base + k
            if pos >= self.world.n():
                break
            y = self.world.draft_token(pos)
            r_ids, r_lps = self.world.pi_row(pos)
            ids.append(y)
            lps.append(r_lps[r_ids.index(y)])
            rows_i.append(r_ids)
            rows_l.append(r_lps)
            if y == EOS:
                break
        return _GenResult(ids, {
            "log_probs": lps,
            "response_topk_ids": rows_i,
            "response_topk_logprobs": rows_l,
        })


class _PositionalVerify:
    def __init__(self, world: _PositionalWorld):
        self.world = world
        self.calls: list[list[int]] = []

    async def __call__(self, seq, params):
        self.calls.append(list(seq))
        assert seq[: len(PREFIX)] == PREFIX
        rows_i, rows_l = [], []
        for r in range(len(seq) - 1):
            pos = r + 1 - len(PREFIX)           # 行 r 预测 seq 位置 r+1 的流坐标
            if pos < 0 or pos >= self.world.n():
                r_ids, r_lps = _row([1, 2], _HIGH)
            else:
                r_ids, r_lps = self.world.pt_row(pos)
            rows_i.append(r_ids)
            rows_l.append(r_lps)
        rows_i.append([0] * OBS)                # 末行 dummy(行语义同构)
        rows_l.append([0.0] * OBS)
        return _GenResult([1], {"prompt_ids": rows_i, "prompt_logprobs": rows_l})


def _run_hook(world, gate, *, horizon: int, seed=42, max_tokens=64,
              b_max=50.0, r_max=50):
    state = TrajectorySteerState(
        gate=gate,
        budget=TrajectoryBudget(b_max=b_max, c=2.5),
        reject_cap=TurnRejectCap(r_max=r_max),
        rng=np.random.default_rng(seed),
    )
    draft = _PositionalDraft(world)
    hook = SpsGenerate(
        draft=draft, verify=_PositionalVerify(world),
        teacher_prefix_ids=PREFIX, student_prompt_ids=PROMPT,
        state=state, tokenizer=_Tok(), topk=8, c=2.5, obs_topk=OBS,
        stop_token_ids={EOS}, draft_horizon=horizon,
    )
    out = asyncio.run(hook(list(PROMPT), {"max_tokens": max_tokens,
                                          "temperature": 1.0}))
    return out, state, draft


def _adaptive(theta_d):
    return IntensityGate(mode="adaptive", theta_d=theta_d, eta_max=0.5)


def _entropy_gate(epsilon=0.0):
    return IntensityGate(mode="entropy", entropy_theta=1.0, eta_max=0.5,
                         epsilon=epsilon, w_min=0.05)


def _assert_bitequal(a, b, *, ledger_too=True):
    out_a, st_a, _ = a
    out_b, st_b, _ = b
    assert list(out_a.token_ids) == list(out_b.token_ids)
    if ledger_too:
        assert st_a.ledger.records == st_b.ledger.records
        assert st_a.ledger.n_candidates == st_b.ledger.n_candidates
        assert st_a.ledger.budget_spent == st_b.ledger.budget_spent


# ---------------------------------------------------------------- 恒等用例

@pytest.mark.parametrize("horizon", [1, 3, 5, 7])
def test_rejection_path_bitequal(horizon):
    """强异议世界(拒绝+重采反复发生):任意 K 与整段逐字节同流同账。"""
    spec = [{"y": 5, "alt": 6, "teacher": "objects", "entropy": "high"}] * 20 \
        + [{"y": EOS, "entropy": "high"}]
    world = _PositionalWorld(spec)
    ref = _run_hook(world, _adaptive(theta_d=1e-9), horizon=0)
    got = _run_hook(world, _adaptive(theta_d=1e-9), horizon=horizon)
    _assert_bitequal(ref, got)
    # 分段确实发生:请求次数更多、单次请求 ≤K
    assert len(got[2].calls) >= len(ref[2].calls)
    assert all(n <= horizon for n in got[2].calls)


def test_gate_bypass_divergence_tokens_and_tilt_records_equal():
    """G-E 门:K 窗全低熵段在 K 路径 bypass、整段路径不 bypass——
    执行 token 流与实际倾斜记录仍逐字节相等;差异只在覆盖面型计数
    (bypass 位不 observe D),这是观察面而非分布面。"""
    spec = ([{"y": 3, "entropy": "low"}] * 4
            + [{"y": 5, "alt": 6, "teacher": "objects", "entropy": "high"}] * 4) * 3 \
        + [{"y": EOS, "entropy": "low"}]
    world = _PositionalWorld(spec)
    ref = _run_hook(world, _entropy_gate(), horizon=0)
    got = _run_hook(world, _entropy_gate(), horizon=4)
    _assert_bitequal(ref, got, ledger_too=False)
    assert ref[1].ledger.records == got[1].ledger.records   # 倾斜位账逐字节同
    # 覆盖面差异方向:K=4 的纯低熵窗触发段级 bypass,整段路径(窗含高熵位)不触发
    assert got[1].ledger.n_gate_bypassed > ref[1].ledger.n_gate_bypassed


def test_probe_epsilon_path_bitequal():
    """ε>0 探针档(entropy 门):draw 只在 base_open 位发生 ⇒ 分段不
    错位 rng 流,全账 bit-equal。"""
    spec = [{"y": 5, "alt": 6, "teacher": "objects", "entropy": "high"}] * 24 \
        + [{"y": EOS, "entropy": "high"}]
    world = _PositionalWorld(spec)
    ref = _run_hook(world, _entropy_gate(epsilon=0.3), horizon=0)
    got = _run_hook(world, _entropy_gate(epsilon=0.3), horizon=5)
    _assert_bitequal(ref, got)


def test_eos_inside_a_window_ends_identically():
    """EOS 落在 K 窗中间:打满窗是分段边界而非终止,自然停仍只看
    「引擎在请求量内停下」——两路径停点一致。"""
    spec = [{"y": 4, "entropy": "high"}] * 6 + [{"y": EOS, "entropy": "high"}]
    world = _PositionalWorld(spec)
    ref = _run_hook(world, _adaptive(theta_d=1e9), horizon=0)   # 门关:纯直通
    got = _run_hook(world, _adaptive(theta_d=1e9), horizon=4)
    _assert_bitequal(ref, got)
    assert list(ref[0].token_ids)[-1] == EOS


def test_horizon_zero_equals_huge_horizon():
    """退化等价:0(不分段)与 K≥max_tokens 完全同构。"""
    spec = [{"y": 5, "alt": 6, "teacher": "objects", "entropy": "high"}] * 12 \
        + [{"y": EOS, "entropy": "high"}]
    world = _PositionalWorld(spec)
    ref = _run_hook(world, _adaptive(theta_d=1e-9), horizon=0)
    got = _run_hook(world, _adaptive(theta_d=1e-9), horizon=10 ** 9)
    _assert_bitequal(ref, got)
    assert got[2].calls == ref[2].calls   # 请求序列也一致(min(remaining, 巨 K)=remaining)
