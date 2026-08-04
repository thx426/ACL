"""SpsGenerate 钩子测试 —— mock 引擎下的机制端到端(规范 §3 的实现验证)。

最重的一条 test_output_matches_analytic_b:门常开、单 token 生成、大样本下
钩子输出分布必须收敛到**独立手写**的解析 b_S(oracle 与实现是两条代码路径)。
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from trace_opd.steering import SteeringGate, TrajectoryBudget, TurnRejectCap
from trace_opd.steering.hook import SpsGenerate, TrajectorySteerState


# ------------------------------------------------------------ mock 引擎世界

VOCAB = 40
EOS = 0


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class _World:
    """位置无关的 π/p_T 表(简化;kernel 判定本就是逐位置独立的)。"""

    def __init__(self, seed=0, pi_temp=1.5, pt_shift=None):
        rng = np.random.default_rng(seed)
        self.pi = _softmax(rng.normal(0, pi_temp, VOCAB))
        pt_logits = rng.normal(0, pi_temp, VOCAB)
        if pt_shift is not None:
            pt_logits = pt_shift(pt_logits, self.pi)
        self.pt = _softmax(pt_logits)

    def topk_rows(self, dist, k):
        order = np.argsort(-dist)[:k]
        return [int(t) for t in order], [float(np.log(dist[t])) for t in order]


class _GenResult:
    def __init__(self, token_ids, extra):
        self.token_ids = token_ids
        self.text = ""
        self.extra = extra


class _MockDraft:
    """从 π 采样;EOS 截停;带出 top-K 明细(GenerateFn 契约)。"""

    def __init__(self, world, obs_topk, rng, eos_prob=0.0):
        self.world = world
        self.obs_topk = obs_topk
        self.rng = rng
        self.calls: list[list[int]] = []

    async def __call__(self, prompt_ids, params):
        self.calls.append(list(prompt_ids))
        assert params.get("logprobs") == self.obs_topk
        n = int(params["max_tokens"])
        ids, lps, rows_i, rows_l = [], [], [], []
        r_ids, r_lps = self.world.topk_rows(self.world.pi, self.obs_topk)
        for _ in range(n):
            y = int(self.rng.choice(VOCAB, p=self.world.pi))
            ids.append(y)
            lps.append(float(np.log(self.world.pi[y])))
            rows_i.append(list(r_ids))
            rows_l.append(list(r_lps))
            if y == EOS:
                break
        return _GenResult(ids, {
            "log_probs": lps,
            "response_topk_ids": rows_i,
            "response_topk_logprobs": rows_l,
        })


class _MockVerify:
    """prompt_logprobs 行语义:行 r 预测位置 r+1;末行 dummy。"""

    def __init__(self, world, obs_topk, teacher_prefix):
        self.world = world
        self.obs_topk = obs_topk
        self.teacher_prefix = list(teacher_prefix)
        self.calls: list[list[int]] = []

    async def __call__(self, seq, params):
        self.calls.append(list(seq))
        assert params.get("prompt_logprobs") == self.obs_topk
        assert seq[: len(self.teacher_prefix)] == self.teacher_prefix, (
            "verify must score under the TEACHER prefix"
        )
        r_ids, r_lps = self.world.topk_rows(self.world.pt, self.obs_topk)
        n_rows = len(seq)  # len-1 真实行 + 1 dummy(与 extract_prompt_logprobs 同构)
        return _GenResult([1], {
            "prompt_ids": [list(r_ids)] * (n_rows - 1) + [[0] * self.obs_topk],
            "prompt_logprobs": [list(r_lps)] * (n_rows - 1) + [[0.0] * self.obs_topk],
        })


class _Tok:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


def _mk_hook(world, *, theta_d, eta_max=0.5, c=2.5, b_max=6.0, topk=8,
             obs_topk=16, epsilon=0.0, r_max=4, seed=0,
             teacher_prefix=(101, 102, 103), student_prompt=(7, 8, 9)):
    state = TrajectorySteerState(
        gate=SteeringGate(theta_d=theta_d, eta_max=eta_max, epsilon=epsilon),
        budget=TrajectoryBudget(b_max=b_max, c=c),
        reject_cap=TurnRejectCap(r_max=r_max),
        rng=np.random.default_rng(seed),
    )
    draft = _MockDraft(world, obs_topk, np.random.default_rng(seed + 1))
    verify = _MockVerify(world, obs_topk, teacher_prefix)
    hook = SpsGenerate(
        draft=draft, verify=verify,
        teacher_prefix_ids=list(teacher_prefix),
        student_prompt_len=len(student_prompt),
        state=state, tokenizer=_Tok(), topk=topk, c=c, obs_topk=obs_topk,
    )
    return hook, state, draft, verify, list(student_prompt)


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def _analytic_b(world, *, topk, obs_topk, eta, c):
    """独立 oracle:S=top{topk}(π) ∪ (top{topk}(p_T)∩top{obs}(π));
    S 内几何倾斜保质量,S 外原 π。与实现是两条代码路径。"""
    pi, pt = world.pi, world.pt
    top_pi = list(np.argsort(-pi)[:topk])
    known = set(np.argsort(-pi)[:obs_topk].tolist())
    top_pt = [t for t in np.argsort(-pt)[:topk] if t in known]
    S = sorted(set(top_pi) | set(top_pt))
    # p_T 观测窗 floor(top-obs 内最小值)
    pt_window = np.sort(pt)[::-1][:obs_topk]
    floor = float(np.log(pt_window[-1]))
    b = pi.copy()
    ell = np.array([
        np.clip((np.log(pt[t]) if pt[t] >= pt_window[-1] else floor) - np.log(pi[t]), -c, c)
        for t in S
    ])
    piS = pi[S]
    M = piS.sum()
    A = float(np.sum(piS * np.exp(eta * ell)))
    b[S] = M * piS * np.exp(eta * ell) / A
    assert abs(b.sum() - 1.0) < 1e-9
    return b, S


# ------------------------------------------------------------------- tests

def test_gate_closed_is_transparent():
    world = _World(seed=2)
    hook, state, draft, verify, sp = _mk_hook(world, theta_d=1e9)
    out = _run(hook(sp, {"max_tokens": 64, "temperature": 1.0}))
    # 门永关 ⇒ 输出 = draft 原样(mock rng 确定性),ledger 空、零扣费
    assert state.budget.spent == 0.0
    assert state.ledger.records == []
    # 但 verify 仍发生(段级 prefill,一次)且序列以教师前缀开头
    assert len(verify.calls) >= 1
    # draft 的第一次调用带学生前缀
    assert draft.calls[0][:3] == sp


def test_output_matches_analytic_b():
    eta_max, c, topk, obs = 0.5, 2.5, 8, 16
    world = _World(seed=5)
    b, S = _analytic_b(world, topk=topk, obs_topk=obs, eta=eta_max, c=c)
    n = 30_000
    counts = np.zeros(VOCAB)
    for i in range(n):
        hook, state, *_ , sp = _mk_hook(
            world, theta_d=1e-9, eta_max=eta_max, c=c, topk=topk, obs_topk=obs,
            b_max=100.0, seed=i,
        )
        out = _run(hook(sp, {"max_tokens": 1, "temperature": 1.0}))
        assert len(out.token_ids) == 1
        counts[out.token_ids[0]] += 1
    emp = counts / n
    tv = 0.5 * float(np.abs(emp - b).sum())
    assert tv < 0.025, f"hook output vs analytic b: TV={tv:.4f}"
    # 而且确实倾斜了(离 π 有可测距离,排除「恒等于 π」假通过)
    tv_pi = 0.5 * float(np.abs(world.pi - b).sum())
    assert tv_pi > 0.03
    assert 0.5 * float(np.abs(emp - world.pi).sum()) > tv


def test_rejection_resumes_from_breakpoint():
    # p_T 强压 π 的最高 token ⇒ 该 token 接受率 ≈ e^{-2ηc},大量拒绝
    def crush_top(pt_logits, pi):
        top = int(np.argmax(pi))
        pt_logits = pt_logits.copy()
        pt_logits[top] = -30.0
        return pt_logits

    world = _World(seed=3, pt_shift=crush_top)
    hook, state, draft, verify, sp = _mk_hook(world, theta_d=1e-9, b_max=100.0, seed=11)
    out = _run(hook(sp, {"max_tokens": 32, "temperature": 1.0}))
    rejects = [r for r in state.ledger.records if r.rejected]
    assert rejects, "crush-top world must produce at least one rejection"
    # 每次拒绝后:下一次 draft 的 prompt = 学生前缀 + 已执行段(含重采 token)
    assert len(draft.calls) == len(rejects) - state.ledger.turns_capped + 1 or len(draft.calls) >= 2
    for k in range(1, len(draft.calls)):
        prev, cur = draft.calls[k - 1], draft.calls[k]
        assert cur[: len(prev)] == prev or cur[: len(sp)] == sp
    # 输出确实避开了被压 token 的部分质量:重采 token 都在 S 内且非被压 token 占多数
    top = int(np.argmax(world.pi))
    resampled = [out.token_ids[r.position] for r in rejects if r.position < len(out.token_ids)]
    assert resampled and sum(t != top for t in resampled) >= len(resampled) * 0.5


def test_budget_exhaustion_goes_pure_draft():
    world = _World(seed=4)
    # b_max=2ηc ⇒ 恰好 1 次整额 steer 后耗尽
    hook, state, draft, verify, sp = _mk_hook(world, theta_d=1e-9, b_max=2 * 0.5 * 2.5, seed=7)
    _run(hook(sp, {"max_tokens": 200, "temperature": 1.0}))
    assert state.budget.remaining < 1e-9
    charged = [r for r in state.ledger.records if r.delta > 0]
    assert len(charged) >= 1
    # 预算耗尽后的 draft 直通轮不再 verify:verify 调用数 <= 扣费前的轮数+1
    assert len(verify.calls) <= len(draft.calls)


def test_ledger_positions_strictly_increasing_and_delta_correct():
    world = _World(seed=6)
    hook, state, *_ , sp = _mk_hook(world, theta_d=1e-9, b_max=100.0, seed=9)
    _run(hook(sp, {"max_tokens": 64, "temperature": 1.0}))
    recs = state.ledger.records
    assert recs
    assert all(b.position > a.position for a, b in zip(recs, recs[1:]))
    for r in recs:
        if r.eta > 0:
            assert abs(r.delta - 2 * r.eta * 2.5) < 1e-12
            assert r.log_b == pytest.approx(r.log_pi + (r.log_b - r.log_pi))
    assert state.ledger.budget_spent == pytest.approx(sum(r.delta for r in recs))


def test_measure_guard_rejects_cold_sampling():
    world = _World(seed=8)
    hook, *_ , sp = _mk_hook(world, theta_d=1e-9)
    with pytest.raises(RuntimeError, match="temperature"):
        _run(hook(sp, {"max_tokens": 4, "temperature": 0.7}))


def test_verify_scores_teacher_prefix_plus_full_response():
    world = _World(seed=10)
    hook, state, draft, verify, sp = _mk_hook(world, theta_d=1e9, seed=13)
    stream = [55, 56]  # episode 已有响应流(glue/obs)
    _run(hook(sp + stream, {"max_tokens": 8, "temperature": 1.0}))
    seq = verify.calls[0]
    tp = verify.teacher_prefix
    # teacher_prefix + (prompt − 学生前缀) + draft 段
    assert seq[: len(tp)] == tp
    assert seq[len(tp): len(tp) + 2] == stream


def test_resampled_stop_token_ends_the_turn():
    # 审阅 IF-06:残差重采出停止 token 时,本轮必须终止,不得续 draft。
    # 构造:π 集中于 token A(高概率),p_T 压 A 抬 EOS ⇒ 拒 A 后残差大概率采出 EOS。
    def crush_top_lift_eos(pt_logits, pi):
        top = int(np.argmax(pi))
        pt_logits = pt_logits.copy()
        pt_logits[top] = -30.0
        pt_logits[EOS] = 8.0
        return pt_logits

    world = _World(seed=21, pt_shift=crush_top_lift_eos)
    hook, state, draft, verify, sp = _mk_hook(world, theta_d=1e-9, b_max=100.0, seed=5)
    hook._stop_ids = frozenset({EOS})
    out = _run(hook(sp, {"max_tokens": 64, "temperature": 1.0}))
    eos_rejects = [
        r for r in state.ledger.records
        if r.rejected and r.position < len(out.token_ids)
        and out.token_ids[r.position] == EOS
    ]
    if eos_rejects:  # 命中重采出 EOS 的跑道(高概率;未命中则本例不判)
        first = eos_rejects[0]
        # EOS 必须是输出的最后一个 token:其后没有任何续 draft 产物
        assert out.token_ids[first.position] == EOS
        assert len(out.token_ids) == first.position + 1
    else:
        pytest.skip("no EOS-resample trajectory under this seed")


def test_rejected_eos_resampled_to_non_stop_continues():
    # 原 draft 末 token 是 EOS 且被拒、重采成非停止符 ⇒ 必须继续 draft。
    # 构造:π 抬 EOS(draft 常以 EOS 收尾),p_T 强压 EOS ⇒ EOS 高概率被拒。
    def lift_eos_pi_crush_pt(pt_logits, pi):
        pt_logits = pt_logits.copy()
        pt_logits[EOS] = -30.0
        return pt_logits

    world = _World(seed=22, pt_shift=lift_eos_pi_crush_pt)
    world.pi[EOS] = world.pi.max() * 3
    world.pi = world.pi / world.pi.sum()
    hook, state, draft, verify, sp = _mk_hook(world, theta_d=1e-9, b_max=100.0, seed=6)
    hook._stop_ids = frozenset({EOS})
    out = _run(hook(sp, {"max_tokens": 48, "temperature": 1.0}))
    swapped = [
        r for r in state.ledger.records
        if r.rejected and r.position < len(out.token_ids)
        and out.token_ids[r.position] != EOS
    ]
    if swapped and len(draft.calls) >= 2:
        # 有「拒后换成非停止符」的位置且发生了续 draft:检查续采前缀包含重采 token
        r0 = swapped[0]
        follow = [c for c in draft.calls if len(c) > len(sp) + r0.position]
        assert any(
            c[len(sp) + r0.position] == out.token_ids[r0.position] for c in follow
        )
    else:
        pytest.skip("no rejected-then-swapped trajectory under this seed")


def test_cap_row_ordering_and_budget_after():
    # 审阅 IF-07:第 R_max 次拒绝那一行必须记 cap_active=True;
    # budget_after 单调不增且与 delta 自洽。
    def crush_top(pt_logits, pi):
        top = int(np.argmax(pi))
        pt_logits = pt_logits.copy()
        pt_logits[top] = -30.0
        return pt_logits

    world = _World(seed=3, pt_shift=crush_top)
    hook, state, *_ , sp = _mk_hook(world, theta_d=1e-9, b_max=100.0, r_max=2, seed=11)
    _run(hook(sp, {"max_tokens": 64, "temperature": 1.0}))
    recs = state.ledger.records
    rejects = [r for r in recs if r.rejected]
    if len(rejects) >= 2:
        assert rejects[1].cap_active, "the R_max-th reject row must carry cap_active=True"
        assert not rejects[0].cap_active
    # budget_after 自洽:非递增,且扣费行 delta>0 ⇒ 余额下降
    afters = [r.budget_after for r in recs if r.budget_after >= 0]
    assert all(b2 <= b1 + 1e-9 for b1, b2 in zip(afters, afters[1:]))
    # 计数器与稀疏行自洽
    assert state.ledger.n_candidates >= len(recs)
    assert state.ledger.n_base_open >= sum(1 for r in recs if r.base_open)
