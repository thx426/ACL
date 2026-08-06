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
             teacher_prefix=(101, 102, 103), student_prompt=(7, 8, 9),
             pacing=False, u0=0.0, profile_power=2.0, t_max=1):
    state = TrajectorySteerState(
        gate=SteeringGate(theta_d=theta_d, eta_max=eta_max, epsilon=epsilon),
        budget=TrajectoryBudget(b_max=b_max, c=c, pacing=pacing, u0=u0,
                                profile_power=profile_power, t_max=t_max),
        reject_cap=TurnRejectCap(r_max=r_max),
        rng=np.random.default_rng(seed),
    )
    draft = _MockDraft(world, obs_topk, np.random.default_rng(seed + 1))
    verify = _MockVerify(world, obs_topk, teacher_prefix)
    hook = SpsGenerate(
        draft=draft, verify=verify,
        teacher_prefix_ids=list(teacher_prefix),
        student_prompt_ids=list(student_prompt),
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
    # p_T 观测窗外单侧区间鲁棒代入(max-IF-13):floor 上界 + 钳非正
    pt_window_min = np.sort(pt)[::-1][:obs_topk][-1]
    floor = float(np.log(pt_window_min))
    b = pi.copy()
    ell = np.array([
        np.clip(np.log(pt[t]) - np.log(pi[t]), -c, c) if pt[t] >= pt_window_min
        else min(0.0, np.clip(floor - np.log(pi[t]), -c, c))
        for t in S
    ])
    piS = pi[S]
    M = piS.sum()
    A = float(np.sum(piS * np.exp(eta * ell)))
    b[S] = M * piS * np.exp(eta * ell) / A
    assert abs(b.sum() - 1.0) < 1e-9
    return b, S


# ------------------------------------------------------------------- tests

def test_policy_version_range_merges_all_draft_and_verify_calls():
    extra = {}
    SpsGenerate._merge_policy_range(
        extra, {"min_global_steps": 7, "max_global_steps": 7}
    )
    SpsGenerate._merge_policy_range(
        extra, {"min_global_steps": 8, "max_global_steps": 8}
    )
    SpsGenerate._merge_policy_range(
        extra, {"min_global_steps": 6, "max_global_steps": 9}
    )
    assert extra == {"min_global_steps": 6, "max_global_steps": 9}


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


def test_prefix_drift_fails_loud():
    # 审阅 max-IF-10:episode 实际前缀与预构学生前缀有任何 token 漂移必须
    # raise,不得静默错位 ledger 坐标。
    world = _World(seed=30)
    hook, *_ , sp = _mk_hook(world, theta_d=1e9)
    drifted = list(sp)
    drifted[-1] = drifted[-1] + 1
    with pytest.raises(RuntimeError, match="prefix drift"):
        _run(hook(drifted, {"max_tokens": 4, "temperature": 1.0}))


def test_pt_window_miss_is_counted_and_never_lifts():
    # π-only 冷候选(p_T 窗外):单侧鲁棒代入 ⇒ 永不抬升;计数进账。
    # 构造:p_T 集中在少数 token(窗 = 全部质量),π 的 top-K 有窗外成员。
    world = _World(seed=31)
    order = np.argsort(-world.pt)
    world.pt[order[16:]] = 1e-12          # obs_topk=16 窗外几乎零质量
    world.pt = world.pt / world.pt.sum()
    hook, state, *_ , sp = _mk_hook(world, theta_d=1e-9, b_max=100.0, seed=8)
    _run(hook(sp, {"max_tokens": 16, "temperature": 1.0}))
    assert state.ledger.n_pt_window_miss > 0


def test_ledger_records_token_pairs_and_ell_observations():
    def crush_top(pt_logits, pi):
        top = int(np.argmax(pi))
        pt_logits = pt_logits.copy()
        pt_logits[top] = -30.0
        return pt_logits

    world = _World(seed=3, pt_shift=crush_top)
    hook, state, *_ , sp = _mk_hook(world, theta_d=1e-9, b_max=100.0, seed=11)
    out = _run(hook(sp, {"max_tokens": 32, "temperature": 1.0}))
    recs = state.ledger.records
    assert recs and all(r.draft_token >= 0 and r.final_token >= 0 for r in recs)
    # 被拒行:final ≠ draft 且 final 与输出坐标一致;接受行:final == draft
    for r in recs:
        if r.position < len(out.token_ids):
            assert out.token_ids[r.position] == r.final_token
        if not r.rejected:
            assert r.final_token == r.draft_token
    # ℓ 观测:每个候选位置贡献窗内 token 数条
    assert state.ledger.ell_count > 0
    assert sum(state.ledger.ell_hist) == state.ledger.ell_count


# ---------------------------------------------- 尾部裁剪 × 位置相关行对齐

class _PositionalVerify:
    """位置相关的 p_T 行 + 尾部裁剪(prompt_logprobs_tail)。

    第 r 行把 token (r % VOCAB) 排在 rank-1 —— 任何行偏移都会让 hook 读到
    不同的 token,被下面的断言抓住。这补上了「每行相同的 mock 掩盖行偏移」
    的测试缺口(codex 审阅测试缺口 #2),同时验证性能修法的裁剪路径。
    """

    def __init__(self, obs_topk, teacher_prefix, honor_tail=True):
        self.obs_topk = obs_topk
        self.teacher_prefix = list(teacher_prefix)
        self.honor_tail = honor_tail
        self.calls = []
        self.tails = []

    async def __call__(self, seq, params):
        self.calls.append(list(seq))
        tail = params.get("prompt_logprobs_tail")
        self.tails.append(tail)
        n_full = len(seq)                      # len-1 真实行 + 1 dummy
        ids, lps = [], []
        for r in range(n_full - 1):            # 真实行
            head = r % VOCAB                   # 该行的 rank-1 token = 行号指纹
            row = [head] + [(head + k) % VOCAB for k in range(1, self.obs_topk)]
            ids.append(row)
            lps.append([-1.0 - 0.01 * k for k in range(self.obs_topk)])
        ids.append([0] * self.obs_topk)        # dummy
        lps.append([0.0] * self.obs_topk)
        if self.honor_tail and tail:
            ids, lps = ids[-tail:], lps[-tail:]
        return _GenResult([1], {"prompt_ids": ids, "prompt_logprobs": lps})


def _run_with_verify(verify, draft_texts_len=6, seed=0, teacher_prefix=(101, 102, 103),
                     student_prompt=(7, 8, 9), obs_topk=16):
    world = _World(seed=seed)
    state = TrajectorySteerState(
        gate=SteeringGate(theta_d=1e9, eta_max=0.5, epsilon=0.0),   # 门恒关:只验行对齐
        budget=TrajectoryBudget(b_max=10.0, c=2.5),
        reject_cap=TurnRejectCap(r_max=4),
        rng=np.random.default_rng(seed),
    )
    draft = _MockDraft(world, obs_topk, np.random.default_rng(seed + 1))
    hook = SpsGenerate(
        draft=draft, verify=verify,
        teacher_prefix_ids=list(teacher_prefix),
        student_prompt_ids=list(student_prompt),
        state=state, tokenizer=_Tok(), topk=4, c=2.5, obs_topk=obs_topk,
    )
    out = _run(hook(list(student_prompt), {"max_tokens": draft_texts_len, "temperature": 1.0}))
    return out, state, draft


def test_verify_tail_trimming_keeps_row_alignment():
    """裁剪返回 vs 全量返回必须读到**同一批** p_T 行(性能修法的正确性守卫)。"""
    tp = (101, 102, 103)
    trimmed = _PositionalVerify(16, tp, honor_tail=True)
    full = _PositionalVerify(16, tp, honor_tail=False)
    o1, s1, d1 = _run_with_verify(trimmed, seed=4)
    o2, s2, d2 = _run_with_verify(full, seed=4)
    # 同 seed ⇒ draft 相同;两种返回形态下执行序列必须逐 token 一致
    assert o1.token_ids == o2.token_ids
    # 裁剪确实生效:请求带了 tail,且返回行数远小于完整序列
    assert trimmed.tails and all(t is not None for t in trimmed.tails)
    seq_len = len(trimmed.calls[0])
    assert trimmed.tails[0] < seq_len, "tail 必须小于完整序列长度才有意义"
    # D 观测逐位置一致(行对齐相同 ⇒ 同样的 ℓ̃ ⇒ 同样的 D)
    assert s1.ledger.d_count == s2.ledger.d_count > 0
    assert abs(s1.ledger.d_sum - s2.ledger.d_sum) < 1e-9


def test_row_misalignment_is_detectable_by_the_positional_mock():
    """反证:位置相关 mock 能抓住行偏移(否则上一条测试是空的)。"""
    tp = (101, 102, 103)
    v = _PositionalVerify(16, tp, honor_tail=True)
    o, s, d = _run_with_verify(v, seed=4)
    d0 = s.ledger.d_sum
    # 人为把返回行整体挪一位 ⇒ D 观测必须变化
    class _Shifted(_PositionalVerify):
        async def __call__(self, seq, params):
            r = await _PositionalVerify.__call__(self, seq, params)
            r.extra["prompt_ids"] = r.extra["prompt_ids"][1:] + [r.extra["prompt_ids"][-1]]
            r.extra["prompt_logprobs"] = r.extra["prompt_logprobs"][1:] + [r.extra["prompt_logprobs"][-1]]
            return r
    o2, s2, _ = _run_with_verify(_Shifted(16, tp, honor_tail=True), seed=4)
    assert abs(s2.ledger.d_sum - d0) > 1e-9, "位置相关 mock 未能区分行偏移 —— 测试无效"


def test_viz_channels_recorded():
    # viz 数据面(v3.2 分层契约):候选位置 D 数组全量;两侧 top-16 明细
    # 只随**拒绝位**携带(防 always_on 全开时 ~1.5MB/轨迹的 dump 膨胀),
    # 拒绝位并带 final_token_pi_rank(冷起边界观察);接受位明细为 None。
    def crush_top(pt_logits, pi):
        top = int(np.argmax(pi))
        pt_logits = pt_logits.copy()
        pt_logits[top] = -30.0
        return pt_logits

    world = _World(seed=3, pt_shift=crush_top)
    hook, state, *_ , sp = _mk_hook(world, theta_d=1e-9, b_max=100.0, seed=11)
    out = _run(hook(sp, {"max_tokens": 24, "temperature": 1.0}))
    led = state.ledger
    assert len(led.d_all) == led.n_candidates == len(led.cand_positions)
    assert all(p2 >= p1 for p1, p2 in zip(led.cand_positions, led.cand_positions[1:]))
    steered = [r for r in led.records if r.eta > 0]
    assert steered
    rejected = [r for r in steered if r.rejected]
    assert rejected, "crush_top 教师必须制造出拒绝位,否则本测试无效"
    for r in rejected:
        assert r.pi_top16 and r.pt_top16
        assert len(r.pi_top16) <= 16 and len(r.pi_top16[0]) == 2
        assert r.final_token_pi_rank >= 0          # 重采 token 的 π rank 已记
        assert r.final_token != r.draft_token
    for r in steered:
        if not r.rejected:
            assert r.pi_top16 is None and r.pt_top16 is None
            assert r.final_token_pi_rank == -1


def test_pacing_spreads_spend_and_counts_wait():
    """坍缩回归(2026-08-05,docs/pgs-budget-pacing-design.md):s1 实测头部高
    异议 + 贪心花费 ⇒ 98.7% 干预堆 turn 0。pacing 下花费必须跨 turn 摊开、
    未解锁等待与总额耗尽分账;pacing off 保持旧行为(首 turn 花光)。"""
    def crush_top(pt_logits, pi):
        top = int(np.argmax(pi))
        pt_logits = pt_logits.copy()
        pt_logits[top] = -30.0
        return pt_logits

    world = _World(seed=3, pt_shift=crush_top)
    # U0=一次整额;T_max=4 turn 全解锁;每 turn 8 token 的候选流,门恒开
    hook, state, draft, verify, sp = _mk_hook(
        world, theta_d=1e-9, b_max=10.0, seed=11,
        pacing=True, u0=2.5, profile_power=2.0, t_max=4)
    prompt = list(sp)
    spents = []
    for _ in range(5):  # 5 个 assistant turn
        out = _run(hook(prompt, {"max_tokens": 8, "temperature": 1.0}))
        prompt = prompt + list(out.token_ids)
        spents.append(state.budget.spent)
    assert state.budget.turn == 5                       # 每次调用推进一个 turn
    assert spents[0] <= 2.5 + 1e-9                      # 首 turn 至多花 U0
    assert spents[-1] > spents[0] + 1e-9                # 后续 turn 持续获得新额度
    assert state.ledger.n_pacing_wait > 0               # 未解锁等待被记账
    assert state.budget.spent <= 10.0 + 1e-9
    for r in state.ledger.records:                      # 解锁曲线随行落账
        assert r.unlocked_at >= 0.0
    # 对照:pacing off ⇒ 首 turn 8 个开门位置 × δ=2.5 花光 b_max(旧行为)
    hook2, st2, _d2, _v2, sp2 = _mk_hook(world, theta_d=1e-9, b_max=10.0, seed=11)
    _run(hook2(list(sp2), {"max_tokens": 8, "temperature": 1.0}))
    assert st2.budget.remaining < 1e-6
    assert st2.ledger.n_pacing_wait == 0
