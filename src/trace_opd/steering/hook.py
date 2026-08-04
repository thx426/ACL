"""SPS 采样钩子 —— GenerateFn 包装器(规范 docs/pgs-final-spec.md §3)。

对 episode 循环完全透明:同一 GenerateFn 契约,返回的 token_ids 就是最终
执行序列(相对保尾核 b_S 精确,命题 A)。教师只在这里出现,训练损失里
没有教师——这是机制红线。

单轮内流程(draft-verify-resample 循环):
  1. DRAFT   纯 π 全速生成剩余段(logprobs=obs_topk 带出 π 的 top-K 明细)。
  2. VERIFY  教师前缀 + 已执行响应流 + draft 段,一次 prompt_logprobs=obs_topk
             打分拿逐位置 p_T top-K(prefill 一次的成本)。
  3. JUDGE   逐位置:S=topk(π)∪topk(p_T)∩known-π → D 门+ε 探针 → 预算允许的
             η → speculative 接受判定;拒绝 ⇒ S 上残差重采,回 1 从断点续。
  4. 每轮拒绝达 R_max ⇒ 该轮余下强制 η=0(行为态,如实记账)。

理论对象口径(审阅 IF-03 定案):本机制的核构造于**引擎采样分布 q**
(draft 路径返回的 logprobs),即 b_S = b_S(q, p_T);speculative 恒等的
接受分母必须与真实 proposal 分布一致,故 q 是唯一自洽的 π 侧口径——命题 A
的精确性相对 b_S(q,·) 成立。q 与训练端 π_old(FSDP bf16→fp32 重算)的
数值残差 ~1e-3/token,与普通 GRPO 默认忽略的 vLLM-FSDP 失配同类同量级;
不额外做 bare verify(那会让接受分母偏离真实 proposal,反而破坏恒等)。
残差量级 P4 冒烟实测(docs/pgs-final-spec.md 实现注记)。

实现口径(引擎数据面的两个封闭点,均为 S 的合法自选而非近似——命题 A 的
精确性相对 b_S 成立,S 的选择只影响倾斜覆盖度):
  * S 的 π 侧候选取 draft 返回的 top-obs_topk;p_T top-K 中 π-rank 超出
    obs_topk 的 token 不入 S(π 极冷 token 抬 e^{2ηc}≈e^{2.5} 倍仍可忽略,
    这与「OPD 只能调已有行为」的能力边界一致)。
  * S 内 p_T 观测窗外的 token(π-only 候选)取**单侧区间鲁棒代入**:
    ℓ̃ = min(0, clip(log floor − log π, ±c)),floor = 该位置观测窗最小
    logprob(真值的上界)。审阅 max-IF-13:纯 floor 代入在 floor > π(t)
    时符号翻转(教师强烈反对的 token 被当成偏好抬升);钳到非正后,
    floor 足够低时保留方向正确的压制(真值更负 ⇒ 实际压得比真值少,
    保守),floor 偏高时退化为中性(不倾斜)——窗外候选永不被抬升。
    替代发生率进记账(steer_n_pt_window_miss)。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import numpy as np

from .budget import TrajectoryBudget, TurnRejectCap
from .gate import SteeringGate
from .kernel import SteeringKernel, truncated_increment
from .ledger import SteeringLedger, TokenSteerRecord
from .sps import accept_or_resample

logger = logging.getLogger(__file__)

__all__ = ["SpsGenerate", "TrajectorySteerState", "position_candidates", "candidate_d"]



@dataclass
class TrajectorySteerState:
    """一条轨迹的 steering 全部可变状态(episode 级生命周期)。"""

    gate: SteeringGate
    budget: TrajectoryBudget
    reject_cap: TurnRejectCap
    ledger: SteeringLedger = field(default_factory=SteeringLedger)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())


def position_candidates(
    *,
    pi_ids: list[int],
    pi_logprobs: list[float],
    pt_ids: list[int],
    pt_logprobs: list[float],
    topk: int,
    c: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, float]]:
    """单个位置的候选集构造:两侧 top-K 明细 →
    (S 的 token_ids, log π|S, ℓ̃|S, S 内 log π 查找表)。

    输入行都是 rank 序(引擎观测窗宽度 >= topk);S 用两侧各自前 topk。
    η 无关 —— D 门决策先于 η 的确定(可测性红线)。
    """
    known_pi = {int(t): float(lp) for t, lp in zip(pi_ids, pi_logprobs)}
    pt_map = {int(t): float(lp) for t, lp in zip(pt_ids, pt_logprobs)}
    pt_floor = min(pt_map.values()) if pt_map else -float("inf")

    s_tokens: list[int] = []
    seen: set[int] = set()
    for t in pi_ids[:topk]:
        t = int(t)
        if t not in seen:
            seen.add(t)
            s_tokens.append(t)
    for t in pt_ids[:topk]:
        t = int(t)
        # p_T 候选必须有已知 π 值才能入 S(实现口径,见模块 docstring)
        if t not in seen and t in known_pi:
            seen.add(t)
            s_tokens.append(t)

    log_pi = np.array([known_pi[t] for t in s_tokens], dtype=np.float64)
    # p_T 窗外单侧区间鲁棒代入(见模块 docstring):floor 上界 + 钳非正
    in_window = np.array([t in pt_map for t in s_tokens])
    n_window_miss = int((~in_window).sum())
    log_pt = np.array(
        [pt_map.get(t, pt_floor) for t in s_tokens], dtype=np.float64
    )
    raw_ell = log_pt - log_pi
    ell = truncated_increment(log_pt, log_pi, c)
    ell = np.where(in_window, ell, np.minimum(0.0, ell))
    # 窗内的未截断 ℓ 是纯观测量(无代入污染)——c 标定与方向观察的数据面
    raw_ell_observed = raw_ell[in_window]
    return (
        np.array(s_tokens, dtype=np.int64), log_pi, ell, known_pi,
        n_window_miss, raw_ell_observed,
    )


def candidate_d(log_pi: np.ndarray, ell: np.ndarray) -> float:
    """D = Var_{a~π|S}[ℓ̃(a)](与 kernel.d_variance 同公式,η 无关)。"""
    pi = np.exp(log_pi)
    w = pi / pi.sum()
    mean = float(np.dot(w, ell))
    return float(np.dot(w, (ell - mean) ** 2))


class SpsGenerate:
    """steering 版 GenerateFn(episode 的生成通道,机制入口)。

    draft / verify 是两个独立的底层 GenerateFn(通常同一 EngineGenerate,
    同一 sticky 路由 key ⇒ 学生与教师两个前缀的 KV 同 replica 复用)。
    """

    def __init__(
        self,
        *,
        draft: Callable[[list[int], dict[str, Any]], Awaitable[Any]],
        verify: Callable[[list[int], dict[str, Any]], Awaitable[Any]],
        teacher_prefix_ids: list[int],
        student_prompt_ids: list[int],
        state: TrajectorySteerState,
        tokenizer,
        topk: int,
        c: float,
        obs_topk: int,
        stop_token_ids: set[int] | frozenset[int] = frozenset(),
    ) -> None:
        self._draft = draft
        self._verify = verify
        self._teacher_prefix = list(teacher_prefix_ids)
        # 审阅 max-IF-10:保存完整学生前缀并逐调校验——只存长度时,上游
        # prompt 构造与 episode 实际前缀有任何 token 级漂移(如 initial_obs
        # 尾随空白差异过了 strip 比较)都会静默错位 ledger 坐标与 verify
        # 组装;fail loud 是唯一正确行为。
        self._student_prompt = [int(t) for t in student_prompt_ids]
        self._student_prompt_len = len(self._student_prompt)
        self._state = state
        self._tokenizer = tokenizer
        self._topk = int(topk)
        self._c = float(c)
        # 观测窗必须与 launcher 配置的 vLLM max_logprobs 同源
        # (config.steering.obs_topk;超过 max_logprobs 的请求会被引擎拒)。
        if int(obs_topk) < int(topk):
            raise ValueError(f"obs_topk({obs_topk}) must be >= topk({topk})")
        self._obs_topk = int(obs_topk)
        # 审阅 IF-06:残差重采可能采出 EOS/停止 token;引擎的停止判定只作用
        # 于 draft 段,replacement 后必须由钩子自己重判,否则会在「已结束」的
        # 前缀后继续 draft,产出停止符之后的 token。
        self._stop_ids = frozenset(int(t) for t in stop_token_ids)

    # ------------------------------------------------------------------ call

    async def __call__(self, prompt_ids: list[int], sampling_params: dict[str, Any]):
        from ..core.episode import GenResult  # 局部 import 避免环

        params = dict(sampling_params)
        max_tokens = int(params.pop("max_tokens"))
        self._assert_measure_compatible(params)
        st = self._state
        st.reject_cap.new_turn()

        stream_len = len(prompt_ids) - self._student_prompt_len
        if stream_len < 0 or list(prompt_ids[: self._student_prompt_len]) != self._student_prompt:
            raise RuntimeError(
                "episode prompt does not start with the pre-built student prompt — "
                "prefix drift (e.g. initial_obs whitespace) would silently misalign "
                "ledger positions and verify assembly"
            )

        executed: list[int] = []
        executed_logpi: list[float] = []
        gen_extra: dict[str, Any] = {}
        finished = False

        while len(executed) < max_tokens and not finished:
            remaining = max_tokens - len(executed)
            draft_out = await self._draft(
                prompt_ids + executed,
                {**params, "max_tokens": remaining, "logprobs": self._obs_topk},
            )
            for key in ("min_global_steps", "max_global_steps", "stop_reason"):
                if key in draft_out.extra:
                    gen_extra.setdefault(key, draft_out.extra[key])
            if draft_out.extra.get("stop_reason") == "aborted":
                if not executed:
                    return draft_out  # 纯透传:episode 按基建故障处理
                gen_extra["stop_reason"] = "aborted"
                break
            gen_ids = list(draft_out.token_ids)
            if not gen_ids:
                break
            # 自然终止判定:引擎在预算内停下(eos)⇒ 本轮生成结束,除非
            # 停点 token 被拒绝重采(那会改变续写条件,回去继续 draft)。
            draft_finished = len(gen_ids) < remaining

            pi_lp = draft_out.extra.get("log_probs")
            pi_topk_ids = draft_out.extra.get("response_topk_ids")
            pi_topk_lps = draft_out.extra.get("response_topk_logprobs")
            if pi_lp is None or pi_topk_ids is None:
                raise RuntimeError(
                    "steering draft returned no logprobs detail — engine passthrough "
                    "missing (vllm_async_server logprobs=K path)"
                )

            steerable = st.budget.remaining > 0 and not st.reject_cap.cap_active
            if not steerable:
                st.ledger.n_bypassed += len(gen_ids)
                executed += gen_ids
                executed_logpi += [float(x) for x in pi_lp]
                finished = draft_finished
                continue

            pt_rows = await self._verify_rows(
                prompt_ids, executed, gen_ids, n_positions=len(gen_ids)
            )

            rejected = False
            for g, y in enumerate(gen_ids):
                y = int(y)
                log_pi_y = float(pi_lp[g])
                position = stream_len + len(executed)
                if st.budget.remaining <= 0 or st.reject_cap.cap_active:
                    st.ledger.n_bypassed += 1
                    executed.append(y)
                    executed_logpi.append(log_pi_y)
                    continue
                s_tokens, s_log_pi, s_ell, known_pi, n_miss, raw_ell = position_candidates(
                    pi_ids=pi_topk_ids[g],
                    pi_logprobs=pi_topk_lps[g],
                    pt_ids=pt_rows[0][g],
                    pt_logprobs=pt_rows[1][g],
                    topk=self._topk,
                    c=self._c,
                )
                st.ledger.n_pt_window_miss += n_miss
                st.ledger.observe_ell(raw_ell)
                d_value = candidate_d(s_log_pi, s_ell)
                st.ledger.observe_d(d_value)
                decision = st.gate.decide(d_value, st.rng)
                st.ledger.n_base_open += int(decision.base_open)
                st.ledger.n_probe_flipped += int(decision.probe_flipped)
                eta = st.budget.admissible_eta(decision.eta)
                if eta <= 0.0:
                    if decision.base_open or decision.probe_flipped:
                        # 门开了但预算不允许(或探针翻关):记账可见,零扣费
                        if decision.eta > 0.0:
                            st.ledger.n_budget_blocked += 1
                        st.ledger.append(TokenSteerRecord(
                            position=position, log_pi=log_pi_y, log_b=log_pi_y,
                            eta=0.0, delta=0.0, d_value=d_value,
                            base_open=decision.base_open,
                            probe_flipped=decision.probe_flipped,
                            propensity=decision.propensity,
                            rejected=False, cap_active=st.reject_cap.cap_active,
                            budget_after=st.budget.remaining,
                            draft_token=y, final_token=y,
                        ))
                    executed.append(y)
                    executed_logpi.append(log_pi_y)
                    continue
                kernel = SteeringKernel(
                    token_ids=s_tokens, log_pi=s_log_pi,
                    ell_tilde=s_ell, eta=eta, c=self._c,
                )
                outcome = accept_or_resample(
                    kernel, y, log_pi_y, st.rng, log_pi_lookup=known_pi
                )
                delta = st.budget.charge(eta)
                st.ledger.budget_spent = st.budget.spent
                # 审阅 IF-07:先推进 cap 状态再落账——第 R_max 次拒绝行必须
                # 记 cap_active=True(旧顺序恒 False,cap 尾段在账上不可见)。
                if outcome.rejected:
                    st.reject_cap.record_reject()
                    if st.reject_cap.cap_active:
                        st.ledger.turns_capped += 1
                st.ledger.append(TokenSteerRecord(
                    position=position,
                    log_pi=outcome.log_pi,
                    log_b=outcome.log_b,
                    eta=eta, delta=delta, d_value=d_value,
                    base_open=decision.base_open,
                    probe_flipped=decision.probe_flipped,
                    propensity=decision.propensity,
                    rejected=outcome.rejected,
                    cap_active=st.reject_cap.cap_active,
                    budget_after=st.budget.remaining,
                    draft_token=y, final_token=int(outcome.token_id),
                ))
                executed.append(int(outcome.token_id))
                executed_logpi.append(float(outcome.log_pi))
                if outcome.rejected:
                    if int(outcome.token_id) in self._stop_ids:
                        finished = True  # IF-06:重采出停止 token,本轮生成结束
                    rejected = True
                    break

            if rejected:
                if finished:
                    break
                continue  # 从断点(含重采 token)续 draft(原 EOS 被拒重采成非停止符同样续)
            finished = draft_finished

        token_ids = executed[:max_tokens]
        return GenResult(
            token_ids=token_ids,
            text=self._tokenizer.decode(token_ids, skip_special_tokens=True),
            extra=gen_extra,
        )

    # -------------------------------------------------------------- internals

    async def _verify_rows(
        self,
        prompt_ids: list[int],
        executed: list[int],
        gen_ids: list[int],
        *,
        n_positions: int,
    ) -> tuple[list[list[int]], list[list[float]]]:
        """教师前缀下对 draft 段逐位置打分,返回 (ids_rows, logprob_rows)。

        行语义(extract_prompt_logprobs):数组第 r 行是预测位置 r+1 的分布,
        末行 dummy。draft 段第 g 个 token 的全局位置 = T+M+E+g,其分布行
        下标 = T+M+E+g-1(T=教师前缀长,M=已有响应流长,E=本轮已执行长)。
        """
        response_stream = prompt_ids[self._student_prompt_len:]
        teacher_seq = self._teacher_prefix + list(response_stream) + list(executed) + list(gen_ids)
        out = await self._verify(
            teacher_seq,
            {
                # 打分请求:确定性口径(温度 1 全分布),1 个丢弃 token
                "max_tokens": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "prompt_logprobs": self._obs_topk,
            },
        )
        if out.extra.get("stop_reason") == "aborted":
            raise RuntimeError("steering verify aborted by the engine")
        rows_ids = out.extra.get("prompt_ids")
        rows_lps = out.extra.get("prompt_logprobs")
        if rows_ids is None or rows_lps is None:
            raise RuntimeError(
                "steering verify returned no prompt_logprobs — engine passthrough missing"
            )
        base = len(teacher_seq) - len(gen_ids) - 1
        ids = [rows_ids[base + g] for g in range(n_positions)]
        lps = [rows_lps[base + g] for g in range(n_positions)]
        return ids, lps

    def _assert_measure_compatible(self, params: dict[str, Any]) -> None:
        """draft 与 verify 必须在同一测度下(温度 1、无截断),否则 ℓ̃ 的
        两侧口径失配 —— fail loud,绝不静默倾斜错误的量。"""
        temp = params.get("temperature", 1.0)
        top_p = params.get("top_p", 1.0)
        top_k = params.get("top_k", -1)
        if not (abs(float(temp) - 1.0) < 1e-9 and float(top_p) >= 1.0 and int(top_k) in (-1, 0)):
            raise RuntimeError(
                f"steering requires temperature=1/top_p=1/top_k=-1 sampling "
                f"(got T={temp}, top_p={top_p}, top_k={top_k}) — the kernel's "
                "pi side must be the true sampling distribution"
            )
