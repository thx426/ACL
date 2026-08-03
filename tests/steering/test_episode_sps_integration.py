"""episode 循环 × SpsGenerate 集成 —— 多轮 glue/obs 混入响应流后,
verify 序列组装与 ledger 坐标仍然正确;门关时端到端透明。

hook 层已证输出分布 ≡ b_S(统计等价);这里补的是 episode 组合层:
SPS 包装器在真实多轮构流(prompt+stream 增长、mask=0 结构段插入)下的
坐标不变量——真机前最后一层可本地验证的正确性。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fakes import FakeTokenizer, ScriptedAdapter, reply  # noqa: E402

from trace_opd.core.episode import GenResult, run_episode  # noqa: E402
from trace_opd.core.template_delta import TemplateDelta, render_ids  # noqa: E402
from trace_opd.core.types import inject_guidance  # noqa: E402
from trace_opd.steering import SteeringGate, TrajectoryBudget, TurnRejectCap  # noqa: E402
from trace_opd.steering.hook import SpsGenerate, TrajectorySteerState  # noqa: E402

TOK = FakeTokenizer()
OBS_TOPK = 8


class _ScriptedDraft:
    """脚本文本 draft(engine 风格:token_ids=采样序列)+ SPS 明细通道。

    π 明细取平坦小分布:每位置 top-K 行 = 定值词表,log_pi(y)=log 0.5,
    与 verify 的 p_T 行完全一致 ⇒ ℓ̃≡0 ⇒ D≡0(门关,透明路径)。
    """

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls: list[list[int]] = []

    async def __call__(self, prompt_ids, params):
        self.calls.append(list(prompt_ids))
        if not self._texts:
            return GenResult(token_ids=[], text="")
        ids = TOK.encode(self._texts.pop(0))[: int(params["max_tokens"])]
        row_ids = list(range(40, 40 + OBS_TOPK))
        row_lps = [float(np.log(1.0 / OBS_TOPK))] * OBS_TOPK
        return GenResult(
            token_ids=ids,
            text=TOK.decode(ids),
            extra={
                "log_probs": [float(np.log(0.5))] * len(ids),
                "response_topk_ids": [list(row_ids)] * len(ids),
                "response_topk_logprobs": [list(row_lps)] * len(ids),
            },
        )


class _EchoVerify:
    """p_T 行 ≡ π 行(ℓ̃≡0);记录并断言收到的教师序列。"""

    def __init__(self, teacher_prefix: list[int]) -> None:
        self.teacher_prefix = list(teacher_prefix)
        self.calls: list[list[int]] = []

    async def __call__(self, seq, params):
        self.calls.append(list(seq))
        assert seq[: len(self.teacher_prefix)] == self.teacher_prefix
        row_ids = list(range(40, 40 + OBS_TOPK))
        row_lps = [float(np.log(1.0 / OBS_TOPK))] * OBS_TOPK
        n = len(seq)
        return GenResult(
            token_ids=[1],
            text="",
            extra={
                "prompt_ids": [list(row_ids)] * (n - 1) + [[0] * OBS_TOPK],
                "prompt_logprobs": [list(row_lps)] * (n - 1) + [[0.0] * OBS_TOPK],
            },
        )


def _adapter():
    return ScriptedAdapter(
        initial_obs="You are in a room. A box is here.",
        responses={"open box": "The box is open.", "take key": "You take the key."},
        win_on="take key",
    )


def _episode_with_sps(texts):
    adapter = _adapter()
    delta = TemplateDelta(TOK)
    student_prompt_ids = render_ids(
        TOK,
        [
            {"role": "system", "content": inject_guidance(adapter.executor_scaffold(), "")},
            {"role": "user", "content": adapter.initial_obs},
        ],
        True,
    )
    doc_ids = TOK.encode("privileged doc: key is inside the box")
    teacher_prefix = doc_ids + student_prompt_ids  # 组装语义只要求"教师自己的前缀"
    draft = _ScriptedDraft(texts)
    verify = _EchoVerify(teacher_prefix)
    state = TrajectorySteerState(
        gate=SteeringGate(theta_d=0.5, eta_max=0.5, epsilon=0.0),
        budget=TrajectoryBudget(b_max=6.0, c=2.5),
        reject_cap=TurnRejectCap(r_max=4),
        rng=np.random.default_rng(0),
    )
    sps = SpsGenerate(
        draft=draft, verify=verify,
        teacher_prefix_ids=teacher_prefix,
        student_prompt_len=len(student_prompt_ids),
        state=state, tokenizer=TOK, topk=4, c=2.5, obs_topk=OBS_TOPK,
    )
    result = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        run_episode(
            adapter=adapter, generate=sps, tokenizer=TOK, delta=delta,
            guidance="", episode_id="sps-it", response_budget=4096,
            sampling_params={"temperature": 1.0},
        )
    )
    return result, state, draft, verify, len(student_prompt_ids)


def test_multi_turn_verify_sequence_tracks_the_stream():
    texts = [reply("look around", "open box"), reply("grab it", "take key")]
    result, state, draft, verify, sp_len = _episode_with_sps(texts)
    assert result.won and result.assistant_turns == 2
    assert len(verify.calls) == 2
    # 第二轮 verify 的序列 = 教师前缀 + 完整响应流(含第一轮生成 + glue/obs)+ 第二轮 draft
    tp = len(verify.teacher_prefix)
    second = verify.calls[1]
    stream_at_turn2 = draft.calls[1][sp_len:]      # episode 传给 draft 的 prompt 尾部
    assert second[tp: tp + len(stream_at_turn2)] == stream_at_turn2
    assert len(second) > len(verify.calls[0])       # 序列随流增长


def test_gate_closed_is_end_to_end_transparent():
    texts = [reply("look around", "open box"), reply("grab it", "take key")]
    result, state, *_ = _episode_with_sps(texts)
    # ℓ̃≡0 ⇒ D≡0 < θ_D ⇒ 门恒关:零 steer、零扣费,但 D 观测照记
    assert state.ledger.records == []
    assert state.budget.spent == 0.0
    assert state.ledger.d_count > 0
    assert state.ledger.d_hist[0] == state.ledger.d_count   # D=0 全落首桶
    # episode 输出与脚本文本一致(透明)
    assert result.trajectory.steps[0].action_norm == "open box"
    assert result.trajectory.steps[1].action_norm == "take key"


def test_response_tokens_match_episode_stream_coordinates():
    texts = [reply("look around", "open box"), reply("grab it", "take key")]
    result, state, draft, verify, sp_len = _episode_with_sps(texts)
    # 第二轮 draft 收到的 prompt = 学生 prompt + episode 已构响应流;
    # 其尾部必须与 result.tokens 的对应前缀区间逐 token 一致(坐标同框)
    stream_at_turn2 = draft.calls[1][sp_len:]
    assert result.tokens.response_ids[: len(stream_at_turn2)] == stream_at_turn2
