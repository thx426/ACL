"""v4 前缀续写评估(research/continuation.py)与三段行布局(arms.role_of)。

契约红线:分位按 policy token(mask=1)累计比例、半轮承接良定义、轮末
延伸规则、重放逐步校验(漂移作废)、预算继承(教师不加时)、reward
有效性规则(finalize/降级/剔除)。全部用 fakes 基建,无引擎依赖。
"""
from __future__ import annotations

import asyncio

import pytest

from fakes import FakeTokenizer, ScriptedAdapter, ScriptedGenerate, reply
from trace_opd.core import protocol
from trace_opd.core.template_delta import TemplateDelta
from trace_opd.research.continuation import (
    ContinuationOutcome,
    aggregate_reward,
    build_prefix_set,
    continue_from_prefix,
)
from trace_opd.steering.arms import ROLE_BARE, ROLE_RESEARCH, ROLE_STEERED, role_of

TOK = FakeTokenizer()


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


# ----------------------------------------------------------- role_of 三段布局

def test_role_of_three_segments():
    # 段序沿 v3.1 既有约定(is_steered_arm 前缀切分):steered 前段、
    # bare 后段、research 尾段。臂身份真值恒以 steer_arm/role 字段为准,
    # 段序是确定性标签细节(副本先验 iid)。
    roles = [role_of(i, 16, 0.5, research_n=8) for i in range(24)]
    assert roles[:8] == [ROLE_STEERED] * 8
    assert roles[8:16] == [ROLE_BARE] * 8
    assert roles[16:24] == [ROLE_RESEARCH] * 8
    with pytest.raises(RuntimeError, match="out of range"):
        role_of(24, 16, 0.5, research_n=8)
    with pytest.raises(RuntimeError, match="session_id"):
        role_of(None, 16, 0.5, research_n=8)


def test_role_of_degrades_to_v31_when_research_n_zero():
    assert [role_of(i, 8, 0.5, 0) for i in range(8)] == (
        [ROLE_STEERED] * 4 + [ROLE_BARE] * 4
    )   # v3.1 原语义逐字节保持
    assert role_of(None, 8, 1.0, 0) == ROLE_STEERED   # 历史全臂容忍 None


# ------------------------------------------------------------- 前缀集纯函数

def _row(turn_texts: list[str], obs_texts: list[str], won: bool = False,
         acted: list[bool] | None = None):
    """构造一条 bare 行:轮生成段(mask=1)+ obs 结构段(mask=0)交替。"""
    ids: list[int] = []
    mask: list[int] = []
    spans: list[list[int]] = []
    steps: list[dict] = []
    acted = acted if acted is not None else [True] * len(turn_texts)
    for i, text in enumerate(turn_texts):
        t_ids = TOK.encode(text)
        spans.append([len(ids), len(ids) + len(t_ids)])
        ids += t_ids
        mask += [1] * len(t_ids)
        if i < len(obs_texts):
            o_ids = TOK.encode(f"<obs>{obs_texts[i]}</obs>")
            ids += o_ids
            mask += [0] * len(o_ids)
        steps.append({
            "action_norm": f"act{i}", "observation": obs_texts[i] if i < len(obs_texts) else "",
            "acted": acted[i],
        })
    return {"won": won, "response_ids": ids, "response_mask": mask,
            "turn_spans": spans, "steps": steps}


def test_prefix_cut_is_policy_token_quantile_mid_turn():
    row = _row(["A" * 40, "B" * 40, "C" * 40], ["o1", "o2", "o3"])
    specs = build_prefix_set([row], [0.05, 0.50], TOK)
    assert len(specs) == 2
    s5, s50 = specs
    # 5%×120=6 个 policy token ⇒ 切在轮 0 第 6 个 token,半轮
    assert s5.turns_used == 0 and s5.replay_actions == ()
    assert s5.half_turn_text == "A" * 6
    assert list(s5.prefix_ids) == TOK.encode("A" * 6)
    # 50%×120=60 ⇒ 轮 1 内(40+20),重放轮 0,半轮文本为轮 1 前 20 字符
    assert s50.turns_used == 1 and s50.replay_actions == ("act0",)
    assert s50.replay_observations == ("o1",)
    assert s50.half_turn_text == "B" * 20
    assert s50.env_steps_used == 1


def test_prefix_turn_end_extends_to_next_turn_boundary():
    # 25%×80=20 恰为轮 0 末 token ⇒ 前缀延伸含 obs 结构,轮 0 进重放,整轮起写
    row = _row(["A" * 20, "B" * 60], ["o1", "o2"])
    (spec,) = build_prefix_set([row], [0.25], TOK)
    assert spec.turns_used == 1 and spec.replay_actions == ("act0",)
    assert spec.half_turn_text == ""
    assert len(spec.prefix_ids) == row["turn_spans"][1][0]   # 到轮 1 起点


def test_prefix_last_turn_end_falls_back_to_half_turn():
    row = _row(["A" * 10], ["o1"])
    (spec,) = build_prefix_set([row], [1.0 - 1e-9], TOK) or [None]
    assert spec is not None
    assert spec.half_turn_text == "A" * 9          # 回退一个 token 强制半轮
    assert spec.turns_used == 0


def test_prefix_skips_unacted_turns_in_replay():
    row = _row(["A" * 30, "B" * 30, "C" * 30], ["fmt", "o2", "o3"],
               acted=[False, True, True])
    (spec,) = build_prefix_set([row], [0.75], TOK)   # 68 ⇒ 轮 2 内
    assert spec.turns_used == 2
    assert spec.replay_actions == ("act1",)          # 轮 0 未 acted,不重放
    assert spec.env_steps_used == 1


def test_prefix_set_prefers_failed_and_spreads_across_trajectories():
    rows = [_row(["W" * 40], ["o"], won=True),
            _row(["X" * 40, "X" * 40], ["o", "o"], won=False),
            _row(["Y" * 40, "Y" * 40], ["o", "o"], won=False)]
    specs = build_prefix_set(rows, [0.05, 0.15, 0.30], TOK)
    assert [s.traj_index for s in specs[:2]] == [1, 2]   # 失败轨迹优先且分散
    assert specs[2].traj_index in (0, 1, 2)


# ------------------------------------------------- 重放 + 半轮续写(编排)

def _adapter(win_on="finish", responses=None):
    return ScriptedAdapter(
        initial_obs="Start room.",
        responses=responses or {"act0": "o1", "act1": "o2", "look": "A room."},
        win_on=win_on,
    )


def _mk_delta():
    return TemplateDelta(TOK)


def test_continuation_replays_then_wins_from_half_turn():
    # 前缀:轮 0 完整(act0 已重放)+ 轮 1 半轮;教师续写补完轮 1 拿到胜局
    row = _row([reply("r0", "act0"), "<reasoning>half"], ["o1"])
    (spec,) = build_prefix_set([row], [0.99], TOK)
    assert spec.half_turn_text.startswith("<reasoning>half"[:5]) or spec.turns_used == 1
    adapter = _adapter()
    gen = ScriptedGenerate(TOK, [" done</reasoning>\n<action>finish</action>"])
    out = _run(continue_from_prefix(
        adapter=adapter, teacher_prefix_ids=TOK.encode("TEACHER:"),
        spec=spec, generate=gen, tokenizer=TOK, delta=_mk_delta(),
        response_budget=4000, sampling_params={"temperature": 1.0},
    ))
    assert out.error is None and out.won
    env = adapter.envs[-1]
    assert env.commands[0] == "act0"               # 重放先行
    assert env.commands[-1] == "finish"            # 半轮拼接解析出的 action
    assert env.closed


def test_continuation_detects_replay_drift():
    row = _row([reply("r0", "act0"), "<reasoning>half"], ["o1"])
    (spec,) = build_prefix_set([row], [0.99], TOK)
    adapter = _adapter(responses={"act0": "SOMETHING ELSE"})   # 环境漂移
    gen = ScriptedGenerate(TOK, [reply("x", "finish")])
    out = _run(continue_from_prefix(
        adapter=adapter, teacher_prefix_ids=[1], spec=spec, generate=gen,
        tokenizer=TOK, delta=_mk_delta(), response_budget=4000,
        sampling_params={},
    ))
    assert out.error is not None and "replay_drift" in out.error
    assert not out.won and adapter.envs[-1].closed


def test_continuation_inherits_env_step_budget(monkeypatch):
    # 预算继承:前缀已用 1 步,总步数上限 2 ⇒ 教师只剩 1 步,用掉即止
    monkeypatch.setattr(protocol, "MAX_ENV_STEPS", 2)
    row = _row([reply("r0", "act0"), "<reasoning>half"], ["o1"])
    (spec,) = build_prefix_set([row], [0.99], TOK)
    adapter = _adapter(win_on="never")
    gen = ScriptedGenerate(TOK, [reply("a", "look"), reply("b", "look")])
    out = _run(continue_from_prefix(
        adapter=adapter, teacher_prefix_ids=[1], spec=spec, generate=gen,
        tokenizer=TOK, delta=_mk_delta(), response_budget=8000,
        sampling_params={},
    ))
    assert out.error is None and out.env_steps == 1     # 只允许 1 新步
    assert len(adapter.envs[-1].commands) == 2          # 重放 1 + 续写 1


def test_continuation_teacher_prefix_precedes_stream():
    row = _row([reply("r0", "act0"), "<reasoning>half"], ["o1"])
    (spec,) = build_prefix_set([row], [0.99], TOK)
    teacher = TOK.encode("DOC-PREFIX|")
    gen = ScriptedGenerate(TOK, [reply("x", "finish")])
    _run(continue_from_prefix(
        adapter=_adapter(), teacher_prefix_ids=teacher, spec=spec,
        generate=gen, tokenizer=TOK, delta=_mk_delta(),
        response_budget=4000, sampling_params={},
    ))
    assert gen.calls[0]["prompt_len"] == len(teacher) + len(spec.prefix_ids)


# --------------------------------------------------------------- reward 聚合

def test_aggregate_reward_rules():
    ok_w = ContinuationOutcome(won=True)
    ok_l = ContinuationOutcome(won=False)
    bad = ContinuationOutcome(error="replay_drift")
    r, valid, deg = aggregate_reward([ok_w, ok_w, ok_l, ok_l], 4)
    assert (r, valid, deg) == (0.5, 4, False)
    r, valid, deg = aggregate_reward([ok_w, bad, bad, bad], 4)
    assert valid == 1 and deg                       # 有效 < 名义半数 ⇒ 降级
    assert r == 1.0
    r, valid, deg = aggregate_reward([bad, bad], 4)
    assert (r, valid, deg) == (0.0, 0, True)        # 全作废:r=0 且降级
