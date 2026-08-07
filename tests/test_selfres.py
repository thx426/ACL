"""v4 编排(train/selfres.py):TaskCoord 协调语义 + research 行全流程。

红线:前缀池在第 bare_expected 条登记时 resolve;argmax 在第
research_expected 个登记时 resolve(降级排除、r*=0 准入门、tie 取短 doc、
重复登记 fail loud);research 行的 λ_r 预乘/warmup 置零/超长 mask 清零/
overflow r=0/降级 doc 不参 argmax;abort 向上 raise 且不登记。
"""
from __future__ import annotations

import asyncio

import pytest

from tests.fakes import FakeTokenizer, ScriptedAdapter, ScriptedGenerate, reply
from trace_opd.core.template_delta import TemplateDelta
from trace_opd.train.selfres import CoordRegistry, TaskCoord, run_research_row

TOK = FakeTokenizer()


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def _bare_row(turn_texts, obs_texts, won=False):
    ids, mask, spans, steps = [], [], [], []
    for i, text in enumerate(turn_texts):
        t_ids = TOK.encode(text)
        spans.append([len(ids), len(ids) + len(t_ids)])
        ids += t_ids
        mask += [1] * len(t_ids)
        o_ids = TOK.encode(f"<obs>{obs_texts[i]}</obs>")
        ids += o_ids
        mask += [0] * len(o_ids)
        steps.append({"action_norm": f"act{i}", "observation": obs_texts[i], "acted": True})
    return {"won": won, "response_ids": ids, "response_mask": mask,
            "turn_spans": spans, "steps": steps}


# ------------------------------------------------------------------ TaskCoord

def test_coord_prefix_resolves_on_last_bare_and_argmax_rules():
    async def main():
        coord = TaskCoord(bare_expected=2, research_expected=3,
                          quantiles=[0.5], tokenizer=TOK)
        assert not coord.prefix_future.done()
        coord.register_bare(_bare_row(["A" * 20], ["o"]))
        assert not coord.prefix_future.done()
        coord.register_bare(_bare_row(["B" * 20], ["o"], won=True))
        specs = await coord.prefix_future
        assert specs and specs[0].traj_index == 0          # 失败轨迹优先
        # argmax:降级行排除;tie 取短 doc
        coord.register_research(0, "LONG DOC BODY", 0.5, False)
        coord.register_research(1, "SHORT", 0.5, False)
        assert not coord.argmax_future.done()
        coord.register_research(2, "ignored-degraded", 0.9, True)
        doc, r = await coord.argmax_future
        assert (doc, r) == ("SHORT", 0.5)                  # 0.9 降级被排除
        with pytest.raises(RuntimeError, match="twice"):
            coord.register_research(1, "x", 0.1, False)
    _run(main())


def test_coord_admission_gate_when_best_is_zero_or_all_degraded():
    async def main():
        c1 = TaskCoord(bare_expected=1, research_expected=2,
                       quantiles=[0.5], tokenizer=TOK)
        c1.register_research(0, "d0", 0.0, False)
        c1.register_research(1, "d1", 0.0, False)
        assert await c1.argmax_future == ("", 0.0)         # r*=0 ⇒ 无教师
        c2 = TaskCoord(bare_expected=1, research_expected=1,
                       quantiles=[0.5], tokenizer=TOK)
        c2.register_research(0, "", 0.0, True)
        assert await c2.argmax_future == ("", 0.0)         # 全降级 ⇒ 无教师
    _run(main())


def test_coord_registry_drops_previous_step_bucket():
    async def main():
        reg = CoordRegistry()
        f = lambda: TaskCoord(bare_expected=1, research_expected=1,
                              quantiles=[0.5], tokenizer=TOK)
        a = await reg.get(1, "t/a", f)
        assert (await reg.get(1, "t/a", f)) is a           # 同 step 复用
        b = await reg.get(2, "t/a", f)
        assert b is not a                                  # 新 step 换桶(回收)
    _run(main())


# ------------------------------------------------------------ research 行全流程

def _mk_session(doc: str | None, error: str | None = None, aborted: bool = False):
    class _Chat:
        def __init__(self):
            self.aborted = aborted
        def training_row(self):
            p = TOK.encode("PROMPT")
            s = TOK.encode("SESSION-GEN")
            return p, s, [1] * len(s)
        def policy_version_fields(self):
            # research 行必须带 policy 版本证据(见 test_research_policy_version_tag.py):
            # trainer 的 _compute_metrics 对每一行都 int() 转换,缺则整步崩。
            return {"min_global_steps": 3, "max_global_steps": 3}
    class _SR:
        def __init__(self):
            self.doc = doc
            self.turns_used = 3
            self.experiments_used = 2
            self.episodes_consumed = 8
            self.error = error
    async def run_session():
        return _SR(), _Chat()
    return run_session


def _row_kwargs(coord, *, doc="D", session_id=0, step=10, lam=0.1, warmup=0,
                max_tokens=10_000, teacher=lambda d: TOK.encode("T:" + d),
                gen_texts=None, samples=2):
    adapter = ScriptedAdapter(initial_obs="Start room.",
                              responses={"act0": "o"}, win_on="finish")
    texts = gen_texts if gen_texts is not None else [reply("x", "finish")] * 8
    gen = ScriptedGenerate(TOK, list(texts))
    return dict(
        session_id=session_id, global_step=step, coord=coord,
        run_session=_mk_session(doc),
        teacher_ids_for_doc=teacher,
        make_generate=lambda tag: gen,
        adapter=adapter, tokenizer=TOK, delta=TemplateDelta(TOK),
        response_budget=4000, continuation_samples=samples,
        continuation_temperature=1.0, research_rl_weight=lam,
        research_warmup_steps=warmup, research_max_sample_tokens=max_tokens,
    )


def test_research_row_full_path_scales_reward_and_registers():
    async def main():
        coord = TaskCoord(bare_expected=1, research_expected=1,
                          quantiles=[0.5], tokenizer=TOK)
        coord.register_bare(_bare_row([reply("r", "act0") + "tail"], ["o"]))
        res = await run_research_row(**_row_kwargs(coord, lam=0.25))
        assert res.fields["research_r_raw"] == 1.0          # 全部续写胜
        assert res.reward_score == pytest.approx(0.25)      # λ_r 预乘
        assert res.fields["research_valid_continuations"] == 2
        doc, r = await coord.argmax_future                  # 已登记并 resolve
        assert doc == "D" and r == 1.0
        assert sum(res.response_mask) == len(TOK.encode("SESSION-GEN"))
    _run(main())


def test_research_row_warmup_zeroes_scaled_reward_only():
    async def main():
        coord = TaskCoord(bare_expected=1, research_expected=1,
                          quantiles=[0.5], tokenizer=TOK)
        coord.register_bare(_bare_row([reply("r", "act0") + "tail"], ["o"]))
        res = await run_research_row(**_row_kwargs(coord, step=3, warmup=5))
        assert res.fields["research_r_raw"] == 1.0
        assert res.reward_score == 0.0                      # warmup:A 必为 0
        assert res.fields["research_warmup_active"] == 1
    _run(main())


def test_research_row_no_finalize_and_overflow_give_zero_and_register():
    async def main():
        coord = TaskCoord(bare_expected=1, research_expected=2,
                          quantiles=[0.5], tokenizer=TOK)
        coord.register_bare(_bare_row([reply("r", "act0") + "tail"], ["o"]))
        r1 = await run_research_row(**_row_kwargs(coord, doc=None, session_id=0))
        assert r1.fields["research_no_finalize"] == 1 and r1.reward_score == 0.0
        r2 = await run_research_row(**_row_kwargs(
            coord, session_id=1, teacher=lambda d: None))   # doc 超上下文
        assert r2.fields["research_doc_overflow"] == 1 and r2.reward_score == 0.0
        assert await coord.argmax_future == ("", 0.0)       # 两行皆 0 ⇒ 无教师
    _run(main())


def test_research_row_overlong_masks_out_but_keeps_reward():
    async def main():
        coord = TaskCoord(bare_expected=1, research_expected=1,
                          quantiles=[0.5], tokenizer=TOK)
        coord.register_bare(_bare_row([reply("r", "act0") + "tail"], ["o"]))
        res = await run_research_row(**_row_kwargs(coord, max_tokens=5))
        assert res.fields["research_overlong"] == 1
        assert set(res.response_mask) == {0}                # 零梯度
        assert res.reward_score > 0                         # r 留组内 baseline
    _run(main())


def test_research_row_abort_raises_without_registering():
    async def main():
        coord = TaskCoord(bare_expected=1, research_expected=1,
                          quantiles=[0.5], tokenizer=TOK)
        kw = _row_kwargs(coord)
        class _AbChat:
            aborted = True
            def training_row(self):
                return [], [], []
        async def run_session():
            class _SR:
                doc = "D"; turns_used = 1; experiments_used = 0
                episodes_consumed = 0; error = None
            return _SR(), _AbChat()
        kw["run_session"] = run_session
        with pytest.raises(RuntimeError, match="aborted"):
            await run_research_row(**kw)
        assert not coord.argmax_future.done()               # 未登记(整 prompt 弃)
    _run(main())
