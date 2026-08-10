"""research 行必须携带 policy 版本证据(min/max_global_steps)。

实测事故(2026-08-07,162 v4 首次冒烟):step1 的 384 行全部正常产出,但
trainer 的 _compute_metrics 对**每一行**做
    np.array([tag["min_global_steps"] for tag in batch.tags], dtype=int)
而 research 行没设这两个键 —— verl 的 default_extra_keys 只保证键存在、
值为 None,于是 int(None) 抛 TypeError 把整步打掉。executor 行靠
`**result.gen_extra` 天然带上,research 行走的是另一条组装路径。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace_opd.train.backends import EngineResearcherChat  # noqa: E402


class _Gen:
    """按脚本返回带/不带版本证据的生成结果。"""

    def __init__(self, extras):
        self._extras = list(extras)
        self.calls = 0

    async def __call__(self, prompt_ids, sampling_params):
        class R:
            token_ids = [1, 2, 3]
            text = "ok"
            extra = self._extras[min(self.calls, len(self._extras) - 1)]
        self.calls += 1
        return R()


def _chat(extras):
    class _Delta:
        post_obs: list[int] = []

        def after_generation(self, tail):
            return []

    class _Tok:
        def encode(self, text, add_special_tokens=False):
            return [9]

    return EngineResearcherChat(
        generate=_Gen(extras), tokenizer=_Tok(), delta=_Delta(),
        temperature=1.0, max_tokens_per_turn=64, session_token_budget=4096,
        max_prompt_tokens=4096, rollout_max_model_len=1000000,
    )


def test_tracks_version_range_across_generations():
    c = _chat([])
    c._track_policy_version({"min_global_steps": 7, "max_global_steps": 7})
    c._track_policy_version({"min_global_steps": 5, "max_global_steps": 9})
    assert c.policy_version_fields() == {"min_global_steps": 5, "max_global_steps": 9}


def test_unwraps_engine_boxed_values_and_ignores_garbage():
    c = _chat([])

    class Boxed:
        data = 11

    c._track_policy_version({"min_global_steps": Boxed(), "max_global_steps": Boxed()})
    assert c.policy_version_fields() == {"min_global_steps": 11, "max_global_steps": 11}
    c._track_policy_version({"min_global_steps": "nope", "max_global_steps": None})
    assert c.policy_version_fields() == {"min_global_steps": 11, "max_global_steps": 11}


def test_no_evidence_yields_empty_so_caller_can_fail_loud():
    """缺证据必须返回空 dict 让调用方 fail loud,不能悄悄填 0 —— 填 0 会让
    版本谱系审计失真,比崩掉更难查。"""
    assert _chat([]).policy_version_fields() == {}


def test_research_row_refuses_to_emit_without_version_evidence():
    src = (ROOT / "src/trace_opd/train/selfres.py").read_text(encoding="utf-8")
    assert "policy_version_fields()" in src
    assert "no policy version evidence" in src
    # 必须在构造 ResearchRowResult 之前校验
    assert src.index("policy_version_fields()") < src.index("return ResearchRowResult(")


def test_version_failure_surfaces_the_underlying_session_error():
    """版本证据为空时,报错必须带出会话的真实死因。

    2026-08-08 事故:researcher 首调 prompt 超 max_prompt_tokens 的 fail-loud
    被 ResearchSession 的 `except Exception` 降级进 snapshot.error,这里若只报
    "no policy version evidence",现场就只剩一个不相干的症状 —— 据此误诊了两轮。
    """
    src = (ROOT / "src/trace_opd/train/selfres.py").read_text(encoding="utf-8")
    head, _, tail = src.partition("policy_version_fields()")
    raise_block = tail[: tail.index("return ResearchRowResult(")]
    assert 'getattr(session_result, "error"' in raise_block, "未从 session_result 取真实死因"
    assert "Underlying session error" in raise_block, "报错未带出底层原因"


def test_row_fields_reach_extra_fields():
    """research 行的 fields 经 **rowres.fields 进 extra_fields —— 契约不能断。"""
    src = (ROOT / "src/trace_opd/train/agent_loop.py").read_text(encoding="utf-8")
    assert "**rowres.fields," in src
