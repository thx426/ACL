"""Review-driven episode edge cases (P1-2 abort, P2-1 drift, P2-3 caps)."""
import pytest

from fakes import FakeTokenizer, ScriptedAdapter, ScriptedGenerate, reply

from trace_opd.core import protocol
from trace_opd.core.episode import GenResult, run_episode
from trace_opd.core.template_delta import TemplateDelta
from trace_opd.core.types import OUTCOME_ENV_ERROR

pytestmark = pytest.mark.asyncio

TOK = FakeTokenizer()
DELTA = TemplateDelta(TOK)


async def run(adapter, generate, *, budget=100000):
    return await run_episode(
        adapter=adapter,
        generate=generate,
        tokenizer=TOK,
        delta=DELTA,
        episode_id="ep",
        response_budget=budget,
        sampling_params={"temperature": 1.0},
    )


async def test_initial_obs_drift_is_env_error():
    adapter = ScriptedAdapter(env_initial_obs="A DIFFERENT scene entirely.")
    result = await run(adapter, ScriptedGenerate(TOK, [reply("x", "look")]))
    assert result.trajectory.outcome == OUTCOME_ENV_ERROR
    assert "mismatch" in result.trajectory.error
    assert result.tokens.response_ids == []
    assert adapter.envs[0].closed  # the drifting env is still closed


async def test_engine_abort_is_env_error():
    class AbortingGenerate:
        async def __call__(self, prompt_ids, sampling_params):
            return GenResult(token_ids=[], text="", extra={"stop_reason": "aborted"})

    adapter = ScriptedAdapter()
    result = await run(adapter, AbortingGenerate())
    assert result.trajectory.outcome == OUTCOME_ENV_ERROR
    assert "aborted" in result.trajectory.error
    assert result.truncated


async def test_assistant_turn_cap_marks_truncated():
    # Alternate parse-able turns with format failures so the turn cap fires
    # long before the env-step budget: 2 turns per env step.
    texts = []
    for _ in range(protocol.MAX_ASSISTANT_TURNS):
        texts.append("no tags at all")
        texts.append(reply("ok", "look"))
    adapter = ScriptedAdapter()
    result = await run(adapter, ScriptedGenerate(TOK, texts))
    assert result.assistant_turns == protocol.MAX_ASSISTANT_TURNS
    assert result.truncated  # guard-cap exit is a truncation, not a clean fail
    env_steps = sum(1 for s in result.trajectory.steps if s.acted)
    assert env_steps < protocol.MAX_ENV_STEPS


async def test_structure_reservation_leaves_room_for_a_turn():
    """After appending glue+obs+post_obs there must always be room for the
    next generation (>=96 reserved) — no dangling mask=0 generation prefix."""
    adapter = ScriptedAdapter()
    result = await run(
        adapter, ScriptedGenerate(TOK, [reply("look", "look")] * 50), budget=400
    )
    assert result.truncated
    mask = result.tokens.response_mask
    # the stream must not END on a structure (mask=0) segment
    assert mask and mask[-1] == 1


class _IdsGenerate:
    """GenerateFn double,engine-backend 风格:token_ids 逐轮照给(可含
    轮末停止符),text 由 fake tokenizer decode——轮末停止形态哨兵的试桩。"""

    def __init__(self, per_turn_ids: list[list[int]]) -> None:
        self._turns = [list(t) for t in per_turn_ids]

    async def __call__(self, prompt_ids, sampling_params):
        if not self._turns:
            return GenResult(token_ids=[], text="")
        ids = self._turns.pop(0)
        return GenResult(token_ids=ids, text=TOK.decode(ids))


async def test_turn_stop_shape_sentinel_clean_vs_other():
    # 轮 1:非 win 动作,engine 形态(以 <|im_end|>=1000001 结尾)→ clean;
    # 轮 2:win 动作,external 形态(停符被剥)且 len < turn_max → other。
    im_end = TOK.convert_tokens_to_ids("<|im_end|>")
    assert DELTA.stop_ids == {im_end}
    turns = [
        TOK.encode(reply("scan", "look")) + [im_end],
        TOK.encode(reply("win now", "do the thing")),
    ]
    result = await run(ScriptedAdapter(), _IdsGenerate(turns))
    meta = result.trajectory.meta
    assert result.trajectory.won
    assert meta["turns_stop_clean"] == 1
    assert meta["turns_stop_other"] == 1
    assert meta["turns_len_capped"] == 0


async def test_turn_stop_shape_sentinel_len_capped(monkeypatch):
    # 单轮恰好打满 turn_max(len == max_tokens)→ capped,即使末 token
    # 恰是停止符也按满额记(引擎侧 length-stop 优先于内容判读)。
    im_end = TOK.convert_tokens_to_ids("<|im_end|>")
    ids = TOK.encode(reply("w", "do the thing")) + [im_end]
    monkeypatch.setattr(protocol, "GEN_MAX_TOKENS_PER_TURN", len(ids))
    result = await run(ScriptedAdapter(), _IdsGenerate([ids]))
    meta = result.trajectory.meta
    assert result.trajectory.won
    assert meta["turns_len_capped"] == 1
    assert meta["turns_stop_clean"] == 0 and meta["turns_stop_other"] == 0
