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
