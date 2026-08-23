import pytest

from fakes import FakeTokenizer, ScriptedAdapter, ScriptedGenerate, reply

from trace_opd.core.episode import run_episode
from trace_opd.core.template_delta import TemplateDelta
from trace_opd.core.types import (
    GUIDANCE_HEADER,
    OUTCOME_ENV_ERROR,
    OUTCOME_SUCCESS,
)

pytestmark = pytest.mark.asyncio

TOK = FakeTokenizer()
DELTA = TemplateDelta(TOK)


async def run(adapter, texts, *, guidance="", budget=100000):
    gen = ScriptedGenerate(TOK, texts)
    result = await run_episode(
        adapter=adapter,
        generate=gen,
        tokenizer=TOK,
        delta=DELTA,
        guidance=guidance,
        episode_id="ep001",
        response_budget=budget,
        sampling_params={"temperature": 0.5},
    )
    return result, gen


async def test_success_episode_stream_and_trajectory():
    adapter = ScriptedAdapter()
    texts = [reply("look first", "look"), reply("now win", "do the thing")]
    result, gen = await run(adapter, texts)

    traj = result.trajectory
    assert traj.outcome == OUTCOME_SUCCESS and traj.won
    assert [s.action_norm for s in traj.steps] == ["look", "do the thing"]
    assert traj.steps[0].thought == "look first"
    assert traj.steps[0].valid  # "An empty room." is not a no-op
    assert result.assistant_turns == 2 and result.invalid_format == 0
    assert not result.truncated
    assert adapter.envs[0].closed

    # token stream: gen1(mask1) | glue+obs+post_obs(mask0) | gen2(mask1)
    gen1 = TOK.encode(texts[0])
    gen2 = TOK.encode(texts[1])
    obs = TOK.encode("An empty room.")
    structure = DELTA.after_generation(gen1) + obs + DELTA.post_obs
    assert result.tokens.response_ids == gen1 + structure + gen2
    assert result.tokens.response_mask == (
        [1] * len(gen1) + [0] * len(structure) + [1] * len(gen2)
    )
    # per-turn sampling params flow through, max_tokens managed by the runner
    assert gen.calls[0]["params"]["temperature"] == 0.5
    assert "max_tokens" in gen.calls[0]["params"]
    # engine extras captured
    assert result.gen_extra["min_global_steps"] == 7


async def test_guidance_injected_into_system_prompt():
    adapter = ScriptedAdapter()
    result, _ = await run(adapter, [reply("w", "do the thing")], guidance="Go straight to it.")
    rendered = TOK.decode(result.tokens.prompt_ids, skip_special_tokens=False)
    # 特权块 = header + 框(参考资料/非任务/需自行核实)+ 正文,见 core.types
    assert GUIDANCE_HEADER in rendered
    assert "Go straight to it." in rendered
    assert result.trajectory.guidance_hash != "bare"


async def test_format_failure_breaker():
    adapter = ScriptedAdapter()
    texts = ["no tags", "still none", "nope"]
    result, _ = await run(adapter, texts)
    traj = result.trajectory
    assert traj.outcome == "DEGENERATE:format"
    assert result.invalid_format == 3
    assert all(not s.acted for s in traj.steps)
    assert result.truncated
    assert adapter.envs[0].commands == []  # nothing ever reached the engine


async def test_format_failure_recovers_before_breaker():
    adapter = ScriptedAdapter()
    texts = ["no tags", reply("ok", "do the thing")]
    result, _ = await run(adapter, texts)
    assert result.trajectory.outcome == OUTCOME_SUCCESS
    assert result.invalid_format == 1


async def test_budget_truncation():
    adapter = ScriptedAdapter()
    texts = [reply("look", "look")] * 50
    result, _ = await run(adapter, texts, budget=200)
    assert result.truncated
    assert len(result.tokens.response_ids) <= 200


async def test_env_open_failure_is_env_error():
    adapter = ScriptedAdapter(fail_open=True)
    result, _ = await run(adapter, [reply("x", "look")])
    assert result.trajectory.outcome == OUTCOME_ENV_ERROR
    assert result.tokens.response_ids == []
    assert "engine exploded" in (result.trajectory.error or "")


async def test_empty_generation_truncates():
    adapter = ScriptedAdapter()
    result, _ = await run(adapter, [])  # generator immediately exhausted
    assert result.truncated
    assert result.trajectory.outcome != OUTCOME_SUCCESS
