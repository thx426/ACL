import pytest

from fakes import make_trajectory

from trace_opd.core.types import OUTCOME_ENV_ERROR, Step, Trajectory
from trace_opd.narrate.assertions import classify
from trace_opd.narrate.factcard import build_fact_card
from trace_opd.narrate.narrator import narrate_episode
from trace_opd.narrate.preprocess import render_for_narrator, render_raw



def _repeating_traj() -> Trajectory:
    traj = make_trajectory()
    traj.steps = [
        Step(t=i + 1, thought="again", action_raw="look", action_norm="look",
             observation="Nothing happens.", valid=False)
        for i in range(5)
    ]
    return traj


@pytest.mark.asyncio
async def test_narrate_success_and_conflict_flag():
    traj = make_trajectory(won=True)  # engine says SUCCESS

    async def lying_narrator(messages, temperature, max_tokens):
        return "The agent wandered. The attempt failed after 1 steps."

    result = await narrate_episode(traj, lying_narrator)
    assert result.usable and result.conflict

    async def honest_narrator(messages, temperature, max_tokens):
        return "The agent acted once and succeeded, completing the task in 1 step."

    result = await narrate_episode(traj, honest_narrator)
    assert result.usable and not result.conflict


@pytest.mark.asyncio
async def test_narrate_skips_env_error_without_llm_call():
    traj = make_trajectory()
    traj.outcome = OUTCOME_ENV_ERROR
    calls = []

    async def narrator(messages, temperature, max_tokens):
        calls.append(1)
        return "x"

    result = await narrate_episode(traj, narrator)
    assert result.skip is not None and result.skip.kind == "env_error"
    assert not calls


@pytest.mark.asyncio
async def test_narrate_soft_lands_on_llm_failure():
    async def broken(messages, temperature, max_tokens):
        raise RuntimeError("server down")

    result = await narrate_episode(make_trajectory(), broken)
    assert not result.usable and "server down" in result.error


def test_classify_empty_and_unparseable():
    traj = make_trajectory()
    traj.steps = []
    assert classify(traj).kind == "empty"
    traj = make_trajectory()
    traj.steps[0].acted = False
    assert classify(traj).kind == "all_unparseable"
    assert classify(make_trajectory()) is None


def test_fact_card_counts_repeats():
    card = build_fact_card(_repeating_traj())
    assert card.longest_repeat_action == "look" and card.longest_repeat_count == 5
    assert card.invalid_count == 5
    assert "longest_repeat" in card.render()


def test_render_folds_repeats_and_render_raw_clamps():
    text = render_for_narrator(_repeating_traj())
    assert "repeated 5x" in text
    raw, rng = render_raw(_repeating_traj(), (2, 99), 10000)
    assert rng == (2, 5) and "RAW LOG" in raw
    tiny, _ = render_raw(_repeating_traj(), None, 80)
    assert "output cut" in tiny
