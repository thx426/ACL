"""Mixed objective (OPD x RL x SFT) — docs/mixed-objective-design.md.

Covers the three places the arms can silently go wrong: the expert sample's
bare-prompt replay, the accept/reject filters, and the config/launcher wiring
that decides how many rollouts run and which loss terms are active.
"""
import pytest

from fakes import FakeTokenizer, ScriptedAdapter, ScriptedGenerate, reply

from trace_opd.core.episode import replay_bare_tokens, run_episode
from trace_opd.core.expert import build_expert_sample
from trace_opd.core.template_delta import TemplateDelta
from trace_opd.core.types import mentions_guidance

TOK = FakeTokenizer()
DELTA = TemplateDelta(TOK)

DOC = "Pick up the thing, then do the thing."


async def _episode(guidance: str, texts):
    return await run_episode(
        adapter=ScriptedAdapter(), generate=ScriptedGenerate(TOK, texts),
        tokenizer=TOK, delta=DELTA, guidance=guidance, episode_id="ep",
        response_budget=100000, sampling_params={},
    )


# --------------------------------------------------------------- the replay


@pytest.mark.asyncio
async def test_replay_swaps_the_prompt_and_keeps_the_response_stream():
    """The guidance lives only in the system message, so replaying an expert
    trajectory under the bare prompt must touch prompt_ids and nothing else —
    that identity is what makes the replay exact instead of a re-tokenization."""
    texts = [reply("look around", "look"), reply("now win", "do the thing")]
    guided = await _episode(DOC, texts)
    assert guided.trajectory.won

    adapter = ScriptedAdapter()
    replayed = replay_bare_tokens(adapter=adapter, tokenizer=TOK, result=guided)

    assert replayed.response_ids == guided.tokens.response_ids
    assert replayed.response_mask == guided.tokens.response_mask
    assert replayed.turn_spans == guided.tokens.turn_spans
    assert replayed.prompt_ids != guided.tokens.prompt_ids

    # And the bare prompt is exactly what a bare rollout of the same task uses.
    bare = await _episode("", texts)
    assert replayed.prompt_ids == bare.tokens.prompt_ids
    # The document is gone from the prompt, not merely reordered.
    assert DOC not in TOK.decode(replayed.prompt_ids, skip_special_tokens=False)
    assert DOC in TOK.decode(guided.tokens.prompt_ids, skip_special_tokens=False)


@pytest.mark.asyncio
async def test_replay_rejects_a_prompt_that_lands_in_a_different_context():
    """A drifted template would put the first response token in a different
    generation state; the replay must fail loud rather than train on it."""
    texts = [reply("now win", "do the thing")]
    guided = await _episode(DOC, texts)
    guided.tokens.prompt_ids = guided.tokens.prompt_ids[:-4] + [7, 7, 7, 7]

    with pytest.raises(RuntimeError, match="diverge"):
        replay_bare_tokens(adapter=ScriptedAdapter(), tokenizer=TOK, result=guided)


@pytest.mark.asyncio
async def test_replay_response_is_byte_identical_to_a_bare_rollout_of_the_same_text():
    """Stronger form of the identity: with the same generated text, a guided and
    a bare episode produce the same response stream. If this ever breaks, the
    SFT arm would be training on tokens the bare policy cannot reproduce."""
    texts = [reply("look around", "look"), reply("now win", "do the thing")]
    guided = await _episode(DOC, texts)
    bare = await _episode("", texts)
    assert guided.tokens.response_ids == bare.tokens.response_ids
    assert guided.tokens.response_mask == bare.tokens.response_mask


# ------------------------------------------------------------- the filters


def test_guidance_reference_filter_catches_the_ways_a_trajectory_names_the_doc():
    assert mentions_guidance("Following the task guidance, I should look first.")
    assert mentions_guidance("THE GUIDANCE says to open the fridge")
    assert mentions_guidance("per the guidance document, take the mug")
    # Ordinary reasoning must survive — the arm is useless if it drops everything.
    assert not mentions_guidance("I should look around, then take the mug.")
    assert not mentions_guidance("")


@pytest.mark.asyncio
async def test_expert_sample_accepts_a_win_and_zeroes_the_mask_otherwise():
    """A rejected expert row stays well-formed with an all-zero mask: several
    verl paths index the last response token, so an empty response would crash,
    while a zero mask contributes exactly nothing to the loss."""
    won = await _episode(DOC, [reply("now win", "do the thing")])
    tokens, fields = _sample(won)
    assert fields["sample_role"] == "sft"
    assert (fields["sft_won"], fields["sft_accepted"], fields["sft_dropped_ref"]) == (1, 1, 0)
    assert any(tokens.response_mask)

    lost = await _episode(DOC, [reply("wrong", "look")] * 3)
    tokens, fields = _sample(lost)
    assert (fields["sft_won"], fields["sft_accepted"]) == (0, 0)
    assert tokens.response_ids and not any(tokens.response_mask)


@pytest.mark.asyncio
async def test_expert_sample_drops_a_win_that_talks_about_the_document():
    result = await _episode(DOC, [reply("the task guidance says to act", "do the thing")])
    assert result.trajectory.won
    tokens, fields = _sample(result)
    assert (fields["sft_won"], fields["sft_dropped_ref"], fields["sft_accepted"]) == (1, 1, 0)
    assert not any(tokens.response_mask)


@pytest.mark.asyncio
async def test_expert_sample_drops_a_win_that_overflows_the_budget():
    result = await _episode(DOC, [reply("now win", "do the thing")])
    _, fields = _sample(result, max_model_len=16)
    assert (fields["sft_won"], fields["sft_accepted"]) == (1, 0)


def _sample(result, max_model_len: int = 40960):
    return build_expert_sample(
        adapter=ScriptedAdapter(), tokenizer=TOK, result=result, max_model_len=max_model_len
    )


# --------------------------------------------------------------- the wiring


_YAML = """
run: {name: t, runs_root: runs, tape: false}
research: {enabled: %s}
narration: {backend: engine}
roles:
  researcher: {endpoint: "http://x:1/v1", model: m}
alfworld: {data_root: /d, worker_python: /p}
train: {model_path: /m, train_file: /t.parquet, val_files: [/v1.parquet]%s}
eval: {}
"""


def _load(tmp_path, train_extra="", research_enabled="true"):
    import importlib.util
    from pathlib import Path

    from trace_opd.common.config import load_config

    path = Path(tmp_path)
    path.mkdir(parents=True, exist_ok=True)
    cfg_path = path / "c.yaml"
    cfg_path.write_text(_YAML % (research_enabled, train_extra))
    cfg = load_config(cfg_path)

    launcher = Path(__file__).resolve().parents[1] / "scripts" / "launch_train.py"
    spec = importlib.util.spec_from_file_location("launch_train_mixed", launcher)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cfg, module


def test_config_rejects_an_sft_arm_that_cannot_produce_samples(tmp_path):
    with pytest.raises(ValueError, match="mix_sft_rollouts"):
        _load(tmp_path, ", mix_sft_weight: 0.5, mix_sft_rollouts: 0")


def test_config_rejects_an_sft_arm_without_a_document_source(tmp_path):
    with pytest.raises(ValueError, match="research.enabled"):
        _load(tmp_path, ", mix_sft_weight: 0.5", research_enabled="false")


def test_config_rejects_a_negative_weight(tmp_path):
    with pytest.raises(ValueError, match="mix_rl_weight"):
        _load(tmp_path, ", mix_rl_weight: -1.0")


def test_launcher_adds_expert_rollouts_only_when_the_sft_arm_is_on(tmp_path):
    cfg, module = _load(tmp_path / "off")
    off = module.build_command(cfg, tmp_path / "run")
    assert "actor_rollout_ref.rollout.n=8" in off
    assert "+distillation.distillation_loss.sft_loss_coef=0.0" in off
    assert "+distillation.distillation_loss.rl_loss_coef=0.0" in off

    cfg, module = _load(tmp_path / "on", ", mix_sft_weight: 0.5, mix_rl_weight: 1.0")
    on = module.build_command(cfg, tmp_path / "run")
    # 8 bare + 4 expert: the bare GRPO group keeps its size, the expert copies
    # ride on top rather than eating into it.
    assert "actor_rollout_ref.rollout.n=12" in on
    assert "+distillation.distillation_loss.sft_loss_coef=0.5" in on
    assert "+distillation.distillation_loss.rl_loss_coef=1.0" in on
