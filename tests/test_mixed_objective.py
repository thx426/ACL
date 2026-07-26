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
async def test_doc_prefix_keeps_the_prompt_the_episode_was_generated_under():
    """The default. The SFT term then optimises policy(.|bare+document), which is
    the OPD teacher — "SFT strengthens the teacher, OPD carries it across". The
    alternative teaches the bare policy directly, which overlaps with OPD and
    leaks: an expert knows where objects are because it read the document."""
    result = await _episode(DOC, [reply("now win", "do the thing")])
    tokens, fields = _sample(result, prefix="doc")
    assert fields["sft_accepted"] == 1
    assert tokens.prompt_ids == result.tokens.prompt_ids      # untouched
    assert DOC in TOK.decode(tokens.prompt_ids, skip_special_tokens=False)
    assert tokens.response_ids == result.tokens.response_ids

    bare_tokens, _ = _sample(result, prefix="bare")
    assert bare_tokens.prompt_ids != result.tokens.prompt_ids  # re-addressed
    assert DOC not in TOK.decode(bare_tokens.prompt_ids, skip_special_tokens=False)
    # Same trajectory either way — only what it is conditioned on differs.
    assert bare_tokens.response_ids == tokens.response_ids


@pytest.mark.asyncio
async def test_reference_filter_applies_to_the_bare_prefix_only():
    """Referring to the document is hallucination without it and simply correct
    with it, so filtering under the doc prefix would drop good samples."""
    result = await _episode(DOC, [reply("the task guidance says to act", "do the thing")])
    assert result.trajectory.won

    _, bare_fields = _sample(result, prefix="bare")
    assert (bare_fields["sft_dropped_ref"], bare_fields["sft_accepted"]) == (1, 0)

    doc_tokens, doc_fields = _sample(result, prefix="doc")
    assert (doc_fields["sft_dropped_ref"], doc_fields["sft_accepted"]) == (0, 1)
    assert any(doc_tokens.response_mask)


def test_expert_sample_rejects_an_unknown_prefix():
    import asyncio

    result = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        _episode(DOC, [reply("now win", "do the thing")])
    )
    with pytest.raises(ValueError, match="must be 'doc' or 'bare'"):
        _sample(result, prefix="both")


@pytest.mark.asyncio
async def test_expert_sample_drops_a_win_that_overflows_the_budget():
    result = await _episode(DOC, [reply("now win", "do the thing")])
    _, fields = _sample(result, max_model_len=16)
    assert (fields["sft_won"], fields["sft_accepted"]) == (1, 0)


def _sample(result, max_model_len: int = 40960, prefix: str = "doc"):
    return build_expert_sample(
        adapter=ScriptedAdapter(), tokenizer=TOK, result=result,
        max_model_len=max_model_len, prefix=prefix,
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


def _load(tmp_path, train_extra="", research_enabled="true", overrides=None):
    import importlib.util
    from pathlib import Path

    from trace_opd.common.config import load_config

    path = Path(tmp_path)
    path.mkdir(parents=True, exist_ok=True)
    cfg_path = path / "c.yaml"
    cfg_path.write_text(_YAML % (research_enabled, train_extra))
    cfg = load_config(cfg_path, overrides=overrides)

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


@pytest.mark.asyncio
async def test_expert_sample_refuses_an_episode_with_no_response():
    """An empty response is the one shape the trainer cannot take (verl writes
    rm_scores[-1] unconditionally), so the contract fails loud here and the
    agent loop falls back to a bare rollout."""
    from trace_opd.core.episode import EpisodeResult, EpisodeTokens
    from trace_opd.core.types import OUTCOME_ENV_ERROR, Trajectory

    traj = Trajectory(episode_id="e", task_id="fake/task_1", guidance_hash="h",
                      sample_idx=0, initial_obs="")
    traj.outcome = OUTCOME_ENV_ERROR
    empty = EpisodeResult(
        trajectory=traj,
        tokens=EpisodeTokens(prompt_ids=[], response_ids=[], response_mask=[]),
        assistant_turns=0, invalid_format=0, truncated=False, gen_extra={},
    )
    with pytest.raises(ValueError, match="no response tokens"):
        _sample(empty)


# ------------------------------------------------- config override channel


def test_override_selects_an_arm_without_touching_the_config_file(tmp_path):
    """Ablation arms are experiment design, not machine setup: they live in the
    config file, and a run picks one with an explicit, recorded override rather
    than a launcher rewriting the file behind the user's back."""
    cfg, module = _load(
        tmp_path,
        overrides=["train.mix_rl_weight=1.0", "train.mix_sft_weight=0.5"],
    )
    assert (cfg.train.mix_rl_weight, cfg.train.mix_sft_weight) == (1.0, 0.5)
    # The raw snapshot the launch manifest stores must show what actually ran.
    assert cfg.raw["train"]["mix_rl_weight"] == 1.0
    cmd = module.build_command(cfg, tmp_path / "run")
    assert "actor_rollout_ref.rollout.n=12" in cmd
    assert "+distillation.distillation_loss.rl_loss_coef=1.0" in cmd


def test_override_without_one_leaves_the_file_value_alone(tmp_path):
    """The failure this replaced: a launcher that rewrote both keys every run
    silently reset a weight set in the config file back to zero."""
    cfg, _ = _load(tmp_path, ", mix_sft_weight: 0.5", overrides=["train.mix_rl_weight=1.0"])
    assert cfg.train.mix_sft_weight == 0.5


def test_override_goes_through_the_same_validation_as_the_file(tmp_path):
    with pytest.raises(ValueError, match="mix_rl_weight"):
        _load(tmp_path, overrides=["train.mix_rl_weight=-1.0"])


def test_override_rejects_a_typo_instead_of_ignoring_it(tmp_path):
    with pytest.raises(ValueError, match="mix_rl_wieght"):
        _load(tmp_path, overrides=["train.mix_rl_wieght=1.0"])
    with pytest.raises(ValueError, match="unknown section"):
        _load(tmp_path, overrides=["trian.mix_rl_weight=1.0"])
    with pytest.raises(ValueError, match="must name a section"):
        _load(tmp_path, overrides=["mix_rl_weight=1.0"])


def test_override_walks_nested_sections(tmp_path):
    """roles.researcher.mode is three levels deep. Splitting the path once put a
    literal "researcher.mode" key under roles, which only surfaced much later as
    a nonsense error from iterating a string's characters (CD-006 report)."""
    from trace_opd.common.config import apply_overrides

    data = {"roles": {"researcher": {"model": "m"}}}
    out = apply_overrides(data, ["roles.researcher.mode=external_pool"])
    assert out["roles"]["researcher"] == {"model": "m", "mode": "external_pool"}
    assert "researcher.mode" not in out["roles"]


def test_override_refuses_to_descend_into_a_non_mapping(tmp_path):
    from trace_opd.common.config import apply_overrides

    with pytest.raises(ValueError, match="train.lr is not a mapping"):
        apply_overrides({"train": {"lr": 1e-6}}, ["train.lr.nested=1"])


# ------------------------------------- config template + experiment copies


# A template must declare every key an experiment config may carry — that is
# what makes an undeclared key a typo rather than a new setting.
_TEMPLATE_YAML = """
run: {name: t, runs_root: runs, tape: false}
research: {enabled: true}
narration: {backend: engine}
roles:
  researcher: {endpoint: "http://x:1/v1", model: m}
alfworld: {data_root: /d, worker_python: /p}
train:
  model_path: /m
  train_file: /t.parquet
  val_files: [/v1.parquet]
  n_gpus: 2
  mix_rl_weight: 0.0
  mix_sft_weight: 0.0
  mix_sft_rollouts: 4
  teacher_scoring_backend: fsdp
eval: {}
"""


def _write(tmp_path, name, body):
    from pathlib import Path

    root = Path(tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(body)
    return path


def test_experiment_config_is_a_complete_copy_not_a_diff(tmp_path):
    """An experiment owns the whole file. It reads on its own months later, and a
    later template edit cannot retroactively change what it meant."""
    from trace_opd.common.config import load_config, template_drift

    template = _write(tmp_path, "template.yaml", _TEMPLATE_YAML)
    experiment = _write(
        tmp_path, "exp.yaml", _TEMPLATE_YAML.replace("mix_sft_weight: 0.0", "mix_sft_weight: 0.5")
    )
    cfg = load_config(experiment)
    assert cfg.train.mix_sft_weight == 0.5
    assert cfg.train.mix_rl_weight == 0.0
    assert template_drift(experiment, template) == {"missing": [], "extra": []}

    # Editing the template afterwards does not reach the experiment...
    template.write_text(_TEMPLATE_YAML.replace("mix_rl_weight: 0.0", "mix_rl_weight: 1.0"))
    assert load_config(experiment).train.mix_rl_weight == 0.0
    # ...and the values themselves are not what drift tracks — only the key set.
    assert template_drift(experiment, template) == {"missing": [], "extra": []}


def test_drift_reports_what_the_template_grew_and_what_the_copy_kept(tmp_path):
    """The cost of copies is drift, so it has to be visible at launch rather than
    discovered as a run quietly using a code default."""
    from trace_opd.common.config import template_drift

    template = _write(tmp_path, "template.yaml", _TEMPLATE_YAML)
    behind = _write(
        tmp_path, "behind.yaml", _TEMPLATE_YAML.replace("  teacher_scoring_backend: fsdp\n", "")
    )
    assert template_drift(behind, template)["missing"] == ["train.teacher_scoring_backend"]

    stale = _write(tmp_path, "stale.yaml", _TEMPLATE_YAML.replace(
        "  mix_sft_rollouts: 4", "  mix_sft_rollouts: 4\n  distill_block_turns: 4"))
    assert template_drift(stale, template)["extra"] == ["train.distill_block_turns"]


def test_drift_compares_the_shape_of_a_role_not_which_roles_exist(tmp_path):
    """Which roles a run serves is its own business; whether a role carries the
    settings the mechanism expects is not."""
    from trace_opd.common.config import template_drift

    template = _write(tmp_path, "template.yaml", _TEMPLATE_YAML)
    extra_role = _write(tmp_path, "exp.yaml", _TEMPLATE_YAML.replace(
        '  researcher: {endpoint: "http://x:1/v1", model: m}',
        '  researcher: {endpoint: "http://x:1/v1", model: m}\n'
        '  narrator_external: {endpoint: "http://y:2/v1", model: n}'))
    assert template_drift(extra_role, template) == {"missing": [], "extra": []}


def test_objective_fingerprint_separates_the_arms(tmp_path):
    """Arms share a default run name, so resuming the wrong one is a forgotten
    flag away. The fingerprint is what the launcher refuses on."""
    from trace_opd.common.config import load_config, objective_fingerprint

    def arm(name, body):
        return objective_fingerprint(load_config(_write(tmp_path / name, "c.yaml", body)))

    a0 = arm("a0", _TEMPLATE_YAML)
    a3 = arm("a3", _TEMPLATE_YAML.replace("mix_rl_weight: 0.0", "mix_rl_weight: 1.0"))
    same = arm("same", _TEMPLATE_YAML)
    assert a0 != a3
    assert a0 == same
    # A machine-only difference must NOT change it: resuming on other hardware
    # is legitimate, resuming under another objective is not.
    assert arm("m8", _TEMPLATE_YAML.replace("n_gpus: 2", "n_gpus: 8")) == a0


def test_shipped_experiment_configs_load_and_stay_in_step():
    """The configs in the repo must actually load against the current schema and
    not have fallen behind the template."""
    from pathlib import Path

    from trace_opd.common.config import load_config, template_drift

    configs = Path(__file__).resolve().parents[1] / "configs"
    template = configs / "trace.yaml"
    for experiment in sorted((configs / "experiments").glob("*.yaml")):
        load_config(experiment)
        assert template_drift(experiment, template)["extra"] == [], experiment
