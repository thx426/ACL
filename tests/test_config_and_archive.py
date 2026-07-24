import json

import pytest

from fakes import make_trajectory

from trace_opd.common.config import load_config
from trace_opd.docstore.archive import ResearchArchive
from trace_opd.narrate.factcard import build_fact_card
from trace_opd.narrate.narrator import NarrationResult
from trace_opd.research.session import SessionResult
from trace_opd.research.tools import EpisodeRecord


def _record(narration_text="The agent looked. The attempt failed after 1 steps."):
    traj = make_trajectory()
    return EpisodeRecord(
        trajectory=traj,
        fact_card=build_fact_card(traj),
        narration=NarrationResult(text=narration_text) if narration_text else None,
        mechanical_log=None if narration_text else "mech",
    )

MINIMAL_YAML = """
run: {name: t, runs_root: runs, tape: false}
research: {n_exp: 5, gt: "on"}
narration: {backend: engine}
roles:
  researcher: {endpoint: "${TEST_TRACE_EP}", model: m, max_concurrency: 2}
alfworld: {data_root: /d, worker_python: /p}
train: {model_path: /m, train_file: /t.parquet, val_files: [/v1.parquet]}
eval: {}
"""


def test_load_config_expands_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_TRACE_EP", "http://x:1/v1")
    path = tmp_path / "c.yaml"
    path.write_text(MINIMAL_YAML)
    cfg = load_config(path)
    assert cfg.research.n_exp == 5 and cfg.research.gt == "on"
    assert cfg.research.k_default == 4  # default preserved
    assert cfg.role("researcher").endpoint == "http://x:1/v1"
    assert cfg.narration.backend == "engine"
    assert cfg.train.val_files == ["/v1.parquet"]
    assert cfg.eval.response_budget == 32768


def test_load_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("research: {n_exp: 5, definitely_not_a_key: 1}")
    with pytest.raises(ValueError, match="definitely_not_a_key"):
        load_config(path)


def test_unknown_role_raises(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(MINIMAL_YAML)
    cfg = load_config(path)
    with pytest.raises(KeyError, match="narrator_external"):
        cfg.role("narrator_external")


def test_rollout_tp_must_divide_n_gpus(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(MINIMAL_YAML.replace(
        "train: {", "train: {n_gpus: 4, rollout_tp: 3, "))
    with pytest.raises(ValueError, match="rollout_tp"):
        load_config(path)


def test_sidecar_sleep_only_on_researcher(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(MINIMAL_YAML.replace(
        "narration: {backend: engine}",
        "narration: {backend: external}",
    ).replace(
        "roles:",
        "roles:\n  narrator_external: {endpoint: e, model: m, sidecar_sleep: true}",
    ))
    with pytest.raises(ValueError, match="narrator_external.sidecar_sleep"):
        load_config(path)


def test_sidecar_sleep_requires_external_research(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(MINIMAL_YAML.replace(
        "research: {n_exp: 5, gt: \"on\"}",
        "research: {enabled: false, tool_protocol: text}",
    ).replace(
        "researcher: {endpoint: \"${TEST_TRACE_EP}\", model: m, max_concurrency: 2}",
        "researcher: {endpoint: e, model: m, sidecar_sleep: true}",
    ))
    with pytest.raises(ValueError, match="sidecar_sleep=true requires"):
        load_config(path)


def test_sidecar_sleep_valid_config_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_TRACE_EP", "http://x:1/v1")
    path = tmp_path / "c.yaml"
    path.write_text(MINIMAL_YAML.replace(
        "model: m, max_concurrency: 2}",
        "model: m, max_concurrency: 2, sidecar_sleep: true}",
    ))
    cfg = load_config(path)
    assert cfg.role("researcher").sidecar_sleep is True
    assert cfg.train.rollout_tp == 2  # default
    assert cfg.train.rollout_n == 8  # default (user ruling: 16 tasks x 8 rollouts)
    assert cfg.train.ulysses_sp == 1  # default keeps the h3 profile


def test_rollout_n_must_be_positive(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(MINIMAL_YAML.replace("train: {", "train: {rollout_n: 0, "))
    with pytest.raises(ValueError, match="rollout_n"):
        load_config(path)


def test_batch_must_divide_across_agent_workers(tmp_path):
    # 30 % 8 != 0: task copies could straddle two worker processes and the
    # research-session dedupe invariant would break.
    path = tmp_path / "c.yaml"
    path.write_text(MINIMAL_YAML.replace("train: {", "train: {train_batch_size: 30, "))
    with pytest.raises(ValueError, match="dedupe"):
        load_config(path)


def test_episode_caps_default_and_validated(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_TRACE_EP", "http://x:1/v1")
    path = tmp_path / "c.yaml"
    path.write_text(MINIMAL_YAML)
    cfg = load_config(path)
    assert cfg.episode.max_env_steps == 30  # user ruling 2026-07-23 (was 50)
    assert cfg.episode.max_assistant_turns == 40
    # turns must exceed steps (the slack absorbs format failures)
    path.write_text(MINIMAL_YAML + "episode: {max_env_steps: 30, max_assistant_turns: 30}\n")
    with pytest.raises(ValueError, match="episode"):
        load_config(path)


def test_ulysses_sp_must_divide_n_gpus(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(MINIMAL_YAML.replace("train: {", "train: {ulysses_sp: 3, "))
    with pytest.raises(ValueError, match="ulysses_sp"):
        load_config(path)


def test_sequence_capacity_check_scales_with_ulysses_sp(tmp_path):
    # 20000 * sp=1 < 2048 + 32768: a full-length sample cannot fit -> reject.
    path = tmp_path / "c.yaml"
    path.write_text(MINIMAL_YAML.replace(
        "train: {", "train: {max_token_len_per_gpu: 20000, "))
    with pytest.raises(ValueError, match="sequence capacity"):
        load_config(path)
    # sp=2 doubles the capacity (40000 >= 34816): loads fine.
    path.write_text(MINIMAL_YAML.replace(
        "train: {", "train: {max_token_len_per_gpu: 20000, ulysses_sp: 2, "))
    cfg = load_config(path)
    assert cfg.train.ulysses_sp == 2


def _session_result(doc):
    return SessionResult(
        finalized=doc is not None,
        no_finalize=doc is None,
        turns_used=3,
        experiments_used=2,
        episodes_consumed=4,
        doc=doc,
        self_assessment="ok",
        flags=[],
        last_exp_successes=3,
        last_exp_attempts=4,
        messages=[{"role": "system", "content": "S"}, {"role": "assistant", "content": "A"}],
        guidance_log=[{"experiment_idx": 1}],
        prompt_tokens=12345,
        completion_tokens=678,
    )


def test_archive_layout(tmp_path):
    archive = ResearchArchive(tmp_path)
    out = archive.save(
        task_id="train/pick-x/trial_1",
        result=_session_result("# doc"),
        records=[_record()],
        policy_version=17,
        extra_meta={"gt": "on"},
    )
    assert out.parent.name == "train__pick-x__trial_1"
    assert (out / "doc.md").read_text() == "# doc"
    assert len((out / "session.jsonl").read_text().splitlines()) == 2
    rows = [json.loads(l) for l in (out / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    # the narration the researcher actually read is archived next to the trajectory
    assert rows[0]["narration"].startswith("The agent looked")
    assert rows[0]["trajectory"]["task_id"] == "fake/task_1"
    assert "FACT CARD" in rows[0]["fact_card"]
    meta = json.loads((out / "meta.json").read_text())
    assert meta["policy_version"] == 17
    assert meta["finalized"] is True
    assert meta["last_exp_successes"] == 3
    assert meta["gt"] == "on"
    # session-total researcher usage is an SFT filtering signal
    assert meta["researcher_prompt_tokens"] == 12345
    assert meta["researcher_completion_tokens"] == 678


def test_archive_no_doc_and_collision(tmp_path):
    archive = ResearchArchive(tmp_path)
    a = archive.save(task_id="t", result=_session_result(None), records=[])
    b = archive.save(task_id="t", result=None, records=[], extra_meta={"error": "boom"})
    assert a != b  # same-second collision resolved by suffix
    assert not (a / "doc.md").exists()
    meta_b = json.loads((b / "meta.json").read_text())
    assert meta_b["error"] == "boom"
    assert "finalized" not in meta_b


def test_archive_meta_survives_unserializable_policy_version(tmp_path):
    """Review P0-2: a wrapper object arriving as policy_version must degrade
    to a string in meta.json, never lose the whole meta file."""

    class Wrapper:
        def __str__(self):
            return "NonTensorData(7)"

    archive = ResearchArchive(tmp_path)
    out = archive.save(
        task_id="t2", result=_session_result(None), records=[], policy_version=Wrapper()
    )
    meta = json.loads((out / "meta.json").read_text())
    assert "NonTensorData" in meta["policy_version"]
