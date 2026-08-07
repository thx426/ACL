import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from trace_opd.common.config import load_config, objective_fingerprint

ROOT = Path(__file__).resolve().parents[1]
CALIB = ROOT / "configs/experiments/162-pgs-s2-sglang-1p7b-researcher-maxopen-calibration.yaml"
MATERIALIZER_PATH = ROOT / "scripts/materialize_162_researcher_maxopen.py"
WRAPPER_PATH = ROOT / "scripts/run_162_researcher_maxopen.py"
AUDITOR_PATH = ROOT / "scripts/audit_162_researcher_maxopen.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAT = _load_module("materialize_maxopen", MATERIALIZER_PATH)
AUDIT = _load_module("audit_maxopen", AUDITOR_PATH)
WRAP = _load_module("run_maxopen", WRAPPER_PATH)


@pytest.fixture(autouse=True)
def credential(monkeypatch):
    monkeypatch.setenv("TRACE_RESEARCHER_API_KEY", "unit-test-researcher-key")
    monkeypatch.setenv("PGS_MAXOPEN_ENTRY_SOURCED", "1")


def _dynamic_meta():
    return {
        "global_step": 1,
        "policy_version": 0,
        "finalized": True,
        "error": None,
        "session_error": None,
    }


def _steering(*, closed):
    count = 0 if closed else 1
    return {
        "steer_count": count,
        "steer_reject_count": 0 if closed else 1,
        "steer_n_base_open": 0 if closed else 1,
        "steer_n_probe_flipped": 0,
        "steer_n_budget_blocked": 0,
        "steer_n_pacing_wait": 0,
        "steer_n_bypassed": 0,
        "steer_turns_capped": 0,
        "steer_d_count": 2,
        "steer_d_hist": [1, 1, 0, 0],
        "steer_d_hist_bins": 4,
        "steer_d_hist_max": 1.0,
    }


def make_complete_run(tmp_path, *, closed=True):
    run = tmp_path / "run"
    step = run / "rollouts/step_00001"
    step.mkdir(parents=True)
    for task_index in range(16):
        archive = f"research/task-{task_index}/archive-v0"
        archive_dir = run / archive
        archive_dir.mkdir(parents=True)
        (archive_dir / "doc.md").write_text(f"document {task_index}")
        (archive_dir / "meta.json").write_text(json.dumps(_dynamic_meta()))
        for sample_index in range(8):
            row = {
                "validate": False,
                "task_id": f"train/task-{task_index}",
                "turn_token_spans": [[0, 2]],
                "policy_version": {"min_global_steps": 0, "max_global_steps": 0},
                "research": {
                    "research_error": 0,
                    "research_doc_missing": 0,
                    "research_policy_version": 0,
                    "research_archive": archive,
                },
                "steering": _steering(closed=closed),
            }
            (step / f"{task_index:02d}-{sample_index}.json").write_text(json.dumps(row))
    return run


def test_calibration_config_is_finite_horizon_nonbinding_and_dynamic():
    cfg = load_config(CALIB)
    st, train = cfg.steering, cfg.train
    assert st.doc_source == "researcher" and cfg.research.enabled
    assert cfg.research.n_exp == 16
    assert cfg.role("researcher").model == "qwen3.6-35b-a3b"
    assert cfg.role("researcher").extra_body["chat_template_kwargs"]["enable_thinking"] is False
    assert train.model_path == "/data3/wushang/model/Qwen/Qwen3-1.7B"
    assert train.n_gpus == 4 and train.rollout_tp == 1 and train.ulysses_sp == 2
    assert train.train_batch_size == 16 and train.rollout_n == 8
    assert train.rollout_max_num_seqs == 256
    assert train.total_epochs == 2 and train.test_freq == train.save_freq == 10
    assert train.max_ckpt_to_keep == 1 and train.loggers == ["console", "swanlab"]
    assert st.epsilon == 0 and st.gate_target_rate == 0.20
    assert st.theta_d == 1_000_000_000 and not st.gate_adaptive
    assert st.b_max == train.response_length * 2 * st.eta_max * st.c == 81920
    assert not st.budget_pacing
    assert st.r_max > train.response_length


def test_all_treatment_fields_move_objective_fingerprint(tmp_path):
    cfg = load_config(CALIB)
    base = objective_fingerprint(cfg)
    cases = {
        "epsilon": 0.01,
        "gate_target_rate": 0.21,
        "b_max": 81919.0,
        "r_max": 32770,
        "budget_pacing": True,
    }
    raw = yaml.safe_load(CALIB.read_text())
    for key, value in cases.items():
        data = json.loads(json.dumps(raw))
        data["steering"][key] = value
        path = tmp_path / f"{key}.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        assert objective_fingerprint(load_config(path)) != base


def test_materializer_requires_complete_zero_intervention_calibration(tmp_path, monkeypatch):
    run = make_complete_run(tmp_path, closed=True)
    output = tmp_path / "formal.yaml"
    audit = tmp_path / "audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "materialize",
            "--calibration-run", str(run),
            "--calibration-config", str(CALIB),
            "--output", str(output),
            "--audit-output", str(audit),
        ],
    )
    assert MAT.main() == 0
    cfg = load_config(output)
    assert cfg.run.name == MAT.FORMAL_NAME
    assert cfg.steering.theta_d == pytest.approx(0.4)
    assert cfg.steering.gate_adaptive
    assert cfg.steering.gate_rate_bounds == (0.10, 0.30)
    assert cfg.steering.epsilon == 0
    assert cfg.steering.b_max == 81920 and cfg.steering.r_max == 32769
    evidence = json.loads(audit.read_text())
    assert evidence["tasks"] == 16 and evidence["trajectories"] == 128
    assert evidence["theta_d"] == pytest.approx(0.4)
    assert evidence["objective_fingerprint"] == objective_fingerprint(cfg)

    with pytest.raises(SystemExit, match="refusing overwrite"):
        MAT.main()


def test_materializer_rejects_partial_degraded_or_intervened_runs(tmp_path):
    run = make_complete_run(tmp_path, closed=True)
    (run / "rollouts/step_00001/00-0.json").unlink()
    with pytest.raises(SystemExit, match="128"):
        MAT._load_step(run)

    run = make_complete_run(tmp_path / "degraded", closed=True)
    path = run / "rollouts/step_00001/00-0.json"
    row = json.loads(path.read_text())
    row["research"]["research_error"] = 1
    path.write_text(json.dumps(row))
    with pytest.raises(SystemExit, match="degraded"):
        MAT._audit_dynamic(run, MAT._load_step(run)[1])

    run = make_complete_run(tmp_path / "intervened", closed=True)
    path = run / "rollouts/step_00001/00-0.json"
    row = json.loads(path.read_text())
    row["steering"]["steer_count"] = 1
    path.write_text(json.dumps(row))
    with pytest.raises(SystemExit, match="changed behavior"):
        MAT._audit_dynamic(run, MAT._load_step(run)[1])


def test_smoke_auditor_proves_every_position_covered_and_no_limiters(tmp_path):
    run = make_complete_run(tmp_path, closed=False)
    data = yaml.safe_load(CALIB.read_text())
    data["steering"]["theta_d"] = 0.2
    data["steering"]["gate_adaptive"] = True
    (run / "effective_config.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    rows = AUDIT._load_rows(run)
    result = AUDIT._audit_rows(run, rows)
    assert result["assistant_tokens"] == result["d_observations"] == 256
    assert result["pacing_wait"] == result["bypassed"] == result["budget_blocked"] == 0
    assert result["turns_capped"] == result["probe_flipped"] == 0
    assert result["tilts"] == 128 and result["base_open_rate"] == 0.5

    path = run / "rollouts/step_00001/00-0.json"
    row = json.loads(path.read_text())
    row["steering"]["steer_n_bypassed"] = 1
    path.write_text(json.dumps(row))
    with pytest.raises(SystemExit, match="steer_n_bypassed"):
        AUDIT._audit_rows(run, AUDIT._load_rows(run))


def test_wrapper_rejects_sentinel_as_formal_and_requires_online_credentials(tmp_path, monkeypatch):
    assert WRAP._config_contract(CALIB, calibration=True)
    with pytest.raises(SystemExit, match="materialized adaptive P80"):
        WRAP._config_contract(CALIB, calibration=False)

    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("research-key\nswan-key\n"))
    monkeypatch.setenv("SWANLAB_MODE", "offline")
    with pytest.raises(SystemExit, match="online SwanLab"):
        WRAP._base_env(dry_run=False)

    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("\n\n"))
    with pytest.raises(SystemExit, match="credential envelope"):
        WRAP._base_env(dry_run=False)

    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(""))
    env = WRAP._base_env(dry_run=True)
    assert env["TRACE_RESEARCHER_API_KEY"] == "dry-run-placeholder"
    assert env["SWANLAB_API_KEY"] == "dry-run-placeholder"
    assert "SWANLAB_MODE" not in env


def test_wrapper_source_never_contains_secret_values_or_proc_scraping():
    source = WRAPPER_PATH.read_text()
    assert "TUNNEL_API_KEY=" not in source
    assert 'Path("/proc")' not in source
    assert "TRACE_RESEARCHER_API_KEY" in source
    assert "SWANLAB_API_KEY" in source
    assert "sys.stdin.readline" in source


def test_secret_scanner_streams_and_skips_checkpoint_blobs(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "safe.txt").write_text("safe")
    checkpoint = run / "checkpoints/global_step_1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.bin").write_bytes(b"secret-value")
    assert AUDIT._scan_secrets([run], {"secret-value"}) == 1
    (run / "leak.txt").write_bytes(b"prefix-secret-value-suffix")
    with pytest.raises(SystemExit, match="credential value leaked"):
        AUDIT._scan_secrets([run], {"secret-value"})
