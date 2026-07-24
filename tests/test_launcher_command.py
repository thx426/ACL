"""build_command — the yaml->verl translation must carry the two new
mechanism switches faithfully: rollout_n always, and the ulysses_sp>1
three-way linkage (SP size + use_remove_padding=True + flash_attention_2)
versus the exact h3 profile (rmpad=False + sdpa) at sp=1.
"""
import importlib.util
from pathlib import Path

from trace_opd.common.config import load_config

_LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "launch_train.py"
_spec = importlib.util.spec_from_file_location("launch_train", _LAUNCHER)
launch_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch_train)

YAML = """
run: {name: t, runs_root: runs, tape: false}
research: {}
narration: {backend: engine}
roles:
  researcher: {endpoint: "http://x:1/v1", model: m}
alfworld: {data_root: /d, worker_python: /p}
train: {model_path: /m, train_file: /t.parquet, val_files: [/v1.parquet]%s}
eval: {}
"""


def _cmd(tmp_path, train_extra=""):
    path = tmp_path / "c.yaml"
    path.write_text(YAML % train_extra)
    cfg = load_config(path)
    return launch_train.build_command(cfg, tmp_path / "run")


def test_default_profile_matches_h3_and_carries_rollout_n(tmp_path):
    cmd = _cmd(tmp_path)
    assert "actor_rollout_ref.rollout.n=8" in cmd
    assert "actor_rollout_ref.model.use_remove_padding=False" in cmd
    assert "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" in cmd
    assert not any("ulysses_sequence_parallel_size" in c for c in cmd)
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=2" in cmd
    # Smoke-run OOM fix (feedback P0): the chunked-prefill budget is a config
    # value now, halved from h3's 4096 for the coexistence topology.
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=2048" in cmd
    # Collapse-watch entropy curve (metric only, entropy_coeff stays 0).
    assert "actor_rollout_ref.actor.calculate_entropy=True" in cmd
    # Micro-batch packing budgets decoupled from the 34816 capacity declaration
    # (feedback P0: 34816 fed 21.2 GiB fp32 logits into both phases).
    assert "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192" in cmd
    assert "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12288" in cmd
    assert not any(c.endswith("token_len_per_gpu=34816") for c in cmd)


def test_ulysses_sp_flips_padding_and_attention(tmp_path):
    cmd = _cmd(tmp_path, ", ulysses_sp: 2")
    assert "actor_rollout_ref.actor.ulysses_sequence_parallel_size=2" in cmd
    assert "actor_rollout_ref.model.use_remove_padding=True" in cmd
    assert (
        "+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2"
        in cmd
    )
    assert not any(c.endswith("=sdpa") for c in cmd)


def test_rollout_n_override_flows_through(tmp_path):
    cmd = _cmd(tmp_path, ", rollout_n: 4")
    assert "actor_rollout_ref.rollout.n=4" in cmd


def test_batched_tokens_override_flows_through(tmp_path):
    cmd = _cmd(tmp_path, ", rollout_max_num_batched_tokens: 4096")
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=4096" in cmd
