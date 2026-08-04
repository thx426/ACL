"""build_command — the yaml->verl translation must carry the mechanism
switches faithfully: rollout_n and the steering→max_logprobs linkage always,
and the ulysses_sp>1 three-way linkage (SP size + use_remove_padding=True +
flash_attention_2) versus the sdpa profile at sp=1. PGS: no distillation
overrides may appear anywhere in the command.
"""
import importlib.util
from pathlib import Path

from trace_opd.common.config import load_config

_LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "launch_train.py"
_spec = importlib.util.spec_from_file_location("launch_train", _LAUNCHER)
launch_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch_train)

YAML = """
run: {name: t, runs_root: runs}
alfworld: {data_root: /d, worker_python: /p}
train: {model_path: /m, train_file: /t.parquet, val_files: [/v1.parquet]%s}
eval: {}
%s
"""


def _cmd(tmp_path, train_extra="", steering_extra=""):
    path = tmp_path / "c.yaml"
    path.write_text(YAML % (train_extra, steering_extra))
    cfg = load_config(path)
    return launch_train.build_command(cfg, tmp_path / "run")


def test_default_profile_and_rollout_n(tmp_path):
    cmd = _cmd(tmp_path)
    assert "actor_rollout_ref.rollout.n=8" in cmd
    assert "actor_rollout_ref.model.use_remove_padding=False" in cmd
    assert "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" in cmd
    assert not any("ulysses_sequence_parallel_size" in c for c in cmd)
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=2" in cmd
    # PGS 默认 grpo_loo(规范 §2.6 条件 iv);对照臂经 train.adv_estimator=grpo
    assert "algorithm.adv_estimator=grpo_loo" in cmd
    # Collapse-watch entropy curve (metric only, entropy_coeff stays 0).
    assert "actor_rollout_ref.actor.calculate_entropy=True" in cmd
    # Micro-batch packing budgets decoupled from the 34816 capacity declaration
    # (feedback P0: 34816 fed 21.2 GiB fp32 logits into both phases).
    assert "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192" in cmd
    assert not any(c.endswith("token_len_per_gpu=34816") for c in cmd)


def test_pgs_loss_surface_is_plain_grpo(tmp_path):
    # PGS: the loss is plain GRPO — a distillation KEY in the launch command
    # means dead config resurrected somewhere. (Match on the key side only:
    # tmp_path values may contain the word via the test name.)
    cmd = _cmd(tmp_path)
    assert not any("distillation" in c.split("=", 1)[0] for c in cmd)


def test_steering_lifts_vllm_max_logprobs(tmp_path):
    # SPS verify needs per-position top-K prompt_logprobs; vLLM's max_logprobs
    # is a LAUNCH-time parameter and must be >= steering.topk.
    cmd = _cmd(
        tmp_path,
        steering_extra=(
            "steering: {enabled: true, doc_table: /tmp/d.jsonl, theta_d: 0.8, topk: 64}"
        ),
    )
    # 联动源是 obs_topk(观测窗)而非 topk —— hook 的 draft logprobs=obs_topk
    # 与 verify prompt_logprobs=obs_topk 都要过 vLLM 校验(codex 审阅 P0)。
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs=256" in cmd
    # off by default: no max_logprobs key at all
    cmd0 = _cmd(tmp_path)
    assert not any("max_logprobs" in c for c in cmd0)


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
