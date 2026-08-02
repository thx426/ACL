#!/usr/bin/env python3
"""Launch the TRACE-OPD training run — every verl CLI argument is generated
from configs/trace.yaml plus a fixed stable-profile block (CD-003 full-data
run: dynamic batching, fsdp2 with offload, chunked top-k distillation,
self-teacher with 0 dedicated GPUs; padding/attention follow train.ulysses_sp
— sp=1 keeps h3's remove_padding=False + sdpa, sp>1 flips to rmpad + FA2).
The resolved command, config snapshot, and any manual overrides are written
to a launch manifest under the run dir — the run must be auditable.

Startup discipline (review P0-3/P1-8): systemic misconfiguration dies HERE —
the external researcher endpoint is pre-flighted (/models must answer and
serve the configured model) before any GPU is touched.

Checkpoints are bound to the run dir (trainer.default_local_dir), so
resume_mode=auto, research archives, and rollout dumps all live and resume
together (review P2-13).

Required environment:
  ALFWORLD_DATA / ALFWORLD_WORKER_PYTHON     consumed via the yaml ${VARS}
  TRACE_MODEL_PATH / TRACE_TRAIN_FILE / TRACE_VAL_*_FILE      ditto
  TRACE_RESEARCHER_ENDPOINT / TRACE_RESEARCHER_MODEL          ditto
  SWANLAB_API_KEY (when swanlab logs online; otherwise SWANLAB_MODE=offline is set)

Usage (server, training venv):
  python trace_opd/scripts/launch_train.py --config trace_opd/configs/trace.yaml \
      [--dry-run] [--skip-preflight]
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "trace_opd" / "src"))

from trace_opd.common.config import (  # noqa: E402
    Config,
    load_config,
    objective_fingerprint,
    template_drift,
)


def preflight_researcher(cfg: Config) -> None:
    """Every pool endpoint must answer /models and serve the configured model —
    otherwise research sessions would silently degrade to doc=None and the run
    would train on near-zero self-KL (review P1-8). Pool mode (comma-separated
    endpoint): >=1 healthy endpoint lets the run proceed (PooledClient fails
    over per call); 0 healthy is FATAL. Auth header included (remote pools)."""
    if not (cfg.research.enabled and cfg.research.backend == "external"):
        return
    role = cfg.role("researcher")
    if not role.endpoint:
        raise SystemExit("FATAL: roles.researcher.endpoint is empty (env var unset?)")
    endpoints = [e.strip() for e in role.endpoint.split(",") if e.strip()]
    healthy: list[str] = []
    errors: list[str] = []
    for ep in endpoints:
        url = ep.rstrip("/") + "/models"
        req = urllib.request.Request(url)
        if role.api_key:
            req.add_header("Authorization", f"Bearer {role.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            served = {m.get("id") for m in data.get("data", [])}
            if role.model not in served:
                errors.append(f"{url}: model {role.model!r} not served ({sorted(served)})")
                continue
            healthy.append(ep)
        except Exception as err:  # noqa: BLE001
            errors.append(f"{url}: {err}")
    for msg in errors:
        print(f"[preflight] WARNING researcher endpoint unhealthy — {msg}")
    if not healthy:
        raise SystemExit(
            f"FATAL: no healthy researcher endpoint (0/{len(endpoints)}): "
            + " | ".join(errors)
        )
    print(
        f"[preflight] researcher OK: {role.model} @ "
        f"{len(healthy)}/{len(endpoints)} endpoint(s) healthy: {', '.join(healthy)}"
    )

    if role.sidecar_sleep:
        root = sidecar_root(role.endpoint)
        probe = root + "/is_sleeping"
        try:
            with urllib.request.urlopen(probe, timeout=15) as resp:
                state = json.loads(resp.read().decode("utf-8"))
        except Exception as err:  # noqa: BLE001
            raise SystemExit(
                f"FATAL: sidecar_sleep=true but {probe} unreachable ({err}) — "
                "the researcher server must run with --enable-sleep-mode and "
                "VLLM_SERVER_DEV_MODE=1 (scripts/serve_researcher.sh)"
            ) from err
        print(f"[preflight] sidecar sleep endpoints OK ({state})")


def sidecar_root(endpoint: str) -> str:
    """vLLM dev-mode control endpoints (/sleep /wake_up /is_sleeping) live at
    the server ROOT, not under /v1."""
    root = endpoint.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


def build_command(cfg: Config, run_dir: Path) -> list[str]:
    t = cfg.train
    if not t.model_path or not t.train_file or not t.val_files:
        raise SystemExit("train.model_path / train.train_file / train.val_files are required")
    agent_cfg = REPO / "trace_opd" / "train" / "agent_loop_config.yaml"
    val_files = "[" + ",".join(f"'{v}'" for v in t.val_files) + "]"
    loggers = "[" + ",".join(f'"{x}"' for x in t.loggers) + "]"
    # Ulysses SP (design §6.1): sp>1 requires packed THD sequences, which in
    # verl means use_remove_padding=True and the flash-attention forward the
    # ulysses monkey patch hooks into. sp=1 keeps the exact h3 profile
    # (rmpad=False + sdpa). Math-equivalent either way; implementation differs.
    if t.ulysses_sp > 1:
        rmpad, attn_impl = "True", "flash_attention_2"
        print(
            f"[launch] ulysses_sp={t.ulysses_sp}: use_remove_padding=True + "
            "flash_attention_2 (departs from the h3 sdpa profile; needs a "
            "working flash-attn install in the training venv)"
        )
    else:
        rmpad, attn_impl = "False", "sdpa"
    sp_args = (
        [f"actor_rollout_ref.actor.ulysses_sequence_parallel_size={t.ulysses_sp}"]
        if t.ulysses_sp > 1
        else []
    )
    # OPD 项的散度形式(config §opd_kl_mode → verl loss_mode;设计文档 §2.7)。
    # top-k 两模式带 top-K 数据面参数;estimator 两模式打分自动降为单点
    # (rollout 侧由注册表 use_topk=False 推出 prompt_logprobs=0/teacher_topk=0),
    # chunked-topk 参数不再传——留着只会在 effective config 里当惰性噪声。
    kl_mode_verl = {
        "forward_kl_topk": "forward_kl_topk",
        "reverse_kl_k3": "k3",
        "reverse_kl_topk": "reverse_kl_topk",
        "reverse_kl_gated": "reverse_kl_gated",
    }[t.opd_kl_mode]
    kl_args = [f"distillation.distillation_loss.loss_mode={kl_mode_verl}"]
    if t.opd_kl_mode in ("forward_kl_topk", "reverse_kl_topk"):
        kl_args += [
            f"distillation.distillation_loss.topk={t.distill_topk}",
            "+distillation.distillation_loss.use_chunked_topk=True",
            "+distillation.distillation_loss.chunked_topk_chunk_size=2048",
        ]
    if t.opd_kl_mode == "reverse_kl_gated":
        kl_args.append(f"+distillation.distillation_loss.gate_beta={t.opd_gate_beta}")
    # Self-research channel (docs/self-research-design.md): its coefficients
    # ride the same distillation_loss config surface. Off (weight=0, default)
    # ships nothing — no config keys, no extra replicas, byte-identical run.
    rrl = cfg.research_rl
    research_args = []
    research_extra_n = 0
    if rrl.weight > 0:
        research_args = [
            f"+distillation.distillation_loss.research_loss_coef={rrl.weight}",
            f"+distillation.distillation_loss.research_update_mode={rrl.update_mode}",
            f"+distillation.distillation_loss.research_adv_clip={rrl.adv_clip}",
            f"+distillation.distillation_loss.research_warmup_steps={rrl.warmup_steps}",
        ]
        research_extra_n = rrl.n_docs
        # The separate mode (and the warmup drop in either mode) splits the
        # batch by role before update_actor; TransferQueue dispatches rows
        # across the dp ranks, so each role's row count must divide n_gpus or
        # a rank ends short. Fixed shapes make this a launch-time check.
        n_research_rows = t.train_batch_size * rrl.n_docs
        n_exec_rows = t.train_batch_size * (
            t.rollout_n + (t.mix_sft_rollouts if t.mix_sft_weight > 0 else 0)
        )
        for label, rows in (("research", n_research_rows), ("executor", n_exec_rows)):
            if rows % t.n_gpus != 0:
                raise SystemExit(
                    f"FATAL: {label} row count {rows} (batch {t.train_batch_size}) is not "
                    f"divisible by n_gpus={t.n_gpus} — the role-split update would leave "
                    "a dp rank short. Adjust research_rl.n_docs / rollout_n / batch size."
                )
    return [
        sys.executable, "-m", "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        f"data.train_files=['{t.train_file}']",
        f"data.val_files={val_files}",
        f"data.train_batch_size={t.train_batch_size}",
        "data.shuffle=True",
        f"data.max_prompt_length={t.max_prompt_length}",
        f"data.max_response_length={t.response_length}",
        "data.return_raw_chat=True",
        # No silent input filtering (design §2): an overlong prompt must fail
        # loudly via truncation=error, not vanish from the task pool.
        "data.filter_overlong_prompts=False",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={t.model_path}",
        f"actor_rollout_ref.model.use_remove_padding={rmpad}",
        f"+actor_rollout_ref.model.override_config.attn_implementation={attn_impl}",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        *sp_args,
        "actor_rollout_ref.actor.use_kl_loss=False",
        # Entropy curve for the collapse watch (PG-4/5 stop-gate signals; h3's
        # entropy-inflation finding is the reference). Metric only —
        # entropy_coeff stays 0 so the loss is untouched. Cost lands in the
        # TRAINING phase (both engines asleep, ~76GB/card free) and the
        # chunked/checkpointed entropy paths below keep it cheap; it does NOT
        # touch the rollout-phase OOM budget.
        "actor_rollout_ref.actor.calculate_entropy=True",
        f"actor_rollout_ref.actor.optim.lr={t.lr}",
        # Explicit schedule = h3's ACTUAL behavior (verl defaults, pinned so a
        # verl default change cannot silently alter the profile).
        f"actor_rollout_ref.actor.optim.lr_scheduler_type={t.lr_scheduler_type}",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps_ratio={t.lr_warmup_steps_ratio}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={t.ppo_mini_batch_size}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        # Micro-batch PACKING budgets, decoupled from max_token_len_per_gpu
        # (feedback P0 2026-07-23): the train-forward logits transient is
        # budget x vocab x 4B fp32 — 34816 meant 21.2 GiB and two OOMs.
        # A tail sample longer than the budget rides in its own micro-batch
        # (verl 6d77dc7 clamps instead of asserting).
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={t.ppo_max_token_len_per_gpu}",
        # 信用分配单位。蒸馏项也读它(distillation/losses.py 的 agg_loss 用
        # actor config 的 loss_agg_mode),所以纯 OPD 下这一项就是 OPD 的聚合口径。
        f"actor_rollout_ref.actor.loss_agg_mode={t.loss_agg_mode}",
        f"actor_rollout_ref.actor.strategy={t.fsdp_strategy}",
        f"actor_rollout_ref.ref.strategy={t.fsdp_strategy}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={t.param_offload}",
        f"actor_rollout_ref.actor.fsdp_config.reshard_after_forward={t.reshard_after_forward}",
        *([f"actor_rollout_ref.actor.fsdp_config.fsdp_size={t.fsdp_size}"] if t.fsdp_size else []),
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=True",
        # CD-006 2026-07-25 用户裁定:去掉 hf_model。它需要 rank0 全量 gather + 单线程写 8GB,
        # 是 201GB/4分钟保存耗时的大头;需要 HF 格式时用 `python -m verl.model_merger` 离线转。
        'actor_rollout_ref.actor.checkpoint.save_contents=["model","optimizer","extra"]',
        # 滚动只留最近 3 个 ckpt(3x201GB≈603GB 恒定),避免 444 步吃掉 17.7TB。
        # PG-5 的峰值 ckpt 存档需另行 pin 到不受清理的目录(待口径)。
        "trainer.max_actor_ckpt_to_keep=3",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={t.rollout_tp}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={t.rollout_gpu_mem}",
        # n student rollouts per task, all scored under the SAME research doc;
        # verl repeats the batch (interleave) and scales ppo_mini_batch_size
        # by n internally, so mini_batch keeps its "tasks" unit.
        # Mixed objective: the SFT arm adds expert copies on top — the agent
        # loop hands the trailing mix_sft_rollouts indices of each task to the
        # document-guided branch, so the bare group stays exactly rollout_n.
        f"actor_rollout_ref.rollout.n="
        f"{t.rollout_n + (t.mix_sft_rollouts if t.mix_sft_weight > 0 else 0) + research_extra_n}",
        f"actor_rollout_ref.rollout.max_model_len={t.rollout_max_model_len}",
        # Chunked-prefill budget. Also sizes the transient full-vocab logits
        # tensor vLLM materializes per chunk during self-teacher prompt_logprobs
        # scoring (chunk x vocab x 2B, ~doubled by the TP all_gather) — the
        # smoke-run OOM root cause at the h3 value 4096 (feedback P0 2026-07-23).
        f"actor_rollout_ref.rollout.max_num_batched_tokens={t.rollout_max_num_batched_tokens}",
        f"+actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs={t.distill_topk}",
        f"actor_rollout_ref.rollout.prompt_length={t.max_prompt_length}",
        f"actor_rollout_ref.rollout.response_length={t.response_length}",
        f"actor_rollout_ref.rollout.temperature={t.rollout_temperature}",
        "actor_rollout_ref.rollout.calculate_log_probs=False",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={t.log_prob_max_token_len_per_gpu}",
        "actor_rollout_ref.rollout.val_kwargs.temperature=0",
        "actor_rollout_ref.rollout.val_kwargs.top_p=1.0",
        "actor_rollout_ref.rollout.val_kwargs.top_k=-1",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=False",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={agent_cfg}",
        f"actor_rollout_ref.rollout.agent.num_workers={t.agent_num_workers}",
        f"trainer.n_gpus_per_node={t.n_gpus}",
        "trainer.nnodes=1",
        f"trainer.logger={loggers}",
        f"trainer.project_name={t.project_name}",
        f"trainer.experiment_name={t.experiment_name}",
        f"trainer.default_local_dir={run_dir / 'checkpoints'}",
        f"trainer.val_before_train={t.val_before_train}",
        f"trainer.test_freq={t.test_freq}",
        f"trainer.save_freq={t.save_freq}",
        f"trainer.total_epochs={t.total_epochs}",
        f"trainer.resume_mode={t.resume_mode}",
        "distillation.enabled=True",
        "+distillation.self_teacher=True",
        "distillation.n_gpus_per_node=0",
        "distillation.nnodes=0",
        f"distillation.teacher_models.teacher_model.model_path={t.model_path}",
        *kl_args,
        "distillation.distillation_loss.use_policy_gradient=False",
        "distillation.distillation_loss.use_task_rewards=False",
        "distillation.distillation_loss.loss_max_clamp=10.0",
        "distillation.distillation_loss.log_prob_min_clamp=-10.0",
        # Mixed objective (docs/mixed-objective-design.md). 0/0 = pure OPD.
        f"+distillation.distillation_loss.opd_loss_coef={t.mix_opd_weight}",
        f"+distillation.distillation_loss.rl_loss_coef={t.mix_rl_weight}",
        f"+distillation.distillation_loss.sft_loss_coef={t.mix_sft_weight}",
        f"+distillation.distillation_loss.tg_loss_coef={t.mix_tg_weight}",
        *research_args,
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.environ.get("TRACE_CONFIG", ""))
    parser.add_argument("--run-name", default="",
                        help="override run.name (e.g. smoke runs get their own dir)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="section.key=value",
        help="override a config setting for this run, e.g. "
             "--set train.mix_rl_weight=1.0 (repeatable, applied after profiles). "
             "Goes through the same validation as the file and is recorded in the "
             "launch manifest.",
    )
    args, passthrough = parser.parse_known_args()
    if not args.config:
        raise SystemExit("--config (or TRACE_CONFIG) is required")
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path, overrides=args.overrides)
    # Each experiment owns a complete copy of the template, so a copy can fall
    # behind a mechanism change. Say so at launch: a key the template grew after
    # this copy was taken silently falls back to the code default, which is not
    # necessarily the value the template recommends.
    template_path = REPO / "trace_opd" / "configs" / "trace.yaml"
    if config_path != template_path and template_path.exists():
        drift = template_drift(config_path, template_path)
        for key in drift["missing"]:
            print(f"[launch] NOTE config is behind the template: {key} is missing "
                  "(falling back to the code default)")
        for key in drift["extra"]:
            print(f"[launch] NOTE config has {key}, which the template does not")
    if args.overrides:
        print(f"[launch] config overrides: {' '.join(args.overrides)}")
    if args.run_name:
        cfg.run.name = args.run_name
        cfg.train.experiment_name = args.run_name  # tracking name follows the run
        # Keep the raw snapshot in step with the dataclass — it is what gets
        # written out below and read back by every worker.
        cfg.raw.setdefault("run", {})["name"] = args.run_name
        cfg.raw.setdefault("train", {})["experiment_name"] = args.run_name

    runs_root = Path(cfg.run.runs_root)
    if not runs_root.is_absolute():
        runs_root = REPO / "trace_opd" / runs_root
    run_dir = runs_root / cfg.run.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Resolve template + profiles + overrides ONCE, write the result, and point
    # every consumer at that file. Before this, the agent-loop workers re-read
    # the template from TRACE_CONFIG and never saw --set: a run launched as
    # `--set train.mix_sft_weight=0.5` had the trainer expecting expert samples
    # while the workers produced none. Two readers of "the config" that resolve
    # it differently is the same class of failure as an env var shadowing a file.
    fingerprint = objective_fingerprint(cfg)
    effective_path = run_dir / "effective_config.yaml"
    header = (
        "# RESOLVED CONFIG — written by launch_train.py, read by every worker.\n"
        "# Do not edit: it is regenerated at each launch from\n"
        f"#   config   : {config_path}\n"
        + "".join(f"#   --set    : {o}\n" for o in args.overrides)
        + f"# objective fingerprint: {fingerprint}\n"
    )
    effective_path.write_text(
        header + yaml.safe_dump(cfg.raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    # Resuming into a run directory whose objective differs means continuing one
    # experiment's checkpoint under another's objective — every ablation arm
    # shares a default run name, so it is one forgotten flag away.
    stamp_path = run_dir / "objective_fingerprint.txt"
    if stamp_path.exists():
        previous = stamp_path.read_text(encoding="utf-8").strip()
        if previous != fingerprint and cfg.train.resume_mode != "disable":
            raise SystemExit(
                f"[launch] FATAL: {run_dir} was created with objective fingerprint "
                f"{previous}, this launch resolves to {fingerprint}. Resuming would "
                "continue one experiment's checkpoints under another's objective.\n"
                "         Use a different --run-name for this arm, or set "
                "train.resume_mode=disable if you really mean to restart here."
            )
    else:
        stamp_path.write_text(fingerprint + "\n", encoding="utf-8")
    print(f"[launch] objective fingerprint: {fingerprint}")

    if not args.dry_run and not args.skip_preflight:
        preflight_researcher(cfg)

    # The vendored verl (trace-opd branch export, trace_opd/verl/VENDORED.md)
    # is prepended to PYTHONPATH so the run imports EXACTLY the patched code
    # shipped with this repo — not whatever verl the venv happens to hold.
    vendored_verl = REPO / "trace_opd" / "verl"
    verl_path = ""
    if (vendored_verl / "verl" / "__init__.py").exists():
        verl_path = str(vendored_verl)
        print(f"[launch] verl from: {verl_path} (vendored)")
    else:
        print("[launch] WARNING: vendored verl missing — falling back to the venv's verl")

    env = dict(os.environ)
    pythonpath = ":".join(
        p for p in (verl_path, str(REPO / "trace_opd" / "src"), env.get("PYTHONPATH", "")) if p
    )
    env.update(
        {
            # The RESOLVED config, not the template: workers must see the same
            # profiles and --set overrides the trainer was launched with.
            "TRACE_CONFIG": str(effective_path),
            "TRACE_RUN_DIR": str(run_dir),
            "TRACE_AGENT_WORKERS": str(cfg.train.agent_num_workers),
            "PYTHONPATH": pythonpath,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",   # mixed-GPU host discipline
            "HYDRA_FULL_ERROR": "1",
        }
    )
    # Settings that reach verl through the environment because verl reads them
    # there. The CONFIG is the source of truth; these are transport, so they are
    # written unconditionally rather than defaulted with `${VAR:-...}` in a shell
    # script — one launch path forgetting the export is how the fsdp scoring
    # backend silently reverted to vLLM on the 162 line (CD-006 requirement 3-F).
    env["TRACE_TEACHER_SCORING_BACKEND"] = cfg.train.teacher_scoring_backend
    env["TRACE_AGENT_TASK_TIMEOUT_S"] = str(cfg.train.agent_task_timeout_s)
    if cfg.train.microscope_distributions:
        env["TRACE_OBS_DIR"] = str(run_dir / "obs")
        env["TRACE_OBS_EVERY"] = str(cfg.train.microscope_dist_every)
    env["VLLM_RPC_TIMEOUT"] = str(cfg.train.vllm_rpc_timeout_ms)
    if cfg.train.dist_backend:
        env["VERL_DIST_BACKEND"] = cfg.train.dist_backend
    else:
        env.pop("VERL_DIST_BACKEND", None)
    if cfg.train.fsdp_mode == "noshard":
        env["VERL_FSDP_NO_SHARD"] = "1"
    else:
        env.pop("VERL_FSDP_NO_SHARD", None)
    print(
        f"[launch] teacher scoring backend = {cfg.train.teacher_scoring_backend}"
        f" | fsdp_mode = {cfg.train.fsdp_mode}"
        f" | dist_backend = {cfg.train.dist_backend or 'verl default'}"
    )
    if cfg.train.distill_microscope:
        # Per-token distillation sampling (verl patch 6f3d089); forwarded to
        # ray workers via verl's call-time env whitelist.
        env["TRACE_DISTILL_MICROSCOPE_DIR"] = str(run_dir / "microscope")
        env["TRACE_DISTILL_SAMPLE_EVERY"] = str(cfg.train.distill_sample_every)
    if "swanlab" in cfg.train.loggers and not env.get("SWANLAB_API_KEY") and not env.get("SWANLAB_MODE"):
        print("[launch] SWANLAB_API_KEY not set — falling back to SWANLAB_MODE=offline")
        env["SWANLAB_MODE"] = "offline"

    # Single-machine time-slicing: hand the researcher server root to the verl
    # sidecar hook (CheckpointEngineManager sleeps/wakes it with the replicas).
    sidecar_url = ""
    if cfg.research.enabled and cfg.research.backend == "external":
        role = cfg.role("researcher")
        if role.sidecar_sleep:
            sidecar_url = sidecar_root(role.endpoint)
            env["TRACE_SIDECAR_URL"] = sidecar_url
            env["TRACE_SIDECAR_SLEEP_LEVEL"] = str(role.sidecar_sleep_level)
            print(f"[launch] sidecar sleep/wake enabled: {sidecar_url}")

    cmd = build_command(cfg, run_dir) + list(passthrough)
    if passthrough:
        print(f"[launch] WARNING: manual verl overrides appended: {passthrough}")

    manifest = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(config_path),
        "config_overrides": list(args.overrides),
        # The file every process actually read, and a hash of the settings that
        # define what this run optimises (resume checks it, see above).
        "effective_config": str(effective_path),
        "objective_fingerprint": fingerprint,
        "config": cfg.raw,
        "overrides": list(passthrough),
        "cmd": cmd,
        "run_dir": str(run_dir),
        "agent_workers": cfg.train.agent_num_workers,
        "sidecar_url": sidecar_url,
        "verl_path": verl_path or "venv",
    }
    manifest_path = run_dir / f"launch_manifest_{time.strftime('%Y%m%d_%H%M%S')}.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"[launch] run_dir={run_dir}")
    print(f"[launch] manifest={manifest_path}")
    print("[launch] " + " ".join(shlex.quote(c) for c in cmd))
    if args.dry_run:
        return 0
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
