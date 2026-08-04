#!/usr/bin/env python3
"""Launch the PGS training run — every verl CLI argument is generated from
configs/pgs.yaml plus a fixed stable-profile block (4B/A800 mainline lineage:
dynamic batching, fsdp2 with offload; padding/attention follow train.ulysses_sp
— sp=1 keeps the sdpa profile, sp>1 flips to rmpad + FA2). The resolved
command, config snapshot, and any manual overrides are written to a launch
manifest under the run dir — the run must be auditable.

Loss surface (docs/pgs-final-spec.md §2.6): plain verl GRPO (native ppo_loss).
No distillation config exists any more — the teacher enters ONLY through the
sampling-side steering hook (SPS), and its behaviour correction W_beh rides
the batch's ``rollout_is_weights`` channel into the native vanilla loss.

Checkpoints are bound to the run dir (trainer.default_local_dir), so
resume_mode=auto and rollout dumps live and resume together.

Required environment:
  ALFWORLD_DATA / ALFWORLD_WORKER_PYTHON     consumed via the yaml ${VARS}
  TRACE_MODEL_PATH / TRACE_TRAIN_FILE / TRACE_VAL_*_FILE      ditto
  SWANLAB_API_KEY (when swanlab logs online; otherwise SWANLAB_MODE=offline is set)

Usage (server, training venv):
  python trace_opd/scripts/launch_train.py --config trace_opd/configs/pgs.yaml \
      [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
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


def build_command(cfg: Config, run_dir: Path) -> list[str]:
    t = cfg.train
    if not t.model_path or not t.train_file or not t.val_files:
        raise SystemExit("train.model_path / train.train_file / train.val_files are required")
    agent_cfg = REPO / "trace_opd" / "train" / "agent_loop_config.yaml"
    val_files = "[" + ",".join(f"'{v}'" for v in t.val_files) + "]"
    loggers = "[" + ",".join(f'"{x}"' for x in t.loggers) + "]"
    # Ulysses SP: sp>1 requires packed THD sequences, which in verl means
    # use_remove_padding=True and the flash-attention forward the ulysses
    # monkey patch hooks into. sp=1 keeps the exact sdpa profile.
    if t.ulysses_sp > 1:
        rmpad, attn_impl = "True", "flash_attention_2"
        print(
            f"[launch] ulysses_sp={t.ulysses_sp}: use_remove_padding=True + "
            "flash_attention_2 (needs a working flash-attn install)"
        )
    else:
        rmpad, attn_impl = "False", "sdpa"
    sp_args = (
        [f"actor_rollout_ref.actor.ulysses_sequence_parallel_size={t.ulysses_sp}"]
        if t.ulysses_sp > 1
        else []
    )
    # SPS verify 需要 per-position top-K prompt_logprobs;vLLM 的 max_logprobs
    # 是启动期参数(默认 20),必须 >= steering.topk,否则 verify 请求在运行
    # 中段被拒(P4 冒烟首查项,docs/pgs-impl-design.md)。
    steering_args = []
    if cfg.steering.enabled:
        # 同源于 hook 的观测窗(steering.obs_topk):draft 的 logprobs=K 与
        # verify 的 prompt_logprobs=K 都要过 vLLM 的 max_logprobs 校验。
        max_logprobs = max(int(cfg.steering.obs_topk), 20)
        steering_args = [
            f"+actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs={max_logprobs}",
        ]
    return [
        sys.executable, "-m", "verl.trainer.main_ppo",
        f"algorithm.adv_estimator={t.adv_estimator}",
        # 审阅 IF-04(方案 A):默认关组内 std 归一化(Dr.GRPO 形态,scale≡1
        # 与样本无关,严格满足条件 iv);开=有偏稳定化消融档。
        f"algorithm.norm_adv_by_std_in_grpo={t.norm_adv_by_std}",
        "algorithm.use_kl_in_reward=False",
        f"data.train_files=['{t.train_file}']",
        f"data.val_files={val_files}",
        f"data.train_batch_size={t.train_batch_size}",
        "data.shuffle=True",
        f"data.max_prompt_length={t.max_prompt_length}",
        f"data.max_response_length={t.response_length}",
        "data.return_raw_chat=True",
        # No silent input filtering: an overlong prompt must fail loudly via
        # truncation=error, not vanish from the task pool.
        "data.filter_overlong_prompts=False",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={t.model_path}",
        f"actor_rollout_ref.model.use_remove_padding={rmpad}",
        f"+actor_rollout_ref.model.override_config.attn_implementation={attn_impl}",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        *sp_args,
        "actor_rollout_ref.actor.use_kl_loss=False",
        # Entropy curve for the collapse watch. Metric only — entropy_coeff
        # stays 0 so the loss is untouched; cost lands in the TRAINING phase.
        "actor_rollout_ref.actor.calculate_entropy=True",
        f"actor_rollout_ref.actor.optim.lr={t.lr}",
        f"actor_rollout_ref.actor.optim.lr_scheduler_type={t.lr_scheduler_type}",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps_ratio={t.lr_warmup_steps_ratio}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={t.ppo_mini_batch_size}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        # Micro-batch PACKING budgets, decoupled from max_token_len_per_gpu:
        # the train-forward logits transient is budget x vocab x 4B fp32.
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={t.ppo_max_token_len_per_gpu}",
        # 信用分配单位(seq-mean-token-mean = 轨迹等权,2026-07-29 用户裁定)。
        f"actor_rollout_ref.actor.loss_agg_mode={t.loss_agg_mode}",
        f"actor_rollout_ref.actor.strategy={t.fsdp_strategy}",
        f"actor_rollout_ref.ref.strategy={t.fsdp_strategy}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={t.param_offload}",
        f"actor_rollout_ref.actor.fsdp_config.reshard_after_forward={t.reshard_after_forward}",
        *([f"actor_rollout_ref.actor.fsdp_config.fsdp_size={t.fsdp_size}"] if t.fsdp_size else []),
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=True",
        # hf_model 保存关闭(rank0 全量 gather + 单线程写 8GB 是保存耗时大头);
        # 需要 HF 格式时用 `python -m verl.model_merger` 离线转。
        'actor_rollout_ref.actor.checkpoint.save_contents=["model","optimizer","extra"]',
        # 滚动保留的 ckpt 数走配置(硬编码会在 manifest 里说谎)。
        f"trainer.max_actor_ckpt_to_keep={t.max_ckpt_to_keep}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={t.rollout_tp}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={t.rollout_gpu_mem}",
        # n student rollouts per task (one GRPO group); verl repeats the batch
        # and scales ppo_mini_batch_size by n internally.
        f"actor_rollout_ref.rollout.n={t.rollout_n}",
        f"actor_rollout_ref.rollout.max_model_len={t.rollout_max_model_len}",
        # Chunked-prefill budget; also sizes the transient full-vocab logits
        # tensor vLLM materializes per chunk during prompt_logprobs scoring.
        f"actor_rollout_ref.rollout.max_num_batched_tokens={t.rollout_max_num_batched_tokens}",
        f"actor_rollout_ref.rollout.max_num_seqs={t.rollout_max_num_seqs}",
        *steering_args,
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
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.environ.get("TRACE_CONFIG", ""))
    parser.add_argument("--run-name", default="",
                        help="override run.name (e.g. smoke runs get their own dir)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="section.key=value",
        help="override a config setting for this run, e.g. "
             "--set steering.eta_max=0.3 (repeatable). Goes through the same "
             "validation as the file and is recorded in the launch manifest.",
    )
    args, passthrough = parser.parse_known_args()
    if not args.config:
        raise SystemExit("--config (or TRACE_CONFIG) is required")
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path, overrides=args.overrides)
    # Each experiment owns a complete copy of the template, so a copy can fall
    # behind a mechanism change. Say so at launch.
    template_path = REPO / "trace_opd" / "configs" / "pgs.yaml"
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
    # every consumer at that file (two readers resolving "the config"
    # differently is the same class of failure as an env var shadowing a file).
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
    # experiment's checkpoint under another's objective.
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

    # Steering 预检:doc 表必须在场且非空(缺表启动 = 全程纯 bare,静默跑穿)。
    if cfg.steering.enabled and not args.dry_run:
        table = Path(cfg.steering.doc_table)
        if not table.is_file():
            raise SystemExit(f"FATAL: steering.doc_table {table} does not exist")

    # The vendored verl is prepended to PYTHONPATH so the run imports EXACTLY
    # the patched code shipped with this repo.
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
    # there. The CONFIG is the source of truth; these are transport.
    env["TRACE_AGENT_TASK_TIMEOUT_S"] = str(cfg.train.agent_task_timeout_s)
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
        f"[launch] fsdp_mode = {cfg.train.fsdp_mode}"
        f" | dist_backend = {cfg.train.dist_backend or 'verl default'}"
        f" | steering = {'ON' if cfg.steering.enabled else 'off'}"
    )
    if "swanlab" in cfg.train.loggers and not env.get("SWANLAB_API_KEY") and not env.get("SWANLAB_MODE"):
        print("[launch] SWANLAB_API_KEY not set — falling back to SWANLAB_MODE=offline")
        env["SWANLAB_MODE"] = "offline"

    # 审阅 IF-10:passthrough 在 fingerprint 计算之后追加——改变优化目标的
    # 键从这里溜进去会绕过 resume 守卫(同一 run 换 objective 续训)。这类
    # 键必须走 --set(参与校验与指纹),不许走 passthrough。
    _OBJECTIVE_PREFIXES = ("algorithm.",)
    _OBJECTIVE_SUBSTRINGS = (
        "loss_agg_mode", "adv_estimator", "optim.lr", "ppo_epochs",
        "ppo_mini_batch_size", "train_files",
    )
    for item in passthrough:
        key = item.split("=", 1)[0].lstrip("+")
        if key.startswith(_OBJECTIVE_PREFIXES) or any(sub in key for sub in _OBJECTIVE_SUBSTRINGS):
            raise SystemExit(
                f"FATAL: passthrough override {item!r} changes the optimisation "
                "objective but bypasses the fingerprint — use --set (config-level) instead"
            )
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
