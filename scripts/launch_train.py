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
import asyncio
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

# 自定位(实验反馈 P1:旧式 REPO/"trace_opd" 假设 worktree 目录名,部署改名
# 会静默加载别的树的 verl/src——最坏一类故障)。ROOT = 脚本所在树的根,
# worktree 叫什么名字都对。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace_opd.common.config import (  # noqa: E402
    Config,
    load_config,
    objective_fingerprint,
    template_drift,
)
from trace_opd.common.llm import OpenAICompatClient  # noqa: E402
from trace_opd.research.toolproto import make_tool_protocol  # noqa: E402


_PREFLIGHT_TOOL = {
    "type": "function",
    "function": {
        "name": "trace_preflight_echo",
        "description": "Return the provided nonce unchanged.",
        "parameters": {
            "type": "object",
            "properties": {"nonce": {"type": "string"}},
            "required": ["nonce"],
            "additionalProperties": False,
        },
    },
}
_PREFLIGHT_NONCE = "trace-researcher-fncall-v1"


def _visible_thinking(message: dict) -> str:
    """返回 formal researcher 响应里可见的 thinking 证据；空串表示干净。"""
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if value not in (None, ""):
            return f"non-empty {key}"
    content = str(message.get("content") or "")
    if "<think>" in content.lower() or "</think>" in content.lower():
        return "visible <think> block"
    return ""


async def _preflight_researcher_protocol(endpoint: str, role) -> None:
    """在一个 endpoint 上真实验证 non-thinking OpenAI fncall 两轮往返。"""
    client = OpenAICompatClient(
        role="researcher-preflight",
        endpoint=endpoint,
        model=role.model,
        api_key=role.api_key,
        max_concurrency=1,
        temperature=0.0,
        max_tokens=256,
        timeout_s=min(float(role.timeout_s), 120.0),
        max_retries=0,
        extra_body=role.extra_body,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "This is a transport preflight. Do not reason aloud. Call "
                "trace_preflight_echo exactly once with nonce "
                + _PREFLIGHT_NONCE
                + ". After its result arrives, respond exactly PREFLIGHT_OK."
            ),
        },
        {"role": "user", "content": "Perform the required tool call now."},
    ]
    try:
        first = await client.chat(messages, tools=[_PREFLIGHT_TOOL])
        thinking = _visible_thinking(first.raw_message)
        if thinking:
            raise RuntimeError(f"thinking mode is not disabled ({thinking})")
        if len(first.tool_calls) != 1:
            raise RuntimeError(
                f"expected exactly one tool call, got {len(first.tool_calls)}"
            )
        call = first.tool_calls[0]
        if call.name != "trace_preflight_echo" or call.arguments != {
            "nonce": _PREFLIGHT_NONCE
        }:
            raise RuntimeError(
                f"malformed tool call name/arguments: {call.name!r} {call.arguments!r}"
            )
        protocol = make_tool_protocol("fncall")
        assistant = protocol.canonicalize_assistant(first)
        messages.extend(
            [
                assistant,
                protocol.tool_result_message(
                    call,
                    json.dumps(
                        {"nonce": _PREFLIGHT_NONCE}, separators=(",", ":")
                    ),
                ),
            ]
        )
        second = await client.chat(messages, tools=[_PREFLIGHT_TOOL], max_tokens=64)
        thinking = _visible_thinking(second.raw_message)
        if thinking:
            raise RuntimeError(f"thinking mode is not disabled ({thinking})")
        if second.tool_calls:
            raise RuntimeError("follow-up unexpectedly requested another tool call")
        if second.text.strip() != "PREFLIGHT_OK":
            raise RuntimeError(
                f"tool-result follow-up is not exact PREFLIGHT_OK: {second.text[:120]!r}"
            )
    finally:
        await client.aclose()


def preflight_researcher(cfg: Config) -> None:
    """真实启动前逐 endpoint 核验认证、模型身份与 non-thinking fncall 往返。"""
    if cfg.research_rl.enabled:
        return  # v4 自体研究员:engine-backed,无 external endpoint 可查
    if not (
        cfg.steering.enabled
        and cfg.steering.doc_source == "researcher"
        and cfg.research.enabled
    ):
        return
    role = cfg.role("researcher")
    if not role.api_key_env:
        raise SystemExit("FATAL: roles.researcher.api_key_env is empty")
    if not role.api_key:
        raise SystemExit(
            f"FATAL: researcher API key environment variable {role.api_key_env!r} is unset or empty"
        )
    endpoints = [e.strip() for e in role.endpoint.split(",") if e.strip()]
    healthy: list[str] = []
    errors: list[str] = []
    for endpoint in endpoints:
        url = endpoint.rstrip("/") + "/models"
        request = urllib.request.Request(url)
        if role.api_key:
            request.add_header("Authorization", f"Bearer {role.api_key}")
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
            served = {str(row.get("id")) for row in payload.get("data", [])}
            if role.model not in served:
                errors.append(f"{url}: model {role.model!r} not served ({sorted(served)})")
                continue
            asyncio.run(_preflight_researcher_protocol(endpoint, role))
            healthy.append(endpoint)
        except Exception as err:  # noqa: BLE001——汇总所有 endpoint 便于一次修齐
            errors.append(f"{url}: {err}")
    for message in errors:
        print(f"[preflight] WARNING researcher endpoint unhealthy — {message}")
    # external_pool 的并发容量是实验合同；真实启动要求 N/N 全健康，
    # 不能以 failover 为借口静默少一半 producer 吞吐。
    if len(healthy) != len(endpoints):
        raise SystemExit(
            f"FATAL: researcher endpoint pool is not fully healthy "
            f"({len(healthy)}/{len(endpoints)}): " + " | ".join(errors)
        )
    print(
        f"[preflight] researcher OK: {role.model} @ "
        f"{len(healthy)}/{len(endpoints)} endpoint(s) healthy: {', '.join(healthy)}"
    )


def preflight_static_doc_table(path: str) -> None:
    """正式启动前核验静态 doc 表存在、非空且每行满足 task_id/doc 合同。"""
    table = Path(path)
    if not table.is_file():
        raise SystemExit(f"FATAL: steering.doc_table {table} does not exist")
    rows = 0
    try:
        with table.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or "task_id" not in row:
                    raise ValueError(f"line {line_no}: missing task_id")
                if "doc" not in row:
                    raise ValueError(f"line {line_no}: missing doc")
                rows += 1
    except Exception as err:  # noqa: BLE001——坏 JSON/编码/缺键均启动前拒绝
        raise SystemExit(f"FATAL: invalid steering.doc_table {table}: {err}") from err
    if rows == 0:
        raise SystemExit(f"FATAL: steering.doc_table {table} is empty")


def build_command(cfg: Config, run_dir: Path) -> list[str]:
    t = cfg.train
    if not t.model_path or not t.train_file or not t.val_files:
        raise SystemExit("train.model_path / train.train_file / train.val_files are required")
    agent_cfg = ROOT / "train" / "agent_loop_config.yaml"
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
    if t.rollout_backend == "sglang" and t.rollout_attention_backend:
        steering_args.append(
            "+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="
            f"{t.rollout_attention_backend}"
        )
    if cfg.steering.enabled and t.rollout_backend == "vllm":
        # 同源于 hook 的观测窗(steering.obs_topk):draft 的 logprobs=K 与
        # verify 的 prompt_logprobs=K 都要过 vLLM 的 max_logprobs 校验。
        # SGLang 无此启动参数(top_logprobs_num 逐请求),后端为 sglang 时跳过。
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
        # v3.3 M1/M2(asym advantage 的 α 调度三参;legacy 档取默认值时
        # trainer 分支不读它们,行为零漂移)。
        f"+algorithm.steered_adv_weight={cfg.steering.steered_adv_weight}",
        f"+algorithm.steered_adv_decay_steps={cfg.steering.steered_adv_decay_steps}",
        f"+algorithm.steered_adv_cap={cfg.steering.steered_adv_cap}",
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
        # rollout 引擎按配置选(vllm | sglang;sglang 侧 verl 自动映射
        # mem_fraction_static←gpu_memory_utilization、max_running_requests←max_num_seqs)
        f"actor_rollout_ref.rollout.name={t.rollout_backend}",
        "actor_rollout_ref.rollout.mode=async",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={t.rollout_tp}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={t.rollout_gpu_mem}",
        # n student rollouts per task (one GRPO group); verl repeats the batch
        # and scales ppo_mini_batch_size by n internally.
        # v4 自体研究员:每 prompt 行数 = executor rollout_n + research_n
        # (三段布局 steering/arms.role_of;research 行独立 GRPO 组)
        f"actor_rollout_ref.rollout.n={t.rollout_n + (cfg.research_rl.research_n if cfg.research_rl.enabled else 0)}",
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
    template_path = ROOT / "configs" / "pgs.yaml"
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
        runs_root = ROOT / runs_root
    run_dir = runs_root / cfg.run.name
    fingerprint = objective_fingerprint(cfg)

    # 改 objective 的 Hydra passthrough 无论 dry-run/正式都必须拒绝；科学键只走 --set。
    objective_prefixes = ("algorithm.",)
    objective_substrings = (
        "loss_agg_mode", "adv_estimator", "optim.lr", "ppo_epochs",
        "ppo_mini_batch_size", "train_files",
    )
    for item in passthrough:
        key = item.split("=", 1)[0].lstrip("+")
        if key.startswith(objective_prefixes) or any(sub in key for sub in objective_substrings):
            raise SystemExit(
                f"FATAL: passthrough override {item!r} changes the optimisation "
                "objective but bypasses the fingerprint — use --set (config-level) instead"
            )

    # dry-run 是真正无副作用模式：严验配置、算指纹/命令并打印；不网络、不写盘。
    if args.dry_run:
        cmd = build_command(cfg, run_dir) + list(passthrough)
        print(f"[launch] objective fingerprint: {fingerprint}")
        print(f"[launch] dry-run run_dir={run_dir}")
        print("[launch] " + " ".join(shlex.quote(c) for c in cmd))
        return 0

    # 正式依赖门必须先于 mkdir/effective/fingerprint，失败不污染 run_dir。
    preflight_researcher(cfg)
    if cfg.steering.enabled and cfg.steering.doc_source == "static":
        preflight_static_doc_table(cfg.steering.doc_table)

    run_dir.mkdir(parents=True, exist_ok=True)

    # Resolve template + profiles + overrides ONCE, write the result, and point
    # every consumer at that file (two readers resolving "the config"
    # differently is the same class of failure as an env var shadowing a file).
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

    # The vendored verl is prepended to PYTHONPATH so the run imports EXACTLY
    # the patched code shipped with this repo.
    vendored_verl = ROOT / "verl"
    verl_path = ""
    if (vendored_verl / "verl" / "__init__.py").exists():
        verl_path = str(vendored_verl)
        print(f"[launch] verl from: {verl_path} (vendored)")
    else:
        print("[launch] WARNING: vendored verl missing — falling back to the venv's verl")

    env = dict(os.environ)
    pythonpath = ":".join(
        p for p in (verl_path, str(ROOT / "src"), env.get("PYTHONPATH", "")) if p
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
    # 勿设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True:与 vLLM
    # CuMemAllocator(sleep/wake 内存池)不兼容,worker 启动即 assert 拒绝
    # (pytorch#147851;pgs_base 首启实测)。显存防御仅靠 rollout_gpu_mem。
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
        "doc_source": cfg.steering.doc_source,
        "research_contract": (
            {
                "backend": cfg.research.backend,
                "tool_protocol": cfg.research.tool_protocol,
                "model": cfg.role("researcher").model,
                "endpoint": cfg.role("researcher").endpoint,
                "n_exp": cfg.research.n_exp,
                "k_default": cfg.research.k_default,
                "k_max": cfg.research.k_max,
                "max_turns": cfg.research.max_turns,
                "session_timeout_s": cfg.research.session_timeout_s,
                "thinking": cfg.role("researcher").extra_body.get(
                    "chat_template_kwargs", {}
                ).get("enable_thinking"),
            }
            if cfg.steering.doc_source == "researcher" else None
        ),
        "verl_path": verl_path or "venv",
    }
    manifest_path = run_dir / f"launch_manifest_{time.strftime('%Y%m%d_%H%M%S')}.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"[launch] run_dir={run_dir}")
    print(f"[launch] manifest={manifest_path}")
    print("[launch] " + " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
