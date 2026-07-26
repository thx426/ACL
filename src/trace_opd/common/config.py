"""Configuration loading — ONE yaml file is the single source of truth for
every tunable in the mechanism (configs/trace.yaml): research session,
narration, role connections, environment, training launcher, evaluation.

``${VARS}`` expand from the environment at load time (empty when unset;
clients fail loudly at connection time). Deployment facts (endpoints, data
roots, interpreter paths) live in env vars, not in the yaml.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


@dataclass
class RoleCfg:
    """Connection parameters for an ``external`` backend role."""

    endpoint: str = ""
    model: str = ""
    api_key: str = ""
    max_concurrency: int = 8
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout_s: float = 300.0
    extra_body: dict[str, Any] = field(default_factory=dict)
    # Single-machine time-slicing: the trainer drives this server through the
    # rollout sleep/wake rhythm (verl sidecar patch). Researcher role only.
    sidecar_sleep: bool = False
    # Depth of that sleep: 1 parks the weights in host RAM, 2 discards them.
    sidecar_sleep_level: int = 1
    # vLLM data-parallel size of the server. >0 turns on client-side sticky
    # routing via the X-data-parallel-rank header (crc32(route_key) % dp_size)
    # so multi-turn sessions hit one engine's prefix cache. 0 = no header.
    dp_size: int = 0
    # How this role is served, stated once instead of inferred from the
    # combination of endpoint/sidecar_sleep/dp_size (CD-006 requirement 3-F):
    #   local_sidecar — a server this run starts and drives through sleep/wake
    #   external_pool — endpoints someone else runs; `endpoint` may list several
    # The launcher and the run scripts both read this instead of each carrying
    # their own environment switch (EXTERNAL_RESEARCHER / REMOTE_RESEARCHER).
    # Ports are NOT a separate setting: `endpoint` is the source of truth and a
    # script that needs the port parses it from there, or the two drift.
    mode: str = "local_sidecar"

    _KNOWN_KEYS = {
        "endpoint", "model", "api_key_env", "max_concurrency",
        "temperature", "max_tokens", "timeout_s", "extra_body", "sidecar_sleep",
        "sidecar_sleep_level", "dp_size", "mode",
    }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "RoleCfg":
        unknown = set(d) - RoleCfg._KNOWN_KEYS
        if unknown:
            raise ValueError(f"role config: unknown keys {sorted(unknown)}")
        api_key_env = str(d.get("api_key_env", "") or "")
        return RoleCfg(
            endpoint=str(d.get("endpoint", "") or ""),
            model=str(d.get("model", "") or ""),
            api_key=os.environ.get(api_key_env, "") if api_key_env else "",
            max_concurrency=int(d.get("max_concurrency", 8)),
            temperature=float(d.get("temperature", 0.7)),
            max_tokens=int(d.get("max_tokens", 1024)),
            timeout_s=float(d.get("timeout_s", 300.0)),
            extra_body=dict(d.get("extra_body", {}) or {}),
            sidecar_sleep=bool(d.get("sidecar_sleep", False)),
            sidecar_sleep_level=int(d.get("sidecar_sleep_level", 1)),
            dp_size=int(d.get("dp_size", 0)),
            mode=str(d.get("mode", "local_sidecar") or "local_sidecar"),
        )


@dataclass
class ResearchCfg:
    enabled: bool = True
    backend: str = "external"        # external | engine (policy-as-researcher)
    tool_protocol: str = "fncall"    # fncall | text
    n_exp: int = 12
    k_default: int = 4
    k_max: int = 8
    max_turns: int = 30
    wrapup_buffer: int = 2
    fetch_raw_limit: int = 20
    fetch_raw_max_chars: int = 16000
    doc_max_words: int = 1200
    gt: str = "off"                  # off | on — GT into the researcher prompt
    episode_temperature: float = 1.0
    researcher_max_tokens: int = 4096
    session_timeout_s: float = 3600.0  # hard wall per research session (defensive)


@dataclass
class NarrationCfg:
    backend: str = "engine"          # engine (policy) | external | none
    max_words: int = 600
    temperature: float = 0.3


@dataclass
class EpisodeCfg:
    """Episode protocol caps (train/research/eval isomorphic, PG-3). Entry
    points push these into core.protocol via configure_episode_caps()."""

    max_env_steps: int = 30          # user ruling 2026-07-23 (was 50); cap=30
                                     # keeps 94.2% of h3 successes (CD-006 data)
    max_assistant_turns: int = 40    # env budget + format-failure slack


@dataclass
class AlfworldCfg:
    data_root: str = ""
    worker_python: str = ""
    spawn_concurrency: int = 24
    start_timeout_s: float = 300.0
    step_timeout_s: float = 120.0
    boot_concurrency: int = 12       # dataset-build engine boots (build_datasets)


@dataclass
class TrainCfg:
    """Single source for the verl launcher (scripts/launch_train.py)."""

    model_path: str = ""
    train_file: str = ""
    val_files: list[str] = field(default_factory=list)
    experiment_name: str = "trace_opd_v1"
    project_name: str = "trace_opd"
    n_gpus: int = 4
    rollout_tp: int = 2               # vLLM tensor_model_parallel_size (user ruling:
                                      # TP2 for long-horizon KV headroom; TP groups
                                      # must land on same-type GPU pairs on zkgy)
    train_batch_size: int = 16
    ppo_mini_batch_size: int = 16
    rollout_n: int = 8                # student rollouts per task, all sharing ONE
                                      # research session/doc (user ruling 2026-07-23,
                                      # relayed via CD-006: 16 tasks x 8 rollouts;
                                      # h3 was n=1). Requires train_batch_size %
                                      # agent_num_workers == 0 (dedupe invariant).
    lr: float = 1.0e-6
    response_length: int = 32768      # user ruling: no-truncation budget (= eval)
    max_prompt_length: int = 1024
    max_token_len_per_gpu: int = 34816   # single-sequence CAPACITY declaration:
                                         # this * ulysses_sp >= prompt+response.
                                         # No longer fed to verl as a micro-batch
                                         # budget (that role split off below after
                                         # the 21-GiB-logits OOMs, feedback P0)
    ppo_max_token_len_per_gpu: int = 8192    # actor train-forward packing budget;
                                             # logits mem = this x vocab x 4B
    log_prob_max_token_len_per_gpu: int = 12288  # old_log_prob inference packing
                                                 # budget (also yields the entropy
                                                 # curve); same logits formula
    rollout_max_model_len: int = 40960   # >= teacher prefix + response (linked)
    ulysses_sp: int = 1               # training-side Ulysses sequence parallel
                                      # (design §6.1). >1 makes the launcher flip
                                      # use_remove_padding=True + flash_attention_2
    rollout_gpu_mem: float = 0.35
    rollout_max_num_batched_tokens: int = 2048  # vLLM chunked-prefill budget; the
                                      # transient logits tensor during self-teacher
                                      # scoring is THIS x vocab x 2B (smoke-run OOM
                                      # root cause at 4096; halved for coexistence)
    rollout_temperature: float = 1.0
    distill_topk: int = 64
    # --- 混合目标(OPD × RL × SFT;docs/mixed-objective-design.md)---
    # 总损失 = OPD + mix_rl_weight × RL + mix_sft_weight × SFT。两个权重默认 0
    # ⇒ 与纯 OPD 逐字节一致(4B 主线无感知)。弱模型(1.7B)上 self-teacher 在
    # student 错误前缀下失去文档语义抓地力,KL 信号起不来,故给 RL 锚(真实
    # reward 的 GRPO)与 SFT 锚(特权文档 rollout 的成功轨迹)。三项同 token 口径,
    # 权重可直接互比;也是正式的消融开关(A0/A1/A2/A3 臂,见设计文档 §6)。
    mix_rl_weight: float = 0.0
    mix_sft_weight: float = 0.0
    mix_sft_rollouts: int = 4         # 每任务额外的带文档 rollout 数(仅
                                      # mix_sft_weight>0 时生效);verl 侧
                                      # rollout.n = rollout_n + 本值,尾部这几
                                      # 个副本走 expert 分支,不参与 OPD/RL
    distill_microscope: bool = True   # sample per-token distillation quantities
                                      # to <run>/microscope/ for the analysis
                                      # suite (docs/analysis-design.md; ~MBs/step)
    distill_sample_every: int = 4     # deterministic 1/N trajectory sample rate
    # --- 从环境变量迁入(CD-006 requirement 3-F;用户 2026-07-26 裁定)---------
    # 这些原先散落在启动脚本的 export 里,而代码侧另有一套 fallback 默认值。
    # 两套真源 = 换一条启动路径就静默换一种行为:`run_a800_4b.sh` 设了
    # `TRACE_TEACHER_SCORING_BACKEND=fsdp`,而 162 主线用的 `run_experiment.sh`
    # 没设,代码 fallback 是 "vllm" ⇒ 打分会悄悄退回 R11 刚根治掉的 vLLM 路径。
    # 现在配置是唯一真源,launcher 从这里写 env,env 降级为纯进程间传递。
    teacher_scoring_backend: str = "fsdp"   # fsdp | vllm(回退路径)
    vllm_rpc_timeout_ms: int = 300000       # EngineCore↔worker RPC 超时;默认
                                            # 10s 会在长序列下误判"引擎已死"
    agent_task_timeout_s: int = 2700        # R3 hang guard;0=禁用
    fsdp_mode: str = "shard"                # shard | noshard(NO_SHARD≈DDP,
                                            # 为卡间通信不可靠的机器保留)
    fsdp_strategy: str = "fsdp2"            # fsdp2 | fsdp(=FSDP1,NO_SHARD 逃生通道要它)
    fsdp_size: int = 0                      # 0=全分片;>0=仅在该大小的组内分片
                                            # (mesh(ddp=n/fsdp_size, fsdp=fsdp_size),
                                            # 显存与通信量的折中)
    param_offload: bool = True              # 参数下沉 CPU。关掉会让训练后的显存高水位
                                            # 不归还,vLLM wake_up 拿不到物理页(07-25 事故)
    reshard_after_forward: bool = True      # verl 直接把它当 zero3_enable:True=用完即
                                            # reshard(ZeRO-3),False=保留到 backward(ZeRO-2,
                                            # 省一次 all_gather 但常驻完整参数)
    dist_backend: str = ""                  # 空=verl 默认(nccl);
                                            # "cpu:gloo,cuda:gloo"=全程绕开 NCCL
    total_epochs: int = 2
    test_freq: int = 10
    save_freq: int = 10
    val_before_train: bool = True
    resume_mode: str = "auto"
    lr_scheduler_type: str = "constant"   # h3's ACTUAL schedule (its notes table
    lr_warmup_steps_ratio: float = 0.0    # saying cosine+0.03 was a misrecord)
    loggers: list[str] = field(default_factory=lambda: ["console", "swanlab"])
    agent_num_workers: int = 8            # verl agent-loop actors; per-process
                                          # concurrency caps are divided by this


@dataclass
class EvalCfg:
    response_budget: int = 32768
    temperature: float = 0.0


@dataclass
class RunCfg:
    name: str = "trace_v1"
    runs_root: str = "runs"
    tape: bool = True


@dataclass
class Config:
    run: RunCfg
    research: ResearchCfg
    narration: NarrationCfg
    roles: dict[str, RoleCfg]
    episode: EpisodeCfg
    alfworld: AlfworldCfg
    train: TrainCfg
    eval: EvalCfg
    raw: dict[str, Any]

    def role(self, name: str) -> RoleCfg:
        try:
            return self.roles[name]
        except KeyError as err:
            raise KeyError(
                f"role {name!r} not configured (roles present: {sorted(self.roles)})"
            ) from err


def _build(dc, d: dict[str, Any]):
    """Populate a dataclass from a dict, rejecting unknown keys loudly."""
    fields = {f.name for f in dc.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(d) - fields
    if unknown:
        raise ValueError(f"{dc.__name__}: unknown config keys {sorted(unknown)}")
    return dc(**d)


_KNOWN_SECTIONS = {"run", "research", "narration", "roles", "episode", "alfworld", "train", "eval"}


def _validate(cfg: "Config") -> None:
    """Semantic checks — misconfigurations die at load time, not mid-training."""
    problems: list[str] = []
    if cfg.research.backend not in ("external", "engine"):
        problems.append(f"research.backend={cfg.research.backend!r} (external|engine)")
    if cfg.research.tool_protocol not in ("fncall", "text"):
        problems.append(f"research.tool_protocol={cfg.research.tool_protocol!r} (fncall|text)")
    if cfg.research.backend == "engine" and cfg.research.tool_protocol != "text":
        problems.append("research.backend=engine requires tool_protocol=text")
    if cfg.research.gt not in ("off", "on"):
        problems.append(f"research.gt={cfg.research.gt!r} (off|on)")
    if cfg.narration.backend not in ("engine", "external", "none"):
        problems.append(f"narration.backend={cfg.narration.backend!r} (engine|external|none)")
    if cfg.research.enabled and cfg.research.backend == "external" and "researcher" not in cfg.roles:
        problems.append("research.backend=external but roles.researcher is missing")
    if cfg.narration.backend == "external" and "narrator_external" not in cfg.roles:
        problems.append("narration.backend=external but roles.narrator_external is missing")
    if not (0 < cfg.research.k_default <= cfg.research.k_max):
        problems.append("research: need 0 < k_default <= k_max")
    if cfg.episode.max_env_steps < 1 or cfg.episode.max_assistant_turns <= cfg.episode.max_env_steps:
        problems.append(
            f"episode: max_env_steps={cfg.episode.max_env_steps} must be >=1 and "
            f"max_assistant_turns={cfg.episode.max_assistant_turns} must exceed it "
            "(the slack absorbs format-failure turns)"
        )
    if cfg.train.agent_num_workers < 1:
        problems.append("train.agent_num_workers must be >= 1")
    if cfg.train.rollout_tp < 1 or cfg.train.n_gpus % cfg.train.rollout_tp != 0:
        problems.append(
            f"train.rollout_tp={cfg.train.rollout_tp} must be >=1 and divide "
            f"n_gpus={cfg.train.n_gpus}"
        )
    if cfg.train.rollout_n < 1:
        problems.append(f"train.rollout_n={cfg.train.rollout_n} must be >= 1")
    if cfg.train.ppo_max_token_len_per_gpu < 1 or cfg.train.log_prob_max_token_len_per_gpu < 1:
        problems.append("train.ppo/log_prob_max_token_len_per_gpu must be >= 1")
    if cfg.train.distill_sample_every < 1:
        problems.append(
            f"train.distill_sample_every={cfg.train.distill_sample_every} must be >= 1"
        )
    # 混合目标:负权重会翻转梯度方向而不报错,必须显式拦住
    import math as _math

    for _name, _value in (
        ("mix_rl_weight", cfg.train.mix_rl_weight),
        ("mix_sft_weight", cfg.train.mix_sft_weight),
    ):
        if not (_math.isfinite(_value) and _value >= 0):
            problems.append(f"train.{_name}={_value} must be finite and >= 0 (0 disables the arm)")
    if cfg.train.mix_sft_rollouts < 0:
        problems.append(
            f"train.mix_sft_rollouts={cfg.train.mix_sft_rollouts} must be >= 0"
        )
    if cfg.train.mix_sft_weight > 0:
        if cfg.train.mix_sft_rollouts < 1:
            problems.append(
                "train.mix_sft_weight>0 requires mix_sft_rollouts>=1 (no expert rollouts, "
                "no expert samples — the SFT term would silently stay zero)"
            )
        if not cfg.research.enabled:
            problems.append(
                "train.mix_sft_weight>0 requires research.enabled (expert rollouts need a document)"
            )
    if cfg.train.teacher_scoring_backend not in ("fsdp", "vllm"):
        problems.append(
            f"train.teacher_scoring_backend={cfg.train.teacher_scoring_backend!r} "
            "must be 'fsdp' or 'vllm'"
        )
    if (
        cfg.train.teacher_scoring_backend == "vllm"
        and cfg.train.rollout_max_num_batched_tokens > 2048
    ):
        problems.append(
            f"train.teacher_scoring_backend=vllm with rollout_max_num_batched_tokens="
            f"{cfg.train.rollout_max_num_batched_tokens}: scoring on the rollout engine "
            "materialises a full-vocabulary logits buffer per prefill chunk "
            "(chunk x vocab x 2B), which OOMed the smoke run at 4096. Either keep the "
            "fsdp backend, or drop the budget to 2048 when falling back to vllm."
        )
    if cfg.train.fsdp_mode not in ("shard", "noshard"):
        problems.append(f"train.fsdp_mode={cfg.train.fsdp_mode!r} must be 'shard' or 'noshard'")
    if cfg.train.vllm_rpc_timeout_ms < 1000:
        problems.append(
            f"train.vllm_rpc_timeout_ms={cfg.train.vllm_rpc_timeout_ms} is below 1s — "
            "a short RPC timeout makes EngineCore declare a live engine dead"
        )
    if cfg.train.agent_task_timeout_s < 0:
        problems.append(f"train.agent_task_timeout_s={cfg.train.agent_task_timeout_s} must be >= 0 (0 disables)")
    for name, role in cfg.roles.items():
        if role.dp_size < 0:
            problems.append(f"roles.{name}.dp_size must be >= 0 (0 disables routing)")
        if role.mode not in ("local_sidecar", "external_pool"):
            problems.append(
                f"roles.{name}.mode={role.mode!r} must be 'local_sidecar' or 'external_pool'"
            )
        if role.mode == "external_pool":
            # Someone else runs these endpoints: this run must not try to drive
            # them through sleep/wake, and dp routing is a single-server notion.
            if not role.endpoint.strip():
                problems.append(f"roles.{name}.mode=external_pool requires a non-empty endpoint")
            if role.sidecar_sleep:
                problems.append(f"roles.{name}: external_pool cannot use sidecar_sleep")
            if role.dp_size > 0:
                problems.append(
                    f"roles.{name}: external_pool cannot use dp_size>0 "
                    "(independent servers, no shared scheduler)"
                )
    if cfg.train.train_batch_size % cfg.train.agent_num_workers != 0:
        problems.append(
            f"train.train_batch_size={cfg.train.train_batch_size} must be divisible "
            f"by agent_num_workers={cfg.train.agent_num_workers}: the rollout_n "
            "copies of a task are interleaved and chunked contiguously across "
            "workers, and divisibility is what keeps all copies of one task in "
            "the same worker process (research-session dedupe invariant)"
        )
    if cfg.train.ulysses_sp < 1 or cfg.train.n_gpus % cfg.train.ulysses_sp != 0:
        problems.append(
            f"train.ulysses_sp={cfg.train.ulysses_sp} must be >=1 and divide "
            f"n_gpus={cfg.train.n_gpus}"
        )
    if (
        cfg.train.max_token_len_per_gpu * cfg.train.ulysses_sp
        < cfg.train.max_prompt_length + cfg.train.response_length
    ):
        problems.append(
            f"sequence capacity max_token_len_per_gpu*ulysses_sp="
            f"{cfg.train.max_token_len_per_gpu * cfg.train.ulysses_sp} is below "
            f"max_prompt_length+response_length="
            f"{cfg.train.max_prompt_length + cfg.train.response_length} — a full-length "
            "sample could not fit one micro batch (raise max_token_len_per_gpu or "
            "ulysses_sp; design.md §6.1)"
        )
    for name, role in cfg.roles.items():
        if role.sidecar_sleep and name != "researcher":
            problems.append(
                f"roles.{name}.sidecar_sleep=true — only the researcher server "
                "is driven by the trainer's sleep/wake rhythm"
            )
    for name, role in cfg.roles.items():
        if "," in role.endpoint:  # endpoint pool (independent replicas)
            if role.sidecar_sleep:
                problems.append(
                    f"roles.{name}: endpoint pool + sidecar_sleep=true — the "
                    "sleep/wake rhythm drives ONE local server, not a pool "
                    "(remote pool deployments must set sidecar_sleep: false)"
                )
            if role.dp_size > 0:
                problems.append(
                    f"roles.{name}: endpoint pool + dp_size>0 — the "
                    "X-data-parallel-rank header targets DP ranks inside one "
                    "server; independent replicas need dp_size: 0 (the pool "
                    "does its own session-sticky balancing)"
                )
    if (
        "researcher" in cfg.roles
        and cfg.roles["researcher"].sidecar_sleep
        and not (cfg.research.enabled and cfg.research.backend == "external")
    ):
        problems.append(
            "roles.researcher.sidecar_sleep=true requires research.enabled "
            "with backend=external (no external server to drive otherwise)"
        )
    if problems:
        raise ValueError("invalid config:\n  - " + "\n  - ".join(problems))


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides onto a parsed config dict.

    For settings that describe the EXPERIMENT rather than the machine — the
    mixed-objective ablation weights above all. Those belong in the config file
    (it is the single source of truth, and the launch manifest snapshots it), but
    running four ablation arms off one file needs a way to say which arm without
    editing the file for every run and remembering to edit it back.

    Applied to the raw dict before the dataclasses are built, so the override
    goes through the same unknown-key rejection and the same validation as a
    value written in the file, and the manifest's config snapshot shows what
    actually ran. Values are parsed as YAML, so ``0.5`` is a float and ``true``
    is a bool, exactly as if they had been typed into the file.
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"config override {item!r} is not of the form section.key=value")
        path, _, value = item.partition("=")
        parts = [p for p in path.strip().split(".") if p]
        if len(parts) < 2:
            raise ValueError(f"config override {item!r} must name a section, e.g. train.mix_rl_weight=1.0")
        if parts[0] not in _KNOWN_SECTIONS:
            raise ValueError(f"config override {item!r}: unknown section {parts[0]!r}")
        # Walk the full path rather than splitting once: nested sections are
        # real (roles.researcher.mode), and treating "researcher.mode" as a
        # literal key under roles produced a role named "researcher.mode" whose
        # value was a string — surfacing much later as "unknown keys
        # ['_','a','e',...]" from iterating that string's characters.
        target = data
        for depth, part in enumerate(parts[:-1]):
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                where = ".".join(parts[: depth + 1])
                raise ValueError(f"config override {item!r}: {where} is not a mapping")
        target[parts[-1]] = yaml.safe_load(value)
    return data


def template_drift(config_path: str | Path, template_path: str | Path) -> dict[str, list[str]]:
    """Compare an experiment's config against the mechanism's template.

    Each experiment owns a COMPLETE copy of the template — not a diff against it.
    A run's config is then self-contained: readable on its own months later,
    diffable against any other experiment, and immune to a template edit
    silently changing what a finished experiment meant.

    The cost of copies is drift, so the template earns its keep by being the one
    place a mechanism change is made and the reference every copy is checked
    against. This reports both directions:

      missing — the template has keys this experiment does not. The mechanism
                grew a setting after the copy was taken; the run falls back to
                the code's default, which may not be the template's recommended
                value. Surfaced at launch, never silent.
      extra   — this experiment has keys the template does not. Usually a
                setting the mechanism removed and the copy still carries.
                (An outright typo is already rejected by the dataclass loader.)
    """
    def flat(data: Any, prefix: str = "") -> set[str]:
        keys: set[str] = set()
        if not isinstance(data, dict):
            return keys
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                keys |= flat(value, path)
            else:
                keys.add(path)
        return keys

    experiment = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    template = yaml.safe_load(Path(template_path).read_text(encoding="utf-8")) or {}
    exp_keys, tpl_keys = flat(experiment), flat(template)
    # Role names are the experiment's business (a run may serve roles the
    # template does not list); compare the SHAPE of a role, not which roles exist.
    def role_shape(keys: set[str]) -> set[str]:
        out = set()
        for key in keys:
            parts = key.split(".")
            if parts[0] == "roles" and len(parts) >= 3:
                out.add("roles.*." + ".".join(parts[2:]))
            else:
                out.add(key)
        return out

    exp_keys, tpl_keys = role_shape(exp_keys), role_shape(tpl_keys)
    return {
        "missing": sorted(tpl_keys - exp_keys),
        "extra": sorted(exp_keys - tpl_keys),
    }


def objective_fingerprint(cfg: "Config") -> str:
    """Short hash of the settings that define WHAT this run optimises.

    Resuming into a run directory whose fingerprint differs means continuing one
    experiment's checkpoint under another experiment's objective — the ablation
    arms all default to the same run name, so this is one forgotten flag away.
    Deliberately narrow: it covers the objective and the batch composition that
    feeds it, not the machine (resuming on different hardware is legitimate).
    """
    t = cfg.train
    material = "|".join(
        str(x) for x in (
            t.model_path, t.mix_rl_weight, t.mix_sft_weight, t.mix_sft_rollouts,
            t.rollout_n, t.train_batch_size, t.distill_topk, t.lr,
            t.teacher_scoring_backend, cfg.research.enabled,
        )
    )
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data = _expand(raw)
    if overrides:
        data = apply_overrides(data, overrides)
    unknown = set(data) - _KNOWN_SECTIONS
    if unknown:
        raise ValueError(f"unknown top-level config sections {sorted(unknown)}")
    roles = {
        name: RoleCfg.from_dict(cfg or {})
        for name, cfg in (data.get("roles") or {}).items()
    }
    cfg = Config(
        run=_build(RunCfg, data.get("run") or {}),
        research=_build(ResearchCfg, data.get("research") or {}),
        narration=_build(NarrationCfg, data.get("narration") or {}),
        roles=roles,
        episode=_build(EpisodeCfg, data.get("episode") or {}),
        alfworld=_build(AlfworldCfg, data.get("alfworld") or {}),
        train=_build(TrainCfg, data.get("train") or {}),
        eval=_build(EvalCfg, data.get("eval") or {}),
        raw=data,
    )
    _validate(cfg)
    return cfg
