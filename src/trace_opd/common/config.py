"""Configuration loading — ONE yaml file is the single source of truth for
every tunable in the mechanism (configs/trace.yaml): research session,
narration, role connections, environment, training launcher, evaluation.

``${VARS}`` expand from the environment at load time (empty when unset;
clients fail loudly at connection time). Deployment facts (endpoints, data
roots, interpreter paths) live in env vars, not in the yaml.
"""
from __future__ import annotations

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
    # vLLM data-parallel size of the server. >0 turns on client-side sticky
    # routing via the X-data-parallel-rank header (crc32(route_key) % dp_size)
    # so multi-turn sessions hit one engine's prefix cache. 0 = no header.
    dp_size: int = 0

    _KNOWN_KEYS = {
        "endpoint", "model", "api_key_env", "max_concurrency",
        "temperature", "max_tokens", "timeout_s", "extra_body", "sidecar_sleep",
        "dp_size",
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
            dp_size=int(d.get("dp_size", 0)),
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
    # --- 信号增强三件套(2026-07-24 用户定稿,cd006-signal-enhancement-final.md)---
    distill_block_turns: int = 4      # K3 分块保鲜打分块宽 W(0=关,退回单次前缀打分);
                                      # 半衰期实测 ≥8 轮,W=4 取 100% 区(用户定)
    distill_loss_soft_cap: bool = True   # K1 clamp 软化:超过 loss_max_clamp 改
                                         # c+log1p(x-c) 压缩,保尖峰有序(不再削平)
    distill_kl_weight_alpha: float = 1.0  # K2 |KL|^α 自适应加权(0=关);批内归一保尺度
    distill_kl_weight_cap: float = 10.0   # K2 单 token 权重上限
    distill_microscope: bool = True   # sample per-token distillation quantities
                                      # to <run>/microscope/ for the analysis
                                      # suite (docs/analysis-design.md; ~MBs/step)
    distill_sample_every: int = 4     # deterministic 1/N trajectory sample rate
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
    # 信号增强三件套参数(codex review 2026-07-24:错误值会静默关闭/反转,必须显式校验)
    import math as _math
    if cfg.train.distill_block_turns < 0:
        problems.append(f"train.distill_block_turns={cfg.train.distill_block_turns} must be >= 0 (0 disables K3)")
    if not (_math.isfinite(cfg.train.distill_kl_weight_alpha) and cfg.train.distill_kl_weight_alpha >= 0):
        problems.append(f"train.distill_kl_weight_alpha={cfg.train.distill_kl_weight_alpha} must be finite and >= 0 (0 disables K2)")
    if not (_math.isfinite(cfg.train.distill_kl_weight_cap) and cfg.train.distill_kl_weight_cap >= 1):
        problems.append(f"train.distill_kl_weight_cap={cfg.train.distill_kl_weight_cap} must be finite and >= 1 (final post-normalization weight bound)")
    for name, role in cfg.roles.items():
        if role.dp_size < 0:
            problems.append(f"roles.{name}.dp_size must be >= 0 (0 disables routing)")
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


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data = _expand(raw)
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
