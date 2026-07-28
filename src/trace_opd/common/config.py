"""Configuration loading — ONE yaml file is the single source of truth for
every tunable in the mechanism (configs/trace.yaml): research session,
narration, role connections, environment, training launcher, evaluation.

``${VARS}`` expand from the environment at load time (empty when unset;
clients fail loudly at connection time). Deployment facts (endpoints, data
roots, interpreter paths) live in env vars, not in the yaml.
"""
from __future__ import annotations

import hashlib
import math
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
class ResearchRLCfg:
    """Self-research 通道(docs/self-research-design.md):policy-as-researcher 的
    research 能力 RL。与执行侧混合目标(train.mix_*)是两个**正交模块**——那三个
    系数优化「任务执行更好」,这里优化「researcher 更好」;二者唯一的耦合是数据流
    (argmax 文档拿 teacher 位),不在 loss 语义上,故本节独立于 TrainCfg。

    weight=0(默认)= 通道关断:不产 research 副本、不加 loss 项、不进 fingerprint,
    与现状逐字节一致。开启硬前提:research.backend=engine + tool_protocol=text
    (research 轨迹必须是 policy 自己的 on-policy 采样,外部 35B 的轨迹不能拿来
    做 policy 的 PG)。"""

    weight: float = 0.0               # research_loss_coef;第二通道的整体系数
    update_mode: str = "joint"        # joint | separate(设计 §7.1:joint=同一
                                      # backward,完美 on-policy,默认;separate=
                                      # 同一全局 step 的第二次 update,UCOB/SEED
                                      # 形态,备选与消融臂)
    n_docs: int = 8                   # N:每任务并行 research 会话数(组间对比的
                                      # 组大小;用户裁定默认 8,快速实验可调小,>=2)
    eval_k: int = 8                   # K:每份文档的 doc-conditioned 评测 rollout 数
                                      # (p_i = wins/K,即 PG-7 文档质量信号)
    adv_clip: float = 2.0             # 组间 advantage 硬 clip(UCOB 同款 ±2.0)
    warmup_steps: int = 30            # 前多少个 global step research 行不给梯度
                                      # (UCOB writer warmup ~30);文档生产与
                                      # teacher 选择从第 1 步照常
    max_sample_tokens: int = 16384    # research 会话 token 样本总长预算;超限则
                                      # 该轨迹降级不进训练面(doc 仍用、reward 仍
                                      # 参与组统计),防超长序列打爆显存峰值


@dataclass
class SftGenCfg:
    """Researcher-SFT 离线采样 pipeline(src/trace_opd/sftgen/;用户需求
    2026-07-28)。与训练机制共享 research/narration/roles/alfworld 各节(同构性
    由共享配置保证——特别是 research.gt 与 research.tool_protocol 两键,SFT 采样
    与 self-evolve 训练读同一个值);本节只放 pipeline 特有键。不影响训练路径。"""

    benchmark: str = "alfworld"       # 分层函数注册键(sftgen/strata.py;新
                                      # benchmark = 新 adapter 包 + 一个分层函数)
    tasks: int = 512                  # 采样任务数 T(用户后续制定;从训练 parquet
                                      # 按 seed 分层采样,与训练任务身份严格同源)
    sessions_per_task: int = 1        # 每任务会话数 M(覆盖优先,默认 1)
    seed: int = 20260728
    eval_k: int = 8                   # 每份 doc 的带文档评测 rollout 数(0=关);
                                      # 口径=机制温度下的 PG-7 信号,直接对标训练时 p_i
    bare_k: int = 8                   # 裸对照 rollout 数(增益 = p_doc − p_bare)
    task_parallel: int = 16           # 并发会话数
    executor_endpoint: str = ""       # 初始 policy 的 vLLM OpenAI 端点(${VAR};
                                      # executor episodes 与 narrator 共用此服务)
    executor_model: str = ""          # 该服务的模型名(Qwen3-4B-Instruct-2507)


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
    # --- OPD 项的散度形式(docs/mixed-objective-design.md §2.7)-----------------
    # 与 mix_opd_weight 正交:系数管强度,这里管 OPD 项内部用哪种散度。四个值:
    #   forward_kl_topk —— 现状默认。teacher top-K 上的截断前向 KL(分布蒸馏,
    #                      dense,K=distill_topk)。4B 主线全部结论建立于此。
    #   reverse_kl_k3   —— 采样 token 单点反向 KL 的 k3 估计器(Schulman
    #                      kl-approx 编号,与已移除的信号增强三件套 K1/K2/K3 无关),
    #                      CD-003 反向 KL 消融同款 ⇒ 与 h3_2507_full_rkl3_from10
    #                      历史曲线严格可比。teacher 打分降为单点(topk=0),
    #                      传输/打分开销反而低于 forward。
    #   reverse_kl_topk —— 与 forward 完全镜像的稠密反向 KL(同一份 top-K 数据
    #                      + 尾桶),消融时**唯一变量是 KL 方向本身**;estimator
    #                      形式一次改三个变量(方向/单点 vs 分布/估计器),回答
    #                      不了"方向贡献几何"。
    #   reverse_kl_gated—— SDAR/SEED 的置信门控 logp-shift(两家数学同构;
    #                      gate=sigmoid(β·Δ) detach,梯度=−gate·∇logπ_S,
    #                      只上拉采样 token、teacher 否决则指数衰减)。他们的
    #                      λ 对应我们的 mix_opd_weight(SDAR 代码默认 0.1、
    #                      论文 0.01),β 对应 opd_gate_beta。
    # 注意:①各模式 loss 数值尺度互不可比,跨模式只比成功率等任务指标,
    # mix_opd_weight 的"等效学习率"含义也随模式变;②CD-003 定论:单换 KL 方向
    # 不救崩塌(方向是二阶因素,只调崩塌速度/深度)——此开关的价值是消融完备性
    # 与"方向 × RL/SFT 臂交互"这一开放问题(SDAR/SEED 的稳定配方 = RL 锚 +
    # 采样点 reverse surrogate,CD-003 只测过纯 OPD × reverse);③estimator 两模式
    # 下 top-k 系观察(student/teacher_mass、overlap_ratio、teacher_topk_entropy)
    # 数据上不存在,自动缺席;显微镜仍落 per-token loss(gated 另有 gate 列)。
    opd_kl_mode: str = "forward_kl_topk"
    opd_gate_beta: float = 5.0        # 仅 reverse_kl_gated 消费;SDAR/SEED 均默认 5.0
    # --- 混合目标(OPD × RL × SFT;docs/mixed-objective-design.md)---
    # 总损失 = OPD + mix_rl_weight × RL + mix_sft_weight × SFT。两个权重默认 0
    # ⇒ 与纯 OPD 逐字节一致(4B 主线无感知)。弱模型(1.7B)上 self-teacher 在
    # student 错误前缀下失去文档语义抓地力,KL 信号起不来,故给 RL 锚(真实
    # reward 的 GRPO)与 SFT 锚(特权文档 rollout 的成功轨迹)。三项同 token 口径,
    # 权重可直接互比;也是正式的消融开关(A0/A1/A2/A3 臂,见设计文档 §6)。
    mix_opd_weight: float = 1.0        # OPD 项自己的系数。**两臂全关时它就是 loss 的
                                      # 整体缩放,也就是有效学习率的乘数**——纯 OPD 基线
                                      # 是在 1.0 上建立的(4B step-9 起飞),改它等于改基线。
                                      # 0 = 完全去掉 OPD 项,即"OPD 在弱模型上还有没有净
                                      # 贡献"那个消融臂(A5)。
    mix_rl_weight: float = 0.0
    mix_sft_weight: float = 0.0
    mix_sft_rollouts: int = 4         # 每任务额外的带文档 rollout 数(仅
                                      # mix_sft_weight>0 时生效);verl 侧
                                      # rollout.n = rollout_n + 本值,尾部这几
                                      # 个副本走 expert 分支,不参与 OPD/RL
    # SFT 项的前缀,决定这一项在优化谁(CD-006 requirements §3-G,用户 2026-07-26 裁定):
    #   doc  —— 保留 expert 生成时的带文档 prompt ⇒ 优化 policy(·|裸+文档) = **OPD 的
    #           teacher 本身**。分工是「SFT 修老师、OPD 做传导」,直击 CD-003 的病因
    #           (teacher 在 student 错误前缀下失去抓地力)。
    #   bare —— 重放到裸 prompt ⇒ 直接教裸 policy。与 OPD 功能重叠,且 A2 首跑实测
    #           **23.6% 的已接受样本在教「零观察依据下断言物体位置」**——expert 带着
    #           文档才知道东西在哪,重放到裸条件后这些断言没有依据,而 expert 轨迹里
    #           没有一条演示「猜错了怎么回退」⇒ 学到自信断言 + 无法恢复。保留作对照臂。
    mix_sft_prefix: str = "doc"       # doc | bare
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
    research_rl: ResearchRLCfg
    sft_gen: SftGenCfg
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


_KNOWN_SECTIONS = {"run", "research", "research_rl", "sft_gen", "narration", "roles", "episode", "alfworld", "train", "eval"}


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
        ("mix_opd_weight", cfg.train.mix_opd_weight),
        ("mix_rl_weight", cfg.train.mix_rl_weight),
        ("mix_sft_weight", cfg.train.mix_sft_weight),
    ):
        if not (_math.isfinite(_value) and _value >= 0):
            problems.append(f"train.{_name}={_value} must be finite and >= 0 (0 disables the arm)")
    if cfg.train.mix_sft_rollouts < 0:
        problems.append(
            f"train.mix_sft_rollouts={cfg.train.mix_sft_rollouts} must be >= 0"
        )
    if not (cfg.train.mix_opd_weight or cfg.train.mix_rl_weight or cfg.train.mix_sft_weight):
        problems.append(
            "train.mix_opd_weight / mix_rl_weight / mix_sft_weight are all 0 — "
            "there is no objective left to optimise"
        )
    if cfg.train.mix_sft_prefix not in ("doc", "bare"):
        problems.append(
            f"train.mix_sft_prefix={cfg.train.mix_sft_prefix!r} must be 'doc' or 'bare'"
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
    # --- sftgen pipeline(轻校验:默认值恒合法,绝不阻训练路径)------------
    sgc = cfg.sft_gen
    if sgc.tasks < 1:
        problems.append(f"sft_gen.tasks={sgc.tasks} must be >= 1")
    if sgc.sessions_per_task < 1:
        problems.append(f"sft_gen.sessions_per_task={sgc.sessions_per_task} must be >= 1")
    if sgc.eval_k < 0 or sgc.bare_k < 0:
        problems.append("sft_gen.eval_k/bare_k must be >= 0")
    if sgc.task_parallel < 1:
        problems.append(f"sft_gen.task_parallel={sgc.task_parallel} must be >= 1")

    # --- self-research 通道(docs/self-research-design.md §7)---------------
    rrl = cfg.research_rl
    if not math.isfinite(rrl.weight) or rrl.weight < 0.0:
        problems.append(f"research_rl.weight={rrl.weight} must be finite and >= 0")
    if rrl.update_mode not in ("joint", "separate"):
        problems.append(f"research_rl.update_mode={rrl.update_mode!r} (joint|separate)")
    if rrl.weight > 0.0:
        if cfg.research.backend != "engine" or cfg.research.tool_protocol != "text":
            problems.append(
                "research_rl.weight>0 requires research.backend=engine and "
                "research.tool_protocol=text — the research trajectories must be the "
                "policy's own on-policy samples (an external researcher's tokens cannot "
                "carry the policy's PG)"
            )
        if not cfg.research.enabled:
            problems.append("research_rl.weight>0 requires research.enabled=true")
        if rrl.n_docs < 2:
            problems.append(
                f"research_rl.n_docs={rrl.n_docs} must be >= 2 (the reward is a "
                "group-relative comparison; one document has no group)"
            )
        if rrl.eval_k < 1:
            problems.append(f"research_rl.eval_k={rrl.eval_k} must be >= 1")
        if not math.isfinite(rrl.adv_clip) or rrl.adv_clip <= 0.0:
            problems.append(f"research_rl.adv_clip={rrl.adv_clip} must be finite and > 0")
        if rrl.warmup_steps < 0:
            problems.append(f"research_rl.warmup_steps={rrl.warmup_steps} must be >= 0")
        if rrl.max_sample_tokens < 1024:
            problems.append(
                f"research_rl.max_sample_tokens={rrl.max_sample_tokens} must be >= 1024"
            )
        # codex review P1-4: the R3 hang guard (agent_task_timeout_s) wraps the
        # WHOLE prompt group — with the research arm on that group serially
        # contains a full researcher session plus its eval rollouts, so the
        # default 2700s guard fires BEFORE the 3600s session timeout and every
        # in-session degradation path becomes unreachable. Require headroom.
        if cfg.train.agent_task_timeout_s != 0 and (
            cfg.train.agent_task_timeout_s
            < cfg.research.session_timeout_s + 1800
        ):
            problems.append(
                f"research_rl.weight>0 requires train.agent_task_timeout_s=0 or >= "
                f"research.session_timeout_s+1800 (got {cfg.train.agent_task_timeout_s} vs "
                f"{cfg.research.session_timeout_s}+1800): the hang guard would kill the "
                "prompt group before the session's own timeout/degradation can run"
            )
        if (
            rrl.max_sample_tokens + cfg.train.max_prompt_length
            > cfg.train.log_prob_max_token_len_per_gpu
        ):
            # Soft warning, not a rejection: the packing budget clamps up to the
            # longest sequence (c48568e), so the run survives — but the
            # old-log-prob forward then peaks at max_sample_tokens x vocab x 4B
            # of fp32 logits (~10 GiB at 16k x 152k), which is exactly the
            # profile that killed earlier runs on the A800.
            print(
                f"[config] WARNING research_rl.max_sample_tokens={rrl.max_sample_tokens} "
                f"exceeds log_prob_max_token_len_per_gpu="
                f"{cfg.train.log_prob_max_token_len_per_gpu}: the log-prob packing "
                "budget will be clamped up by the longest research row — watch the "
                "old-log-prob peak memory on the first step"
            )
        if rrl.max_sample_tokens + cfg.train.max_prompt_length > cfg.train.rollout_max_model_len:
            problems.append(
                f"research_rl.max_sample_tokens={rrl.max_sample_tokens} + "
                f"max_prompt_length={cfg.train.max_prompt_length} exceeds "
                f"rollout_max_model_len={cfg.train.rollout_max_model_len}"
            )

    if cfg.train.opd_kl_mode not in (
        "forward_kl_topk", "reverse_kl_k3", "reverse_kl_topk", "reverse_kl_gated"
    ):
        problems.append(
            f"train.opd_kl_mode={cfg.train.opd_kl_mode!r} must be one of "
            "forward_kl_topk / reverse_kl_k3 / reverse_kl_topk / reverse_kl_gated"
        )
    if not (_math.isfinite(cfg.train.opd_gate_beta) and cfg.train.opd_gate_beta > 0):
        # β→0 把 gate 退化成平坦 0.5(半强度 logp-shift),负 β 反转到
        # teacher 否决的 token 上——都不是任何实验想表达的意思
        problems.append(
            f"train.opd_gate_beta={cfg.train.opd_gate_beta} must be finite and > 0"
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
            t.model_path, t.mix_opd_weight, t.mix_rl_weight, t.mix_sft_weight,
            t.mix_sft_rollouts,
            t.mix_sft_prefix,
            t.rollout_n, t.train_batch_size, t.distill_topk, t.lr,
            t.teacher_scoring_backend, cfg.research.enabled,
        )
    )
    # KL 模式条件式追加,而非无条件进列表:加字段那一刻有三个消融臂在生产上跑
    # (A0/A1/A2-doc,均为 forward_kl_topk),无条件追加会改变它们已写入 manifest
    # 的指纹,让下一次 resume 被我们自己的守卫拒掉。默认值不进 hash ⇒ 旧指纹
    # 逐字节不变;非默认模式照常拒绝跨模式 resume;β 只在 gated 模式下有语义,
    # 也只在那时参与。
    if t.opd_kl_mode != "forward_kl_topk":
        material += f"|opd_kl_mode={t.opd_kl_mode}"
        if t.opd_kl_mode == "reverse_kl_gated":
            material += f"|opd_gate_beta={t.opd_gate_beta}"
    # self-research 通道同一纪律:默认关断(weight=0)不进 hash ⇒ 存量 run 指纹
    # 逐字节不变;开启后 N/K/模式/warmup 都定义"优化什么",跨值 resume 拒绝。
    rrl = cfg.research_rl
    if rrl.weight > 0.0:
        material += (
            f"|research_rl={rrl.weight},{rrl.update_mode},{rrl.n_docs},"
            f"{rrl.eval_k},{rrl.adv_clip},{rrl.warmup_steps},{rrl.max_sample_tokens}"
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
        research_rl=_build(ResearchRLCfg, data.get("research_rl") or {}),
        sft_gen=_build(SftGenCfg, data.get("sft_gen") or {}),
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
