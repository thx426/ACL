"""Configuration loading — ONE yaml file is the single source of truth for
every tunable in the mechanism (configs/pgs.yaml): episode protocol,
environment, training launcher, steering, evaluation.

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
    """外部模型角色的连接配置；密钥只由环境变量在运行期解析。"""

    endpoint: str = ""
    model: str = ""
    api_key: str = ""
    api_key_env: str = ""
    max_concurrency: int = 8
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout_s: float = 300.0
    extra_body: dict[str, Any] = field(default_factory=dict)
    sidecar_sleep: bool = False
    sidecar_sleep_level: int = 1
    dp_size: int = 0
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
            api_key_env=api_key_env,
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
    """每个(step, task)一次的在线文档生产合同。"""

    enabled: bool = False
    backend: str = "external"
    tool_protocol: str = "fncall"
    n_exp: int = 12
    k_default: int = 4
    k_max: int = 8
    max_turns: int = 30
    wrapup_buffer: int = 2
    fetch_raw_limit: int = 20
    fetch_raw_max_chars: int = 16000
    doc_max_words: int = 3000
    gt: str = "off"
    episode_temperature: float = 1.0
    researcher_max_tokens: int = 8192
    session_timeout_s: float = 3600.0


@dataclass
class NarrationCfg:
    """research 实验轨迹的叙述后端；engine 复用当前 SGLang rollout。"""

    backend: str = "engine"
    max_words: int = 1000
    temperature: float = 0.3


@dataclass
class EpisodeCfg:
    """Episode protocol caps (train/research/eval isomorphic, PG-3). Entry
    points push these into core.protocol via configure_episode_caps()."""

    max_env_steps: int = 30          # user ruling 2026-07-23 (was 50); cap=30
                                     # keeps 94.2% of h3 successes (CD-006 data)
    max_assistant_turns: int = 40    # env budget + format-failure slack
    # 单轮生成上限(reasoning+action)。此前是 protocol.py 的硬编码常量 2048;
    # 2026-07-30 用户裁定提到 4096 并改为配置项。
    # 依据(CD-008 实测,原 rkl):打满该上限的轮 step1-8 恒 0、s16 越 1%、s17 达 3.97%,
    # 而**任一轮打满该轨迹的 won 率只剩 10.5%**(未打满的 70.4%)——打满的那一轮拿不到
    # </action> 闭合,动作解析必然失败,于是"卡住→写更长→打满→新的无效动作"闭环。
    # ⚠️ 它不进 objective_fingerprint(见 protocol.configure_episode_caps 的说明):
    #    改的是采样协议不是优化目标,故跨值 resume 是允许的。要还原历史 run 的采样
    #    行为,在其配置里显式写回 2048。
    # ⚠️ 总预算 train.response_length 不变,单轮放宽会更快耗尽它 ⇒ 轮数可能减少。
    max_tokens_per_turn: int = 2048


@dataclass
class AlfworldCfg:
    data_root: str = ""
    worker_python: str = ""
    spawn_concurrency: int = 128    # 全局 ALFWorld 引擎并发 boot 上限(按 worker 数折算);
                                    # 用户裁定 2026-08-04:拉到 128 充分利用 80 核
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
    rollout_backend: str = "vllm"    # rollout 引擎:vllm | sglang(2026-08-05 用户
                                     # 裁定切换 SGLang:原生 logprob_start_len
                                     # tail-only + radix 前缀复用,根治 verify
                                     # 平方墙钟;vllm 保留为回摆通道。执行层,
                                     # 不进 objective_fingerprint)
    rollout_attention_backend: str = ""  # 仅 sglang 后端:attention 内核显式指定
                                     # (空=引擎自动)。0.5.8 在 A800(sm80)上的
                                     # 自动选择待 PoC 门 2 验证;若起服失败按
                                     # 环境册填 flashinfer / triton / fa_x
    rollout_max_num_batched_tokens: int = 2048  # vLLM chunked-prefill budget; the
                                      # transient logits tensor during prompt_logprobs
                                      # scoring (SPS verify) is THIS x vocab x 2B
                                      # (smoke-run OOM root cause at 4096)
    rollout_max_num_seqs: int = 128 # 每个 vLLM replica 的并发序列上限(单 TP 上限;
                                    # 用户裁定 2026-08-04)。verl 默认 1024,显式钉死
                                    # 便于与 agent 侧并发对账
    rollout_temperature: float = 1.0
    # --- 损失的信用分配单位(verl actor.loss_agg_mode;蒸馏项与 policy 项共用)---
    # token-mean(verl 默认,4B 主线 rkl/sp2 全部历史结论建立于此):
    #   loss = Σ_token loss / batch_num_tokens —— 分母是全批 token 总数,
    #   故**每条轨迹按其 token 数获得梯度权重**。CD-008 实测:step-17 时
    #   85.5% 的 token 与 70.8% 的 loss 来自最终失败的轨迹,最长 10% 的行拿走
    #   21.2% 的 token;而长尾恰恰是每 token 信息最低的部分
    #   (ρ(length, loss mean) = −0.856),即长失败序列拥有最多"投票权"。
    # seq-mean-token-mean(**现默认**,2026-07-29 用户裁定):每条轨迹先 token-mean,
    #   再在序列间等权 ⇒ 信用单位从 token 变为轨迹,长尾不再按长度加权。
    # ⚠️ 2026-07-29 起默认值由 verl 的 token-mean 改为 seq-mean-token-mean。
    #    这会改变**所有未显式设置该项的配置**的有效梯度分布 —— 包括
    #    162-4b-a0 / a800-4b-a0 / 162-1p7b-* 等既有实验配置。
    #    要复现 2026-07-29 之前的任何 run,必须在其配置里显式写回
    #    `loss_agg_mode: token-mean`(4B 主线 rkl/sp2 的全部历史结论建立于 token-mean)。
    loss_agg_mode: str = "seq-mean-token-mean"
    # --- GRPO advantage 口径 ---
    # grpo_loo(PGS 默认):baseline 严格 leave-one-out。
    # grpo:verl 原生口径(与历史 run 可比的对照臂)。
    adv_estimator: str = "grpo_loo"
    # scale 口径(审阅 IF-04 裁定,方案 A):默认 False = 不做组内 std 归一化
    # (Dr.GRPO 形态,scale≡1 与样本无关——严格满足规范条件 iv 且零爆炸)。
    # True 时 grpo_loo 用 max(std_loo, std_full),std_full 含当前样本,是
    # **有偏稳定化消融档**(孤例组的 scale 依赖自身 outcome),不作主档。
    norm_adv_by_std: bool = False
    # --- 从环境变量迁入(CD-006 requirement 3-F;用户 2026-07-26 裁定)---------
    # 这些原先散落在启动脚本的 export 里,而代码侧另有一套 fallback 默认值。
    # 两套真源 = 换一条启动路径就静默换一种行为:`run_a800_4b.sh` 设了
    # `TRACE_TEACHER_SCORING_BACKEND=fsdp`,而 162 主线用的 `run_experiment.sh`
    # 没设,代码 fallback 是 "vllm" ⇒ 打分会悄悄退回 R11 刚根治掉的 vLLM 路径。
    # 现在配置是唯一真源,launcher 从这里写 env,env 降级为纯进程间传递。
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
    max_ckpt_to_keep: int = 4       # 滚动保留的 actor ckpt 数(用户裁定 2026-08-04;
                                    # 单 ckpt ≈45G ⇒ 峰值 ~180G。旧值 3 曾硬编码在
                                    # launcher 里——参数真值必须在配置层可见)
    val_before_train: bool = True
    resume_mode: str = "auto"
    lr_scheduler_type: str = "constant"   # h3's ACTUAL schedule (its notes table
    lr_warmup_steps_ratio: float = 0.0    # saying cosine+0.03 was a misrecord)
    loggers: list[str] = field(default_factory=lambda: ["console", "swanlab"])
    agent_num_workers: int = 16     # **必须 = train_batch_size**(用户长期裁定):
                                    # 每 worker 恰好领一个任务的全部 rollout_n 副本
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
class SteeringCfg:
    """PGS steering 配置(规范 docs/pgs-final-spec.md)。enabled=False ⇒ 纯 bare
    rollout,与 GRPO 底座逐字节一致。"""
    enabled: bool = False
    # 缺省 static 保持历史配置/指纹语义；researcher 每 step 在线产文档。
    doc_source: str = "static"       # static | researcher
    gate_mode: str = "adaptive"    # 门控模式(机制 v3.2,用户裁定 2026-08-06):
                                   # adaptive = D 门 + ε 探针 + θ_D 率控自适应(历史行为);
                                   # always_on = 无门判定,每个候选位置恒 η=η_max
                                   # (maxopen 档:outcome reward + bare 臂锚的语境下
                                   # 允许最大程度干预;θ 标定/探针/率告警整体退役)。
                                   # 进 objective_fingerprint(改变行为分布 b)
    doc_table: str = ""            # static:task_id→doc 的 jsonl 表；researcher 可空
    eta_max: float = 0.5           # 倾斜强度上限
    c: float = 2.5                 # 截断参数(单点话语权限幅;目标语境实测标定)
    b_max: float = 20.0            # D_∞ 逐轨迹预算(nats;用户裁定 2026-08-04 v2:
                                   # 10→20,≈8 次整额干预,600-900 决策位的 1-2% 作用面;
                                   # per-token W 界 e^{±2ηc} 与本值无关)
    topk: int = 64                 # 保尾核候选集 S 的两侧 top-K
    obs_topk: int = 256            # draft/verify 的引擎观测窗(top-K 明细宽度);
                                   # launcher 联动 vLLM max_logprobs ≥ 本值(P0:两处
                                   # 必须同源,hook 请求超过 max_logprobs 会被引擎拒)
    theta_d: float = 0.0           # θ_D 的 EMA **初值**(标定 run P(1−q) 回填;0=未标定
                                   # 占位,启动校验拒 0;自适应关闭时即冻结阈值)
    gate_adaptive: bool = True     # 率控自适应开关(用户裁定 2026-08-04:θ_D 每 step
                                   # EMA 追踪目标门率;标定 run 必须置 false 冻结哨兵)
    gate_target_rate: float = 0.10 # 目标门率 q(用户裁定默认 10%;消融 {5,10,20}%)
    gate_ema_beta: float = 0.1     # EMA 慢速率 β
    epsilon: float = 0.05          # ε 探针概率
    gate_rate_bounds: tuple = (0.05, 0.25)   # spec 预注册区间
    r_max: int = 0                 # 每轮拒绝上限;0=无上限(v3.2 默认,用户裁定
                                   # 2026-08-06:干预不受轮内拒绝数限制,bare 臂+
                                   # outcome reward 锚定);≥1=历史 cap 行为态
    w_beh_mode: str = "per_token"  # per_token | per_decision | vtrace(消融开关,规范 §2.6)
    budget_pacing: bool = False    # 预算解锁曲线(2026-08-05 用户裁定,修 s1 头部坍缩;
                                   # docs/pgs-budget-pacing-design.md):false=旧行为
                                   # (t=0 全解锁)。S2 起 true
    budget_u0: float = 5.0         # pacing 初始拨付(nats;=2×δ_full,保头部纠错)
    budget_profile_power: float = 2.0  # 解锁曲线族 p:1=匀速,2=速率线性衰减(默认,
                                   # 用户提议档:预算重心前移但全程非零)
    steered_fraction: float = 1.0  # 分臂采样(机制 v3.1,用户裁定 2026-08-06):
                                   # 每任务 n 个 rollout 中 steered 臂占比,其余
                                   # 副本为纯 π 的 bare 臂(空账 W≡1),同组进
                                   # GRPO。1.0=历史全臂行为(不进指纹);<1 时
                                   # round(n×fraction) 为 steered 数,按
                                   # session_id 确定性分配(steering/arms.py)

    def __post_init__(self) -> None:
        # yaml 1.1 把 1.0e9(无符号指数)解析成字符串——数值字段一律强转,
        # 转不动就在这里死,而不是在校验/运行期比较时抛 TypeError。
        for f in ("eta_max", "c", "b_max", "theta_d", "epsilon",
                  "gate_target_rate", "gate_ema_beta",
                  "budget_u0", "budget_profile_power", "steered_fraction"):
            setattr(self, f, float(getattr(self, f)))
        self.topk = int(self.topk)
        self.obs_topk = int(self.obs_topk)
        self.r_max = int(self.r_max)


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
    steering: SteeringCfg
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


_KNOWN_SECTIONS = {
    "run", "research", "narration", "roles", "episode", "alfworld", "train", "eval", "steering"
}


def _validate(cfg: "Config") -> None:
    t = cfg.train
    if not t.model_path:
        raise ValueError("train.model_path is required")
    if t.rollout_backend not in ("vllm", "sglang"):
        raise ValueError(f"train.rollout_backend 必须是 vllm|sglang,得到 {t.rollout_backend!r}")
    if t.rollout_n < 1:
        raise ValueError("train.rollout_n must be >= 1")
    st = cfg.steering
    if st.doc_source not in ("static", "researcher"):
        raise ValueError(
            f"steering.doc_source 必须是 static|researcher,得到 {st.doc_source!r}"
        )
    if st.enabled:
        # 审阅 IF-13:数值面先过有限性——NaN/inf 能绕过大小比较,错误会
        # 延迟到 rollout 中段才爆。b_max 例外:+inf 是 maxopen 档的一等语义
        # (无限预算,v3.2),仍拒 NaN/−inf。
        for name in ("eta_max", "c", "theta_d", "epsilon"):
            if not math.isfinite(getattr(st, name)):
                raise ValueError(f"steering.{name} 必须是有限数,得到 {getattr(st, name)!r}")
        if math.isnan(st.b_max) or st.b_max == -math.inf:
            raise ValueError(f"steering.b_max 必须是非负数或 +inf,得到 {st.b_max!r}")
        if math.isinf(st.b_max) and st.budget_pacing:
            raise ValueError(
                "steering.b_max=inf 与 budget_pacing=true 不相容(解锁曲线无定义)"
                "——maxopen 档请显式 budget_pacing: false"
            )
        if st.gate_mode not in ("adaptive", "always_on"):
            raise ValueError(
                f"steering.gate_mode 必须是 adaptive|always_on,得到 {st.gate_mode!r}"
            )
        if st.gate_mode == "always_on":
            # maxopen 档参数纪律(参数真值必须显式可见,不做静默忽略):
            # 门语义整体退役 ⇒ 探针/率控自适应必须显式关闭,θ_D 免标定。
            if st.epsilon != 0.0:
                raise ValueError(
                    "gate_mode=always_on 时无门决定可翻转,须显式 epsilon: 0"
                )
            if st.gate_adaptive:
                raise ValueError(
                    "gate_mode=always_on 时 θ_D 无语义,须显式 gate_adaptive: false"
                )
        if st.doc_source == "static" and not st.doc_table:
            raise ValueError(
                "steering.enabled 且 doc_source=static 需要 doc_table(task_id→doc jsonl)"
            )
        if not (0.0 < st.eta_max < 1.0):
            raise ValueError(f"steering.eta_max 必须在 (0,1),得到 {st.eta_max}")
        if st.c <= 0 or st.b_max < 0:
            raise ValueError("steering.c 必须 > 0 且 b_max >= 0")
        if st.topk < 1:
            raise ValueError(f"steering.topk 必须 >= 1,得到 {st.topk}")
        if st.r_max < 0:
            raise ValueError(f"steering.r_max 必须 >= 0(0=无上限),得到 {st.r_max}")
        if st.budget_pacing:
            if st.budget_profile_power <= 0:
                raise ValueError(
                    f"steering.budget_profile_power 必须 > 0,得到 {st.budget_profile_power}")
            if not (0.0 <= st.budget_u0 <= st.b_max):
                raise ValueError(
                    f"steering.budget_u0 必须在 [0, b_max],得到 {st.budget_u0}")
        if st.gate_adaptive:
            if not (0.0 < st.gate_target_rate < 1.0):
                raise ValueError(f"steering.gate_target_rate 必须在 (0,1),得到 {st.gate_target_rate}")
            if not (0.0 < st.gate_ema_beta <= 1.0):
                raise ValueError(f"steering.gate_ema_beta 必须在 (0,1],得到 {st.gate_ema_beta}")
        rb = st.gate_rate_bounds
        if len(rb) != 2 or not (0.0 <= float(rb[0]) < float(rb[1]) <= 1.0):
            raise ValueError(
                f"steering.gate_rate_bounds 必须是 [lo, hi] 且 0<=lo<hi<=1,得到 {rb!r}"
            )
        if st.gate_mode == "adaptive" and st.theta_d <= 0:
            raise ValueError(
                "steering.theta_d 未标定(<=0)——先用目标文档语境预扫描 D 分布定 P85"
                "(gate_mode=always_on 免标定)"
            )
        if st.w_beh_mode not in ("per_token", "per_decision", "vtrace"):
            raise ValueError(f"未知 w_beh_mode {st.w_beh_mode!r}")
        # 分臂采样(v3.1):fraction ∈ (0,1];取整后 steered 臂至少 1 个副本
        # (=0 等价关 steering,应直接用 base 配置,拒绝歧义形态)。
        if not (0.0 < st.steered_fraction <= 1.0):
            raise ValueError(
                f"steering.steered_fraction 必须在 (0,1],得到 {st.steered_fraction}"
            )
        if st.steered_fraction < 1.0:
            from ..steering.arms import steered_arm_count

            k = steered_arm_count(t.rollout_n, st.steered_fraction)
            if k < 1:
                raise ValueError(
                    f"steering.steered_fraction={st.steered_fraction} 在 "
                    f"rollout_n={t.rollout_n} 下取整得 0 个 steered 副本——"
                    "全 bare 请直接用 steering.enabled=false 的 base 配置"
                )
            if k >= t.rollout_n:
                raise ValueError(
                    f"steering.steered_fraction={st.steered_fraction} 在 "
                    f"rollout_n={t.rollout_n} 下取整得整组 steered——"
                    "全臂请显式写 steered_fraction: 1.0"
                )
        if st.obs_topk < st.topk:
            raise ValueError(
                f"steering.obs_topk({st.obs_topk}) 必须 >= topk({st.topk})——"
                "观测窗小于 S 的 K 会让候选集静默缩水"
            )

        if st.doc_source == "researcher":
            rcfg = cfg.research
            role = cfg.roles.get("researcher")
            problems: list[str] = []
            if not rcfg.enabled:
                problems.append("research.enabled 必须为 true")
            if rcfg.backend != "external":
                problems.append("research.backend 必须为 external")
            if rcfg.tool_protocol != "fncall":
                problems.append("research.tool_protocol 必须为 fncall")
            if rcfg.gt != "off":
                problems.append("research.gt 必须为 off")
            if cfg.narration.backend != "engine":
                problems.append("narration.backend 必须为 engine")
            if t.train_batch_size != t.agent_num_workers:
                problems.append(
                    "train.train_batch_size 必须等于 agent_num_workers，保证同任务副本同 worker"
                )
            if t.agent_task_timeout_s and t.agent_task_timeout_s <= rcfg.session_timeout_s:
                problems.append(
                    "train.agent_task_timeout_s 必须为 0 或显著大于 research.session_timeout_s"
                )
            if not (0 < rcfg.k_default <= rcfg.k_max):
                problems.append("research 需要 0 < k_default <= k_max")
            if rcfg.max_turns < 1 or rcfg.wrapup_buffer < 0:
                problems.append("research.max_turns>=1 且 wrapup_buffer>=0")
            if rcfg.n_exp < 1 or rcfg.fetch_raw_limit < 0:
                problems.append("research.n_exp>=1 且 fetch_raw_limit>=0")
            if rcfg.fetch_raw_max_chars < 1 or rcfg.doc_max_words < 1:
                problems.append("research.fetch_raw_max_chars/doc_max_words 必须 >=1")
            if rcfg.researcher_max_tokens < 1:
                problems.append("research.researcher_max_tokens 必须 >=1")
            if not math.isfinite(rcfg.session_timeout_s) or rcfg.session_timeout_s <= 0:
                problems.append("research.session_timeout_s 必须是有限正数")
            if role is None:
                problems.append("roles.researcher 缺失")
            else:
                if role.mode != "external_pool":
                    problems.append("roles.researcher.mode 必须为 external_pool")
                if role.sidecar_sleep:
                    problems.append("roles.researcher.sidecar_sleep 必须为 false")
                if role.dp_size != 0:
                    problems.append("roles.researcher.dp_size 必须为 0")
                if not role.endpoint.strip():
                    problems.append("roles.researcher.endpoint 不能为空")
                if not role.model.strip():
                    problems.append("roles.researcher.model 不能为空")
                if not role.api_key_env.strip():
                    problems.append("roles.researcher.api_key_env 变量名不能为空")
                if not role.api_key:
                    problems.append(
                        f"roles.researcher.api_key_env={role.api_key_env!r} 未设置或为空"
                    )
                if role.max_concurrency < 1:
                    problems.append("roles.researcher.max_concurrency 必须 >=1")
                if role.max_tokens < 1 or role.timeout_s <= 0 or not math.isfinite(role.timeout_s):
                    problems.append("roles.researcher.max_tokens/timeout_s 必须为正")
                thinking = (
                    role.extra_body.get("chat_template_kwargs", {})
                    .get("enable_thinking")
                )
                if thinking is not False:
                    problems.append(
                        "roles.researcher.extra_body.chat_template_kwargs.enable_thinking 必须为 false"
                    )
            if problems:
                raise ValueError("invalid researcher doc source:\n  - " + "\n  - ".join(problems))


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides onto a parsed config dict.

    For settings that describe the EXPERIMENT rather than the machine — the
    steering ablation switches above all. Those belong in the config file
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
            raise ValueError(f"config override {item!r} must name a section, e.g. steering.eta_max=0.3")
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
    return {
        "missing": sorted(tpl_keys - exp_keys),
        "extra": sorted(exp_keys - tpl_keys),
    }


def _doc_table_digest(path: str) -> str:
    """doc 表**内容**摘要(审阅 IF-10:只哈希路径时同路径换内容不触发守卫)。
    文件不在本机时(如在本地 dry-run 服务器配置)返回哨兵——真正的 resume
    守卫发生在服务器上,那里 launcher 预检已保证文件存在。"""
    if not path:
        return ""
    p = Path(path)
    if not p.is_file():
        return "absent"
    return hashlib.sha1(p.read_bytes()).hexdigest()[:12]


def objective_fingerprint(cfg: "Config") -> str:
    """定义「优化什么」的短哈希(resume 守卫;机器无关)。
    steering 机制参数、advantage 口径与 doc 表内容全部参与:
    换其中任何一个 = 换优化目标(审阅 IF-10 扩面)。"""
    t, st = cfg.train, cfg.steering
    # static 路径必须逐字节保持 fa11dc8 的旧材料，既有 checkpoint 才能续跑；
    # researcher 另加 source 哨兵与非 secret 科学合同。
    material = "|".join(
        str(x) for x in (
            t.model_path, t.rollout_n, t.train_batch_size, t.lr,
            t.loss_agg_mode, t.adv_estimator, t.norm_adv_by_std,
            t.ppo_mini_batch_size,
            st.enabled, st.doc_table, _doc_table_digest(st.doc_table),
            st.eta_max, st.c, st.b_max, st.topk, st.obs_topk,
            st.theta_d, st.epsilon, st.r_max, st.w_beh_mode,
            st.gate_adaptive, st.gate_target_rate, st.gate_ema_beta,
            st.budget_pacing, st.budget_u0, st.budget_profile_power,
        )
    )
    # 分臂采样(v3.1)改变优化目标(组内一半样本换回 π 采样),必须进指纹;
    # 条件追加:fraction=1.0(历史全臂)不改 material,既有 run 指纹逐字节不变
    # (researcher source 哨兵同款先例)。
    if st.enabled and st.steered_fraction < 1.0:
        material += f"|steered_fraction={st.steered_fraction}"
    # maxopen 门控模式(v3.2)改变行为分布 b(处处倾斜),必须进指纹;
    # 条件追加:adaptive(历史行为)不改 material。
    if st.enabled and st.gate_mode != "adaptive":
        material += f"|gate_mode={st.gate_mode}"
    if st.doc_source == "researcher":
        rcfg, ncfg, role = cfg.research, cfg.narration, cfg.role("researcher")
        research_material = {
            "source": "researcher",
            "research": {
                "enabled": rcfg.enabled,
                "backend": rcfg.backend,
                "tool_protocol": rcfg.tool_protocol,
                "n_exp": rcfg.n_exp,
                "k_default": rcfg.k_default,
                "k_max": rcfg.k_max,
                "max_turns": rcfg.max_turns,
                "wrapup_buffer": rcfg.wrapup_buffer,
                "fetch_raw_limit": rcfg.fetch_raw_limit,
                "fetch_raw_max_chars": rcfg.fetch_raw_max_chars,
                "doc_max_words": rcfg.doc_max_words,
                "gt": rcfg.gt,
                "episode_temperature": rcfg.episode_temperature,
                "researcher_max_tokens": rcfg.researcher_max_tokens,
                "session_timeout_s": rcfg.session_timeout_s,
            },
            "narration": {
                "backend": ncfg.backend,
                "max_words": ncfg.max_words,
                "temperature": ncfg.temperature,
            },
            "role": {
                "model": role.model,
                "temperature": role.temperature,
                "max_tokens": role.max_tokens,
                "extra_body": role.extra_body,
                "mode": role.mode,
                "sidecar_sleep": role.sidecar_sleep,
                "dp_size": role.dp_size,
            },
        }
        import json

        material += "|" + json.dumps(
            research_material, ensure_ascii=False, sort_keys=True, separators=(",", ":")
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
        name: RoleCfg.from_dict(role_cfg or {})
        for name, role_cfg in (data.get("roles") or {}).items()
    }
    steering_raw = dict(data.get("steering") or {})
    if isinstance(steering_raw.get("gate_rate_bounds"), list):
        steering_raw["gate_rate_bounds"] = tuple(steering_raw["gate_rate_bounds"])
    cfg = Config(
        run=_build(RunCfg, data.get("run") or {}),
        research=_build(ResearchCfg, data.get("research") or {}),
        narration=_build(NarrationCfg, data.get("narration") or {}),
        roles=roles,
        episode=_build(EpisodeCfg, data.get("episode") or {}),
        alfworld=_build(AlfworldCfg, data.get("alfworld") or {}),
        train=_build(TrainCfg, data.get("train") or {}),
        eval=_build(EvalCfg, data.get("eval") or {}),
        steering=_build(SteeringCfg, steering_raw),
        raw=data,
    )
    _validate(cfg)
    return cfg
