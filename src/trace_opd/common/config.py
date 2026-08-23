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
    n_exp: int = 8                 # 每会话实验预算(用户裁定 2026-08-07:16→8,
                                   # 降 episode 开销;历史 12→16→8)
    k_default: int = 2             # 每次 run_experiment 的 episode 数(4→2)
    k_max: int = 4                 # 硬上限:tools.py 用 min(k_req, k_max),
                                   # 留大则模型填大即绕过 k_default
    max_turns: int = 30
    wrapup_buffer: int = 2
    fetch_raw_limit: int = 20
    fetch_raw_max_chars: int = 16000
    doc_max_words: int = 3000
    gt: str = "off"
    episode_temperature: float = 1.0
    researcher_max_tokens: int = 8192   # 单轮生成峰值上限(弹性,非典型量)
    session_timeout_s: float = 3600.0
    # §7.5(用户裁定 2026-08-08):上限(保险丝)与控制器解耦,两键**必配**
    # (research.enabled 时 >0,无默认推导)。
    session_token_budget: int = 0  # 会话流预算(生成+工具返回+结构,**不含
                                   # prompt**)= 实际控制器;超即截断轨迹
                                   # (length 优雅终局)。允许 < max_turns ×
                                   # researcher_max_tokens(单轮上限是峰值弹性,
                                   # 典型轮远小于它;4B 探针会话总量 p90=42k/
                                   # max=58.8k 是定值依据)
    max_prompt_tokens: int = 0     # researcher 首调 prompt(system+模板 tools
                                   # 块+kickoff)上限,超即 fail-loud——scaffold/
                                   # 工具描述膨胀的暴露面(4B 实测中位 1426/
                                   # max 1598);旧 model_len−2048 的 2048 是
                                   # 无引用魔数,已废


@dataclass
class ResearchRLCfg:
    """机制 v4.0(用户裁定 2026-08-06,docs/pgs-self-researcher-design.md):
    自体研究员双目标 RL——policy 本体扮演 researcher,每任务 research_n 个
    独立会话产 doc,以「教师在 bare 轨迹前缀上续写的成功比例」为 reward
    (前缀续写 = 在学生状态分布上评教师价值,DAgger 式评估分布),
    独立 GRPO 组 joint 反传。enabled=false ⇒ v3.2 现状逐字节不变。"""

    enabled: bool = False
    research_n: int = 8            # 每任务研究会话数 = research GRPO 组大小
    prefix_quantiles: list[float] = field(
        default_factory=lambda: [0.05, 0.15, 0.30, 0.50, 0.75]
    )                              # policy 生成 token(mask=1)累计比例分位;
                                   # 不对齐 turn 边界(半轮承接是特性)
    continuation_samples: int = 4  # 每 (doc, prefix) 续写次数 K_c(用户裁定默认 4)
    continuation_temperature: float = 1.0   # 续写采样温度(纯 p_T,无 SPS)
    researcher_temperature: float = 1.0     # 自体 researcher chat 温度(用户裁定
                                   # 默认 1.0;external 35B 模式仍用
                                   # roles.researcher.temperature)
    research_rl_weight: float = 0.1  # λ_r:research 行 advantage 标量乘(seq-mean
                                   # 行等权口径下即角色间权重比,无需分角色分母)
    research_warmup_steps: int = 0   # 该步数内 research 行 advantage 置 0(不反传)
    research_max_sample_tokens: int = 0  # research 行零梯度降级阈。**0(默认)=
                                   # 联动会话 token 墙**(max(4096,
                                   # rollout_max_model_len−2048)):凡在会话预算内
                                   # 完成的正常轨迹恒有全梯度(需求 §7.4,用户裁定
                                   # 2026-08-08——旧默认 24576 < 墙 38912,正常长
                                   # 会话被静默零梯度)。显式配 >0 才作防御降级
                                   # (超过 ⇒ mask 全 0,r 留组内 baseline,doc 照用)

    def __post_init__(self) -> None:
        for f in ("continuation_temperature", "researcher_temperature",
                  "research_rl_weight"):
            setattr(self, f, float(getattr(self, f)))
        self.research_n = int(self.research_n)
        self.continuation_samples = int(self.continuation_samples)
        self.research_warmup_steps = int(self.research_warmup_steps)
        self.research_max_sample_tokens = int(self.research_max_sample_tokens)
        self.prefix_quantiles = [float(q) for q in self.prefix_quantiles]


@dataclass
class NarrationCfg:
    """research 实验轨迹的叙述后端；engine 复用当前 SGLang rollout。"""

    backend: str = "engine"
    max_words: int = 1000          # prompt 指令词数(叙述篇幅要求),非解码上限
    temperature: float = 0.3
    max_tokens: int = 0            # 单次叙述解码上限;0 = 旧推导
                                   # max(1000, 3×max_words)(兼容)
    max_prompt_tokens: int = 0     # 叙述 prompt(system+轨迹渲染)上限,超出对
                                   # 轨迹渲染段截中重渲(叙述本就是压缩,截输入
                                   # 不 fail);research.enabled 时必配 >0


@dataclass
class SpreadsheetbenchCfg:
    """SpreadsheetBench(verified 400)专属配置;xlsx 数据与执行资源。
    部署事实走 env(SSB_DATA),不进 yaml 字面量。"""

    data_root: str = ""             # 任务 xlsx 根目录(必填,SSB_DATA)
    exec_concurrency: int = 32      # run_python 子进程全局并发门(按 worker
                                    # 折算;CPU 饥饿教训:环境子进程抢光 CPU
                                    # 会饿死推理引擎)
    python_timeout_s: int = 120     # 单次 run_python 超时
    recalc_timeout_s: float = 120.0  # 评分前 LibreOffice 公式重算超时
                                    # (官方 open_spreadsheet.py 语义;
                                    # soffice 路径走 SSB_SOFFICE/PATH)
    obs_max_chars: int = 3500       # 工具输出截断(< protocol 全局 4000,
                                    # 保 <tool_response> 包裹完整)
    #: 交互协议:fncall(CD-010 原档,run_python+submit 两工具,Qwen3 系
    #: 验证)| text(2026-08-17 用户裁定,经典线 Qwen2.5 直接 text 化:
    #: ```python 围栏 + ```output 回注 + ```submit 终局,one-shot 内嵌
    #: system;依据 = Math 域 FC→text +21.6pp 同为写代码任务、SQA 域
    #: Qwen2.5 FC 违规率 11.8% 实测、大段代码进 JSON 转义是已证痛点)。
    #: 默认保持 fncall(Math 同款纪律:裁定只在 configs/classic/ 显式
    #: 写落地,不静默改变旧配置行为)。
    protocol: str = "fncall"



@dataclass
class MathCfg:
    """Math benchmark(DAPO-17k × 六竞赛评测源)。无文件资产——任务全
    自足于 parquet extra_info,无 data_root。"""

    exec_concurrency: int = 32      # run_python 子进程全局并发门(按 worker 折算)
    python_timeout_s: int = 30      # 数学计算脚本超时(短;SSB 是 120)
    verify_timeout_s: float = 20.0  # math-verify 子进程硬超时(2026-08-13:
                                    # 病态表达式 sympy 满核自旋拖死整臂;
                                    # signal 超时装不进 to_thread ⇒ 子进程击杀)
    obs_max_chars: int = 3500       # 工具输出截断(保 <tool_response> 包裹)
    #: 交互协议:fncall(CD-010 原档,run_python+submit 两工具)| text
    #: (2026-08-13 用户裁定,经典线 Qwen2.5:```python 围栏 + ```output 回注
    #: + \boxed{} 终答,对照 ToRL/SimpleTIR/Qwen2.5-Math-TIR)| single
    #: (2026-08-16 用户裁定,对齐 MENTOR arXiv 2510.04140 ICLR2026 的单轮
    #: setting:无工具、无沙箱,一次 completion 即终局;<think></think> +
    #: \boxed{} 两标签)。默认保持 fncall(镜像 alfworld:裁定只在
    #: configs/classic/ 显式写落地,不静默改变旧配置行为)。
    protocol: str = "fncall"
    #: single 档的格式分权重。MENTOR 的 outcome:format = 9:1 ⇒ 0.1
    #: (论文 Appx C p18;代码 examples/reward_function/math.py 的
    #: `format_weight=0.1`,score = (1-w)·accuracy + w·format)。
    #: ⚠ >0 时 score 变连续值({0, w, 1-w, 1}),而 core 的
    #: `won = score >= 1.0` 会把「答对但格式错」(1-w) 判成失败 ⇒ 必须
    #: 同时开 train.reward_from_score,否则装配期 fail loud(_validate)。
    #: 置 0 = 纯二值答案正确率(格式仍记账进 valid/score,只是不进 reward)。
    #: 仅 single 档消费;fncall/text 档的终答走 submit,无格式分概念。
    format_weight: float = 0.1


@dataclass
class WebshopCfg:
    """WebShop(1000 商品社区 RL 小集;经典转向 2026-08-12 裁定,见
    envs/webshop/adapter.py)。无 data_root——环境是 162 上的薄 HTTP 服务
    (classic_bench/webshop_service.py),goal 经 parquet goal_idx 绑定。"""

    # 输出协议:"text"(<reasoning> + <action>search[q]/click[t]</action>,
    # 动作语法为 WebShop 原生、与 GiGPO/verl-agent 逐字同款)| "fncall"
    # (OpenAI 格式两工具 search/click)。2026-08-16 用户裁定默认 text,与
    # ALFWorld/SearchQA/Math 三域对齐:Qwen2.5 系对 FC 遵循不足(SearchQA
    # FC 档实测 11.8% 轨迹因连续格式违规判 DEGENERATE;Math 上 FC→text
    # 协议效应 +21.6pp),且两家相关工作在 WebShop 上都用文本动作。
    # 协议进 objective_fingerprint(无条件追加——WebShop 无历史训练 run)。
    protocol: str = "text"
    #: 服务基址。**逗号分隔可给多个分片** —— 单进程 stdlib http.server 受
    #: GIL 限制,正式训练的 128 级并发(16×8,文献口径见 OPS-LOG §9)需要多
    #: 进程分片;episode 按 goal_idx 取模确定性绑定到某个分片(sid 是分片
    #: 本地的,reset/step/release 必须同分片)。
    service_url: str = "http://127.0.0.1:8310"
    reset_timeout_s: float = 60.0   # /reset 超时(服务端 acquire 上限 30s)
    step_timeout_s: float = 30.0    # /step /release 超时
    request_concurrency: int = 32   # HTTP 并发门(≈服务端实例池大小;按
                                    # worker 折算)
    obs_max_chars: int = 3500       # 页面文本截断(商品列表页很长)
    preflight: bool = True          # 装配期 /meta 真实校验(fail loud)
    #: ★ 修复 vendored 判分器的 option 匹配 bug(2026-08-16 用户裁定,默认开)。
    #: 病灶:`get_reward` 把 `goal_options.items()`(tuple)传给期望 str 的
    #: `get_option_reward`,而 `normalize_color` 里的 `in` 在 str 上是子串、
    #: 在 tuple 上是成员 ⇒ 只有 purchased 侧被归一化,两侧对不上 ⇒
    #: **6.63% 的 goal 连 gold 答案自己都拿不到满分**(上限 0.667–0.889),
    #: 二值 reward 下 won 恒 False、该组零梯度。
    #: 修复后实测(400 goals):gold 满分率 94.1%→**100%**,错选项 p50 0.714、
    #: 错商品 p50 0.000 **均不变** ⇒ 区分度保持。
    #: ⚠ 这**改变 reward 定义** ⇒ 无条件进 objective_fingerprint;且与
    #: GiGPO/verl-agent/SEED 的口径断裂(他们跑的是有 bug 的版本),报数对拍
    #: 必须标注。开关必须与服务端 `--option-fix` 一致,preflight 逐分片校验。
    fix_option_matching: bool = True
    # 重试/退避:服务端池满回 429(忙,不是坏)⇒ 客户端指数退避而非判故障。
    # 2026-08-16 与服务端的 30s 快速 429 + sid TTL 回收配套落地(修 2026-08-12
    # 的 52/60 ENV_ERROR 池泄漏事故)。
    max_retries: int = 4
    retry_backoff_s: float = 1.0
    retry_backoff_max_s: float = 8.0


@dataclass
class SearchQACfg:
    """SearchQA(Search-R1 风格 wiki-search QA;经典转向 2026-08-12 裁定,
    见 envs/searchqa/adapter.py)。无 data_root——任务自足于 extra_info;
    检索是远端 HTTP 服务(162 wiki-18 + e5,只读共享)。"""

    # 输出协议:"text"(Search-R1/SEED 同款 <think>/<search>/<answer> 标签)
    # | "fncall"(OpenAI 格式两工具)。2026-08-13 用户裁定回退 text:
    # Qwen2.5 系(2024)对 function-call 遵循不足,FC 档实测 11.8% 轨迹因连续
    # 格式违规被判 DEGENERATE:format,而 4 步预算下一次失误即 25%;两家相关
    # 工作也都用文本标签并在 prompt 内置 one-shot。协议进 objective_fingerprint。
    protocol: str = "text"
    retriever_url: str = "http://127.0.0.1:8300"  # /retrieve 服务基址
    topk: int = 3                   # Search-R1 论文口径 topk=3
    # 单批检索超时(含 e5 编码)。2026-08-13 冒烟 v4:162 上 **35B researcher
    # 与 faiss retriever 共享同样 4 张 GPU**(每卡 35B 58.8G + 索引 10G),
    # 机制臂 128 个 researcher 会话打满 35B 时,retriever 响应从 0.1s 退化到
    # 12s。此时服务端只是**慢**不是**坏**(failure=0,重试均成功),而 30s
    # 超时会把慢误判成故障 ⇒ episode 白白重试、丢弃已生成 token。放宽到 90s
    # (batcher 等待上限 = 3×+30 = 300s),让排队自然消化。
    search_timeout_s: float = 90.0
    # 在途批请求上限(批量化后语义=并发"批"数,不是并发 query 数)。
    # ⚠ 2026-08-13 冒烟 v3 事故:此值曾为 64,双臂即 128 批 × 64 query =
    # 8192 query 同时在途,队列深度爆炸 ⇒ 服务端单发往返从 0.10s 退化到
    # 13.7s(136×),客户端大面积 batcher 超时。实测容量曲线(batch=64):
    #   在途 1/2/4/8/16/32 → 507/687/669/729/700/711 q/s
    #   中位延迟 0.10/0.18/0.39/0.68/1.19/2.09s
    # ⇒ 吞吐在在途 2 即饱和,再加只涨延迟不涨吞吐。
    # 再扫单请求批大小(服务端内部 retrieval_batch_size=512,原配 64 没吃满):
    #   批 64/128/256/512 × 在途2 → 632/897/1123/**1384** q/s
    # ⇒ 最优点 = **批 512 × 在途 2**(1384 q/s,中位 0.62s/最大 0.86s);
    # 双臂合计 4 批在途 × 512 = 2048 query,服务端仍在最优区。
    # 需求侧:每步 1024 eps × ~2.7 检索 = 2765 query ÷ 1384 ≈ 2s,
    # 占每步总时长(3-5 分钟)不到 1% ⇒ 检索已彻底不是瓶颈。
    search_concurrency: int = 2
    # --- micro-batching(2026-08-13 实测,162 retriever;用户裁定"环境侧
    # 吞吐调到最大以支撑高 RL rollout 并发")---------------------------------
    # 服务端 /retrieve 原生收 queries 数组,而原实现每 episode 单 query 单请求,
    # 吞吐被请求开销封死:并发 16→14.6 q/s、并发 256→15.7 q/s(拉并发无用)。
    # 批量后:batch8→90.5、batch32→331.9、**batch64 并发16→498.1 q/s(34×)**。
    # RL rollout 下 1024 episodes × ~2.7 次检索 = 2765 次/步:单 query 模式
    # 检索独占 184s/步,批量后 5.6s/步。
    search_max_batch: int = 512     # 单请求合并的 query 数;对齐服务端内部
                                    # retrieval_batch_size=512(实测 1.95× 于 64)
    search_batch_wait_ms: float = 15.0  # 攒批等待窗口;高并发下瞬间攒满,
                                    # 该值只在稀疏尾部生效(不拖长整体)
    obs_max_chars: int = 3500       # 工具输出截断(同 MathCfg 语义)
    preflight: bool = True          # 装配期真实检索一次(fail loud:服务
                                    # 不可达时零分会伪装成能力问题)


@dataclass
class SciworldCfg:
    """ScienceWorld(ETO 口径 24 任务;引入裁定 2026-08-08 见
    envs/scienceworld/adapter.py)。引擎 JAR 随 pip 包内置,无 data_root;
    每 episode 一个 JVM worker 子进程(本地实测 boot+load ≈1.2s,冷启)。"""

    worker_python: str = ""         # SCIWORLD_WORKER_PYTHON(装 scienceworld+
                                    # py4j 的 venv;PATH 需有 java>=1.8)
    simplifications: str = "easy"   # 社区两谱系一致口径(ETO/PGPO 硬编码、
                                    # RLVMR 默认;用户裁定跟随)。electrical
                                    # 任务引擎自动豁免 noElectricalAction
    env_step_limit: int = 100       # 引擎侧步数兜底(episode 预算仍由
                                    # episode.max_env_steps 主导,RLVMR 同款)
    spawn_concurrency: int = 64     # JVM boot 并发门(按 worker 折算;
                                    # CPU 饥饿教训同 alfworld)
    start_timeout_s: float = 120.0  # JVM boot+load(本地 1.2s,服务器留余量)
    step_timeout_s: float = 60.0
    boot_concurrency: int = 16      # build_dataset 期引擎 boot 并发


@dataclass
class TerminalCfg:
    """终端 CLI 域(TMax-15K × ACS 云沙箱;envs/terminal/adapter.py,引入
    裁定 2026-08-10 terminal-env-proposal.md)。部署事实走 env(TERMINAL_*),
    不进 yaml 字面量。协议面常量(截断 10k、单工具 bash、提交标记、wrapper
    路径)是 TMax 忠实字面量,故意**不做**配置——改它们=改任务表面。"""

    protocol: str = "fncall"        # 终端域 FC 原生,无 text 档
    backend: str = "acs"            # acs = 阿里云 Agent Sandbox(cloudbox);
                                    # docker = 162 环境池(dockerbox)。
                                    # A800 本机不可跑环境(LXC/CapEff=0/mount
                                    # 被拦,TMax 任务 100% 需 root),故 A800
                                    # 训练时**必须** docker + 162
    docker_host: str = ""           # backend=docker 时的 DOCKER_HOST;空则取
                                    # 环境变量。A800→162 用
                                    # unix:///tmp/docker162.sock(先跑
                                    # ~/dockfwd.sh 建转发;ControlMaster 必须关)
    snapshot_root: str = ""         # TERMINAL_SNAPSHOTS:per-task 环境快照目录
                                    # (<root>/<task_id>.tgz)。非空且该任务有
                                    # 快照 ⇒ **跳过 setup.sh**,改为起基底容器
                                    # + 解开快照(实测 provision 107s→~3s,
                                    # 文件树/md5/用户身份与原生 setup 逐一致)。
                                    # 复刻 TMax 官方"预建镜像"设计;快照在
                                    # docker 宿主本地,经 bind-mount 进容器,
                                    # 客户端零传输。缺快照的任务自动回落 setup.sh
    snapshot_index: str = ""        # TERMINAL_SNAPSHOT_INDEX:**客户端可见**的
                                    # 快照清单(每行一个 task_id)。必需项 ——
                                    # snapshot_root 是 **docker 宿主侧**路径
                                    # (bind-mount 由宿主解析),客户端(A800)
                                    # 看不到那个目录,不能用 is_file() 判存在;
                                    # 而 docker 对不存在的 bind 源会**自动建空
                                    # 目录**,静默把 tar 弄坏。索引为空时退回
                                    # 本地存在性检查(客户端==宿主的场景)
    task_data_root: str = ""        # TERMINAL_TASK_DATA(解包的 per-task 资产根:
                                    # <root>/<task_id>/{setup.sh,tests/,instruction.md})
    kubeconfig: str = ""            # TERMINAL_KUBECONFIG(集群 kubeconfig 路径)
    namespace: str = "default"
    image: str = ""                 # TERMINAL_IMAGE(ACR 胖基镜像全名:tag)
    image_pull_secret: str = "acr-zjk"
    run_label: str = ""             # Pod label trace-run=<值>(残留沙箱按标签清理)
    pod_cpu: str = "1"              # TMax 同款沙箱资源(convert_to_harbor.py:244-248)
    pod_memory: str = "2Gi"
    spawn_concurrency: int = 32     # 沙箱 Pod 创建全局并发门(按 worker 折算;
                                    # 保护 ACS 创建速率与 K8s API)
    create_timeout_s: float = 300.0  # 创建→Running(冒烟实测 23s 无预热池;
                                    # 胖基镜像首拉 GB 级,留余量)
    setup_timeout_s: float = 600.0  # 运行时 setup.sh(任务初始化,深读 §6.3)
    command_timeout_s: float = 120.0  # 单条 bash 命令(TMax Table 13)
    test_timeout_s: float = 600.0   # 判分沙箱内 timeout(TMax _MIN_TEST_TIMEOUT_S)
    test_grace_s: float = 300.0     # 判分外层墙钟余量:外层 = test_timeout_s +
                                    # 此值(默认 900,修 TMax 120<600 倒挂)
    #: 注入 episode 容器的环境变量(``K=V`` 逗号分隔)。TB 的 tests/test.sh
    #: 用 ``uvx -p 3.13 -w pytest…`` 判分,uv 默认从 github.com 取 Python
    #: standalone 与包 —— GitHub 在 162 不可达,不改走国内镜像则**每个 TB
    #: 任务 reward 恒 0**(2026-08-12 golden 实测:官方参考解也 0 分;注入后
    #: 同一任务 reward=1)。test.sh 一字不改。TMax 任务不消费这些变量。
    container_env: str = ""
    #: 注入容器 /etc/hosts 的条目(``host=ip`` 逗号分隔)。**为什么值得**:
    #: test.sh 无条件 `curl https://astral.sh/uv/...|sh`,该脚本再去 github.com
    #: 拉二进制;GitHub 不可达时 curl 要**空等 129 秒 TCP 超时**才放弃,而 uv
    #: 早已预置在镜像里 —— 这 129s 是每次判分的纯浪费。把 github.com 指到
    #: 127.0.0.1 后 curl 立即 refused,实测单次判分 **122-786s → 26s**
    #: (reward 仍正确产出 1)。当前无境外通路,黑洞化不损失能力;将来接了
    #: 代理清空此项即可。
    container_hosts: str = ""
    score_at_budget: bool = False   # False = TMax 训练语义(不提交 ⇒ 0,不跑
                                    # verifier);True = TB 式末步仍判分(评测近似)


@dataclass
class SweCfg:
    """SWE 域(R2E-Gym-Subset × ACS 云沙箱;envs/swe/,协商稿
    2026-08-11-swe-env-proposal.md)。三工具 FC(str_replace_editor/
    execute_bash/submit,openhands 套件)+ 容器内 pytest 判分(judge.py
    移植 R2E _calculate_reward_r2e,二值)。特权(GT patch/期望输出)在
    privileged_root,judge 从此读 expected_output_json,不进 agent 面。"""

    protocol: str = "fncall"        # SWE 域 FC 原生
    data_root: str = ""             # SWE_DATA(dataset 根:train.parquet +
                                    # privileged/<instance_id>.json)
    kubeconfig: str = ""            # SWE_KUBECONFIG(集群 kubeconfig)
    namespace: str = "default"
    image_registry: str = ""        # SWE_IMAGE_REGISTRY(ACR 前缀,拼
                                    # <registry>/<repo>_final:<hash>;空则用
                                    # 数据集原 docker_image = dockerhub)
    image_pull_secret: str = "acr-zjk"
    run_label: str = ""             # Pod label trace-run=<值>(残留清理)
    pod_cpu: str = "2"              # SWE 单任务 ~1 core(NeMo Gym 实测)+ 余量
    pod_memory: str = "4Gi"
    spawn_concurrency: int = 16     # 沙箱创建全局并发门
    create_timeout_s: float = 300.0  # 创建→Running(SWE 镜像 0.3-0.5GB 压缩)
    command_timeout_s: float = 120.0  # 单条工具命令(R2E CMD_TIMEOUT=120)
    setup_timeout_s: float = 300.0  # episode 开局容器内 setup(symlink/pyc 清理/
                                    # r2e_tests 归位;raw 镜像必需,见 adapter)
    test_timeout_s: float = 300.0   # 判分 pytest 超时(R2E reward_timeout=300)
    test_grace_s: float = 60.0      # 判分外层墙钟余量(外层=test_timeout+此值)
    max_obs_chars: int = 10000      # observation 截断(R2E 工具内部 10k 字符
                                    # 同量级;execute_bash 上游不截断,我们补)
    prebaked_setup: bool = False    # True = 镜像已烘 setup(M2 后),开局只按
                                    # tests_visible 建/撤软链;False = raw 镜像
                                    # 走完整 setup(每 episode 一次合并 exec)
    budget_exhaust_reward_scale: float = 0.5  # R2E α:预算耗尽仍判分,分数 ×α
                                    # (先例 0.5,动机=约 20% 轨迹撞墙、其中
                                    # 近 1/4 已修好;设 1.0 关闭惩罚)
                                    # ⚠ 影响效果的参数,档位待用户确认
    grading_mode: str = "deferred"  # swebench(Verified)判分:deferred =
                                    # episode 只收 patch、分数离线由官方
                                    # harness 出(**报表口径**,训练环境零
                                    # swebench 依赖);inline = 沙箱内跑官方
                                    # eval_script 判(给 in-loop val 曲线,
                                    # 需装 swebench 包)。r2e 源不受此影响
    tests_visible: bool = True      # True = R2E 训练 setting(r2e_tests 软链
                                    # 对 agent 可见,DeepSWE/SWE-Master 同款);
                                    # False = 对齐 SWE-bench Verified 评测口径
                                    # (F2P 测试不可见),供消融
    allow_git: bool = False         # git 封禁(先例意图=agent 完全无 git);
                                    # True 仅供调试,正式实验恒 False

    def _validate(self) -> None:
        for k in ("spawn_concurrency", "max_obs_chars"):
            if getattr(self, k) <= 0:
                raise ValueError(f"swe.{k} must be > 0")
        for k in ("command_timeout_s", "test_timeout_s", "create_timeout_s",
                  "setup_timeout_s"):
            if getattr(self, k) <= 0:
                raise ValueError(f"swe.{k} must be > 0")
        if self.grading_mode not in ("deferred", "inline"):
            raise ValueError(
                f"swe.grading_mode 必须是 deferred|inline,得到 {self.grading_mode!r}"
            )
        if not 0.0 <= self.budget_exhaust_reward_scale <= 1.0:
            raise ValueError(
                "swe.budget_exhaust_reward_scale 必须在 [0,1](R2E α;1.0=不罚)"
            )


@dataclass
class WebArenaCfg:
    """WebArena 域(WebArena-Verified 812 × 站点池;envs/webarena/,引入
    裁定 2026-08-10 webarena-env-proposal.md §9)。评测桥钉死
    webarena-verified==1.2.3(evalbridge.PINNED_VERSION);交互面 = WebServ
    step_browser 语义 + terminate 结构化终局(FinalAgentResponse)。"""

    protocol: str = "fncall"        # 单工具 step_browser,无 text 档
    pool: str = "static"            # static(固定 URL,冒烟/只读)| acs
                                    # (Checkpoint/Restore SandboxSet 池,隔离)
    #: pool=static:site → base URL(如 {"shopping": "http://localhost:7770"})
    static_base_urls: dict[str, str] = field(default_factory=dict)
    #: pool=acs:site → SandboxSet 名(每站点一池,RUNBOOK §12.5 配方)
    sandboxsets: dict[str, str] = field(default_factory=dict)
    kubeconfig: str = ""            # WEBARENA_KUBECONFIG(pool=acs 必填)
    namespace: str = "default"
    work_dir: str = ""              # episode 产物根(HAR/agent_response/eval)
    run_label: str = ""             # 领用注解标识 + 产物归属
    headless: bool = True
    spawn_concurrency: int = 16     # 并发浏览器会话门(chromium ~200MB/个)
    acquire_timeout_s: float = 300.0  # 站点池领用等待上限(池竭即 fail)
    #: 单条观测字符上限(head 60% + tail 40%)。WebServ 不截断、靠 16k ctx
    #: 硬约束;此值影响效果,默认 40000 为工程保底——正式训练值待用户裁定。
    max_obs_chars: int = 40_000
    #: HAR 保留策略:False = 评测后即删(实测 4-8MB/episode,训练会线性
    #: 吃盘;agent_response/eval_result 小件始终保留);True = 全保(审计/
    #: 提交包场景)。
    keep_har: bool = False
    #: static 池对 MUTATE 任务的显式放行(默认拒绝:静态池无状态隔离,
    #: mutate 互踩;只有单任务串行调试才可打开)。
    allow_static_mutate: bool = False


@dataclass
class EnvironmentCfg:
    """任务环境选择(envs/registry.py 的注册键)。机制层永不感知具体
    benchmark;装配层凭 name 分派工厂。默认 alfworld = 历史行为,且默认值
    不进 objective_fingerprint(条件追加,保既有 run 指纹逐字节不变)。"""

    name: str = "alfworld"


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
    #: §7.6(用户裁定 2026-08-08):executor 流预算独立于 verl padding 维度
    #: ——train.response_length 回归单一职责(容量/padding 维度,须 ≥ 各角色
    #: 流预算的 max)。0 = 跟随 train.response_length(历史等值行为的显式化,
    #: 非魔数推导);>0 = 显式 executor 流预算(如 executor 16K + researcher
    #: 48K 时此键 16384、response_length 49152)。
    session_token_budget: int = 0
    max_tokens_per_turn: int = 2048
    #: Qwen3.5 等 thinking-family 模板的思考模式(engine 侧全局:executor /
    #: v4 自体 researcher / 叙述共用 render_ids)。False = 预填空 think 块
    #: (模型直接出正文);True = 模型自己写思考再出正文。
    #: 改变每轮生成的形状 ⇒ 进 objective_fingerprint。
    #: ⚠ 开启会显著抬高 token 消耗(历史实测同答案 8 → 167 token),
    #: session_token_budget / max_tokens_per_turn 须随之重估。
    enable_thinking: bool = False
    #: 跨 benchmark 的推理标签(输出协议面;真源 core/protocol.REASONING_TAG,
    #: 入口经 configure_reasoning_tag 注入)。2026-08-16 用户裁定:全线弃用
    #: `<think>` 统一 `<reasoning>` —— thinking-family 模板(Qwen3.5/3.6,含
    #: 拟用作分布干预教师的 35B)会**剥离历史 assistant 轮的 `<think>` 块**
    #: (推理静默丢失),且其 non-thinking 模式是预填空 think 块、与"模型自己
    #: 写 `<think>`"直接冲突;`<think>` 在该族词表里还是特殊 token。
    #: 取证:scripts/probe_think_tags.py、scripts/probe_strip_scope.py。
    #: 设成 "think" 可回退(复现历史 run / 与相关工作逐字对拍),会打告警。
    #: ⚠ 改的是任务表面 ⇒ 进 objective_fingerprint(按 env 的历史默认条件
    #:   追加,既有 run 指纹逐字节不变;见 objective_fingerprint)。
    reasoning_tag: str = "reasoning"


@dataclass
class AlfworldCfg:
    #: 输出协议:"text"(<reasoning>/<action> ReAct 协议)| "fncall"(统一
    #: Function Call)。协议改变任务表面(system prompt/输出格式/观察包裹),
    #: 进 objective_fingerprint(条件追加,text 保历史指纹)。
    #:
    #: 2026-08-13 用户裁定:经典 benchmark 线(Qwen2.5-7B × ALFWorld)**回退
    #: text/ReAct**,不再用 FC——与相关工作对齐(verl-agent/GiGPO、OPID
    #: 〔On-Policy Skill Distillation,arXiv 2606.26790〕、SEED 三家 ALFWorld
    #: prompt 逐字节同源,均为 <think>+<action> 文本协议;RAGEN 为
    #: <think>+<answer>;ReAct/Reflexion 原版为 "> think:"/裸命令)。我方标签
    #: 用 <reasoning> 而非 <think> 是硬约束:连续对话流下 Qwen chat template
    #: 会剥离历史轮 content 里的字面 <think> 块(见 core/protocol.py 头注),
    #: 相关工作每轮重建 prompt 故不受影响。
    #: **默认值仍保持 fncall**:configs/experiments/ 下 15 个 PGS 线历史配置
    #: 不写该键、吃默认值,改默认会把它们静默切协议(param-truth 第四层事故
    #: 型)。裁定只在 configs/classic/ 两臂显式写 `protocol: text` 落地。
    protocol: str = "fncall"
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
    use_fused_kernels: bool = False   # verl fused linear+CE(logits 不落地)。
                                      # Qwen3.5 248k 词表 × 长序列的训练显存刚需
                                      # (32k×248k bf16 ≈ 16G/条);数学等价执行档,
                                      # 不进 objective_fingerprint。verl 侧对
                                      # qwen3_5 有专用实现(monkey_patch.py:270)
    fused_kernels_backend: str = "torch"  # torch | triton(仅 use_fused_kernels
                                      # 时消费;torch 首发保守档,triton 待验)
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
    #: 训练 reward 的取数口径。False(历史行为,全部既有 run)= 二值
    #: `1.0 if traj.won else 0.0`;True = 直接用 `traj.score`(ActionOutcome.score
    #: 透传的连续终评)。
    #: 2026-08-16 用户裁定引入,为 math single 档对齐 MENTOR 的
    #: `0.9·accuracy + 0.1·format` —— 二值通道会把 0.9(答对/格式错)判成 0,
    #: 比 MENTOR 更严。开启后 won 退化为纯观察指标(仍是 score>=1.0),
    #: 曲线要看 rollout_extra/score。
    #: ⚠ 改的是优化目标本身 ⇒ 进 objective_fingerprint(条件追加,
    #:   False 时不改 material,既有 run 指纹逐字节不变)。
    #: ⚠ 只有 score 天然连续的环境该开;二值环境开了等价于原行为(score∈{0,1})。
    reward_from_score: bool = False


@dataclass
class EvalCfg:
    response_budget: int = 32768
    temperature: float = 0.0
    #: 训练内 val 的每题采样次数 → verl val_kwargs.n;指标名随之变成
    #: `mean@N`(metric_utils.py:1093 按实际 n_resps 命名),N>1 时 verl 另给
    #: best@N / worst@N。1 = 历史 greedy 单次口径。
    #: ⚠ N>1 必须配 temperature>0,否则 N 次贪心解码逐字节相同、平均无意义。
    n: int = 1


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
    gate_mode: str = "adaptive"    # 门控模式(v3.2 硬门档 + v3.3 软强度档,
                                   # docs/pgs-v33-mixed-policy-design.md M4):
                                   # adaptive = D 硬门 + ε 探针 + θ_D 率控自适应(历史行为);
                                   # always_on = 无门判定,每个候选位置恒 η=η_max(maxopen);
                                   # entropy = G-E 软强度 w=min(1, h/H_q)(h=学生截断熵,
                                   #   draft 侧免费;w<w_min 的整段跳过教师 verify);
                                   # d_var = G-D 软化 w=min(1, D/θ_D)(保教师异议信息,
                                   #   verify 先行,阈值自适应复用 D 门基建);
                                   # soft_or = G-SOR w=1−(1−w_E)(1−w_D)(两盲区互补,
                                   #   需两侧信号,无 bypass 红利)。
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
    epsilon: float = 0.0   # ε 探针默认关(2026-08-17 用户裁定;因果归因读数按需显式开)          # ε 探针概率
    gate_rate_bounds: tuple = (0.05, 0.25)   # spec 预注册区间
    r_max: int = 0                 # 每轮拒绝上限;0=无上限(v3.2 默认,用户裁定
                                   # 2026-08-06:干预不受轮内拒绝数限制,bare 臂+
                                   # outcome reward 锚定);≥1=历史 cap 行为态
    executor: str = "python"       # SPS 循环宿主(Engine-Resident SPS P1,用户
                                   # 裁定 2026-08-10;docs/plans/2026-08-10-
                                   # engine-resident-sps-design.md):
                                   #   python = 训练进程内循环,每周期 2 次
                                   #     RPC(历史行为);
                                   #   engine = server actor 进程内循环,每
                                   #     turn 1 次 RPC(仅 sglang 后端)。
                                   # 纯执行层,采样分布逐位恒等 ⇒ 不进指纹;
                                   # 切换前须过对拍门(设计文档 §4)。
    draft_horizon: int = 0         # bounded K:单次 draft 的 token 上限(0=
                                   # 整段)。拒绝丢弃 suffix 的上界即 K−1——
                                   # SSB 近平方放大的解。仅 engine 档允许
                                   # 非零:HTTP 路径上小 K = 每周期固定往返
                                   # 成本 × 周期数暴涨(CD-006 已证)。
    w_beh_mode: str = "per_token"  # per_token | per_decision | vtrace | off |
                                   # shaped_reject(v3.3 M3,spec §2.6 版本演进):
                                   # off = steered 行为分布免 IS(MENTOR 口径,ratio
                                   #   分母统一 π_old;偏差由 asym advantage 的 [·]₊
                                   #   与 α 承接)——v3.3 推荐默认;
                                   # shaped_reject = 仅拒绝位乘 f(π)=π/(π+γ)(LUFFY
                                   #   反向加权:低概率教学位拿更大梯度),接受位=1
    w_beh_shaped_gamma: float = 0.1  # shaped_reject 的 γ(LUFFY 实证默认 0.1)
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
    # ---- v3.3 M1/M2:不对称 advantage 与 α 有限期调度(用户裁定 2026-08-07;
    # docs/pgs-v33-mixed-policy-design.md)----
    adv_mode: str = "legacy_shared"  # legacy_shared = v3.1 口径(两臂同组共享
                                   # LOO 基线;既有 run 语义,指纹不变);
                                   # asym = bare 组内 LOO(no-std 不变)+
                                   # steered=[R−mean_bare]₊×α_t(失败零梯度
                                   # 不惩罚;MENTOR Eq.13 口径,s3 弥散化病根
                                   # 的结构性修复)。进 objective_fingerprint;
                                   # asym 要求 train.adv_estimator=pgs_asym_grpo
    steered_adv_decay_mode: str = "cosine_steps"  # 调度族枚举(设计 M2 留位;
                                   # 当前唯一实现 cosine_steps,后续自适应版
                                   # 加枚举值不动结构)。asym 档进指纹
    steered_adv_weight: float = 1.0    # α_0(asym 档 steered 超额的权重初值)
    steered_adv_decay_steps: int = 0   # T_decay:α_t=α_0·½(1+cos(π·t/T)),
                                   # t≥T 后恒 0 **且 steered 臂停产**(教师=
                                   # 有限期脚手架,MENTOR 同款;建议取总步数
                                   # 2/3 给副动作族慢热留窗)。0=不衰减
                                   # (α 恒 α_0,steered 不停产)
    steered_adv_cap: float = 0.0   # 组内 steered advantage 总额上限(>0 激活;
                                   # MENTOR expert_cap 同款防御闸,默认关)
    # ---- v3.3 M4:软强度门控参数(gate_mode=entropy|d_var|soft_or)----
    entropy_target_rate: float = 0.05  # 满强度位置的目标占比(阈值 H_q 追踪
                                   # 熵分布的 P_{1−rate};MENTOR entropy_p=0.95
                                   # 即 rate=0.05)
    entropy_theta_init: float = 0.0    # H_q 起点。0 = 自标定(默认;warmup
                                   # 哨兵起步、零干预,首个 step 边界从真实熵
                                   # 分布分位落地——MENTOR 首步 γ_p=999 同款,
                                   # 无 benchmark 锚定);>0 = 显式标定起点。
                                   # ⚠ 绝对 nats 值跨 benchmark/模型不可迁移
                                   # (1.28 占位值事故,2026-08-11 诊断)。
    entropy_theta_update: str = "direct"  # H_q 更新方式:direct=分位直标
                                   # (每 step set,无 EMA 时延;通用化默认)
                                   # | ema=历史 robust-EMA 慢更(复现旧 run;
                                   # 要求显式 theta_init 与 floor)。进指纹。
    entropy_theta_ratchet: bool = True  # direct 档:θ 不低于首次标定值——
                                   # 策略熵坍缩时干预自动淡出而非门变宽,
                                   # 取代绝对值 floor(自标定,无量纲)。
                                   # 仅 direct 档生效;进指纹。
    entropy_floor: float = 0.0     # H_q 硬地板(仅 ema 档;防阈值追着熵坍缩
                                   # 跑,MENTOR max(q80,1.5) 先例;ema 档须 >0,
                                   # direct 档必须保持 0——ratchet 取代)
    intensity_w_min: float = 0.05  # w<w_min 的位置视为关门(bypass;entropy 档
                                   # 整段全 bypass 时跳过教师 verify——工程红利)

    def __post_init__(self) -> None:
        # yaml 1.1 把 1.0e9(无符号指数)解析成字符串——数值字段一律强转,
        # 转不动就在这里死,而不是在校验/运行期比较时抛 TypeError。
        for f in ("eta_max", "c", "b_max", "theta_d", "epsilon",
                  "gate_target_rate", "gate_ema_beta",
                  "budget_u0", "budget_profile_power", "steered_fraction",
                  "steered_adv_weight", "steered_adv_cap", "w_beh_shaped_gamma",
                  "entropy_target_rate", "entropy_theta_init", "entropy_floor",
                  "intensity_w_min"):
            setattr(self, f, float(getattr(self, f)))
        self.steered_adv_decay_steps = int(self.steered_adv_decay_steps)
        # YAML 1.1 陷阱(审阅 L17):不带引号的 `w_beh_mode: off` 被解析成
        # 布尔 False(CLI override 经 yaml.safe_load 同样中招)——按用户意图
        # 归一;True 无对应档,落进枚举校验被拒。
        if self.w_beh_mode is False:
            self.w_beh_mode = "off"
        self.topk = int(self.topk)
        self.obs_topk = int(self.obs_topk)
        self.r_max = int(self.r_max)


@dataclass
class Config:
    run: RunCfg
    research: ResearchCfg
    research_rl: ResearchRLCfg
    narration: NarrationCfg
    roles: dict[str, RoleCfg]
    episode: EpisodeCfg
    environment: EnvironmentCfg
    alfworld: AlfworldCfg
    spreadsheetbench: SpreadsheetbenchCfg
    math: MathCfg
    searchqa: SearchQACfg
    webshop: WebshopCfg
    scienceworld: SciworldCfg
    terminal: TerminalCfg
    webarena: WebArenaCfg
    swe: SweCfg
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
    "run", "research", "research_rl", "narration", "roles", "episode", "alfworld",
    "train", "eval", "steering", "environment", "spreadsheetbench", "math",
    "scienceworld", "terminal", "webarena", "swe", "searchqa", "webshop",
}


def _validate(cfg: "Config") -> None:
    t = cfg.train
    if not t.model_path:
        raise ValueError("train.model_path is required")
    if cfg.alfworld.protocol not in ("fncall", "text"):
        raise ValueError(
            f"alfworld.protocol 必须是 fncall|text,得到 {cfg.alfworld.protocol!r}"
        )
    if cfg.math.protocol not in ("fncall", "text", "single"):
        raise ValueError(
            f"math.protocol 必须是 fncall|text|single,得到 {cfg.math.protocol!r}"
        )
    if cfg.webshop.protocol not in ("fncall", "text"):
        raise ValueError(
            f"webshop.protocol 必须是 fncall|text,得到 {cfg.webshop.protocol!r}"
        )
    if cfg.environment.name == "math" and cfg.math.protocol == "single":
        # 单轮语义:一次 completion 即终局(adapter 恒 done=True)。caps 大于 1
        # 不会改变行为,但会让配置读起来像多轮 —— 参数真值必须一眼可读
        # (记忆 config-vs-manifest-truth),故显式钉死。
        if (cfg.episode.max_env_steps, cfg.episode.max_assistant_turns) != (1, 1):
            raise ValueError(
                "math.protocol=single 是单轮协议,episode.max_env_steps 与 "
                "max_assistant_turns 必须都为 1,得到 "
                f"{cfg.episode.max_env_steps}/{cfg.episode.max_assistant_turns}"
            )
        if not 0.0 <= cfg.math.format_weight < 1.0:
            raise ValueError(
                f"math.format_weight 必须在 [0,1),得到 {cfg.math.format_weight}"
            )
        if cfg.math.format_weight > 0.0 and not cfg.train.reward_from_score:
            # 静默降级的后果:格式错但答对的样本 score=0.9 ⇒ won=False ⇒
            # reward=0,format 分不但没生效,还把正确答案打成完全失败。
            raise ValueError(
                f"math.format_weight={cfg.math.format_weight} > 0 需要同时设 "
                "train.reward_from_score=true —— 否则 core 的 won=score>=1.0 "
                "会把「答对但格式错」(score=1-w) 判成 reward 0,比 MENTOR 更严"
            )
    if cfg.terminal.protocol != "fncall":
        raise ValueError(
            f"terminal.protocol 仅支持 fncall(TMax 语料即 FC 单工具 bash,"
            f"无 text 档),得到 {cfg.terminal.protocol!r}"
        )
    if cfg.terminal.backend not in ("acs", "docker"):
        raise ValueError(
            f"terminal.backend 必须是 acs|docker,得到 {cfg.terminal.backend!r}"
        )
    if cfg.terminal.test_timeout_s < 600.0:
        raise ValueError(
            f"terminal.test_timeout_s({cfg.terminal.test_timeout_s}) 必须 >= "
            "600(TMax _MIN_TEST_TIMEOUT_S;更小会把慢 verifier 伪装成任务失败)"
        )
    if cfg.terminal.test_grace_s < 60.0:
        raise ValueError(
            f"terminal.test_grace_s({cfg.terminal.test_grace_s}) 必须 >= 60"
            "(外层墙钟必须显著大于沙箱内判分 timeout,修 TMax 倒挂)"
        )
    if cfg.webarena.protocol != "fncall":
        raise ValueError(
            f"webarena.protocol 仅支持 fncall(单工具 step_browser,无 text "
            f"档),得到 {cfg.webarena.protocol!r}"
        )
    if cfg.webarena.pool not in ("static", "acs"):
        raise ValueError(
            f"webarena.pool 必须是 static|acs,得到 {cfg.webarena.pool!r}"
        )
    if cfg.swe.protocol != "fncall":
        raise ValueError(
            f"swe.protocol 仅支持 fncall(R2E openhands 三工具,无 text 档),"
            f"得到 {cfg.swe.protocol!r}"
        )
    cfg.swe._validate()
    import math as _math
    if cfg.webarena.spawn_concurrency < 1:
        raise ValueError("webarena.spawn_concurrency 必须 >= 1")
    if (not _math.isfinite(cfg.webarena.acquire_timeout_s)
            or cfg.webarena.acquire_timeout_s <= 0):
        raise ValueError("webarena.acquire_timeout_s 必须是有限正数")
    if cfg.webarena.max_obs_chars < 0:
        raise ValueError("webarena.max_obs_chars 必须 >= 0(0=不截断)")
    if cfg.environment.name == "webarena" and cfg.webarena.pool == "static":
        # 静态池无 episode 间状态隔离:mutate 任务并发互踩。只拦作为训练
        # 环境的使用;冒烟脚本直构工厂不经此路径。
        raise ValueError(
            "webarena.pool='static' 不可用于训练(无状态隔离,mutate 任务"
            "互踩);训练用 pool='acs'(Checkpoint/Restore 池)"
        )
    # §7.6:executor 流预算(episode.session_token_budget,0=跟随维度)必须
    # 被 padding 维度容下(verl 只补不截)——与 research 侧同款不变式。
    _exec_budget = cfg.episode.session_token_budget or cfg.train.response_length
    if _exec_budget > cfg.train.response_length:
        raise ValueError(
            f"episode.session_token_budget({cfg.episode.session_token_budget})"
            f" 超过 train.response_length({cfg.train.response_length})——"
            "executor 行进该 padding 维度且 verl 只补不截;维度应配为各角色"
            "流预算的 max"
        )
    # §7.1/§7.5/§7.6 预算关系陈述(信息,不评判):单轮上限的职责是允许个别
    # 轮打到峰值(弹性),典型轮远小于它——总预算 < 轮×单轮乘积是常态设计。
    _turn_budget = cfg.episode.max_assistant_turns * cfg.episode.max_tokens_per_turn
    if cfg.train.response_length:
        who = ("executor 流预算兜底" if _exec_budget < _turn_budget
               else "轮×单轮乘积先到")
        print(
            f"[config] budget: executor 流预算={_exec_budget}"
            f"(episode.session_token_budget"
            f"{'=0→跟随维度' if not cfg.episode.session_token_budget else ''}),"
            f"轮×单轮乘积={_turn_budget}({who});padding 维度="
            f"train.response_length={cfg.train.response_length},"
            f"CPU 内存 ∝ 行数×维度。"
        )
    # eval 采样口径自洽(用户裁定 2026-08-09 引入 avg@N):
    if cfg.eval.n < 1:
        raise ValueError(f"eval.n 必须 >=1,得到 {cfg.eval.n}")
    if cfg.eval.n > 1 and cfg.eval.temperature <= 0:
        raise ValueError(
            f"eval.n={cfg.eval.n} > 1 要求 eval.temperature > 0(得到 "
            f"{cfg.eval.temperature}):温度为 0 时 N 次贪心解码逐字节相同,"
            "avg@N 退化为 mean@1 却多付 N 倍评测成本"
        )
    if t.rollout_backend not in ("vllm", "sglang"):
        raise ValueError(f"train.rollout_backend 必须是 vllm|sglang,得到 {t.rollout_backend!r}")
    if t.fused_kernels_backend not in ("torch", "triton"):
        raise ValueError(
            f"train.fused_kernels_backend 必须是 torch|triton,得到 {t.fused_kernels_backend!r}"
        )
    if t.rollout_n < 1:
        raise ValueError("train.rollout_n must be >= 1")
    st = cfg.steering
    if st.doc_source not in ("static", "researcher"):
        raise ValueError(
            f"steering.doc_source 必须是 static|researcher,得到 {st.doc_source!r}"
        )
    if not st.enabled and (
        t.adv_estimator == "pgs_asym_grpo" or st.adv_mode not in ("", "legacy_shared")
    ):
        # 反向联动必须站在 enabled 门之外(审阅 L3/L16:原先整块在
        # `if st.enabled:` 内,steering 关掉即绕过"两处必须联动"守卫——
        # estimator 会在无臂标签的 batch 上把全部行当 unknown 训练)。
        raise ValueError(
            "steering.enabled=false 与 pgs_asym_grpo/adv_mode=asym 不相容——"
            "asym advantage 需要分臂标签,请开 steering 或换回 grpo_loo"
        )
    # Engine-Resident SPS(P1)执行档:st.enabled 门外校验——错误组合在
    # steering 关闭时同样是配置矛盾,应当即拒。
    if st.executor not in ("python", "engine"):
        raise ValueError(
            f"steering.executor 必须是 python|engine,得到 {st.executor!r}"
        )
    if st.draft_horizon < 0:
        raise ValueError(f"steering.draft_horizon 必须 >= 0,得到 {st.draft_horizon}")
    if st.executor == "python" and st.draft_horizon > 0:
        raise ValueError(
            "steering.draft_horizon > 0 仅在 executor=engine 下允许——HTTP 路径"
            "上的小 K 是每周期固定往返成本 × 周期数暴涨(CD-006 吞吐分析已证);"
            "python 档请保持 0(整段 draft)"
        )
    if st.executor == "engine" and t.rollout_backend != "sglang":
        raise ValueError(
            f"steering.executor=engine 仅支持 train.rollout_backend=sglang"
            f"(P1 实现面;得到 {t.rollout_backend!r})——vLLM 路径请用 python 档"
        )
    if st.enabled:
        # 审阅 IF-13:数值面先过有限性——NaN/inf 能绕过大小比较,错误会
        # 延迟到 rollout 中段才爆。b_max 例外:+inf 是 maxopen 档的一等语义
        # (无限预算,v3.2),仍拒 NaN/−inf。
        for name in ("eta_max", "c", "theta_d", "epsilon",
                     "steered_adv_weight", "steered_adv_cap",
                     "w_beh_shaped_gamma", "entropy_theta_init",
                     "entropy_floor", "intensity_w_min"):
            if not math.isfinite(getattr(st, name)):
                raise ValueError(f"steering.{name} 必须是有限数,得到 {getattr(st, name)!r}")
        if math.isnan(st.b_max) or st.b_max == -math.inf:
            raise ValueError(f"steering.b_max 必须是非负数或 +inf,得到 {st.b_max!r}")
        if math.isinf(st.b_max) and st.budget_pacing:
            raise ValueError(
                "steering.b_max=inf 与 budget_pacing=true 不相容(解锁曲线无定义)"
                "——maxopen 档请显式 budget_pacing: false"
            )
        if st.gate_mode not in ("adaptive", "always_on", "entropy", "d_var", "soft_or"):
            raise ValueError(
                "steering.gate_mode 必须是 adaptive|always_on|entropy|d_var|soft_or,"
                f"得到 {st.gate_mode!r}"
            )
        if st.gate_mode in ("entropy", "soft_or"):
            if st.entropy_theta_update not in ("direct", "ema"):
                raise ValueError(
                    "steering.entropy_theta_update 必须是 direct|ema,"
                    f"得到 {st.entropy_theta_update!r}"
                )
            if st.entropy_theta_update == "ema":
                # 历史行为完整复现档:标定纪律(theta_d 同款)——初值与
                # 硬地板必须显式用目标语境标定回填,拒绝未标定占位。
                if st.entropy_theta_init <= 0:
                    raise ValueError(
                        "steering.entropy_theta_init 未标定(<=0)——ema 档"
                        "先用目标语境 dump 的 pi_topk 熵分布定 P_{1−rate} 初值,"
                        "或改用 entropy_theta_update: direct(分位自标定)"
                    )
                if st.entropy_floor <= 0:
                    raise ValueError(
                        "steering.entropy_floor 必须 > 0(ema 档防阈值追熵坍缩"
                        "的硬地板,MENTOR max(q80,1.5) 先例)"
                    )
            else:
                # direct 档(2026-08-11 剂量通用化):绝对 nats 常数一律拒绝
                # ——floor 由 ratchet(θ 不低于首次标定值)取代;theta_init
                # 0=自标定 warmup(零干预起步),>0=显式起点。
                if st.entropy_floor > 0:
                    raise ValueError(
                        "steering.entropy_floor 仅 ema 档有语义——direct 档的"
                        "坍缩防线是 entropy_theta_ratchet(自标定,无量纲),"
                        "请删除 entropy_floor 或改 entropy_theta_update: ema"
                    )
                if st.entropy_theta_init <= 0 and not st.gate_adaptive:
                    raise ValueError(
                        "steering.entropy_theta_init=0(自标定)依赖 tracker "
                        "(gate_adaptive: true)在 step 边界落定阈值——固定阈值"
                        "运行请显式给 entropy_theta_init"
                    )
            if not (0.0 < st.entropy_target_rate < 1.0):
                raise ValueError(
                    f"steering.entropy_target_rate 必须在 (0,1),得到 {st.entropy_target_rate}"
                )
        if st.gate_mode in ("always_on", "entropy", "d_var", "soft_or"):
            # always_on 也消费 w_min(decide 的软档分支 base_open 判定与 hook
            # 段级 bypass;审阅 L15:w_min>1 会把 maxopen 静默变成零干预)。
            if not (0.0 <= st.intensity_w_min < 1.0):
                raise ValueError(
                    f"steering.intensity_w_min 必须在 [0,1),得到 {st.intensity_w_min}"
                )
        if st.gate_mode in ("entropy", "soft_or") and st.obs_topk < 16:
            # 熵信号 = draft top-obs_topk 明细的前 16 个(steering/hook.py 的
            # ENTROPY_TOPK;审阅 L20)——窗口窄于 16 会静默换熵口径,标定失配。
            raise ValueError(
                f"steering.obs_topk({st.obs_topk}) 必须 >= 16(熵信号截断宽度"
                " ENTROPY_TOPK;与标定同口径)"
            )
        if not (0.0 <= st.epsilon < 0.5):
            # 配置期即拒(审阅 L18:原先只在首条 steered 轨迹构造 IntensityGate
            # 时才炸,静态可查的错误被推迟成半程崩溃)。
            raise ValueError(f"steering.epsilon 必须在 [0, 0.5),得到 {st.epsilon}")
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
        if st.gate_mode in ("adaptive", "d_var", "soft_or") and st.theta_d <= 0:
            raise ValueError(
                "steering.theta_d 未标定(<=0)——先用目标文档语境预扫描 D 分布定 P85"
                "(gate_mode=always_on/entropy 免标定)"
            )
        if st.w_beh_mode not in ("per_token", "per_decision", "vtrace",
                                 "off", "shaped_reject"):
            raise ValueError(f"未知 w_beh_mode {st.w_beh_mode!r}")
        if st.w_beh_mode == "shaped_reject" and st.w_beh_shaped_gamma <= 0:
            raise ValueError(
                f"steering.w_beh_shaped_gamma 必须 > 0,得到 {st.w_beh_shaped_gamma}"
            )
        # ---- v3.3 M1/M2 校验(asym advantage 与 α 调度)----
        if st.adv_mode not in ("legacy_shared", "asym"):
            raise ValueError(
                f"steering.adv_mode 必须是 legacy_shared|asym,得到 {st.adv_mode!r}"
            )
        if st.adv_mode == "asym":
            if t.adv_estimator != "pgs_asym_grpo":
                raise ValueError(
                    "steering.adv_mode=asym 要求 train.adv_estimator=pgs_asym_grpo"
                    f"(得到 {t.adv_estimator!r})——两处必须联动,防半配置"
                )
            if st.steered_fraction >= 1.0:
                raise ValueError(
                    "steering.adv_mode=asym 需要分臂(steered_fraction<1)——"
                    "steered 超额的基线是同组 bare 均值,全臂无 bare 可比"
                )
            if st.steered_adv_weight <= 0:
                raise ValueError(
                    f"steering.steered_adv_weight 必须 > 0,得到 {st.steered_adv_weight}"
                )
            if t.norm_adv_by_std:
                raise ValueError(
                    "adv_mode=asym 下 estimator 不做 std 归一——"
                    "train.norm_adv_by_std=true 会被静默忽略且污染指纹,"
                    "请显式 false(参数真值纪律,审阅 L1)"
                )
        elif t.adv_estimator == "pgs_asym_grpo":
            raise ValueError(
                "train.adv_estimator=pgs_asym_grpo 要求 steering.adv_mode=asym"
                "(estimator 需要臂标签与 α,legacy 口径不产)"
            )
        else:
            # legacy 档的 α 三参完全惰性——非默认值必是配置错觉,拒绝静默
            # 忽略(always_on 档参数纪律先例;审阅 L19)。
            if (st.steered_adv_weight != 1.0 or st.steered_adv_decay_steps != 0
                    or st.steered_adv_cap != 0.0):
                raise ValueError(
                    "steered_adv_{weight,decay_steps,cap} 仅在 adv_mode=asym 下"
                    "生效——legacy_shared 配非默认值会被静默忽略,请删除或切 asym"
                )
        if st.steered_adv_decay_mode != "cosine_steps":
            raise ValueError(
                f"未知 steered_adv_decay_mode {st.steered_adv_decay_mode!r}"
                "(当前仅 cosine_steps)"
            )
        if st.steered_adv_decay_steps < 0 or st.steered_adv_cap < 0:
            raise ValueError("steered_adv_decay_steps/steered_adv_cap 必须 >= 0")
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
            selfres = cfg.research_rl.enabled
            if selfres:
                # v4 自体研究员(engine + TextProtocol):external 角色面
                # (endpoint/key/thinking)整体不适用,通用数值检查照旧。
                if rcfg.backend != "engine":
                    problems.append("research_rl 下 research.backend 必须为 engine(自体)")
                if rcfg.tool_protocol != "text":
                    problems.append("research_rl 下 research.tool_protocol 必须为 text")
            else:
                if rcfg.backend != "external":
                    problems.append("research.backend 必须为 external")
                if rcfg.tool_protocol != "fncall":
                    problems.append("research.tool_protocol 必须为 fncall")
            if rcfg.gt != "off":
                problems.append("research.gt 必须为 off")
            # §7.5:预算三键必配(上限/控制器解耦,无默认推导——用户裁定)
            if rcfg.session_token_budget <= 0:
                problems.append(
                    "research.session_token_budget 必配(>0;会话流预算,不含 "
                    "prompt。定值依据:4B researcher 探针会话总量 p90=42122/"
                    "max=58828)"
                )
            if rcfg.max_prompt_tokens <= 0:
                problems.append(
                    "research.max_prompt_tokens 必配(>0;researcher 首调 "
                    "prompt 上限,4B 实测中位 1426/max 1598)"
                )
            if cfg.narration.max_prompt_tokens <= 0:
                problems.append(
                    "narration.max_prompt_tokens 必配(>0;叙述 prompt 上限,"
                    "超出截轨迹渲染段)"
                )
            # 保险丝不等式(FAIL 级)——**仅 selfres(自体研究员)**:
            # ① prompt+流 ≤ 模型上下文(否则引擎必爆);
            # ② research 行 stream 进 verl response padding 维度(只补不截),
            #    budget 超维度 = 超长行不截静默进训练;
            # ③ research 行 prompt 进 verl prompt padding 维度,同理。
            # external 外部研究员(2026-08-13 用户裁定):会话 token 流是外部
            # 模型(35B/256K 级上下文)生成、不进训练 batch 的任何 padding
            # 维度,预算只受外部服务 max_model_len 约束(preflight/运行时由
            # 端点自拒),不得绑到 policy 的训练维度上压低会话预算。
            if selfres:
                if (rcfg.session_token_budget > 0 and rcfg.max_prompt_tokens > 0
                        and rcfg.session_token_budget + rcfg.max_prompt_tokens
                        > t.rollout_max_model_len):
                    problems.append(
                        f"research.session_token_budget({rcfg.session_token_budget})"
                        f" + max_prompt_tokens({rcfg.max_prompt_tokens}) 超过 "
                        f"train.rollout_max_model_len({t.rollout_max_model_len})"
                    )
                if rcfg.session_token_budget > t.response_length:
                    problems.append(
                        f"research.session_token_budget({rcfg.session_token_budget})"
                        f" 超过 train.response_length({t.response_length})——research "
                        "行 stream 进该 padding 维度且 verl 只补不截"
                    )
                if rcfg.max_prompt_tokens > t.max_prompt_length:
                    problems.append(
                        f"research.max_prompt_tokens({rcfg.max_prompt_tokens}) 超过 "
                        f"train.max_prompt_length({t.max_prompt_length})(research 行 "
                        "prompt 的 padding 维度)"
                    )
            # narration 后端(2026-08-13 用户裁定放开 external):
            #   engine   — policy 自述自己的轨迹(历史默认;researcher 读到的是
            #              "学生眼中的轨迹")
            #   external — 外部强模型叙述,与 researcher 同端点。动因是吞吐:
            #              narration 每个 researcher episode 调一次,实测
            #              1415 次/步 ×(1141 prefill + 308 decode)⇒ 每步给
            #              7B 引擎压上 1.62M prefill + 0.44M decode,是单项最大
            #              的 prefill 来源。不引入新特权(researcher 本就有权
            #              看该轨迹全文),但叙述视角由学生变为强模型。
            if cfg.narration.backend not in ("engine", "external"):
                problems.append(
                    f"narration.backend 必须是 engine|external,得到 "
                    f"{cfg.narration.backend!r}"
                )
            if cfg.narration.backend == "external" and not (
                "narrator" in cfg.roles or "researcher" in cfg.roles
            ):
                problems.append(
                    "narration.backend=external 需要 roles.narrator 或 roles.researcher"
                )
            # 同任务副本必须落同一 worker(steering 组内配对 + 每 task 只跑一次
            # researcher)。**准确条件是整除而非相等**(2026-08-13 查清 verl 语义):
            # verl 对已 repeat n 次的 batch(batch_size×n 行)按 worker 数**连续
            # 切块**(agent_loop.py: prompts.chunk(len(workers))),而
            # DataProto.repeat 默认 interleave=True ⇒ 行序为 [t0×n, t1×n, …]。
            # 于是每 worker 拿 batch_size×n/workers 行,只要它是 n 的整数倍,
            # 每个 worker 就恰好拿到整数个**完整 task 组** ⇒ 条件即
            # batch_size % workers == 0(相等只是其特例)。
            # 放宽的动因:每个 worker 是独立进程、各持一个检索 batcher;
            # workers=batch_size=128 时每进程只见 1/128 的稀疏 query 流,
            # 实测 avg_batch=1(合批完全失效)且 128×在途 = 256 并发小请求,
            # 检索退化到 15 q/s、wait 恶化到 47s(冒烟 v5)。
            if t.agent_num_workers < 1 or t.train_batch_size % t.agent_num_workers:
                problems.append(
                    f"train.train_batch_size({t.train_batch_size}) 必须能被 "
                    f"agent_num_workers({t.agent_num_workers}) 整除,"
                    "保证同任务副本同 worker(verl 按 worker 数对 interleave 后的 "
                    "batch 连续切块)"
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
            if selfres:
                pass                     # 自体模式无 external 角色面
            elif role is None:
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

    rrl = cfg.research_rl
    if rrl.enabled:
        # v4 自体研究员依赖 researcher 文档基建与 steering 教师面
        # (docs/pgs-self-researcher-design.md §5)。
        if not (st.enabled and st.doc_source == "researcher" and cfg.research.enabled):
            raise ValueError(
                "research_rl.enabled 要求 steering.enabled + doc_source=researcher "
                "+ research.enabled(自体研究员复用 researcher 会话基建)"
            )
        if rrl.research_n < 2:
            raise ValueError(
                f"research_rl.research_n 必须 >= 2(组间对比),得到 {rrl.research_n}"
            )
        qs = rrl.prefix_quantiles
        if not qs or any(not (0.0 < q < 1.0) for q in qs) or sorted(qs) != qs:
            raise ValueError(
                f"research_rl.prefix_quantiles 必须是 (0,1) 内升序非空列表,得到 {qs!r}"
            )
        if rrl.continuation_samples < 1:
            raise ValueError(
                f"research_rl.continuation_samples 必须 >= 1,得到 {rrl.continuation_samples}"
            )
        for name in ("continuation_temperature", "researcher_temperature"):
            v = getattr(rrl, name)
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"research_rl.{name} 必须是有限正数,得到 {v!r}")
        if not (math.isfinite(rrl.research_rl_weight) and rrl.research_rl_weight >= 0):
            raise ValueError(
                f"research_rl.research_rl_weight 必须是非负有限数,得到 {rrl.research_rl_weight!r}"
            )
        if rrl.research_warmup_steps < 0 or rrl.research_max_sample_tokens < 0:
            raise ValueError(
                "research_rl.research_warmup_steps 须 >=0 且 research_max_sample_tokens 须 >=0"
                "(0 = 联动会话 token 墙,§7.4)"
            )


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
    # 任务环境本身当然定义优化目标;条件追加:默认 alfworld(历史唯一环境)
    # 不改 material,既有 run 指纹逐字节不变。
    if cfg.environment.name != "alfworld":
        material += f"|environment={cfg.environment.name}"
    # FC 协议统一(2026-08-08 用户裁定)改变任务表面(system prompt/输出
    # 协议/观察包裹)= 换优化目标;条件追加:text(历史行为)不改 material,
    # 在跑 text run 的指纹逐字节不变,误用新代码续跑旧 run 会被 resume 守卫拦下。
    if cfg.environment.name == "alfworld" and cfg.alfworld.protocol != "text":
        material += f"|alfworld_protocol={cfg.alfworld.protocol}"
    # SearchQA 同理(2026-08-13):文本标签协议 vs FC 是两套任务表面
    # (system prompt / 输出格式 / 观察包裹 <information> vs <tool_response>),
    # 换协议 = 换优化目标。**无条件追加**——SearchQA 是新域,无需保护历史
    # 指纹,无条件写入才能保证两种协议的 run 指纹必然不同(alfworld 那条是
    # 条件式,只为兼容既有 text run)。
    if cfg.environment.name == "searchqa":
        material += f"|searchqa_protocol={cfg.searchqa.protocol}"
    # Math 同理(2026-08-13 文本 TIR 裁定):围栏协议 vs FC 是两套任务表面
    # (system prompt / ```python vs tool_call / ```output vs <tool_response>
    # / \boxed 终答 vs submit 参数)。本树无 math 历史训练 run,无条件追加
    # (searchqa 同款)。
    # WebShop 同理(2026-08-16 用户裁定 text 档):文本动作 vs FC 是两套任务
    # 表面(system prompt / <action>search[q]</action> vs tool_call / 裸观察
    # vs <tool_response> 包裹)。本树无 webshop 历史训练 run,无条件追加。
    if cfg.environment.name == "webshop":
        material += f"|webshop_protocol={cfg.webshop.protocol}"
        # option 修复直接改变 reward 的定义(6.63% 的任务从"恒 won=False"
        # 变成"可达满分")⇒ **无条件**追加,两种口径的 run 指纹必然不同。
        material += f"|webshop_option_fix={int(cfg.webshop.fix_option_matching)}"
    if cfg.environment.name == "math":
        material += f"|math_protocol={cfg.math.protocol}"
        # single 档的 reward 由 (1-w)·acc + w·format 定义 ⇒ 权重是优化目标的
        # 一部分。条件追加(仅 single),fncall/text 的既有指纹逐字节不变。
        if cfg.math.protocol == "single":
            material += f"|math_format_weight={cfg.math.format_weight}"
    # SSB 同理(2026-08-17 用户裁定 text 档):围栏协议 vs FC 是两套任务表面
    # (system prompt one-shot / ```python vs tool_call / ```output vs
    # <tool_response> / ```submit vs submit())。**条件追加**(alfworld 同款
    # 方向反转:SSB 历史 run 全是 fncall,fncall 不改 material 保既有指纹
    # 逐字节不变;text 追加,误用新协议续跑旧 run 被 resume 守卫拦下)。
    if (cfg.environment.name == "spreadsheetbench"
            and cfg.spreadsheetbench.protocol != "fncall"):
        material += f"|ssb_protocol={cfg.spreadsheetbench.protocol}"
    # reward 取数口径(二值 won vs 连续 score)直接定义"优化什么";
    # 条件追加以保 False(全部既有 run)的指纹逐字节不变。
    if cfg.train.reward_from_score:
        material += "|reward_from_score=1"
    # 推理标签是任务表面的一部分(system prompt / one-shot / 解析)⇒ 换标签
    # = 换优化目标。**按 env 的历史实际值**条件追加:每个 env 保持其历史
    # 默认时不改 material(既有 run 指纹逐字节不变),换掉才追加。
    #   alfworld 历史 = reasoning(它一直用 <reasoning>)
    #   searchqa / math 历史 = think(SearchQA 沿用 Search-R1 原生标签;
    #     math single 档初版逐字移植 MENTOR)
    # ⇒ 2026-08-16 统一到 reasoning 后,alfworld 指纹不变、searchqa/math
    #   指纹改变 —— 后者正是我们要的(换协议 = 新曲线,旧 ckpt 被守卫拦下)。
    _legacy_rtag = {"alfworld": "reasoning", "searchqa": "think", "math": "think"}
    if cfg.episode.reasoning_tag != _legacy_rtag.get(cfg.environment.name, "reasoning"):
        material += f"|reasoning_tag={cfg.episode.reasoning_tag}"
    # thinking 模式改变每轮生成的形状(空 think 块 vs 模型自写思考),
    # 是协议面而非执行档:两种模式的 run 不可混为同一条曲线。条件追加
    # 以保 non-thinking 历史 run 的指纹逐字节不变。
    if cfg.episode.enable_thinking:
        material += "|thinking=on"
    # 分臂采样(v3.1)改变优化目标(组内一半样本换回 π 采样),必须进指纹;
    # 条件追加:fraction=1.0(历史全臂)不改 material,既有 run 指纹逐字节不变
    # (researcher source 哨兵同款先例)。
    if st.enabled and st.steered_fraction < 1.0:
        material += f"|steered_fraction={st.steered_fraction}"
    # maxopen 门控模式(v3.2)改变行为分布 b(处处倾斜),必须进指纹;
    # 条件追加:adaptive(历史行为)不改 material。
    if st.enabled and st.gate_mode != "adaptive":
        material += f"|gate_mode={st.gate_mode}"
    # v3.3 M4 软强度门参数改变行为分布 b;条件追加(entropy/soft_or 档才有语义)。
    if st.enabled and st.gate_mode in ("entropy", "soft_or"):
        material += (
            f"|entropy_gate={st.entropy_target_rate}:{st.entropy_theta_init}"
            f":{st.entropy_floor}"
        )
        # 剂量通用化(2026-08-11):direct 分位直标改变 θ 轨迹 = 改变行为
        # 分布 b,必须进指纹;条件追加——ema(历史行为)不改 material,
        # 显式配 ema 的旧 run 指纹逐字节不变,可继续 resume。
        if st.entropy_theta_update != "ema":
            material += (
                f"|entropy_theta_update={st.entropy_theta_update}"
                f":{int(st.entropy_theta_ratchet)}"
            )
    if st.enabled and st.gate_mode in ("entropy", "d_var", "soft_or"):
        material += f"|intensity_w_min={st.intensity_w_min}"
    # v3.3 M1/M2 不对称 advantage 改变优化目标;条件追加(legacy_shared 不改
    # material,既有 run 指纹逐字节不变)。α 三参数决定 steered 梯度权重与
    # 停产点,同批进。
    if st.enabled and st.adv_mode != "legacy_shared":
        material += (
            f"|adv_mode={st.adv_mode}"
            f"|steered_adv={st.steered_adv_weight}"
            f":{st.steered_adv_decay_steps}:{st.steered_adv_cap}"
            f":{st.steered_adv_decay_mode}"
        )
    # v3.3 M3 shaped_reject 的 γ 决定拒绝位梯度权重;条件追加。
    if st.enabled and st.w_beh_mode == "shaped_reject":
        material += f"|w_beh_gamma={st.w_beh_shaped_gamma}"
    # v4 自体研究员(research_rl)改变优化目标(双 RL 目标 + 教师文档来源
    # 从共享单会话换为组内 argmax),全套进指纹;条件追加保历史指纹。
    rrl = cfg.research_rl
    if rrl.enabled:
        material += "|" + "|".join(
            str(x) for x in (
                "research_rl", rrl.research_n, rrl.prefix_quantiles,
                rrl.continuation_samples, rrl.continuation_temperature,
                rrl.researcher_temperature, rrl.research_rl_weight,
                rrl.research_warmup_steps, rrl.research_max_sample_tokens,
            )
        )
    if st.doc_source == "researcher":
        rcfg, ncfg = cfg.research, cfg.narration
        # v4 自体研究员(research_rl.enabled)没有 external 角色面 —— researcher
        # 就是被训练的 policy 本身,连接参数(endpoint/model/temperature/
        # sidecar)整体不存在。此处强取 roles["researcher"] 会 KeyError 直接
        # 拦死 selfres 的启动(2026-08-12 实测)。自体档改记一个稳定标记:
        # 它同样能把 external↔selfres 的切换体现成指纹变化(角色面消失本身
        # 就是优化目标的改变),而 selfres 自己的超参已在上面的 research_rl
        # 段进入 material。
        role = cfg.roles.get("researcher")
        role_material = ({
            "model": role.model,
            "temperature": role.temperature,
            "max_tokens": role.max_tokens,
            "extra_body": role.extra_body,
            "mode": role.mode,
            "sidecar_sleep": role.sidecar_sleep,
            "dp_size": role.dp_size,
        } if role is not None else {"mode": "selfres_engine"})
        research_material = {
            "source": "researcher",
            # M2(2026-08-08)重写了 text 通道的 researcher 表面:schema 块
            # 移出 system prompt、tools 交模板渲染、wire 解析家族化——
            # tool_protocol 字符串没变而分布变了。哨兵使新代码指纹必与旧
            # 不同:v4 在跑 run(researcher token 在 loss 里)用新代码
            # resume 会被守卫拦下,而不是静默换优化目标(审阅 2026-08-08)。
            "researcher_wire": "fc_native_v2",
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
            "role": role_material,
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
        research_rl=_build(ResearchRLCfg, data.get("research_rl") or {}),
        narration=_build(NarrationCfg, data.get("narration") or {}),
        roles=roles,
        episode=_build(EpisodeCfg, data.get("episode") or {}),
        environment=_build(EnvironmentCfg, data.get("environment") or {}),
        alfworld=_build(AlfworldCfg, data.get("alfworld") or {}),
        spreadsheetbench=_build(
            SpreadsheetbenchCfg, data.get("spreadsheetbench") or {}
        ),
        math=_build(MathCfg, data.get("math") or {}),
        searchqa=_build(SearchQACfg, data.get("searchqa") or {}),
        webshop=_build(WebshopCfg, data.get("webshop") or {}),
        scienceworld=_build(SciworldCfg, data.get("scienceworld") or {}),
        terminal=_build(TerminalCfg, data.get("terminal") or {}),
        webarena=_build(WebArenaCfg, data.get("webarena") or {}),
        swe=_build(SweCfg, data.get("swe") or {}),
        train=_build(TrainCfg, data.get("train") or {}),
        eval=_build(EvalCfg, data.get("eval") or {}),
        steering=_build(SteeringCfg, steering_raw),
        raw=data,
    )
    _validate(cfg)
    return cfg
