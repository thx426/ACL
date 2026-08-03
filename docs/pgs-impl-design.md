# PGS 实现设计（pgs-sps 分支）

CD-006 · 2026-08-04 · 机制规范：`docs/pgs-final-spec.md`（v3.0-final）
**用户裁定的实施原则**：新方案代码完全独立、干净；旧机制代码只保留复用部分，
无关代码清理干净。

---

## 1. 代码处置清单（保留 / 清理 / 新增）

### 1.1 保留（基建，新方案直接复用）

| 路径 | 角色 | 处置 |
|---|---|---|
| `core/episode.py` | 逐轮 episode runner（rollout 骨架） | 保留；加一个干净的可选 steering 钩子 |
| `core/protocol.py` | 交互协议/轮边界/预算常量 | 保留 |
| `core/types.py` | EpisodeResult/Tokens 等 | 保留（扩展记账字段） |
| `envs/alfworld/*` | 环境适配（adapter/worker/gt） | 保留 |
| `train/backends.py` | EngineGenerate（vLLM 客户端） | 保留；扩展 logprobs 透传 |
| `train/build_datasets.py` | 数据构建（含 gtwalk 通道） | 保留 |
| `common/config.py` | 配置系统骨架 | 保留骨架；清无关配置组、加 `steering` 组 |
| `verl/`（vendored） | 训练循环/GRPO/PPO/FSDP | 保留；训练端最小改动（W_beh 权重） |
| agent_loop 中 `_teacher_prompt_ids` 的前缀构造 | 教师前缀（文档在 system prompt 最前部——KV 复用前提已满足，NF09 核实完毕） | 逻辑迁入新模块 |

### 1.2 清理（与新机制无关的旧机制代码）

| 路径/内容 | 属于 | 理由 |
|---|---|---|
| `verl/verl/trainer/distillation/` 的 OPD/tg 损失（reverse_kl_topk、tg_soft_ce、teacher_topk 观测落盘） | 旧 OPD 机制 | 新损失中无教师、无蒸馏项 |
| agent_loop 中 research 闭环（`_research*`、`research_group.py`、researcher 端点/隧道逻辑） | H1 在线研究闭环 | 第一版文档来源=静态表（离线制备），无在线 researcher |
| narration、sft_gen 通道 | 旧机制辅助 | 无关 |
| `microscope/`、`viz/`、旧 obs 管线 | 旧 OPD 观测 | 新方案有自己的记账；分析工具随需另建 |
| 配置组：`research`、`research_rl`、`narration`、`sft_gen`、distillation 系（`opd_*`、`mix_tg_*` 等） | 旧机制 | 清理；保留 `mix_rl`→升级为主通道 |

清理方式：整文件删除 + agent_loop/config 内的分支剪除；每步跑 import/单测防断链。

### 1.3 新增（新机制的独立代码）

| 路径 | 内容 |
|---|---|
| `src/trace_opd/steering/__init__.py` | 包入口 |
| `src/trace_opd/steering/kernel.py` | **保尾核**：S 集合固定、ℓ̃ 截断、b_S 构造、密度比界断言；纯函数 |
| `src/trace_opd/steering/gate.py` | D 计算（保尾核对象上）、Θ_D 冻结阈值 + robust-EMA、ε 探针（独立随机数、propensity 记录）、可测性红线的实现约束 |
| `src/trace_opd/steering/budget.py` | D_∞ 预算余额制（δ=2ηc 保守记账）、R_max 状态机（cap-active 行为态） |
| `src/trace_opd/steering/sps.py` | SPS 接受-拒绝-残差重采（相对 b_S 的 speculative 恒等）、记账装配 |
| `src/trace_opd/steering/ledger.py` | 逐 token 记账结构（log_pi/log_b/eta/delta/gate_state/steer_flag/reject_count）→ extra_fields |
| `src/trace_opd/steering/episode_hook.py` | 与 episode.py 的集成钩子（draft→verify→resample→续 draft 循环） |
| `tests/steering/test_kernel.py` 等 | §5 测试计划 |
| `configs/pgs.yaml` | 新配置（steering 组 + mix_rl 底座） |

## 2. 引擎能力确认（已探明）

```
draft 逐 token log π    TokenOutput.log_probs（replica.py:43）；vLLM server 侧
                        sampling_params["logprobs"] 透传（vllm_async_server.py:595,671）
verify/scoring          extract_prompt_logprobs 已存在（:663，num_prompt_logprobs
                        取自 SamplingParams.prompt_logprobs）——scoring 调用 =
                        prompt_ids 带全序列 + max_tokens=1 + prompt_logprobs=K
参数通道                sampling_params dict 直通 SamplingParams(**)（:607）
                        ⇒ logprobs/prompt_logprobs 不需改 verl 引擎，只需在
                        EngineGenerate/TokenOutput→GenResult 链路把它们带出来
教师前缀 KV             文档在 system prompt 静态最前部（原 _teacher_prompt_ids
                        构造），prefix caching 跨轮可复用——NF09 前提已满足
```

**风险位**：`extract_prompt_logprobs` 的输出去向（extra_fields 的键名/格式）需
实测确认；top-k 上限 `max_logprobs` 需在 vLLM 启动参数放宽到 ≥64。

## 3. SPS 数据流（映射到调用）

```
每轮（episode_hook 内）：
① draft   EngineGenerate(prompt+stream, {**sp, logprobs: 1})
          → gen_ids + log π(y)（draft 路径，仅监控用）
② verify  两次 scoring 调用（同一 verify 通道，数值统一）：
          bare:    generate(prompt+stream+gen_ids, {max_tokens:1, prompt_logprobs:K})
          teacher: generate(teacher_prompt+stream+gen_ids, {max_tokens:1, prompt_logprobs:K})
          → 逐位置两侧 top-K logprobs（S_j 的原料）
③ 扫描    kernel.build(S_j, ℓ̃, η_j) → b_S；gate/budget 决定 η_j（σ(h_j,…) 红线）
          接受检验 min(1, b(y_j)/π(y_j))（π 取 verify 路径值）
④ 拒绝于 j*：sps.resample_residual(S_{j*}) → y′；截断 stream 至 j*⊕y′；
          回 ① 续 draft（R_max 状态机管制）
⑤ 定稿    ledger 写 extra_fields；环境执行
```

verify 的两次 scoring 与 draft 的比例：每轮 1 次 draft + R 次续段，每段 2 次
scoring；教师侧 prompt 静态前部相同 ⇒ prefix cache 增量。

## 4. 训练端改动（最小面）

- rollout 记账把 `log_b` 写入 batch（沿 extra_fields→TQ 通道，§7 修复后链路可用）
- verl 侧 GRPO/PPO loss：advantage 乘 `W_beh`（消融开关：单点 w_t / 前缀 W_t /
  V-trace 截断）；`r_upd = π_θ/π_old` 照旧走 clip——**分离式**（NF02）
- GRPO 组内 baseline+scale 改 leave-one-out（NF03 条件 iv）
- 定位：`mix_rl` 通道升级为主损失（旧 OPD 项清理后即纯 RL + W_beh）

## 5. 测试计划

| 层 | 内容 |
|---|---|
| kernel 单测 | b_S 归一性；S 外恒等 π；密度比 ∈ e^{±2ηc}（含构造到界的极端例）；η=0 ⇒ b≡π |
| sps 统计测试 | 小词表（V=50）模拟：SPS 输出经验分布 vs 解析 b_S 的 TV < 阈值（万次采样）；拒绝率 = TV(b,π) 校验 |
| budget 单测 | 余额制扣费/耗尽/缩 η；R_max 状态机；记账 δ=2ηc |
| gate 单测 | 可测性（η 不依赖 y_j——接口层强制）；探针 propensity 记录 |
| ledger 单测 | 字段完整性；与 turn_spans 对齐 |
| 集成冒烟 | mock generate 的 episode 全流程；真实引擎冒烟（162，C3 结束后） |

## 6. 分阶段

```
P1  清理 commit（1.2 清单）+ steering 包核心（kernel/gate/budget/sps/ledger）+ 全部单测   ← 本轮
P2  backends logprobs 透传 + episode_hook 集成 + agent_loop 精简改造
P3  训练端 W_beh + LOO + 配置组 + pgs.yaml
P4  真实引擎冒烟（162）→ 记账/数值一致性核验（draft vs verify 的 π 残差实测）
```

## 7. 风险清单

1. prompt_logprobs 输出格式/键名（P2 首个验证项）；max_logprobs 启动参数
2. scoring 调用的 KV/吞吐实测（NF09 算力账）
3. draft 与 verify 的 π 数值残差（~1e-3 预期，smoke 必测）
4. 清理面误伤（每步 import+测试把关）
