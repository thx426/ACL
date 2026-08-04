# PGS 完整实施深度代码审阅（阶段性落盘）

> 审阅区间：`c9d2466..c2096e3`，共 8 个 commit（`c623d85` → `c2096e3`）。  
> 审阅对象固定为提交 `c2096e3` 的 Git 对象；审阅途中工作树出现了非本审阅产生的并发未提交修改，均不纳入判断，也未被本审阅触碰。  
> 本文件按用户要求分阶段落盘；下列计数与核对表会在审阅完成时更新为最终值。

## 总体结论（阶段性）

实现的数学纯模块整体质量较好，保尾核、残差采样和保守预算的主干已经形成；但当前 8-commit 快照存在一个会阻断正式 steering 轨迹进入 TransferQueue 的 P0 接线错误，并在 LOO scale、审计记账、配置单源与规范保护措施上存在数个实质缺口，现阶段不应直接进入 P4 正式实验。

当前已确认：**P0 1 条 / P1 5 条 / P2 2 条**（阶段性计数）。

## Findings

### IF-01 — P0 — `verl/verl/trainer/ppo/v1/agent_loop_tq.py:190-195`

**问题陈述**

PGS 与 verl 原生 rollout correction 的互斥判断写成了
`self.config.algorithm.get("rollout_correction", None) is not None`。vendored verl 的 Hydra 默认配置始终实例化 `algorithm.rollout_correction` 映射，即使其中 `rollout_is`、`rollout_rs` 都是 `null`；因此“功能关闭”仍会被判为冲突。

**失败场景**

使用仓库提供的 `162-pgs-s1.yaml` 启动正式 steering；只要第一个位置开门（或探针翻开）使 `steer_positions` 非空，TQ 入队前即抛出 `RuntimeError("steering W_beh and algorithm.rollout_correction ...")`。标定臂因门恒关、账本为空可能通过，导致问题恰好延迟到正式实验才暴露；该 prompt 随后被标为 failure，PGS 训练数据无法正常入队。

**修复建议**

互斥条件应判断原生功能是否真的会写权重，例如 `rollout_is is not None` 或 `rollout_rs is not None`（并覆盖 bypass 模式的实际写入条件），而不是判断配置对象是否存在；同时在 launcher 明确钉死 PGS 档的原生 rollout correction 关闭状态，并增加“真实 Hydra 默认配置 + 一条非空 steering ledger + TQ postprocess”的本地集成测试。

### IF-02 — P1 — `verl/verl/trainer/ppo/core_algos.py:639-699`

**问题陈述**

`grpo_loo` 的 baseline 严格 LOO，但 scale 使用 `max(std_loo, std_full)`；`std_full` 包含当前样本自己的 reward。这与权威 spec §2.6“baseline 与 scale 均 leave-one-out”及命题 C 条件 iv 不一致。“无偏性只由 baseline 承担”的实现注释在数学上不成立：一个依赖当前动作/reward 的随机分母同样会改变期望梯度。

**失败场景**

二元 7 胜 1 负组中，孤立负样本的 `std_loo=0`，代码改用包含该负样本的 `std_full`；多数正样本通常使用各自的 `std_loo`。正负样本因此按依赖自身 outcome 的不同尺度进入梯度，即使 W_beh 完全正确，也不再满足命题 C 所列的交叉拟合条件。代码确实避免了 `1/epsilon` 爆炸，但代价是引入未量化偏差，不能继续称为严格 LOO scale。

**修复建议**

在“理论正确档”中改用仅依赖其他样本的有界 scale（例如预注册常数、跨组/历史冻结尺度，或对 `std_loo=0` 采用与当前样本无关的固定 floor）；若保留 `max(std_loo,std_full)` 作为工程稳定档，应明确标为有偏实现、从命题 C 条件 iv 中移除，并单独做 scale 口径消融和偏差说明。

### IF-03 — P1 — `src/trace_opd/steering/hook.py:201-230`；`src/trace_opd/steering/ledger.py:62-86`

**问题陈述**

spec §4 要求逐 token 记录完整 `gate_state`、预算余额、`steer_flag` 与 `reject_count`；实际 ledger 只在执行了正 η，或“`base_open or probe_flipped` 且 η=0”时追加稀疏行。最常见的“主门关闭且探针未翻转”位置完全消失；预算耗尽或 cap-active 后直通的 token 也不记录。现有字段还缺逐位置预算余额、明确的 steer flag 和累计 reject count，而 `steer_count=len(records)` 会把零 η 的关闭/阻断行也计作 steer。

**失败场景**

主门命中率 15%、epsilon=5% 时，大部分候选位置没有逐 token gate 记录。事后无法从 ledger 重建 probe 分配的控制总体、预算 noncompliance 或 cap-active 后的行为状态；`steer_count` 曲线也可能把“门开但被翻关/预算阻断”误报为实际干预。训练权重在这些未干预位置仍为 1，因而不一定立即数值错误，但 NF07 的可识别性修法和 spec §4 的审计红线没有落地。

**修复建议**

分离两种数据面：行为校正可继续使用只含非零 ratio 的稀疏表；审计 ledger 必须逐生成 token 写全 `D/base_open/assigned_open/probe/propensity/eta/delta/budget_remaining/cap_active/steer_flag/reject_count`。为计数另设 `actual_steer_count`，不要复用记录行数。

### IF-04 — P1 — `src/trace_opd/train/agent_loop.py:308-317`；`src/trace_opd/common/config.py:182-185`

**问题陈述**

配置声明 `steering.obs_topk` 是 hook 与 vLLM `max_logprobs` 的单一真源，launcher 也按该值配置引擎；但构造 `SpsGenerate` 时没有传入 `obs_topk=st_cfg.obs_topk`，hook 实际静默采用 `4 * topk` 默认值。

**失败场景**

合法配置 `topk=64, obs_topk=128` 会让 launcher 将 vLLM 上限设为 128，而 hook 仍请求 `logprobs=256/prompt_logprobs=256`，首个 SPS 请求在引擎侧失败。反向配置 `obs_topk=512` 则 manifest/fingerprint 声称观察 512，实际只观察 256，候选覆盖与实验定义不符。

**修复建议**

在 `SpsGenerate(...)` 显式传 `obs_topk=st_cfg.obs_topk`；增加非默认 `obs_topk` 的 agent-loop 接线测试，并断言 launcher 上限、hook 请求宽度和 fingerprint 三者一致。

### IF-05 — P1 — `src/trace_opd/steering/gate.py:74-95`；`src/trace_opd/train/agent_loop.py:198-201`

**问题陈述**

spec §2.3 要求训练中 robust-EMA 慢更新，并在触发率越界时“告警降级”。实现只提供未被生产路径调用的 `robust_quantile_update()`；`rate_alarm()` 仅在单条轨迹累计至少 50 次 gate decision 后返回字符串，agent loop 也只记录 warning，不执行降级。门对象每条轨迹重建，典型短轨迹甚至永远达不到 50 次阈值。

**失败场景**

策略分布漂移后，冻结 P85 使开门率接近 100%；系统仍按余额在轨迹最前部持续干预，仅可能输出一条 warning，既不切回纯 π，也不聚合跨轨迹触发率。由此违反预注册率界保护，并放大“早花晚饿”偏差。

**修复建议**

要么按 spec 在训练步边界聚合 D/门率并更新下一步冻结阈值，越界时显式将后续 rollout 降级为 η=0；要么正式裁定为“全程冻结、只告警”的较弱版本并同步修改 spec。无论哪种，都应跨轨迹聚合，而不是把 50 次门决策门槛放在 episode 局部对象中。

### IF-06 — P1 — `src/trace_opd/steering/hook.py:157-160,192-194,273-311`

**问题陈述**

权威 spec §2.5 和实现设计 §3 都规定 bare/teacher 两次 verify，并要求接受比中的 π 来自同一 verify scoring 口径；当前 hook 只做 teacher verify，π top-K 和 sampled-token logprob直接取 draft 生成结果。代码因此没有实现文档声明的双 verify，也无法测量/约束 draft-vs-bare-verify 数值残差。

**失败场景**

当生成路径与 prompt scoring 路径因 logits processor、量化、`logprobs_mode` 或内核数值产生差异时，实际核是“以 draft q 为 π 构造的另一 b”，而不是规范所定义、记账所声称的 bare verify π 核；命题 A 最多能相对这个实现对象成立，不能按现有文字直接宣称相对规范对象精确。P4 清单要求测的 π 残差也因 bare verify 根本不存在而无数据来源。

**修复建议**

先明确唯一数学对象：若坚持 spec，应增加 bare scoring、验证 q≈π_verify，并按裁定处理残差；若为了 speculative 精确性选择实际 draft q 作为 proposal/核的 π 侧，则应修改 spec、命名和记账说明，并显式记录 q-vs-bare residual。两种口径都需锁死所有 logits processor，而不能维持“文档双 verify、代码单 verify”的状态。

### IF-07 — P2 — `src/trace_opd/common/config.py:332-344`

**问题陈述**

`objective_fingerprint()` 未包含新增且直接改变梯度的 `train.adv_estimator`，也未包含 `ppo_mini_batch_size` 等命题 C 单快照/单步相关口径；`doc_table` 只哈希路径字符串，不哈希内容。

**失败场景**

同一 run 目录先用 `grpo`，再把配置改成 `grpo_loo` 并 `resume_mode=auto`，fingerprint 不变，旧 checkpoint 会在另一 advantage 定义下继续训练而不触发守卫。同路径 doc JSONL 被替换也同样不可见。

**修复建议**

把所有会改变采样分布、advantage、loss 聚合或更新次数的字段纳入规范化 fingerprint；对 doc 表至少加入内容摘要（以及可选的数据版本/行数）。为每个字段增加“改单项必改 fingerprint”的参数化测试。

### IF-08 — P2 — `src/trace_opd/steering/ledger.py:31-55`；`scripts/calibrate_theta_d.py:61-81`

**问题陈述**

D 直方图上限硬编码为 `6.25=c^2`（只适配 c=2.5）。配置允许其他 c；c>2.5 时所有超界 D 被压进末桶，标定脚本仍输出不超过 6.25 的“suggested theta_d”，仅在末桶占比大于 2% 时打印 warning。

**失败场景**

以 c=3 做目标语境标定，真实 P85 若为 7.2，会被输出为末桶内约 6.2 的 floor；正式 run 的开门率高于预期。末桶占比不超过 2% 时高分位（如 P99）仍可能失真却不告警；即使告警，脚本仍给出可被误用的建议值。

**修复建议**

让直方图范围随运行配置 c 写入 ledger 元数据，聚合时拒绝混合不同坐标系；目标分位落入 overflow bucket 时不要给数值建议，改为 fail-loud 或要求扩大 bins 后重跑。另校验 `q∈[0,1]` 与 `sum(hist)==d_count`。

### IF-09 — P1 — `src/trace_opd/train/agent_loop.py:207,341-360`；`src/trace_opd/core/episode.py:181-187,292-295`

**问题陈述**

基础设施错误只在“尚无任何 response token”时重试并最终抛出；一旦错误发生在多轮 episode 的第二轮或更晚，已有的前轮 token 使两个 `and not result.tokens.response_ids` 条件都为假，`ENV_ERROR` 结果直接返回。随后输出层以 `traj.won` 计算奖励，基础设施错误因此被编码为 0 分样本，而不是从训练批次剔除。

**失败场景**

第一轮执行成功并产生 40 个 token；第二轮 vLLM verify 超时、环境 step 抛异常或 async server abort。`run_episode` 返回 `outcome=ENV_ERROR` 且保留前 40 个 token，`_student_rollout()` 不重试、不抛错，最终把一个本来未知的任务结果作为失败轨迹送进 GRPO。若某类长轨迹更易遇到超时，这会形成与长度/状态相关的系统性负奖励偏差。

**修复建议**

任何 `ENV_ERROR` 都不得成为有效 attempt。对整个 episode 使用新环境状态和新 steering state 重试一次；再次失败则抛出/由 TQ 明确标记该 rollout 失败并补样，绝不能按 `won=False` 计 0 分。增加“首轮已有 token、次轮 generate/verify/env-step 分别失败”的三组单测，并断言没有 reward 样本入队。

### IF-10 — P1 — `src/trace_opd/core/episode.py:147-154`；`src/trace_opd/train/agent_loop.py:280-292`；`src/trace_opd/steering/hook.py:115-149,288-289`

**问题陈述**

episode 对数据集初始观察与环境实际观察只比较 `strip()` 后文本；SPS 却预先用 `adapter.initial_obs` 构造 student/teacher prompt，并只保存 `student_prompt_len`。生成时既不保存也不校验实际 student prompt token 前缀是否逐 token 相同，随后按预计算长度切 `response_stream`。这与注释声称的“逐字节同参”不符，也没有实现审阅要求中的 teacher/student prompt 一致性断言。

**失败场景**

数据集观察为 `"state"`，环境返回 `"state\n"`：一致性检查放行，但聊天模板可能多出一个 token。首次 generate 时 `stream_len` 已为 1，ledger position 从 1 开始；teacher verify 的 `response_stream` 还夹入 student prompt 尾 token。若长度恰好不变但 token ID 改变，则断言也不会触发，却在与实际 student prefix 不同的 teacher prefix 上计分，导致 b、位置记账和行为权重全部错位。

**修复建议**

要么 steering 路径要求原始初始观察精确相等并在差异时降级/报错，要么等环境 reset 得到真实观察后再构造两套 prompt。`SpsGenerate` 应保存完整 `student_prefix_ids`，每次调用先断言 `prompt_ids[:len(prefix)] == prefix`，并据此切 response；同时对 teacher prefix 做完整、而非仅生成尾部的可验证对齐。补空白、Unicode 归一化和“等长不同 token”测试。

### IF-11 — P1 — `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py:684-696`；`verl/verl/workers/rollout/vllm_rollout/utils.py:659-671`

**问题陈述**

response top-K 行先以 `token_id=0, logp=0.0` 填充，再按 `rank-1` 覆盖；prompt top-K 则以 `None` 填充。两条路径都只跳过 sampled-token 的 `rank>K`，没有验证 rank 非空、位于 `[1,K]`、互异且最终无空洞。只要 vLLM 版本/API、logits processor 或返回异常造成 rank 缺失，response 路径就会把“概率 1 的 token 0”伪装成真实候选，prompt 路径则把 `None` 留到更下游才失败。

**失败场景**

K=3 的一行只返回 rank 1 和 3（另有 sampled token rank 7）。代码产生 `ids=[t1,0,t3]`、`logps=[lp1,0,lp3]`；核把 token 0 当成 π 概率 1 的候选，可能因 S 质量超过 1 直接 abort，也可能在 ID 重复/后续字典覆盖时静默构造错误 b。相同空洞发生在 prompt scoring 时会以 `float(None)`/索引错误迟发失败，错误位置和请求上下文已经丢失。

**修复建议**

先收集所有 `rank<=K` 项，严格验证 ranks 恰为 `1..K`、token ID 唯一、logprob 有限，再生成稠密行；sampled token `rank>K` 可继续单独保留在 sampled logprob 通道，但绝不能用于填 top-K。验证失败应携带 request ID、token position、K 和原始 ranks fail-loud。增加 rank=None、重复、越界、空洞以及 sampled-token rank>K 的契约测试；该契约还需纳入支持的每个 vLLM 版本真机冒烟。

### IF-12 — P2 — `verl/verl/trainer/config/ppo_trainer.yaml:42-43`；`verl/verl/trainer/config/distillation/distillation.yaml:1-113`；`verl/verl/workers/engine/fsdp/transformer_impl.py:654-703,1250-1467`；`src/trace_opd/core/episode.py:323-360`

**问题陈述**

P1.5 所称“旧机制全面出清”并未完成。默认 PPO Hydra defaults 仍装载一整棵 distillation 配置，其 `_target_` 指向已经不存在的 `verl.workers.config.DistillationConfig`；FSDP（以及 Megatron/VeOmni）仍保留 `research_mask`、`tg_mask`、`distillation_use_topk`、`distillation_only` 活分支；trace_opd 还留有只服务旧 SFT/mixed-objective 的 `replay_bare_tokens()`，以及旧机制命名/docstring。当前主路径没有读取 `config.distillation`，所以默认关闭时不直接改变 PGS loss，但“纯 GRPO 面、无旧机制配置/代码”的守护结论不成立。

**失败场景**

调试工具或未来公共代码尝试把默认 `config.distillation` 转成 dataclass，立即因目标类已删而 import/instantiate 失败；用户看到仍存在的 `distillation.enabled` 并尝试启用，会进入半删除状态。若旧 batch key 偶然泄漏，engine 中休眠分支还可重新改变 logprob/归一化行为，而现有字符串守护测试因检查面不全不会报警。

**修复建议**

若目标确为彻底根除，删除 defaults 项、distillation YAML/生成配置残段、三种 engine 的专用分支、旧 replay/API/docstring，并用覆盖 `src/`、`verl/`、configs 的允许名单式 `git grep` 守护；若这些是有意保留的 verl 原生通用能力，则恢复完整 dataclass/测试、去掉 trace-opd 旧机制耦合，并在设计文档明确其不被 PGS launcher 接线。不能维持“配置仍暴露、目标类已删除”的半死状态。

## 忠实性核对表（待完成）

| spec 条款 | 代码位置 | 阶段性判定 |
|---|---|---|
| §2.1 保尾核 | `steering/kernel.py` | 主公式、S 外恒 π、S 质量守恒与 ratio 界正确 |
| §2.2 截断 | `kernel.py:28-36` | 正确 |
| §2.3 D 门/探针 | `gate.py`、`hook.py` | 可测接口基本正确；慢更新、越界降级与逐 token 审计未闭合（IF-03/05） |
| §2.4 D∞ 预算 | `budget.py`、`hook.py` | `δ=2ηc` 与余额缩 η 主路径正确；最终边界继续核查 |
| §2.5 SPS | `sps.py`、`hook.py` | 单点恒等与拒绝续采主路径正确；双 verify 口径不忠实（IF-06），异常/EOS 路径继续核查 |
| §2.6 训练 | `steering_weights.py`、`agent_loop_tq.py`、`core_algos.py` | clip 外乘权重的 loss 形态正确；TQ 默认配置阻断（IF-01），scale 非严格 LOO（IF-02） |
| §3 命题 A-D | 多处 | A/B 的纯模块对象基本成立；C 的实现条件未完全满足；D 为规范数例、代码无额外外推 |
| §4 记账 | `ledger.py`、`hook.py` | 不完整（IF-03） |

## 测试缺口清单（待完成）

- 缺真实 Hydra 默认配置下的 TQ/W_beh 入队集成测试，因而未发现 IF-01。
- verify mock 每行分布相同，任何行偏移一位的 bug 都会通过；需要位置相关 oracle。
- 拒绝续采断言含宽松的 `or len(draft.calls) >= 2`，没有逐次证明“修正前缀 + 重采 token”精确一致。
- 缺 EOS/stop token 被拒、空 draft、draft/verify aborted、畸形或不完整 top-K 行、输出超过 max_tokens 的测试。
- 缺多轮中发生真实拒绝后的 ledger 坐标测试；现有 episode 集成只覆盖 D=0、门恒关透明路径。
- 缺默认关闭 rollout correction 与 PGS 字段混合批次的实际 TransferQueue 测试。
- `grpo_loo` 数值测试验证了稳定性，却没有验证 scale 对本样本 outcome 的依赖及相应偏差。
- 已知只能真机验证的 prefix-cache×prompt_logprobs、`logprobs_mode` 与 draft/verify 数值残差仍按 P4 清单保留。

## 四个实现口径定案的独立判定（待完成）

1. **S 的两个封闭点**：阶段性判定为“作为预先固定的自选 S，不破坏相对该 b_S 的 SPS 恒等；会降低教师冷 token 覆盖，需如实命名，不能解释为理想双 top-K 并集”。
2. **p_T 观测窗外 floor 代入**：阶段性判定为“定义了可采样的近似核，clip 饱和时无误差；未饱和时所谓方向保守需要结合行提取/floor 完整性继续核查”。
3. **`max(std_loo,std_full)`**：不满足严格 LOO/命题 C，但可作为明确有偏的稳定性档；见 IF-02。
4. **借原生 `rollout_is_weights` 通道**：loss 中 clip 外乘权重的选择本身正确且改动面小；当前互斥守卫使默认配置不可运行，故落地判定暂为不通过；见 IF-01。
