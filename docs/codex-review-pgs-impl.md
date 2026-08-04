# PGS 完整实施深度代码审阅

> 审阅区间：`c9d2466..c2096e3`，共 8 个 commit（`c623d85` → `c2096e3`）。
>
> 审阅对象固定为提交 `c2096e3` 的 Git 对象。审阅途中外部并发工作将分支推进到 `6194b94` 并继续留下未提交修改；本报告后半程全部使用显式 `c2096e3:<path>` 复核，新增提交与工作树修改均不纳入判断，也未被本审阅触碰。

## 总体结论

**一句话判断：机制的纯数学模块扎实、模块边界也基本清楚，但该 8-commit 快照尚不具备正式实验就绪性——两个 P0 分别使 SPS 请求超过 vLLM 上限、使非空 steering ledger 无法入队，另有行为核近似、LOO、异常样本处理和审计闭环等 P1 会影响机制结论。**

最终计数：**P0 2 条 / P1 10 条 / P2 5 条**。建议先修完全部 P0/P1，再进入限定范围的 P4 真机冒烟；当前结果不能用于宣称已忠实实现 v3.0-final 的全部命题条件。

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

### IF-03 — P1 — `src/trace_opd/steering/hook.py:201-258`；`src/trace_opd/steering/ledger.py:17-86`

**问题陈述**

spec §4 要求逐 token 记录完整 `gate_state`、预算余额、`steer_flag` 与 `reject_count`；实际 ledger 只在执行了正 η，或“`base_open or probe_flipped` 且 η=0”时追加稀疏行。最常见的“主门关闭且探针未翻转”位置完全消失；预算耗尽或 cap-active 后直通的 token 也不记录。更具体地，触发 R_max 的记录在 `record_reject()` 之前硬写 `cap_active=False`，而后续 cap-active token 在入口直接 `continue`，所以导出的 `steer_cap_active` 正常情况下根本记不到 1。现有字段还缺逐位置预算余额、明确的 steer flag 和累计 reject count，而 `steer_count=len(records)` 会把零 η 的关闭/阻断行也计作 steer。

**失败场景**

主门命中率 15%、epsilon=5% 时，大部分候选位置没有逐 token gate 记录。某一 turn 第 R_max 次拒绝后，汇总虽可能增加 `turns_capped`，逐位置数组却全是 `cap_active=0`，无法定位从哪一 token 起进入 base-only 状态。事后也无法从 ledger 重建 probe 分配的控制总体或预算 noncompliance；`steer_count` 曲线还可能把“门开但被翻关/预算阻断”误报为实际干预。训练权重在这些未干预位置仍为 1，因而不一定立即数值错误，但 NF07 的可识别性修法和 spec §4 的审计红线没有落地。

**修复建议**

分离两种数据面：行为校正可继续使用只含非零 ratio 的稀疏表；审计 ledger 必须逐生成 token 写全 `D/base_open/assigned_open/probe/propensity/eta/delta/budget_remaining/cap_active/steer_flag/reject_count`。为计数另设 `actual_steer_count`，不要复用记录行数。

### IF-04 — P0 — `scripts/launch_train.py:73-81`；`src/trace_opd/steering/hook.py:124-134,159,297`；`src/trace_opd/train/agent_loop.py:308-317`

**问题陈述**

`c2096e3` 没有 `SteeringCfg.obs_topk`。launcher 只把 vLLM 启动上限设为 `max(topk,20)`，而 `SpsGenerate` 在调用方未传参时静默采用 `4*topk`，并把这个更大值同时用于 draft `logprobs` 和 verify `prompt_logprobs`。所有仓库配置均为 `topk=64`，所以引擎上限是 64、hook 实际请求是 256；P4 清单所写的“max_logprobs=256”没有落进命令生成。

**失败场景**

直接运行提供的 `162-pgs-calib.yaml` 或 `162-pgs-s1.yaml`：首个 SPS draft 发送 `logprobs=256`，vLLM 以配置上限 64 拒绝请求，episode 走基础设施失败路径；即使 draft 偶然越过校验，teacher verify 的 `prompt_logprobs=256` 仍会失败。标定和正式 S1 都无法产生机制数据，因此是独立于 IF-01 的全链阻断。

**修复建议**

引入显式 `steering.obs_topk`（默认 256/或由 `4*topk` 唯一派生），由同一值驱动 launcher 的 vLLM 上限与 `SpsGenerate` 两类请求，并校验 `obs_topk>=topk`；不要在 hook 内保留隐藏倍率默认。增加“真实生成命令 + 构造后的 hook 请求参数”集成断言，覆盖 calib/base/S1 三份配置。

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

### IF-07 — P2 — `src/trace_opd/common/config.py:323-335`；`scripts/launch_train.py:179,297-310`

**问题陈述**

`objective_fingerprint()` 未包含新增且直接改变梯度的 `train.adv_estimator`，也未包含 `ppo_mini_batch_size` 等命题 C 单快照/单步相关口径；`doc_table` 只哈希路径字符串，不哈希内容。launcher 还把所有未知参数原样追加到 verl 命令末尾，受保护的算法字段可在 fingerprint 计算后被 Hydra 的末项覆盖；manifest 虽记录字符串，却不参与 resume 守卫。

**失败场景**

同一 run 目录先用 `grpo`，再把配置改成 `grpo_loo` 并 `resume_mode=auto`，fingerprint 不变，旧 checkpoint 会在另一 advantage 定义下继续训练而不触发守卫。同路径 doc JSONL 被替换也同样不可见。即使日后把字段补进 Config，命令尾传 `algorithm.adv_estimator=grpo`、loss mode 或 PPO epochs 仍可让实际命令与 fingerprint 分叉。

**修复建议**

把所有会改变采样分布、advantage、loss 聚合或更新次数的字段纳入规范化 fingerprint；对 doc 表至少加入内容摘要（以及可选的数据版本/行数）。对 protected Hydra keys 禁止 passthrough 覆盖，或解析最终有效命令后再做规范化摘要。为每个字段和一组尾部覆盖增加“改单项必改 fingerprint/受保护项拒绝启动”的参数化测试。

### IF-08 — P2 — `src/trace_opd/steering/ledger.py:31-55`；`scripts/calibrate_theta_d.py:61-81`

**问题陈述**

D 直方图上限硬编码为 `6.25=c^2`（只适配 c=2.5）。配置允许其他 c；c>2.5 时所有超界 D 被压进末桶，标定脚本仍输出不超过 6.25 的“suggested theta_d”，仅在末桶占比大于 2% 时打印 warning。

**失败场景**

以 c=3 做目标语境标定，真实 P85 若为 7.2，会被输出为末桶内约 6.2 的 floor；正式 run 的开门率高于预期。末桶占比不超过 2% 时高分位（如 P99）仍可能失真却不告警；即使告警，脚本仍给出可被误用的建议值。

**修复建议**

让直方图范围随运行配置 c 写入 ledger 元数据，聚合时拒绝混合不同坐标系；目标分位落入 overflow bucket 时不要给数值建议，改为 fail-loud 或要求扩大 bins 后重跑。另校验 `q∈[0,1]` 与 `sum(hist)==d_count`。

### IF-09 — P1 — `src/trace_opd/train/agent_loop.py:207,340-359`；`src/trace_opd/core/episode.py:181-187,292-295`

**问题陈述**

基础设施错误只在“尚无任何 response token”时重试并最终抛出；一旦错误发生在多轮 episode 的第二轮或更晚，已有的前轮 token 使两个 `and not result.tokens.response_ids` 条件都为假，`ENV_ERROR` 结果直接返回。随后输出层以 `traj.won` 计算奖励，基础设施错误因此被编码为 0 分样本，而不是从训练批次剔除。

**失败场景**

第一轮执行成功并产生 40 个 token；第二轮 vLLM verify 超时、环境 step 抛异常或 async server abort。`run_episode` 返回 `outcome=ENV_ERROR` 且保留前 40 个 token，`_student_rollout()` 不重试、不抛错，最终把一个本来未知的任务结果作为失败轨迹送进 GRPO。若某类长轨迹更易遇到超时，这会形成与长度/状态相关的系统性负奖励偏差。

**修复建议**

任何 `ENV_ERROR` 都不得成为有效 attempt。对整个 episode 使用新环境状态和新 steering state 重试一次；再次失败则抛出/由 TQ 明确标记该 rollout 失败并补样，绝不能按 `won=False` 计 0 分。增加“首轮已有 token、次轮 generate/verify/env-step 分别失败”的三组单测，并断言没有 reward 样本入队。

### IF-10 — P1 — `src/trace_opd/core/episode.py:147-154`；`src/trace_opd/train/agent_loop.py:280-292`；`src/trace_opd/steering/hook.py:118-148,287-288`

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

### IF-12 — P2 — `verl/verl/trainer/config/ppo_trainer.yaml:42-43`；`verl/verl/trainer/config/distillation/distillation.yaml:1-113`；`verl/verl/workers/engine/fsdp/transformer_impl.py:654-703,1250-1467`；`verl/tests/utils/test_special_megatron_kl_loss_tp.py:24-25`；`src/trace_opd/core/episode.py:323-360`

**问题陈述**

P1.5 所称“旧机制全面出清”并未完成。默认 PPO Hydra defaults 仍装载一整棵 distillation 配置，其 `_target_` 指向已经不存在的 `verl.workers.config.DistillationConfig`；FSDP（以及 Megatron/VeOmni）仍保留 `research_mask`、`tg_mask`、`distillation_use_topk`、`distillation_only` 活分支；trace_opd 还留有只服务旧 SFT/mixed-objective 的 `replay_bare_tokens()`，以及旧机制命名/docstring。四个 vendored 测试文件也仍 import 已删除的 `verl.trainer.distillation.*` 模块。当前主路径没有读取 `config.distillation`，所以默认关闭时不直接改变 PGS loss，但“纯 GRPO 面、无旧机制配置/代码”的守护结论不成立，完整 verl test collection 也不再闭合。

**失败场景**

调试工具或未来公共代码尝试把默认 `config.distillation` 转成 dataclass，立即因目标类已删而 import/instantiate 失败；用户看到仍存在的 `distillation.enabled` 并尝试启用，会进入半删除状态。完整收集 `verl/tests/` 会先在上述遗留测试的 import 阶段失败。若旧 batch key 偶然泄漏，engine 中休眠分支还可重新改变 logprob/归一化行为，而现有字符串守护测试因检查面不全不会报警。

**修复建议**

若目标确为彻底根除，删除 defaults 项、distillation YAML/生成配置残段、三种 engine 的专用分支、旧 replay/API/docstring，并用覆盖 `src/`、`verl/`、configs 的允许名单式 `git grep` 守护；若这些是有意保留的 verl 原生通用能力，则恢复完整 dataclass/测试、去掉 trace-opd 旧机制耦合，并在设计文档明确其不被 PGS launcher 接线。不能维持“配置仍暴露、目标类已删除”的半死状态。

### IF-13 — P1 — `src/trace_opd/steering/hook.py:20-22,76-93`；`docs/pgs-impl-design.md:109-113`

**问题陈述**

对属于 π 侧 top-K、但落在 teacher top-K 窗外的 token，代码用 teacher 第 K 名 logprob（floor）代替真实 `log p_T`。真实值只知道“不大于 floor”；这个上界代入并不保证文档暗示的方向保守，甚至可把 `log p_T-log π` 的符号翻转。SPS 和预算命题仍相对这个人为定义的近似核成立，但该核不再保证沿真实 teacher 偏好倾斜，D 门也会由同一伪差异触发。

**失败场景**

某 π-only 候选满足 `π(t)=1e-4`、teacher 第 K 名 `p_floor=1e-3`，而真实 `p_T(t)=1e-8`，取 `c=2.5`。真实增量应为 `clip(log(1e-8/1e-4))=-2.5`，floor 代理却给出 `log(10)≈+2.30`：实现把一个 teacher 强烈反对的 token 当成偏好 token 提升。只要此类位置未达到注释所假设的“负侧 clip 已饱和且 floor 不影响”，形式上的 privileged guidance 方向就可能反转；当前 ledger 也没有统计发生率。

**修复建议**

不要把上界当点估计。最稳妥的是扩大/精确取得 S 内 teacher logprob；若必须 top-K 封闭，可对未知项取中性 `log p_T:=log π`（令该 token 的增量为 0），或用区间鲁棒核并把最坏方向写进规范。至少记录 floor-substitution 数、`floor>π(t)` 的符号不确定率和质量占比，先用真实模型验证“多数已饱和”的经验前提；在此之前不要称其为方向保守。

### IF-14 — P2 — `verl/verl/trainer/ppo/steering_weights.py:27-31,87-96`

**问题陈述**

三种展开的代码行为与测试基本一致，但命名/注释夸大了两个消融：`per_decision` 实际是“截至当前位置的前缀乘积（PDIS）”，不是把完整轨迹乘积施加到所有 score term 的 trajectory IS；`vtrace` 只是逐 token 的 `min(1,π/b)`，没有 V-trace 的 value target、`c` trace 系数或时序递推。对终局 outcome reward，前缀权重尤其不能自动等同于完整轨迹换测度。

**失败场景**

实验表把 `w_beh_mode=per_decision` 标成“完整 trajectory IS”并据此声称满足命题 C，或把 `vtrace` 档结果解释为标准 V-trace；实际早期 token 的 loss 没有乘未来动作比，后者也只是局部截断权重。即使代码按设计运行，论文中的 estimand/偏差结论仍会被误报。

**修复建议**

把模式改名或至少在配置、实现文档和实验表中统一标为 `per_token`、`prefix_pdis`、`clipped_per_token`；若真要提供 full-trajectory 档，应把整条有效 response 的 log-ratio 和应用到所有有效 score term，并单独处理数值稳定性。只有实现完整 V-trace 递推时才使用 `vtrace` 名称。

### IF-15 — P1 — `src/trace_opd/steering/hook.py:172-174,253-262`；`src/trace_opd/train/agent_loop.py:308-317`

**问题陈述**

hook 只用原 draft 是否提前结束来判断 EOS/stop；拒绝后从残差分布实际执行的 replacement token 没有重新做停止判断。任何 rejection 都设置 `rejected=True` 并无条件 `continue` 新一轮 draft，因此“残差恰好采到 EOS/stop”时仍会在已终止前缀后继续生成。这破坏了断点续采的停止语义，任务书点名要求的 EOS×拒绝交互没有被实现。

**失败场景**

draft 在普通 token `y` 上被拒绝，残差分布采出 EOS。代码先把 EOS 追加到 `executed`，然后再次调用 draft；vLLM 不会因为 prompt 尾部已有 EOS 自动把这个新请求视为上一请求已停止，于是返回 EOS 后的额外 token。最终 decode 又可能因 `skip_special_tokens=True` 隐去中间 EOS，环境收到拼接文本，ledger/W_beh 也包含本不应存在的后续位置，实际 episode 分布不再是按 b_S 逐 token 直到停止的分布。

**修复建议**

将本次采样配置实际生效的所有 stop token ID（至少 tokenizer EOS，以及框架配置的 stop IDs）显式传入 hook；每次 append **实际执行 token** 后立即判断停止。若 replacement 是 stop，结束本轮且不再 draft；若原 draft 的 stop token 被拒绝并替换成非 stop，则丢弃原 suffix 后从 replacement 前缀继续。增加这两个对偶分支以及 stop-string/tokenizer 多 EOS 的测试。

### IF-16 — P1 — `src/trace_opd/train/agent_loop.py:187-200`；`verl/verl/trainer/ppo/v1/agent_loop_tq.py:189-211`；`verl/verl/utils/tensordict_utils.py:927-930`

**问题陈述**

只有 `state is not None` 的轨迹才导出 ledger，且 TQ 只有在 `extra.get("steer_positions")` 非空时才创建顶层 `rollout_is_weights`。因此同一 PGS batch 中“至少一次实际 steering”和“全程门关/缺 doc/doc overflow”的行具有不同 schema。下游 `list_of_dict_to_tensordict()` 不是做 key 并集，而是只取第一行 keys，再按这些 key 索引所有行；W_beh 字段于是具有顺序相关的静默丢失/崩溃行为。

**失败场景**

batch 两行 A/B：A 全程门关，B 有非零 steering。若 A 在第一行，首行 keys 不含 `rollout_is_weights`，B 已算出的权重被整个忽略，倾斜轨迹以全 1 未校正训练；若 B 在第一行，转换器会读取 A 的缺失 key 并抛 `KeyError`。doc missing/overflow、预算始终未用或短响应均可自然制造 A，因此这不是畸形输入。

**修复建议**

用明确的“PGS 协议在场”标志判断，而不是用非空 ledger 判断；steering-enabled 训练的每条轨迹都导出相同字段，空账展开为长度匹配的全 1 dense weight。也可在 batch 组装层对 key 并集补默认值，但必须 fail-loud 检查 schema。增加 A/B 两种顺序、缺 doc、overflow、全门关与非空 ledger 混合的 TQ 测试，并断言 steered 行的非 1 权重最终到达 actor batch。

### IF-17 — P2 — `src/trace_opd/common/config.py:221-240`；`src/trace_opd/steering/budget.py:19-45`；`src/trace_opd/steering/gate.py:39-47`

**问题陈述**

`SteeringCfg` 已解决 YAML 科学计数法字符串并做 float/int 强转，但统一 `_validate()` 没有检查数值有限性，也没有检查 `topk>=1`、`r_max>=1` 和合法的 `gate_rate_bounds`。Python 对 NaN 的大小比较为 false，导致 `c/b_max/theta_d` 的现有条件可被绕过；部分整数错误会延迟到状态构造，另一些配置会静默改变机制。

**失败场景**

YAML 写 `theta_d: .nan`：配置成功加载，`d_value >= theta_d` 永远为 false，正式 steering 静默退化成门恒关；`b_max: .nan` 通过 `<0` 检查，预算 `remaining=max(0,nan)` 的行为可把干预折成 0 或在一次扣费后污染状态。`topk: 0` 则直到首个候选位置才形成空 S/除零，浪费一次正式 rollout 才失败；逆序 rate bounds 只会产生无意义告警。

**修复建议**

在配置加载完成后集中检查所有浮点字段 `math.isfinite`，并验证 `topk>=1`、`obs_topk>=topk`、`r_max>=1`、`0<=lo<hi<=1`；底层 `TrajectoryBudget/SteeringGate/SteeringKernel` 仍保留同样的防御，以免绕过 Config 直接构造。增加 `.nan/.inf`、0/负整数和逆序 bounds 的参数化 fail-fast 测试。

## 可读性与结构评价

整体模块边界是本实施的强项：`kernel.py`、`sps.py`、`gate.py`、`budget.py` 都是小而纯的数学/状态模块，约束靠近数据结构，易做性质测试；`hook.py` 把教师限制在采样端，训练侧只接稀疏账本，红线在目录结构上可见。verify 行语义的注释也准确解释了 `r→r+1` 和末行 dummy，实际下标核对通过。

主要结构债务集中在三处：

- backend 的 logprob 明细用若干动态 `extra` key 和平行 list 隐式约定，没有 typed row/schema validator，因而宽度、rank、行数、有限性和 stop 契约分别散落在 hook/server（IF-04/11/15）。建议定义一个经过单点校验的 `ScoredTokenBlock`/协议对象。
- agent loop 只把 `student_prompt_len` 交给 hook，而不是把经过环境 reset 的完整 prefix 作为不变量，导致字符串/模板一致性无法在边界层证明（IF-10）。prompt 构造应由同一阶段完成并传完整 token prefix。
- `SteeringLedger` 同时承担训练稀疏 ratio 表、因果审计和曲线汇总，三种粒度互相妥协（IF-03/16）。应把 correction table 与 audit/aggregate schema 分离，并给每个字段定义是否逐 token、逐 decision 或逐 trajectory。

需要纠正的文档失真包括：`SpsOutcome.log_pi`/hook 文字暗示同一 bare verify、实际取 draft q（IF-06）；prefix PDIS 和单步截断被称为 trajectory IS/V-trace（IF-14）；包级 docstring、Hydra defaults 和 engine 注释仍宣称旧 distillation/research 机制（IF-12）。关键输入契约目前还多处用可被 `python -O` 删除的 `assert`；修 IF-10/11 时应改为带请求/位置上下文的显式异常。

## 忠实性核对表

| spec 条款 | 代码位置（均指 `c2096e3`） | 判定 |
|---|---|---|
| §2.1 固定 S 与保尾核 | `steering/hook.py:59-95`；`steering/kernel.py:40-141` | **部分通过。** S 在当前 token 判定前由两侧行分布固定；核在 S 外严格等于 proposal、S 内保持原质量且归一化正确。两个观测封闭点定义了另一合法 operational S，但不是字面理想并集；teacher floor 还可翻转方向（IF-13）。 |
| §2.2 截断增量 | `steering/kernel.py:28-35,72-94` | **通过。** `clip(log p_T-log q,±c)`、几何倾斜和 `|log(b/q)|<=2ηc` 的实现/断言一致。这里的 π 侧实际是 draft q（见 IF-06）。 |
| §2.3 D 门与 ε 探针 | `steering/gate.py:29-95`；`steering/hook.py:205-216` | **部分通过。** D、主门和 probe 的函数签名不接当前 proposal token，满足可测性红线；随机翻转概率也正确。训练期 robust-EMA、跨轨迹率界和越界降级未接线，审计总体不全（IF-03/05/08）。 |
| §2.4 D∞ 预算 | `steering/budget.py:16-46`；`steering/hook.py:185,201-250` | **通过数学主路径。** 逐位置先按余额缩 η、再扣 `δ=2ηc`，恰好耗尽后转 base，不会透支；轨迹密度比界随扣费相加。耗尽/cap 尾段的逐 token 状态没有如实落账（IF-03），不影响采样界但影响可审计性。 |
| §2.5 SPS 与 R_max | `steering/sps.py:31-78`；`steering/hook.py:155-311` | **部分通过。** 单点接受 `min(1,b/q)`、残差 `norm((b-q)_+)`、拒绝后丢后缀并从修正前缀续采、每 turn cap 主路径正确。`_verify_rows` 的 `base=T+M+E-1` 与“行 r 预测 r+1”语义对齐。缺 bare verify，残差 EOS 不终止，prompt 精确前缀、异常与 rank 契约也未闭合（IF-06/09/10/11/15）。 |
| §2.6 W_beh 与 LOO | `steering_weights.py:42-99`；`agent_loop_tq.py:189-211`；`core_algos.py:640-699,1340-1420` | **代码形态部分通过、集成不通过。** vanilla loss 先对 `πθ/πold` clip，再在外部乘 W，位置正确；per-token 与前缀积展开/尾部延续正确。默认 Hydra 守卫会拒绝非空账（IF-01），混合账字段又会丢权重或崩溃（IF-16）；scale 不是严格 LOO（IF-02），模式命名有误（IF-14）。 |
| §3 命题 A | `kernel.py`；`sps.py`；`hook.py` | **有条件成立。** 若 vLLM top-K 行完整，输出单点精确服从以真实 draft q 构造的 operational `b_S(q,p_T)`；相对规范所写 bare-verify π、理想 p_T 值和完整停止轨迹，现代码不能直接引用命题 A（IF-06/11/13/15）。 |
| §3 命题 B | `kernel.py:91-94,139-141`；`budget.py:30-46` | **成立于 operational q-kernel。** S 外 ratio=1，S 内逐点界和轨迹累计界正确。它不自动证明 q≈训练端 `π_old`。 |
| §3 命题 C | `core_algos.py:640-699,1340-1420`；`steering_weights.py` | **不满足默认实现条件。** 默认 per-token 不是整轨无裁剪换测度，q 与 `π_old` 未同对象验证，scale 包含当前 outcome；单快照/单 epoch 虽靠 verl 默认与整批 mini-batch 配置近似满足，也未被完整 fingerprint/守卫锁死。只能引用 spec 已承认的一阶有偏工程档，不能声称期望梯度等于 `∇J`。 |
| §3 命题 D | `tests/steering/test_steering.py` | **仅窄范围通过。** 单点二元数例有对应数值测试；实现没有把它外推到 episode，这一点忠实。 |
| §4 extra_fields | `steering/ledger.py:17-86`；`steering/hook.py:201-258` | **不通过。** 行为校正需要的稀疏 `log_q/log_b` 足够，但规范要求的逐 token gate/budget/cap/steer/reject 状态无法重建，且 `cap_active` 逐行字段事实上记不到真值（IF-03）。 |
| “奖励只来自 verifiable outcome；loss 无教师”红线 | `train/agent_loop.py:207`；`scripts/launch_train.py:84-111`；`core_algos.py:1340-1420` | **活跃 PGS 路径通过。** reward 由 `traj.won` 产生，教师只在采样 hook；vanilla policy loss 无蒸馏项。但默认配置/engine 仍有半死 distillation/research 残留（IF-12），故“全面根除”不通过。 |

### NF01–NF10 修法回归

| 前轮编号 | 本轮判定 | 落地情况 |
|---|---|---|
| NF01 固定行为对象 | **核心已修。** | `SteeringKernel` 明确定义固定 S 的保尾核，S 外 q 不变，接受/残差/记账共用该对象。当前对象实际是 q-kernel，floor 是额外近似；见 IF-06/13。 |
| NF02 行为比进入 PPO clip | **公式已修、集成未通。** | `core_algos.py` 在 PPO clip 后乘 `rollout_is_weights`，不再用 W_beh 污染 update ratio；但 IF-01/16 使该字段在正式批次拒绝、丢失或崩溃。 |
| NF03 命题 C 条件缺失 | **部分修复。** | 规范已撤回“二阶偏差”，加入分离式 W 和 `grpo_loo`；默认仍非整轨 IS、q 未与 `π_old` 同对象校验、scale 非 LOO，故条件 iii/iv 仍不满足（IF-02/06/14）。 |
| NF04 η 可测性/R_max 行为态 | **采样层已修，审计未闭合。** | gate API 不接 proposal token，R_max 达到后 turn 尾部转 η=0；残差 EOS 仍破坏停止态，cap-active 逐 token 账为假（IF-03/15）。 |
| NF05 ITT credit 冷启动 | **从第一版移出。** | 当前 PGS 没有在线 credit/剂量学习活路径，固定 B∞；原问题不再适用于本机制实现。 |
| NF06 sham/跨快照 LCB | **从第一版移出。** | sham 和 LCB credit 不在当前采样/训练主链；无须在本 8-commit 实现中修补，但旧机制残留清理仍不彻底（IF-12）。 |
| NF07 D 门/阈值/propensity | **部分修复。** | D 使用同一 operational S，P85 标定面、ε propensity 与 rate alarm 已有；robust-EMA/越界降级未接、逐 token 控制总体和 histogram 坐标仍不完整（IF-03/05/08）。 |
| NF08 五臂 estimand/功效 | **原冲突已移出机制层。** | 当前只提供 calib/base/S1 首批配置，不再把旧 T/U/S 五臂写成一个含混 estimand；种子/功效仍需另行预注册，代码本身不能证明充分。 |
| NF09 成本式/重复 verify/cache | **实现语义部分修复，真机未闭环。** | hook 在每次拒绝后丢 suffix 并重新 draft+teacher verify，成本应按重复次数计算；prefix-cache 可复用和实际墙钟仍只在 P4 清单，且代码没有规范所写的 bare verify。 |
| NF10 c 与 B∞ 解释 | **核心已修。** | 实现用完整密度比界 `2ηc` 扣费，而非只把 c 当“最大抬升”；B∞=6 只提供有限界、不等于方差安全，当前规范未再作 episode 安全外推。 |

## 测试缺口清单

按任务书口径，目标集合有 88 个 collected case（源码为 53 个测试函数，另有参数化展开）。本审阅逐文件检查了这些测试；因审阅中分支被外部推进且工作树持续有并发改动，没有在非 `c2096e3` 工作树上运行 pytest，以下判定来自静态代码/测试审阅，不把后续修复的结果冒充原快照结果。

现有覆盖的强项：纯 kernel 的 S 质量/保尾/密度界、预算与 gate 单元边界；SPS 单点接受/残差恒等及约 20 万次统计等价；hook 正常路径约 3 万次统计；三种 W_beh 展开；`grpo_loo` 的 n=1/n=2/全同/孤例稳定性；D=0、门恒关的 episode×SPS 透明集成；vendored 文件的若干字符串守护。

仍需补齐：

1. **两个 P0 的真实配置测试**：从 `162-pgs-{calib,s1}.yaml` 构造最终命令与 hook，断言 `max_logprobs >= logprobs/prompt_logprobs`；用真实 Hydra defaults 加一条非空 ledger 走完整 TQ postprocess，证明关闭的 rollout correction 不误判。
2. **TQ mixed schema**：全门关/缺 doc/overflow 与非空 steering 轨迹按两种顺序混合，断言字段恒齐、空账为全 1、非空权重抵达 actor；当前测试只对展开函数本身，不覆盖 `list_of_dict_to_tensordict()`（IF-16）。
3. **verify 行 oracle**：每个位置使用不同 top-K/概率，逐项证明行 r 预测 r+1 及 `base=T+M+E-1`；现有 mock 各行相同，偏移一行也会通过。
4. **真正的多轮拒绝路径**：第一轮拒绝后插入环境 glue，第二轮再次拒绝，核对 prompt、ledger position、turn cap 重置、budget 余额与 W 展开；当前 episode 集成只走 D=0 透明路径。
5. **停止边界**：普通 token 被拒后 residual=EOS 应立即停；draft EOS 被拒且 replacement 非 stop 应继续；覆盖多 EOS/自定义 stop ID。再测恰好 R_max、恰好耗尽预算及下一 token 的 base-only 状态。
6. **异常契约**：首轮/后续轮分别注入 draft aborted、verify aborted、env step exception；已有 response 的 `ENV_ERROR` 必须整轨丢弃。另覆盖空 draft、缺 logprob 明细、行数不齐、后端返回超过 remaining `max_tokens`，确保不产生 phantom ledger/budget。
7. **prompt 精确一致性**：初始观察前后空白、Unicode 归一化、等长不同 token 和聊天模板差异；每次调用都验证完整 student prefix，而非只比较长度/尾部。
8. **top-K 行契约**：rank=None、重复、负/零/越界、空洞、sampled token rank>K、token ID 重复和非有限 logprob；验证失败要定位 request/position，不允许 sentinel 进入核。
9. **floor/封闭点性质**：构造 `p_T(t)<<π(t)<p_floor` 的符号反转例；统计真实模型中 teacher cold-token 排除质量、floor 替代率、`p_floor>π(t)` 率和 D/门决定变化。当前测试只证明近似核自洽，没有证明方向保守。
10. **统计功效**：20 万次纯单点测试对常见概率差异足够，但 3 万次 hook 测试难发现低于约 1% 的稀有 residual/顺序偏差，也没有检验两位置联合分布。应预注册 TV/卡方阈值与功效，覆盖多个质量尺度、稀有拒绝和两位置 joint target，避免单 seed 偶然通过。
11. **LOO 理论性质**：现有测试固化了 n=1/n=2/孤例“数值不爆炸”，却没有做“改变当前 reward 不得改变本样本 scale”的性质测试；理论正确档应先定义外部/frozen scale，再测该独立性。全同组应继续保持零 advantage。
12. **配置/恢复/标定**：逐字段验证 fingerprint 改变，doc 内容同路径替换、protected passthrough、`.nan/.inf`、topk/rmax/rate bounds；标定需测混合 c 坐标、目标分位落 overflow、非法 q 及 `sum(hist)!=d_count`。
13. **根除守护**：守护范围扩到 `src/`、`verl/`、Hydra defaults、生成 YAML 和 vendored tests；完整 collection 不应再 import 已删除的 distillation 模块。

已知 P4 真机项仍包括 prefix-cache×`prompt_logprobs`、`logprobs_mode`、q（draft）对训练端 `π_old` 的数值残差、反复拒绝后的 KV/cache 墙钟与显存。**本轮新增只能真机可靠验证的项目**是：所有声明支持的 vLLM 版本是否都返回完整/唯一 rank 1..K；真实 EOS/stop 在 response token IDs 中的表现；Ray TransferQueue 混合 PGS schema；server cancel/abort 后 sticky route 与 KV 状态能否安全重用。

## 四个实现口径定案的独立判定

1. **S 的两个封闭点——条件通过。** S 是机制可自选的 prefix-measurable 固定集：π 侧只用已观察窗口、teacher top-K 中没有已知 q 值的冷 token 不入 S，不会破坏相对这个 operational `b_S` 的 SPS 恒等、保尾或预算界。代价不是“数值近似”而是**目标覆盖改变**：最强 teacher 建议若在 q 中很冷会被系统性忽略，不能再把 S 描述成字面 `topK(q)∪topK(p_T)`。正式实验应记录 teacher top-K 被排除数、q 质量上界与最大可提升质量；这些量足够小且预注册时，该裁定是 sound 的工程封闭。
2. **p_T 观测窗外 floor 代入——当前论证不通过。** 它确实定义了一个可归一化、可精确采样且仍受 D∞ 预算约束的近似核；若真实与 proxy 都落在同一负侧 clip 饱和区，结果无误差。但 floor 是未观察概率的上界，不是保守点估计，未饱和时可从 `-c` 翻到正值并逆 teacher 方向（IF-13）。在真实率未量化前不能称“方向保守”；推荐取得精确值、用中性增量 0，或使用区间鲁棒规则。
3. **`scale=max(std_loo,std_full)`——理论主档不通过，工程消融可保留。** 它解决 7胜1负和 n=2 的爆炸，但 `std_full` 包含本样本 outcome，违反命题 C 条件 iv；baseline LOO 不能单独恢复无偏性（IF-02）。主档应使用与本样本无关的 frozen/跨组尺度或 scale≡1；现实现只能标为“有偏稳定化消融”，并与 n=1 原始 score、全同组零 advantage 的边界分别说明。
4. **借原生 `rollout_is_weights` 通道——架构选择通过，`c2096e3` 落地不通过。** vanilla loss 对该字段的消费位置正好在 PPO clip 之外，复用通道比新增第二套 loss plumbing 更小、更清晰；与原生 rollout correction 互斥也是正确约束。落地必须判断 `rollout_is/rollout_rs` 是否真正启用，而不是配置映射是否存在，并给每条 PGS 行补全同 schema 的 dense 权重（IF-01/16）；同时钉死 vanilla loss、禁止 protected passthrough 覆盖。满足这些条件后，该定案是四项中最 sound 的工程选择。
