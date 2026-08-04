# PGS 完整实现深度代码审阅

## 总体结论

**实施质量判断：数学核心与主要模块边界总体扎实，但固定快照 `c2096e3` 仍不具备进入 P4 真机冒烟或正式训练的条件。** 两个默认路径上的确定性阻断分别使 vLLM 首次 SPS 请求超出启动上限（IF-05）和首条非空 steering ledger 在 TransferQueue 写入前被错误拒绝（IF-06）；此外，同通道 verify、异常轨迹出清、停止语义、混合批权重 schema、严格 LOO 条件和 fully-async 原生活路径仍有 P1 问题。

- 最终发现计数：**P0 2 项 / P1 9 项 / P2 5 项**，共 16 项。
- P0 gate：修复 IF-05、IF-06，并补真实默认配置下的启动契约测试后，才值得进入 P4。
- P1 gate：在正式训练前解决 IF-01、IF-02、IF-03、IF-07、IF-08、IF-09、IF-10、IF-11；若仓库仍承诺 vendored verl 的 fully-async 能力，还必须修复 IF-12。
- 机制红线判定：当前活跃 PGS actor loss 中没有教师、蒸馏或 mixed-objective 项，`W_beh` 的理论乘法位置也在 PPO clip 之外；该红线通过。但这不抵消默认执行链尚不可用以及若干规范命题证据条件未满足的问题。
- 验证边界：本轮为固定 git 对象的静态审阅，没有把工作树中的并发修改当作证据，也没有执行 pytest；“88 个测试”是函数静态计数，不是本轮通过数。

## 审阅范围与方法

- 工作树：`/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/trace_opd-pgs`
- 分支：`pgs-sps`
- 审阅快照：`c9d2466..c2096e3`（8 个提交）
- 权威规范：`docs/pgs-final-spec.md`（v3.0-final）
- 实现定案：`docs/pgs-impl-design.md`
- 取证口径：仅审阅用户指定的已提交对象 `c2096e3`；审阅期间分支前移及工作树原有/并发未提交改动均不纳入结论。
- 执行约束：只读静态审阅；除本报告外未修改仓库文件，未启动训练、GPU、网络测试或长时间任务。

## Findings：忠实性

### IF-01 — P1 — 逐 token ledger 未实现规范 §4 的最小完备字段，且 `cap_active` 永远记为假

- 位置：`src/trace_opd/steering/ledger.py:16-28,62-86`；`src/trace_opd/steering/hook.py:185-190,201-227,240-256`
- 问题陈述：规范要求每个生成 token 可追溯 `log_pi/log_b/eta/delta`、完整 gate 状态、预算余额、`cap_active`、`steer_flag` 与拒绝次数。当前 ledger 只保存发生部分 steering 决策的稀疏记录，字段中没有预算余额、显式 `steer_flag`、累计 `reject_count`；普通 gate-closed token、预算/拒绝上限触发后的 token 和非 steering token 都不落账。更严重的是记录构造时把 `cap_active=False` 写死，而拒绝计数及 cap 状态是在该记录写入之后更新，因此输出的 `steer_cap_active` 实际上恒为 0。
- 失败场景：第 4 次拒绝触发 `R_max=4` 时，该位置先以 `cap_active=false` 落账，随后才进入 cap；剩余 token 直接按 π 续采且不再落账。离线分析既无法定位行为态切换点，也无法区分“gate 正常关闭”“预算耗尽”“拒绝上限触发”；ε 探针的低 D 对照样本也因普通 gate-closed 决策被省略而不完整。训练权重本身未必立刻错误，但规范中的可审计性、因果诊断与命题 D 的运行时证据链不成立。
- 修复建议：让每个响应 token 都产生一条结构固定的 ledger 记录（纯 π token 使用 `log_b=log_pi, eta=delta=0`）；至少补齐 `budget_remaining_before/after`、`steer_flag`、累计拒绝次数与 gate 原始/翻转/propensity 字段。cap 触发位置应明确记录状态转移，并对随后纯 π 区间逐 token 记录 `cap_active=true`。若为体积采用稀疏编码，应显式编码默认区间及行为态切换点，并由唯一、经过测试的展开器还原规范中的稠密语义。

### IF-02 — P1 — `grpo_loo` 的 `max(std_loo,std_full)` 使用当前样本 outcome，不能支持规范命题 C 的“基线与尺度均 LOO”无偏性主张

- 位置：`verl/verl/trainer/ppo/core_algos.py:639-697`；`docs/pgs-impl-design.md:102-104`
- 问题陈述：LOO baseline 确实排除了当前样本，但分母取 `max(std_loo,std_full)`；其中 `std_full` 包含当前样本 reward，因此归一化倍率仍依赖当前 action/outcome。实现注释把 scale 称为只影响幅度、不会影响无偏性，这在倍率随当前样本变化时不成立，也不忠实于规范 §2.6/命题 C 的条件。
- 失败场景：同一 prompt 的二元 reward 组中只有一个正例时，孤例自身会改变 `std_full`，并可能触发 max 分支；其梯度倍率因自身 outcome 而变化。即使 LOO baseline 条件均值为零，乘上该 outcome-dependent 随机倍率后，规范所声称的条件无偏关系不再由现有推导保证。
- 修复建议：若必须保留防爆下限，使用与当前 action/outcome 独立的冻结或交叉拟合尺度（例如上一批/全局 EMA floor），或不做样本内尺度归一化；同时增加数值与统计测试。若团队接受该稳定性启发式，则需在 spec/设计文档中明确降级为“有意引入偏差的工程折中”，删除命题 C 已被严格满足的表述。

### IF-03 — P1 — SPS 没有执行规范要求的 bare/teacher 同通道双 verify，π 来自 draft response logprobs

- 位置：`src/trace_opd/steering/hook.py:153-176,192-212,273-310`
- 问题陈述：规范 §2.5 与实现设计均要求对同一固定序列做 bare 与 teacher 两次 scoring，并从同一 verify 数据通道得到 π、p_T 及接受比。当前实现只做 teacher prompt-logprobs verify；π 的 top-K 与已采 token logprob直接取 draft generate response-logprobs。这样 sampler 可能仍能相对实际 proposal q 正确，但它不是文档所规定、也不是代码注释声称的“同 verify 通道”实现，且没有运行时证据证明 response 与 prompt 两条 vLLM logprobs 路径的语义/数值等价。
- 失败场景：若 vLLM 的 response-logprobs 与 prompt-logprobs 在 `logprobs_mode`、处理前/处理后 logits、数值精度或插件路径上不同，则构造候选核时的 π 与 p_T 不在同一口径，`b/π` 也无法直接套用规范证明；P4 所列的数值残差检查当前无数据可做。
- 修复建议：按 spec 增加 bare verify，并明确区分“实际 draft proposal q”与“策略 π”：验收比必须相对真实 q，同时用同通道 π/p_T 构核；若坚持复用 draft 行，应先修订权威规范，证明该 API 返回的就是真实 q，并在启动时/抽样时 fail-loud 比较 bare prompt 与 response 行的 top-K、采样 token logprob 残差。完成真机验证前，不应把同通道精确性当成已落实事实。

### IF-04 — P2 — 规范要求的跨 step 鲁棒阈值更新和异常降级没有接入训练生命周期

- 位置：`src/trace_opd/steering/gate.py:73-95`；`src/trace_opd/train/agent_loop.py:198-201`
- 问题陈述：`robust_quantile_update()` 存在但没有调用点；gate rate 越界只在 episode 完成后记 warning，不会切换行为态或对下一 step 的冻结阈值做 EMA 更新。因而实现实际是“整次运行固定 θ_D + 仅告警”，与规范 §2.3 的 frozen-within-step、step 间鲁棒更新及越界降级不一致。
- 失败场景：模型分布漂移后 D 几乎全高于阈值，gate rate 长期接近 100%。当前系统只打印告警，仍持续 steering，直到预算/R_max 各自限制；无法获得规范承诺的自动降级行为，也无法让阈值适应新分布。
- 修复建议：在明确的 optimizer-step 边界聚合 D，计算 winsorized/quantile 目标并只更新下一 step 使用的冻结阈值；将越界连续次数和降级状态纳入共享、可持久化的控制状态。若产品决定 θ_D 全程固定且告警不降级，应同步修订 spec 和设计文档，避免“已实现自适应/降级”的错误声明。

### 本维度判定

保尾核、截断增量、SPS 接受/残差公式、η 的接口可测性、`delta=2*eta*c` 与预算缩放、R_max 计数及 native `rollout_is_weights` 在 PPO clip 外相乘的主干实现基本可对应规范；但 verify 通道、可审计 ledger、LOO scale 与 gate 生命周期尚不能支持“v3.0-final 全条款忠实落地”的结论。NF01、NF02、NF04、NF07、NF10 的核心修法可见；NF03 仅 baseline 部分严格落实，NF09 的同通道数值验证仍停留在 P4 假设。

## Findings：正确性（对抗性反例）

### IF-05 — P0 — launcher 将 vLLM `max_logprobs` 配成 64，但 hook 的真实请求宽度是 256

- 位置：`src/trace_opd/steering/hook.py:43-45,124-134,155-160,289-298`；`scripts/launch_train.py:73-81`；`tests/test_launcher_command.py:58-70`
- 问题陈述：hook 的默认观测窗是 `OBS_TOPK_FACTOR * topk = 4 * 64 = 256`，draft `logprobs` 和 teacher `prompt_logprobs` 均请求该宽度；launcher 却只设置 `max_logprobs=max(steering.topk,20)=64`。设计文档 P4 清单明确要求 256，而测试又把 64 固化为期望值。
- 失败场景：直接使用 `162-pgs-calib.yaml`，或为 `162-pgs-s1.yaml` 填入已标定 θ_D 后启动。第一次 SPS draft 即发出 `logprobs=256`，超过 vLLM 启动上限 64，SamplingParams/服务端校验拒绝请求；校准臂和正式 PGS 都无法产生首个 token。
- 修复建议：把 `obs_topk` 设为 `SteeringCfg` 的显式单一真源，校验 `obs_topk >= topk > 0`；hook、launcher、fingerprint 共用它，launcher 配置 `max_logprobs >= obs_topk`。契约测试必须同时实例化 hook 并核对其请求宽度，不能只测试命令字符串中的 `topk`。

### IF-06 — P0 — 默认“关闭”的 verl rollout correction 映射被误判为已启用，首条非空 steering ledger 无法入队

- 位置：`verl/verl/trainer/ppo/v1/agent_loop_tq.py:184-206`；`verl/verl/trainer/config/ppo_trainer.yaml:39-43`；`verl/verl/trainer/config/algorithm/rollout_correction.yaml:5-20`
- 问题陈述：互斥守卫判断 `algorithm.rollout_correction is not None`。Hydra 默认配置始终组合出该映射；语义上的关闭状态是映射内 `rollout_is: null`、`rollout_rs: null`，并不是整个对象为 `None`。因此 PGS 与原生 correction 实际并未同时启用，也会被判为冲突。
- 失败场景：修复 IF-05 后运行正式 S1；任一轨迹首次产生非空 `steer_positions`，TQ put 前就抛出 “both write rollout_is_weights”。校准臂因 θ_D 极大、ledger 为空可能通过，导致阻断恰好延迟到正式实验才暴露。
- 修复建议：抽取与 verl 实际写权重条件一致的 `rollout_correction_enabled()`，只在 `rollout_is`、`rollout_rs` 或 bypass 的真实冲突模式开启时拒绝；launcher 同时显式钉死 PGS 档的原生 correction。增加“真实默认 Hydra 映射 + 非空 ledger 可入队”和“真实启用原生 correction 必须拒绝”两类测试。

### IF-07 — P1 — 中途 abort/异常且已有历史响应时，残缺轨迹会以 reward 0 进入 GRPO

- 位置：`src/trace_opd/core/episode.py:174-191,270-310`；`src/trace_opd/train/agent_loop.py:175-208,319-361`；`src/trace_opd/steering/hook.py:155-168,289-301`
- 问题陈述：episode 正确把 engine abort/verify 异常标为 `OUTCOME_ENV_ERROR`，但 `_student_rollout()` 只有在 `response_ids` 为空时才重试或抛错；只要之前已有一轮响应，错误结果就直接返回。`run()` 随后对所有非 win 轨迹统一赋 reward 0，把基础设施失败冒充为 verifiable environment outcome。
- 失败场景：第 1 轮交互正常，第 2 轮 teacher verify 被 weight-sync 中止。episode 保留第 1 轮 response，故不满足“响应为空”的重试条件；残缺 trajectory 以失败奖励进入同一 GRPO group。若 abort 发生在当前 hook 已执行若干位置之后，ledger 还可能含没有提交到最终 response 的尝试 token。
- 修复建议：对任何 `OUTCOME_ENV_ERROR` 整 episode 作废，并用新 route、新环境和新 `TrajectorySteerState` 重试；第二次仍失败应显式 drop/raise，绝不能赋 0。ledger 应按 turn 事务提交，或在 episode 作废时整体丢弃。

### IF-08 — P1 — residual replacement 若采到 EOS/stop，hook 仍继续 draft 并返回 stop 后 token

- 位置：`src/trace_opd/steering/hook.py:169-174,251-262`
- 问题陈述：自然停止只由原 draft 段是否短于 `remaining` 推断；任何拒绝都无条件进入下一轮 draft。残差重采替换了原 token 后，hook 没有重新执行 EOS、token stop 或文本 stop-sequence 判定。
- 失败场景：普通 draft token 被拒，`norm(max(0,b-pi))` 恰好采到 EOS。代码先把 EOS 加入 `executed`，再以该前缀请求更多 token，最终输出包含 EOS 后的内容；这不再是带同一停止规则的 b_S 自回归轨迹，也可能改变 episode action parser 的结果。反向情形“原 draft EOS 被拒并替换成非 EOS”则确实应续采，不能简单沿用原 stop reason。
- 修复建议：把 engine 的全部停止口径（EOS IDs、token stops、文本 stops）显式传入 hook；每个 replacement 后运行同一增量 stop matcher。分别测试 replacement=EOS、replacement 完成多 token stop，以及原 EOS 被拒后替换为非 EOS。

### IF-09 — P1 — vLLM top-K 行用合法 token/probability 作占位，rank 空洞会伪造 token 0、概率 1

- 位置：`verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py:677-700`；`verl/verl/workers/rollout/vllm_rollout/utils.py:645-675`
- 问题陈述：response extractor 先用 `[0]*K` 与 `[0.0]*K` 填充，再按 rank 覆盖；没有校验 rank 非空、在 1..K 内、唯一且完整覆盖。缺项/重复项会留下“token id 0、logprob 0”的伪数据。prompt extractor 虽用 `None`，但只靠可被 `python -O` 移除的长度 assert，也没有验证 rank 集合；`rank=None` 会在比较时异常。sampled token 的 rank>K 时跳过、另用自身 logprob 的逻辑本身正确。
- 失败场景：目标 vLLM 版本或插件返回 top-K 加 sampled token，但某一行 rank 缺洞/重复。伪 token 0 以概率 1 进入 π 观测窗，使 π(S)>1 后崩溃，或在未及时触发不变量时构造错误核；错误发生在 rollout 中段且缺少请求/位置上下文。
- 修复建议：prompt/response 共用一个 extractor：先收集 `(rank,token,logprob)`，校验 finite、token 唯一且 rank 集严格等于 `1..K`，不完整就显式 fail loud；不要用任何合法 token 或概率作 sentinel。补 rank hole、duplicate、None、非有限值及 sampled-rank>K 测试。

### IF-10 — P1 — `rollout_is_weights` 只在 ledger 非空时存在，PGS 混合批没有统一字段 schema

- 位置：`src/trace_opd/train/agent_loop.py:170-201`；`verl/verl/trainer/ppo/v1/agent_loop_tq.py:179-206`；`verl/verl/utils/tensordict_utils.py:918-948`
- 问题陈述：只有 `steer_positions` 非空才创建 dense 权重字段；gate 全关、doc missing/overflow、预算为零及 baseline 轨迹均不携带该字段。同一个 GRPO group 可同时含有字段和无字段的 row，而 TensorDict/训练 loss 把一个 key 视为整批存在或整批缺失；代码没有定义缺失 row 应填全 1 的契约。本地 `list_of_dict_to_tensordict` 还只取第一条记录的 key 集合，外部 TransferQueue 对跨 key 缺列的最终行为未被仓库证明。
- 失败场景：同一 prompt 的 8 条轨迹中，一条开门产生 W_beh，另一条门全关。批取数时可能 stack 失败，或整批不带 `rollout_is_weights`，使已倾斜轨迹按权重 1 训练；两种结果都破坏分离式校正。
- 修复建议：只要轨迹属于 PGS 协议，就总是写与 response 对齐的 dense 权重；空 ledger 明确写全 1。把“字段存在”与“实际发生 steering”分开，并在真实 TransferQueue 上做 empty/non-empty/doc-missing 混合批到 actor loss 的端到端测试。这是本次新增的真机确认项。

### IF-11 — P1 — resume fingerprint 漏掉 advantage/objective 参数，且在计算后可被 verl passthrough 覆盖

- 位置：`src/trace_opd/common/config.py:323-335`；`scripts/launch_train.py:179,213-241,297-310`
- 问题陈述：fingerprint 自称描述“优化什么”，但不含 `train.adv_estimator`，也不含严格单步条件、观测窗/logprobs 口径及原生 rollout correction；doc table 只哈希路径，不哈希内容。更直接的是未知 verl CLI 参数在 fingerprint 生成后原样附加，能够覆盖 `algorithm.adv_estimator` 等目标参数而不改变 guard。
- 失败场景：同一 run 先按 `grpo_loo` 训练，恢复时追加 `algorithm.adv_estimator=grpo`；命令的最终 objective 已变化，但 fingerprint 相同，旧 checkpoint 被允许继续。原路径下替换 doc table 内容也同样绕过守卫。
- 修复建议：对最终解析后的 canonical objective 配置做哈希，至少覆盖 estimator/scale 口径、epochs、mini-batch、loss aggregation、全部 steering/观测参数和 correction；禁止或解析会改 objective 的 passthrough，再计算 fingerprint。doc table 使用内容 digest 或不可变版本 ID。

### 本维度反例核查结论

- `extract_prompt_logprobs` 的行 `r` 预测 token `r+1`；hook 的 `base=len(teacher_seq)-len(gen_ids)-1` 在当前序列拼接下是正确的。
- 多轮的 `position=stream_len+len(executed)` 将结构 token 也计入 response 坐标，与 verl 的 `response_mask` 坐标一致；拒绝后 replacement 入前缀、旧 draft 后缀丢弃的主路径正确。
- 残差公式、S 外接受率 1、预算不足时缩 η 到恰好余额，以及第 R_max 次拒绝后下一位置强制 π 的行为主路径正确；问题在停止语义和 ledger 可见性。
- vendored vLLM 会把请求 `max_tokens` clamp 到上下文允许范围，正常 backend 不应 over-return；但 hook 自身仅在返回时切 `executed[:max_tokens]`，没有对畸形 backend 的超额 rows/ledger fail-loud，仍应补防御性测试。
- 空 draft、部分 abort、verify 异常和 top-K 畸形目前没有完整契约测试；其中部分 abort 已构成 IF-07，其余列入测试缺口。

## Findings：根除完整性

### IF-12 — P1 — P1.5 连带删除 fully-async rollouter 的非蒸馏方法，vendored verl 原生活路径确定性损坏

- 位置：`verl/verl/experimental/fully_async_policy/fully_async_rollouter.py:735-746,803-807`；`verl/verl/experimental/separation/ray_trainer.py:136-143`；删除发生于提交 `033c44d`
- 问题陈述：清除 `_create_teacher_model_manager()` 时，同一相邻代码块中的三个非蒸馏方法也被删除：fully-async 专用的 `_create_actor_rollout_classes()` no-op、`_create_reward_model_class()` no-op，以及 `_create_continuous_iterator()`。当前 `init_workers()` 仍调用继承的 `_create_worker_classes()`；它会分派到基类 `_create_actor_rollout_classes()` 并直接抛 `NotImplementedError`。即使绕过初始化，`_feed_samples()` 仍调用已不存在的 `_create_continuous_iterator()`。
- 失败场景：任何使用 vendored `FullyAsyncRollouter.init_workers()` 的原生 fully-async 训练，在创建 worker class 阶段立即 `NotImplementedError`；若外部定制恰好绕过该阶段，开始喂样本时又以 `AttributeError: ... _create_continuous_iterator` 失败。该问题与 distillation 是否启用无关，是旧机制出清对活路径的确定性误伤。
- 修复建议：恢复这三个方法的非 teacher 版本：前两个保持 fully-async 所需的 no-op，iterator 恢复按 epoch 遍历 dataloader；只删除 teacher manager 初始化及 client 传参。增加一个不启 GPU 的结构/最小对象测试，至少证明 fully-async subclass 覆盖所有基类抽象 hook，且 `_feed_samples` 引用的方法存在。

### IF-13 — P2 — “旧机制全面出清”仍留下默认配置、engine 半死分支与 trace_opd 死代码/入口

- 位置：`verl/verl/trainer/config/ppo_trainer.yaml:39-43`；`verl/verl/trainer/config/distillation/distillation.yaml:1-113`；`verl/verl/trainer/config/_generated_ppo_trainer.yaml:815-818`（其余 generated 变体同类）；`verl/verl/workers/engine/fsdp/transformer_impl.py:654-704,1250-1467`；`verl/verl/workers/engine/megatron/transformer_impl.py:538,672,909-1066`；`verl/verl/workers/engine/veomni/transformer_impl.py:807-830`；`verl/verl/trainer/constants_ppo.py:121-144`；`src/trace_opd/core/episode.py:323-361`；`src/trace_opd/core/types.py:184-213`；`README.md:1-45`；`pyproject.toml:3-17`
- 问题陈述：默认 Hydra 仍组合 `distillation` 节点，其 `_target_` 指向已经删除的 `DistillationConfig/DistillationLossConfig`；generated configs、OPD examples/tests/docs也仍暴露这套失效 API。三种 engine 继续读取 `distillation_use_topk/distillation_only/teacher_*`，FSDP 还保留 `research_mask/tg_mask` 统计；环境变量转发仍有 distillation microscope/observation pack。trace_opd 自身残留无人引用的 mixed-objective `replay_bare_tokens()`、`FactCard`，包说明、README、pyproject 仍宣称 researcher/self-distillation 并指向已删模块/config。现有根除测试只扫描有限执行文件，因而对这些位置全盲。
- 失败场景：按 vendored verl 仍保留的 OPD 示例或对 `config.distillation` 调用标准 dataclass instantiate，会解析到不存在的 target；外部调用者若写入残留 engine flags，则进入没有完整 producer/loss/config 类的半死分支。仓库首页的 quick-start 还会调用已删除的 `configs/trace.yaml` 和 researcher 包。PGS 默认 loss 当前仍是纯 PPO/GRPO，不会因此静默复活教师项，但“干净代码基”和通用 vendored 功能都不真实。
- 修复建议：明确二选一边界。若目标是 PGS 专用 vendor，删除 distillation defaults/generated 配置、示例/失效测试、engine 专用分支、旧环境变量和 trace 死代码/陈旧入口；若要保留上游 OPD 能力，则完整恢复为隔离、非默认、可测试的上游模块，且不得与 PGS 字段共享或进入 PGS launcher。根除守护应改为覆盖 `src/`、`configs/`、`verl/verl/trainer/config` 和三种 engine 的允许名单式扫描，而不是少数文件字符串断言。

### 本维度判定

活跃 PGS launcher 没有 distillation key，`ActorRolloutRefWorker` 的 actor loss 也恒为 plain `ppo_loss`；教师没有重新进入训练损失，机制红线本身通过。P1.5 对主 V1/V0 PPO 的 teacher manager、mixed sample-role、padding 与 distillation loss 接线删除基本成套，但 fully-async 的连带误删（IF-12）和广泛半死表面（IF-13）使“全面出清且未误伤原生功能”的提交声明不成立。

## Findings：可读性与结构

### IF-14 — P2 — D 直方图 schema 将上界硬编码为默认 `c=2.5`，标定脚本仍对不可辨识分位数给出伪精确建议

- 位置：`src/trace_opd/steering/ledger.py:31-55,83-85`；`scripts/calibrate_theta_d.py:24-26,61-81`
- 问题陈述：D 的理论范围为 `[0,c²]`，但 ledger 全局固定 `D_HIST_MAX=6.25`，没有随 run 落盘 `c`、range 或 schema version。`c>2.5` 时所有高值压入末桶；即便 `c=2.5`，末桶也同时包含正常的 `[6.152...,6.25]`，脚本却把它统称为 `>6.25` overflow。若目标 P85 落入压缩质量，脚本仍线性插值并打印一个四位小数的 `suggested theta_d`。
- 失败场景：配置 `c=3`，20% 的 D 位于 6.25 到 9 之间。真实 P85 可落在该区间任意位置，聚合脚本仍给出约 6.2 的建议；正式 run 开门率会显著偏高。聚合不同 c/schema 的 rollout 目录时也没有 fail-loud。
- 修复建议：把 `hist_min/max/bins/c/schema_version` 与每条账一起落盘，上界至少取本 run 的 `c²`；更稳妥的是使用可合并 quantile sketch。聚合时拒绝混合 schema；若 overflow 的累计质量覆盖请求分位数，必须拒绝输出建议值，只报告下界和需要重跑。

### IF-15 — P2 — `SteeringCfg` 的类型入口集中但不变量校验不完整，错误被推迟到 rollout 中段

- 位置：`src/trace_opd/common/config.py:173-195,221-240`；`src/trace_opd/steering/gate.py:32-47,74-83`；`src/trace_opd/steering/budget.py:19-24,49-55`
- 问题陈述：科学计数法字符串通过数值强转已修复，但启动校验仍缺 `topk>0`、`r_max>=1`、`epsilon`、`gate_rate_bounds` 的长度/类型/次序/范围，以及 `c/b_max/theta_d` finite；也未在 steering 启用时提前锁定 rollout 温度/截断测度。部分错误直到 per-episode 构造甚至第 50 次 gate alarm 才暴露。
- 失败场景：`topk: 0` 能过 launcher，首位置构造空 S 后失败；`c: .nan`、`b_max: .inf` 可绕过大小比较并产生无意义预算；三元素或字符串 `gate_rate_bounds` 直到 `lo,hi=...`/float 比较时才异常。错误会表现成 rollout failure，而不是清楚的配置拒绝。
- 修复建议：把所有机制不变量放在唯一配置入口，使用 `math.isfinite`，显式要求 `topk>=1`、`r_max>=1`、`0<=epsilon<0.5`、`0<=lo<hi<=1`，并校验 steering 下 `temperature=1/top_p=1/top_k=-1`。为 YAML 字符串、NaN/inf、零值和非法数组做参数化测试。

### IF-16 — P2 — 关键机制契约依赖可移除的 `assert`，多处 docstring 与实际数据流相反

- 位置：`src/trace_opd/train/agent_loop.py:125-155,280-292`；`src/trace_opd/core/episode.py:100-102,143-162`；`src/trace_opd/steering/hook.py:13,147-148`；`src/trace_opd/steering/kernel.py:39-55,91-94`；`src/trace_opd/steering/sps.py:22-52`；`src/trace_opd/steering/ledger.py:1-6,62-67`；`verl/verl/workers/rollout/vllm_rollout/utils.py:661-669`
- 问题陈述：teacher/student 生成尾部一致、prompt 长度、核质量/密度比和 prompt top-K 结构使用 `assert`，在 `python -O` 下会消失。`SteeringKernel` 声称构造后不可变但 dataclass/NumPy 数组均可修改；kernel/sps 声称 π 来自 verify，实际来自 draft；ledger 顶部声称未 steer 位置逐位置齐全，`finalize()` 又明确是稀疏；hook/budget 声称 cap 尾段如实记账，实际不是；episode 暗示训练会重试 ENV_ERROR，实际只重试空响应。`vtrace` 名称也高估了实际的单点 ratio clipping。
- 失败场景：生产镜像开启 `PYTHONOPTIMIZE=1`，同时模板或 top-K 数据漂移，机制红线不再 fail-loud。另一个更具体的坐标问题是：episode 只要求 live `initial_obs.strip()==dataset initial_obs.strip()`，但 hook 的 `student_prompt_len` 用 dataset 字符串预先渲染；若只差首尾空白而 token 长度变化，首轮 `stream_len` 会把 prompt 尾部误当 response（或变负），teacher 行与 ledger position 随即错位。
- 修复建议：所有数据/机制边界改为显式 `ValueError/RuntimeError`。要么要求 initial_obs 字节级一致，要么在环境启动、拿到 live prompt 后再构造 teacher prefix 与 SpsGenerate，确保只存在一份 prompt token 真源。kernel 应复制输入并设为只读（仅 `frozen=True` 不足以冻结 ndarray）；逐项校正 docstring 到真实 sparse ledger、draft q 和异常语义，`vtrace` 重命名为 `clipped_per_token` 或实现完整算法。

### 本维度判定

`kernel/sps/gate/budget/ledger` 的职责拆分、纯 NumPy 数学面与 episode GenerateFn 包装边界总体清楚，`rollout_is_weights` 也集中在一个展开模块和一个写入点；问题主要集中在跨模块契约没有单一真源，以及大量历史/证明性注释比代码走得更远。对这类机制实现，错误承诺会直接扩大下一轮修改的风险，应与功能 bug 同批清理。

## 忠实性核对表

| 规范条款 | 主要代码位置 | 判定 | 核对摘要 |
|---|---|---|---|
| §2.1 保尾核 | `steering/kernel.py:28-141`；`steering/hook.py:59-95` | **条件通过** | S 内几何倾斜、S 外恒 π、质量守恒和密度比界实现正确；实际 S 是有限观测 surrogate，且 π 行来自 draft response 通道（IF-03）。 |
| §2.2 截断增量 | `steering/kernel.py:28-36`；`steering/hook.py:74-95` | **条件通过** | `clip(log p_T-log π,±c)` 正确；teacher 窗外 floor 是保守 surrogate，不是不可观测原 p_T 核的精确实现。 |
| §2.3 D 门、ε 探针与可测性 | `steering/gate.py:24-95`；`steering/hook.py:185-212` | **部分通过** | η 的接口输入不含当前 proposal token，D 和随机探针先于采样决定；但跨 step 鲁棒更新/异常降级未接线，普通关闭决策也未完整落账（IF-01、IF-04）。 |
| §2.4 D∞ 预算 | `steering/budget.py:17-86`；`steering/hook.py:214-227` | **数学通过、审计不通过** | `δ=2ηc`、余额不足时缩 η、累计上界主路径正确；ledger 缺余额 before/after，不能从产物重建完整轨迹证据（IF-01）。 |
| §2.5 SPS 精确采样 | `steering/sps.py:18-82`；`steering/hook.py:153-310` | **部分不通过** | 单点接受 `min(1,b/q)`、残差 `norm(max(0,b-q))` 和拒绝断点续采正确；未执行规范的 bare/teacher 同通道 verify，默认请求宽度不可运行，replacement stop 语义错误（IF-03、IF-05、IF-08）。 |
| §2.5 R_max 行为态 | `steering/budget.py:89-109`；`steering/hook.py:240-273` | **行为通过、记账不通过** | 第 R_max 次拒绝后后续位置转纯 π 的主路径正确；cap 转移及纯 π 尾段未被如实记录（IF-01）。 |
| §2.6 分离式 W_beh | `steering_weights.py:42-93`；`core_algos.py:1390-1423`；`agent_loop_tq.py:179-206` | **公式通过、执行链不通过** | W 在 PPO clip 形成后相乘，per-token/per-decision 展开基本正确；默认互斥误判和混合批缺字段阻断/破坏实际链路（IF-06、IF-10）。 |
| §2.6 LOO advantage | `core_algos.py:639-697` | **不通过** | baseline 排除了自身，但 `std_full` 含当前 outcome，严格 LOO scale 条件及命题 C 推导不成立（IF-02）。 |
| §4 逐 token 记账 | `steering/ledger.py:16-86`；`agent_loop.py:188-201` | **不通过** | 当前是部分 steering 决策的稀疏账，缺规范要求的完整 gate/预算/cap/reject/纯 π token 语义（IF-01）；D histogram 还固定默认 c（IF-14）。 |
| 命题 A：SPS 精确服从 b_S | `sps.py:18-82`；`hook.py:153-310` | **核级通过、系统级有条件** | 对同一固定 q/b 行，接受—残差恒等正确；系统声明仍依赖 q≡π、同对象 scoring、完整 top-K 与正确 stop 语义（IF-03、IF-08、IF-09）。 |
| 命题 B：逐点比值界与轨迹 D∞ 界 | `kernel.py:98-141`；`budget.py:49-86`；`hook.py:214-227` | **数学通过、运行证据不足** | 对 surrogate kernel 的 `|log(b/π)|≤2ηc` 和保守扣费推导成立；现有账无法独立复核每个位置及状态转移（IF-01）。 |
| 命题 C：分离式校正无偏性条件 | `core_algos.py:639-697,1390-1423`；`steering_weights.py` | **不通过** | loss 的乘法位置正确；严格 LOO scale 失败，且默认 W 通路不可用。per-token 模式本就是文档承认的有偏近似，不能代替命题条件。 |
| 命题 D：单点二元 worked example | `kernel.py:28-141`；`docs/pgs-final-spec.md:113` | **数值口径成立、无专门回归** | 通用核公式可复现 `b=0.219… / G=8 时 86.2% / δ=1.48`；实现没有把该组参数做独立回归测试，且代码未把它外推为 episode 保证。 |
| 教师不进入损失红线 | `launch_train.py:213-241`；`core_algos.py:1390-1423` | **通过** | 活跃 PGS 命令和 actor loss 中无教师、蒸馏或 mixed-objective 项；教师只在采样侧出现。 |

### NF01–NF10 修法落地核对

- **NF01：核心落地。** q/π top-K 可观测性约束和固定候选集合已编码；但实际 q/π scoring 对象仍需按 IF-03 澄清。
- **NF02：结构落地、默认链未落通。** W 的乘法位置正确且 teacher 未进入 loss；默认 correction 互斥判定使真实非空链路失败（IF-06）。
- **NF03：只完成 LOO baseline。** per-token 偏差已在设计中承认、per-decision 口径已实现；当前 scale 仍依赖自身 outcome（IF-02）。
- **NF04：行为核心落地、审计未落地。** gate 签名和 R_max 行为态存在，cap/纯 π 尾段 ledger 不真实（IF-01）。
- **NF05/NF06：不再作为 PGS loss 机制。** 旧信用分配/伪奖励路径未进入当前主目标；相关研究残留仍见 IF-13。
- **NF07：基础数据面部分落地。** D、propensity、直方图有实现；assigned/actual/budget-blocked、完整对照账和 EMA/降级仍不完整（IF-01、IF-04、IF-14）。
- **NF08：未发现其旧实验叙事重新进入主机制。** 当前实验矩阵不构成对该既往问题的新增实现保证。
- **NF09：静态 prompt 路由部分落地。** teacher/student 尾部一致性有断言，prefix-cache 与 prompt-logprobs 被列入 P4；同通道 verify 尚未落实（IF-03），prompt 真源还有 IF-16。
- **NF10：核心落地。** `δ=2ηc`、预算缩放和尾段纯 π 行为正确；逐 token 审计证据仍缺 IF-01 所列字段。

## 测试充分性

### 静态盘点与已覆盖面

- 在固定对象 `c2096e3` 上按 `def test_`/`async def test_` 静态计数，共 88 个测试函数；用户允许执行的范围为其中 51 个：`tests/steering/` 35 个，`tests/test_vendored_verl.py` 16 个。其余 37 个覆盖 episode、protocol、launcher、adapter/eval 等基础面。
- 本轮没有执行 pytest：审阅期间分支/测试文件被并发修改，直接运行会验证错误快照；把 `c2096e3` 解包到临时目录又违反“唯一写操作为报告”的约束。以下判断来自逐测试静态审阅，不能把“88 个函数”表述成“本轮 88 tests passed”。
- `test_steering.py` 对保尾质量、全词表 ratio 界、η=0、尾部恒 π、残差恒等、D gate propensity、预算恰好耗尽和 R_max 单体状态机覆盖较扎实；200,000 次单点模拟同时核对输出 TV、拒绝率和解析 log_b。
- `test_sps_hook.py` 有独立解析 oracle 的 30,000 次单 token 统计测试，且显式证明结果不是恒 π；还覆盖 gate closed、基本 budget tail、测度 guard、teacher 序列拼接和拒绝后继续调用 draft。
- W_beh 的三个模式、空 ledger、越界位置和单 token ratio bound 有纯 NumPy 测试；`grpo_loo` 覆盖 baseline 排除自身、孤例有限、全同、singleton 和组隔离；episode×SPS 覆盖两轮结构 token 坐标的 gate-closed 透明路径。
- vendored 守护确认 active loss 没有 distillation 分支、W 位于 vanilla clip 外，并检查若干删除目录/符号；这些属于轻量静态烟雾，不是执行链集成测试。

### 关键缺口清单

1. **启动契约**：没有把 hook 的真实 `obs_topk=4*topk` 与 launcher `max_logprobs` 放入同一测试；`test_launcher_command.py:58-70` 反而把错误的 64 固化为期望值，漏掉 IF-05。
2. **真实配置/TQ 链**：没有用 Hydra 默认 `rollout_correction` 映射测试互斥谓词，亦没有一条 CPU 端到端轨迹让 ledger 经 TQ put/get、worker field selection 到 vanilla loss；因此 IF-06/IF-10 均未被发现。
3. **verify 行对齐**：hook 与 episode mock 的每一行 p_T 完全相同，`base` 偏移任意一行仍会通过；需要每个位置拥有唯一分布/marker 的 oracle，并直接断言 draft 第 g 个 token取到全局 `T+M+E+g-1` 行。
4. **统计等价功效**：现有 hook 统计只测位置无关、单 token 分布和一个 seed；缺可枚举的多 token 条件分布、EOS/stop 后的停止轨迹概率、早位置拒绝后重新构核，以及多 seed/置信区间。30k 阈值能排除恒 π，但对局部小偏差功效有限。
5. **拒绝断点**：`test_rejection_resumes_from_breakpoint()` 的 `A or len>=2` 与“下一 call 只要仍以 student prompt 开头”几乎不能证明旧 suffix 被丢弃；应逐 call 保存 draft proposal、replacement 和期望前缀，精确相等比较。
6. **停止与异常**：缺 replacement=EOS、replacement 完成多 token stop、原 EOS 被拒后换成非 EOS；缺空 draft 的错误分类、已有 executed 时 draft abort、teacher verify abort/异常、以及 engine over-return 后 ledger/response 截断一致性。
7. **多轮真实 steering**：episode 集成固定 `ell=0`、D=0、ledger 为空；缺第二轮发生拒绝、跨轮 budget 耗尽、每轮 R_max reset/cap 尾段和 mask=0 结构坐标的联合测试。
8. **top-K 数据边界**：没有 rank hole、duplicate、None、非有限 logprob、行长不匹配和 sampled token rank>K 测试；后者代码逻辑正确但仍需回归保护。
9. **ledger 审计**：测试手工构造一条 `cap_active=True` 记录，却没有通过 hook 真的触发第 R_max 次拒绝；也未检查预算 before/after、gate assigned/actual/budget-blocked、累计拒绝和纯 π 尾段逐 token 完备性。
10. **W_beh**：缺 mixed empty/non-empty/doc-missing schema、实际 response_mask、跨结构 token 的 per-decision 前缀、尾截后轨迹权重以及非有限 ratio；`vtrace` 测试只验证单点 `min(1,W)`，没有验证完整 V-trace（因为实现也不是）。
11. **LOO 数学性质**：测试通过源码切片 `exec`，没有验证 registry/多轨迹调用链；n=2 只测关闭 std 的情形。现测试明确断言 `max(std_loo,std_full)`，是在固化 IF-02，而非验证命题 C 无偏性。需 n=2 norm-on 与小 bandit Monte Carlo 梯度期望测试。
12. **异常 outcome 红线**：episode abort 测试只覆盖首轮空响应；缺“已有一轮成功响应后 abort/verify exception 不得赋 0 入训”的 agent-loop 测试（IF-07）。
13. **config/resume/calibration**：缺 NaN/inf/非法 bounds/topk/rmax，缺只改变 adv_estimator、doc 内容或 passthrough 后 resume 必须拒绝，缺 P85 落在 overflow 时拒绝建议、混合 histogram schema fail-loud。
14. **根除与误伤**：vendored 字符串 allowlist 没扫默认 config/generated config、三种 engine 和 trace 死代码；没有 fully-async subclass contract 测试，故 IF-12/IF-13 均通过现有守护。
15. **prompt 一致性**：缺 live initial_obs 与 dataset 只差首尾空白但 tokenization/长度不同的用例，也没有在 `python -O` 语义下证明红线仍显式报错（IF-16）。

### 真机边界

- 设计文档已知的 prefix cache × `prompt_logprobs`、真实 `logprobs_mode`、draft/bare-verify 数值残差及显存/墙钟仍应按 P4 执行。
- 本轮新增需要真实集成环境最终确认的项目：TransferQueue 对跨 key 缺列的精确语义及 `rollout_is_weights` 是否整批送达（IF-10）；目标 vLLM 版本的 top-K rank 完整性/非 top-K sampled token 形态；engine 是否暴露足够 stop metadata 让 residual replacement 复用完全相同的停止判定。前两项即使可用 mock 做本地防御测试，也不能替代目标版本的契约冒烟。

## 对四个实现口径定案的独立判定

### ① S 的两个封闭点：有条件 sound

- 代码位置：`src/trace_opd/steering/hook.py:59-95`；设计位置：`docs/pgs-impl-design.md:107-113`
- 判定：π 候选只取固定 `obs_topk` 行、p_T top-K token 只有在 π 行中可观测才入 S，这些选择都由当前位置前缀及两条分布行决定，不依赖当前 proposal token。于是 S 仍是 proposal 前固定的自选集合；相对于这个重新定义的有限观测核，SPS 恒等与密度比界不因覆盖率降低而失效。
- 限制：它不再等同于规范文字中理想的 `topk(π) ∪ topk(p_T)`，会系统性排除“教师很热、策略极冷”的 token，是明确的能力边界/近似，不应被表述成原始并集的零误差实现。建议落账被排除的 p_T 候选数量与质量代理；该判定还以 IF-03 的 q/π 对象澄清、IF-09 的 top-K 行完整为前提。

### ② p_T 观测窗外用位置 floor：作为保守 surrogate sound，不是原 p_T 核的精确实现

- 代码位置：`src/trace_opd/steering/hook.py:74-95`
- 判定：对 π 侧候选但不在 p_T 观测窗的 token，真实 `log p_T` 不高于第 K 名 floor；用 floor 代入会使 surrogate `ell` 不小于真实 `ell`，即避免过度压低学生已有 token。若 floor 版本已 clip 到 `-c`，真实版本也必为 `-c`，此时零误差；未饱和时，SPS 只对 surrogate kernel 精确。
- 限制：单 token 增量的“少压制”方向是保守的，但经 S 内共同归一化后，不能外推成对所有 token 概率或任务效果都单调保守。应记录 floor 使用率、饱和率，并把理论对象明确写成 `b_S^floor`，不要与不可观测的原始 p_T 核混称。

### ③ `scale=max(std_loo,std_full)`：数值稳定有效，作为命题 C 主实现不 sound

- 代码位置：`verl/verl/trainer/ppo/core_algos.py:639-697`
- 判定：该 floor 确实避免 7:1 二元 reward 组中孤例 `std_loo=0` 导致的 `1/epsilon` 爆炸，数值动机成立；但 `std_full` 和 max 分支依赖当前 outcome，违反规范要求的 scale LOO 条件，不能继续声称严格无偏。详见 IF-02。
- 建议：主口径采用固定/冻结且与当前样本独立的 floor，或直接关闭组内 std 归一化；现公式若保留，应命名为“有偏稳定化”消融并与严格主实现分开。

### ④ 复用 native `rollout_is_weights`：设计 sound，`c2096e3` 的接线不 sound

- 代码位置：`verl/verl/trainer/ppo/steering_weights.py:42-93`；`verl/verl/trainer/ppo/core_algos.py:1390-1423`；`verl/verl/trainer/ppo/v1/agent_loop_tq.py:184-206`
- 判定：vanilla policy loss 先只对 `pi_theta/pi_old` 形成 PPO dual-clip，再在 `pg_losses` 上乘 W_beh，符合 NF02 的分离式位置要求；复用原生字段比另建 teacher-aware loss 更小、更不易把教师带进目标。`per_decision` 的前缀积跨 mask=0 结构 token 延续，也与轨迹 IS 坐标一致。
- 限制：IF-06 会在默认 Hydra 配置下阻断非空 ledger，IF-10 没有定义混合批的全 1 行；另外 `vtrace` 模式实际只是 `min(1,π/b)` 的逐 token clipping，并非带 value bootstrap/trace recursion 的完整 V-trace，名称应收窄。只有修复接线、锁定同步单快照/单 epoch 条件并做真实 TQ→actor loss 测试后，才能把该定案判为落地通过。
