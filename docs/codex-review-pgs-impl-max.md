# PGS 完整实现深度代码审阅

<!-- FINAL_SUMMARY: 收尾时在此处补入总体结论与最终计数。 -->

## 审阅范围与方法

- 工作树：`/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/trace_opd-pgs`
- 分支：`pgs-sps`
- 审阅快照：`c9d2466..c2096e3`（8 个提交）
- 权威规范：`docs/pgs-final-spec.md`（v3.0-final）
- 实现定案：`docs/pgs-impl-design.md`
- 取证口径：仅审阅已提交的 `HEAD` 快照；工作树中原有未提交改动不纳入结论。
- 执行约束：只读静态审阅；除本报告外未修改仓库文件，未启动训练、GPU、网络测试或长时间任务。

## 1. 忠实性

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

