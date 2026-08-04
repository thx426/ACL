# PGS 完整实施深度代码审阅

## 总体结论

**当前实施尚不具备进入 P4 真机冒烟或正式训练的条件。** 数学核心的保尾核、截断增量、残差分布与 `2ηc` 预算实现质量较好，NF01、NF02、NF10 的核心修法也能在代码中找到；但默认启动路径存在两个确定性阻断：hook 请求 `obs_topk=256` 而 launcher 只把 vLLM `max_logprobs` 配到 64，以及默认 Hydra `rollout_correction` 映射会被错误判定为已启用并与 PGS 权重冲突。另有同一 π scoring 通道未落实、LOO scale 自包含、异常轨迹误入训练、重采后停止语义等实质性正确性问题。

发现计数：**P0 = 2，P1 = 8，P2 = 5**。

审阅范围为 `c9d2466..HEAD`；Git 实际返回 **8 个 commit**（`c623d85` 至 `c2096e3`），不是任务描述中的 9 个；给出的 commit 列表本身也为 8 个。本报告已覆盖该区间全部变更，并参考了可读的两轮既往审阅报告。

验证边界：严格只读审阅；未启动训练、未访问网络/GPU。允许范围内 45 个本地单测通过；`grpo_loo` 的 6 个测试因当时只读沙箱不能创建 Python 临时目录而未执行，但已静态逐项审阅。仓库共有 88 个 `test_*` 函数，指定允许范围为其中 51 个。`git diff --check c9d2466..HEAD` 通过，38 个现存变更 Python 文件均通过 AST 解析。

## 发现

### IF-01 — P0 — launcher 的 `max_logprobs` 小于 hook 的真实请求宽度

位置：`src/trace_opd/steering/hook.py:43-45,124-134,157-160`；`scripts/launch_train.py:73-81`；`tests/test_launcher_command.py:58-67`。

问题陈述：`SpsGenerate` 默认把观测窗设为 `OBS_TOPK_FACTOR * topk = 4 * 64 = 256`，draft 和 teacher verify 都请求 256 个 logprobs；launcher 却只设置 `max_logprobs=max(topk, 20)=64`。实现文档 §7.3 明确写的是 256，而测试把错误的 64 固化成了期望值。

失败场景：用 `162-pgs-calib.yaml` 或填好 `theta_d` 的 `162-pgs-s1.yaml` 启动时，第一次 draft 即发送 `logprobs=256`，vLLM server 只允许最多 64，SamplingParams 校验拒绝请求；校准和正式 PGS 都无法生成首个 token。

修复建议：把 `obs_topk` 变成 `SteeringCfg` 的显式字段并校验 `obs_topk >= topk > 0`；hook、launcher、fingerprint 共用该字段；launcher 设置 `max_logprobs >= obs_topk`。测试应联合断言 hook 请求宽度和启动上限，而不是只断言 `topk`。

### IF-02 — P0 — 默认关闭的 verl rollout correction 被判成冲突，首条 steering 轨迹必失败

位置：`verl/verl/trainer/ppo/v1/agent_loop_tq.py:184-206`；`verl/verl/trainer/config/ppo_trainer.yaml:39-43`；`verl/verl/trainer/config/algorithm/rollout_correction.yaml:5-20`。

问题陈述：互斥检查使用 `algorithm.rollout_correction is not None`。但默认 Hydra 配置始终组合出一个映射；其中 `rollout_is: null`、`rollout_rs: null` 才表示语义上关闭。因此默认配置也会触发冲突。launcher 没有把整个对象覆盖为 `null`。

失败场景：修复 IF-01 后，任一轨迹产生非空 `steer_positions`，TQ put 前即抛出 “both write rollout_is_weights”；即使原生 rollout IS/RS 都关闭，训练批仍无法进入队列。

修复建议：抽取统一的 `rollout_correction_enabled()`，仅在 `rollout_is`、`rollout_rs` 或 bypass 的实际写权重模式启用时判冲突；在启动期做一次全局不变量检查。至少增加一条使用真实默认 Hydra 映射的测试，以及一条确实启用原生 correction 的冲突测试。

### IF-03 — P1 — 规范要求的 bare verify 未实现，核与 W 使用 draft q 而非同一 verify π

位置：`src/trace_opd/steering/hook.py:176-180,192-194,205-236,273-311`；`src/trace_opd/steering/sps.py:39-48`；`docs/pgs-final-spec.md:79-88`；`docs/pgs-impl-design.md:71-87`。

问题陈述：每段只有 teacher `prompt_logprobs` scoring；π 侧 top-K 和执行 token logprob 直接来自 draft 返回值。规范和实现设计均要求 bare、teacher 两次同一 scoring 通道，接受率和记账中的 π 取 bare verify，并记录 draft-vs-verify 残差。当前代码既没有 bare verify，也无法产生该残差；`sps.py` 的 docstring 还错误声称 `draft_log_pi` 来自 verify。

失败场景：vLLM `processed_logprobs`、采样精度或 logits processor 与 prompt scoring 有细微差异时，代码实际构造的是相对 draft proposal q 的核并记录 `q/b`，而训练目标所需的是 `π_old/b`。SPS 对代码内 q-kernel 可保持恒等，但不能据此声称规范中的 `b_S(π,p_T)` 和命题 A/C 已落实；数值漂移也无监控可发现。

修复建议：恢复 bare verify，显式比较 draft q 与 bare verify π；为保持 speculative 恒等，应预注册容差并明确对象：只有确认 `q≡π` 后才用 bare π 构核，否则接受分母必须是实际 proposal q，同时把理论声明改为 q-kernel 并对训练侧单独校正 `π_old/q`。锁定并断言 `logprobs_mode`、repetition/presence/frequency penalties、stop processor 等完整测度参数。

### IF-04 — P1 — `max(std_loo,std_full)` 让 scale 重新依赖当前样本，违反 LOO 条件

位置：`verl/verl/trainer/ppo/core_algos.py:639-697`，尤其 `683-692`；`tests/steering/test_grpo_loo.py:47-86`。

问题陈述：baseline 是 LOO，但 scale 取 `max(std_loo, std_full)`；`std_full` 含当前样本的 reward，且 `max` 的分支选择也随当前 reward 变化。注释中“无偏性由 baseline 承担”不成立：乘在 score function 上的随机 scale 若依赖当前 action/outcome，同样会改变期望梯度。n=2 时 scale 全部来自 full group，和规范“baseline 与 scale 均 LOO”直接冲突。

失败场景：二元 reward 组中，当前动作是否成为孤例会改变自身分母；同一策略的成功/失败样本被按包含自身结果的不同尺度加权，哪怕使用整轨 IS，也不再满足命题 C 条件 iv。现有测试只验证有限和手算一致，实际是在固化这项偏差。

修复建议：优先采用不按组 std 归一化（Dr.GRPO/固定 scale=1）；若必须防爆，使用预注册常数 floor `max(std_loo, sigma_floor)`，其中 floor 不由当前组当前样本决定，或用跨批冻结/交叉拟合的 scale。增加小型 bandit Monte Carlo 梯度期望测试，而不只测数值输出。

### IF-05 — P1 — 中途 abort/异常且已有响应时会被当成环境失败参与 GRPO

位置：`src/trace_opd/core/episode.py:181-191,270-310`；`src/trace_opd/train/agent_loop.py:203-208,325-361`；`src/trace_opd/steering/hook.py:164-168,300-301`。

问题陈述：`run_episode` 正确把 engine abort/异常标成 `OUTCOME_ENV_ERROR`，但 `_student_rollout` 仅在 `response_ids` 为空时重试和抛错。有任意已完成轮的部分轨迹会直接返回，`run()` 再把所有非 win 结果赋 reward 0。这与“基础设施故障不算环境 outcome”及只用 verifiable env reward 的红线冲突。

失败场景：第一轮环境交互成功，第二轮 teacher verify abort。结果含第一轮 token，因此不重试；该不完整样本以 reward 0 进入 GRPO。若 abort 发生在 hook 已执行但尚未交给 episode 的段内，ledger/D 统计还会包含最终 response 之外的尝试 token。

修复建议：所有 `OUTCOME_ENV_ERROR` 都应整 episode 作废并以全新 route/steering state 重试；第二次仍失败则显式 drop/raise，绝不能赋 0。steering 记账应按 turn 事务提交，或在整 episode 作废时整体丢弃。

### IF-06 — P1 — residual 重采出 EOS/stop 后仍继续 draft，输出含 post-stop token

位置：`src/trace_opd/steering/hook.py:169-175,251-262`。

问题陈述：自然停止仅依据原 draft 段长度判断；任何拒绝都会无条件 `continue`。若 residual replacement 自身是 EOS、特殊 stop token，或补全了文本 stop sequence，hook 不重新执行停止判定。

失败场景：draft 在位置 j 给出普通 token 并被拒绝，残差从 S 采到 EOS。代码把 EOS 放入 `executed` 后以“已经 EOS 的前缀”继续调用 draft，最终返回 EOS 后的额外 token；这不是带相同停止规则的 `b_S` 自回归轨迹，也可能改变 episode 解析结果。

修复建议：从 tokenizer 与 sampling params 获得有效 EOS/stop 条件；每个 replacement 后立即按与 engine 相同的规则更新 finished。文本 stop sequence 需在 replacement 后重跑增量 stop matcher。增加“replacement=EOS”“replacement 完成多 token stop”“原 draft EOS 被拒绝后换成非 EOS”三类测试。

### IF-07 — P1 — 行为状态 ledger 不完整，`cap_active` 字段实际上几乎恒为 0

位置：`src/trace_opd/steering/ledger.py:1-6,16-29,62-85`；`src/trace_opd/steering/hook.py:185-204,216-250,253-261`；`src/trace_opd/steering/budget.py:49-69`。

问题陈述：规范 §4 要求逐 token 的预算余额、`cap_active`、`steer_flag`、累计 `reject_count`。当前 record 缺预算余额、显式 steer flag 和 reject count；预算耗尽或 cap 生效后的 token 被直接旁路、不记账；发生第 `R_max` 次拒绝时 record 先写死 `cap_active=False`，随后才更新 cap。因而导出的 `steer_cap_active` 无法表示真正的 cap 尾段。模块顶层“未 steer 位置仍逐位置齐全”的说明也与稀疏实现矛盾。

失败场景：某轮第 4 次拒绝后余下 100 token 按 π 生成，ledger 只显示第 4 次拒绝前 `cap_active=0`，后续没有任何行；审阅者无法区分门关、预算阻断和拒绝上限阻断，也无法复建实际自适应行为状态或核验 NF04。

修复建议：对每个可执行 assistant token 记录行为状态，或提供与 response 同坐标的完整状态数组；至少记录 budget before/after、assigned/open、actual steer、budget_blocked、累计 reject_count、cap before/after。第 `R_max` 次拒绝和 cap 尾段必须可见；训练权重可继续只保留稀疏非单位 ratio 列。

### IF-08 — P1 — vLLM top-K 提取会把 rank 空洞伪造成 token 0、概率 1

位置：`verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py:677-700`；`verl/verl/workers/rollout/vllm_rollout/utils.py:645-675`。

问题陈述：response top-K 先以 `[0] * K`、`[0.0] * K` 填充，再按 rank 覆盖；没有校验 rank 非空、范围合法、唯一且恰好覆盖 1..K。缺 rank 会留下伪造的 token 0 与 `logprob=0`。prompt 提取虽用 `None`，但只用 `assert len`，同样不校验 rank 完整性，且 `rank is None` 会在比较时异常。sampled token rank>K 的跳过逻辑本身正确。

失败场景：某 vLLM 版本返回 top-K 加 sampled token，但 top-K rank 有空洞/重复；伪 token 0 被加入 S，`π(S)` 可能大于 1 而中途崩溃，或用错误概率构核。优化模式下 prompt 路径的 `assert` 还会消失。

修复建议：先收集 `(rank, token_id, logprob)`，再验证有限值、token 唯一和 rank 集合严格等于 `1..K`，不完整就带 request/position 上下文 fail loud；不要使用合法 token/probability 作为 sentinel。对 prompt/response 共用一个 extractor，并补 rank hole、duplicate、None、sampled rank>K 测试。

### IF-09 — P1 — `rollout_is_weights` 只在非空 ledger 写入，批内字段 schema 不统一

位置：`verl/verl/trainer/ppo/v1/agent_loop_tq.py:179-206`；`verl/verl/utils/tensordict_utils.py:918-948`；`src/trace_opd/train/agent_loop.py:170-201`。

问题陈述：只有 `extra.get("steer_positions")` 为真才创建权重字段。门全关、doc missing、doc overflow、预算为零或 baseline 轨迹都不带字段；同一训练批可能同时出现有字段和无字段的 key。`list_of_dict_to_tensordict` 又以第一个 dict 的 keys 为准。TransferQueue 包未 vendoring 在仓库中，跨 key 缺字段究竟报错、取交集还是填值无法由本地代码证明，但任何一种非显式语义都可能丢掉行为校正。

失败场景：一个 GRPO group 中一条轨迹有 steering record，另一条因 ε/门关闭而空。TQ 批取数时可能无法 stack，或因首条无字段而整批不把 `rollout_is_weights` 送进 actor，造成已倾斜轨迹未经校正训练。

修复建议：只要该 rollout 使用 PGS 协议，就始终写入与 response 等长的 dense 权重；空 ledger 明确写全 1。将“PGS 字段存在”与“本轨迹发生 steer”分开，并在真实 TQ 上增加混合 empty/non-empty/doc-missing 的批级测试。此项是本次新增的真机验证项。

### IF-10 — P1 — resume fingerprint 未覆盖实际优化目标，且可被 verl passthrough 绕过

位置：`src/trace_opd/common/config.py:323-335`；`scripts/launch_train.py:179,213-241,297-309`。

问题陈述：fingerprint 声称覆盖“优化什么”，但遗漏 `train.adv_estimator`，也不覆盖 advantage std 策略、PPO epochs/mini-batch 的单步条件、`obs_topk`/logprobs 口径、原生 rollout correction 等。更直接的是未知 CLI 参数在 fingerprint 计算后原样追加，可把 `algorithm.adv_estimator` 等覆盖成另一值。doc table 只哈希路径、不哈希内容或版本。

失败场景：同一 run 先以 `grpo_loo` 训练，后用 passthrough `algorithm.adv_estimator=grpo` 恢复；fingerprint 不变，checkpoint 在不同 objective/advantage 定义下继续。原路径上的文档表被替换内容也不会触发 guard。

修复建议：对最终解析后的 canonical objective 配置做哈希，至少包含 estimator、norm/std policy、epochs、mini-batch、loss mode、全部 steering/观测参数、rollout correction；限制或解析会改变 objective 的 passthrough。doc table 加内容 digest/不可变版本标识。

### IF-11 — P2 — Θ_D robust-EMA 与超限降级只写在文档/辅助函数中，生产路径未接线

位置：`src/trace_opd/steering/gate.py:1-8,68-95`；`src/trace_opd/train/agent_loop.py:198-201`；`scripts/calibrate_theta_d.py:11-15`；`configs/pgs.yaml:116-123`。

问题陈述：`robust_quantile_update()` 没有生产调用；每 episode 都从固定配置重建 gate。rate alarm 只打印日志，没有预注册降级动作。校准脚本和配置甚至明确说“全程冻结、不自动调”，与权威规范 §2.3 的“训练中 robust-EMA 慢更新；超限告警降级”不一致。

失败场景：训练分布漂移使开门率持续 60%，系统仅在单轨迹决策数达到 50 后打印一次 warning；不会降低 η/关门，也不会更新 Θ_D。长轨迹之外由于 gate 每 episode 重置，很多异常连告警门槛都到不了。

修复建议：在训练 step 边界聚合 D、冻结一步后做可恢复的 robust-EMA 更新，并定义确定的降级状态机；门率应跨 episode/step 统计。若第一版确实裁定永久冻结，则先修改权威 spec，不能让实现文档单方面覆盖它。

### IF-12 — P2 — D 直方图固定在 `c=2.5`，overflow 仍会输出伪精确 P85

位置：`src/trace_opd/steering/ledger.py:31-35,51-55`；`scripts/calibrate_theta_d.py:61-81`。

问题陈述：直方图上界硬编码 6.25；`c>2.5` 的值全部压入末桶。即使默认 c=2.5，末桶也同时包含合法区间 `[6.152...,6.25]`，脚本却把末桶统称为 `>6.25` overflow。若目标分位落在溢出质量内，线性桶内插值没有辨识依据，但脚本仍打印一个 `suggested theta_d`。

失败场景：`c=3` 且 20% D 值大于 6.25，P85 的真实值可能在 6.3 到 9 之间任意位置；脚本仍给出约 6.2 的建议并只附 warning，正式门率会显著偏高。

修复建议：把 histogram range、c 和 schema version 一同落盘，范围至少为当前 `c²`；更好使用可合并 quantile sketch。若 overflow 累积质量覆盖目标 quantile，必须拒绝给出建议值，而不是给 floor estimate。

### IF-13 — P2 — Steering 配置只做类型强转，缺少有限性与结构校验

位置：`src/trace_opd/common/config.py:174-195,221-240`；`src/trace_opd/steering/gate.py:32-47`；`src/trace_opd/steering/budget.py:19-24,49-55`。

问题陈述：YAML 科学计数法字符串坑已通过强转修复，但启动校验仍未检查 `topk>0`、`r_max>=1`、`gate_rate_bounds` 长度/次序/范围，以及 `c/b_max/theta_d` 为 finite。`NaN`/`inf` 可绕过部分大小比较，错误直到 rollout 中段才出现。

失败场景：`topk: 0` 能通过 `_validate`，首个位置构造空 S 后在 `candidate_d`/kernel 崩溃；`c: .inf` 产生无限预算扣费和无意义截断；三元素 rate bounds 直到 `lo, hi = ...` 才报错。

修复建议：在唯一配置入口用 `math.isfinite` 和完整不变量校验；显式要求 `topk>=1`、`r_max>=1`、`0<=lo<hi<=1`，并校验校准哨兵的有限上界策略。为字符串、NaN、inf、0、错误数组增加参数化测试。

### IF-14 — P2 — P1.5“旧机制全面出清”不完整，仍有失效配置与半死分支

位置：`verl/verl/trainer/config/ppo_trainer.yaml:42-43`；`verl/verl/trainer/config/distillation/distillation.yaml:1-20,58-83`；`verl/verl/trainer/constants_ppo.py:121-137`；`verl/verl/workers/engine/fsdp/transformer_impl.py:1250-1466`；`verl/verl/workers/engine/megatron/transformer_impl.py:538,672,909-1066`；`verl/verl/workers/engine/veomni/transformer_impl.py:808-820`；`src/trace_opd/core/episode.py:323-361`；`src/trace_opd/core/types.py:62-78`；`src/trace_opd/__init__.py:1`。

问题陈述：活跃 PGS loss 路径确实没有教师项，这是红线上的好结果；但默认配置仍组合 distillation 节点，其 `_target_` 指向已经删除的 `Distillation*Config`。engine 中仍保留 `distillation_use_topk/distillation_only/teacher_*` 分支，环境变量转发仍提到已删除的 microscope/losses。trace_opd 侧还留有无人引用的 mixed-objective `replay_bare_tokens`、SFT leakage helper 和 self-distillation 包说明。现有 vendored 守护测试的字符串 allowlist 没覆盖这些位置。

失败场景：旧 verl 配置或 engine 调用者设置残留 key 时，进入没有完整生产者/配置类的半死路径；生成配置继续宣称一个实际上无法实例化的 distillation 功能。维护者也可能误以为这些分支仍是 PGS 需要，从而重新接回教师 loss。

修复建议：二选一并明确边界：若目标是 PGS 专用干净 vendor，删除默认组合、generated YAML、engine 分支、环境变量和 trace 死代码/陈旧说明；若要保留 upstream distillation，则完整恢复为隔离的可选上游能力，但不得与 PGS 共享字段或默认组合。扩展守护测试到配置、engine 和文档入口。

### IF-15 — P2 — 关键红线依赖可被 `python -O` 去掉的 assert，且多处 docstring 与真实行为相反

位置：`src/trace_opd/train/agent_loop.py:150-154`；`src/trace_opd/steering/hook.py:147-148`；`src/trace_opd/steering/kernel.py:39-55,91-95`；`src/trace_opd/steering/ledger.py:1-6,62-67`；`verl/verl/workers/rollout/vllm_rollout/utils.py:661-669`。

问题陈述：teacher/student 生成前缀一致性、prompt 长度、核不变量和 prompt top-K 结构部分依赖 `assert`；优化模式会全部移除。`SteeringKernel` 声称构造后不可变但 dataclass 不是 frozen；ledger 声称未 steer 位置逐位置记录但实际稀疏；kernel/sps 声称 π 来自 verify 但实际来自 draft。对于机制红线，错误文档会直接误导后续修改。

失败场景：生产镜像使用 `PYTHONOPTIMIZE=1` 且 teacher/bare template 尾部不一致，系统不会 fail loud，teacher 行与 student token 坐标错配后仍继续采样和记账。

修复建议：把所有机制/数据边界不变量改为显式 `ValueError/RuntimeError`；真正冻结 kernel 的输入和派生状态，或删掉不可变承诺；同步 docstring 到实际稀疏 ledger 与 π 数据来源，并用测试验证优化模式不影响红线。

## 忠实性核对表

| 规范条款 | 代码位置 | 判定 | 说明 |
|---|---|---|---|
| §2.1 保尾核 | `steering/kernel.py:28-146`；`steering/hook.py:59-95` | 条件通过 | S 内保质量、S 外 π、密度比界公式正确；依赖 top-K 行完整，且实际 π 来自 draft q。 |
| §2.2 截断 | `kernel.py:28-36`；`hook.py:92-94` | 通过 | `clip(log p_T-log π, ±c)` 正确，floor 属实现近似口径。 |
| §2.3 D 门/可测性 | `gate.py:29-66`；`hook.py:205-216` | 部分通过 | gate API 不接 proposal token，D 公式正确；但数据是在整段 draft 后取得，且 EMA/降级未接线、状态记账不全。 |
| §2.4 `D_∞` 预算 | `budget.py:16-46`；`hook.py:216,238-250` | 数学通过/审计部分失败 | `δ=2ηc`、余额缩 η、耗尽后 π 正确；缺逐位置余额与全轨迹落盘核验。 |
| §2.5 SPS | `sps.py:31-78`；`hook.py:155-262` | 部分失败 | 固定单点核的接受/残差恒等正确，拒绝后丢后缀并续采正确；bare verify、停止语义、默认 launcher 容量未成立。 |
| §2.5 `R_max` 行为态 | `budget.py:49-69`；`hook.py:201-204,253-261` | 行为正确/记账失败 | 恰第 `R_max` 次后关闭，下一段走 π；cap-active 尾段不落账。 |
| §2.6 分离式 W_beh | `steering_weights.py:42-93`；`core_algos.py:1390-1422`；`losses.py:85-112` | 结构通过/默认链路阻断 | 权重确实乘在 PPO clip 结果之外；IF-02 使默认 PGS 无法把权重送达，IF-09 有混批风险。 |
| §2.6 LOO advantage | `core_algos.py:639-697` | 失败 | baseline LOO；scale 通过 `std_full` 包含自身，不满足条款。 |
| §4 逐 token 记账 | `ledger.py:16-85` | 失败 | logπ/logb/η/δ/D/门/propensity 有；预算余额、steer_flag、reject_count 和真实 cap 尾段缺失。 |
| 命题 A | `kernel.py` + `sps.py` | 单点实现通过，系统声明未闭合 | mock 统计测试支持固定 q-kernel；默认启动阻断、q/π 口径和 stop 规则使规范对象上的完整轨迹声明未成立。 |
| 命题 B | `kernel.py:84-95`；`budget.py:30-46` | 数学通过 | 全词表逐点界与保守扣费正确；建议增加实际 ledger 的 `abs(sum logπ-logb) <= B∞` 端到端断言。 |
| 命题 C | `steering_weights.py`；`core_algos.py:639-697` | 不满足 | 默认 per-token 本就按规范承认有一阶偏差；即使切 per-decision，scale 自包含、q/π 数值口径和单步 resume 配置守卫仍不闭合。 |
| 命题 D | `tests/steering/test_steering.py` | 非系统保证 | 单点数值示例/核性质可验证，不应外推 episode；代码没有额外错误声明。 |
| 教师不进 loss 红线 | launcher、`core_algos.py`、`losses.py` | 通过 | 活跃 loss 是纯 GRPO/PPO，无 teacher/distillation term；IF-14 是清理和半死路径问题，不等于当前 loss 已复活。 |

### NF01–NF10 修法落地结论

- NF01：核心已落地，保尾核是唯一显式行为对象；受 IF-03 的 q/π 口径和 IF-08 数据完整性制约。
- NF02：loss 乘法位置已正确分离；默认通路被 IF-02 阻断。
- NF03：per-token 偏差被文档承认，per-decision 可展开；LOO scale 未按条件 iv 落地，命题 C 仍不能声明。
- NF04：gate 签名的因果接口与 `R_max` 状态机已落地；完整行为状态记账未落地。
- NF05/NF06：credit 与 sham 已按定稿移出第一版，不再构成当前实现分支。
- NF07：propensity 只记账不进训练权重，D 用保尾核对象计算；robust-EMA、降级和完整 assigned/actual/budget-blocked 记录未闭合。
- NF08：五臂功效设计已明确移出机制规范；当前首批配置不重新声称三 seed 可作最终判决。
- NF09：教师文档位于动态对话前、同 route 复用成立；真实 prefix-cache × prompt-logprobs 仍是已知 P4 项。
- NF10：密度比上界与扣费统一为 `2ηc`，已正确落地。

## 测试覆盖与缺口

### 已覆盖

- 88 个测试函数中，PGS 指定范围有 51 个：kernel/gate/budget/SPS、hook、W_beh、多轮 episode 集成、`grpo_loo` 和 vendored 清理守护。
- 本次安全执行 45 个：`test_behavior_weights.py`、`test_steering.py`、`test_sps_hook.py`、`test_episode_sps_integration.py`、`test_vendored_verl.py`，全部通过（7.06 秒）。
- `test_output_matches_analytic_b` 用 30,000 次独立单 token mock 样本检验 TV；能排除“实现恒等于 π”的明显假通过。
- 已覆盖单点核正规性/尾部、残差支撑、预算恰耗尽、拒绝续采、基础多轮坐标、三个 W 模式的机械展开、LOO 的 n=1/全同/孤例有限性，以及若干旧路径字符串守护。

### 关键缺口

1. 没有把 hook `obs_topk` 请求和 launcher `max_logprobs` 放进同一个契约测试；现有 launcher 测试反而固定错误值。
2. 没有用真实默认 Hydra `rollout_correction` 对象测试互斥判断，也没有 PGS 权重穿过完整 TQ→worker→vanilla loss 的 CPU 集成测试。
3. 没有 bare verify、draft-vs-verify π 残差、`logprobs_mode`/完整 processor 测度测试。
4. 统计等价测试只验证位置无关、单 token 分布；缺多 token 条件分布、早位置拒绝后重建核，以及多个随机种子/置信界。当前 breakpoint 断言 `A or len>=2` 较弱，不能逐次证明“丢弃全部旧后缀”。
5. 缺 residual replacement 为 EOS/stop、原 EOS 被拒绝、空 draft、部分 abort、verify 异常、执行尾截断的对抗测试。
6. 多轮集成只走 `D=0` 门全关路径；没有多轮真实拒绝、跨轮预算耗尽、`R_max` cap 尾段与 ledger 坐标联合测试。
7. top-K 提取无 rank hole/duplicate/None、sampled rank>K、非有限 logprob 测试。
8. W_beh 无 empty/non-empty/doc-missing 混合 TQ batch；per-decision 无跨结构 token、尾截与实际 response mask 联合测试。
9. `grpo_loo` 测试验证了当前公式而非无偏性；缺 norm=true 的 n=2 理论判定和 bandit Monte Carlo 梯度测试。其 6 个测试本次因只读临时目录限制未运行。
10. fingerprint 无“只改 adv_estimator/文档内容/passthrough 后必须拒绝 resume”的测试；配置无 NaN/inf/非法 bounds 测试。
11. D 直方图无 quantile 落入 overflow 时拒绝建议值的测试。
12. 旧机制守护是有限字符串 allowlist，未覆盖默认 config、generated config、engine 的半死 distillation 分支及 trace_opd 死代码。

已知 P4 项（prefix cache × prompt_logprobs、真实 `logprobs_mode`、draft/verify 数值残差、墙钟/OOM）仍需按实现文档执行。**本次新增只能在真实集成环境最终确认的项目**是：TransferQueue 对同批 key 字段不齐时的精确语义及 `rollout_is_weights` 是否完整送达；同时应在真实 vLLM 版本核验 top-K rank 完整性与 stop metadata 能否支持 replacement 后重判。

## 四个实现口径定案的独立判定

### 1. S 的两个封闭点

**判定：有条件 sound。** 把 π 候选限制在预先取得的 `obs_topk`，并排除 π-rank 超窗的 p_T token，仍然是在 proposal 前由前缀决定的固定 S；相对于这个重新定义的有限观测核，命题 A/B 不因覆盖度下降而失效。代价是它不再等同规范文字中的完整 `topk(p_T)` 并集，能力边界应明确标成实现近似。建议记录被排除的 p_T 候选数/质量代理。该判定以 top-K 行完整且 IF-03 的 q/π 对象澄清为前提。

### 2. p_T 观测窗外使用位置 floor

**判定：作为保守 surrogate 是 sound，但不是对原始 p_T 核的零误差实现。** 真实 p_T 低于观测 floor，以 floor 代入会让负向 `ell` 绝对值减小，即少压学生已有 token，方向上保守；当随后 clip 到 `-c` 时确实零误差。未饱和时 SPS 只对 floor-surrogate kernel 精确。应记录 floor 使用率/未饱和率，并把理论对象显式写成 surrogate `b_S`。

### 3. `scale=max(std_loo,std_full)`

**判定：理论上不 sound。** 它能解决二元 reward 孤例的数值爆炸，但 `std_full` 和 max 分支依赖当前 outcome，违反权威规范的 LOO scale 条件，也破坏命题 C 的无偏声明。可以保留为明确标注“有偏稳定化”的消融，主实现应改为常数/冻结 floor 或关闭组内 std 归一化。

### 4. 复用 verl 原生 `rollout_is_weights`

**判定：设计本身 sound，当前接线不 sound。** vanilla policy loss 在形成 PPO dual-clip 损失后再乘该字段，正好实现 NF02 的分离式校正；复用原生通道比另写 loss 更小、更可靠。但 IF-02 的错误互斥谓词会阻断默认路径，IF-09 的字段不齐可能使混合批丢权重。修复这两项并做真实 TQ 端到端验证后，该定案可保留。

## 最终建议

先修 IF-01、IF-02 并增加启动级契约测试；随后按顺序处理 IF-03、IF-04、IF-05、IF-06，之后再做 P4。当前不应以“88 tests”或单点统计等价测试作为上线依据，因为它们没有覆盖两个默认配置阻断和完整行为/训练通路。
