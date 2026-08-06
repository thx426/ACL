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

## 4. 训练端改动（落地定案，2026-08-04）

**零新 loss 代码**：verl 原生 `ppo_loss` → `compute_policy_loss_vanilla` 已内建
`rollout_is_weights` 通道，权重乘在 dual-clip **之外**——恰是 NF02 分离式校正
`W_t · min(r_t·Â, clip(r_t)·Â)` 的原生正确形态。落地：

- `verl/trainer/ppo/steering_weights.py`：稀疏 ledger → 稠密 per-token 权重
  （`per_token` 默认 / `per_decision` 前缀积 / `vtrace` ρ̄=1 截断）；
  逐 token `|log(π/b)| > 2ηc` 即记账损坏 fail-loud
- 写入点唯一：`agent_loop_tq.py` TQ put 前展开；与 `algorithm.rollout_correction`
  互斥检查（同字段两个写者 = 事故）
- 通路：KVBatchMeta.fields=None ⇒ worker 全字段自动取数 ⇒ `ppo_loss` 的
  `rollout_is_weights` 分支命中，无中间白名单
- advantage：注册 `grpo_loo`（baseline 严格 LOO；scale=max(std_loo, std_full)
  ——**规范级缺陷修订**：二元 reward 下孤例的 std_loo=0 会给出 −1/ε 的
  advantage（7胜1负组是常态），full-std floor 封住；无偏性由 baseline 承担）
- 命题 C 条件 v（单快照单步）：ppo_epochs=1（verl 默认）+ mini_batch=全批

## 5. 实现口径定案（引擎数据面）

1. **S 的两个封闭点**（S 是自选固定集，选择只影响覆盖度、不破坏命题 A）：
   π 侧候选限 draft 的 obs_topk（=4×topk）观测窗；p_T top-K 中 π-rank 超窗的
   token 不入 S（π 极冷 token 抬 e^{2ηc}≈e^{2.5} 倍仍可忽略，与「OPD 只能调
   已有行为」的能力边界一致）。S 内 p_T 窗外的 token 用该位置观测 floor 代入
   ——clip 到 −c 时零误差，未饱和时误差有界且方向保守。
2. **logprobs 透传**：vllm_async_server 的 `logprobs` 支持 int K（top-K 明细行
   `response_topk_ids/logprobs`，rank 序）；verify 用 `prompt_logprobs=K`
   （行语义：第 j 行预测位置 j+1，末行 dummy——`extract_prompt_logprobs`）。
3. **测度守卫**：draft 侧要求 T=1/top_p=1/top_k=-1（失配 fail-loud）；verify
   请求显式同口径。`logprobs_mode` 的实际语义列 P4 首查。
4. **θ_D 标定**：ledger 逐候选位置 observe_d → 64-bin 直方图随 dump 落盘；
   标定 run 三件套 = theta_d=1e9 + epsilon=0 + gate_rate_bounds=[0,1]
   （行为=纯 bare、零干预）；`scripts/calibrate_theta_d.py` 聚合出 P85。
5. **doc 表**：`gt_docs_rl_train_v2.jsonl`（162：trace_opd-pgc/data/out/，
   2875 条）与 `_load_doc_table` 格式零转换兼容，首实验直接复用。

## 6. 进度

```
P1   ✅ steering 包核心 + 12 单测（c623d85）
P1.5 ✅ 旧机制全面出清：trace_opd 七包 + verl 蒸馏面根除（033c44d）
P2   ✅ SPS 钩子全链路 + 引擎透传 + agent_loop 接线 + 7 测试（daa171a）
P3   ✅ W_beh 展开（rollout_is_weights）+ 三口径 + grpo_loo（7646332, c84008f）
P3.5 ✅ D 直方图标定数据面 + calibrate_theta_d.py（a9dfc9f）
P4   ⏳ 真机冒烟 —— 162 被 C3 占（step 38/97，~08-05/06 完）；A800 sshd 两端口
        reset 连不上（等端口恢复或 C3 结束）
```

## 7. P4 冒烟清单（上机首查，按序）

1. `prompt_logprobs`/`logprobs=K` 真实输出格式与 `logprobs_mode` 语义
   （raw vs processed；T=1 全分布下应一致）
2. **prefix cache × prompt_logprobs 冲突**：vLLM 对 cached prefix 可能不产
   logprobs 行——若冲突，教师路由禁 prefix cache 或接受重算（实测定）
3. max_logprobs=256（obs_topk）启动参数生效；verify 段级 prefill 的墙钟
   （NF09 算力账：每轮 ≤(R_max+1) 次 prefill）
4. draft/verify 的 π 数值残差（bf16 采样 vs 打分，~1e-3 预期）
5. θ_D 标定 run（三件套配置）→ P85 → 正式 run 配置
6. 冒烟对照：steering off vs on（θ_D 标定后）首 2 step 的 steer_count/
   budget_spent/reject 率曲线（rollout_extra 通道）与墙钟差


## 8. sp=2 / rmpad 路径下 W_beh 的正确性核查(2026-08-04,用户裁定 sp 配置化后)

**结论:W_beh 通路与 ulysses_sp / use_remove_padding 正交,sp=2 下正确。** 链路:

1. rollout 侧:`agent_loop_tq` 展开的 dense 权重是 **response 坐标**的 1D tensor
   (逐轨迹变长),TQ 以 nested 形态存取——与训练端序列布局(padded/THD)无关;
2. loss 层:`ppo_loss` 对 `response_mask/old_log_probs/advantages/rollout_is_weights`
   统一 `to_padded_tensor()` 回 **padded 2D 空间**;`model_output["log_probs"]` 在
   sp>1 时已由 engine 的 `gather_outputs_and_unpad` 归并回全长再 nested,经
   `no_padding_2_padding` 同样落 padded 2D——vanilla loss 的逐元素乘全部发生在
   同一 padded 坐标系,rmpad/sp 只是 forward 的内部实现;
3. sp=2+rmpad 的 GRPO 主路径是 162 的历史成熟档(4B/SP2 主线全程),PGS 未触碰
   该 forward 路径(蒸馏分支删除只移除了死代码)。

**运行时终验**(替代无法本地复现的分布式数据流):s1 首个有 steer 的 step,
swanlab `actor/rollout_is_weight_off1_frac` 必须 >0(run-guide 步骤⑤ 首查项)。

## 9. 分臂采样(机制 v3.1;用户裁定 2026-08-06,规范附注第 8 条)

**动机**:全臂干预的三重问题——训练 score 测 E_b[R] 污染逐步能力读数;
组内成功率被抬升 ⇒ GRPO 组内方差坍缩 + policy 自然失败样本稀疏化(负梯度
从高频低幅劣化为低频高幅);per-token W_beh 不校正前缀测度,命题 C 一阶
偏差项随 B_∞ 线性放大(400 vs 20 ⇒ 20×)。分臂恢复 bare 读数与原生负
梯度,混合组 baseline 把"教师能成/policy 失败"转成对比信号;等价于
**轨迹级 50% 探针**(token 级 ε 探针保留在 steered 臂内)。

**实现链**(全部有单测):

1. `steering/arms.py`:`steered_arm_count = round(n×fraction)`;
   `is_steered_arm(session_id, n, fraction)` 前缀切分。fraction<1 时
   session_id 缺失/越界 **fail loud**(静默回退=整组变回全 steered,
   正是分臂要修的污染);fraction=1.0 恒 True 且容忍 None(历史路径)。
2. `session_id` 来源:verl TQ 层 `agent_loop_tq._run_prompt` 逐副本
   `session_id=i` → 基类 `_run_agent_loop` 的 `**kwargs` 透传 →
   `TraceAgentLoop.run()`(vendored 断言锁链:test_vendored_verl)。
3. `agent_loop.run()`:bare 臂 `steer_ctx["doc"]=""` ⇒ `_make_generate`
   走纯 EngineGenerate(零 verify/零倾斜/空账 W≡1)——复用 IF-09 的
   "空账=全 1、schema 恒齐"路径,TQ/loss 层零改动。`doc_missing` 保持
   任务级语义(两臂同值);臂身份独立字段 `steer_arm`(账+dump+曲线)。
   researcher 文档仍按 (step,task) 生产一次;bare 臂 await 同一共享任务
   只为携带同一份 research_fields(schema 恒齐),文档不注入。
4. 曲线:`metric_utils.compute_rollout_extra_metrics` 产
   `won_bare_rate`(去混杂能力读数)/`won_steered_rate`/`steer_arm_gap`
   (δ=q−p,教师在线增益;空臂批跳过不发 NaN)。
5. 指纹:fraction=1.0 不进 material(既有 run 指纹逐字节不变可续);
   <1.0 追加 ⇒ 换指纹换 run_dir。

**已知次级效应与留白**(规范附注第 8 条):steered 正信号放大/bare 珍贵
成功稀释(观察不预修);消融留白 = 口径 B(两臂独立组)/口径 C(steered
只作 baseline,W_beh 全免)/δ 驱动 fraction 退火。**θ_D 自适应只吃
steered 臂的 D 观测**:每 worker 每 step 观测量减半,min_obs=500 下更新
周期约 4→7 step(预期内,changelog 已告知实验侧)。
