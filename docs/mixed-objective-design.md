# 混合目标机制设计:OPD × RL × SFT(mixed-objective)

> 设计真源。2026-07-26 用户提出;分支 `mixed-objective`。
> 定位:**正式的、可开关、可消融的通用机制**——不是 1.7B 一次性补丁。
> OPD 混 SFT/RL 本身是方案后续重要消融实验的组成部分(用户原话)。
> 触及 PG-1/PG-5"第一轮纯 OPD"的扩展 → 进完成报告 Proposed Global Decisions 报备 master。

## 0. 动机:为什么弱模型需要混合目标

**用户实验观察(经由 4B/1.7B 两条线的大量真实 run)**:4B 仅靠 OPD loss(forward_kl_topk
纯蒸馏)即可持续自我改进,KL 信号强度有效(step-9 行为起飞实证);1.7B 则始终看不到效果
——teacher 仅是同权重 policy + 特权文档前缀,对 student 自身 rollout 的(大量错误)轨迹
状态打分,**弱模型的 doc 语义注意力在错误轨迹前缀下迅速衰减,产不出有效学习信号**。

这与本仓既有证据链完全吻合:

| 既有证据 | 与本设计的关系 |
|---|---|
| 深度衰减 50×(t1-3 → t26+),65.6% 信号在 turn 1-3 | 弱模型上衰减更快,OPD 唯一信号源枯竭 |
| framing 探针:错误循环态下任何条件于学生上下文的 teacher 均无法表达纠正 | **on-policy-only 蒸馏有机理天花板** → S7(guided 轨迹混入)升根本解 = 本设计 SFT 臂 |
| guided(doc in context)96% vs 蒸后 bare 7-12% | 瓶颈在转移通道;guided 轨迹是现成的高质量专家分布 |
| CD-003 崩塌机理:真修复 = 冻结 teacher + **锚** | RL(真实 reward)与 SFT(外部成功轨迹)正是"锚"的两种形式,无自蒸馏正反馈回路 |
| 观察㊶:doc 再生使 loss 2×(moving teacher 目标漂移) | RL 锚可容忍 moving teacher(SEED 全程 self-evolving 不崩的文献判决,见 §1) |

## 1. 文献基座(2026-07-26 调研,全部真实论文)

### 1.1 RL × OPD 混合(用户点名两篇,同一文献簇)

**SDAR — Self-Distilled Agentic Reinforcement Learning**([arXiv:2605.15155](https://arxiv.org/abs/2605.15155)):
- 目标:`L = L_GRPO + λ·L_SDAR`,**RL 为主干,OPSD 为 gated 辅助**。
- OPSD 项(token 级,采样 token 上):`ℓ_t = g_t·(sg[log π_T(y_t|s_t⁺)] − log π_θ(y_t|s_t))`,
  gate `g_t = σ(β·Δ_t)`,`Δ_t = sg[log π_T(y_t|s⁺) − log π_θ(y_t|s)]`(teacher-student 差,detach)。
  正差(teacher 更信任)强化、负差(teacher 否决)软衰减,gate 有界防梯度爆炸。
- 超参:**λ=0.01、β=5、GRPO G=8 组内标准化、clip ε=0.2、lr 1e-6**;环境含 ALFWorld。
- **对我们最关键的两个判决**:
  ① 无 gate 的朴素 GRPO+OPSD 在 **Qwen3-1.7B 上灾难退化(32.0% vs GRPO 基线 46.1%)**
  ——蒸馏梯度无界压垮 RL 信号;SDAR 修复后 **53.9%(+7.8)**。这正是用户 1.7B 观察的文献印证,
  也给出一个**邻近文献参照**(同环境同基座:ALFWorld × Qwen3-1.7B)。
  **⚠ 不是可直接判输赢的 baseline**(外部审阅 2026-07-26 核对原文):SDAR 用采样 token 的
  reverse-KL surrogate + gate 而我们用 top-k forward KL、其 OPD 系数 0.01 而我们 A3 是 1、
  它含 fixed-reference KL 而我们不启用、response length 512 vs 我们 32768、其 ALFWorld 总分
  是六类等权平均而我们的 val 是两 split 内按任务平均;skill 来源与训练步数亦不同。
  **写进论文时只能作"邻近工作参照",不得宣称达到或超过 53.9。**
  ② teacher 负差三来源:skill 质量 / skill 利用失败 / **multi-turn drift(轮深处 gap 变宽)**
  ——与我们深度衰减测量同一现象。λ=0.1 梯度支配、0.001 不足,0.01 甜点;gap gating > entropy gating。

**SEED — Self-Evolving On-Policy Distillation**([arXiv:2607.14777](https://arxiv.org/abs/2607.14777)):
- `L = L_rl + λ_opd·L_opd`;GRPO(N=8)+ gated token 级 OPD(gate 同 SDAR 形式,β_opd);
  πθ_old 既收轨迹又当 analyzer 产 hindsight skill,**teacher 随 policy 迭代进化而全程不崩**。
  **⚠ 不能把稳定性单独归因于 RL 锚**(外部审阅指出):SEED 同时用了 SFT 初始化、gated OPD、
  GRPO、fixed-reference KL 与同步旧策略快照,原文没有隔离出 RL 单独的稳定作用。
  它只能说明"RL 锚 + moving teacher 可以共存",不能说明"RL 锚是稳定的原因"。
- 与我们的差异:SEED 的 skill 是事后 hindsight(从完成轨迹提取);我们的 doc 是任务先验
  (researcher 现场实验产出),信息含量更强。SEED 的 OPD 只在采样 token 上(logp shift),
  我们的 forward_kl_topk 是 top-k 分布蒸馏(信号更密)。

### 1.2 SFT(特权专家轨迹)混合

**PACT — Privileged Trace Co-Training for Multi-Turn Tool-Use Agents**
([arXiv:2606.16215](https://arxiv.org/abs/2606.16215)):特权轨迹共训的直接先例,可移植结论:
- **剥离特权上下文训练裸 policy**(privileged 生成的轨迹,SFT 时 strip 掉特权信息)
  ——强迫学到可泛化行为而非对特权线索的依赖;
- rollout 保持 prompt-only,expert trace 只用于优化;SFT 只监督可控的 reasoning/tool-call 段;
- 成功过滤(binary success)+ **λ_SFT 先高后衰减**(curriculum:早期 SFT 重、policy 稳定后 RL 接管)。

> **⚠ 引用更正(2026-07-26,外部审阅核对原文后指出)**:本文档此前称"PACT 实证:过滤
> reasoning 中显式引用特权信息的轨迹是 essential,不过滤则伤 unseen"——**该论断在 PACT
> 原文中不存在**。PACT 的 leakage 过滤针对的是"不泄漏未来 tool call 的安全 reasoning
> prefix",不是过滤"the guidance says…"这类自述引用;原文也没有对应的 unseen 消融。
> 我们的词面过滤(§3.3 第 2 条)因此是**本项目提出的启发式,待自身消融验证**,不是已被
> PACT 证明的设计。**误引根源:采信了论文的二手摘要(WebFetch 由小模型总结)而未核对
> 原文——凡要写进设计依据的论文结论,必须回原文核对。**

**权重设计谱系**(供二期参考):CHORD([arXiv:2509.06948](https://arxiv.org/html/2509.06948v2)
相关工作节)= 全局权重 + token 级权重(up-weight 不确定 expert token、down-weight 大偏离 token);
LUFFY = off-policy expert trace 与 on-policy rollout 同组 Mixed-Policy GRPO(需 shaped-policy IS
修正,复杂度高,**本设计不采用**、记为替代路线);SRFT = 熵引导动态权重。
反方证据存档:[arXiv:2604.23747](https://arxiv.org/pdf/2604.23747)(SFT-then-RL 优于混合)
——正因存在争议,本机制的**可开关消融设计**本身就有论文价值。

## 2. 目标函数(规范形)

```
L_total = w_opd · L_OPD + w_rl · L_RL + w_sft · L_SFT
```

- **L_OPD**(现状,不动):forward_kl_topk(top-64)+ 硬 clamp(10)+ SUM 聚合。
  `w_opd=1, w_rl=0, w_sft=0` 时**不产生任何额外目标计算,训练数学语义与改动前相同**
  (4B 主线零感知,默认值)。**措辞更正**:此前写"逐字节一致"过强——rollout dump 无条件
  多了 `sample_role`/`sft_accepted` 两个字段、launcher 命令多两个零系数、manifest 与 viz
  schema 也变了;参数更新的数值等价性**尚未做同 checkpoint 同 batch 的一步差分验证**。
- **L_RL**:GRPO——won∈{0,1} 轨迹级 reward,`{uid}_{session_id}` 组内标准化 advantage
  (G=8 bare rollouts,verl `adv_estimator=grpo` **现已在算**,见 §4),PPO-clip
  vanilla policy loss(verl 上游 `distillation_ppo_loss` 组合路径现成)。
  follow SDAR/SEED:不加 ref-KL 正则第一期(我们无 ref 模型常驻,显存紧;SEED 的 βKL
  是防漂移的二道锚,我们的 OPD 项本身即强正则,一期观察 entropy 曲线再定)。
- **L_SFT**:guided 成功轨迹上的 token 级 CE(response 生成段,mask 语义同现构流),
  **bare 上下文形态**(PACT 剥离原则):输入是裸 system prompt + 轨迹 obs/action 流重构,
  评测同构(PG-3)。

**归一化口径(实现中核实,重要)**:`config.global_batch_info` **只由 `ppo_loss` 写入**,
而纯 OPD 路径从不调用它 ⇒ **现行 OPD 项一直是"每微批 token 平均"**(4B 主线就跑在这个
口径上)。因此:
1. **L_SFT 同样用每微批 token 平均**(分母 = 该微批的 expert token 数,clamp≥1)——
   三项同族,`w_sft` 才是"一个均值对另一个均值"的权重,且在各臂含义一致;
   无 expert token 的微批贡献恰为 0(agg_loss 的 token-mean 会在此除零产 NaN,故自算)。
2. **开 RL 臂会污染 OPD 项的归一化**:`ppo_loss` 的副作用写入 config,而 config 跨微批
   存活 ⇒ 第二个微批起 OPD 项静默换成全局分母(尺度跳 ~微批数 × dp_size),**只在开 RL 的臂
   发生** ⇒ 消融将比较两个不同尺度的 OPD 目标。实现里 OPD 项在 `rl_loss_coef>0` 时对
   **清空后的快照**求值,把各臂钉死在纯 OPD 口径上(A0/A2 与上游 `use_task_rewards` 语义均不受影响)。

**系数哲学(消融自由,预设两档)**:

| profile | w_opd | w_rl | w_sft | 依据 |
|---|---|---|---|---|
| pure-opd(默认,=现状) | 1 | 0 | 0 | 4B 已实证有效 |
| 1.7B 起步建议 | 1 | 1 | 0.5 | OPD 保持主干口径 + RL 全额锚 + SFT 半权(PACT 衰减思想的静态保守版);真实配比由消融定 |

SDAR 的 RL 主干哲学(w_opd≈0.01 量级)作为消融臂之一,不作默认——我们与 SDAR 的差异
是 doc 信息含量(现场研究产物 vs 检索 skill)与 OPD 形式(top-k 分布 vs 采样 token),
OPD 应保留争夺主干的资格,这正是消融要回答的问题。

**Gate(σ(β·Δ) token 门控)一期不实现**,理由:①我们的 KL 有 clamp+topk+可调 w_opd
三重有界,SDAR 所防的"无界蒸馏梯度"在我们设置已有保护;②gate 对 top-k 分布蒸馏的移植
语义需单独设计(SDAR/SEED 均为采样 token 形式);③刚裁定移除三件套(K2 教训:加权类
组件必须先有实验判据)。混合模式首跑显微镜观察后,若 KL 项在错误轨迹深处仍产生有害
梯度,gate 作为二期选项(与 K1 重审同批立项)。

## 3. 数据流:三臂样本的生产与隔离(以下为**已实现**口径)

### 3.1 batch 构成(mixed 模式)

```
每 step 每任务:rollout.n = rollout_n(8,bare) + mix_sft_rollouts(4,guided)= 12 副本
  副本 0-7(bare,现状不变):裸 rollout → teacher(=policy+doc)打分 → OPD;
                              won reward → GRPO 组(8 条同组)→ RL
  副本 8-11(expert,新增):await 研究定稿 doc → run_episode(guidance=doc)
      → 通过三道过滤 → bare 化重放(§3.2)→ sample_role="sft" → CE
      → 未通过:同一行保留,response_mask 整条置 0 → 各处贡献恰为 0
```

- **副本分流**:同 (step,task) 的 12 个副本落同一 AgentLoopWorker(verl interleave
  分块 dispatch,`train_batch_size % agent_num_workers == 0` 已强校验——research
  shared_task 去重同一前提,生产已验证)。发号 = per-(step,task) 进程内计数器
  (`_next_replica_index`,与 `research_tasks` 同款 dict;读写间无 await 故对同一
  事件循环上的协程原子);号 ≥ `rollout_n` 走 expert 分支。
- **expert rollout = `run_episode(guidance=doc)`**:现成入口,doc 注入走
  `inject_guidance`(PG-7 GUIDANCE_HEADER 单一真源,与 teacher prompt / 研究会话
  executor episodes 同构)。**这 4 条的 won 率同时兑现 PG-7 的文档质量信号**;
  曲线 `rollout_extra/sft_won/mean ÷ sft_role/mean` = 文档注入成功率。
- **研究失败(doc=None)/ expert episode 环境错**:该 slot **保持 expert 身份、全零 mask**,
  什么都不贡献,并上 `sft_missing_no_doc` / `sft_missing_env_error` 曲线。
  **不再降级为普通 bare 副本**(2026-07-26 外部审阅指出的偏差):研究失败与任务难度相关,
  转成 bare 会让 SFT 臂在**恰恰是难任务上**多拿 on-policy 数据,与对照臂的样本构成系统性
  不同——"SFT 臂更稳"就可能只是它多训了几条。
- **拒收样本为何不置空 response**:`AgentLoopOutput.as_dict` 有
  `rm_scores[-1] = reward_score`,空 response 会 IndexError;全零 mask 则是训练侧
  已经承受过的形态(hang guard 取消 prompt 即产生它,空批防御 commit 已收编)。

### 3.2 SFT 样本的 bare 化重放(`core/episode.py::replay_bare_tokens`)

**先厘清两种"轨迹"的区别(否则极易混淆)**:
- **OPD 的 teacher 与 student 共用同一条轨迹**——学生自己 rollout 出来的那条。teacher 取
  `[system+doc] + 学生已生成的前 t 个 token`,student 取 `[system] + 同样这 t 个 token`,
  两者对**同一位置 t+1** 给分布,KL 就在这些位置上算。即 **teacher 被迫在学生的前缀
  (含错误前缀)上表态**——这正是弱模型失效的机理入口。
- **SFT 臂的 expert rollout 是另一条完全不同的轨迹**:policy 带着 doc 自己走出来的、成功的
  那条。它的价值恰恰在于**不是**学生的轨迹——它落在学生 on-policy 分布够不着的正确中段
  状态上(framing 探针证明的 on-policy-only 天花板的解法)。

guided 轨迹是 doc 条件下产生的,直接拿来 SFT 会教出"有 doc 时做对",与内化目标相反
(PACT 明确:必须剥离特权上下文)。**重放无需重新 tokenize**,依据是:guidance 只存在于
system message(即只存在于 `prompt_ids`);而 response 流的每一段——生成 token(模型输出)、
turn glue 与 post_obs(模板常量)、observation(环境返回)——**没有一段依赖 system 里
有没有 doc**。所以 guided 轨迹的 response 流接到裸 prompt 后,仍是一个合法的、与学生
rollout 同构的序列。故重放 = 原样接到**重新渲染的裸首轮 prompt** 上,并断言两个 prompt 的
生成前缀尾部一致(与特权 teacher prefix 同款不变量)。
`test_replay_response_is_byte_identical_to_a_bare_rollout_of_the_same_text` 锁的正是
"构流不依赖 system prompt"这条性质(固定生成文本、对比两种 prompt 下的构流结果),
**不是**"guided 与 bare 会走出同一条轨迹"——真如此的话 doc 就毫无价值了。

### 3.3 三道过滤(`core/expert.py::build_expert_sample`,verl-free 故可单测)

1. **未 won** → 拒(模仿失败样本就是在教失败);
2. **轨迹自述引用文档**(`mentions_guidance`:`task guidance` / `the guidance` /
   `guidance says` / `guidance document`)→ 拒。bare 条件下这是幻觉式引用(模型引述一份
   它看不见的文档)。**这是本项目的启发式,尚无实证背书**(见 §1.2 的引用更正),
   故意做宽:4 条里误杀 1 条只损失一个样本,漏放则污染行为;误杀率由 CD-006 首验回报,
   本身也应进消融。
3. **重放超序列预算** → 拒。
拒收即 `response_mask` 全 0,并分别计数上报(`sft_won`/`sft_accepted`/`sft_dropped_ref`)。

### 3.4 隔离(四处,缺一即污染)

| # | 位置 | 若不做会怎样 |
|---|---|---|
| 隔离-1 | **TQ put 清空 expert 行的 `response_mask`/`loss_mask`**(`agent_loop_tq`) | expert token 进 OPD/PG 分子,且进全局 `batch_num_tokens` 分母 ⇒ **开 SFT 臂会顺带缩放 OPD loss**,A2/A3 臂与 A0 不可比 |
| 隔离-2 | **独立 GRPO 组**(`_compute_advantage` 给 uid 加 `:sft` 后缀) | expert 行 ~96% won 抬高任务组均值 ⇒ **压缩成功 bare 行的正 advantage、放大失败 bare 行的负 advantage**(二元 reward 下组内只要有失败,成功样本的 advantage 必为正;全成功时为 0,不会变负——此处此前写"拿到负 advantage"是错的,外部审阅 2026-07-26 指出) |
| 隔离-3 | **行为指标只读 on-policy 行**(`_compute_metrics` 的 `on_policy_mask`:`compute_data_metrics` / 方差代理 / `num_turns` / 除 `sft_*` 外的 reward_extra) | **`critic/score/mean` 被抬高**——而它是判停与起飞判据的主曲线。吞吐/计时仍计全部行(确实耗算力) |
| 隔离-4 | **观测层**:显微镜跳过全零 mask 行;`collect` 不索引 `sample_role=="sft"` 的 dump;viz 全部聚合查询过滤 role(schema 加列 + `PRAGMA user_version` 迁移),但 step 下钻仍列出并打 `expert` 标签 | 显微镜:span-vs-mask 完整性自检**误报 mismatch**(CD-006 每次 resume 的首验判据);viz:score/曲线/完整性审计全被抬高 |

### 3.5 teacher 打分跳过(算力)

expert 行 `teacher_mode="sft"`:fsdp 打分入口直接零填充 teacher_ids/logprobs 而**不做
前向**(省 4/12 的打分算力;这些值被隔离-1 的零 mask 全部屏蔽,无人读取)。vLLM 回退
后端不改——它会按 bare 语义给这些行打一次分,浪费但正确,**回退路径零风险**优先。

## 4. 代码触面(已实现;verl 改动走真源仓库 → 重导出)

verl(真源 `../verl` trace-opd,commit `a61c654` + `93a0109`):

| # | 位置 | 改动 |
|---|---|---|
| V1 | `workers/config/distillation.py` | `rl_loss_coef`/`sft_loss_coef`(默认 0)+ 校验(非负有限;与 `use_task_rewards` 互斥——同一项两种权重口径) |
| V2 | `trainer/distillation/losses.py` | `distillation_ppo_loss` 三项组合;新增 `sft_ce_loss`(`sft_mask` 选中的 token 上 `-log_prob`,同 `agg_loss` 口径);显微镜跳过全零 mask 行 |
| V3 | `trainer/ppo/v1/agent_loop_tq.py` | TQ put 处切分:`sft_mask` = expert 行的真 mask,`response_mask`/`loss_mask` 清零(隔离-1) |
| V4 | `trainer/ppo/v1/trainer_base.py` | `_mixed_objective_enabled` + `_compute_advantage` 的 uid `:sft`(隔离-2);`_compute_metrics` 的 `on_policy_mask`(隔离-3);`distillation_only` 在任一臂开启时关闭 |
| V5 | `trainer/distillation/teacher_topk_fsdp.py` + `experimental/agent_loop/agent_loop.py` | `teacher_mode="sft"`:零填充不前向 |

**已存在、无需动的**:GRPO advantage 全链路(`adv_estimator=grpo` 已配置且每步在算)、
reward 通道(`reward_score = won` → rm_scores → token_level_rewards)、old_log_prob
阶段(V1 主循环无条件跑)、`ppo_loss`(上游 vanilla + clip)。
**RL 臂本质上只是把已在计算的量接进 loss。**

trace_opd 侧:

| # | 位置 | 改动 |
|---|---|---|
| T1 | `common/config.py` | `mix_rl_weight` / `mix_sft_weight` / `mix_sft_rollouts`(默认 0/0/4)+ 校验(负值、`mix_sft_weight>0` 却无 expert rollout 或无 research 均拦) |
| T2 | `core/episode.py` | `replay_bare_tokens()`(§3.2) |
| T3 | `core/expert.py`(新) | `build_expert_sample()` 三道过滤 + 拒收置零(**verl-free,故可单测**) |
| T4 | `core/types.py` | `mentions_guidance()` |
| T5 | `train/agent_loop.py` | `_next_replica_index` / `_expert_rollout` / expert 分支与 doc=None 降级 / dump 加 `sample_role` |
| T6 | `train/curve.py` | `_MIXED_SFT` 四键(受批级常量 gate,保同质性契约) |
| T7 | `scripts/launch_train.py` | `rollout.n = rollout_n + mix_sft_rollouts`(SFT 臂开时);两系数转发 |
| T8 | `analysis/collect.py`、`viz/{index,queries,static/app.js}` | 隔离-4(collect 不索引 expert dump;viz schema 加 `sample_role` + `PRAGMA user_version=2` 迁移 + 聚合过滤 + 下钻打标) |
| T9 | `configs/trace.yaml` | 三键 + 消融臂注释 |

测试:`tests/test_mixed_objective.py`(11)+ `test_viz_index.py` 隔离用例 + 
`test_vendored_verl.py` 两条执行路径断言;全量 **152/152**。

## 4.5 开关契约:关闭一个臂 = 不产生该臂的任何计算/存储(用户要求,非"系数置 0")

**每个臂独立可关,且关闭时该臂特有的开销全部消失**,而不是"照跑再乘 0"。逐项:

| 臂 | 关闭时不发生的事 | 由什么 gate |
|---|---|---|
| **SFT** | ①**不多跑 4/12 的 rollout**(最大一笔,约 1/3 的 rollout 墙钟)②agent loop 不发副本号、不进 expert 分支、不做重放与过滤 ③**TQ 不分配也不传输 `sft_mask`**(否则每行一个全零 [R] 张量,每 step 数十 MB 纯浪费)④loss 不调 `sft_ce_loss` ⑤`distillation_only` 保持 True(verl 因此跳过 `log_probs` 计算) | ①launcher 的 `rollout.n` 表达式;②`sft_arm` 短路(权重是第二个条件);③**`sample_role` 标记只在该臂开启时盖章**,TQ 侧以"该字段是否存在"为开关;④⑤系数判定 |
| **RL** | ①不调 `ppo_loss` ②不做 `global_batch_info` 快照/恢复 ③`distillation_only` 保持 True | 系数判定 |

**两笔属于 verl 主循环、无条件发生的既有开销**(改动前的纯 OPD 也在付,不由本机制引入,
但如实记录):
- `_compute_advantage`(step 7)每步都算 GRPO advantage。CPU 侧张量操作(毫秒级),且
  `advantages` 还供曲线使用 ⇒ 不值得关。
- `_compute_old_log_prob`(step 4)是**整批一次额外前向**(`log_prob_max_token_len_per_gpu`
  就是给它的预算)。纯 OPD 不需要 `old_log_probs`,但该阶段**顺带产出 `actor/entropy` 曲线**
  ——崩塌监控的关键信号 ⇒ 关掉会失去它。若将来要省这一趟前向,须先把 entropy 曲线改由
  训练前向承担(`calculate_entropy=True` 已开),属独立优化课题,**不在本机制范围**。

**一条工程教训入档**:新增 per-token 字段必须同步 `construct_minimal_padding_template`
——它 clone 源行全部字段后只覆盖一份固定清单,漏掉的字段会让合成的 padding 行携带
别的行的全长张量。`sft_mask` 因此曾会崩溃或虚增 `sft/expert_tokens`(已修 + 形状断言
+ vendored 断言防回归)。

## 5. 观察与判据

- **显微镜适配**:microscope 行加 `role` 字段;SFT 样本 KL=0 天然不占蒸馏地形,
  分析侧按 role 分层。
- **RL 臂健康**:`actor/pg_loss`、advantage 分布、clip 率、entropy(RL 推高 entropy
  是预期方向,防坍缩);**GRPO 已知边界:全组同 reward(8 条全败/全胜)时 advantage=0,
  低成功率早期 RL 信号稀疏**——1.7B 起步成功率 ~5%,预计前期 RL 臂大多任务零信号,
  SFT 臂正是为此兜底(guided 96% won 率保证 SFT 样本供给稳定)。
- **SFT 臂健康**:`sft_loss` 曲线、guided won 率(=文档质量信号)、过滤计数;
  bare 轨迹对 guided 轨迹的行为吸收(by-family 行为指标,CD-006 现有口径)。
- **判停组合签名不变**(design §8),新增:sft_loss 持续上升 + score 不动 = 专家分布
  与 policy 脱耦(SFT 失效信号);entropy 骤降 + maxrep 升 = 崩塌签名照旧。

## 6. 消融矩阵(论文 C 系 claims 的机制扩展)

| 臂 | w_opd | w_rl | w_sft | 回答什么 | 必跑? |
|---|---|---|---|---|---|
| A0 纯 OPD | 1 | 0 | 0 | 基线 | **必跑,且必须与其他臂配对重跑**(见下) |
| A1 OPD+RL | 1 | 1 | 0 | RL 锚是否足以救活弱模型 | 必跑 |
| A2 OPD+SFT | 1 | 0 | 0.5 | 专家分布注入是否独立有效(S7 判决) | 必跑 |
| A3 全混 | 1 | 1 | 0.5 | 完整机制 | 必跑 |
| **A4 纯 RL** | 0 | 1 | 0 | GRPO 裸基线 | **必跑**(见下,原列为"可选"是错的) |
| **A5 RL+SFT(无 OPD)** | 0 | 1 | 0.5 | **OPD 在弱模型上是否还有净贡献** | 必跑 |

A0-A3 构成"OPD 恒为 1、RL×SFT 做 2×2 因子实验",可估计 RL 增量(A1−A0)、SFT 增量
(A2−A0)与交互(A3−A2−A1+A0)。**但只有这四臂答不了三个问题**(外部审阅 2026-07-26):
- **A4 必跑**:没有纯 RL 基线,就无法说"混合优于单纯 RL",而那是 SDAR/SEED 那一支的主张;
- **A5 必跑**:OPD 恒为 1 意味着**从没检验过 OPD 在弱模型上是否还有正贡献**——完全可能
  是 RL+SFT 在起作用而 OPD 在拖后腿,四臂设计对此完全盲;
- **A0 必须配对重跑**,不能拿 1.7B v2 的历史数据当对照:那之间隔着代码变更、数据顺序、
  researcher 换代(27B→35B),把这些差异算进"目标函数效应"是最容易犯的错。

**执行顺序的更正**:此前写"A3 失败则 A1/A2 大概率也失败,故先跑 A3"——**该推断不成立**。
A3 完全可能因 RL 与 SFT 梯度冲突而失败,同时 A1 或 A2 单独成功。**不得以 A3 的成败决定
是否执行拆解臂。**

**三重混杂必须在报告里显式处理**(开 SFT 臂同时改变了三件事):①环境交互数与生成算力
(每任务多 4 条 rollout);②训练 batch 的序列数与动态 packing;③每个裸 token 在微批中的
权重。因此固定 step 的比较**既不是固定样本量也不是固定算力**。论文至少要同时报:
固定 task-step / 固定裸 rollout 数 / 固定总环境 episode / GPU-hour。

**统计单位是任务,不是轨迹**:每任务 8 条 rollout 共享同一份文档,轨迹之间不独立;
显著性按任务(必要时按 seed)聚类计算,≥3 seeds。

**专家数据是 best-of-4 后再按成功条件化的样本**,不是任务分布上的无偏专家数据。
容易被文档救活的任务会稳定贡献 SFT,难任务持续贡献零 ⇒ 只看总体成功率可能把"容易任务
被强化"读成"弱模型整体被救活"。须按任务族报告 expert 接受率、至少一条被接受专家样本的
任务覆盖率、以及有/无专家样本的任务后续裸成功率。

## 7. 风险与边界

1. **n=12 的资源账**:训练 token +50%(128→192 轨迹/step)、rollout 并发 +64 episodes、
   打分不增(sft 跳过)。1.7B 显存余量足(v2 时代 2 卡跑过);**数字以实跑为准,严禁预估入报告**。
2. **SFT 过拟合**:每 step 全新样本(非重复 buffer)+ lr 1e-6 + w_sft≤1,风险低;
   监控 sft_loss 与 bare 行为脱耦签名。
3. **引用过滤的召回边界**:词面启发式抓不住隐式引用(reasoning 复述 doc 内容但不点名)。
   一期接受(PACT 同为启发式);显微镜 token 层可后验审计。
4. **GRPO 稀疏性**(§5):低成功率期 RL 臂近零信号——这不是 bug,是 A1 消融要测的机理
   (RL 冷启动困境正是 SFT 臂存在的理由,PACT 的 curriculum 结论)。
5. **两阶段 vs 混合争议**(2604.23747):我们的消融矩阵天然覆盖(A2 先跑若干 step 再开
   RL ≈ SFT-then-RL,调度层免费获得),结论开放。
6. **PG-1/PG-5 扩展**:纯 OPD 承诺被打开,进 Proposed Global Decisions;崩塌安全带逻辑
   反而增强(RL/SFT=锚,PG-5 的输入过滤由 won 过滤天然满足)。

## 8. 分期

- **P1(本轮,✅ 完成)**:V1-V5 + T1-T9 + 单测 + vendored 重导出 + 本文档;
  默认全关(4B 主线零感知,`rl/sft_loss_coef=0` 与纯 OPD 逐字节一致)。
- **P2(CD-006 实验窗,待服务器)**:1.7B A3 首跑。**首验清单**:
  1. `rollout_extra/sft_role/mean` ≈ 4/12 = 0.333(副本发号正确);
  2. `rollout_extra/sft_won/mean ÷ sft_role/mean` ≈ 0.9 量级(文档注入成功率,
     对得上 h1 的 doc 重放 8/8 与 CD-006 的 guided 96%);
  3. `sft/expert_tokens` > 0,`sft/loss`(= 每微批 expert token 平均 CE,跨 rank SUM
     聚合)有限且随训练下降;
  4. `critic/score/mean` **不因开臂而跳变**(隔离-3 生效的判据——对照同 step 的
     A0 臂或开臂前后一步);
  5. `actor/pg_loss`、advantage 分布、clip 率(RL 臂);低成功率早期 GRPO 组内
     全败 ⇒ advantage=0 ⇒ RL 信号稀疏,**这是预期而非 bug**(SFT 臂正为此兜底);
  6. 显微镜 `mask_mismatch` 仍为 0(隔离-4 生效)。
  超参微调(w_sft 衰减调度、SDAR 式 gate)按首跑证据立项,不预先做。
- **P3(论文期)**:消融矩阵全跑 + 与 SDAR/SEED 锚点对比。
