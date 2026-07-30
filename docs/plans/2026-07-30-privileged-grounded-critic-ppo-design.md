# 特权文档条件化 Grounded Critic + PPO：最终方案机制设计

> 状态：机制设计 v1.2；协议已定稿，代码实现待开始，当前尚不可直接 launch（2026-07-30）
> 适用对象：TRACE-OPD / ALFWorld / 4B actor 方案探索线
> 设计性质：独立于 OPD 的新 RL 分支；本文是该分支的权威机制规格
> 配套材料：[研究判断与可行性备忘录](../privileged-critic-research-and-feasibility.md)、
> [OPD 显微镜任务说明](../opd-microscope-analysis-taskbrief.md)

## 0. 执行摘要

本方案验证一个独立于 OPD 的假设：长程 LLM Agent 的 outcome-only actor–critic 难以
高效学习，不一定只是因为没有足够 reward，也可能因为 bare critic 必须从长历史、部分
可观察状态和稀疏终局结果中独自重建任务结构。TRACE-OPD 的 researcher 已能通过额外
compute、真实环境实验和失败复盘，为每个训练 batch 中的每个任务生成一份绑定当前
policy version 的在线特权文档。该文档可以作为 critic 的训练期状态特征，降低 return
prediction 的条件不确定性和表征难度。

本文是 **implementation-ready experiment specification**，不是“代码已经实现”的声明。
当前仓库尚无新分支所需的独立配置、launcher、critic dual-view routing 和 gate/statistics
脚本；必须完成 §13–§14 并生成通过校验的 resolved manifest 后，状态才能从
protocol-complete 变为 launch-ready。

正式机制是 **Privileged Grounded Actor–Critic**：

- bare actor 只读取原始任务与多轮交互历史；部署接口不变；
- researcher 对当前任务和当前 policy 进行独立在线实验，产出文档 \(z\)；
- C1 critic 读取 \(z\) 与 bare student 的真实前缀，C0 critic 不读取 \(z\)；
- 两个 critic 都只用对应 bare student rollout 的真实环境结果
  \(G=\mathbf 1[\mathrm{won}]\) 训练；
- critic 不生成 teacher next-token distribution，不给 semantic step score，也不把文档
  自评混入 reward；
- actor 使用 PPO/GAE；第一版取 \(\gamma=\lambda=1\)，因此特权 critic 首先是一个
  grounded、action-independent 的更强 baseline；
- 采用 VERL 原生独立 critic worker，actor 与 critic 参数、optimizer 和梯度完全隔离；
- actor 和 critic loss 均采用 **seq-mean-token-mean**；
- 正式训练前进行 8 个 online critic-only warmup batch；若任一臂未通过共同安全门，
  两臂一起延长到 16 个 batch；
- warmup 后进行严格计数的 40 个 actor update；只比较 C0-fast 与 C1-fast，不增加
  shuffled-document 第三臂；
- actor 分叉后用双方共同评分的 held-out trajectory union 检验 value，避免把不同
  on-policy rollout 难度误认成 critic 优劣；
- 主终点是固定验证集上的 bare actor success，而不是 critic loss、researcher 成功率
  或训练 batch score。

第一版要回答的严格命题是：

> 在 reward、actor 输入、rollout budget、初始化和 PPO 配方完全相同时，matched online
> privileged document 是否能使 grounded critic 更准确，并进一步让 bare PPO actor
> 获得更高、更稳定的环境成功率？

它不试图同时解决 thinking 长度、环境步数、context overflow、反事实 action credit 或
semantic process reward。那些问题可以在本假设通过后开独立实验线，不能混入首轮因果
比较。

---

## 1. 背景、问题诊断与路线切换

### 1.1 为什么不把新方案继续绑定在 OPD 上

此前 privileged-context OPD 的基本做法是：student 在 bare 前缀下产生实际 token
序列，带特权文档的 self-teacher 在同一 student 序列上提供 next-token distribution，
再逐 token 对齐 student 分布。实验表明这条路径可以提高任务分数，但也出现了持续的
thinking 膨胀和上下文耗尽。

**seq-mean-token-mean** 修复了一个真实的权重问题：长失败轨迹不再仅凭 token 数多而在
loss 中占更大总权重。但它只改变样本聚合口径，不能修复目标错位。带文档 teacher 与
bare student 的输入语义并不对等；teacher 在 student 自己不会自然到达的错误前缀上，
可能持续上调“语义上合理的推理叙述”，而不是直接优化尽快完成环境任务。逐 token KL
由此可能奖励更多被 teacher 认可的 thinking，却不保证更快行动或更高终局成功。

新方案不再把 researcher 文档用作 token teacher 的特权前缀，而是把它降格为 critic
的训练期辅助状态。环境真实 return 保持为唯一监督来源。这样既保留了文档中的任务理解，
又删除了“teacher 语义认可等于任务 reward”的核心错位。

### 1.2 新路线的研究直觉

原始 bare critic 必须直接学习：

\[
h_t\longrightarrow G,
\]

其中 \(h_t\) 是冗长、多轮、部分可观察的交互历史，\(G\) 通常只有终局 0/1。它需要同时
推断目标结构、当前进度、关键前置条件、失败恢复路径和剩余可解性。

researcher 则已经通过额外 compute 完成了部分推断：

\[
(x,\pi_k,\text{independent experiments})
\longrightarrow z_{k,i}.
\]

因此 C1 学习的是：

\[
(h_t,z_{k,i})\longrightarrow G.
\]

这不增加标签信息，却有机会降低 \(G\) 在给定输入下的条件不确定性，并把隐含任务结构
变成更容易利用的语义变量。

### 1.3 角色命名：本方案没有 token teacher

本文固定使用以下术语：

| 角色 | 输入 | 输出 | 是否直接训练 actor |
|---|---|---|---|
| actor / student | bare task 与真实交互历史 \(h_t\) | response token / 环境动作 | 是 |
| researcher | 任务、当前 policy 版本、独立实验环境 | 在线文档 \(z_{k,i}\) | 否 |
| critic | C0: bare prefix；C1: 文档 + bare prefix | 每个有效 response token 的 value | 通过 advantage 间接参与 |
| environment | actor 的实际动作 | observation 与终局 won/lost | 提供唯一 reward |

researcher 不是本方案中的 teacher：它不返回 next-token distribution，不评价 main
rollout 的当前 token，也不提供 value target。critic 的 target 始终来自 bare rollout
的环境真实结果。

---

## 2. 目标、非目标与可声称结论

### 2.1 第一阶段目标

1. 验证 online privileged document 是否在共同 held-out rollout 分布上提高 bare return
   prediction。
2. 验证更好的 value 是否改善 PPO advantage 的信噪比、梯度稳定性和最终 bare actor
   success。
3. 在当前 VERL 主流 PPO/GAE 实现内完成验证，尽量复用原生 standalone critic。
4. 建立严格的数据血缘、梯度隔离、逻辑计数和 prequential 评估，保证结果可审计。

### 2.2 明确非目标

第一阶段不包含：

- OPD、self-teacher KL 或任何 token distillation；
- researcher/LLM judge 生成的 semantic process reward；
- shuffled-document 训练臂；
- grounded branch-\(Q\)、candidate reranking 或 PRM；
- token cost、environment-step cost、overflow constraint；
- 以 critic score 替换环境 outcome；
- actor 与 critic 共享 backbone；
- 静态、跨 step 复用的通用文档；
- 为追求吞吐而牺牲机制完整性的压缩实现。

长度、environment steps 和 overflow 可以保留为诊断日志，但不进入首轮 reward、停止门
或主结论。

### 2.3 首轮可以与不可以声称什么

若 C1 胜过 C0，可以声称：

> 在线特权文档为 grounded value prediction 提供了可利用的增量信息，并通过更好的
> baseline 改善了有限样本 PPO 的学习效果。

不能仅由该结果声称：

- 文档给出了正确的逐步 process reward；
- critic 已学到某个动作的反事实因果贡献；
- thinking token 获得了无偏的局部信用；
- privileged critic 必然改变理想 policy gradient 的期望方向；
- 该机制已经解决长度膨胀。

---

## 3. 数学定义与机制边界

### 3.1 决策变量

令一个 Agent episode 的第 \(t\) 个 assistant response 为：

\[
\begin{aligned}
h_t&=(x,o_0,y_0,u_0,\ldots,o_t),\\
y_t&=(y_{t,1},\ldots,y_{t,L_t})
\sim\pi_\theta(\cdot\mid h_t),\\
u_t&=P(y_t),
\end{aligned}
\]

其中：

- \(x\) 是原始任务；
- \(o_t\) 是环境 observation；
- \(y_t\) 包含 thinking 与 action serialization；
- \(P\) 是 parser；
- \(u_t\) 是环境真正执行的 command。

训练 reward 只有：

\[
G_e=\mathbf 1[\text{episode }e\text{ won}].
\]

所有未成功、invalid、truncated 或达到环境上限的 episode 在本阶段都取 \(G_e=0\)。

### 3.2 在线文档

对训练 batch \(b\)、任务 \(i\)，令 \(v(b)\) 是该批实际使用的 actor policy version：

\[
z_{b,i}
\sim q_\rho
\left(
z\mid x_i,\pi_{v(b)},\xi^{\mathrm{research}}_{b,i}
\right),
\]

其中 \(\xi^{\mathrm{research}}\) 是 researcher 的独立实验随机性。文档的规范身份键是
**(batch_id, policy_version, task_id)**。运行时只有在“一个 batch 只允许一个 policy
version”的断言已通过时，才可用 **(batch_id, task_id)** 做进程内去重；落盘 metadata
和 resume 校验必须保存完整三元组。这对 warmup 很重要：warmup 中 policy version 保持
0，但每个新 batch 仍须重新研究并生成 fresh document。同一三元组下的 8 条 main
rollout 共享一份文档。

### 3.3 Actor 与两个 critic

actor 在 C0/C1 中完全相同：

\[
\pi_\theta(y_{t,r}\mid h_t,y_{t,<r}).
\]

C0 critic：

\[
V_{\phi_0}(h_t,y_{t,<r}).
\]

C1 critic：

\[
V_{\phi_1}(z_{b,i},h_t,y_{t,<r}).
\]

critic 为 response 中每个有效 assistant token \(y_{t,r}\) 预测当前 token 采样前的
success value。

### 3.4 Action-independent baseline 的合法性

文档可以与 main rollout 并发生成，不要求在墙钟时间上先完成。真正的硬条件是文档的
数据血缘不能依赖对应 main rollout 的当前或未来动作、observation、outcome：

\[
z_{b,i}\perp y_{b,i,j,t}
\mid x_i,\pi_{v(b)},h_{b,i,j,t}.
\]

在 actor 不读取 \(z\) 且当前动作不影响 \(z\) 时：

\[
\begin{aligned}
&\mathbb E_{a_t\sim\pi_\theta}
\left[
\nabla_\theta\log\pi_\theta(a_t\mid h_t)
V_\phi(h_t,z)
\right]\\
&=V_\phi(h_t,z)
\nabla_\theta\sum_{a_t}\pi_\theta(a_t\mid h_t)=0.
\end{aligned}
\]

因此它可以作为训练期 asymmetric baseline。researcher 使用同一 policy 的独立实验
并不破坏该性质；读取 main rollout 或根据本批失败反向改写文档会破坏它。

这个零期望证明只覆盖未做 trajectory-length reweighting、batch whitening 和 PPO
clipping 的参考 score-function estimator。首轮实际使用的
**seq-mean-token-mean** 会除以由采样行为决定的 trajectory 长度，VERL
**masked_whiten** 的 batch mean/std 也依赖本批动作与 return，PPO clipping 进一步改变
surrogate。因此不能把上式外推为“实际 whitened PPO 更新严格无偏”。本实验检验的是：
在共同的、允许存在有限样本偏差的 VERL surrogate 下，C1 相对 C0 是否产生更好的净效果。

### 3.5 \(\gamma=\lambda=1\) 的准确含义

VERL GAE 为：

\[
\delta_t=r_t+\gamma V_{t+1}-V_t,
\qquad
A_t^{\mathrm{GAE}}
=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}.
\]

第一版取 \(\gamma=\lambda=1\)，且 reward 只在终局出现，因此中间 value 相消：

\[
A_t^{\mathrm{raw}}=G_e-V_t,
\qquad
R_t=G_e.
\]

这是一条故意保守的机制线：critic 改善 baseline 和有限 batch 下的估计质量，不用自己的
预测创造额外 reward。若未来改为 \(\lambda<1\)，value 会通过 bootstrap 更强地改变逐步
advantage，可能降低方差，也可能因 value 误差引入更大偏差，必须另开实验。

---

## 4. 4B Actor 冷启动

### 4.1 为什么 warmup 之前必须先有可学习的 actor

若 bare actor 几乎处处失败，grounded critic 的正确输出就是接近 0。critic warmup
不能凭文档创造正标签，也不应被要求这样做。全零数据只会把两个 critic 都推向低常数，
无法检验文档是否改善成功概率建模。

因此 actor 冷启动是本方案的前置条件，不是特权 critic 的一部分。

### 4.2 先锁 split，再产生数据

在任何大模型采样、SFT/RFT 或 checkpoint 选择之前，固定五个互不重叠的 manifest：

| Split | 用途 | 是否产生梯度 | 是否可用于选 checkpoint |
|---|---|---:|---:|
| cold-start-train | 大模型采样与 4B SFT/RFT | 是，仅冷启动 | 否 |
| cold-start-dev | 格式率与 10%–25% success 窗口选择 | 否 | 是，仅冷启动 |
| RL-train | warmup、online research 与 PPO | 是 | 否 |
| critic-audit | 两个 critic 的共同 held-out value 审计 | 否 | 否 |
| RL-eval seen/unseen | C0/C1 固定 bare validation 与主终点 | 否 | 否 |

前四者从 ALFWorld train pool 预先做 task-family 分层划分；官方
**valid_seen=140 / valid_unseen=134** 全量保留为 RL-eval，不抽样。任务 ID、gamefile、
环境 seed、family 和 split hash 在采样前冻结。不得使用 critic-audit 或 RL-eval 轨迹筛选
SFT/RFT 数据、调整 warmup 或选择 checkpoint；任何 task ID 跨 split 重叠都使 run
manifest 无效。

### 4.3 数据产生方式

使用更大的 executor model 在与最终 actor 完全相同的 bare 接口下采样：

- 输入只有原始任务、真实 observation 与标准 executor scaffold；
- 不输入 researcher 文档、ground truth plan 或 teacher explanation；
- 让大模型在真实环境中执行完整 episode；
- 只用环境 **won** 判断轨迹是否成功；
- 保存完整多轮 message、thinking/action 格式、环境 observation 与 parser 结果；
- 成功轨迹用于 SFT 或 rejection fine-tuning；
- 数据按任务 family 和难度去重、平衡，防止只学会简单任务格式。

### 4.4 冷启动目标

选择一个 4B checkpoint，使其：

- parser/格式有效率至少约 95%；
- bare validation success 约 10%–25%；
- 已具备基本工具使用和任务执行能力；
- 尚未饱和，给后续 RL 留出可观测提升空间。

不选择成功率最高或已经接近上限的 checkpoint。优先选择第一个进入目标窗口的 checkpoint。
若一次 SFT 后仍低于窗口，可进行一次基于 4B 自身成功 rollout 的 RFT；若仍近乎全零，
停止 critic 实验并继续改善冷启动，而不是改用 teacher 分数伪造 value label。

### 4.5 冷启动验收

cold-start-dev 至少固定 256 个按 family 分层的任务，并以每任务一次 bare greedy episode
验收。启动 warmup 的硬门是：格式有效率至少 95%，point success 位于 10%–25%，至少
32 条真实成功且覆盖至少 3 个 task family。若样本数不足以同时满足计数门，则扩大 dev，
不能用重复 rollout 冒充独立任务。

验收后锁定：

- actor checkpoint hash；
- tokenizer、chat template、scaffold 和 parser 版本；
- 训练/验证任务 split；
- train sampling 参数与 validation greedy 参数；
- overall 与每类任务的格式有效率和 success；
- 上述成功数、family 覆盖及其原始 episode manifest。

---

## 5. 实验臂与唯一变量

### 5.1 两个且仅两个训练臂

| Arm | Actor 输入 | Critic 输入 | Critic target | Reward |
|---|---|---|---|---|
| C0-fast | bare \(h\) | bare \(h,y_{<t}\) + 固定 no-doc marker | bare rollout \(G\in\{0,1\}\) | 同一真实 outcome |
| C1-fast | bare \(h\) | matched online \(z\) + bare \(h,y_{<t}\) | 同一 bare rollout \(G\in\{0,1\}\) | 同一真实 outcome |

不增加 shuffled-document arm。C0 已回答“没有文档时标准 critic + PPO 如何表现”；首轮
资源只用于判断 matched online document 的端到端净增益。

### 5.2 所有共同变量

C0/C1 必须共享：

- 4B actor 冷启动 checkpoint；
- critic backbone checkpoint 与 value-head 初始化 seed；
- 训练任务顺序、每步任务数和每任务 rollout 数；
- researcher 模型、prompt、实验 budget 和调用拓扑；
- actor prompt、rollout temperature、parser 和环境；
- 终局 0/1 reward；
- GAE、advantage whitening、PPO clipping 和 loss aggregation；
- actor/critic optimizer 配置；
- critic warmup 规则与 actor update 数；
- validation split、解码、频率和统计脚本；
- critic-audit split、source rollout seeds、频率和统计脚本；
- checkpoint/resume 和日志字段。

“唯一机制开关”指两臂在 **actor 尚未分叉时的初始干预**；正式训练后 actor、rollout 和
在线文档会成为该干预的下游中介，预期不再逐位相同。唯一初始机制开关为：

~~~text
privileged_critic.mode = none | matched_online_document
~~~

### 5.3 C0 也运行 shadow researcher

C0 仍然对每个训练任务执行与 C1 相同的在线 research，保存文档和元数据，但 critic
忽略文档。这样 researcher 带来的调度、并发、环境调用和系统负载在两臂中一致。效率
不是首轮主指标，但不能让系统拓扑成为隐藏变量。

### 5.4 严格梯度隔离

VERL 已原生支持独立 critic worker，因此第一版直接选择 standalone critic：

- actor 与 critic 是两个独立 4B 实例；
- critic 从 actor 冷启动 checkpoint 复制 backbone 初始化，但之后不共享参数；
- value loss 只更新 critic；
- actor loss 只更新 actor；
- researcher 无梯度；
- critic value 和 advantage 在 driver 上作为 detached tensor 进入 actor loss。

不采用共享 backbone + value head。共享实现会让 privileged critic loss 通过共享参数直接
改写 actor，把“更好的 baseline”与“文档通过共享权重被蒸入 actor”混在一起。

---

## 6. 在线 Researcher 文档生命周期

### 6.1 每个 training batch 重新生成

对训练 batch \(b\) 及其冻结的 actor policy \(\pi_{v(b)}\)，每个训练任务 \(i\) 生成：

\[
z_{b,i}=\operatorname{Research}(x_i;\pi_{v(b)},\text{fresh experiments}).
\]

文档以 **(batch_id, policy_version, task_id)** 为规范身份，是在线产物而不是固定手册，
也不跨 batch 读取旧 archive。即使 warmup 中 policy 不变，或同一 task 再次出现，新
batch 也必须使用 fresh research seed 重新生成文档。

### 6.2 与 main rollout 并发，但必须独立

保留当前 **agent_loop.py** 的并发结构：research 与 bare student rollout 可以同时开始，
本步 policy 权重保持冻结，critic 计算前等待文档完成。这满足效率和统计要求，前提是：

- research episode 使用独立环境实例和 RNG；
- researcher API 不接收 main rollout payload；
- main rollout 不接收文档；
- researcher archive 与 main rollout archive 使用不同 lineage ID；
- 文档不能在看见 main outcome 后二次改写。

这里的“pre-action”是因果/数据血缘意义，而非墙钟先后意义。

### 6.3 一个 task-step 一份文档

每步 16 个任务、每任务 8 条 bare rollout：

~~~text
(batch_id b, task i; policy_version v(b))
    ├── one online document z(k,i)
    ├── bare rollout j=1
    ├── ...
    └── bare rollout j=8
~~~

8 条 rollout 共享同一文档，提供同任务、同文档下的随机行为覆盖。统计上它们是相关样本；
置信区间和 calibration 分析必须按 task/document cluster 处理。

### 6.4 分支漂移是机制的一部分

warmup 期间 C0/C1 actor 相同，任务和 RNG 配对时可获得近似相同的文档与 main rollout。
正式 actor update 后两个 policy 自然分叉：

\[
z^0_{k,i}\sim q(z\mid x_i,\pi^0_k),
\qquad
z^1_{k,i}\sim q(z\mid x_i,\pi^1_k).
\]

此后文档不同不是实验污染，而是 online closed-loop treatment 的组成部分。C0 仍生成
自己的 shadow 文档；C1 使用与自己当前 policy 匹配的文档。不得把另一臂或旧 policy
的文档搬来使用。

### 6.5 文档失败与缺失

researcher 单样本失败不能删除或替换 main rollout。处理规则：

- C0 critic 输入固定标记 **[NO_PRIVILEGED_DOCUMENT]**；
- C1 正常时输入完整 matched document；
- C1 research 失败时输入 **[DOCUMENT_UNAVAILABLE]**；
- 记录 failure reason、archive、document hash 和 availability；
- 不因 research 失败额外重采一个更容易任务；
- 不静默把失败样本标成正常 matched-document。

缺失标记本身可能携带“researcher 对该任务也困难”的特权信息，第一版允许 C1 使用，
但必须分别报告 available/unavailable 子集。

### 6.6 文档与 policy provenance

每份文档至少保存：

~~~text
batch_id
task_id
policy_version
critic_update_id
actor_checkpoint_hash
researcher_model/version
research_prompt_hash
research_seed / episode_ids
environment/gamefile/version
created_at
document_hash
research_archive_path
document_available
failure_reason
~~~

---

## 7. Critic 输入、token 对齐与监督

### 7.1 双视图 batch

同一条 bare trajectory 构造 actor view 与 critic view：

~~~text
actor view:
    [bare prompt h0]
    [student actual multi-turn response stream y]

C0 critic view:
    [fixed critic header]
    [NO_PRIVILEGED_DOCUMENT]
    [same bare prompt/history]
    [same response stream y]

C1 critic view:
    [fixed critic header]
    [matched online document z]
    [same bare prompt/history]
    [same response stream y]
~~~

两个 view 的 response token IDs、response mask、turn spans 和 environment outcome 必须
逐位相同。只有 critic prompt/context 不同。

### 7.2 Critic prompt 构造

使用普通文本 delimiter，不新增 tokenizer special token：

~~~text
<critic_context version="1">
<privileged_document>
... matched document / explicit marker ...
</privileged_document>
<bare_actor_context>
... original task scaffold and observations ...
</bare_actor_context>
~~~

critic 是 token-classification/value-head 模型，不需要生成自然语言分数。header 的作用
只是稳定区分文档与 actor history。C0/C1 使用同一模板。

现有 **teacher_prompt_ids** 的渲染和“generation boundary 尾部一致”检查可以复用思想，
但必须新增语义独立的 **critic_prompt_ids**；不得继续用 teacher 字段承载新机制。

### 7.3 每个 response token 的 pre-action value

critic forward 对完整 causal sequence 输出逐 token scalar。VERL
**no_padding_2_padding()** 会把输出左移一位：

\[
\text{value assigned to }y_r
=f_\phi(z,h,y_{<r}),
\]

而不是 \(f(z,h,y_{\le r})\)。这是 action-independent baseline 的必要条件。

需要用单测确认：

- 改变 token \(y_r\) 本身，不得改变分配给 \(y_r\) 的 value；
- 改变未来 token，不得改变更早 token value；
- 改变过去 prefix，应允许后续 value 改变。

### 7.4 Mask 规则

沿用当前 episode tokenization：

- assistant 实际生成的 thinking/action token：**response_mask=1**；
- observation、tool glue、chat template framework token：**response_mask=0**；
- padding：0；
- 截断后不存在的 token：0。

第一版保留 native token-value，不采用早期研究备忘录中的“只在 turn endpoint 训练”
候选。**turn_spans** 仍用于分深度诊断，但不改变首轮 loss mask。

### 7.5 Value target 与 loss

在 \(\gamma=\lambda=1\) 下，每个有效 response token 的 return 为同一终局结果：

\[
R_{e,r}=G_e\in\{0,1\}.
\]

value loss 使用 PPO clipped MSE：

\[
\begin{aligned}
V^{\mathrm{clip}}_{e,r}
&=\operatorname{clip}
(V^{\mathrm{new}}_{e,r},V^{\mathrm{old}}_{e,r}-\epsilon_V,
V^{\mathrm{old}}_{e,r}+\epsilon_V),\\
L_{e,r}
&=\tfrac12\max\left[
(V^{\mathrm{new}}_{e,r}-G_e)^2,
(V^{\mathrm{clip}}_{e,r}-G_e)^2
\right].
\end{aligned}
\]

聚合采用：

\[
L_V
=\frac1B\sum_{e=1}^{B}
\frac1{|M_e|}\sum_{r\in M_e}L_{e,r}.
\]

即 **seq-mean-token-mean**：每条 trajectory 在 critic loss 中等权，trajectory 内有效
token 取均值。长失败轨迹不再因 token 更多而获得更高总权重。

### 7.6 上下文处理纪律

首轮优先保证完整文档和完整 bare trajectory 都能进入 critic：

- critic context capacity 必须在 smoke test 中按真实最长样本验证；
- 不允许静默左截断文档或历史；
- 不允许 actor view 与 critic view 的 response 尾部错位；
- 若单样本文档不可用，使用显式 unavailable marker 并记录；
- 系统性容量不足属于启动失败，应调整 critic context/并行配置，而不是让结果混入大量
  隐式 bare fallback。

这属于输入正确性要求，不是本阶段的 efficiency objective。

---

## 8. Critic Warmup：正式 policy evaluation 阶段

### 8.1 目的

VERL 会从 actor checkpoint 加载 critic backbone，并创建一个单标量 value head。value
head 初始为随机状态。若立刻更新 actor，第一批 PPO advantage 会由随机 value、有限
batch whitening 和 clipping 共同决定。

warmup 的目标是：在固定 \(\pi_0\) 下让两个 critic 学会基本的
\(V^{\pi_0}\)，尤其让 C1 学会如何联合读取文档与 trajectory prefix，然后才让 baseline
参与 actor 更新。

它不是 optimizer LR warmup，也不是离线固定文档预训练。

### 8.2 每个 warmup batch

1. actor 固定为冷启动 \(\pi_0\)；
2. 采样 16 个训练任务；
3. 每个任务进行 fresh online research；
4. 每个任务产生 8 条 fresh bare rollout；
5. 环境给真实 0/1 outcome；
6. 用更新前 critic 计算 value 和 prequential 指标；
7. 计算 return；
8. 只更新 critic；
9. actor 参数、optimizer、scheduler 均不更新；
10. 下一批重新 research、重新 rollout。

### 8.3 数据规模与相关性

固定 8 个 warmup batch：

\[
16\ \text{tasks}
\times8\ \text{rollouts}
\times8\ \text{batches}
=1024\ \text{trajectories}.
\]

真正更接近独立单元的是：

\[
16\times8=128\ \text{task-document clusters}.
\]

若冷启动 success 为 10%–25%，预期有约 102–256 条成功 trajectory。优化可以使用所有
token，但 warmup calibration、置信区间和覆盖判断按 cluster 统计。

### 8.4 初始化与 optimizer

C0/C1 使用完全相同设置：

| 项 | 值 |
|---|---:|
| critic backbone | 同一个 4B SFT/RFT actor checkpoint |
| value head | VERL/HF 原生随机初始化，同 pair seed |
| critic training | full-parameter |
| optimizer LR | 1e-5 |
| LR scheduler | constant |
| optimizer LR warmup | 0 |
| weight decay | 0.01 |
| gradient clip | 1.0 |
| value clip range | 0.5 |
| critic PPO epochs | 1 |
| loss aggregation | seq-mean-token-mean |

不额外定制零初始化 head；actor 已冻结，原生随机 head 不会污染 actor，且更贴近 VERL
标准路径。

### 8.5 就绪门

warmup 门分为“立即作废的正确性门”和“允许共同延长的 readiness 门”。

正确性门任一失败都立即停止，不用更多 batch 掩盖：

- actor 参数 hash 与 warmup 前逐位相同；
- actor optimizer step 和 scheduler step 均为 0；
- 固定 actor-logit probe 逐位相同，固定环境 validation success 相同；
- critic 参数确实变化；
- value/loss/pre-clip gradient 无 NaN 或 Inf；
- token shift、response mask 和 return 对齐测试通过；
- C0/C1 critic update 数完全一致；
- researcher/main lineage 不交叉；
- 两臂 researcher document availability 均至少 95%，且每个样本数不少于 8 个 cluster
  的 task family 至少 90%；C1 实际送入 critic 的 matched-doc rate 必须与 archive
  availability 一致。

在 batch 8 的共同 barrier 上，对每一臂分别计算以下 readiness 条件；任一臂失败则两臂
共同延长：

| 项 | batch-8 阈值与口径 |
|---|---|
| 真实成功 trajectory | 累计至少 64 条 |
| 含至少一条成功的 task-document cluster | 累计至少 32 个 |
| 出现成功的 task family | 至少 3 类 |
| pre-update raw value MSE | \(M_V^{(2)}\le\max(1.25M_{\rm const}^{(2)},M_{\rm const}^{(2)}+0.05)\) |
| 常数 baseline | \(M_{\rm const}\) 只用该预测之前的 batch outcome 估计；第一批用 cold-start-dev 成功率 |
| success/failure 排序安全性 | 最近 2 批的 sequence-balanced \(\bar V_{\rm succ}-\bar V_{\rm fail}\ge-0.05\) |
| value scale | 最近 2 批所有有效 token 的 \(P_{99}(|V|)\le5\) |

gate 脚本对“最近 2 批”的唯一计算契约如下。对每个批次 \(q\)，先在预测该批之前用所有
更早 warmup outcome 估计 \(p_q\)；只有 \(q=1\) 使用 frozen cold-start-dev success。
然后在批内先按 trajectory 做 sequence mean，得到
\(M_{V,q}\) 与 \(M_{\mathrm{const},q}=\operatorname{mean}_e(G_e-p_q)^2\)。barrier 使用两个
**批均值的等权平均**：

\[
M_V^{(2)}=\tfrac12(M_{V,b-1}+M_{V,b}),\qquad
M_{\rm const}^{(2)}=\tfrac12(M_{{\rm const},b-1}+M_{{\rm const},b}).
\]

不得先把两批 token/trajectory 池化再重新估一个常数。排序项也先逐 trajectory 取
sequence-mean value，再在最近两批合并后的 success/failure trajectory 间求差；若最近
两批任一类别为 0 条，则该项为 **undefined = readiness fail**：W8 时触发共同 extend，
W16 时触发 abort。`warmup_gate.py` 的 commit hash、输入 manifest hash、逐项中间量和最终
decision 都写入 `warmup_decision.json`，禁止两个 arm 使用不同实现。

raw MSE 是 readiness 主口径。value head 无界，因此 clipped-to-[0,1] Brier/ECE 只作
calibration 诊断，不能与 raw MSE 混称。

由于 **ppo_epochs=1** 且一个 global mini-batch 覆盖全部 128 条 trajectory，value
forward 与 old value 在同一 optimizer step 前计算，**vf_clipfrac** 预期接近 0；它和
gradient clip rate只记录，不作为 warmup pass gate。

不要求 C1 在 warmup 结束时已经胜过 C0；那是实验假设，不是启动门。

### 8.6 固定 8、共同延长到 16

- 默认：8 个 critic-only batch；
- 若任一 critic 未通过共同数值/安全门：C0/C1 都继续 warmup 到 16；
- 禁止 C0 8 批、C1 16 批的按臂早停；
- batch 16 使用同一 readiness 公式：覆盖指标在累计 16 批上计算，MSE/排序/value
  scale 仍只看最近 2 批；
- 16 批后仍失败：停止 pilot，诊断正例覆盖、alignment、LR、截断和输入，而不是强行
  更新 actor。

两臂在 batch 8 必须同时停在 checkpoint barrier。一个 pair-level gate job 读取两臂
manifest，写出唯一的 **warmup_decision.json**（pass / extend / abort）及触发指标。
两个 run 只接受同一 decision 文件后才能 resume；batch 16 再执行一次同样 barrier。

### 8.7 VERL off-by-one

当前 VERL 判断为：

~~~python
if trainer.critic_warmup > global_steps:
    # skip actor update
~~~

首个训练 batch 的 **global_steps=1**，因此 **critic_warmup=8** 实际只跳过 7 个 actor
update。若不改上游条件，逻辑上的 8 个 warmup batch 应映射为 **critic_warmup=9**。

本文首轮复用 native 条件，通过分阶段 run control 落地：

| 阶段 | trainer.total_training_steps | trainer.critic_warmup | 结束时逻辑状态 |
|---|---:|---:|---|
| W8 barrier | 8 | 9 | 8 critic / 0 actor |
| W8 pass 后正式续跑 | 48 | 9 | 48 critic / 40 actor |
| W8 失败后的 W16 barrier | 16 | 17 | 16 critic / 0 actor |
| W16 pass 后正式续跑 | 56 | 17 | 56 critic / 40 actor |

所有阶段都从上一个 committed checkpoint resume。constant LR scheduler 保证扩展总步数
不会改变已执行 update 的学习率；若以后使用非 constant scheduler，本表不再等价，必须
重新设计。objective fingerprint 固定的是“8 批、共同门控、最多延长到 16”的逻辑规则；
phase-specific 的 native total/warmup 值写入 phase manifest，并且只允许按上表变化。

实现不能只信配置名，必须记录并断言：

~~~text
W8:  critic_only_updates == 8  and actor_updates == 0
W16: critic_only_updates == 16 and actor_updates == 0
final: actor_updates == 40
~~~

---

## 9. 正式 Online PPO 闭环

### 9.1 每个逻辑 actor update 的时序

warmup 结束时 actor policy version 为 \(\pi_0=\pi_{\theta_0}\)。对第
\(u=1,\ldots,40\) 次正式 actor update：

1. **冻结 policy snapshot**：本步所有 research 和 main rollout 使用
   update 前的 \(\pi_{u-1}=\pi_{\theta_{u-1}}\)。
2. **在线 research**：每个任务产生绑定 \(\pi_{u-1}\) 的当步 matched document。
3. **bare rollout**：actor 不读取文档，每任务产生 8 条 trajectory。
4. **真实 reward**：环境产生 \(G_{k,i,j}\in\{0,1\}\)。
5. **old actor log-prob**：在 bare actor view 上计算并冻结。
6. **old critic value**：使用更新前 \(\phi_{u-1}\) 在 critic view 上计算并冻结。
7. **advantage/return**：从真实 reward 和 old value 计算。
8. **update critic**：用本批 return 更新到 \(\phi_u\)。
9. **update actor**：使用第 7 步冻结的 advantage 更新到 \(\theta_u\)，policy version
   从 \(u-1\) 变为 \(u\)。
10. **commit step**：保存 checkpoint/provenance；下一步再用新 policy 重新 research。

不能先用本批 return 更新 critic，再用更新后 value 重算同一批 advantage。那会把
in-sample value 拟合泄漏进 policy update。

### 9.2 Advantage 与 actor loss

raw advantage：

\[
A_{e,r}^{\mathrm{raw}}
=G_e-V_{\phi_{u-1}}(z,h,y_{<r}).
\]

保留 VERL 原生 **masked_whiten**：

\[
\widehat A_{e,r}
=\frac{A_{e,r}^{\mathrm{raw}}-\mu_{\mathrm{token}}}
{\sigma_{\mathrm{token}}+\epsilon}.
\]

PPO ratio：

\[
\rho_{e,r}(\theta)
=\exp\left[
\log\pi_\theta(y_{e,r}\mid h,y_{<r})
-\log\pi_{\theta_{u-1}}(y_{e,r}\mid h,y_{<r})
\right].
\]

actor loss：

\[
L_\pi
=-\frac1B\sum_e\frac1{|M_e|}\sum_{r\in M_e}
\min\left[
\rho_{e,r}\widehat A_{e,r},
\operatorname{clip}(\rho_{e,r},1-\epsilon,1+\epsilon)
\widehat A_{e,r}
\right].
\]

即 actor 也使用 **seq-mean-token-mean**。

### 9.3 首轮 PPO 配置

| 项 | 值 |
|---|---:|
| algorithm | PPO + GAE |
| advantage estimator | gae |
| \(\gamma\) | 1.0 |
| \(\lambda\) | 1.0 |
| train tasks / batch | 16 |
| bare rollouts / task | 8 |
| trajectories / actor step | 128 |
| actor PPO mini-batch | 16 tasks，VERL 内部乘 n=8 得 128 |
| critic PPO mini-batch | 16 tasks，VERL 内部乘 n=8 得 128 |
| actor PPO epochs | 1 |
| critic PPO epochs | 1 |
| actor LR | 1e-6 |
| actor LR scheduler | constant |
| actor LR warmup | 0 |
| actor clip ratio | 0.2 |
| entropy coefficient | 0 |
| entropy calculation | 开，仅日志 |
| reference KL loss | 关 |
| in-reward KL | 关 |
| actor loss aggregation | seq-mean-token-mean |
| critic loss aggregation | seq-mean-token-mean |
| train rollout temperature | 1.0 |
| actor updates | 40 |

关闭 OPD 后不需要 teacher model、teacher top-k、distillation loss 或 reference policy。
同时固定 **train_batch_size=16、rollout.n=8、actor/critic ppo_mini_batch_size=16**，并将
research-RL、mixed SFT、expert copy 等所有额外 row 通道置 0。于是每个逻辑训练 batch
必须恰好包含 16 个 task cluster、128 条 main bare trajectory；research episode 只生成
文档，不进入 actor/critic 训练 batch。

在这组配置下，**ppo_epochs=1** 且 global mini-batch 等于全部 128 条 trajectory，因此
每个逻辑 update 对 actor 和 critic 各恰好调用一次 optimizer step。dynamic micro-batch
只做显存分块和梯度累积，不能增加 optimizer/scheduler step。launcher 在每步断言：

~~~text
main_rows == 16 * 8 == 128
actor_optimizer_steps_delta  == 0 during warmup, else 1
critic_optimizer_steps_delta == 1
actor_scheduler_steps_delta  == actor_optimizer_steps_delta
critic_scheduler_steps_delta == 1
~~~

这一配方还有一个必须诚实记录的性质：在唯一一次 full-batch forward 时
\(\pi_\theta=\pi_{\rm old}\)、\(V_{\rm new}=V_{\rm old}\)，所以 actor ratio clipping 与
value clipping 通常不会真正触发，clip fraction 应接近 0。它保留 VERL PPO 数据流，但
就首轮每步而言更接近“带 learned baseline 的单步 on-policy policy gradient”，不能把
clipping 当成已经实现了有效 trust region。若未来增加 PPO epochs 或拆分 mini-batch，
那会改变优化机制，必须另开配置与比较。

### 9.4 Advantage whitening 的已知边界

**seq-mean-token-mean** 只改变最终 actor/value loss 聚合；VERL GAE 的
**masked_whiten** 仍使用所有有效 response token 计算均值和方差，所以长 trajectory
仍会在 normalization moments 中占更多权重。

首轮保留该行为，原因是要检验 privileged critic 是否能改善标准 VERL PPO，而不是同时
修改 advantage estimator。但必须保存：

- raw advantage 与 whitened advantage；
- token-weighted 和 episode-balanced mean/std；
- advantage 与 response length 的相关性；
- quantile、sign 和极值；
- actor gradient norm 与跨 batch 梯度方向稳定性。

若 C1 value 明显更准而 actor 无增益，whitening 是第一后续诊断：再开
sequence-balanced normalization 或只中心化、不除 std 的独立实验。不能在首轮结果出现
前静默改变。

---

## 10. 逻辑计数、验证、Checkpoint 与恢复

### 10.1 四个逻辑计数器

不要用一个 VERL **global_steps** 同时表示所有含义：

| Counter | 8-batch warmup 结束 | 40 actor update 结束 |
|---|---:|---:|
| batch_id | 8 | 48 |
| critic_update_id | 8 | 48 |
| actor_update_id | 0 | 40 |
| policy_version | 0 | 40 |

若 warmup 延长到 16，则最终 **batch_id=critic_update_id=56**，但
**actor_update_id=policy_version=40**。

researcher 路由、archive 和日志同时保存 **batch_id** 与 **policy_version**，避免把
warmup global step 误当作 actor 已更新。

### 10.2 Validation 按 actor update 调度

验证点：

~~~text
actor_update = 0             冷启动基线
warmup_end                   actor 应完全不变
actor_update = 5,10,...,40   正式曲线
~~~

不能简单用 **global_steps % 5 == 0**，否则 warmup 会错位。validation scheduler 必须
读取 **actor_update_id**。critic audit 另按 warmup_end 与 actor update
10/20/30/35/40 调度，不得混入 bare validation job。

### 10.3 Validation 只测 bare actor

验证时：

- 不运行 researcher；
- 不输入文档；
- 不运行 critic；
- 使用固定 seen/unseen task split；
- 固定环境 seed；
- greedy decoding：temperature 0、top-p 1、top-k -1；
- 主结果来自环境 success。

warmup 前后 deterministic validation 必须完全一致；任何变化都说明 actor 被误更新或
验证随机性未锁定。具体比较 actor checkpoint hash、固定 logit-probe tensor、274 个 task
的 command trace hash 和 success vector；参数/logit 不同是立即作废，参数/logit 相同但
环境 trace 不同则先修复 validation determinism，不进入正式训练。

### 10.4 Step transaction 与 resume

每个正式 step 视为一个事务：

~~~text
begin step(policy hash)
  -> research artifacts
  -> main rollout artifacts
  -> old value/logprob/advantage
  -> critic update
  -> actor update
  -> checkpoint + committed marker
end step
~~~

若在 actor/critic 只完成一半时崩溃，不得带着半步 optimizer 状态重新生成文档继续。恢复
只能：

1. 从上一个 committed checkpoint 完整重跑本步；或
2. 若框架支持事务级精确恢复，复用同一 doc/rollout artifacts 完成本步。

checkpoint 至少包含：

- actor model/optimizer/scheduler；
- critic model/optimizer/scheduler；
- 四个逻辑计数器；
- actor、critic、prompt、tokenizer、parser、环境 fingerprint；
- dataloader/task RNG；
- committed step marker；
- document/rollout manifest hash。

---

## 11. 实验规模、主终点与统计判据

### 11.1 Fast 对照矩阵

只跑：

| Pair | Arm 0 | Arm 1 |
|---|---|---|
| seed 1 | C0-fast | C1-fast |
| seed 2 | C0-fast | C1-fast |
| seed 3 | C0-fast | C1-fast |

每个 pair 使用相同 actor/critic 初始化 seed、任务顺序和 validation seed。warmup 期间
actor 相同，应尽量获得相同 bare rollout；若 async engine 无法保证 bitwise determinism，
至少保证任务、采样 seed 和 policy hash 配对并记录差异。

正式训练后 actor 和 online documents 分支化是预期行为。

### 11.2 主终点

RL-eval 使用冻结的全部 **140 个 valid_seen + 134 个 valid_unseen** 任务；每个 arm、seed、
checkpoint 对每个任务恰好跑一次 bare greedy episode，不允许按结果缺行或抽取子集。令
\(S^{d}_{a,s,u}\) 是 arm \(a\)、seed \(s\)、actor update \(u\) 在 split
\(d\in\{\mathrm{seen},\mathrm{unseen}\}\) 上的任务平均成功率。先对两个 split 宏平均：

\[
S_{a,s,u}=\tfrac12
\left(S^{\mathrm{seen}}_{a,s,u}+S^{\mathrm{unseen}}_{a,s,u}\right),
\]

再取预先锁定的最后三个 checkpoint：

\[
S_{a,s}=\frac{S_{a,s,30}+S_{a,s,35}+S_{a,s,40}}3,
\qquad
d_s=S_{C1,s}-S_{C0,s},
\qquad
\Delta_{\mathrm{success}}=\frac13\sum_{s=1}^{3}d_s.
\]

宏平均防止 140/134 的轻微规模差决定总分；seen、unseen 原始分数仍必须分别报告。fast
的 **end-to-end success gate** 同时要求：

1. \(\Delta_{\mathrm{success}}\ge5\) 个百分点；
2. paired hierarchical bootstrap 的 95% percentile interval 下界高于 0；
3. 至少 2/3 个 \(d_s>0\)；
4. 至少 3 个 task family 的配对差为正，且逐一移除任一 family 后的总体差仍大于 0；
5. 两臂在每个 checkpoint 的 manifest 都严格为 140/134，同一 task ID、environment seed、
   parser 和 protocol hash 配对。

bootstrap 固定为 10,000 次：先配对地重采 seed，再在每个 seed 内分别重采 seen/unseen
task ID；同一 task 的两个 arm 与 30/35/40 三个 checkpoint 始终绑定在一起，不能把时间点
当成独立样本。`statistics_protocol.json` 在训练前冻结唯一分析脚本 hash、bootstrap seed
20260730、10,000 次区间重采以及 100,000 次 Monte Carlo power simulation。success power
固定模拟绝对 +5pp、\(\alpha=0.05\)、task-level pair correlation=0、between-seed SD=0.05；
baseline 只取 frozen cold-start-dev 宏平均。不得在看见 C0/C1 结果后换相关系数或脚本。
若预计功效或有效样本下的功效低于 80%，未通过区间门只能记为“fast 未证实/统计上不充分”，
不能声称负结果；通过仍只算可行性证据，正式期需增加 seed 并原样冻结 fast 超参数。

### 11.3 共同 held-out critic audit

正式更新后 C0/C1 actor、文档和 rollout 分布都会分叉，因此两臂各自在自己训练 batch
上的 on-policy prequential MSE **不可直接横比**：差异可能来自 critic，也可能只是一个
actor 产生了更容易预测的轨迹。为使“C1 value 更好”可证伪，在 warmup end 以及 actor
update **10、20、30、35、40** 做只读的共同审计：

`u=0` 使用 warmup-end critic 与未更新的 actor；`u>0` 使用逻辑 actor update `u` 完成并
commit 后的 actor/critic checkpoint。audit 开始前同时冻结两臂 checkpoint，不能让一臂
先用 audit 结果继续训练。

1. 使用预先冻结且从不训练的 16 个 critic-audit 任务；
2. C0 actor 和 C1 actor 在每个任务各产生 4 条 fresh bare rollout；
3. 每个 source actor/task 独立生成一份绑定该 source policy 的 matched online document；
4. 合并为同一组 \(16\times(4+4)=128\) 条 trajectory-document pair；
5. 两个冻结的 critic checkpoint 都对这 **完全相同的 128 条轨迹** 评分：C0 输入
   no-doc marker，C1 输入每条轨迹所属 source policy 的 matched document；
6. audit 样本不更新 actor、critic、researcher，不参与 checkpoint 选择，也不进入 RL-eval。

research 失败的 audit row 仍保留 outcome，并给 C1 显式 unavailable marker；不得删行。
各 audit checkpoint 的 document availability 也必须达到整体 95% / 合格 family 90%，否则
该 pair 是系统有效性失败，不用只保留成功文档计算一个乐观 MSE。

这不是第三条训练臂，也不是 shuffled-document 消融；它只是双方共同考试。令对 critic
\(c\) 和 trajectory \(e\)：

\[
m_{c,e}=\frac1{|M_e|}\sum_{r\in M_e}
\left(V_c(\text{audit prefix}_{e,r})-G_e\right)^2.
\]

先在同一 document cluster 的 4 条 rollout 内平均，再对 16 个 task 平均，最后对 C0/C1
两个 **source-actor 分布各给一半权重**，得到 \(M_{c,s,u}\)。critic 主终点同样取
30/35/40：

\[
M_{c,s}=\tfrac13\sum_{u\in\{30,35,40\}}M_{c,s,u},
\quad
d^V_s=M_{C1,s}-M_{C0,s},
\quad
R_V=1-\frac{\sum_sM_{C1,s}}{\sum_sM_{C0,s}}.
\]

**value-information gate** 同时要求：\(R_V\ge5\%\)、\(d^V_s<0\) 至少 2/3 seeds，且
以 seed 为最高层、audit task/source-policy document 为嵌套配对层的 10,000 次 bootstrap
中，\(M_{C1}-M_{C0}\) 的 95% interval **上界低于 0**。raw MSE 是唯一 gate；clipped
Brier/ECE、ranking 和 explained variance 只作补充解释。

value power 同样由上述 frozen 脚本计算：目标为 raw MSE 相对下降 5%、pair correlation=0；
误差方差只从不参与训练/选模的 warmup-end (`u=0`) common audit 估计并乘 1.5 安全系数，
在正式 actor update 前写回同一 protocol manifest。若预计功效低于 80% 且 value gate 未
通过，value 结论是 **inconclusive**，不是“文档没有 value 信息”；不得通过追加有利
audit task 或改相关系数补救。

两臂各自训练流上的 pre-update MSE 仍必须记录，用于发现 drift 和 critic lag，但在 actor
分叉后只作 arm-internal 诊断，不承担跨臂机制结论。

### 11.4 统计单位

- validation 主结果按 task/episode 统计；
- common audit 的 critic 主结果按 task/source-policy-document cluster 配对统计；
- warmup/train critic 指标按各自 task-document cluster 统计，不跨臂直接比较；
- seed 是算法复现实验的最高层独立单位；
- 同文档下 8 条 rollout 不能当 8 个独立文档样本；
- final-three-point 平均减少单个 checkpoint 偶然波动，但不能把三个时间点当独立 seed。

### 11.5 机制链与结论分级

端点 success 之外，至少验证：

\[
z
\rightarrow
\text{better pre-update }V
\rightarrow
\text{better raw advantage/gradient}
\rightarrow
\text{better bare success}.
\]

raw/whitened advantage、gradient norm、方向相似度和长度相关性没有唯一可公认的“更好”
阈值，因此只作机制诊断，不设 pass gate。任一主 gate 若未通过且其预注册功效低于 80%，
总体结论先记 **inconclusive**，不进入下表的负向格。对功效充分的结果再分四种：

| value-information gate | success gate | 允许结论 |
|---:|---:|---|
| 通过 | 通过 | 完整主假设获得 fast 支持：文档改善 grounded value，且改善 PPO 学习 |
| 通过 | 未通过 | value 假设成立，但当前 \(\lambda=1\)/whitening/单步 PG 未消费该优势 |
| 未通过（功效充分） | 通过 | 观察到 treatment uplift，但 privileged-value 中介归因未证实；先查泄漏与分布差 |
| 未通过（功效充分） | 未通过（功效充分） | 当前 privileged grounded critic 配方不获支持 |

因此，只有 success 提高而共同 audit value 未改善时，不能直接归因于 privileged critic；
同样，只有 value 改善也不能声称 RL 已被改进。

---

## 12. 日志与分析指标

### 12.1 Grounded outcome

- train bare success；
- fixed validation overall success；
- seen/unseen success；
- success by task family；
- invalid/truncated 作为诊断分解；
- 每个 actor update 的任务构成。

**critic/score/mean** 在旧代码里表示 rollout 平均 reward，不是 value-model 输出。新分支
应避免继续用这个歧义名称，明确使用 **outcome/train_success** 和 **value/**。

### 12.2 Value 质量：必须使用 update 前 prediction

- **value/pre_raw_mse**：训练流中更新前 value 的 raw MSE；token 先在每条 trajectory
  内取均值，再按 task-document cluster 汇总；
- **audit/raw_mse**、**audit/raw_mse_delta**、**audit/relative_reduction**：共同 held-out
  audit 的主 critic 指标，并标记 source_actor；
- clipped-to-[0,1] Brier / ECE；
- explained variance；
- prediction mean/std/min/max；
- success 与 failure 的 value separation；
- calibration curve / ECE；
- 按 task family、document availability、trajectory depth 分层；
- within-task 8-rollout ranking；
- C0/C1 相对常数 base-rate predictor 的增益；
- critic update 后的 train loss 只作优化监控，不能替代 prequential 指标。

value head 输出不强制限制在 [0,1]。训练仍用 raw scalar MSE；calibration 可同时报告 raw
和仅用于评估的 clipped-to-[0,1] 版本，但不能在日志里混称。

### 12.3 Critic 优化

- vf_loss；
- vf_clipfrac；
- critic gradient norm / clip rate；
- critic LR；
- value update magnitude；
- C0/C1 每步实际 optimizer step 数；
- context length、dynamic micro-batch 数与被拒绝样本数。

### 12.4 Advantage 与 actor 更新

- raw / whitened advantage mean/std/quantile；
- token-weighted / sequence-balanced 两套 moments；
- advantage 与 response length、return、task family 的相关性；
- PPO clip fraction；
- old-to-new policy approximate KL；
- entropy（日志，系数为 0）；
- actor gradient norm；
- actor parameter update norm；
- 相邻 batch 梯度方向相似度或方差代理。

### 12.5 Researcher 与文档

- document availability；
- document token/word length；
- researcher experiments/episodes；
- independent research success；
- doc hash 与 policy hash；
- 完整 document identity **(batch_id, policy_version, task_id)**；
- 同 task-step 的 8 条 rollout 是否共享同一 doc hash；
- 是否错误复用了旧 policy 文档；
- researcher/main lineage 是否发生交叉。

---

## 13. 当前代码上的实现设计

### 13.1 新分支、新配置，不改造成 OPD 混合臂

在独立开发分支实现，例如：

~~~text
privileged-grounded-critic-ppo
~~~

新增独立配置与 launcher，例如：

~~~text
configs/privileged_critic_ppo.yaml
scripts/launch_privileged_critic_ppo.py
~~~

不要把现有 **trace.yaml** 的 OPD/mixed-objective 参数继续扩张成新方案。新 launcher 明确：

~~~text
algorithm.adv_estimator=gae
critic.enable=true
distillation.enabled=false
actor_rollout_ref.actor.use_kl_loss=false
algorithm.use_kl_in_reward=false
~~~

### 13.2 Agent loop：复用 research，删除 teacher 语义

当前 [agent_loop.py](../../src/trace_opd/train/agent_loop.py) 已具备：

- 每 **(global_step, task_id)** 去重的 online research；
- research 与 bare rollout 并发；
- 一份文档共享 8 条 rollout；
- bare student response IDs/mask；
- research archive 和 per-sample metadata；
- 文档注入 prompt 的构造工具。

新分支把现有 `global_step` 路由显式升级为规范三元组
**(batch_id, policy_version, task_id)**，并断言一个 batch 只有一个 policy hash；否则不能
仅凭当前二元 key 证明 fresh/matched 属性。

新分支应新增：

~~~text
critic_prompt_ids
critic_doc_available
critic_doc_hash
critic_policy_version
critic_research_lineage_id
main_rollout_lineage_id
~~~

现有 **teacher_prompt_ids**、**teacher_routing_key** 和 teacher top-k scoring 不进入新路径。

### 13.3 Trainer：建立 critic batch view

当前 VERL **_compute_values(batch)** 和 **_update_critic(batch)** 默认读取 actor batch。
新数据流应为：

~~~text
rollout output
  ├── build_actor_batch()
  └── build_critic_batch(mode=none|matched_online_document)

old_log_probs = actor.compute_log_prob(actor_batch)
old_values    = critic.compute_values(critic_batch)

actor_batch.values = align(old_values)
actor_batch = compute_return_and_advantage(actor_batch)

critic_batch.values        = actor_batch.values
critic_batch.returns       = actor_batch.returns
critic_batch.response_mask = actor_batch.response_mask

critic.update(critic_batch)
actor.update(actor_batch)
~~~

**old_values** 返回形状必须严格为 **[batch, actor_response_length]**，然后才能与 reward、mask
合并。C0/C1 走同一个 builder 和 trainer 路径，仅 document mode 不同。

### 13.4 原生 VERL critic 复用点

直接复用：

- Role.Critic 独立 worker/resource pool；
- HF AutoModelForTokenClassification(num_labels=1) value model；
- no_padding_2_padding 的 causal left shift；
- GAE；
- PPO clipped value loss；
- critic optimizer/checkpoint；
- actor/critic FSDP 与 dynamic batching。

需要窄改：

- critic 专用 prompt/batch；
- actor/critic batch 字段路由；
- raw advantage 与 prequential value metrics；
- logical actor-update counters；
- warmup off-by-one 断言；
- validation 以 actor-update 为调度依据；
- provenance 与 gradient-isolation 检查。

不需要重写 PPO loss、GAE 数学或自制共享 value head。

### 13.5 配置单一真源

新配置至少显式声明：

~~~yaml
experiment:
  arm: C0-fast | C1-fast
  paired_seed: ...
  objective_version: privileged_grounded_critic_v1

data:
  train_batch_size: 16
  cold_start_train_manifest: ...
  cold_start_dev_manifest: ...
  rl_train_manifest: ...
  critic_audit_manifest: ...       # exactly 16 frozen tasks
  val_seen_manifest: ...           # exactly 140 tasks
  val_unseen_manifest: ...         # exactly 134 tasks

models:
  actor_init_path: ...
  actor_init_hash: ...
  critic_backbone_path: ...        # same cold-start weights
  critic_backbone_hash: ...        # must equal actor_init_hash
  critic_value_head_seed: ...
  tokenizer_hash: ...

capacity:
  critic_context_capacity: ${RESOLVED_CRITIC_CONTEXT_CAPACITY}
  critic_model_override: ${RESOLVED_CRITIC_MODEL_OVERRIDE}
  critic_ppo_max_token_len_per_gpu: ${RESOLVED_CRITIC_PPO_TOKEN_BUDGET}
  critic_forward_max_token_len_per_gpu: ${RESOLVED_CRITIC_FORWARD_TOKEN_BUDGET}
  critic_infer_max_token_len_per_gpu: ${RESOLVED_CRITIC_INFER_TOKEN_BUDGET}

research:
  enabled: true
  backend: external
  n_exp: 12
  k_default: 4
  k_max: 8
  max_turns: 30
  doc_max_words: 3000
  episode_temperature: 1.0
  model_fingerprint: ...
  endpoint_fingerprint: ...
  prompt_hash: ...

episode:
  max_env_steps: 30
  max_assistant_turns: 40
  max_tokens_per_turn: 2048
  parser_hash: ...
  environment_hash: ...

privileged_critic:
  mode: none | matched_online_document
  online_research: true
  logical_warmup_batches: 8
  max_warmup_batches: 16
  actor_updates: 40
  document_identity: [batch_id, policy_version, task_id]
  missing_document_marker: "[DOCUMENT_UNAVAILABLE]"
  no_document_marker: "[NO_PRIVILEGED_DOCUMENT]"
  truncation: error
  audit:
    actor_updates: [0, 10, 20, 30, 35, 40]
    rollouts_per_source_actor: 4

algorithm:
  adv_estimator: gae
  gamma: 1.0
  lam: 1.0
  use_kl_in_reward: false

actor_rollout_ref:
  model:
    path: ${ACTOR_INIT_PATH}
  actor:
    strategy: ${RESOLVED_ACTOR_STRATEGY}
    loss_agg_mode: seq-mean-token-mean
    ppo_mini_batch_size: 16
    ppo_epochs: 1
    use_dynamic_bsz: true
    ppo_max_token_len_per_gpu: ${RESOLVED_ACTOR_PPO_TOKEN_BUDGET}
    grad_clip: 1.0
    optim:
      lr: 1.0e-6
  rollout:
    n: 8
    temperature: 1.0
    response_length: 32768
    max_model_len: 40960

critic:
  enable: true
  strategy: ${RESOLVED_CRITIC_STRATEGY}
  model:
    path: ${ACTOR_INIT_PATH}
    tokenizer_path: ${ACTOR_INIT_PATH}
    use_remove_padding: true
    enable_gradient_checkpointing: true
    override_config: ${RESOLVED_CRITIC_MODEL_OVERRIDE}
  loss_agg_mode: seq-mean-token-mean
  ppo_mini_batch_size: 16
  ppo_epochs: 1
  use_dynamic_bsz: true
  ppo_max_token_len_per_gpu: ${RESOLVED_CRITIC_PPO_TOKEN_BUDGET}
  forward_max_token_len_per_gpu: ${RESOLVED_CRITIC_FORWARD_TOKEN_BUDGET}
  ppo_infer_max_token_len_per_gpu: ${RESOLVED_CRITIC_INFER_TOKEN_BUDGET}
  cliprange_value: 0.5
  grad_clip: 1.0
  optim:
    lr: 1.0e-5
    lr_warmup_steps_ratio: 0.0
    weight_decay: 0.01

distillation:
  enabled: false

research_rl:
  weight: 0.0

auxiliary_rows:
  mix_sft_weight: 0.0
  expert_rollouts: 0
~~~

这里沿用当前 researcher 角色的已确认 profile；模型名称、endpoint、prompt、工具协议、
thinking 开关和全部调用参数必须解析成 fingerprint，而不是只保存 `${ENV_VAR}` 字面量。
episode 的 30/40/2048 是当前任务接口协议，不是 token/step cost objective，也不进入 reward。

`RESOLVED_CRITIC_CONTEXT_CAPACITY` 在真实 longest-sample smoke 后写成数值。它不是一个
可直接传给当前 `CriticConfig` 的字段；launcher 必须将它校验/转译到模型实际支持的
`critic.model.override_config`（仅在目标 Qwen checkpoint 支持该 override 时）以及原生
`critic.ppo_max_token_len_per_gpu`、`critic.forward_max_token_len_per_gpu` 和
`critic.ppo_infer_max_token_len_per_gpu`。当前组件 YAML 未显式列出最后一项而 trainer
inference 会读取它，因此 launcher 必须以 Hydra `+critic.ppo_infer_max_token_len_per_gpu`
或等价结构化 compose 方式注入，并由 dataclass/实际 worker 读回测试确认。不得虚构
`critic.max_model_len`。最终必须满足：

~~~text
critic capacity >= critic header + maximum document + full bare history/response + 8-token margin
observed critic input length <= resolved capacity
truncated critic samples == 0
rejected main rows == 0
~~~

原生 VERL resolved config 至少必须真实出现
`actor_rollout_ref.model.path`、`actor_rollout_ref.actor.use_kl_loss`、`critic.strategy`、
`critic.model.path`、`critic.model.use_remove_padding` 和上述三个 critic token-budget 字段；
wrapper-level 的 `models.*` / `capacity.*` 只负责 provenance，不能代替 native Hydra 路径。
launcher 必须先用 OmegaConf compose + dataclass validation 证明不存在 unknown/missing key。

FSDP strategy、nodes/GPUs、offload、sequence parallel、dynamic token budget、actor/critic resource
pool 和实际软件 commit 都必须进入 resolved resource manifest；C0/C1 使用同一 topology，
仅 C1 因真实文档自然具有更长序列。硬件值由 smoke 后冻结，不能在 pair 中临时改变。

launcher 还必须保存 phase-specific native `total_training_steps/critic_warmup`、五个 split
hash、researcher profile、actor/critic/checkpoint hash、tokenizer/chat-template/scaffold、
parser/environment/protocol、loss aggregation、GAE/whitening、batch cardinality和 validation/
audit 口径，以及 `warmup_gate.py` / statistics script hash 组成的 objective fingerprint。
跨 C0/C1 除 `privileged_critic.mode` 外有任何初始
字段差异，或跨 cold-start checkpoint、aggregation mode、phase decision 的 resume，均
fail loud。

---

## 14. 验证与测试计划

### 14.1 静态与单元测试

1. **Actor/critic response identity**：两个 view 的 response IDs 与 mask 逐位相同。
2. **Prompt separation**：C1 文档 token 只存在于 critic prompt，不存在于 actor prompt。
3. **Causal shift**：value for \(y_r\) 来自 \(y_{<r}\)。
4. **Observation mask**：observation/glue value 不进入 loss。
5. **Return algebra**：\(\gamma=\lambda=1\) 时每个有效 token return 等于终局 \(G\)。
6. **Loss aggregation**：将某 episode 的整段有效 token-loss 序列整体重复一遍，其
   sequence mean 与该 episode 总权重不变；不得把“复制任意单个 token”误作此性质。
7. **Document grouping**：同 task-step 8 条 rollout doc hash 相同。
8. **Online refresh**：相邻 policy step 不读取旧 document artifact。
9. **No main-rollout leakage**：research request schema 不包含 main response/outcome。
10. **Missing document marker**：失败样本不被删除或静默改成 normal matched sample。
11. **Split isolation**：五个 manifest 的 task/gamefile ID 两两无交集，RL-eval 精确为
    seen 140 / unseen 134。
12. **Batch cardinality**：正式/warmup batch 精确为 16 clusters / 128 main rows，且不包含
    research、SFT、expert 或 padding-as-sample rows。
13. **Audit pairing**：两个 critic 收到相同 audit trajectory ID、outcome、response IDs/mask；
    差别仅为 no-doc 与 matched-doc critic context。
14. **Native config compose**：wrapper config 转译后可被当前 VERL OmegaConf/dataclass 完整
    compose；不存在 unknown/missing key，actor/critic strategy、model path 和 token budget
    均为 resolved 数值。
15. **Gate golden cases**：固定小数据覆盖 last-two batch、无 success、无 failure、低 power、
    pass/extend/abort/inconclusive，锁定脚本输出与 hash。

### 14.2 梯度隔离测试

在一个 critic-only update 前后：

- critic parameter hash 必须变化；
- actor parameter hash 必须逐位相同；
- actor optimizer state 必须不变；
- actor logits 在固定输入上必须相同。

在 actor-only 测试更新前后，critic 参数不得变化。任何失败都使实验作废。

### 14.3 Warmup 集成测试

小规模运行 2 tasks × 2 rollouts：

- 连续执行指定 critic-only batch；
- 确认 actor update count 为 0；
- 确认每批使用 fresh research 和 bare rollout；
- 确认 pre-update value 在 critic update 前持久化；
- warmup 前后 deterministic actor validation 完全相同；
- 检查 native **critic_warmup** 到逻辑批数的映射；
- 强制两个 arm 在 W8 barrier 等待同一 `warmup_decision.json`，覆盖 pass、extend、abort；
- 验证 W8→formal 与 W8→W16→formal 两条路径最终都恰好产生 40 次 actor update。

### 14.4 正式 step 数据流测试

一个 actor step 中断言：

~~~text
policy hash used by research == policy hash used by main rollout
critic hash for value       == critic-before-update hash
advantage computed before critic update
actor update uses frozen advantage
next research uses actor-after-update policy hash
main rows == 128
actor optimizer/scheduler step delta == 1
critic optimizer/scheduler step delta == 1
~~~

### 14.5 Resume 测试

分别在以下位置注入故障：

- research 完成后；
- rollout 完成后；
- critic 更新后、actor 更新前；
- actor 更新后、commit 前。

验证恢复不会出现半步 critic、重复 actor update、旧文档配新 policy 或 step counter 漂移。

### 14.6 端到端 smoke gate

正式三 seed 前，单 pair 小步运行并确认：

- C0/C1 都能加载独立 4B critic；
- critic context 不发生静默截断；
- reward 只有环境 outcome；
- distillation/teacher/ref worker 不存在；
- 8 个 rollout 的文档共享正确；
- common audit 能让两个 critic 对同一 128-row union 只读评分，且不改变任何参数；
- value、return、advantage、loss 和 actor update 全部 finite；
- 显存/序列容量可完成真实最长样本。

---

## 15. 失败模式、解释与停止规则

| 观察 | 更可能原因 | 决策 |
|---|---|---|
| 冷启动仍几乎全零 | 没有真实成功覆盖 | 返回 SFT/RFT，不启动 critic |
| warmup actor hash 改变 | 梯度/optimizer 隔离失败 | 实验作废 |
| C1 prompt 进入 actor | 特权信息泄漏 | 实验作废 |
| 文档读取了 main outcome | action/outcome leakage | 实验作废 |
| warmup value 全部崩到低常数 | 正例不足或输入/优化失败 | 两臂共同延长；仍失败则停止 |
| common audit 中 C1 raw MSE 不优于 C0 | 文档无增量信息、未被利用或长上下文增加难度 | value gate 失败；先查 alignment/coverage |
| success/value gate 未过且对应 power <80% | fast 样本不足以裁决 | 标记 inconclusive，不作负向机制结论 |
| 各自 on-policy MSE 看似改善、common audit 不改善 | actor 分布难度不同造成假象 | 以 common audit 为准，不接受 value 主张 |
| common audit value 更好、actor 无改善 | whitening、\(\lambda=1\)、单步 PG 或 critic lag 成为瓶颈 | value 假设通过，RL 改进未通过；进入预注册 follow-up |
| C1 actor 改善但 common audit value 不改善 | 泄漏、分支分布、随机性或其他中介 | treatment 正向但机制未确认，先审计 |
| common audit value 与 actor 均通过 gate | 主假设获得 fast 支持 | 扩大 seed 做正式确认 |
| C1 训练早期好、后期失准 | policy/doc 非平稳，critic lag | 提高在线校准/共同 critic update，不用旧文档 |
| advantage 与长度仍强相关 | whitening moments 或 token-level surrogate | 记录；首轮不改，按 follow-up 验证 |
| critic prompt 系统性超长 | context 配置不足 | 修容量；不得静默截断后继续解释 |
| 同 task-step doc hash 不一致 | research 去重/共享失败 | 修实现后重跑 |
| actor update 数不是严格 40 | warmup/global-step 计数错误 | 结果不可比，重跑 |

首轮停止条件：

1. 冷启动无法获得足够真实成功；
2. strict isolation、causal lineage 或 token alignment 任一失败；
3. warmup 到 16 批仍未达到数值安全门；
4. C1 大量样本因系统性 context 问题退化成 bare；
5. 三 seed 的 common-audit value 和 bare success 都没有一致正向迹象，且冻结样本量的
   统计功效足以排除预注册最小效应；功效不足时只能记为 inconclusive。

停止只否定当前 privileged \(V\)-critic 配方，不否定 researcher 文档用于 SFT、搜索、
规划或其他独立路线的价值。

---

## 16. 执行阶段与决策树

### Phase 0：4B cold start

~~~text
larger bare executor sampling
  -> environment-success filtering
  -> 4B SFT/RFT
  -> cold-start-dev: format >=95%, bare success 10%-25%, >=32 successes / >=3 families
~~~

不通过则继续 cold start。

### Phase 1：数据面与隔离 dry run

实现双视图、standalone critic、online doc provenance、mask/shift、logical counters。通过
§14 的 unit/integration/smoke tests 后才能训练。

### Phase 2：配对 warmup

每个 seed 的 C0/C1：8 个 online critic-only batch；共同安全门失败则一起到 16。保存
warmup-end checkpoint。

### Phase 3：C0-fast / C1-fast

从各自 warmup-end checkpoint 开始，执行 40 个 actor update，按 actor update 0/5/.../40
进行 bare validation；在 0/10/20/30/35/40 做 §11.3 的只读 common critic audit。

### Phase 4：结果裁决

~~~text
任一未通过的主 gate 对应 power <80%？
 ├─是：整体 inconclusive，先扩大预注册样本
 └─否：检查 value-information gate
    ├─否
    │  ├─success gate 也未过：当前配方不获支持
    │  └─success gate 通过：只有 treatment uplift，value 中介未确认，先审计
    └─是
       ├─success gate 通过：完整主假设获 fast 支持，扩大 seed 正式复现
       └─success gate 未过：value 假设成立，RL 改进未成立；再查 whitening/lambda/critic lag
~~~

不得因 C1 未立即提高 actor，就直接把 critic 输出加进 reward；那会跳到另一个有偏机制。

---

## 17. 通过首轮后的独立扩展线

这些是未来候选，不属于本文首轮机制：

### 17.1 Bootstrapped GAE

若 C1 的 common-audit value 明显更好但 actor 增益不足，可在共同优胜 checkpoint 上对比：

~~~text
lambda = 1.0  vs  0.95
~~~

只改 \(\lambda\)，保留环境 reward。它检验更强 bootstrap 是否能消费 privileged value。

### 17.2 Advantage normalization

若 raw advantage 改善被 token-weighted whitening 掩盖，对比：

- native token-weighted whitening；
- sequence-balanced whitening；
- mean-center only，不除 std。

一次只改 normalization，不与 \(\lambda\) 同时改变。

### 17.3 Grounded \(Q\)/PRM

若目标转向动作反事实 credit，需要通过 branch rollout 训练
\(Q^{\pi}(h,z,y)\) 或 canonical command \(Q_{u,C}\)。它是新估计对象，不能把当前
\(V(h,z)\) 逐步求和当作 process reward。

### 17.4 共享 actor–critic backbone

只在 standalone critic 的效果已经成立、且未来需要节省资源时考虑。共享实现必须承认
critic loss 会成为 actor representation learning 信号，不再是当前严格因果实验。

---

## 18. 最终锁定决策清单

- [x] 新建完全独立 RL 分支，不绑定 OPD。
- [x] researcher 保留为 online privileged feature constructor，不是 token teacher。
- [x] 五个用途 split 在任何采样/选 checkpoint 前冻结，RL-eval 保留全量 140/134。
- [x] 每个 training batch、每个任务 fresh document，规范身份为
  (batch_id, policy_version, task_id)；同三元组 8 rollout 共享。
- [x] research 可与 main rollout 并发，但数据血缘严格独立。
- [x] actor 永远 bare；validation 也只测 bare actor。
- [x] critic target 只用 bare rollout 环境 0/1 return。
- [x] C0/C1 只有 critic 是否读取 matched online document 不同。
- [x] 只比较 C0-fast 与 C1-fast，不加 shuffled-document。
- [x] 使用 VERL 原生独立 4B critic，严格梯度隔离。
- [x] critic backbone 从 4B SFT/RFT actor checkpoint 初始化；full-parameter 训练。
- [x] actor/critic 都用 seq-mean-token-mean。
- [x] GAE 取 \(\gamma=\lambda=1\)；保留 native advantage whitening。
- [x] 8 个 critic-only warmup batch；共同失败则两臂一起到 16。
- [x] warmup 后严格执行 40 个 actor update。
- [x] 每批精确 16 tasks × 8 main rollouts；actor/critic 各一个 full-batch optimizer step，
  额外训练 row 通道全部关闭。
- [x] validation 每 5 个 actor update，主终点为最后 30/35/40 三点 bare success 均值。
- [x] fast 使用 3 个 paired seeds，目标 C1 至少 +5pp、配对方向为正、2/3 seeds 正向。
- [x] 用两臂共同的 held-out 128-row audit 检验 value；不把分叉后的各自 on-policy MSE
  当成跨臂证据，也不增加 shuffled-document。
- [x] warmup/statistics gate 使用单一脚本 hash；任一未通过 gate 在 power<80% 时只记
  inconclusive。
- [x] 不在首轮加入 token/step/overflow objective。
- [x] C1 common-audit value 更好但 actor 无增益时，先查 whitening、\(\lambda\) 和
  critic lag。
- [x] 任何 actor privilege 泄漏、main-outcome 文档泄漏或 warmup actor 更新都使实验作废。

---

## 19. 代码与证据入口

### 当前代码

- [训练 launcher](../../scripts/launch_train.py)
- [在线 researcher 与 AgentLoop](../../src/trace_opd/train/agent_loop.py)
- [多轮 episode token/mask 构造](../../src/trace_opd/core/episode.py)
- [VERL PPO trainer](../../verl/verl/trainer/ppo/ray_trainer.py)
- [VERL GAE / PPO / value loss](../../verl/verl/trainer/ppo/core_algos.py)
- [VERL causal value shift](../../verl/verl/workers/utils/padding.py)
- [VERL critic loss](../../verl/verl/workers/utils/losses.py)
- [VERL critic 配置](../../verl/verl/trainer/config/critic/critic.yaml)

### 项目证据

- [OPD 显微镜分析任务说明](../opd-microscope-analysis-taskbrief.md)
- [数据盘点](../analysis/T0-data-inventory.md)
- [长度膨胀分析](../analysis/T2-length-inflation.md)
- [Teacher signal 分析](../analysis/T3-teacher-signal.md)
- [完整研究与文献备忘录](../privileged-critic-research-and-feasibility.md)

### 理论背景

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Asymmetric Actor Critic for Image-Based Robot Learning](https://arxiv.org/abs/1710.06542)
- [Unbiased Asymmetric Reinforcement Learning under Partial Observability](https://arxiv.org/abs/2105.11674)
- [Learning Using Privileged Information](https://jmlr.org/papers/v16/vapnik15b.html)

本文至此完成 v1.2 机制冻结。任何实现中对 reward、actor 输入、文档生命周期、
\(\gamma/\lambda\)、loss aggregation、warmup、gradient isolation 或实验臂的改变，都必须
更新设计版本和 objective fingerprint，不能作为“等价工程调整”静默进入运行。
