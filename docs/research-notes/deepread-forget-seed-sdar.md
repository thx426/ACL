# 多目标干扰/遗忘论文精读 × SEED/SDAR 联合调度侧源码深读

> 对象：TRACE-OPD 四项 loss 一次反传（`L = w_opd·OPD + w_rl·RL_exec + w_sft·SFT + w_res·RL_research`）的角色遗忘风险、
> verl 框架级自检、以及 RL+蒸馏联合的成熟实践。
> 论文基路径 `.conductor/branches/CD-002/papers-mtl/`；源码 `work/refs/SEED/`、`work/refs/SDAR/`；
> 自检对象 `trace_opd/verl/`（version `0.9.0.dev`）。
> 所有结论均标注论文节/式号或 `file:line`。

---

# A) 论文精读

## A1. MIFO — Mitigating Forgetting Between SFT and RL (2510.04454)

**核心诊断（§3）**：SFT 与 RL 的参数更新是**不对称**的。§3.1 用两个实验证明：

- **在线梯度稀疏化**（每步 Bernoulli 丢 50% 梯度坐标）：SFT 曲线约 40 步就追平基线、收敛差 **0.7pp**；RL 差 **4.23pp**（Fig 1）。
- **事后更新剪枝**（对 Δθ = θ_T − θ_0 随机剪 p_post）：剪枝率 0→0.5 时 RL 从 35.5 掉到 29.9（**−5.6**）；SFT 反而在 0.1 处升到 29.6（+0.3）再缓降到 28.4（−0.9）（Fig 2）。

§3.2 Fig 3 显示 SFT 的逐层 ‖Δθ‖₂ 全面大于 RL。结论：**SFT 冗余、RL 吝啬，所以 SFT 覆盖 RL**（"SFT forgets RL"）。
附录 C 的 Lemma 1 / Theorem 1 给出形式化：决策冗余比 DR_x ≡ ‖Δθ‖₂·‖g‖₂/(ε − m₀)，且 DR^SFT ≥ DR^RL（式 6）。

### 算法（§4，伪码见附录 D / p15 Algorithm 1）

**① 数据 + token 选择（§4.1）**

- RL 用 GRPO；对每个 question 统计 rollout 准确率 `acc(q)`，把 `acc(q) ≤ p` 且外部解答可验证（`extract(s) = a`）的样本进 `Buffer_FT`；Buffer 满 S 就切到 SFT，Buffer 排空再切回 RL。
- SFT **只在高熵 token 上算 loss**：`H_t = −Σ_j p_{t,j} log p_{t,j}`，取样本内熵的 top-ρ 分位阈 `τ_ρ(q,s)`，

  `L_SFT(θ) = −(1/|s|) Σ_t 1[H_t ≥ τ_ρ(q,s)] · log π_θ(s_t | q, s_<t)`

**② 参数冻结（§4.2）**

- 记一个 RL→SFT 区间内 `Δθ^j_{RL,i} = ‖θ^j_{RL,end,i} − θ^j_{RL,start,i}‖₂`；
- 历史重要度 `C̄_i = α·C_{i−1} + (1−α)·Δθ_{RL,i}`（`C_0 = 0`）；
- 取 `I_i = TopK(C̄_i, k)` 得二值掩码 `M_i`，SFT 阶段令 `∇_θ L_SFT ← (1−M_i) ⊙ ∇_θ L_SFT`；
- SFT 完成后**全部解冻**进入下一轮 RL。`C_i = M_i ⊙ C̄_i` 传给下一区间。α=0 记作 MIFO†。

### 超参（附录 D / p16）

`p = 1/8`，`ρ = 0.2`，`k = 0.5`，`α = 0.5`。
GRPO：rollout bs 128、8 rollouts/query、update bs 64、lr 1e-6、temp 1.0、entropy coef 0.001。
SFT：bs 128、lr 1e-5。3 epochs、4×8 A100、**VERL 框架**、二值 reward。RoPE base 10k→40k、context 16,384。

### 效果与消融（Table 3 / p8，Qwen2.5-Math-1.5B）

| 配置 | Overall | 平均长度 |
|---|---|---|
| base | 15.0 | 4043 |
| + Interleave | 34.0 | 4421 |
| + Interleave + ES（熵选择） | 39.2 | 3246 |
| + Interleave + PF（参数冻结） | 38.4 | 4490 |
| **MIFO（ES+PF）** | **40.0** | **2518** |

即 **ES 主要降长度、PF 主要稳分数，二者互补**。Fig 6 证实高熵 token 选择确实压小了 SFT 的逐层更新幅度（仅 17–19 层例外）。
主表：1.5B 上 MIFO 40.0（最优），7B 上 54.3 vs SRFT 54.1；数据用量仅为 SRFT 的 1.5% SFT / 20.4% RL（Fig 5 右）。

### ★ 必须知道的自我否定（附录 E.4 / Fig 13）

作者补做了"随机剪枝 vs RL-TopK 选择性剪枝"，结论是**两者无显著差异，RL TopK 甚至略输给随机**。
所以 PF 的收益来自"**限制 SFT 更新幅度**"这一事实，而非"精确识别了 RL 关键参数"。

### 对 TRACE-OPD 的适用性判决

**机理适用、算法要降级使用。**

- 我们的 research 侧是 SFT 冷启动 + PG，executor 侧是 RL——恰是 MIFO 所说"大幅度更新覆盖小幅度更新"的结构，但**方向相反**（我们担心 RL 冲掉 SFT 出来的能力）。
- Theorem 1 只保证 `DR^SFT ≥ DR^RL`，反向不成立，因此**不能照搬"冻结 RL 关键参数"**——要冻的是 research-critical 参数。
- 因 Fig 13，落地时**先试便宜的 random-50% 冻结**，验证有效再上 TopK。

---

## A2. MGS — Advancing General-Purpose Reasoning Models with Modular Gradient Surgery (2602.02301)

### Sequential RL 的两种失效（§3.1–3.2）

- **Forgetting**：后训域抹掉前训域（Chat 分数 −7.0）。
- **Rigidity**（作者定义的新现象）：先训的域**限制后训域的上限**。

关键是**非对称**：Math→Chat 的 Chat 缺口 −9.6，而 Chat→Math 的 Math 缺口只有 −1.3（Fig 2）。
§3.2 用熵解释（Fig 3）：Math 训练降熵、Chat 训练升熵，**先训 Math 会把后续 Chat 锁死在低熵**；
因此**高熵、行为灵活的域应当先训**。附录 E / Table 5 给全量：Math-only 2ep 的 Chat avg 仅 3.3；Chat4ep→Math2ep 能到 Chat 47.2 / Math 61.6。

### Mixed RL 的失效（§3.3–3.4）

即使 9:1 极端配比也追不上单域专家（Fig 4）→ **调数据比例解决不了冲突**。诊断量两个（Fig 5）：

- 域间梯度**余弦相似度**频繁 < 0；
- Math/Chat 的**梯度范数比**在对数轴上跨数量级 → 冲突与失衡同时存在。

附录 F / Fig 10 进一步给**分模块**余弦：`L25 Input LayerNorm` 长期大幅负相关，而 `L25 Attn V-Proj`、`L25 MLP Down` 基本 ≥ 0，`L5 Input LN` 在 ±0.5 间震荡 → **冲突是局部的，不是全局的**。

### 手术算法（§4 Algorithm 1；全版 Algorithm 2 见附录 D / p18）

把 θ 按 `model.named_parameters()` 划成不相交模块 Φ（逐层的 Attn / MLP / LayerNorm）。
对每对任务 (i, j)（j 走 `RandomPermutation({1..K}\{i})`），**在每个模块 φ 内独立做 PCGrad**：

```
v_i, v_j = Flatten(g^PC_i[φ]), Flatten(g_j[φ])
if v_i · v_j < 0:
    v_i ← v_i − (v_i·v_j)/(‖v_j‖² + ε) · v_j
    g^PC_i[φ] ← Unflatten(v_i)
return Δθ = Σ_i g^PC_i
```

全局手术（GGS）只是把 Φ 退化成单个模块（Algorithm 2 的 `Mode = GLOBAL` 分支）。

### 开销（附录 F / p19）

- 显存：`O(T×N)`，FSDP 下单卡 `O(T × N/world_size)`，峰值 `≈ (T_buffer + T_projected) × N/world_size × Bytes`。
- 时间：理论 `O(M·T²)`（M = 模块数），**实测 ~163 秒 / ~30 小时训练**，可忽略。

### 效果幅度（Table 2 / p7；Table 3 / p8）

- Qwen-2.5-7B 总均 **44.9** vs Naïve Mixing 40.4（**+4.5，11.1%**）；Llama-3.1-8B **30.2** vs 25.9（**+4.3，16.6%**）。
- **GGS（全局手术）常不如模块级**：Qwen 44.2 / Llama 30.1，作者解释为全局手术过于保守——单个不重要模块的冲突会迫使整个模型梯度被投影（§5.1）。
- 三任务（Math+Chat+IF，Llama）：MGS 32.6 > GGS 30.3 > Naïve 27.3。
- Fig 7 模块消融：**排除 LayerNorm 导致双轴暴跌**，尽管它参数最少；只对 Top-1% 范数模块做手术会让 Chat 升到 55 但 Math 掉到 45。

### 对 TRACE-OPD 的适用性判决

**诊断部分立刻可用，手术部分症状驱动。**

- 我们的四项 loss 天然是 Mixed 训练。Fig 5 / Fig 10 的两张图（全局 cos + 分模块 cos + 范数比）是现成的"是否冲突"判据，成本只是每 N 步存一次分组梯度。
- 手术本身在 T=2（research / executor）时开销很低，但**要等诊断给出"个别模块为负、全局非负"的证据再上**；若全局就是负的，MGS 自己的数据说明全局手术未必优于什么都不做。
- §3.1 的非对称结论**直接背书**我们"先 research SFT 冷启动、后 executor RL"的顺序——反过来会吃 rigidity。

---

## A3. SFT-then-RL Outperforms Mixed-Policy Methods (2604.23747)

### bug (a) — DeepSpeed CPU-offload optimizer

**机理（§2.1）**：ZeRO Stage 1/2 + CPU-offloaded Adam 时，offload 代码把"梯度拷回 CPU"放在只有 `micro_step_id == 0` 才执行的 `else` 分支里。中间 micro-batch 在 GPU 上正确累加，却从不触发拷贝，于是 optimizer step 时 **CPU 侧只看到第一个 micro-batch 的梯度**。

**确切位置与补丁（附录 C / p15）**：`deepspeed/runtime/zero/stage_1_and_2.py` 约 1394 行：

```diff
  if self.micro_step_id > 0:
      accumulate_gradients()
- else:
-     copy_gradients_to_cpu()
+ copy_gradients_to_cpu()
```

2024-09 由 DeepSpeed PR #6550 引入，修复已合入 PR #7967（识别于 DeepSpeed v0.18.9）。
**三个条件全部满足才触发**：ZeRO-1/2 + optimizer offload + gradient accumulation 步数 > 1。
影响 TRL / OpenRLHF / Llama-Factory。**verl 不用 DeepSpeed，因此不受影响**——受影响的是那些论文的 SFT 基线。

**症状指纹（Fig 2）**：grad norm 系统性**显著偏低** + loss 均值被抬高；**只有 optimizer 这一项会动 grad norm**。

### bug (b) — OpenRLHF per-mini-batch loss 聚合

**机理（§2.2）**：算的是 **mean of per-mini-batch means**，而不是真正的 per-token mean。每个 mini-batch / 每个 DP rank 先各自算局部均值再平均，于是**每个 mini-batch 无论含多少有效 token 都占同样权重**。该 bug 继承自预训练代码（那里 data packing 保证各 rank token 数相等），SFT 里 prompt/response 长度不齐，**几乎每一步都触发**。

**修复伪码（附录 D / p15，Algorithm 2）**：

```
S_k ← MaskedSum(−ℓ_i, m_i)        # 本 rank loss 和
n_k ← Σ_i m_i                     # 本 rank token 数
N   ← AllReduce(n_k, SUM)         # 全局 token 数（跨所有 DP rank）
L_k ← S_k / N × D                 # D = DP size，抵消 DDP 的梯度平均
backward(L_k)
```

verl 已于 **2025-11 PR #3994** 修复；OpenRLHF / Llama-Factory 截至成文未修（OpenRLHF PR #1216 待合）。

**症状指纹（Fig 2）**：只增大 loss 的**方差**，不移动均值，**不影响 grad norm**。

### 分离效果与修正后结论

**Table 2**（Qwen2.5-Math-7B，3 seed）：

| SFT 配置 | Avg |
|---|---|
| OpenRLHF baseline | 48.3 ± 0.8 |
| + fix loss aggregation | 49.1 ± 0.9 |
| + fix optimizer | 53.4 ± 0.4 |
| + fix both | 54.0 ± 0.2 |
| verl（独立实现） | 53.8 ± 0.1 |

→ **optimizer bug 吃掉绝大部分差距，loss 聚合是小而可测的额外项。**

**Table 3** 另给一条警示：仅仅照搬 SRFT 的 SFT 超参（10× 低 lr、2× batch、linear schedule）就能让基线从 53.8 掉到 48.3——**5.5 分的"方法收益"可以纯由超参制造**。

**Table 1 / Table 4**：修正后 SFT 单独 52.2 已超 LUFFY(46.3) / ReLIFT(48.8) / Prefix-RFT(51.8)；SFT→RL 57.0 超次优 SRFT(53.2) **+3.8**；Llama-3.1-8B 上 43.7 vs HPT 21.5（**+22.2**）。
**附录 B**：截断到 50 步 RL 的版本 3.63e19 FLOPs，低于 LUFFY 6.65e19 与 ReLIFT 8.76e19。

RL 超参（附录 A）：rollout bs 128、8 rollouts/prompt、mini-batch 64、lr 1e-6 constant、entropy 0、temp 1.0、max resp 8192、500 步；DAPO 式非对称 clip（ε_low=0.2, ε_high=0.28）+ token-level loss，去 KL、去长度归一与标准误归一。

---

### ★ verl（fsdp 后端）自检清单

我已核对 TRACE-OPD 自带的 verl（`trace_opd/verl/`，version `0.9.0.dev`），**结构上两个 bug 都已不在**：

| 事实 | 位置 |
|---|---|
| `agg_loss` 已是全局归一版：`masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size`；`dp_size > 1` 而 `batch_num_tokens is None` 会**直接抛错**。正是附录 D 的修复形式。 | `verl/verl/trainer/ppo/core_algos.py:1138-1173` |
| `forward_backward_batch` 先 `all_reduce(loss_mask.sum(), SUM, dp_group)` 得全局 `batch_num_tokens`，再写入 `dp_size`，**然后**才切 micro-batch，每个 micro-batch 直接 `loss.backward()`。**全树没有 `/gradient_accumulation` 的除法。** | `verl/verl/workers/engine/fsdp/transformer_impl.py:643-657` |
| `sft_loss` 用同一模式；且规定一旦全局归一生效，metric 聚合必须切成 SUM | `verl/verl/workers/utils/losses.py:31-52`；`:60-80` |
| 全树只有 `verl/utils/ulysses.py` 提到 deepspeed，**无 DeepSpeed 训练路径** → bug (a) 原样不适用 | — |
| FSDP1 路径**强制关闭 actor 的 CPUOffload**，注释明写 "causes incorrect results when using grad accumulation"（bug (a) 的 FSDP 同源问题）；FSDP2 路径的 `offload_policy` 仍可被 `engine_config.offload_policy` 打开 → **唯一残留开关** | `transformer_impl.py:399-410`；`:433-441` |

**因此自检重点不是"框架有没有 bug"，而是"我们新加的三项 loss 有没有走上那条已修好的路"。七条：**

**C1 — 掩码/分母一致性。**
逐项确认 OPD / SFT / RL_research 的 `loss_mask` 与 `batch_num_tokens` 是否同源。若 OPD 用 `opd_step_mask ⊂ response_mask` 而分母仍是全局 response token 数，则
**有效 w_opd = 名义 w_opd × (OPD 激活 token / 全局 token)**，逐 batch 漂移。
*测法*：固定一批数据，人为把 OPD 激活步比例从 100% 降到 10%，看 `w_opd·opd_loss` 对总 loss 的贡献是否随之缩小 10×。两种语义都可用，但必须知道是哪种并写进配置文档。

**C2 — DP 不变性单测（最强的 bug(b) 判据）。**
同一份数据、同一 seed、关 dropout，分别以 world_size = 1 / 2 / 4 跑 1 step，比较四项 loss 标量与 grad_norm，应在 1e-5 内一致。

**C3 — micro-batch 不变性单测。**
固定 mini_batch，把 `ppo_micro_batch_size_per_gpu` 从 32 降到 8（micro 数 ×4），loss 与 grad_norm 应不变。若 grad_norm 随 micro 数变化 → 某项分母写死在 micro 级。

**C4 — research 项是否绕开归一。**
若 RL_research 像 SEED 那样另起 forward/backward，它**完全绕过** `batch_num_tokens`。SEED 的做法是自己乘 `world_size / global_batch_size`（`refs/SEED/verl/workers/actor/dp_actor.py:739`）。对照检查我们的 research 项有没有等价缩放；没有的话 w_res 的有效尺度会随 batch 内 research 样本数变化。

**C5 — grad-accum 交叉验证。**
把 `ppo_mini_batch_size` 翻倍（累积步数翻倍）跑 1 step，grad_norm 应基本不变；若系统性缩小约 2×，说明某项被按 micro 数重复除。

**C6 — offload 断言。**
run 配置里断言 actor 的 `optimizer_offload=False`、且 fsdp2 的 `offload_policy` 未开（`transformer_impl.py:433-441`）。ref policy 开 offload 无害。

**C7 — metric 语义。**
确认四项的 metric 聚合都是 SUM 而非 MEAN（`losses.py:60-80` 的规则），否则 swanlab 上的 loss 曲线与真实反传量不成比例——与已知的"loss 是 SUM 聚合"一致，改动任一项时要一起复核。

---

# B) 源码：RL + OPD 联合调度侧

## B1. SEED（`work/refs/SEED/`）

### ① skill 生产管线怎么接进训练循环

- **入口**：`verl/trainer/ppo/ray_trainer.py:2340 _prepare_seed_teacher_signals`，位置在 rollout 之后、advantage 之前，与主循环**同一个 batch**。
- **谁产**：`seed/analysis.py:253 SEEDEpisodeAnalyzer`，backend 可选 `openai` 或 `policy_vllm`（用 policy 自己产 skill，即 self-evolve）；ALFWorld 脚本用 `policy_vllm`。
  输入由 `analysis.py:184 build_episode_records` 把一条轨迹解码成 `(observation, response, step_reward, action_valid)` 的 step 记录，输出 JSON：`episode_skill` + `step_skills`。
- **何时**：**每个 training step 都跑**。非 `policy_vllm` 后端时丢进 `ThreadPoolExecutor` 与后续阶段重叠（`ray_trainer.py:3145-3152`）。
- **进不进梯度：默认不进。** skill 文本只用来拼 teacher prompt。只有同时开 `algorithm.seed.skill_gen.enable=True` 且 `actor.skill_gen_loss_coef > 0`，"产 skill 的那次生成"本身才作为 PG 样本进梯度——**这正是我们 RL_research 的对应物**。
  **默认配置两个都是关的**（`verl/trainer/config/ppo_trainer.yaml:59` 与 `:294`），ALFWorld 脚本也没打开。
- **skill_gen 的 reward**（`seed/skill_gen.py:44-91`）：
  `reward = downstream_gain_coef · gain + valid_json_bonus + non_empty_skill_bonus − too_long_penalty`，clip ±2；
  其中 `gain` = 在成功轨迹的 OPD 激活 token 上取 `teacher_gap` 均值（`ray_trainer.py:785-811`），失败轨迹置 0（`failed_reward_mode=zero`）。

> **关键差异**：SEED 的 research reward 是"文档让 teacher 比 student 高多少 logprob"（**一步、密集**），
> 不是 TRACE-OPD 的"下游 rollout 成功率"（**两步、稀疏**）。

### ② batch 构成与 `opd_step_mask` 生成逻辑

- 一个 batch 是所有轨迹所有 step 展平；**RL 项吃全部 step，OPD 项只吃 `teacher_signal_mask = episode_skill_mask | step_skill_mask`**（`ray_trainer.py:2881`）。
  该 mask 是 step 级 1D，在 `core_algos.py:528-541` 与 `response_mask` 相乘变 token 级。
- **两条 critical mask 生成路径**：
  1. **实际默认路径** `ray_trainer.py:2536-2540`——凡"该轨迹的分析成功产出非空 skill"，**该轨迹全部 step 置 True**，即 episode 级全量，**不做步筛**。
  2. **统计路径** `seed/analysis.py:76-181 select_critical_steps_by_stats`——按 `anchor_obs` 分组（复用 gigpo 的 step group），组内 reward 方差需 ≥ `var_quantile`(默认 0.75) 分位且组大小 ≥ `min_group_size`(2)；
     样本得分 `group_var × max(group_mean − reward, 0)`；每轨迹取 `topk_per_traj`(=1)。
     语义："**同一状态下别人做得更好而你做砸了的那一步**"。
- **teacher log-prob**（`ray_trainer.py:2773-2820`）：把 skill 拼进 prompt 重建 prompt batch（`traj_collector.build_prompt_batch`），与原 responses 拼接后调 `actor_rollout_wg.compute_log_prob` ——
  **teacher = 同一份权重 + 文档前缀**，与我们一致（moving teacher）。
- **OPD loss 本体**（`core_algos.py:559-565`）：`gate = sigmoid(β·(teacher_logp − student_logp)).detach()`；`loss_mat = gate × (teacher_logp − log_prob)`；`agg_loss(loss_mat, opd_mask, loss_agg_mode)`。

### ③ λ 之外的平衡/调度手段（最有价值的部分）

- **互斥而非叠加**：`ppo_trainer.yaml:272-273` 明写两个 `*_teacher_advantage_w` "ignored when `actor.opd_loss_coef > 0`"——
  **skill 要么走 advantage 要么走 OPD loss，不并存**。
- **时间窗开关**：`algorithm.seed.opd_start_after_steps` / `opd_stop_after_steps`（`ray_trainer.py:585-595`、`844-882`），是**开/关**而非线性 ramp。ALFWorld 脚本两者都是 `null`（全程开）。
- **数据筛**：`seed.failed_only` / `failed_only_after_steps`——训练后期只对失败轨迹做分析 + OPD（`analysis_mode` 从 `teacher_bootstrap` 切到 `failed_episode_opd`）。
- **置信门**：`opd_gate_beta = 5`，teacher 不强于 student 的 token 自动降权。
- **★ 研究侧梯度隔离**：`skill_gen` 的反传是**独立的 `zero_grad → backward_skill_gen_loss → _optimizer_step`**
  （`verl/workers/actor/dp_actor.py:971-980`），与 RL/OPD **完全不在同一次反传里**。
  → **SEED 对"角色梯度互相冲掉"的实际答案是：时间上交替，而非空间上加权。**
- 且其归一是对的：`dp_actor.py:739`
  `weighted_loss = loss_coef × (world_size / global_batch_size) × Σ loss_terms`，与论文 3 附录 D 的修复形式同构。

> ⚠️ **不要连坑一起搬**：SEED 的 RL/OPD 主路径仍是老 verl——`dp_actor.py:946` 是
> `loss = policy_loss / self.gradient_accumulation`，配合 `core_algos.py:410-411` 的局部 `masked_mean`，
> **正是论文 3 §2.2 描述的 mean-of-means 形态**。我们的 verl 0.9.0.dev 已不是这样。

### ④ 全套超参（`examples/seed_trainer/_common/alfworld.sh`）

```
lr 1e-6 | ppo_mini_batch_size 256 | ppo_micro_batch_size_per_gpu 32
train_batch 16 任务 × group 8 | env.max_steps 30 | history_length 5
use_kl_loss=True, kl_loss_coef=0.01, kl_loss_type=low_var_kl
opd_loss_coef = 0.01          ← w_opd / w_rl 的实际比例
opd_gate_beta = 5.0
skill_gen_loss_coef 未传 (=0，关)
step_advantage_w = 0 | episode_skill_teacher_advantage_w = 0 | step_skill_teacher_advantage_w = 0
gamma 0.95 | invalid_action_penalty_coef 0.1
max_prompt_length 2048 | max_response_length 512
total_epochs 160 | 8 GPU | Qwen2.5-3B-Instruct (SFT 后)
```

---

## B2. SDAR（`work/refs/SDAR/`）

### ① 三个入口同一 trainer

`verl/trainer/main_sdar.py:167-169` 用 `SkillSDRayTrainer`，并在 `open_dict` 中把 `algorithm.sdar.*` 拷进
`actor.use_sdar_loss / sdar_loss_coef / sdar_gate_beta`（`main_sdar.py:47-50`）。

### ② skill 来源与 SEED 完全不同：离线静态文档，不在线提炼

`skills/alfworld/*.md` 由 `skill_mapping.json` 索引；`SkillProvider`（`verl/trainer/ppo/rlsd_utils.py:33-142`）按 gamefile / data_source / prompt 关键词选文件，`skill_all=true` 时拼全部。
teacher prompt = `"[Privileged Skill Information]\n{skill}\n\n" + 原 prompt`（`rlsd_ray_trainer.py:111-112`），左截断到 `max_prompt_length`。
→ **skill 完全不进梯度、训练中不更新。** 所以 SDAR 侧没有"角色遗忘"问题可参考，它只提供"蒸馏与 RL 如何共存"的经验。

### ③ batch 构成与组结构：无 step 级筛选

`skillsd_ray_trainer.py:160-162` 对整批每个样本都算 `teacher_log_probs`，SDAR loss 用**完整 `response_mask`**（`sdar_utils.py:52`）。
→ **SDAR = 全量 token OPD，SEED = 关键步 OPD。**

### ④ loss 本体

`sdar_utils.py:42-52`：
```
delta = teacher_logp − student_logp.detach()
gate  = sigmoid(gate_beta · delta).detach()
loss  = agg_loss(gate × (teacher_logp − student_logp), response_mask, loss_agg_mode)
```
与 SEED `compute_opd_loss` **数学同构**。
另有 SDL 变体（`skillsd_utils.py:40-50`）：`ℓ = student − teacher`，`k3 = exp(clamp(−ℓ, max=20)) − 1 + ℓ`，再乘在线重要性比 `ρ_on = exp(clamp(student − old, max=10))`。

### ⑤ 联合调度手段

- 系数：`sdar_coef = 0.01`、`gate_beta = 5.0`（`examples/sdar_trainer/run_alfworld_qwen3.sh:7-8`）——与 SEED **独立收敛到同一数量级**。
- **`pg_loss_coef` 是唯一的 RL 侧配重**（`verl/workers/actor/dp_actor.py:412-419`）。三个 example 族就是这一个旋钮的消融：

  | 族 | 差异 | 含义 |
  |---|---|---|
  | `sdar_trainer` | pg=1, β=5 | RL + 门控 OPD |
  | `opsd_trainer` | **`pg_loss_coef=0`** | 纯 OPD，无 RL |
  | `grpo_opsd_trainer` | **`gate_beta=0`** → gate ≡ 0.5 | 无门控 OPD + RL |

  这是一套现成、干净的消融设计，可直接照搬到我们的 w_opd / w_rl 消融。

- **★ λ 调度是死代码（重要教训）**：
  `skillsd_ray_trainer.py:57-62 _get_sdl_lambda` 实现了线性 warmdown，但 `:227-230` **只把它写进 metrics**；
  实际系数是 `main_skillsd.py:49` 在**启动时一次性**拷进 `actor.sdl_loss_coef` 的静态值，`dp_actor.py:441` 读的就是这个静态值。
  → **`warmdown_steps` 对 loss 完全无效**；shipped 脚本一律写 `warmdown_steps=-1`，正好掩盖了它。
  对照 RLSD：它的 λ（`rlsd_ray_trainer.py:179-183`）用在 trainer 侧的 advantage 计算里（`rlsd_utils.py:145-183`），所以**是有效的**。

> 这是"**调度写在 trainer、系数读在 actor、中间没有每 step 传递通道**"的真实样本，
> 正是我们担心的第 ② 类框架坑。任何 λ ramp 都必须做端到端验证。

### ⑥ 超参（`examples/sdar_trainer/run_alfworld_qwen3.sh`）

```
lr 1e-6 | ppo_mini_batch_size 256 | ppo_micro_batch_size_per_gpu 32
train_batch 16 × group 8 | env.max_steps 50
use_kl_loss=True, kl_loss_coef=0.01, low_var_kl
sdar_coef 0.01 | gate_beta 5.0 | skill_all false
max_prompt_length 2048 | max_response_length 512
tensor_model_parallel_size 2 | total_epochs 150 | 8 GPU
Qwen3-1.7B，且 +data.apply_chat_template_kwargs.enable_thinking=False
```

---

# C) 防角色遗忘的落地建议

## 必做（不依赖症状）

1. **把 RL_research 从四项联合反传中拆出来**，改成独立 `zero_grad → backward → step`，或至少独立 backward + 自带 `world_size/global_batch_size` 归一。
   依据：SEED 正是这么做的（`dp_actor.py:971-980`，归一在 `:739`）。
   当前四项一次反传时，research 样本数远小于 executor step 数，**w_res 的有效尺度会随 batch 内 research 占比逐 step 漂移**——这是"executor 冲掉 research"最廉价的解释，必须先排除。

2. **先跑 C2 / C3 两个不变性单测（DP 数、micro 数），再谈任何 w_\* 调参。**
   论文 3 的教训是框架 bug 能造成 5.7 分的假差距，所有 w_* 的相对结论都建立在归一正确之上。

3. **分开记录四项的 grad norm 而非 loss。**
   MIFO §3.1 / Fig 3 的核心量是参数更新幅度；判断"executor 冲掉 research"唯一直接的指标是两侧梯度范数之比，loss 值在不同 mask 下不可比。

4. **断言 actor 侧 offload 关闭**（清单 C6）。

5. **冷启动顺序按 MGS §3.1 的非对称性排**：高熵、行为灵活的角色先训。
   research（自由文本写文档）是高熵侧，executor（受限动作空间）是低熵侧——现有的"research SFT 先冷启动"是对的；
   **绝不要改成"先 executor RL 再补 research"**，那会吃 Fig 2 里 −9.6 那一档的 rigidity。

## 症状驱动

6. **若 research 能力退化**（文档质量 / 长度 / JSON 合法率下降）：上 MIFO 的参数冻结。
   在 research 步之前，用最近若干 executor RL 步的 |Δθ_j| 建 `C̄_i = α·C_{i−1} + (1−α)·Δθ_RL,i`（α=0.5），冻结 top-k=50%，research 步后解冻。
   **先用 random-50% 冻结验一次**——MIFO 附录 E.4 / Fig 13 自承 random ≈ TopK；若 random 也有效，才说明"限制幅度"是对的解。

7. **若 research 输出变长 / 发散**：只在高熵 token 上算 research 的 SFT loss，阈值取每样本熵的 top ρ=0.2 分位。
   Table 3 显示 ES 单独把 response 从 4421 压到 3246，而 PF 单独会拉到 4490，二者互补。

8. **若怀疑是梯度冲突而非幅度**：先测，别直接上手术。
   照 MGS §3.4 / 附录 F，每 N 步存 research 与 executor 两侧的分模块梯度，算 per-module cos 与 norm ratio。
   判据：**全局 cos 长期 ≥ 0 而仅个别模块为负 → 用模块级 MGS；全局 cos 频繁 < 0 → 才考虑全局手术**（Table 2 显示 GGS 常不如模块级、有时不如 naive）。

9. **若确认冲突要上 MGS**：T=2，开销可接受（显存 `O(T×N/world_size)`，时间 ~163s/30h）。
   **务必把 LayerNorm 纳入手术范围**——Fig 7 显示排除 LayerNorm 会让两轴同时暴跌，尽管它参数最少。

10. **若 research 是"从未学会"而非"学会又忘"**：按论文 3 §4.1 的 Llama 教训，这是 SFT 冷启动不足，不是遗忘。
    判据是 research reward 从头贴地板且 500 步不涨（LUFFY 在 Llama 上训练 reward 500 步平坦）。
    此时应加大 SFT 而非调 w_res——与我们已有的"**OPD 只能调已有行为、SFT 是引入新动作的唯一通道**"完全一致。

---

# D) RL + 蒸馏联合的成熟实践清单

1. **λ_opd 取 0.01 量级。** SEED `opd_loss_coef=0.01` 与 SDAR `sdar_coef=0.01` 是两个独立工作在同一环境（ALFWorld）上收敛到的同一数量级。
2. **置信门 `sigmoid(β·gap)` 且 detach，β=5。** teacher 不强于 student 的 token 自动降权，天然抑制 moving-teacher 正反馈；`gate_beta=0` 是现成的消融臂（`grpo_opsd`）。
3. **teacher = 同权重 + 文档前缀**，走 `compute_log_prob` 而非另起模型；prompt 左截断。
4. **蒸馏与 advantage 二选一，不叠加**（SEED `ppo_trainer.yaml:272-273` 的显式互斥）。
5. **λ 调度用时间窗（start/stop step）而非连续 ramp**；若一定要 ramp，**必须端到端验证系数每 step 真的传到了 actor**（SDAR `_get_sdl_lambda` 是反面教材）。
6. **需要筛关键步时**，用"组内 reward 方差高 + 低于组均值"的统计判据（`select_critical_steps_by_stats`），每轨迹 top-1。
7. **研究/提炼侧的 loss 独立反传，并自带全局样本数归一。**
8. **蒸馏侧的最小可观测集**：`gate_mean`、`gate_active_ratio`、`teacher_gap_mean`、`active_token_ratio`（SEED 四个 metric 齐备，`dp_actor.py:954-959`）——判断"蒸馏在不在干活"的最小充分集。
9. **保留 per-token gap dump 开关**（SDAR `SAVE_SDAR_DEBUG=1`，按 `test_freq` 落 jsonl，`skillsd_ray_trainer.py:233-276`），出问题时能直接定位是哪些 token 在被拉。

---

## 附：本文引用的源码位置速查

| 主题 | 位置 |
|---|---|
| verl 全局归一 `agg_loss` | `trace_opd/verl/verl/trainer/ppo/core_algos.py:1138-1173` |
| verl FSDP 全局 token 数 all_reduce | `trace_opd/verl/verl/workers/engine/fsdp/transformer_impl.py:643-657` |
| verl CPUOffload 强制关闭（FSDP1）/ 残留开关（FSDP2） | 同上 `:399-410` / `:433-441` |
| verl sft_loss 归一模板 + SUM 聚合规则 | `trace_opd/verl/verl/workers/utils/losses.py:31-52`, `:60-80` |
| SEED OPD loss | `refs/SEED/verl/trainer/ppo/core_algos.py:495-576` |
| SEED skill_gen 独立反传 | `refs/SEED/verl/workers/actor/dp_actor.py:670-759`, `:971-980` |
| SEED 老式 `/gradient_accumulation`（勿搬） | `refs/SEED/verl/workers/actor/dp_actor.py:946` |
| SEED teacher 信号构建 | `refs/SEED/verl/trainer/ppo/ray_trainer.py:2340-2894` |
| SEED 关键步统计筛选 | `refs/SEED/seed/analysis.py:76-181` |
| SEED skill_gen reward | `refs/SEED/seed/skill_gen.py:44-91` |
| SEED 超参 | `refs/SEED/examples/seed_trainer/_common/alfworld.sh` |
| SDAR loss | `refs/SDAR/verl/trainer/ppo/sdar_utils.py:14-67` |
| SDAR `pg_loss_coef` 配重 | `refs/SDAR/verl/workers/actor/dp_actor.py:412-419`, `:446-459` |
| SDAR λ warmdown 死代码 | `refs/SDAR/verl/trainer/ppo/skillsd_ray_trainer.py:57-62`, `:227-230`；`verl/trainer/main_skillsd.py:49` |
| SDAR SkillProvider / teacher prompt | `refs/SDAR/verl/trainer/ppo/rlsd_utils.py:33-142`；`rlsd_ray_trainer.py:39-161` |
