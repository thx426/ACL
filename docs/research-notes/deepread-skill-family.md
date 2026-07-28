# Skill×RL 多能力联合训练三库源码深读报告

> 调研对象:Skill1(arXiv 2605.06130)、SKILL0/SkillZero(arXiv 2604.02268)、SkillRL(arXiv 2602.08234)
> 目的:为 TRACE-OPD 新增第四目标(policy-as-researcher 的 research 能力 RL)提供可移植实现经验
> 日期:2026-07-28

基路径统一省略为 `<CB>/` = `/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/.conductor/branches/CD-002/codebases/`

**一句话结论**:三库都号称"多能力共同进化",但**没有任何一个真的把多个 loss 项加权求和做一次反传**。它们的统一做法是**把所有能力的信号折叠成"每条轨迹一个标量 reward",再走单一 GRPO**;能力之间的平衡靠 **reward 量纲(λ 系数)、组内对照(control/experiment 半组)、以及"辅助信号间歇施加/退火"** 三招,而不是靠 loss 权重。这与 TRACE-OPD 计划的 `L = w_opd·OPD + w_rl·RL_exec + w_sft·SFT + w_res·RL_research` 四项同时反传是**根本不同的路线**——三库无一提供该路线的现成实证。

---

## 一、Skill1(arXiv 2605.06130,最高优先)

### 1.1 文件定位与相对 verl-agent 的改动面

| 角色 | 文件 |
|---|---|
| 入口 | `<CB>/skill1/verl/trainer/main_ppo.py:161-162`(实例化 `TrajectoryCollector`) |
| **真实 rollout 主循环** | `<CB>/skill1/agent_system/multi_turn_rollout/rollout_loop.py`(1138 行) |
| 死代码 | `metarl_rollout_loop.py` —— `multi_turn_rollout/__init__.py:16` 只导出 `rollout_loop` 的类,**metarl 版本不被任何地方 import**,勿参考 |
| 技能库/reward 计算 | `agent_system/environments/env_manager.py`(4900+ 行,5 个 env 类各自复制一份同样逻辑;ALFWorld 类在 `:887-2046`) |
| 技能库数据结构 | `agent_system/memory/memory.py:208-543` |
| reward→token | `agent_system/reward_manager/episode.py:72-82` |
| 优势 | `verl/trainer/ppo/core_algos.py:114-160`(GRPO)、`gigpo/core_gigpo.py:138-171`(GiGPO,**发布脚本未用**) |
| 训练循环改动 | `verl/trainer/ppo/ray_trainer.py:200-225`(invalid-action 惩罚)、`:1394-1431`(adv)、`:695-736`(ref-policy 产文档) |

上游 verl 是老版本(v0.2 时代,含 `vllm_v_0_5_4`),diff 噪声极大,不建议逐文件 diff;真正的论文改动全部集中在 `agent_system/` + `rollout_loop.py` + `ray_trainer.py` 三处。

### 1.2 多能力如何进 loss(精确形态)

一次 rollout 内,**同一个 policy 依次扮演 4 个角色**,每个角色的输出都是一段独立生成的 response:

```
Phase 0  query      → 生成检索 query          (rollout_loop.py:405-448)
Phase 0b rerank     → 对 top_k=3 候选重排序   (rollout_loop.py:450-498)
Phase 1  play ×50   → 环境交互                (rollout_loop.py:500-563)
Phase 2  distill    → 反思写技能 JSON         (rollout_loop.py:579-663)
```

**关键实现事实(与论文叙述有出入,务必注意)**:

1. **query / rerank / play 三种 step 被 append 进同一条轨迹 `total_batch_list[i]`,共享同一个 `uid` 和 `traj_uid`**(`:448`、`:498`、`:556`)。它们各自的 step-level `rewards` 是:
   - query step:`q_rewards[i] * λ_rerank`,在 distill 阶段结束后**回填**到 `total_batch_list[i][0]['rewards']`(`:659-662`)
   - rerank step:`NDCG * λ_rerank`(`:490`)
   - play step:环境 reward(ALFWorld = `10.0 * won`,见 `env_package/alfworld/envs.py:48-52`)

2. **`credit_assignment()` 把整条轨迹所有 active step 的 `rewards` 求和,得到一个标量,再赋给该轨迹的每一个 step**(`rollout_loop.py:950-966`)。所以:
   ```
   score(轨迹 i) = Σ play_rewards + NDCG·λ_rerank + q_contrastive·λ_rerank
   ```
   这个同一个标量被 `episode.py:82` 写到**每个 step response 的最后一个 token** 上,再由 GRPO 按 `uid` 组归一化。**三种能力共享完全相同的 advantage**——这就是论文"one reward, unified evolution"的全部含义。没有多 loss、没有多头、没有加权求和。

3. **⚠️ 最重要的发现:distill(技能生成)阶段的 token 从未进入训练 batch。** `rollout_loop.py` 全文的 `total_batch_list[i].append(...)` 只有三处(448/498/556),distill 阶段的 `batch` 在 `:616` union 完输出、`:618` decode 出文本后就只用于计算 reward 和写日志,**没有任何 append**。
   更进一步:`:651-652` 把 distill reward 加到 numpy 数组 `episode_rewards[i]`,但发布脚本设了 `+algorithm.credit_assignment=True`(`launch_scripts/alfworld/train_alfworld.sh:144`),而 `credit_assignment` 分支(`:884-887`)**完全忽略** `total_episode_rewards`,只重算 step `rewards` 之和。→ **在发布配置下,distill reward 对梯度零贡献,distill 输出的 token 也不训练。** "蒸馏能力"实际只通过"写进技能库→影响后续 batch 的检索内容"这一**数据通路**演化,不是 RL 通路。
   (只有 `credit_assignment=False` 走 `no_credit_assignment`(`:987-1024`)时 distill reward 才生效,但两个发布脚本都设了 True。)

### 1.3 各能力的 reward 精确形式

**rerank(选择-利用)= NDCG**,`env_manager.py:1592-1634`:
ground truth 是候选按 `utility_score` 降序;`DCG = Σ gain_i / log2(i+2)`,`reward = DCG_pred / DCG_ideal ∈ [0,1]`。解析失败或候选 <2 → reward = 0(**注意:失败与最差表现同为 0,不可区分**)。训练时只保留 rerank 后的 top-1 技能进 prompt(`:1585-1586`,`rerank_train_top1=True`)。

**query(选择)= 组间对照差**,`env_manager.py:1109-1156`:
```
contrastive = experiment_wins/n_exp − control_wins/n_ctrl
```
组内 `group_n=8` 个 rollout 按位置切成 control(不带技能)/ experiment(带技能)两半(`:1013`,`group_split_index = group_n // 2`)。**这是 TRACE-OPD research reward 最直接的对标物**。

⚠️ 但发布脚本设 `full_group_memory=True`(`train_alfworld.sh:161`)→ `group_split_index=0`(`:1014-1015`)→ control 组为空、`n_ctrl = max(0,1) = 1`、`control_wins` 恒为 0 → **contrastive 退化成 experiment 组的裸成功率,对照消失**。且 `:72-73` 有断言 `full_group_memory` 与 `group_outperformance` 互斥。

**distill(蒸馏)= 一阶差分**,`env_manager.py:1289-1296`(`distill_reward_type=first_order_diff`):
```
u_hat  = max(检索到的技能的 utility_score)     # u_hat_aggregation=max
r_tau  = 1.0 if 任务成功 else 0.0
reward = r_tau − u_hat
```
即"这次实际结果**超出**所用技能历史效用的部分"。另有 potential-based 的 `improvement = max(0, φ_t − φ_{t-1})`(`:1345-1349`),φ 来自 JSON 里 subtask 完成比例、成功则强制 1.0;组内可选中心化(`:1367-1372`,发布脚本关掉 `group_relative_intrinsic_rewards=False`)。基线 `task_potential_history` 用**单调取大**更新(`:1376-1380`),EMA 代码被注释掉了(`:1381-1391`)。

**技能库效用 = EMA**,`memory.py:365-368`:`u ← (1−α)·u + α·score`,α=0.05,score 为该 rollout 是否成功(0/1)(`env_manager.py:1166-1193`)。

**检索 = 相关度与 UCB 的凸组合**,`memory.py:493-503`:`final = w_rel·relevance + (1−w_rel)·(utility + scale·√(ln N / count))`。⚠️ 发布脚本设 `relevance_weight=1.0` → **UCB 项权重为 0,`train_retrieve_type=ucb` 形同虚设**,退化为纯相似度检索(README 里 "Pure similarity retrieval" 与此一致,但与配置名字矛盾)。

### 1.4 组结构

- 组 = 同一任务的 `env.rollout.n = 8` 个 rollout,`uid` 在 `rollout_loop.py:362-368` 按 `i % n == 0` 换新。
- **异质样本不隔离**:query / rerank / play 三种 step 全在同一 `uid` 组里一起做 mean/std 归一化。
- `core_algos.py:121,135-137` 的 `compute_mean_std_cross_steps=True`(默认)意味着**组内统计量跨 step 计算**——每个 step 是一个样本。由于同一轨迹所有 step 携带相同分数,**长轨迹在组均值里的权重更大**(隐式长度加权)。
- control / experiment 两半也**不隔离**(`env_manager.py:1042-1047`),这是设计意图:组均值天然成为"无技能基线",带技能的 rollout 只有赢过无技能的才拿到正 advantage。
- 全对/全错组:DAPO 式动态过滤在 `utils.py:133-184`(`filter_group_data`,组内 reward 全相同则整组丢弃,反复重采样至多 `max_num_gen_batches=10` 次),但 **`algorithm.filter_groups.enable` 默认 False(`verl/trainer/config/ppo_trainer.yaml:255-257`)且两个发布脚本都没打开** → 实际保留全对/全错组(advantage=0,只是空转)。

### 1.5 稳定化手段全清单

| 手段 | 位置 | 发布配置下是否生效 |
|---|---|---|
| 辅助 reward 量纲压制:λ_rerank=λ_distill=0.3,而任务 reward=10.0×won | `train_alfworld.sh:163-164`;`envs.py:48-52` | ✅ 辅助信号 ≤ 主信号的 **3%** |
| 蒸馏系数指数退火 `exp(−5·progress)`;`hard_cutoff` 模式在 10% 进度后硬切 0 | `rollout_loop.py:54-79` | ❌ `intrinsic_reward_coefficient=1.0 > 0` → 走常数分支,不退火 |
| 内在 reward 上下 clip | `rollout_loop.py:635-638` | ❌ 默认 −5 → 不 clip |
| 组内中心化内在 reward | `env_manager.py:1367-1372` | ❌ 关闭 |
| 记忆 warmup(前期禁检索) | `env_manager.py:1018-1023` | ❌ `memory_start_cutoff=0.0` |
| KL to frozen ref,`low_var_kl`,coef 0.01 | `train_alfworld.sh:119-122` | ✅ |
| 非法动作惩罚 −0.1(打在最后一个 token) | `ray_trainer.py:200-225` | ✅(= 成功 reward 的 1%) |
| 技能库退休/剪枝(`utility × log2(count+1)`) | `memory.py:371-421` | 部分(`check_and_prune` 调用被注释掉,`env_manager.py:1084`) |
| GRPO 组归一化(除 std) | `core_algos.py` | ✅ `norm_adv_by_std_in_grpo` 默认 True |
| **用冻结 ref policy 生成技能文档** | `ray_trainer.py:695-736`;`rollout_loop.py:597-606` | ❌ `distill_reference_policy=False` |

⚠️ 最后一条对 TRACE-OPD 有直接价值但**实现是残缺的**:`_get_ref_rollout_wg` 的 docstring 明说 `ref_policy_wg` 没有 rollout 引擎、调用会崩,所以它**永远返回 `actor_rollout_wg`**;非 LoRA 情况下只是打了个 `meta_info['use_ref_policy']=True` 的标记,**实际仍用 actor 权重生成**,并打印 WARNING(`:723-732`)。即"冻结 teacher 产文档"在该库里只对 LoRA 成立。

### 1.6 蒸馏项(token 级)

**没有。** 全库无 token-level KD/teacher-forcing loss,`distill` 一词在此指"从轨迹提炼技能文本",不是 policy distillation。唯一的 KL 是 verl 标配的 `kl_loss` to frozen ref(coef 0.01,`low_var_kl`)。因此**不存在"蒸馏项与 RL 项尺度协调"的现成实现可抄**。

### 1.7 超参全表(ALFWorld,`launch_scripts/alfworld/train_alfworld.sh`)

| 项 | 值 | 行 |
|---|---|---|
| adv_estimator / lr | grpo / 1e-6 | 102,115 |
| train_batch × rollout.n | 16 × 8 = 128 traj | 20,22,169 |
| ppo_mini / micro_per_gpu | 256 / 16 | 117-118 |
| max_prompt / max_response | 16384 / 2048 | 107-108 |
| kl_loss | True, coef 0.01, low_var_kl | 119-122 |
| invalid_action_penalty | True, 0.1 | 138-139 |
| top_k / α(EMA) | 3 / 0.05 | 146,148 |
| λ_distill / λ_rerank | 0.3 / 0.3 | 163-164 |
| intrinsic_coef / hard_cutoff | 1.0 / False | 141-142 |
| credit_assignment / step_gamma | True / 0.95(默认) | 144;`rollout_loop.py:42` |
| relevance_weight | 1.0(→UCB 失效) | 162 |
| full_group_memory | True(→对照失效) | 161 |
| env.max_steps / TP / GPU | 50 / 4 / 8×1 node | 168,127,175-176 |
| total_epochs / save / test | 150 / 50 / 20 | 177-180 |
| model | Qwen2.5-7B-Instruct | 30 |

WebShop 脚本(`launch_scripts/webshop/train_webshop.sh`)超参同构,仅 batch/长度不同。

### 1.8 已知问题与 workaround(源码注释)

- `ray_trainer.py:723-732`:ref-policy 生成技能的 WARNING(见 1.5)。
- `rollout_loop.py:986-1025`:`no_credit_assignment` 的注释坦承"discounted return 该填什么取决于算法用法",实现上直接填 final reward。
- `env_manager.py:1381-1391`:EMA 基线更新被整段注释,改为单调取大——即基线只升不降,任务变难时不会回落。
- `core_gigpo.py:110`:`assert traj_active_masks.all()` 要求同轨迹 active mask 全 1,与 query/rerank step 的 `active_masks=0`(当 `selection_trainable=False` 或 λ=0 时,`rollout_loop.py:444,494`)冲突 → **GiGPO 与 selection 关闭态不兼容**,这大概是发布脚本用 `grpo` 而非 `gigpo` 的原因。
- 5 个 env 类里同一段 reward 逻辑复制了 5 份(`step_distill` 在 `:527/1200/2362/3113/4167`),改一处必须改五处。

---

## 二、SKILL0 / SkillZero(arXiv 2604.02268)

### 2.1 机制精确形态

**技能生成者不存在,更谈不上被训练。** 技能是**手写静态 markdown**:`<CB>/skill0/skills/alfworld/{general_skills,pick_and_place,clean,heat,cool,look_at_obj_in_light}.md` + `skill_mapping.json`。

论文的"internalization"是一个**课程机制,不是 loss**(`agent_system/environments/curriculum_skill_manager.py`):
- 每次验证跑**两遍**:带技能 / 不带技能,取 per-task 成功率之差 `delta`(`verl/trainer/ppo/ray_trainer.py:1011-1022`)。
- `delta < 0` 的技能直接删除;其余按 delta 降序保留 top-`max_set`(`curriculum_skill_manager.py:154-176`)。
- `max_set` 按训练进度走 schedule `[6,3,0]`(`:104-114`)——**最终把 in-context 技能全部撤走**,逼 policy 把技能内化进权重。
- 断言 `num_validations % len(schedule) == 0`(`:64-68`)。

**第二个能力(唯一有独立 reward 的)= 观测压缩选择**,`agent_system/reward_manager/episode_with_compression.py:58-140`:
```
成功轨迹: r_comp = log(compression_factor) · coef        (coef=0.01)
失败轨迹: r_comp = −log(compression_factor) · penalty_coef (默认 0)
score    = episode_reward + r_comp                        (:139)
```
仍是**加到同一个标量上、打同一个 token**,单 GRPO 反传。

### 2.2 稳定化清单(SKILL0 独有的两招最值得抄)

1. **间歇施加**:`apply_compression_reward = global_steps % compression_reward_every_n_steps == 0`,发布配置 **N=8**(`episode_with_compression.py:67`;`scripts/train_alfworld_skillzero_7b.sh:71`)。辅助 reward **每 8 步才施加一次**,其余 7 步恒为 0。这是"防辅助信号淹没主信号"的最直白手段,且不改 loss 结构。
2. **对数压缩量纲**:用 `log(factor)` 而非 `factor−1`,配 coef=0.01(注释里保留了被弃用的线性版本 `:119`)。
3. 失败轨迹的惩罚系数**独立于**成功轨迹的奖励系数,默认 0——即**只在成功时给辅助奖励,失败时不惩罚**,避免把探索期的噪声灌进梯度。
4. 无 KL loss(`use_kl_loss=False`,`train_alfworld_skillzero_7b.sh:38`),靠 invalid-action 惩罚 0.1 + GRPO 组归一化稳定。
5. 三个断言强制系数非负、周期 ≥1(`:49-51`)。

### 2.3 超参(`scripts/train_alfworld_skillzero_7b.sh`)

grpo / lr 1e-6 / train_batch 16 × n=8 / mini 256 / micro 8 / prompt 3072 / resp 512 / **无 KL loss** / invalid 0.1 / curriculum `[6,3,0]` / compression coef 0.01、every 8 steps / total_training_steps 180 / 4 GPU / Qwen2.5-VL-7B。纯文本版 `train_alfworld_text_skill.sh`:prompt 5120、150 steps、无 curriculum(只挂静态 `alfworld_skills_nl.md`)。

---

## 三、SkillRL(arXiv 2602.08234)

### 3.1 机制精确形态

**技能生成者是外部冻结的 Azure OpenAI o3,完全不在梯度里**(`agent_system/memory/skill_updater.py:17-97`,`self.model = "o3"`)。技能库离线由 o3 生成(`skill_generation/{alfworld,search,webshop}.py`),产物落盘为 `memory_data/alfworld/claude_style_skills.json`。

"递归进化"是**训练循环里的数据级回路**(`verl/trainer/ppo/ray_trainer.py:941-1008`):
1. 从**训练 batch**(注释明确说是为了避免验证集泄漏,`:944-945`)统计各 task 的 success_rate;
2. **任一 task 成功率 < `update_threshold=0.4` 才触发**(`:961-971`),否则跳过;
3. 取失败轨迹交给 o3 分析,最多产 `max_new_skills=3` 条新技能(`skill_updater.py:93`),`dyn_NNN` ID 由本地重新分配以防冲突(`:124-134`);
4. `retrieval_memory.add_skills(...)` 写回技能库。

RL 侧是**纯 GRPO 执行任务,单一 outcome reward,无任何辅助 loss、无蒸馏**。检索**不做组内对照**——所有 rollout 一视同仁挂技能(`agent_system/environments/env_manager.py:150-161`)。能力提升靠**两阶段**:先 SFT(HF 上放了 `*-7B-SFT`),再 RL,不是联合。

### 3.2 稳定化清单

- 触发阈值门控(0.4)——**技能库只在表现差时才变**,避免库随机漂移干扰 policy;
- 每次最多 3 条新技能,增量受限;
- 检索时**动态技能全部保留**、静态技能只填满 `top_k` 剩余名额(`skills_only_memory.py:359-371`,注释解释:否则静态库一大,新技能会被 `[:top_k]` 静默切掉);
- KL to ref coef 0.01 `low_var_kl` + invalid 0.1(`run_alfworld_skills_train_update.sh:42-44,61-62`)。

### 3.3 超参(`examples/grpo_trainer/run_alfworld_skills_train_update.sh`)

grpo / lr 1e-6 / train_batch 16 × n=8 / mini 128 / micro 4 / prompt 4096 / resp 512 / KL 0.01 low_var_kl / invalid 0.1 / top_k 6 / update_threshold 0.4 / max_new_skills 3 / TP 2 / 8 GPU / test_freq 5 / 150 epochs。

---

## 四、三库横向对比

### 共同实践(三库一致,可视为该技术线的默认 answer)

1. **能力不进 loss,只进 reward 标量**。三库都是单 GRPO、单次 backward、单 advantage。"多目标加权求和一次反传"在这条技术线上**没有先例**。
2. **辅助能力的 reward 量纲被压到主 reward 的 1–3%**:Skill1 λ=0.3 vs 任务 10.0;SKILL0 coef=0.01 且 8 步一次;SkillRL 干脆不给辅助 reward。
3. **组 = 同任务 n=8 rollout,异质样本不隔离**,靠组均值当基线。
4. **技能库更新是"写文件"而非"回传梯度"**,与 RL 通路解耦,天然避免了双角色梯度耦合。
5. 稳定三件套完全一致:KL-to-frozen-ref coef 0.01(SKILL0 除外)、invalid-action 惩罚 0.1、GRPO 组归一化。
6. lr=1e-6、train_batch=16、group=8 三库完全相同(都继承 verl-agent 默认)。

### 分歧

| 维度 | Skill1 | SKILL0 | SkillRL |
|---|---|---|---|
| 技能由谁产 | policy 自己(但**不训练该阶段 token**) | 人手写,固定 | 外部冻结 o3 |
| 组内对照 | 有(control/exp 半组)但发布配置关掉 | 只在**验证**层面对照(跑两遍) | 无 |
| 防淹没主手段 | λ 系数 + 指数退火(未启用) | **间歇施加 every-N-steps** | 阈值门控 + 不给辅助 reward |
| 技能撤除 | 无(靠 utility 退休) | **课程 `[6,3,0]` 强制清零** | 无 |
| 阶段划分 | 单阶段 | 单阶段 | **SFT → RL 两阶段** |

---

## 五、对 TRACE-OPD 双角色联合训练的可移植经验

1. **[对照 reward 直接可抄,但必须保住对照组]** Skill1 的 `contrastive = exp_success_rate − ctrl_success_rate`(`env_manager.py:1152-1156`)就是我们 research reward 的同构物:N 份文档 × K 次 doc-conditioned rollout 求成功率,减去**无文档基线**。⚠️ Skill1 自己在发布配置里用 `full_group_memory=True` 把对照组清空了(`:1014-1015`),导致 reward 退化成裸成功率、失去"文档是否有增益"的语义。**我们的 8×8 设计必须显式保留裸 rollout 作为 baseline,且要么放在同组当均值,要么单独算差值——不要两者都不做。**

2. **[辅助目标量纲定在主目标的 1–3%,而不是 0.5–1×]** 三库一致把辅助信号压到 1–3%(Skill1: 0.3/10.0;SKILL0: 0.01×log)。我们的 `w_res` 若按"两个 RL 目标同等重要"取 0.5,与全部已知实践相差 15–50 倍。建议 `w_res` 初值按"research 项对总 loss 的贡献 ≈ RL_exec 的 3–5%"反推,再逐步抬。

3. **[间歇施加优于连续小权重]** SKILL0 的 `compression_reward_every_n_steps=8`(`episode_with_compression.py:67`)是三库中唯一**结构化**的防淹没手段:辅助梯度只在 1/8 的 step 出现,其余步完全干净。对"轨迹级、超长、噪声大"的 RL_research 特别合适——把它做成**每 K 步才反传一次的 research 步**,比每步给一个小权重更能避免噪声持续污染 executor 的梯度方向。**这条最推荐优先试。**

4. **[退火/门控要留,但别忘了打开]** Skill1 写了完整的指数退火 `exp(−5·progress)` 和 10% 硬切(`rollout_loop.py:54-79`),SkillRL 写了 `update_threshold=0.4` 门控——但 Skill1 的退火**在发布配置里因为 `intrinsic_coef=1.0 > 0` 走了常数分支而从未生效**。教训:这类开关要在训练日志里打印实际生效值,否则会以为在退火其实没有。

5. **[全对/全错组:三库都保留、不过滤]** DAPO 式过滤代码 Skill1 有(`utils.py:133-184`)但默认关闭。对我们的 research 组更值得开:N=8 份文档若 K 次 rollout 成功率全等,该组 advantage 恒 0,白烧 8×8=64 次 rollout。**建议在 research 侧单独启用组过滤**(执行侧可不开),因为 research rollout 的采样成本是 executor 的 K 倍。

6. **[异质角色同组归一化是可行的,但有隐式长度加权]** Skill1 把 query/rerank/play 三种截然不同的 response 放进同一 GRPO 组共享 advantage,且 `compute_mean_std_cross_steps=True`(`core_algos.py:121`)使**每个 step 是一个样本**——长轨迹在组均值中权重更大。若把 research 会话(超长、少步)与 executor 轨迹(多步)混同一组,executor 会主导组统计量,research 的 advantage 会被系统性压偏。**建议 research 与 executor 用不同的 group key(不同 uid 前缀),各自归一化后再按 w 加权**,不要指望共享一个组均值。

7. **[冻结 teacher 产文档:该路线在开源实现里是残缺的]** Skill1 的 `distill_reference_policy` 意图正是我们现在的 researcher(冻结权重产特权文档),但 `ray_trainer.py:707-732` 的注释明说 `ref_policy_wg` **没有 rollout 引擎、调用 generate 会崩**,非 LoRA 情况下只打标记不换权重。若要"policy-as-researcher 但用旧权重产文档",**必须自己解决 vLLM 权重双份/换权问题**(或走 LoRA:base=ref),没有现成代码可抄。

## 六、三条警告

1. **不要把 Skill1 当作"多能力联合 loss"的实证依据。** 它的 distill 阶段 token 从不进 batch(`rollout_loop.py` 全文只有 448/498/556 三处 append),distill reward 在发布配置下也被 `credit_assignment=True` 分支丢弃。论文宣称的"三能力统一进化",在代码里只有 selection(query+rerank)和 utilization(play)真的进了梯度,**distillation 是纯数据通路**。引用该工作时必须区分这一点,否则会在 review 里被打。

2. **三库无一提供 token-dense 蒸馏项与轨迹级 RL 项的尺度协调实证。** 我们的 `w_opd·OPD(token dense)` + `w_rl·RL(轨迹二值)` 的量纲冲突问题,在这三库里**完全没有对应经验**——它们唯一的 KL 是 verl 标配的 ref-KL(coef 0.01),且 SKILL0 连这个都关了。这块只能靠 CD-003 的 OPD 崩塌机理与 `loss_agg_mode`(默认 `token-mean`,`ppo_trainer.yaml:56`)自己定。注意 `token-mean` 下超长 research 会话的每 token 权重会被摊薄,而 OPD 的 token-dense 项不会——这是要单独核算的第四类不均衡,三库都没碰过。

3. **别照抄发布脚本的参数名字面含义。** Skill1 至少三处"配置说了但没生效":`train_retrieve_type=ucb` 被 `relevance_weight=1.0` 架空(`memory.py:503`);`full_group_memory=True` 架空组内对照;`intrinsic_reward_coefficient=1.0` 架空退火。这类"配置文件不是运行真相"的坑与已记录的经验一致——**移植时以代码分支为准,并在自己的实现里把生效值打进 manifest**。

---

## 附:最相关的可读源文件(绝对路径)

- `<CB>/skill1/agent_system/multi_turn_rollout/rollout_loop.py`(四阶段主循环、credit assignment、λ 注入)
- `<CB>/skill1/agent_system/environments/env_manager.py:1090-1700`(对照 reward、NDCG、first_order_diff、组划分)
- `<CB>/skill1/agent_system/memory/memory.py:355-520`(EMA 效用、UCB 检索、退休)
- `<CB>/skill0/agent_system/reward_manager/episode_with_compression.py:58-140`(间歇施加辅助 reward)
- `<CB>/skill0/agent_system/environments/curriculum_skill_manager.py:104-176`(delta 驱动的技能撤除课程)
- `<CB>/skillrl/verl/trainer/ppo/ray_trainer.py:941-1008`(阈值门控的技能库更新)
