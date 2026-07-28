# UCOB 源码逐行深读报告

> 调研对象:`.conductor/branches/CD-002/codebases/ucob/`(github.com/TU2021/UCOB 浅克隆,仅 1 个 commit `d38292c`,无 issue、无历史)
> 目的:为 TRACE-OPD 新增第四目标(policy-as-researcher 的 research 能力 RL)提供多能力联合训练的可移植实现细节
> 完成日期:2026-07-28

## 0. 文件定位

| 角色 | 文件 | 行数 |
|---|---|---|
| 入口 + rollout 采集 + CBSD 辅助批构造 + reflection 生成 + U(m) 更新 | `verl/trainer/main_ucob.py` | 4626 |
| trainer(`UCOBRayTrainer(RLSDRayTrainer)`)、`fit()` | `verl/trainer/ppo/ucob_ray_trainer.py` | 1543 |
| **四项 loss 的真实汇聚点** `update_policy` | `verl/workers/actor/dp_actor.py` | 1032 |
| OPD / top-K 蒸馏核 | `verl/trainer/ppo/sdar_utils.py` | 244 |
| `agg_loss` 归一化语义 | `verl/trainer/ppo/core_algos.py` | 728 |
| 记忆库 + U(m) EMA | `agent_system/memory/reflection_memory.py` | 1752 |
| 配置默认 | `verl/trainer/config/ucob_trainer.yaml` | 161 |
| 实验值 | `examples/ucob_trainer/run_alfworld_qwen3_1.7b_full.sh` | 279 |

---

## 1. 机制精确形态(以代码为准)

### 1.1 最重要的结构性事实:**不是四项一次反传,是「三项一次 + 第四项另起一次」**

`update_policy` 里实际相加的只有三项(`dp_actor.py:819-904`):

```
policy_loss = pg_loss*pg_loss_coef − entropy_loss*entropy_coeff   # :836
            + kl_loss*kl_loss_coef                                 # :846
            + cbsd_loss*cbsd_coef                                  # :904
loss = policy_loss / gradient_accumulation ; loss.backward()       # :1017-1018
```

reflection writer 的 PG **不在这里**。它走 `_update_reflection_actor`(`ucob_ray_trainer.py:856-889`),在主 `update_actor` 完成之后**再调一次** `self.actor_rollout_wg.update_actor(chunk)`(`:879`)——而 `update_policy` 内部自带 `_optimizer_step()`(`dp_actor.py:1028`),所以这是**独立的优化器步**,不是同一份梯度。

原因写在注释里(`main_ucob.py:4619-4620`):

> `# Reflection self-training uses a longer response length than acting rollouts.`
> `# Keep it out of the rollout tensor batch to avoid padding every action update.`

reflection 响应固定 padding 到 2048(`main_ucob.py:2670-2684`),执行 rollout 只有 512(`ucob_trainer.yaml:7`)。为避免每个 action 更新都按 2048 padding,他们把 reflection 批塞进 `meta_info` 带出(`main_ucob.py:4618-4621`),trainer 里 pop 出来(`ucob_ray_trainer.py:1203`)单独训。

### 1.2 第二个关键结构:**CBSD 行与 RL 行在 token 级完全互斥**

`dp_actor.py:736-739`:

```python
cbsd_row_mask = (cbsd_loss_mask.sum(dim=-1, keepdim=True) > 0).to(dtype=torch.float32)
response_mask = base_response_mask * (1.0 - cbsd_row_mask)
```

凡带 CBSD mask 的行,其 `response_mask` **被整行清零**,于是该行不进 `pg_loss`(`:823`)、不进 `entropy_loss`(`:833`)、不进 `kl_loss`(`:844`)。反过来普通 RL 行的 `cbsd_loss_mask` 全零,不进 CBSD loss。**没有任何一个 token 同时承载两个目标**——这是他们规避梯度冲突的主手段,不是靠调系数。

CBSD loss 用自己的 mask 归一化(`dp_actor.py:863 / 888 / 901` 都传 `response_mask=cbsd_loss_mask`)。

### 1.3 归一化口径

`agg_loss`(`core_algos.py:395-433`),默认 `token-mean`(`ppo_trainer.yaml:56`),即 `masked_mean = (v*m).sum()/(m.sum()+1e-8)`(`torch_functional.py:148`,全零 mask 返回 0 不 NaN)。

三项各自在**自己的 mask 上做 token-mean**,所以相对梯度量级**与各目标的行数无关**,只由系数决定。这是个真实的平衡性质。

`seq-mean-token-mean` 分支被**改过**(`core_algos.py:415-422`),加了 `active_rows` 守卫,只对有效 token 的行求均值——原版 verl 没有。这正是为了配合 1.2 的整行清零(否则分母会把清零行也算进去稀释 PG)。

### 1.4 CBSD 精确形态

- **分组键**:`(uid, stringify(anchor_obs))`(`main_ucob.py:2120-2123`)——同一任务实例 + 同一环境观测,即 GiGPO 的 anchor state。
- **teacher 选择**:组内 `source_branches`(默认 `[good, none]`,双向)中 **return 最高者**(`:2230`),不是逐对高教低。
- **student 视图**:目标是「对侧 prompt 视图」(`:2231`,`_opposite_cbsd_branch` `:1664-1665`),即把 winner 的**同一段 response** 放到 skill↔no-skill 的另一个 prompt 上下文下重算 logprob。对侧 prompt 在 rollout 时就已存好在 step 里(`ucob_cbsd_alt_prompt_ids`,`:1785-1790`),取不到就整对丢弃。
- **gap**:`raw_gap = winner.return − baseline_return`,baseline 取 target 自身 return;若 target 就是 winner 本身则取组均值(`:2252-2253`)。
- **ε_CBSD 门槛**:`_cbsd_effective_teaching_gap`(`:1667-1673`),`raw_gap <= margin` 直接返回 `None` 丢弃(margin=0.05)。
- **权重**:`weight = min(max(gap/tau, 0.0), weight_max)`(`:2184`),与论文 `w=clip(Δ/τ,0,w_max)` 完全一致。
- **action token 加权**:scope=response 时 action span 内 token 额外乘 `1+(action_weight−1)`(`:2190-2192`),action_weight=2.0。
- **每组配对上限**:按 gap 降序取前 `max_pairs_per_group=2`(`:2270-2272`)。
- **teacher 前向**:`compute_log_prob_with_logits` 用**当前 policy 权重 + 参考 prompt 视图**(`ucob_ray_trainer.py:701`)——这就是「同参数双视图」,teacher 不是独立权重。
- **loss 核**(默认 `topk_opd`,`sdar_utils.py:122-244`):对 teacher top-32 做 `log_softmax` 归一(`:164-166`),`topk_kl = Σ p_ref·(log p_ref − log p_tgt)`(`:176`),即 **top-K 前向 KL**。
- **置信门**:`gate = sigmoid(gate_beta * (log π_ref − log π_tgt)).detach()`(`sdar_utils.py:180-189`;采样版 `:59-65`),β=5.0,**门是 detach 的,不回传梯度**。
- **stop-grad 位置**:`sdar_utils.py:57`(采样版 reference)、`:156-157`(top-K reference logits/logsumexp)、`:171-172`(`reference_probs_k` 与 `log_ratio` 中的 ref 项)。

### 1.5 reflection writer 完整数据流(最重要)

**① 生成**:rollout 过程中,用 policy 自身单轮生成——`actor_rollout_wg.generate_sequences(reflection_batch_input_padded)`(`main_ucob.py:3687`)。prompt 是「成功轨迹 vs 失败轨迹」对比模板(`:3190` / `:3454`)。**是单轮生成,不是多轮 agent 会话。**

**② 存储**:prompt 与 response 的**纯文本**写进 memory item 的 metadata(`_reflection_training_metadata` `:1018-1026`),打上 `reflection_train_backend="vllm_self_reflection"` 标记;state 池写入点 `:3713-3720`,task 池 `:3866`。落盘为 json(`ucob_trainer.yaml:99`)。

**③ U(m) EMA**:`reflection_memory.py:1281-1284`

```python
item["utility_score"] = ((1.0 - update_beta) * current_avg) + (update_beta * float(new_score))
```

β 从 `beta` / `task_utility_beta` / `state_utility_beta` 读(`:1485-1494`),代码默认 0.05,**YAML 与脚本均为 0.2**。`initial_utility_score` 在 `baseline_relative` 模式下为 **0.0**(`:1497-1498`,YAML `:120`)。

**④ new_score = δ_m 的计算**(`main_ucob.py:2477-2552`,`utility_update=baseline_relative`):
- 按 `env.rollout.n=8` 切组(`:2492, 2497`);组内前 4 条无记忆、后 4 条有记忆(`_rollout_skill_mask` `:3917-3920`)。
- **task 池**:`task_credit = mean(success[good]) − mean(success[baseline])`(`:2505-2507`)——组级成功率差,即 δ_m。
- **state 池**:`state_credit = success[单条轨迹] − mean(success[baseline])`(`:2521`)——更细粒度。
- 两池用**不同 β** 提交(`:2534, 2542`)。
- ⚠️ 若组内无 baseline 行,`baseline_success` 静默取 0.0(`:2505`)。

另有备选模式 `_update_step_relative_memory_utility`(`:2418-2475`):按 `(uid, anchor_obs)` 分组,`score = 0.5 + 0.5*(R − mean)/tau` 并 clip 到 [0,1]。full 配置未启用。

**⑤ 重放样本选择**(`_select_reflection_train_records_for_pool` `:2748-2817`):
- 候选来源:**本 step 实际被检索使用过**的记忆(`_collect_current_step_memory_counts` `:2734-2746`,只统计 `dynamic_memory_branch=="good"` 的 step)。
- 过滤链:必须是 `vllm_self_reflection`(`:2772`)→ prompt/response 非空(`:2776`)→ `lifetime_count ≥ min_lifetime_count`(`:2779`)→ `|utility| ≥ min_abs_utility=0.02`(`:2782`)。
- utility 先 clip 到 [−1,1](`:2790`)。
- 排序:`(step_count, |utility|, lifetime_count)` 三级降序(`:2796-2803`),取前 `max_samples=16`。
- **候选 < 2 直接返回空**(`:2805`)。

**⑥ advantage 附着**(`:2808-2817`):

```python
mean = np.mean(rewards);  std = max(np.std(rewards), reflection_advantage_eps)
advantage = (utility - mean) / std
advantage = clip(advantage, ±reflection_advantage_clip)
return [item for item in candidates if abs(advantage) > 0.0]      # 零优势行丢弃
```

**μ_pool/σ_pool 的「pool」= 本 step 选中的候选集,按 task/state 分池各算一次**——既不是全库,也不是整个 batch。σ 有下限 0.05,advantage 硬 clip ±2.0。

advantage 被**广播到 response 的每个 token**(`:2715-2719`),`returns` 同值(`:2727`)。

**⑦ 重放执行**(`ucob_ray_trainer.py:711-763, 856-889`):
- `multi_turn=False`、`calculate_entropy=False`(`:713-714`);清掉所有 CBSD/DSDAR 键(`:718-724`)。
- `reflection_loss_weight` **乘在 advantages/returns 上**,不是当 loss 系数(`:736-740`)。
- `old_log_probs` 在重放时用**当前 policy 现算**(`:743`);ref_log_prob 也算(`:749-755`)→ **reflection 行同样吃 KL 惩罚**。
- `teacher_log_probs` / `sdar_loss_weight` 置零(`:758-760`),关掉 SDAR。
- 按 `reflection_train_mini_batch_size=16` 分块(`:861-868`),**每块一次 `update_actor` = 一次优化器步**(`:872-880`)。第一块 ratio≡1(clip 不触发),后续块 old_log_prob 已陈旧,PPO clip 才真正生效。
- warmup 门:`current_training_step > reflection_train_warmup_steps`(`:1012-1016`),30 步内完全不训 writer。

### 1.6 训练调度(单个 step 内的严格顺序)

1. rollout(`main_ucob.py:4582-4588`),期间检索记忆 + 生成 reflection
2. **U(m) EMA 更新**(`:4294-4299`)—— 在任何 policy 更新之前
3. 构造 CBSD 辅助批(`:4597-4600`)、reflection 训练批(`:4601`)
4. trainer 拆分三路(`ucob_ray_trainer.py:1203-1213`);`compute_advantage` **只作用于执行行**(`:1351`,CBSD/reflection 行此时已被摘走)
5. `update_actor(执行行 + CBSD 行交错拼接)`(`:1475-1487`)
6. `_update_reflection_actor(...)`(`:1490-1496`)—— 独立优化器步

即:**记忆更新 → 执行/CBSD 更新 → writer 更新**,同步串行,不是跨 step 交替。

---

## 2. 默认超参全表

「代码默认 / YAML / alfworld 脚本」三列;后两者一致时合并。

### CBSD

| 键 | 代码默认 | YAML + 脚本 | 出处 |
|---|---|---|---|
| enable | False | **true** | `main_ucob.py:680` / `yaml:78` / `sh:37` |
| **coef (λ_CBSD)** | **0.05** | **0.1** | `main_ucob.py:182`,`dp_actor.py:853` / `yaml:79` / `sh:38` |
| **margin (ε_CBSD)** | 0.05 | 0.05 | `main_ucob.py:682` / `yaml:81` |
| **tau** | 0.2 | 0.2 | `:683` / `yaml:82` |
| **weight_max** | 2.0 | 2.0 | `:684` / `yaml:83` |
| max_pairs_per_group | 2 | 2 | `:690` / `yaml:84` |
| loss_type | `nll` | **`topk_opd`** | `:691`,`dp_actor.py:633` / `yaml:87` |
| opd_granularity | `turn` | `turn` | `:692` / `yaml:88` |
| opd_scope | `action` | **`response`** | `:698` / `yaml:89` |
| **topk_k (K)** | 32 | 32 | `main_ucob.py:185` / `yaml:90` |
| gate_enable | True | true | `:187`,`dp_actor.py:867` / `yaml:91` |
| **gate_beta (β)** | 5.0 | 5.0 | `:190`,`dp_actor.py:870` / `yaml:92` |
| topk_norm_to_one | True | true | `:192` / `yaml:93` |
| topk_clip_log_ratio | False | false | `:196` / `yaml:94` |
| action_weight | 1.0 | **2.0** | `:704` / `yaml:95` |
| action_mask_enable | True | true | `:681` / `yaml:80` |
| source_branches | `good,none` | `[good,none]` | `:706-708` / `yaml:85` |
| target_policy | `opposite` | `opposite` | `:709` / `yaml:86` |
| max_turns_per_trajectory | 0(不限) | 0 | `:705` / `yaml:96` |
| force_source_teacher | False | 未设 | `:685-689` |

### Reflection writer

| 键 | 代码默认 | YAML + 脚本 | 出处 |
|---|---|---|---|
| reflection_train_enable | False | **true** | `main_ucob.py:778` / `yaml:146` |
| **warmup_steps** | 0 | **30** | `:782-787` / `yaml:148` |
| max_task/state_samples_per_step | 16 / 16 | 16 / 16 | `:788-793` / `yaml:149-150` |
| min_lifetime_count task/state | **1 / 10** | **1 / 1** | `:794-799` / `yaml:151-152` |
| min_abs_utility | 0.02 | 0.02 | `:800-802` / `yaml:153` |
| **advantage_eps(σ 下限)** | 0.05 | 0.05 | `:817` / `yaml:156` |
| **advantage_clip** | 2.0 | 2.0 | `:818` / `yaml:157` |
| advantage_group_by | `memory_pool` | `memory_pool` | `:804-816` / `yaml:155` |
| **loss_weight (λ_refl)** | 1.0 | **1.0** | `ucob_ray_trainer.py:736` / `yaml:160` |
| train_mini_batch_size | `len(aux)` | 16 | `ucob_ray_trainer.py:861-865` / `yaml:158` |
| train_response_length | — | 2048 | `main_ucob.py:819-822` / `yaml:159` |
| reflection_reward_mode | — | `batch_zscore` | `yaml:154` |

### 记忆库 / U(m)

| 键 | 代码默认 | YAML + 脚本 | 出处 |
|---|---|---|---|
| **beta (EMA)** | **0.05** | **0.2** | `reflection_memory.py:31,1485,1508` / `yaml:111` |
| task_utility_beta | ←beta | 0.2 | `:1486-1488` / `yaml:112` |
| state_utility_beta | ←beta | 0.2 | `:1489-1494` / `yaml:113` |
| initial_utility_score | 0.0(baseline_relative)/0.5 | 0.0 | `:1495-1500` / `yaml:120` |
| utility_update | `outcome` | **`baseline_relative`** | `:1484` / `yaml:119` |
| step_utility_tau | 1.0 | 1.0 | `main_ucob.py:763` / `yaml:128` |
| retrieve_type / ucb_scale | — | `ucb` / 0.1 | `yaml:114-115` |
| task_top_k / state_top_k | — | 3 / 3 | `yaml:106-107` |
| alpha(相关性权重) | — | 0.1 | `yaml:110` |
| max_context_tokens | — | 4096 | `yaml:129` |

### RL / PPO / 优化器

| 键 | 值 | 出处 |
|---|---|---|
| adv_estimator | `gigpo`,mode `mean_norm`,step_advantage_w 1.0 | `yaml:52,56-59` |
| gamma | 0.95 | `yaml:53` |
| **entropy_coeff** | **0.0(完全无熵正则)** | `yaml:23`,`sh:28` |
| use_kl_loss / coef / type | True / **0.01** / `low_var_kl` | `yaml:24-26` |
| use_kl_in_reward | False | `yaml:54` |
| clip_ratio / low / high / c | 0.2 / 0.2 / 0.2 / 3.0 | `ppo_trainer.yaml:52-55` |
| **loss_agg_mode** | **`token-mean`** | `ppo_trainer.yaml:56` |
| pg_loss_coef | 1.0 | `dp_actor.py:831` |
| ppo_epochs | **1** | `ppo_trainer.yaml:64` |
| ppo_mini_batch / micro_per_gpu | 64 / 8 | `yaml:21-22` |
| train_batch_size / rollout.n | **16 / 8**(→128 轨迹,64 有记忆 + 64 无) | `sh:112,114` |
| use_dynamic_bsz | False | `ppo_trainer.yaml:48` |
| **lr / warmup** | **1e-6 / 无 warmup**(`lr_warmup_steps=-1`,ratio 0.0,`warmup_style=constant`) | `yaml:20`;`ppo_trainer.yaml:73-77` |
| weight_decay / grad_clip | 0.01 / **1.0** | `ppo_trainer.yaml:79,50` |
| actor shuffle | False | `ppo_trainer.yaml:65` |
| norm_adv_by_std_in_grpo | True | `ppo_trainer.yaml:238` |
| invalid_action_penalty | True / 0.1 | `yaml:30-31` |
| **filter_groups.enable** | **false(全对全错组不过滤)** | `yaml:60-61`,`sh:172` |
| dsdar.adaptive_enable / topk_enable | false / false | `sh:24-25` |
| max_prompt / response_length | 8192 / **512** | `yaml:6-7` |
| max_model_len / gpu_mem_util / TP | 10240 / 0.6 / 1 | `yaml:37-38,34` |
| env.max_steps / total_epochs / GPUs | 50 / 150 / 8×1 | `sh:264,277,271` |
| val temp / top_p | 0.4 / 1.0 | `yaml:43-44` |

**1.7B vs 7B**:`diff run_alfworld_qwen3_1.7b_full.sh run_alfworld_qwen25_7b_full.sh` 只有 2 行差异——`MODEL_PATH`(第 13 行)与 `experiment_name`(第 117 行)。**跨 4 倍模型规模没有重调任何一个超参**,包括 lr、λ_CBSD、λ_refl、batch。

---

## 3. 稳定化手段全清单

**结构性(最强,不靠调参)**

1. **行级目标互斥**:CBSD 行整行退出 PG/entropy/KL(`dp_actor.py:736-739`)。
2. **reflection 独立反传 + 独立优化器步**(`ucob_ray_trainer.py:872-880`),彻底隔离超长稀疏目标。
3. **CBSD 行均匀交错进主批**(`_concat_interleaved_batches` `:892-916`,按 `(pos+1)*aux_len//total_len` 均分),保证每个梯度累积 micro-batch 都同时见到两个目标,避免某个 micro-batch 只剩单一目标。
4. **各目标在自己的 mask 上 token-mean**,梯度占比与行数解耦。
5. **`seq-mean-token-mean` 的 `active_rows` 守卫**(`core_algos.py:415-422`,改自原版 verl),配合整行清零防止分母稀释。
6. `masked_mean` 分母 `+1e-8`(`torch_functional.py:148`),全零 mask 返回 0 而非 NaN。

**信号质量门(即「无有效信号就不产生梯度」)**

7. CBSD margin ε=0.05,`gap <= ε` 整对丢弃(`main_ucob.py:1671`)。
8. CBSD 每组配对上限 2(`:2271-2272`)。
9. 置信门 `σ(5·Δ)` detach(`sdar_utils.py:65,188`),teacher 并不更自信的 token 自动降权。
10. 权重 clip `Δ/τ ∈ [0, 2]`(`:2184`)。
11. reflection:`|U| ≥ 0.02`、`lifetime_count ≥ 阈值`、**候选 < 2 直接放弃**、零优势行丢弃(`:2782, 2779, 2805, 2817`)。
12. reflection advantage 的 σ 下限 0.05 + 硬 clip ±2.0(`:2810-2815`)——防止同质候选集把噪声放大成巨大 advantage。
13. U(m) 的 EMA(β=0.2)本身就是对单 step 噪声的低通滤波。
14. reflection warmup 30 步(`:1012-1016`),等 U(m) 攒够统计量再开训。
15. top-K KL 的 `norm_to_one`(在 K 个 logit 上重新 log_softmax,`sdar_utils.py:164-166`)。

**通用**

16. KL 到冻结 ref,coef 0.01,`low_var_kl`;**reflection 行同样吃**(`ucob_ray_trainer.py:749-755`)。
17. 全局 grad_clip 1.0(`ppo_trainer.yaml:50`)。
18. lr 1e-6、ppo_epochs=1、无 warmup、constant schedule。
19. 长度守卫:超 `max_model_len` 的 reflection 行直接跳过(`main_ucob.py:2702-2710`)。

**明确不存在的**:梯度范数均衡 / GradNorm / PCGrad 类冲突消解、动态权重、λ 退火(`_get_sdl_lambda` `ucob_ray_trainer.py:1129-1134` 存在但只服务已关闭的 legacy SDL)、熵正则(coeff=0)、DAPO 全对全错组过滤(`filter_groups.enable=false`,实现在 `agent_system/multi_turn_rollout/utils.py:133-170`)。**系数全是死值,靠结构而非调度保稳。**

---

## 4. 对 TRACE-OPD 双角色联合训练的可移植经验

1. **把 research 目标从执行 batch 里彻底摘出去,单独一次 `update_actor`**——UCOB 正是因为 writer 响应 2048 vs 执行 512 而这么做(`main_ucob.py:4619-4620`),我们的 research 会话长度差距只会更大,混批的 padding 浪费会吃掉大部分算力。
2. **让 OPD 行与 RL 行行级互斥**:凡进入蒸馏的行,整行从 PG/KL/entropy mask 中清零(照抄 `dp_actor.py:736-739`),这样 `w_opd` 与 `w_rl` 不再竞争同一批 token,系数选择的敏感度大幅下降。
3. **若采用 2,必须同步修 `seq-mean-token-mean` 的分母**(加 `active_rows` 守卫,`core_algos.py:415-422`),否则被清零的行会稀释 PG;用 `token-mean` 则天然免疫。
4. **各目标一律在自己的 mask 上归一化**,使梯度占比与该目标的行数解耦——这样 research 行数在各 step 波动(有时 0 行)不会造成有效学习率抖动。
5. **把辅助行均匀交错进主批**(`_concat_interleaved_batches` `:892-916`),别拼在尾部,否则最后几个 micro-batch 变成纯单目标。
6. **research reward 用「组内 z-score + σ 下限 + advantage 硬 clip」三件套**(`:2808-2816`):σ 下限(0.05)专治「8 份文档成功率几乎相同」时把噪声放大成巨大 advantage,clip(±2.0)封顶。
7. **组内候选 < 2 就整组放弃,零优势行显式丢弃**(`:2805, 2817`)——这是他们对「部分组无有效信号」的全部处理,比 DAPO 重采样便宜得多。
8. **给 research 目标设 warmup 步数**(UCOB 用 30,`:1012-1016`),让 executor 先训到有区分度的成功率,再开 research RL;这直接对应「executor 已训多步、researcher 冷启动」的节奏差。
9. **research reward 用「用文档 vs 不用文档」的组内成功率差,而不是绝对成功率**(`:2505-2507`)——绝对成功率会被任务难度污染,差分自动扣掉难度基线。我们的 K=8 doc-conditioned rollout 应配一组 no-doc baseline rollout。
10. **对 research 信号先做 EMA 再进 PG**(β=0.2,`reflection_memory.py:1281-1284`),让同一份文档跨多个 step 累积证据后再产生梯度,而不是单 step 的 8 次 rollout 直接当 reward。
11. **权重系数乘在 advantage 上而不是 loss 上**(`ucob_ray_trainer.py:736-740`),可复用现成 PG 管线;但要清楚这**不缩放同批的 KL 项**。
12. **蒸馏项用 top-K(K=32)前向 KL + 在 K 个 logit 上重归一**(`sdar_utils.py:164-176`),显存/带宽远低于全词表,且 `norm_to_one` 让它成为良定义的分布距离。
13. **置信门 `σ(β·(logπ_teacher − logπ_student))` 且 detach**(`:65,188`,β=5):teacher 并不更自信的 token 自动接近 0 权重——这是比全局系数更细粒度的「防止 teacher 教坏」手段,可直接加到我们的特权前缀 OPD 上。
14. **teacher 权重与 student 同参数、只换 prompt 视图**(`ucob_ray_trainer.py:701`),与「文档作特权前缀」结构同构,可直接复用其 gap/权重口径。
15. **设 teaching gap 门槛 ε 并统计被过滤比例**(`:1671`,指标 `analysis/mech/margin_filtered_ratio` `ucob_ray_trainer.py:189-333`)——这个比例是判断「机制是否在空转」的一线指标,建议我们也埋。
16. **每组配对数设上限**(2,`:2271-2272`),防止某几个高方差组霸占蒸馏预算。
17. **research 重放按小块(16 行)分次 `update_actor`**(`:861-880`),既控显存,又让块间产生真实的 PPO ratio ≠ 1,clip 才有意义。
18. **在 rollout 阶段就把两种 prompt 视图都存进 step**(`ucob_cbsd_alt_prompt_ids`,`:1785-1790`),取不到就跳过该对——避免训练期重建 prompt 的对齐错误。

---

## 5. 警告 / 不适用点

1. **⚠️ reflection 重放走的是文本再分词,不是原始 token id**(`:2670-2684` 用 `tokenizer.encode(text)`,而生成端 `batch_decode(..., skip_special_tokens=True)` `:3689-3692`)。这意味着 (a) 重放序列可能与实际采样序列的 token 切分不同,PG 的 on-policy 假设被破坏;(b) **EOS 被剥掉,永远训不到停止符**。若照搬,务必存原始 token id。
2. **⚠️ `baseline_success` 在组内无 baseline 行时静默取 0.0**(`:2505`),会把 credit 虚高成绝对成功率。我们的 N=8 research 会话若不保证配套 no-doc 组,会踩同样的坑。
3. **⚠️ reflection 是单轮生成**(`:3687`),不是多轮 agent 会话。我们的 research 会话是超长多轮 + 工具调用,UCOB 在「多轮 writer 的 credit assignment、中间轮 mask、工具调用 token 是否计入 loss」上**没有任何可参考实现**——这是纯空白区。
4. **⚠️ 论文/README 承诺的四项联合,代码里 λ_refl 与另三项从不共享一次 backward**,因此 UCOB **完全没有验证过**「轨迹级超长稀疏目标与 token 级 dense 目标同批反传」的稳定性。若坚持四合一,不能引用 UCOB 作为可行性依据。
5. **⚠️ 熵正则完全关闭(coeff=0)、`filter_groups` 关闭**——UCOB 对「角色干扰导致熵坍缩」和「全对全错组」都没有主动防护,只靠 KL(0.01)兜底。researcher 冷启动阶段熵行为会与 executor 差异很大,不能假设 UCOB 的配置能覆盖。
6. **⚠️ 1.7B → 7B 零重调**(见 §2 末)。这说明这套超参**没有经过规模敏感性验证**,把 0.1 / 0.2 / 5.0 这些值直接搬到 Qwen3-4B 前应自己扫。
7. **⚠️ `min_lifetime_count_state` 代码默认 10 但 YAML/脚本设成 1**(`:797-799` vs `yaml:152`)——这类「代码默认与实验值大幅背离」的键要逐个核对,别读代码默认当结论。
8. **⚠️ reflection 行虽然构造了 `loss_mask`(`ucob_ray_trainer.py:726-735`),但 `use_loss_mask` 因 `multi_turn=False` 且 `action_loss_mask_enable=false` 而为 False**(`dp_actor.py:617`),实际走的是 `attention_mask` 分支。此处两者数值恰好等价(`main_ucob.py:2712`),但这是个易碎的隐式依赖,移植时容易变成真 bug。
9. **⚠️ CBSD 的 `loss_type` 代码默认是 `nll`(纯 `-log_prob` 加权,`dp_actor.py:897-903`),而实验用的是 `topk_opd`**。若只读 `.get()` 默认会得到完全不同的机制。
10. **不适用**:UCOB 的 memory 是**检索式知识条目**(sentence-transformer + UCB 检索,`yaml:114-123`),writer 训的是「写出一条会被检索命中的经验」;我们的 researcher 训的是「跑一场实验会话」。U(m) 的 EMA 之所以成立,是因为**同一条记忆会跨多个 step 被反复检索**,才攒得出统计量。若每任务 8 份文档是一次性的,EMA 无处附着——需要先想清楚「跨 step 复用的载体是什么」。

---

## 6. 与「论文机制(已知)」描述不同的发现

| # | 论文/任务书描述 | 代码实际 | 出处 |
|---|---|---|---|
| 1 | `L = L_RL + λ_CBSD·L_CBSD + λ_refl·L_refl + β_KL·L_KL` 单一 loss | **只有前两项 + KL 同批反传;L_refl 是独立的第二次 `update_actor` + 独立优化器步** | `dp_actor.py:836-904` vs `ucob_ray_trainer.py:872-880` |
| 2 | 「anchor-state 组内高回报教低回报」 | 不是逐对高教低,而是**组内唯一 winner 教其余所有 target**;且「学生」是 winner 的**同一段 response 换到对侧 prompt 视图**下,不是低回报者自己的 response | `:2230-2233, 2143-2159` |
| 3 | CBSD 是「双向自蒸馏」 | 双向体现在 `source_branches=[good,none]`(谁赢谁当 teacher),但**单个组内方向唯一**,不是同时双向 | `:2227-2233`,`yaml:85` |
| 4 | 置信门 `g=σ(β·[logπ(y*\|ref)−logπ(y*\|tgt)])` | 公式一致,但**主 loss 不是被门控的 NLL,而是 teacher top-32 上的前向 KL**;门只是乘在 top-K KL 上,且 detach | `sdar_utils.py:176, 188-189` |
| 5 | `A_refl=(U(m)−μ_pool)/σ_pool` | μ/σ 只对**本 step 选中的候选集、按 task/state 分池**各算一次;另有 σ 下限 0.05 与 advantage 硬 clip ±2.0,且零优势行被丢弃 | `:2808-2817` |
| 6 | `U(m)←(1−β)U(m)+β·δ_m` | 一致;但 **β 代码默认 0.05、实验用 0.2**,且 task/state 两池可用不同 β;`δ_m` 对 state 池是**逐轨迹**而非组级 | `reflection_memory.py:1281-1284`;`main_ucob.py:2507, 2521, 2534, 2542` |
| 7 | 「clipped PG 重放」 | 复用标准 PPO clip 0.2,但 `old_log_probs` 在重放时现算(`:743`)→ **第一个 chunk ratio 恒等于 1,clip 不触发**,退化为带 advantage 的 REINFORCE;仅后续 chunk 才有真实 ratio | `ucob_ray_trainer.py:743, 872-880` |
| 8 | λ_refl 是 loss 系数 | **乘在 advantages/returns 上**,不是 loss 系数;因此它**不缩放同批的 KL 项** | `ucob_ray_trainer.py:736-740` |
| 9 | reflection writer 训练自身产出 | 只训 `reflection_train_backend == "vllm_self_reflection"` 的条目,即**只训 policy 自己(vLLM)生成的**;external_api 生成的反思写进库但**永不参与训练** | `:2772`,`:1020` |
| 10 | (未提) | reflection 候选来源限定为**本 step 真被检索命中**的记忆,且按 `step_count` 降序优先——命中越多越优先训 | `:2734-2746, 2796-2803` |
| 11 | 骨架 GiGPO | 确认 `adv_estimator=gigpo`、`mode=mean_norm`、`step_advantage_w=1.0`、gamma 0.95 | `yaml:52-59` |
| 12 | (未提) | **CBSD 与 reflection 行都在 `compute_advantage` 之前被摘走**,GRPO advantage 只算在执行行上 | `ucob_ray_trainer.py:1203-1213` vs `:1351` |

**不稳定现象与 workaround(必答问题 8)**:README、示例 README、代码注释、git 历史中**没有任何不稳定现象记录或 workaround**。全仓仅两条相关注释:`main_ucob.py:4619-4620`(reflection 出批的 padding 理由)与 `:3919`(`env.rollout.n` 为奇数时的分组警告)。浅克隆无 issue 可查。
