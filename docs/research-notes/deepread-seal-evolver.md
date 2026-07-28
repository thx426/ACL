# SEAL / EvolveR 官方代码库逐行深读

> 调研日期:2026-07-28
> 调研对象基路径 `CB` = `/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/.conductor/branches/CD-002/codebases`
> 方法:实际打开文件逐行读关键实现,所有结论带 `file:line` 出处。未运行任何代码;凡属推断而非实测的,文内显式标注。

## 结论先行

1. **SEAL 仓库里没有任何 GRPO/PPO 代码残骸**,论文的"reward 依赖被更新参数时 GRPO/PPO 不稳"声明在代码层**无法核实**。
2. 但代码里留下了一组量化指标,显示 SEAL 真正的病可能是**评分器信噪比小于 1**(组间信号 std 0.1244 < 组内噪声 std 0.1592),而**不一定是漂移**。这对我们的论点是重要的修正。
3. **EvolveR 发布配置把经验相关的 reward 权重全设成 0**(`info_gain=0, experience=0`),其提炼与检索环节**完全没有梯度**。我们要做的"research 能力 RL"它没做过 —— 既是新颖性,也意味着无先例可抄。

---

# 一、SEAL(`CB/seal`)

## 1.1 机制精确形态

SEAL 是两套独立实现,别混谈。**与 TRACE-OPD 同构的是 general-knowledge 那套**(自然语言文档 + 连续 reward),few-shot 那套的 self-edit 其实是个超参 JSON。

### few-shot(ARC)

| 环节 | 实现 |
|---|---|
| self-edit 采样 | `SamplingParams(max_tokens=128, temperature=0.8)`,`CB/seal/few-shot/self-edit.py:432-436` |
| self-edit 内容 | **不是文档,是训练配置 JSON**:4 个增强开关 + strategy/lr/epochs,`CB/seal/few-shot/utils/prompts.py:3-41` |
| 规模 | 12 tasks × 15 self-edits(训);10 × 5(评),`CB/seal/few-shot/README.md:19,70` |
| 去重 | config 转 hashable tuple 存 set,重复则重采;JSON 解析失败也重采,`self-edit.py:461-478` |
| 内层更新 | LoRA r=128 α=16,target q/v/gate/down/up,bs=2,cosine,warmup=11,`self-edit.py:395-407` + `arclib/update_model.py:227-241` |
| reward | temperature=0/n_sample=1 推理 → 多增强视图投票出 2 个 attempt → gold 是否命中,**二值**,`arclib/eval.py:54-57` |
| ReST-EM | **严格 correct 过滤,无阈值无 top-k**,`BC-self-edit.py:67`;SFT on prompt+response,只对 response 段算 loss(`:100-110`);LoRA r=16 α=16, epochs=8, bs=5, lr=5e-5(`README.md:51-56`) |
| 外循环轮数 | **1 轮** |

### general-knowledge(SQuAD)

| 环节 | 实现 |
|---|---|
| self-edit 采样 | k=5/篇,**temperature=1.0, top_p=0.95**, max_tokens=8192,`scripts/make_squad_data.sh:22-25` |
| self-edit 内容 | 自然语言 "implications" 列表,`src/data_generation/make_squad_data.py:17-22` |
| 内层更新 | LoRA r=32 α=64, epochs=10, **lr=1e-3**, bs=1,`scripts/query_server.sh:22` |
| 训练序列构造 | completion 按 `---`/换行切条,每条前缀 title,**末尾总是追加原文 context**,上限 30 条,`src/utils.py:193-237` |
| 评测 | vLLM 动态 load/unload adapter,temperature=0,max_tokens=64;**GPT-4.1 as judge** yes/no,`src/utils.py:153-186` |
| reward | `adapter_gain = adapter_acc − baseline_acc`,`src/inner/TTT_server.py:344-353` |
| 重复评测 | **eval_times=3**(同一 completion 跑 3 次完整 finetune+eval 取均值),`src/query/query_server.py:146-196` |
| ReST-EM | **按 `adapter_mean` 排序取 top-k,k_best 默认 1**,`src/EM/build_SFT_dataset.py:20-30,38-43` |
| SFT | trl SFTTrainer,LoRA r=64 α=128 全 7 proj,epochs=2,lr=3e-4,有效 batch=1×5×2,DeepSpeed stage3,`scripts/train_SFT.sh:18-26` |
| 外循环轮数 | **2 轮** |

**一个设计上的洞**:`gain_mean` 算了、存了、打印了,但 `build_SFT_dataset.py:41-43` 的 `--metric` 可选项只有 `adapter_mean` 和 `proxy_mean` —— **基线扣除进了日志,没进决策路径**。

## 1.2 GRPO/PPO 不稳的代码证据:零

全仓 grep `grpo|ppo|policy_loss|advantage|reinforce|logprob`,唯一命中是 `requirements.txt:9` 的 `trl==0.18.1` 和 `src/EM/train_SFT.py:14` 的 `from trl import SFTConfig, SFTTrainer`。没有实现残骸、没有被注释掉的分支、没有相关注释。git log 只有一个 squash commit `6d9c9f9`。

**所以"SEAL 说 GRPO 不稳"只能作为论文断言引用,不能替它断言机理。** 但代码里有间接的量化证据(见 1.4),指向一个和漂移不同的病因。

## 1.3 内层更新开销 vs TRACE-OPD

general-knowledge 一轮 ReST-EM 数据 = 50 篇 × 5 completions × 3 次 = **750 次完整 LoRA finetune+eval**,每次约 310 个 optimizer step,外加每题一次 GPT-4.1 调用。硬件 2×A100/H100(`CB/seal/README.md:65`)。

我们的 doc-conditioned 前向即评:N=8 × K=8 = 64 次 rollout/任务,**零梯度、零 adapter 落盘、零外部 judge**。成本结构差一个数量级以上,并且完全绕开了 SEAL 的 adapter 生命周期管理坑。

## 1.4 reward 方差处理 —— 最有价值的一节

- 重复次数:general-knowledge 3 次,few-shot **1 次**。
- 种子:**每个请求独立随机** `seed = (int(time.time()*1e6)+step) & 0xFFFFFFFF`,`TTT_server.py:157-159`。**不是共同随机数(CRN)**。
- 归一化:无;基线扣除算了但没用(见 1.1 末)。

`query_server.py:13-19` 的 docstring 亲手定义了两个诊断量:

- `mean_adapter_std_over_completions` — *"higher ⇒ more variance the RL selector can exploit"*(组间散度 = **信号**)
- `mean_adapter_std_within_completions` — *"lower ⇒ more stable signal"*(组内重复散度 = **噪声**)

**实测数字**(`CB/seal/general-knowledge/results/query_server/train/`):

| run | 信号(组间) | 噪声(组内) |
|---|---|---|
| rank_iter0 | 0.1244 | **0.1592** |
| rank_iter1 | 0.1320 | **0.1905** |

**噪声大于信号。** eval_times=3 后组均值标准误 ≈ 0.1592/√3 = 0.092,与 0.1244 仍同量级 → SNR ≈ 1.35。这意味着 top-1 硬选择有相当概率选错。这是 SEAL 评分器质量的硬数字,也是它必须退回 ReST-EM 的一个充分理由 —— **和"reward 依赖被更新参数"是两回事**。

## 1.5 训练规模与效果

`CB/seal/general-knowledge/results/query_server/eval/*.json`,200 篇 val,k=1,eval_times=1:

| 配置 | adapter accuracy |
|---|---|
| 无微调 baseline | 0.3267 |
| naive(只喂原文) | 0.3351 |
| **SEAL base(自编辑)** | 0.3968 |
| **SEAL iter1** | 0.4268 |
| **SEAL iter2** | **0.4699** |
| GPT-4.1 合成数据 | 0.4631 |

2 轮 ReST-EM **+7.3pp,并超过 GPT-4.1 合成数据**。且 iter1→iter2 增益(+4.3pp)大于 base→iter1(+3.0pp),**尚未饱和**。

> ⚠️ `CB/seal/few-shot/final_results.json` 只有 slot `'0'`、8 任务、0 正确,是残缺中途产物,**不是论文数字,别引用**。

**灾难性遗忘实测**(`results/continual_self_edits/overall_results.json`,64 sequences × 8 datapoints,base=iter2,配套代码 `src/continual/continual_self_edits.py`):

- 行 0(base 模型)在 8 个 datapoint 上 ≈ 0.25–0.34
- d0 刚学完 0.3825 → 8 步顺序合并后 **0.1894**,**低于 base 的 0.3309**
- 对角线(刚学的那个)也在衰减:0.3825 → 0.4988 → 0.4462 → 0.4397 → 0.3764 → 0.4489 → 0.3846 → **0.3397**

→ **顺序合并 self-edit 不仅遗忘旧知识,新知识吸收能力本身也在衰减。**

## 1.6 稳定化清单(SEAL)

1. ReST-EM 而非 policy gradient(唯一算法级手段)
2. 内层 LoRA 每次 `reset_lora()`:B 置零、A 恢复初值,`arclib/update_model.py:133-140`
3. adapter 用完即删 `shutil.rmtree` + `unload_adapter`,`TTT_server.py:339-341`
4. 步数硬闸:`epochs*len(train_data)//2 > 375` → epochs=0(产出未训练 adapter),`self-edit.py:553-555`
5. 非法 config → epochs=0 兜底而非崩溃,`self-edit.py:515-523,542-550`
6. 评测端 temperature=0(`few-shot/README.md:31`;`TTT_server.sh:21`)
7. eval_times=3 平均(general-knowledge)
8. 每轮 ReST-EM 后 `merge_and_unload` 存全权重,下轮从干净 checkpoint 起,`train_SFT.py:84-87` / `BC-self-edit.py:154`
9. self-edit 去重(few-shot,config 级 hashable 去重)

> ⚠️ **一处隐患需点名(未运行验证)**:`TTT_server.py:296` 每个请求都对**循环外加载一次的同一个 `base_model` 对象**(`:127-132`)调 `get_peft_model`。PEFT 会就地把子模块替换成 LoRA 层;第二次调用时 `base_model` 已带着上次**训练过的** LoRA 权重。few-shot 侧有显式 `reset_lora()`,general-knowledge 侧**没有对应物**。若该推断成立,评分器会随 step 单调漂移 —— 恰是"reward 依赖被更新参数"最恶劣的具体形态,而且它出现在 SEAL 自己的评分器里。
>
> **对我们的行动项**:若做 server 化评分器,把"跨请求权重不变"写进冒烟测试。

---

# 二、EvolveR(`CB/evolver`)

## 2.0 先说最要紧的一条

`CB/evolver/scripts/train_grpo-3b.sh:143-146`:

```
rewards.weights.format=0.1  rewards.weights.outcome=1.0
rewards.weights.info_gain=0  rewards.weights.experience=0
```

默认值 `{format:0.05, outcome:1.0, info_gain:0.2, experience:0.2}`(`verl/trainer/config/ppo_trainer.yaml:193`)被脚本覆盖。而 `evolver/rewards/__init__.py:62-90` 对 weight ≤ 0 的组件**根本不计算**(用 `RewardOutputMock` 占位)。

→ **`experience_reward_fn` 和 `info_gain_reward_fn` 在发布配置下是死代码。跑通的只有 outcome EM + format。**

## 2.1 经验生命周期哪些环节被训练

| 环节 | 训练? | 信号 / loss |
|---|---|---|
| 提炼 distill | **否** | 纯 prompting;`reflection()` 在 `update_actor` 之后调用(`verl/trainer/ppo/ray_trainer.py:1061` vs `:1089-1093`),完全在 loss 之外 |
| 检索 retrieve | **否** | 冻结 bge-m3 + Milvus ANN,无 loss(`experience_manager.py:84`) |
| 利用 utilize | **是** | GRPO + outcome EM。学的是"何时发 `<search_experience>`、拿到 principle 怎么用" |
| 库评分 | 否(非梯度) | Laplace 平滑 `(success+1)/(usage+2)`,`experience_manager.py:707` |

提炼走 `_batch_llm_call_with_actor`(`experience_manager.py:816`),用的是训练中的 actor,但只调 `generate_sequences`,`recompute_log_prob: False`,输出 decode 成字符串即丢弃(`:866-881`)。**零梯度路径,已逐行确认。**

→ **EvolveR 的"自演化"是:梯度只走利用侧,提炼侧靠 actor 变强而间接变好 —— 一个隐式、无梯度的正反馈环。**

## 2.2 提炼者与 solver 是否同权重 / 联合还是交替

- **同一份权重**(`_batch_llm_call_with_actor` 直接用 `self.actor_rollout_wg`,`experience_manager.py:816-819`)
- 可切换为外部 API 模型(`summary_llm_api.url` 非空则走 `:740` 的 `_batch_llm_call_with_api`),默认 `null` → 用 actor
- **交替,不是联合**:一个完整 RL step 走完(rollout → reward → advantage → update_actor)才 reflection。**无联合 loss、无系数**
- 调度:`organize_interval=1`(每步反思,`train_grpo-3b.sh:149`;default 是 50)、`export_interval=50`、`clean_interval=10`

## 2.3 经验库怎么演化 / 防退化

**增**:每步按 `trajectory_choice_ratio=0.25` 从 `n_agent=8` 中抽 2 条,`selection_strategy='balanced'` 尽量一半成功一半失败(`ray_trainer.py:761-781`)。成功 → Guiding Principle,失败 → Cautionary Principle,均要求"一句话描述 + (subject,predicate,object) 三元组 JSON"(`evolver/experience/prompts.py:10-74`)。轨迹日志超 4096 字符截断(`experience_manager.py:1040-1041`)。

**去重(两级)**:

1. batch 内:仅对**同 question 同 type**的候选两两配对,用 `MATCH_PRINCIPLE_PROMPT` 让 LLM 判 Yes/No(`max_new_tokens=8`),建图取连通分量代表元,`experience_manager.py:572-635`
2. 对库内:`_batch_find_matching_principles` → 命中则 merge,否则新建,`:1075-1092`;embedding 侧阈值 `SIMILARITY_THRESHOLD=0.85`(`experience/config.py:3`)

**评分**:`metric_score = (success_count+1)/(usage_count+2)`,新建时初始 0.5(`experience_manager.py:707,1114`)

**删 / 防退化**:

- `clean_low_metric_principles(threshold=0.3)`,每 10 步一次(`ray_trainer.py:1104-1119`)
- 检索时即时清理:含特殊 token 的、描述重复的、**以 ≥5 个 `!` 开头的"塌缩" principle**,直接 delete(`experience_manager.py:448-484`)
- `_create_new_principle` 拒收含 `<think>`/`<answer>` 等特殊 token 的描述(`:1098-1106`)
- 每 principle 只留最近 3 条成功 / 2 条失败轨迹;doc 截 200 字符、thought 截 300 字符(`experience/config.py:6-7,22-23`)

> 「塌缩 principle 检测」是他们真的踩过坑的化石证据:模型会把 principle 生成成 `!!!!!!!`。

## 2.4 稳定化手段全清单与超参

- **GRPO 组内归一化** `(score − group_mean)/(group_std + 1e-6)`,按 prompt index 分组,`n_agent=8`,`verl/trainer/ppo/core_algos.py:138-153`
- `use_kl_loss=true`, `kl_loss_coef=0.001`, `kl_loss_type=low_var_kl`(`train_grpo-3b.sh:105,119-120`)
- **state masking:`mask_sections=['information','experience']`** —— 检索回来的内容整段替换成 pad,不进 loss:`evolver/llm_agent/generation.py:396-411` 造 `info_mask` → `ray_trainer.py:1144` 变 `loss_mask`
- `clip_ratio=0.2`, `entropy_coeff=0.001`(脚本未覆盖,取 `ppo_trainer.yaml:39-40` 默认)
- `lr=1e-6`, `lr_warmup_steps_ratio=0.02`, `grad_clip=1.0`, `ppo_epochs=1`(严格 on-policy,无样本重用)
- `train_batch_size=128`, `ppo_mini_batch_size=128`, `ppo_micro_batch_size=32`
- rollout `temperature=1.0`(训练);`val_temperature=0.6` 且 `val_do_sample=false`
- `max_turns=10`, `retriever.topk=3`, `TOP_K_PRINCIPLES=3`
- 冷启动 SFT:LLaMA-Factory LoRA r=8, lr=1e-4, 3 epochs, `max_samples=1000`, cutoff 2048(`LLaMA-Factory/examples/train_lora/qwen2_5vl_lora_sft.yaml`,**文件名有误导性**,内容已被改成 Qwen2.5-3B-Instruct)—— **README 完全没提这一步**
- 规模:Qwen2.5-3B-Instruct,8 GPU,`total_training_steps=1000`,5 epochs,NQ+HotpotQA

---

# 三、给 TRACE-OPD 的结论

## ① 如果 research GRPO 真不稳,ReST-EM 备选臂的精确配方

对标 general-knowledge 那套(自然语言文档 + 连续 reward),不是 few-shot 的二值版。

**采样**:保持 N=8 research 会话/任务,temperature 1.0 / top_p 0.95(同 SEAL `make_squad_data.sh:23-24`)。**别降温度** —— ReST-EM 完全靠组内多样性吃饭。

**打分**:p_i = K=8 次 doc-conditioned rollout 成功率。这已经强于 SEAL(它 eval_times=3 且独立随机种子;我们是同一 executor 快照)。

**筛选(核心分歧点,给两档)**:

- **A 档(照抄 SEAL)**:每任务按 p_i 排序取 top-1,无阈值。优点:简单,有 +7.3pp 实证。缺点:一组全烂也会选出一个,把噪声写进 SFT。
- **B 档(推荐)**:top-1 **且**过两道闸 ——
  - (a) `p_max ≥ 无文档基线 + 1/K`
  - (b) `p_max − p_median ≥ 1/K`(组内确有区分度)
  - 两闸不过则**整组丢弃**。
  - 理由:SEAL 把 `gain` 算出来却没进选择路径(`build_SFT_dataset.py:41-43` 的 metric choices 里根本没有 gain),这是它设计上的一个洞;我们有无文档基线,补上成本为零。
- **别用 `k_best > 1`**:SEAL 默认 1,top-2 与 top-1 的 p 差通常落在噪声内。

**SFT**:loss 只算 research 会话的 response 段,prompt 段 mask(照 `BC-self-edit.py:100-110`)。**每轮 ReST-EM 结束就 merge 进权重、丢掉优化器状态,不要让 research 的 LoRA 跨轮累积**(`train_SFT.py:84-87`)。

**超参起点**(SEAL general-knowledge 实测值,`train_SFT.sh:18-26`):epochs=2, lr=3e-4(LoRA r=64 α=128)或 2e-5(全参),有效 batch ≈ 10。
注意 few-shot 侧是 epochs=8 / lr=5e-5 / bs=5,两套差异很大 → **这组超参 SEAL 自己没调透,我们需要自己扫**。

**轮数**:SEAL 停在 iter2 时还没饱和(iter1→iter2 增益大于 base→iter1),**给 3–4 轮预算**。

**与四项联合训练的接口**:ReST-EM 臂只需要 `argmax_i p_i`,不需要梯度 —— 所以**"最优文档作 self-teacher 特权前缀"那条完全不受影响,OPD / 执行侧 RL / SFT 三项一行不改**。这是个很干净的降级路径,**建议直接在代码里做成一个 flag,而不是两套分支**。

## ② 对"组间同 step 对比免疫评分器漂移"论点的证据

### 支持(强)

1. **EvolveR 的 GRPO 就是这个结构且跑通了**:组内归一化、`n_agent=8`、组内共享同一 rollout 快照(`core_algos.py:138-153`),3B / 8 卡 / 1000 步,ICML 2026 接收。这是该结构在 agentic self-evolve 场景下的存在性证明。
2. **SEAL 自己的 docstring 就是按"组间散度 = 可利用信号 / 组内散度 = 噪声"设计指标的**(`query_server.py:13-19`)。它的作者同样认为组间对比是信号所在;它只是**没把这个结构用进 policy gradient**,而是选了 top-1 硬选择。
3. **SEAL 的 reward 确实以最糟的形式依赖被更新参数**:内层 LoRA 从当前 policy 权重出发训练,policy 一变,整个 reward 尺度就平移。我们的组内共享快照切断了这条通道。

### 反驳 / 必须注意

1. **组间对比免疫的是"漂移",不是"低 SNR" —— 而后者才是 SEAL 数字暴露出来的病。**
   - p_i ~ Binomial(8,·)/8,p=0.5 时单点 std ≈ √(0.25/8) ≈ **0.177**,**这就是我们的噪声底**。
   - 若组间真实散度不显著超过它,GRPO 除以 `group_std` 反而会把纯噪声**放大**成看似很大的 advantage。**这比漂移更现实。**
   - **强烈建议开训前做一次零成本诊断**:跑一批任务,对每任务算 8 个 p_i 的 std,与 √(p̄(1−p̄)/8) 相比。**信号/噪声 < 1.5 就先加大 K,而不是调 RL 超参。** 这就是把 SEAL 那对指标翻译到我们的设定。
2. **SEAL 的不稳声明在代码里无法核实**,可能是漂移,也可能就是上述 SNR 问题(它 eval_times=3 且独立随机种子,SNR 比我们差)。**论文里请按"论文断言"引用,别替 SEAL 断言机理。**
3. **组内共享快照 ≠ 跨 step 可比。** executor 快照每步在变,"第 t 步的 p=0.6"和"第 t+100 步的 p=0.6"不是一回事。对 PG 本身无害(advantage 是组内的),但会**污染监控曲线**和任何跨 step 的文档质量比较 —— 要看 research 是否真变好,得**固定一个 held-out executor 快照单独评**。

## ③ 其他可移植经验与警告

### 该抄的

1. **state masking(最高优先级)**:doc-conditioned rollout 里特权文档前缀是外部注入的,**绝不能进 policy loss**,否则 policy 会去拟合自己没生成的 token。verl 里现成:`actor.state_masking=true` + `algorithm.state_masking.mask_sections`(`generation.py:396-411` → `ray_trainer.py:1144`)。
2. **评测态严格 reset**:SEAL few-shot 的 `reset_lora()`(B=0 / A=初值,`update_model.py:133-140`)。我们的 executor 快照每组必须是同一份且零残留。
3. **塌缩检测**:EvolveR 显式删除以 ≥5 个 `!` 开头的 principle + 拒收含特殊 token 的描述(`experience_manager.py:477-484,1098-1106`)。我们的 research 文档必然出现同类退化(重复、空洞、复读特殊 token)。**这与已有的 thought 逐字重复率指标是同一类判据,可直接复用那套阈值逻辑。**
4. **平衡采样**:存轨迹时一半成功一半失败(`ray_trainer.py:764-778`),别只喂成功的。
5. **诊断指标对进常规监控面板**:`std over 8 docs` vs `binomial noise at K=8`。

### 该警惕的

1. **EvolveR 的"经验 reward"是死代码**(`experience=0, info_gain=0`)。若打算引用 EvolveR 作为"经验质量可以直接给 reward"的先例 —— **不能引**。它跑通的只有 outcome EM + format。
2. **EvolveR 提炼侧没有梯度**:"policy-as-researcher 的 research 能力被 RL 训练"这件事它**没做**。对我们既是新颖性,也意味着没有先例可抄,踩坑全靠自己。
3. **SEAL 顺序 self-edit 的灾难性遗忘实测**(d0: 0.3825 → 0.1894,低于 base 0.3309;对角线 0.4988 → 0.3397)。我们的 OPD 若是"最优文档作 teacher 反复蒸",本质上是同类的顺序权重更新 —— **这条曲线可直接作为"冻结 teacher + 锚"结论的外部佐证**。
4. **`TTT_server.py:296` 的 PEFT 重复包裹**(见 1.6 隐患):若做 server 化评分器,把"跨请求权重不变"写进冒烟测试。
5. **EvolveR 仓库未经复现验证**:`evolver/experience/milvusdb/start_server.sh` 有 `\$` 转义 bug 和 `cd ./evolver/evolver/experience/milvusdb` 路径重复;`evolver/search/retrieval.sh` 有作者硬编码的 home 路径。**别假设开箱即用。**
6. **`CB/seal/few-shot/final_results.json` 是残缺产物**(8 任务 0 正确),不是论文数字,别引用。

---

## 附:关键文件索引

### SEAL

| 用途 | 路径 |
|---|---|
| few-shot self-edit 生成 + 内层训练驱动 | `CB/seal/few-shot/self-edit.py` |
| few-shot ReST-EM(二值过滤 SFT) | `CB/seal/few-shot/BC-self-edit.py` |
| few-shot 内层 LoRA(含 `reset_lora`) | `CB/seal/few-shot/arclib/update_model.py` |
| few-shot reward 度量 | `CB/seal/few-shot/arclib/eval.py` |
| gk 内层 TTT 服务(reward 计算) | `CB/seal/general-knowledge/src/inner/TTT_server.py` |
| gk 外循环驱动 + 方差诊断指标 | `CB/seal/general-knowledge/src/query/query_server.py` |
| gk ReST-EM 数据构建(top-k 选择) | `CB/seal/general-knowledge/src/EM/build_SFT_dataset.py` |
| gk SFT 训练 | `CB/seal/general-knowledge/src/EM/train_SFT.py` |
| gk 训练序列构造 + GPT-4.1 judge + proxy reward | `CB/seal/general-knowledge/src/utils.py` |
| 灾难性遗忘实验 | `CB/seal/general-knowledge/src/continual/continual_self_edits.py` |

### EvolveR

| 用途 | 路径 |
|---|---|
| 主训练入口脚本(全部超参) | `CB/evolver/scripts/train_grpo-3b.sh` |
| verl 配置默认值(experience/rewards 段) | `CB/evolver/verl/trainer/config/ppo_trainer.yaml` |
| 经验管理器(提炼/检索/去重/评分/清理) | `CB/evolver/evolver/experience/experience_manager.py` |
| 三条蒸馏 prompt | `CB/evolver/evolver/experience/prompts.py` |
| 蒸馏/检索超参常量 | `CB/evolver/evolver/experience/config.py` |
| reward 聚合器 | `CB/evolver/evolver/rewards/__init__.py` |
| 多轮 rollout + info_mask 构造 | `CB/evolver/evolver/llm_agent/generation.py` |
| 训练循环 + reflection 钩子 + loss_mask | `CB/evolver/verl/trainer/ppo/ray_trainer.py` |
| GRPO 组内归一化 | `CB/evolver/verl/trainer/ppo/core_algos.py` |
