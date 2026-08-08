# Related Work 资产索引(CD-010,2026-08-07)

`related_work/` 存放与 PGS(采样端分布干预 + 混合 on/off-policy GRPO)直接相关的
论文 PDF 与参考代码库。**本目录除本文件外全部 gitignore**(见根 .gitignore);
复现方式与版本以本文件为准。papers/ 为 arxiv PDF(`curl -sL arxiv.org/pdf/<id>`),
code/ 为浅 clone(`git clone --depth 1`,commit 已钉)。

## 论文清单(papers/)

| 文件 | venue | 一句话 | 与 PGS 的关系 |
|---|---|---|---|
| 2510.04140_MENTOR_ICLR2026.pdf | **ICLR 2026** | token 级混合分布(按学生熵软插值到 expert)+ spec-sampling 执行 + 分组不对称 advantage(on 组自比;expert 行 [R−mean_on]₊×α,α cosine 衰减) | **最近同类**;执行层与 SPS 同构;advantage 口径是 s3 病灶的文献答案 |
| 2504.14945_LUFFY.pdf | NeurIPS 2025 | off-policy 示范混组 GRPO + policy shaping f(π)=π/(π+γ) 反转低概率 token 梯度权重,防熵坍缩 | W=π/b 打折悖论的反方向实证;token 级 prefix_mask 混合处理模式 |
| 2507.13266_QuestA.pdf | 2025 | prompt 层注入部分解 hint,50%→25% 课程退火,提 pass@1 与 pass@k | hint 在文本层(与我们信息状态红线不同路);退火课程参照 |
| 2508.08641_MiGrATe.pdf | 2025 | TTT 搜索场景 mixed-policy GRPO:on + greedy(历史高分)+ neighborhood 同组共享基线 | off 样本全正样本锚的特例;共享基线在该语境无害的参照 |
| 2509.24203_GRPO_secretly_offpolicy.pdf | ICLR 2026 | 组相对 REINFORCE 的原生 off-policy 解释(理论) | 混合训练的理论基础,论文引用用 |
| 2507.18071_GSPO.pdf | 2025(Qwen) | 序列级 IS 替代 token 级(token 级 IS 噪声大) | λ± 块级校正的同方向大厂实证 |
| 2601.18795_ReuseYourFLOPs.pdf | 2026 | 极端 off-policy 前缀条件化做难题 RL | prefix 条件化通道参照(与 v4 前缀续写远亲) |
| 2605.15155_SDAR.pdf | 2026 | **agentic 多轮 + 自条件化教师(π+特权上下文)**;OPSD 作 gated 辅助 loss(λ=0.01,β=5),三式 sigmoid 门:entropy σ(βh)/gap σ(βΔ)/soft-OR;Δ=sg(logπ⁺(y)−logπ(y)) detached | **门控设计的直接参照**(用户点名);教师结构与 PGS 相同;"平均 gap 恒负、无门控蒸馏有害"的结构发现;ALFWorld +9.4%,Skill-GRPO 表面依赖 vs SDAR 内化的实证 |
| 2606.29476_CRAFT.pdf | 2026 | sibling rollouts 反事实信用分配(SDAR 族) | 信用集中方向的参照,未细读 |
| 2606.26790_OPID.pdf | 2026 | on-policy skill distillation(SDAR 族) | 同族竞品,论文引用用 |
| 2608.03223_SD_RewardShaping.pdf | 2026 | 自蒸馏 reward shaping(SDAR 族) | 同族竞品,论文引用用 |

## 代码库(code/)

### MENTOR(`code/MENTOR`,commit b1cc4c6,2026-03-01;EasyR1 系 verl fork + vllm patch)

与我们同族架构(verl + rollout 引擎 patch)。关键文件与读法:

- **advantage(病灶 1 的答案)** `verl/trainer/core_algos.py:171`
  `compute_mixedpolicy_grpo_outcome_advantage`:on 行 mean/std **只用 non-expert
  样本**算(基线污染结构性排除);expert 行 `clamp_min(R − mean_on, 0) × expert_weight`
  (失败零梯度不惩罚);`expert_cap` 组内总额防御闸(默认 99 不激活)。
- **α 调度** `verl/trainer/ray_trainer.py:633`:cosine 从 expert_weight=1.0 衰减,
  `expert_weight_decay_totalstep=120` 后恒 0,**且 α=0 后 mixed rollout 停产**
  (`if use_mentor and expert_weight > 0` 才跑 spec 分支)——教师是有限期脚手架。
- **无行为分布 IS(病灶 2 的答案之一)** `ray_trainer.py:672`:old_log_probs 对
  全 batch(含 mixed 行)统一用 actor 重算——mixed 行 PPO ratio 分母是 π_θold,
  **不存在 W_beh 等价物**;偏差全部由 [·]₊ 与 α 兜。
- **混合数学** `vllm/model_executor/layers/rejection_sampler.py:30`
  `SpecGRPOSampler._batch_modified_rejection_sampling`:
  `scaling = clamp(H_draft/entropy_threshold, max=1)`;
  `π_mix = π + (π*−π)·scaling`(概率域线性插值,**软插值非硬门**,学生自信处
  教师无权干预);然后标准 spec-sampling accept/残差重采。
  阈值挂载点 `vllm_rollout_spmd.py:109`:`entropy_threshold = max(q80, 1.5)`
  ——**1.5 硬地板**防阈值追着熵坍缩跑。
- **阈值反馈环** `verl/workers/actor/dp_actor.py:373`:actor update 时算 batch 熵
  P95(`entropy_p=0.95`)回传 trainer,下一步生效(滞后一步,无 EMA)。
- **执行工程** `vllm_rollout_spmd.py:92`:**teacher 当主模型、学生当 draft**,
  verify 下沉 vLLM 原生 spec-decode(`num_speculative_tokens=4`,自定义
  acceptance_method=SpecGRPOSampler)——logprobs 永不出引擎(天然规避我们踩过的
  prompt_logprobs 序列化 20× 墙钟);教师引擎是**每步临时子进程**(math 单轮可行,
  agentic 多轮不可照抄)。
- **超参真值** `examples/train.sh`:n=8 其中 **n_mixedpolicy=4(4:4 配比,与我们
  8:8 同比例)**;teacher=OpenR1-Qwen-7B(同尺寸 distill);entropy_p=0.95;
  max_response_length=8192;小 KL(low_var_kl,1e-2)。

### LUFFY(`code/LUFFY`,commit 01f4ae0,2026-03-20;verl fork)

- **on/off 混合 loss** `luffy/verl/verl/mix_src/mix_core_alg.py:69`
  `compute_token_on_off_policy_loss`:`prefix_mask` 为 **token 级** off 掩码
  (off 轨迹= R1 前缀 + 学生续写,一条轨迹内混合处理);on token 走 PPO
  ratio+clip;off token 权重 = **f(π_θ)=π/(π+0.1)**(π_φ=1 简化,不除行为
  概率),带 off_max/min_clip 防御。多种 reshape 消融开关内置
  (p_div_p_0.1/0.3/0.5、pow、sqrt)。
- 混组共享基线(1 off : 7 on),有意利用不对称做自动课程(弱时模仿/强时探索)。
- `ExGRPO/` 子目录为后续经验回放工作,未细读。

## 与 PGS 的映射备忘(详见 docs/pgs-s3-diagnosis-cd010.md 与 changelog)

s3 两病灶 ↔ 文献答案:①混合组共享 LOO 基线(bare 被系统性压制→弥散化膨胀)
↔ MENTOR 分组不对称 advantage;②W_beh=π/b 忠实打折教学位 ↔ MENTOR 免 IS /
LUFFY 反向 shaping。我们保留的独有点:教师=π 自身+特权文档条件化(无词表
约束、steered 轨迹文本层信息状态干净、doc 可 researcher 自进化)、agentic
多轮语境(死锁截断、常驻引擎 tail-only verify)。

---

# Benchmark 资产(benchmarks/<name>/;需求 §4 逐环境引入的调研与 baseline 对照物)

每个 benchmark 一个子目录:`papers/`(arXiv PDF)+ `code/`(浅 clone,commit 钉此)。
用途:①引入前读社区**代码实现**定关键抉择(不是只读论文);②实验/论文期
baseline 对照与口径核对。

## scienceworld(2026-08-08)

**papers/**:2203.07540(ScienceWorld 本体,EMNLP22)/ 2403.02502(ETO,ACL24)/
2405.11930(IPR,EMNLP24)/ 2506.01475(PGPO,ACL25F)/ 2507.22844(RLVMR,Tencent)

**code/**(commit·日期):ETO `a2fc5da·2024-10-30` / IPR `93335a3·2024-10-18` /
PGPO `db0d74e·2025-08-07` / ScienceWorld 官方 `c6edd47·2026-08-04` /
DigitalHuman(RLVMR 子目录,sparse)`59ff469·2026-01-29`

**四抉择的社区实测答案(读码结论,两谱系对照)**:

| 抉择 | ETO 谱系(ETO/IPR/PGPO,SFT+DPO) | RLVMR(verl GRPO) |
|---|---|---|
| 数据/划分 | **24 任务**(非 30);train 1483 / dev 194 / test 211,seen/unseen 均**变体级**(`eval_agent/data/sciworld/*_indices.json`) | 29 任务;L0 同分布 / L1 变体级 / **L2 任务级 held-out**(train 19 任务 vs test 10 任务不相交;`variations_idx/L*_idx.json`) |
| reward | `raw_score`∈[0,1] 取**轨迹峰值**(`sciworld_env.py: if reward > self.state.reward` 只增不减)→ 天然免负分 | **二元 10×won**,won=isCompleted∧score>0(宽松:超步限带正分也算;`envs.py:compute_reward`);dense 信号来自 meta-reasoning 标签非环境分 |
| simplifications | **"easy" 硬编码**(ETO/PGPO `reset()` 源码) | **默认 "easy"**(`env_manager.py:563`) |
| 步预算 | **per-task 表** `max_steps.json`:10(lifespan)–120(melting-point),均值≈40;计对话步含格式失败步 | **全局 30**(`run_sciworld.sh: env.max_steps=30`)+ 引擎 envStepLimit=100 兜底 |
| 观察增强 | 无 admissible list(代码里注释掉了) | **每步注入 valid_actions 全表+对象表+示例** |
| 架构 | 对话式 ReAct(`Action:` 正则)+ ICL few-shot | verl-agent step-level(每步独立 prompt,max_response 1024/步) |

另:RLVMR 带 SciWorld **专家轨迹 SFT 冷启动**(`data/sciworld_expert_traj` +
`examples/sft/cold_start/`)——与我们 4B SFT 冷启动先例同构,可比对照。
ETO 的 `replace_sciworld_score.py` 是对官方包 step 的 monkey-patch(加
raw_score 字段),IPR 同款。
