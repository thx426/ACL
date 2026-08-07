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
