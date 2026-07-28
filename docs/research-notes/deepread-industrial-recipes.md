# 工业级多域 RL post-training 配方调研

调研日期 2026-07-28 · 全部断言标注一手/二手 · 原文未写一律记「未公开」

---

## 结论先行

1. **2026 年工业界主流已倒向「联合/混合多域 RL」**,且出现了明确的立场反转与明确的反对声明:MiniMax 在 M1(先 reasoning 后渐进混入)到 M2(每步四域同时取数)之间自我推翻;Olmo 3 直接写明试过「先 math warm-up 再切混合」的分阶段方案并判定 suboptimal。
2. **但没有任何一家做过「联合 vs 分阶段」的受控消融。** 唯一存在的真消融是 GLM-4.5V 的单域 vs mix-all 5×5 迁移矩阵,以及 Olmo 3 的单域 vs 混合双曲线 —— 两者都是「单域 vs 混合」,不是「分阶段 vs 联合」。这是文献空白。
3. **分阶段派全部要额外造一个「回补阶段」**:GLM-4.5 用 expert 自蒸馏、NVIDIA 用 MOPD、DeepSeek-R1 用 rejection sampling + SFT、Qwen3 用 Stage-3 拒绝采样自复现。分阶段的真实成本不是训练顺序,是必须多养一套回补机制。
4. **跨域 reward 尺度归一化是集体盲区**:七家里零家做 per-domain advantage normalization。实际起作用的是两条替代路线 —— 把所有域压进同一有界尺度(k1.5 全域 {0,1}、Olmo 3 全域 [0,1]、MiniMax-M1 三档量表),或者靠 GRPO 组内归一化「顺带」抹平(但没人讨论其副作用)。
5. **最强的单条实证是 Olmo 3 的单域崩塌曲线**:只训 instruction-following,IFEval 只多涨 1.3pp,AlpacaEval 从 ~50 断崖式崩到 ~4。

---

## A. 工业技术报告逐家

### A1. Tülu 3(AllenAI)— 联合,配比全公开

一手:https://arxiv.org/pdf/2411.15124v3 (arXiv HTML 渲染被截断,须读 PDF) · https://allenai.org/blog/tulu-3-technical

- **组织方式:联合。** §6.1 原文 "We combine all prompts together, results in a mixture of roughly 30,000 prompts with ground truth labels"。RLVR 是 SFT→DPO→RLVR 流水线的第 4 阶段,故相对 SFT 分阶段、跨域联合。
- **配比:全公开(Table 22)。** GSM8K 7,473 / MATH 7,500 / IF-verifiable 14,973 = **29,946**(HF 数据集行数已核)。但报告从未把这个比例当设计选择讨论,也没有额外采样权重。405B 反而只用 MATH(§8.1,因 GSM8K 已饱和、IFEval 无增益)。
- **归一化:无 per-domain。** 全域统一 α=10 二值奖励(§6 式 8,"We set α = 10 based on pilot experiments and did not tune it further"),advantage 是 batch 级全局 whitening。**明确拒绝混入 RM 分数**(§6.2.1 小标题就叫 "Do Not Use the Scores from RM.",理由是引入噪声)。
- **失衡:承认且有表。** Table 23(8B DPO→RLVR):目标域 GSM8K +3.3 / MATH +1.7 / IFEval +1.3,非目标域 MMLU −0.5、TruthfulQA −1.1、Safety −1.7,总均值仅 +0.4。§6.4 原文承认高分模型 "tended to perform worse in other metrics, dragging down their overall average"。对策全是钝器:KL 锚(8B β=0.05)、每 40–100 步选点、405B 直接砍域。
- **监控:** §6.2 "we look at the verifiable rewards, the KL divergence, and the response length" + 中间 ckpt 的 per-domain eval(App. E.1 明确警告会过拟合)。代码里有未文档化的 per-verifier `objective/{gsm8k,math,ifeval}_correct_rate`。
- **消融:无联合 vs 分阶段。** 单任务 run 存在但只有曲线(Fig 19),无数字表。
- 补:Tülu 3.1 换 GRPO 去 RM,均值 64.8→66.3,但 **Safety 85.5→81.2(−4.3)且未作解释** —— 又一条未被承认的失衡。

### A2. Olmo 3(AllenAI)— ★ 证据最强的一家

一手:https://arxiv.org/abs/2512.13961(118 页,须下载 PDF 读)· https://allenai.org/blog/olmo3

- **组织方式:联合,且明确否掉分阶段。** §4.4.3 "Our final RL run ended up mixing carefully-filtered data from all domains roughly equally"。§5.4.1 脚注 56 逐字:"alternative RL setups—for example, **first warming up on math-only data and then switching to a mixed dataset without math—resulted in suboptimal performance**"。这是工业报告里唯一一条对 staged RL 的直接负面判定。
- **配比:完全公开(Table 20)。** Dolci-Think-RL 共 104,869:Precise-IF 30,186(28.8%)/ Math 30,186(28.8%)/ Code 23,110(22.0%)/ General-Chat 21,387(20.4%)。IF 与 Math prompt 数**精确相等**,General 三源**精确各 7,129** —— 「roughly equal + 略偏 math/IF」的落地形态。Instruct 版 171,950,配比改为 Math 37.2 / Gen 29.0 / IF 22.1 / Code 11.6。
- **归一化:无 per-domain。** 尺度对齐全靠**所有 verifier 都输出 [0,1] 标量**(Figure 16)。算法 = GRPO + DAPO(clip-higher 0.272、token-level loss、零梯度组过滤)+ Dr GRPO 去 std 归一化 + **无 KL loss、无 entropy bonus**。
- **★ 失衡:给出了全调研最干净的单域崩塌实证(§4.5, Figure 20)。** 在 Think-SFT-7B 上跑 1000 步,IFEval-only vs Mixed:单域把 IFEval 推到 83.8(混合 82.5,仅多 1.3pp),代价是 **AlpacaEval 从 ~50 断崖崩到 ~4**,拐点在 500→600 步(41.2→23.0→9.8)。原文归因明确:"the **LM-judge reward ensures that the model continues to produce well-formed chat responses**"。
  ⚠️ 上述 AlpacaEval 数值系子代理从矢量图读数(原文未印表),趋势与量级可靠,单点数字勿直引。
- **另一条:混合的税在域间极不均匀(Figure 21,图读数)。** 同为 1K 步,混合相对单域的 train reward:IF **−1.7**、General **−1.0**、Math **≈0**(两条曲线基本重合)。即混合压的是最易 reward-hack 的两域,math 免疫。
- **长跑仍有慢性失衡(Table 14,原印数字)。** 同一 mixture 从 750 步延到 2300 步:IFBench +20.5、AIME25 +5.6,但 **AlpacaEval 74.2→69.1(−5.1)**、GPQA −1.4、BBH −1.2。全链看 32B 的 AlpacaEval:DPO 78.6 → RL(750) 74.2 → RL(2300) 69.1,**即使混合,chat 质量仍随 RL 步数单调下滑**。
- **监控(Figure 41,九格 dashboard):** overall reward、**KL(被监控但不进 loss,纯诊断量)**、response length(先降后升)、5 个 per-verifier reward 分域曲线。**不监控 entropy。** 另有一张极实用的**评测方差表**(§4.1.1)用于区分噪声:高方差 GPQA 1.48 / AlpacaEval 1.24 / IFEval 0.88,低方差 MATH 0.25 / MMLU 0.22 / PopQA 0.16。
- **难度过滤:8 rollouts,砍 pass rate > 62.5% 的(只砍太易,不砍全错);general-chat 用 F1∈[0.1,0.8] 双边过滤。** RL-Zero 更宽松,只砍 8/8 全对。
- **消融:** 有 SFT 起点 vs DPO 起点(avg 71.9 vs **74.1**,Table 22);有 active sampling(维持满 batch,标准做法 1.0 跌到 ~0.5);**无联合 vs 分阶段的数值消融**。§6.2 承认 RL-Zero mix run "each domain is under-optimized compared to the single-domain setup",但 ⚠️ 该对比有严重步数混淆(mix 只跑 600 步,单域跑 2000–4500 步),不可直接引为「混合更差」。

### A3. Qwen3 — 严格分阶段,且公开承认接受损失

一手:https://arxiv.org/pdf/2505.09388v1(**HTML 版截断且完全缺 §4.4 General RL,勿用**)

- **组织方式:严格分阶段,RL 中间还夹一个 SFT。** 四阶段:Long-CoT 冷启 SFT → **Reasoning RL** → Thinking Mode Fusion(**继续 SFT**,§4.3 "we conduct continual supervised fine-tuning (SFT) on the Reasoning RL model")→ **General RL**。两次 RL 从不混批、从不交替。
- **配比:基本未公开。** Stage 2 只给 **3,995 query-verifier pairs**;Stage 4 只有 "over 20 distinct tasks",query 数与各类占比全部未公开。
- **归一化:完全未公开。** §4.4 列了三类 reward(rule-based / model-based with reference,用 Qwen2.5-72B-Instruct 打分 / model-based without reference,标量 RM),但**如何组合、加权、裁剪、对齐尺度只字未提**。Stage 4 用什么算法也未公开(GRPO 只在 Stage 2 声明)。
- **★ 失衡:承认得最直白,并选择不修(§4.7 结论 3)。** 原文:"the performance in thinking mode **actually decreases** after these two training stages … we conjecture this degradation is due to the model being trained on a broader range of general tasks, which may compromise its specialized capabilities … **we choose to accept this performance trade-off**"。Table 22(32B,Stage2→Stage4):IFEval +12.0、Multi-IF +11.6、ToolUse +22.2,但 **AIME'24 −2.4、LiveCodeBench −2.7、MMLU-Redux −0.5、GPQA −0.4**。
  结构性读法:**AIME 掉分主要发生在 Stage-3 的 SFT(−1.9)而非 Stage-4 的 RL(−0.5)**;LiveCodeBench 则相反(SFT −1.2 / RL −1.5)。
- **对策:仅一条,且在 SFT 层。** §4.3 "To ensure that the performance of the Stage 2 model is not compromised by the additional SFT, the 'thinking' data is generated via **rejection sampling on Stage 1 queries using the Stage 2 model itself**" —— 上一阶段自身输出的 on-policy 自复现。Stage 4 无 KL 锚、无 replay、无 per-domain 门控。
- **监控:** training reward、validation 性能、**entropy 被显式当控制目标**("controlling the model's entropy to increase steadily or remain stable",但控制方法未公开)。另自建 4 个内部基准 CounterFactQA / LengthCtrl / ThinkFollow / ToolUse。不报 KL、不报 length。
- **消融:无联合 vs 分阶段**,只有 stage-additive 的 Table 22。
- 旁证(与我们直接相关):**蒸馏 vs RL,§4.7 Table 21** —— 同起点,RL 花 **17,920 GPU·h**,on-policy 蒸馏花 **1,800 GPU·h**(≈1/10)且效果更好;更关键的是 "**reinforcement learning does not lead to any improvement in pass@64 scores**"(RL 把 AIME'24 pass@1 推了 +12.6,pass@64 却纹丝不动停在 90.0)。

### A4. DeepSeek-R1 — 四阶段,但第二次 RL 内部是「同批求和 + 晚掺入」

一手:https://arxiv.org/pdf/2501.12948 · **Nature 版(2025-09)信息量远大于 arXiv 版**:https://www.nature.com/articles/s41586-025-09422-z(直连会被认证挡,可走 Europe PMC `PMC12443585`,SI PDF 为 `41586_2025_9422_MOESM1_ESM.pdf`)

- **组织方式:四阶段流水线,但第二次 RL 是同 batch 联合。** Nature Methods 式(8):**Reward = Reward_reasoning + Reward_general + Reward_language**。
  **★ 关键限定(仅 Nature 版有):** "The stage comprises a total of **1,700 training steps**, during which general instruction data and preference-based rewards are incorporated **exclusively in the final 400 steps**. We find that **more training steps with the model-based preference reward signal may lead to reward hacking**" —— 即「联合但只暴露 400/1700 步」的晚掺入策略。
- **配比:Nature SI Table 1 公开。** Math 26K / Code 17K(另 8K bug-fixing)/ STEM 22K / Logic 15K / General 66K(另 12K harmlessness)。**但 batch 内采样权重未公开**。
- **归一化:直接相加、等权、无 per-domain。** "The accuracy, reward and format reward are combined with the same weight";language 一致性奖励 "directly summing them"。二值规则奖励与连续 RM 分数直接相加,**量纲问题原文完全没讨论** —— 全调研中最粗糙的一处。唯一归一化来自 GRPO 的 per-prompt 组内相对优势。
- **失衡:Nature SI 有量化证据。** ① **Supplementary Fig. 4 图注:"Reward hacking: the reward exhibits an increasing trend as the performance on CodeForces decreases for training."** ② SI Table 9:R1-Zero(纯 reasoning RL)相对 V3-Base **IF-Eval 58.6→46.6**、AlpacaEval 仅 24.7(V3 为 70.0)、Aider 仅 12.2(V3 为 49.6);四阶段把这些补回(83.3 / 87.6 / 53.3),代价是 **GPQA 75.8→71.5、CNMO 88.1→78.8**。
- **对策:** rejection sampling + SFT 回补(600k reasoning + 200k non-reasoning);通用 reward 晚掺入 400 步;拒绝 neural PRM。
- **监控:** AIME pass@1(16 采样均值)vs steps、平均 response length、**reward 与下游 benchmark pass@1 同图**(SI Fig 4,这正是 reward hacking 的检测手段)、language consistency 比例。
- **消融:无联合 vs 分阶段。** 有 LC-reward 消融(仅曲线,"comparable on math, slight degradation on coding",无数字)。
- 细节(对双角色有用):**helpfulness reward 只作用于 final summary**("minimizing interference with the underlying reasoning process"),**harmlessness 作用于含 reasoning 的整个响应**。

### A5. Kimi k1.5 / K2 — 联合;k1.5 靠二值化根除尺度问题,K2 靠 PTX 防遗忘

一手:https://arxiv.org/html/2501.12599v1 · https://arxiv.org/abs/2507.20534(K2,id 已核)

**k1.5**
- 组织:**单一联合 RL**(文本+视觉 joint),分阶段的维度是**难度而非域**(curriculum);long2short 是独立后置阶段。
- 配比:**RL 侧未公开**(⚠️ 易混点:§2.5.2 的 500k general / 200k coding 等是 **vanilla SFT** 数据,不是 RL prompt set)。
- **★ 归一化:强制全域 r(x,y,y*) ∈ {0,1}** —— 连自由形式答案也训一个 RM 来「判对错」而非打偏好分。**从根上回避了跨域尺度混合**,与 R1 的异构直接相加恰成对照。
- 失衡:承认的是 **overthinking(长度膨胀)** 而非跨域遗忘。对策 = length penalty(先关后开的两段式 warm-up)+ long2short。防 hacking:排除选择/判断/证明题;用「无 CoT 直接猜、8 次内猜中就删」过滤。
- 监控:training accuracy 与 response length 联合演化、per-problem 成功率 s_i、pass rate 难度代理、token efficiency。
- 采样权重(明确公式):**prioritized sampling 按 1−s_i 成比例**;curriculum 无公式,做法是全量 warm-up 后只训 hard。
- 消融:有 curriculum vs uniform、negative gradient vs ReST —— **均只有曲线无数字**;**无多域联合 vs 单域**。

**K2**
- 组织:**全调研中最明确宣称联合的一家**,abstract 与 §3.2.3 都用 "joint RL training"。
- 配比 / 归一化 / 监控指标 / RL 消融:**全部未公开**(报告 RL 章无一条训练曲线、无一个消融数字)。
- **失衡:明确承认。** §3.2.3 开篇 "a primary challenge is achieving **consistent performance improvements across all domains**"。三条对策:① **per-task-type token budget**(超出即截断+惩罚,唯一的 per-domain 差异化处理,数值未公开);② **PTX loss** —— 把人工精选高质量数据作为辅助损失混入 RL 目标,"prevent the potential forgetting … mitigates the risk of overfitting to the limited set of tasks";③ temperature decay(探索→利用)。
- **★ 最有价值的机制:closed-loop critic(§3.2.2)。** "**On-policy rollouts generated from verifiable-reward prompts are used to continuously update the critic** … grounds its more subjective judgments in verifiable data" —— 用可验证域的 on-policy rollout 持续再训 critic,让可验证信号锚定不可验证域的评判标准。

### A6. GLM-4.5 / GLM-4.5V(Zhipu)— expert-then-distill,以及唯一的真消融矩阵

一手:https://arxiv.org/abs/2508.06471(无 HTML,须读 PDF)· https://arxiv.org/abs/2507.01006(4.1V/4.5V)
GLM-4.6 **无技术报告**,RL 配方未公开(HF 卡指回 4.5)。⚠️ 二手源把 RLCS 算作 GLM-4.6 是张冠李戴,RLCS 出自 2507.01006。

**GLM-4.5:expert-then-distill 属实,是全文骨架。** §3 原文:"In stage 1 (**Expert Training**), we construct expert models specializing in three domains: **Reasoning, Agent, and General chat**. In stage 2 (**Unified Training**), we employ **self-distillation** techniques to integrate multiple experts"。合并方式**不是权重合并**,是用各 expert 的输出对 base 重新 SFT(§3.1 Overall SFT,"By distilling from the output of distinct experts"),蒸馏前过四级 rejection sampling。合并后是否还有统一 RL:**报告表述有歧义,判为未明确公开。**

- **★ 全调研唯一的「拆 vs 合」工程判据(§3.4):** "We incorporate step-wise rule-based RL **directly into our general RL framework due to their similar output lengths and convergence speeds**. For end-to-end multi-turn RL, **we first train specialized expert models and then distill these experts into the main model**." —— 输出长度与收敛速度相近就联合,差异大就拆 expert 再蒸馏。
- 配比:跨域比例未公开;域内公开 Holistic RL ≈5,000 prompts(7/33/139 三级分类)、IF 约束体系 7 major / 151 minor。
- 归一化:跨域未公开。域内:GRPO 去 KL;agentic 目标只做组内均值中心化;code RL 用 **token-weighted mean loss** 替代 sequence-mean 以缓解长度偏置。
- 失衡:**跨域遗忘未讨论**,只承认长度维度且措辞很强 —— "introducing RL stages with shorter maximum lengths can cause the model to **'unlearn' its long-context capabilities** … a significant and **irreversible** drop"。对策 = 直接单阶段 64K RL。
- 监控:reward 方差退化(全 1/全 0 batch)、平均 reward 停滞判收敛后提温、**留出集扫温度取「相对最优跌幅 ≤1%」的最大温度**、pass@8=0 且 pass@512>0 界定极难题。
- 消融:**无 expert-then-distill vs 联合 RL 对比**(最大证据缺口)。有:两阶段难度课程 83.4 vs 81.8;**单阶段 64K 83.4 vs 多阶段渐进长度 80.6**;science RL 专家验证单选 65.8 vs 混杂 62.9。

**GLM-4.5V(2507.01006)—— 同实验室在视觉线选了联合,并做了 4.5 缺的那个消融。**
- **★ §6.3 Figure 6 单域 vs mix-all 5×5 矩阵**(单域样本量严格等于该域在 mix-all 中的量),行=训练域、列=评测域(STEM / OCR&Chart / Grounding / GUI / General-VQA),数值为相对 SFT ckpt 的提升:
  - **Mix All:+6.6, +5.2, +30.1, +6.0, +3.4**
  - Grounding 单域:−0.5, **−4.6**, +30.4, +8.1, −0.4
  - OCR&Chart 单域:+3.0, +1.2, **−14.5**, +5.8, +1.3
  - 结论原文:mix-all "delivers clear gains over any single-domain RL in **three out of five** areas … however, **mixed-domain training does not improve grounding or GUI-agent performance**"。
  - 即:混合在 5 域里赢 3;单域训练之间存在**剧烈负迁移**(OCR 训练把 Grounding 打到 −14.5),混合把负迁移抹平了。
- **★ 联合的真实失败模式(§5.2, Figure 5):** "**any weakness in the reward signal for a single capability can derail the entire training** … a flaw in the reward for the multi-image QA task led to **model collapse across all domains**"(step 150 后连 reward 质量很高的 STEM 也停滞,MMMU/MathVista/AI2D 急跌)。**最弱的 verifier 决定整个联合 run 的生死。**
- 另一条关键定性(§5.3.1):"**Domain interference in RL is less pronounced than in SFT.**"

### A7. MiniMax M1 → M2 — 一年内自我推翻,「分阶段 vs 联合」最强一手反证

一手:https://arxiv.org/abs/2506.13585(M1)· https://arxiv.org/abs/2605.26494(M2,标题 "The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence",2026-05-26,已核)

**M1:分阶段渐进混合。** §4.3 标题即 "Curriculum of Incorporating Diverse Data":"**we start with only the reasoning-intensive tasks with rule-based reward, and then gradually mix in the general domain tasks**",并声称 "It **prevents catastrophic forgetting** of specialized skills"(⚠️ 纯声明,无消融支撑)。
- 配比:各域数据量公开(math ≈50K / logic 53K / code 30K / SWE 数千 / general 25K),混合比例与权重曲线未公开。长度阶梯 40K→48K→…→80K。算力 512×H800 / 3 周 / $534,700。
- 尺度对齐:把各域压进**有界离散量表** —— 可验证域 0/1、有 GT 的一般域五级量表、无 GT 的 pairwise **−1/0/+1**。
- **★ generative RM 的长度偏置(§4.2.2 专节):** "GenRMs **preferred longer outputs** over potentially superior concise alternatives … inducing reward hacking"。离线三招(多样化训练数据/对抗样本/改结构)**被判定失败**("frequently failed to prevent length bias during RL training"),真正管用的是**在线监控 + 触发式 GenRM 重标定**。
- 监控:在线长度偏置指标、生成序列 perplexity 收敛、**输出长度 99 分位是否逼近窗口上限**(长度阶梯升级准入判据)、重复检测(连续 3,000 token 每个 p>0.99 即早停)。
- 消融:有 CISPO vs DAPO vs GRPO(~2× 加速,50% 步数追平 DAPO);**无多域组织方式消融**。

**M2:§6.1.6 标题就叫 "Mixed-Domain RL Training",同时否掉单域与顺序分阶段。** 逐字:
> "**Single-domain RL training … risks catastrophic forgetting** … Conversely, **sequential multi-stage training across domains induces negative transfer, as gains in one domain erode performance in previously trained domains.** We adopt a **mixed-domain RL training strategy** … within each stage, training data is drawn **simultaneously from four domains: reasoning, coding, agent, and general**. This joint optimization ensures that the policy gradient updates are informed by a diverse task distribution **at every training step**"
- 按 stage 调三个轴:**domain mixing ratio**(早期偏 reasoning+general 打底,后期抬 agent+coding)、**context length(per-domain 粒度)**、**difficulty**。三轴具体数值全部未公开。
- 优势估计换成 reward-to-go + trajectory-level baseline(非 GRPO 组内 std);CISPO 裁剪改为非对称、下界取 0。
- 消融:**无 mixed vs staged vs single-domain 的对照数字**(只有定性论断)。

### A8. NVIDIA Nemotron-Cascade 2 — 反潮流的分阶段派,配 MOPD 回补

一手:https://arxiv.org/abs/2603.19220(63 页)· 前作 NC1 https://arxiv.org/abs/2512.13607

- **Cascade RL = 按域顺序分阶段 RL,明确与 R1/Qwen3 的联合路线对立。** NC1 §1 逐字:"Cascade RL **trains models sequentially across domains**, in contrast to approaches such as DeepSeek-R1 and Qwen3, which **blend diverse prompt distributions from all (reasoning) domains for joint RL training**."
- NC2 顺序:SFT → IF-RL → **Multi-domain RL** → **MOPD** → RLHF → Long-context RL → Code RL → SWE RL。**顺序不是常量**(§4.1.1 "The optimal ordering … is **not a universal constant**")。排序原则是 "identify an ordering that **minimizes negative interference across domains**"。
- **★ MOPD(多域 on-policy 蒸馏)= 显式的回补阶段。** §4.1.1:"It is **effective at recovering benchmark performance that may have regressed during earlier, more specialized stages** of the cascade RL"。§4.4 给病因:"it does **not fully eliminate capability drift** as the number of training environments increases … certain RLVR training often reduces model entropy and shortens reasoning traces, thus can negatively impact mathematical reasoning performance, while RLHF-oriented optimization can partially trade off against instruction-following behavior."
  机制:3 个 **冻结的历史 checkpoint** 当 domain teacher(math teacher = 初始 SFT ckpt;RLHF teacher = 从 SFT 分叉单训的支线;multi-domain teacher = IF+MD 阶段后的 ckpt),student 自己 rollout,按域路由到对应 teacher,token 级 **reverse-KL 作为 stop-gradient 优势**塞进 policy-gradient(而非直接最小化 KL),配截断重要性权重 [0.5, 2.0] 与 30 步 warm-up(2e-7→2e-6)。**40–52 步收敛**,reverse-KL→0 即收敛判据。
- **部分域仍走联合(§4.3),判据与 GLM-4.5 撞车:** 合并 STEM-MCQA 55% / agentic tool-calling 30% / structured-output 15% 成一个阶段,理由 ①"we do not observe performance degradation" ②"**the response lengths and verification times of these datasets are similar**"。反向证据(§4.6):"incorporating other domains during long-context RL **negatively affects performance on unrelated benchmarks**"。
- 归一化:**无 per-domain**;跨域尺度问题是**用架构(分阶段)绕开的,不是用数值归一化解决的**。唯一通用归一化是 GRPO 组内标准化(去 KL、去 entropy bonus)。
- 失衡的量化证据:**未公开** —— NC2 把 NC1 的逐阶段 benchmark 表与曲线**全部删掉了**,所有回退描述均为定性。MOPD 唯一带数字的是效率(ArenaHard v2:RLHF 160 步 80.7/71.2 vs **MOPD 52 步 85.5/71.0**;AIME25 GRPO 25 步 89.9→91.0 vs MOPD 30 步 →92.0 "recovers teacher-level performance")。
- **消融:无 cascade vs joint 数值对照(未公开)。** 唯一「分阶段 vs 混合」数值消融在 SFT 层(§3.2.9:纯 agentic Pass@1 48.9 vs agentic+agentless 49.9)。
- 监控:Figure 3 仅 reverse-KL / grad_norm / AIME25 三格;per-benchmark-category 验证隐含存在(teacher 选择依赖它)但**曲线未公开**;response length 是多处早停判据。

### A9. Seed1.5-Thinking / Llama 4(补充)

**Seed1.5-Thinking**(https://arxiv.org/abs/2504.13914)—— 完全联合,且把域间干扰讲得最直白:
- §4.2 "a **unified reinforcement learning framework that seamlessly fuses data from a broad range of domains**"(verifiable / general / 两者混合三类)。
- **★ 承认得最彻底:** "we face the challenge of **interference between different data domains** … This interference can arise from **disparities in difficulty levels, the risk of reward-hacking** … These issues make it **extremely difficult to achieve uniform and simultaneous improvements across all capabilities**." 对策 = **Online Data Distribution Adaptation**(把静态 prompt 分布变成自适应分布),但**具体算法、公式、超参全部未公开**。
- **★ 唯一明确把「跨范式 reward 尺度冲突」当问题解决的一家(§3.2):** 用 **pairwise generative RM** 替代常规打分式 RM,靠压掉离群分让一般域分布不至于与 verifier 的 0/1 差太远 —— "**mitigating outlier score generation** … therefore **avoiding significant discrepancies in score distributions with the verifier**"。
- 配比:域内公开(STEM 100k / logic ≈10k / SFT 400k = 300k verifiable + 100k non-verifiable),**跨域混合比例未公开**。
- 一条有用的过滤判据:剔除**得分方差低**的 prompt,以及在前代 RL 中 reward 提升已超阈值的 prompt,理由 "overoptimizing such samples leads to **premature collapse of the model's exploration space**"。

**Llama 4**(https://ai.meta.com/blog/llama-4-multimodal-intelligence/)—— 博客体裁,唯一给出明确过滤比例的一家:
- 流水线重排:"**lightweight SFT > online RL > lightweight DPO**",动机是 "**SFT and DPO can over-constrain the model, restricting exploration during the online RL stage**"。
- **难度过滤比例:Maverick "removed more than 50% of our data tagged as easy"、Behemoth "prune 95% of the SFT data, as opposed to 50% for smaller models"。**
- 连续在线 RL:"alternated between training the model and then using it to **continually filter and retain only medium-to-hard difficulty prompts**";"**constructing training batches with mixed prompts from multiple capabilities** were instrumental";"sampling from a **variety of system instructions** was crucial in ensuring that the model **retained its instruction following ability**"。
- 归一化 / 监控 / 消融:全部未公开。

---

## B. 2026 新工作清单(arXiv id 已逐条核验)

**先说重复项**(与既有覆盖撞车,不必再读):CTWA = arXiv **2602.06869**《Uncovering Cross-Objective Interference in Multi-Objective Alignment》;RDPO = **2605.13641**;MGS = **2602.02301**《Modular Gradient Surgery》;GDPO = **2601.05242**(NVIDIA,多 reward 下 GRPO 组归一化会把不同 reward 组合塌成同一 advantage,故解耦每个 reward 各自归一化)—— 若 brief 里的「GD²PO」即此,则重复;若不是,值得补读,它是少数直面「多 reward 归一化」的工作。

| # | arXiv id | 一句话机制 | 相对既有覆盖的增量 | 建议 |
|---|---|---|---|---|
| 1 | **2510.19178**《Imbalanced Gradients in RL Post-Training of Multi-Task LLMs》 | 实证:RL post-training 中不同任务梯度幅度差异巨大,**且大梯度 ≠ 大学习增益**;失衡无法用 training reward / advantage 解释,源于任务固有差异 | **对 GradNorm/PCGrad 一类「按梯度范数均衡」的直接证伪性证据** —— 既有覆盖全是方法,这是唯一说「这些方法的前提可能不成立」的诊断工作。作者明说只提出问题不给解法 | ★深读 |
| 2 | **2606.30406**《MOPD: Multi-Teacher On-Policy Distillation for Capability Integration》 | per-domain 专家 RL 产出 teacher,再在 **student 自己的 rollout 上**多教师蒸馏合并 | **唯一把 Mix-RL / Cascade RL / Off-policy FT / Param-Merge 放同一张表比较的学术工作** —— 正好补上 GLM-4.5 与 NVIDIA 都缺的那个消融;且「专家可并行独立开发」的结构与我们双角色同型 | ★★最值得深读 |
| 3 | **2605.25443**《Harmony in Diversity: MCPO》 | 多域对比式策略优化:他域可迁移轨迹作正样本、错误 rollout 作负样本,做**表示级**一致性对齐 | 干扰缓解从**梯度层移到表示层**,且目标是「促进跨域知识共享」而非仅「避免打架」—— 既有覆盖(PCGrad/GradNorm/MGS)全在梯度层 | 深读 |
| 4 | **2505.24871**《MoDoMoDo》 | 学一个「混合权重 → RL 微调结果」的预测器,再据此优化配比;**+5.24% vs 均匀混合**,+20.74% vs 起点 | **唯一对「配比怎么定」给出可计算答案的工作**(工业界全是手调 + roughly equal) | 深读(⚠️ 2025-05,略早于 H2) |
| 5 | **2507.05386**《RFT Naturally Mitigates Forgetting in Continual Post-Training》(v6 2026-06) | RL 微调遗忘显著小于 SFT,且**明确否定 KL 惩罚与 CoT 是主因**,归因于选择性更新机制 | 为 GLM-4.5V 的 "Domain interference in RL is **less pronounced than in SFT**" 提供机理侧独立佐证;也反向说明「RL 改不动的东西」 | 选读 |
| 6 | **2603.06610**《CapTrack》 | 把遗忘重定义为**多维行为漂移**(鲁棒性 / 默认行为 / 校准 / 多语言),而非只是知识丢失;IFT 伤最重,偏好优化能部分恢复 | **监控口径的增量** —— 工业报告的失衡监控普遍只看 benchmark 均值 | 选读 |
| 7 | **2605.09608**《Geometry Conflict / GCWM》 | 遗忘 = 任务诱导的协方差几何与模型演化状态几何失配;提出 **data-free** 的 Wasserstein barycenter 合并,用几何冲突当门控信号 | 无需 replay 的对策,与「必须留 replay 数据」的工业做法正交 | 选读 |

---

## C. 横向结论:联合 vs 分阶段

**主流选择已明显倒向联合/混合,但不是压倒性的。**

- **联合派:** Olmo 3(四域 roughly equal 一次训完)、Kimi K2("joint RL stage")、Seed1.5-Thinking("unified … seamlessly fuses")、MiniMax-M2(每步四域同时取数)、Tülu 3(30K 单 mixture)、GLM-4.5V(mix-all)。
- **分阶段派:** Qwen3(四阶段)、GLM-4.5(expert-then-distill)、NVIDIA Nemotron-Cascade 1/2(sequential domain-wise)、MiniMax-M1(渐进混入)、DeepSeek-R1(四阶段,但末段 RL 内部同批求和)。

**三条可靠的横向观察:**

1. **分阶段的真实成本是必须额外造一个回补机制。** GLM-4.5 的 expert 自蒸馏、NVIDIA 的 MOPD、R1 的 rejection-sampling SFT、Qwen3 的 Stage-3 自复现 —— 四家独立收敛到同一形态:**用冻结的历史 checkpoint 或分域 expert 的输出,把被后续阶段训掉的能力蒸馏回来**。NVIDIA 说得最直白:cascade "does **not fully eliminate capability drift**"。
2. **两家独立收敛到同一条「拆/合」工程判据:输出长度与验证耗时是否相近。** GLM-4.5 §3.4("similar output lengths and convergence speeds" → 直接混进联合 RL;否则拆 expert 再蒸馏)与 NVIDIA §4.3("response lengths and verification times of these datasets are similar")措辞几乎一致。这解释了为什么 agentic/long-context(长轨迹、慢验证)总被单拎出来。
3. **联合的真实失败模式不是「优化打架」,而是「最弱的 verifier 拖垮全场」。** GLM-4.5V §5.2 是唯一直接实证:multi-image QA 单项 reward 有缺陷 → **all domains collapse**,连 reward 质量很高的 STEM 也停滞。

**与学术结论的呼应与冲突:**

- **呼应「on-policy 联合更稳」:** Olmo 3 Figure 20(单域 IF-RLVR 把 AlpacaEval 打到 ~4,混合则稳)、GLM-4.5V(单域间剧烈负迁移,mix-all 抹平)、MiniMax-M2(明确判 sequential 有 negative transfer)。三家方向一致。
- **冲突:NVIDIA 是唯一反例**,且立场是论述性的(NC1 §1 直接批评 joint 训练 "complicates the RL infrastructure … ultimately leading to suboptimal performance"),**但两代都没给 joint 基线数字**。所以这条冲突的证据强度不对称:联合派有实证曲线,分阶段派只有论述 + 一个回补机制。
- **呼应「SFT 比 RL 更伤」:** Qwen3 Table 22 显示 AIME 掉分主要发生在 Stage-3 的 **SFT**(−1.9)而非 Stage-4 的 RL(−0.5);GLM-4.5V "domain interference in RL is less pronounced than in SFT";arXiv 2507.05386 从机理侧同向。
- **对「SFT-then-RL 反超混合」的补充而非反驳:** Qwen3 证明**新行为模式(non-thinking)只能靠 SFT 引入**(Stage 3 是 continual SFT),这与我方 `opd-capability-boundary` 结论一致 —— RL 只调已有行为分布,新动作走 SFT。同时 Llama 4 给出反向约束:"SFT and DPO can **over-constrain** the model, restricting exploration during the online RL stage",所以 SFT 要「轻」。

**唯一确定的空白:七家 + 两条产品线,零个「联合 vs 分阶段」的受控消融。** 现存的两个真消融都是「单域 vs 混合」(GLM-4.5V 5×5 矩阵、Olmo 3 Figure 20/21)。若我们能给出双角色场景下的联合 vs 分阶段对照,是有位置的。

---

## D. 适用性提炼:对 TRACE-OPD 双角色最值得抄的 5 条

**① 联合同批训练,并靠「锚 reward」而非正则防单角色过优化。**
出处:Olmo 3 §4.5(一手,强)—— 全程 **no KL loss、no entropy bonus、no per-domain normalization**,唯一防崩机制是把 general-chat LM-judge reward 混进 mixture,原文 "the **LM-judge reward ensures that the model continues to produce well-formed chat responses**"。单域时 AlpacaEval 崩到 ~4,混合时守住 ~52。
对我们:researcher 的 reward(下游 rollout 成功率)天然是 executor 的锚;反过来 executor 通道缺锚。**风险点是只训 executor 侧时 researcher 的产出质量无人守**,应考虑给 executor 批次里恒定混入一路由 researcher 文档质量驱动的 reward。注意 Olmo 的锚是**外部固定 judge(Qwen3-32B,权重不动)**,结构上正好绕开我们 CD-003 定位的 moving self-teacher 正反馈病因。

**② 把两个通道的 reward 压进同一有界尺度,不要异构直接相加。**
出处:Kimi k1.5 §2.3.1 全域 {0,1}(一手,强)、Olmo 3 Figure 16 全域 [0,1](一手,强)、MiniMax-M1 三档量表(一手,中强);反面教材 DeepSeek-R1 的二值规则奖励 + 连续 RM 分数直接等权相加,**原文对量纲问题只字未提,且自己吃了 reward hacking**(Nature SI Fig 4)。Seed1.5 §3.2 是第三条路:用 pairwise generative RM 压离群分,让主观域分布不至于与 verifier 的 0/1 差太远。
对我们:executor 的 OPD/GRPO 信号与 researcher 的下游成功率若尺度不一,优先做**有界化**而不是加权调参。

**③ reward 曲线必须与下游 benchmark 曲线同图看。**
出处:DeepSeek-R1 Nature SI Fig 4(一手,强,图注即 "the reward exhibits an increasing trend as the performance on CodeForces decreases")、GLM-4.5V Figure 5(一手,强)、Olmo 3 Figure 41 九格 per-verifier dashboard(一手,强)。
对我们:这与 `thought-repeat-metric` 那条「测模型状态而非任务难度」同构。**具体可抄的是 Olmo 3 的 per-verifier 分域 reward 面板 + KL 作纯诊断量不进 loss**;另外 Olmo 3 §4.1.1 那张评测方差表(GPQA 1.48 / AlpacaEval 1.24 高方差,MATH 0.25 / MMLU 0.22 低方差)是判断「这次波动是不是噪声」的现成工具。

**④ 用「输出长度与验证耗时是否相近」决定两个角色能不能同批。**
出处:GLM-4.5 §3.4 与 NVIDIA NC2 §4.3(两家独立收敛,一手,强);配套 Kimi K2 的 **per-task-type token budget**(一手,中强)。
对我们:**executor 轨迹与 researcher 文档的长度分布差异很大**,这正是该判据判「拆」的情形。两个可选做法 —— 要么按 K2 给每类任务独立 token budget(超出截断+惩罚),要么按 GLM/NVIDIA 拆开训再蒸馏合并。考虑到我们已有 4B/A800 的显存天花板与长序列敏感性(见 `a800-4b-vram-ceiling`),**per-role budget 是成本最低的一步**。

**⑤ 难度过滤砍掉一半以上,且优先砍「太易」而非「太难」。**
出处:四家独立收敛(强)—— Llama 4 砍 >50%(Behemoth 95%)、Olmo 3 砍 pass rate > 62.5%(8 rollouts,**只砍太易,不砍全错**;general-chat 另用 F1∈[0.1,0.8] 双边)、GLM-4.5 砍响应最短的 50% prompt 换 +2–4%、MiniMax-M1 只留 pass@10∈(0,0.9)、Seed1.5 剔除得分方差低的样本。
对我们:274 任务池若未做难度分层,这是**性价比最高的单点改动**;且 Olmo 3 的 active sampling(持续重采样直到凑满非零梯度 batch)解决了 GRPO 的 batch 缩水问题,基础设施代价不高。

**次优先(未进前 5 但值得记档):** DeepSeek-R1 的「model-based reward 只在最后 400/1700 步暴露」是控制 reward hacking 的低成本手段;MiniMax-M1 §4.2.2 证明 **generative RM 的长度偏置离线修不掉**,只有在线监控 + 触发式重标定有效;Kimi K2 的 closed-loop critic(用可验证域 on-policy rollout 持续再训 critic)与我方 CD-003「冻结 teacher + 锚」方向一致;NVIDIA MOPD 的 teacher 全部是**冻结历史 checkpoint**、reverse-KL 作 stop-gradient 优势、40–52 步短脉冲收敛 —— 与我方 E1/E2 同向,但他们**没给「不冻结会怎样」的对照**,不构成对我们机理结论的独立验证。

---

## 附:核验过的一手源索引

| 报告 | arXiv / URL | 备注 |
|---|---|---|
| Tülu 3 | https://arxiv.org/pdf/2411.15124v3 | HTML 渲染截断,须读 PDF |
| Olmo 3 | https://arxiv.org/abs/2512.13961 | 118 页,须下载 PDF |
| Qwen3 | https://arxiv.org/pdf/2505.09388v1 | **HTML 版缺 §4.4 General RL,勿用** |
| DeepSeek-R1 | https://arxiv.org/pdf/2501.12948 · Nature https://www.nature.com/articles/s41586-025-09422-z | **Nature 版 Methods+SI 信息量远大于 arXiv**;可走 Europe PMC `PMC12443585` |
| Kimi k1.5 | https://arxiv.org/html/2501.12599v1 | |
| Kimi K2 | https://arxiv.org/abs/2507.20534 | id 已核 |
| GLM-4.5 | https://arxiv.org/abs/2508.06471 | 无 HTML |
| GLM-4.5V/4.1V | https://arxiv.org/abs/2507.01006 | **唯一的单域 vs mix-all 消融矩阵** |
| MiniMax-M1 | https://arxiv.org/abs/2506.13585 | |
| MiniMax-M2 | https://arxiv.org/abs/2605.26494 | 2026-05-26,id 与标题已核 |
| Nemotron-Cascade 2 | https://arxiv.org/abs/2603.19220 | 前作 NC1 https://arxiv.org/abs/2512.13607 |
| Seed1.5-Thinking | https://arxiv.org/abs/2504.13914 | |
| Llama 4 | https://ai.meta.com/blog/llama-4-multimodal-intelligence/ | 博客,无曲线无消融 |
| GLM-4.6 | https://huggingface.co/zai-org/GLM-4.6 | **无技术报告,RL 配方未公开** |
