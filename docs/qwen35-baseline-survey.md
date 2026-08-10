# Qwen3.5 系列相关工作调研:可对照 baseline 收集

**日期**:2026-08-08 · **分支**:CD-006 · **用途**:主力实验模型迁移到 Qwen3.5 系列前,
收集**同基座 × 同 benchmark** 的可比 baseline
**检索**:arXiv(cs.AI/cs.CL/cs.LG,2026-01-01 起)三轮 + Web 检索;共筛出含 Qwen3.5 的论文
约 70 篇,下表只列**规模落在 0.8B/2B/4B/9B 且 benchmark 与我们重叠**的,以及少数必要的
大规模对照。

---

## 0. Qwen3.5 系列规模谱(供选型)

Dense:**0.8B / 2B / 4B / 9B / 27B**(Apache 2.0);MoE:**35B-A3B / 122B-A10B / 397B-A17B**。
官方公布的基线(来源见文末):

| benchmark | 2B | 4B | 9B |
|---|---|---|---|
| GSM8K | 62.2 | 86.5 | **89.5** |
| GPQA-Diamond | — | — | **81.7**(对照 GPT-OSS-120B 71.5)|

另有第三方全谱测量:GSM8K 从 **41%(0.8B)** 到 **96.8%(397B-A17B)**(2604.02315)。

---

## 1. ★★★ 最直接可比:Harness-R1(ALFWorld + WebShop + DBBench,Qwen3.5-9B)

**arXiv 2608.02276**(2026-08-03)· 目标 agent = **frozen Qwen3.5-9B**,engineer 也是 9B。

**这篇给出了按 ALFWorld family 拆分的完整对照表**,是我们目前能找到的最可比 baseline:

| Method | Pick | Look | Clean | Heat | Cool | Pick2 | **ALFWorld All** | WebShop Succ | DBBench | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen3.5-9B(default harness)** | 75.8 | 66.7 | 16.9 | 28.8 | 9.2 | 47.5 | **40.6** | 31.2 | 61.0 | 44.3 |
| ReAct | 79.8 | 54.8 | 27.3 | 27.4 | 10.3 | 53.3 | 43.4 | 37.4 | 61.7 | 47.5 |
| Self-Refine | 70.7 | 45.2 | 11.7 | 28.8 | 6.9 | 57.4 | 39.0 | 29.0 | 57.3 | 41.8 |
| Reflection | 91.9 | 90.5 | 24.7 | 56.2 | 19.5 | 73.8 | 59.2 | 43.6 | 64.7 | 55.8 |
| **Qwen3.5-397B** | 76.8 | 66.7 | 14.3 | 31.5 | 11.5 | 47.5 | **41.2** | 32.8 | 63.3 | 45.8 |
| GPT-5.5 | 80.8 | 78.6 | 31.2 | 28.8 | 17.2 | 35.2 | 43.2 | 36.6 | 64.0 | 47.9 |
| GLM-5.2 | 77.8 | 78.6 | 18.2 | 27.4 | 16.1 | 54.9 | 45.0 | 36.0 | 65.3 | 48.8 |
| DeepSeek-V4-Pro | 72.7 | 69.0 | 13.0 | 30.1 | 16.1 | 47.5 | 41.0 | 32.4 | 64.3 | 45.9 |
| Harness-R1(本文)| 77.8 | 81.0 | 58.4 | 43.8 | 34.5 | 39.3 | **53.2** | 42.2 | 65.3 | 53.6 |
| **Agent SFT** | 91.9 | 100.0 | 46.8 | 56.2 | 48.3 | 85.2 | **71.2** | 42.6 | 63.7 | 59.2 |
| Agent SFT + Harness-R1 | 93.9 | 100.0 | 81.8 | 74.0 | 72.4 | 86.1 | **84.0** | 43.0 | 65.7 | 64.2 |

**其实验设置**(便于口径对齐):ALFWorld train 1380+1280 / val 99 / **test 500**;
target temperature **0**(greedy)、max response **4096**、**thinking disabled**、tool choice auto;
RL 用 GRPO(clip 0.20/0.28、truncated IS max weight 2.0、entropy & KL 系数均 0)、rollout temp 0.7。

**三条对我们特别有价值的读数**:

1. **`Qwen3.5-397B`(41.2)几乎打不过 `Qwen3.5-9B`(40.6),只高 0.6pp**;GPT-5.5 也只有 43.2。
   → ALFWorld 上**模型规模不是瓶颈**,这对"靠机制而非靠更大基座"的路线是直接支撑。
2. **family 难度序与我们实测完全一致**:Cool(9.2)< Clean(16.9)< Heat(28.8)最难,
   Pick(75.8)/ Look(66.7)最易。我们 step1/step2 里 `pick_heat`、`pick_cool` 的 bare 常为 0%,
   同一现象。
3. **Agent SFT 单独就到 71.2**,是很强的 baseline;而我们当前是 **9B raw 直接 RL,无 SFT**。

**与我们当前读数的口径差**(不可直接比,列出以便后续对齐):我们 9B raw 的 val 是
seen 30.71% / unseen 26.87%(274 条、greedy、**纯 bare zero-shot 无 harness**);
他们 40.6% 是 500 条 test + "default harness"。差距的一部分来自 harness/few-shot,
一部分来自评测集与划分。**若要正式对照,需按其口径重跑一次**。

---

## 2. ★★ 同类机制(经验/技能自演化 → 注入 executor),ALFWorld

| 论文 | 基座 | benchmark | 关键数字 |
|---|---|---|---|
| **EmbodiSkill**(2605.10332)| **Qwen3.5-27B**(frozen executor)| ALFWorld, EmbodiedBench | ALFWorld **93.28%**;比作为直接 agent 的 GPT-5.2 高 31.58pp。training-free 的技能自演化 |
| **ACCORD**(2606.16432)| Qwen3.5-27B-FP8 | AppWorld, ALFWorld | 开源模型 +10.1;ALFWorld +7.4(GPT-5-mini)。无需训练的上下文 grounding |
| Uncertainty Decomposition(2606.19559)| Qwen3.5-35B | WebShop, ALFWorld(+澄清变体)| 五个 backbone 对照,clarification F1 提升 |

**EmbodiSkill 与我们的 researcher 文档机制最同构**(都是"从自身轨迹提炼可复用经验 →
喂给 frozen executor"),但它是 **training-free** 且用 27B;我们是**把经验作为特权信息做权重内化**。
这正是我们的差异点,也是它可以作为"不训练版上界"的参照。

---

## 3. ★★★ 数学/推理(2B / 4B / 9B)

| 论文 | 基座 | benchmark | 关键数字 |
|---|---|---|---|
| **PIPO**(2605.27255)| **Qwen3.5-4B 和 9B** | AIME 2025, GPQA-Diamond, LiveCodeBench v6, LongBench v2 | pass@4 最高 **+7.15**;**方法里用了 On-Policy Distillation(OPD)** 训练 confidence head |
| **Gradient Starvation in Binary-Reward GRPO**(2605.07689)| **Qwen3.5-9B** | GSM8K, MATH-500 | 实测 group size 4 时 **degeneracy rate 0.69**;Sign advantage `A=2r−1` 达 **73.8%** vs 标准 DrGRPO **28.4%**(+45.4pp, p<0.0001)|
| **ExComm**(2605.22102)| **Qwen3.5-4B** | AIME 2024/2025, GAIA | 比最强 baseline 平均 +5.0% |
| **SVR**(2607.28457)| **Qwen3.5-2B** | MATH | 自验证精炼 + 联合 verdict-confidence RL |
| InsightReplay(2605.14457)| Qwen3.5(8B/30B 档)| AIME, HMMT, GPQA-Diamond, LiveCodeBench v5 | 24 个设置全部提升,均值 +1.65 |
| Cast a Wider Net(2605.27000)| **Qwen3.5-9B** | LiveCodeBench, MATH | Pass@K 协同策略优化 |
| Lean4 Autoformalization(2603.24372)| **Qwen3.5-2B** | MATH | cycle consistency 微调 |
| Domyn-Small(2607.20448)| 以 **Qwen3.5-9B** 作 peer baseline | GPQA-Diamond, IFEval, MATH | 其 10B 模型 GPQA-Diamond 50.0;**token 用量仅为 Qwen3.5-9B 的 1/3** |

**`Gradient Starvation` 那篇对我们有直接方法论价值**:它在 **Qwen3.5-9B + GSM8K** 上实测了
GRPO 组内全对/全错导致 advantage 恒零的退化率(group size 4 时 0.69)。
我们用 `pgs_asym_grpo`(bare 组内 LOO),同样面临这个结构;其 Sign advantage 是一个现成对照臂。

---

## 4. ★★ Qwen3.5 的 hybrid 架构特性(GatedDeltaNet + attention)

这几篇直接研究我们正在用的架构,对训练配置有参考价值:

| 论文 | 基座 | 发现 |
|---|---|---|
| **S0 Tuning**(2604.01168)| **Qwen3.5-4B**(GatedDeltaNet hybrid)| 只调每层 recurrent 的初始 state 矩阵、冻结全部权重:HumanEval greedy pass@1 **+23.6pp**;跨域迁移 MATH-500 +4.8pp、GSM8K +2.8pp。**零推理开销 PEFT** |
| **Where Should LoRA Go?**(2604.22127)| **Qwen3.5-0.8B**(sequential hybrid)| **attention 通路**加 LoRA 用 5–10× 更少参数就胜过全模型适配;**改 recurrent 主干在 sequential hybrid 上是破坏性的(GSM8K −14.8pp)**,在 parallel hybrid 上却是建设性的(+8.6pp)|
| Component Ablation(2603.22473)| Qwen3.5-0.8B | hybrid 组件消融 |

**第二篇的结论值得留意**:Qwen3.5 是 sequential hybrid,若日后做 LoRA/部分微调,
**动 recurrent 主干可能是负收益**。我们现在是全参数 FSDP 训练,暂不受影响,但选型时要记着。

---

## 5. ★ 其它 agentic(规模匹配但 benchmark 不重叠,备选)

| 论文 | 基座 | benchmark |
|---|---|---|
| Multi-Turn RL for Tool-Calling(2604.02869)| Qwen3.5-4B | **tau-bench** |
| ABSeeker(2608.05102)| Qwen3.5-4B | BrowseComp(长程搜索)|
| Speculate While You Reason(2607.25816)| Qwen3.5-4B | 工具调用预测,joint agent-speculator RL |
| Cooperative Coevolution(2608.02391)| Qwen3.5-4B | MATH,资源受限 agentic 后训练 |
| RLVR for Tool-Use Agents(2607.01465)| Qwen3.5-4B | Atlassian 工作流 |
| WinDOM(2606.25964)| **Qwen3.5-2B** | OSWorld,**self-family distillation**(自蒸馏,与我们同源思路)|
| StructAgent(2607.11388)| Qwen3.5-9B/27B | OSWorld |
| SearchMaster(2608.01822)| Qwen3.5-9B | 搜索 agent 自博弈 |

---

## 6. 选型与对照建议(供裁定,非结论)

1. **9B 是当前 baseline 最密集的档**:Harness-R1、Gradient Starvation、PIPO、Cast a Wider Net、
   Domyn-Small 都用它,且 Harness-R1 给了 ALFWorld family 级完整表。**若要"同基座同 benchmark"
   的强可比性,9B 优先**。
2. **4B 次之**(PIPO 双档、ExComm、tau-bench、ABSeeker),**2B 最稀疏**(SVR、WinDOM、Lean4),
   数学/推理有零星工作,agentic 长程几乎没有 —— 若做 2B 需自建 baseline。
3. **口径对齐是使用这些数字的前提**:Harness-R1 用 ALFWorld **test 500 条 / greedy /
   thinking disabled / max response 4096**;我们目前是 **274 条(seen140+unseen134)/ greedy /
   thinking disabled / response 32768**。要引用它作对照,得先按其口径补一次评测。
4. **可直接采纳的对照臂**:ReAct、Self-Refine、Reflection(prompt-based)、Agent SFT
   ——Harness-R1 已在同基座同环境下跑好,我们复现同口径后即可并列。

---

## 来源

- [Harness-R1 (arXiv 2608.02276)](https://arxiv.org/abs/2608.02276)
- [EmbodiSkill (arXiv 2605.10332)](https://arxiv.org/abs/2605.10332)
- [ACCORD (arXiv 2606.16432)](https://arxiv.org/abs/2606.16432)
- [PIPO (arXiv 2605.27255)](https://arxiv.org/abs/2605.27255)
- [Gradient Starvation in Binary-Reward GRPO (arXiv 2605.07689)](https://arxiv.org/abs/2605.07689)
- [ExComm (arXiv 2605.22102)](https://arxiv.org/abs/2605.22102)
- [SVR (arXiv 2607.28457)](https://arxiv.org/abs/2607.28457)
- [InsightReplay (arXiv 2605.14457)](https://arxiv.org/abs/2605.14457)
- [S0 Tuning (arXiv 2604.01168)](https://arxiv.org/abs/2604.01168)
- [Where Should LoRA Go? (arXiv 2604.22127)](https://arxiv.org/abs/2604.22127)
- [Beyond the Assistant Turn (arXiv 2604.02315)](https://arxiv.org/abs/2604.02315)
- [Domyn-Small (arXiv 2607.20448)](https://arxiv.org/abs/2607.20448)
- [Multi-Turn RL for Tool-Calling Agents (arXiv 2604.02869)](https://arxiv.org/abs/2604.02869)
- [WinDOM (arXiv 2606.25964)](https://arxiv.org/abs/2606.25964)
- [Qwen 3.5 Benchmarks Guide](https://www.digitalapplied.com/blog/qwen-3-5-agentic-ai-benchmarks-guide)
- [Qwen3.5 family overview (StableLearn)](https://stable-learn.com/en/qwen35-native-multimodal-agent-model/)
- [Alibaba Qwen 3.5 Agentic AI Benchmark 2026](https://www.buildmvpfast.com/blog/alibaba-qwen-3-5-agentic-ai-benchmark-2026)

---

# 第二部分:Agentic benchmark 选型(2026-08-08 追加)

**目的**:为 Qwen3.5 系列(尤其 4B/9B)寻找**更难、更高质量、且已有可比 baseline** 的
agentic benchmark。**方法**:arXiv 五轮检索聚合去重 **87 篇**含 Qwen3.5 的论文,
统计各 benchmark 的出现频次,并区分"该 benchmark 在 4B/9B 档是否真有 baseline"。

## 1. 频次统计与一个反直觉结论

| benchmark | 出现篇数 | **其中含 4B/9B 的** | 说明 |
|---|---|---|---|
| SWE-bench | 9 | **1** | 热门,但小模型档几乎空白 |
| Terminal-Bench | 7 | **0** | **完全没有 4B/9B baseline** |
| BrowseComp | 4 | 2 | deep search,9B 有 |
| ALFWorld | 3 | 1 | Harness-R1(9B,family 级完整表)|
| OpenClaw / Claw-Eval | 3+2 | 3 | 个人助理,4B/9B 都有 |
| OSWorld | 2 | 1 | **StructAgent 有 9B 明确数字** |
| GAIA | 2 | 2 | 4B/9B 都有 |
| Online-Mind2Web / WebVoyager | 2+2 | (Fara-1.5 覆盖 4B/9B/27B)| |
| tau-bench | 1 | 1 | 4B |
| WebShop / DBBench | 1 / 1 | 1 / 1 | 与 ALFWorld 同篇(Harness-R1)|
| AppWorld | 1 | 0 | 只有 27B-FP8 |
| EmbodiedBench | 1 | 0 | 只有 27B |

**反直觉结论**:**Terminal-Bench 与 SWE-bench 是当下最热的 agentic benchmark,
但在 4B/9B 档 baseline 近乎为零** —— 相关工作全部集中在 27B / 35B-A3B / 122B-A10B / 397B-A17B。
这两个 benchmark 对 9B 以下模型难度过高,**引入它们拿不到可比 baseline**,不建议作为主战场。

(统计口径说明:仅统计**摘要中出现**的 benchmark 名,论文正文里的补充实验不计,
故为下界;且检索 query 本身对某些名字有偏好。)

## 2. 在 4B/9B 档真有 baseline 的 benchmark(按可用度排序)

### ★★★ OSWorld-Verified —— 最推荐的"更难"选项

**StructAgent**(2607.11388):`Qwen3.5-9B` **27.0% → 46.9%**;`Qwen3.5-27B` **31.6% → 62.2%**;
开源 SoTA 78.9%(MiniMax-M3)。同框架还泛化到 Minecraft。

- **难度合适**:9B 起点仅 27.0%,远低于 ALFWorld 的 40.6%,提升空间大且不天花板。
- **长程属性强**:真实桌面操作,累积观测、中间编辑、失败重试、部分完成 —— 与我们
  "特权文档指导长程执行"的假设高度契合。
- ⚠ **需先确认接口形态**:StructAgent 说同时支持 LLM 与 VLM backbone,`Qwen3.5-9B`(纯文本)
  能上说明有 a11y-tree / 文本化观测通道。**引入前必须验证我们的纯文本 policy 能接进去**。

### ★★★ BrowseComp 系 —— deep search 长程

**SearchMaster**(2608.01822):`Qwen3.5-9B` 六个 deep-search benchmark 平均
**38.19% → 51.52%**,BrowseComp-Plus 单项 **+30.1pp**。另有 **ABSeeker**(2608.05102,4B)、
SearchArt(27B,BrowseComp 70.06 / BrowseComp-ZH 74.39)、UNIBROWSE(35B-A3B)。

- 9B 起点 38.19%,难度合适;**多篇工作、多个规模**,baseline 密度好。
- 需要联网检索环境,基础设施成本比 ALFWorld 高。

### ★★★ Online-Mind2Web / WebVoyager —— 有三档同基座 SoTA

**Fara-1.5**(2606.20785):在 **Qwen3.5 的 4B / 9B / 27B 三档**上训 computer-use agent,
**每档都刷到 size class 的 SoTA**,权重 MIT 开源:

| | Online-Mind2Web | WebVoyager |
|---|---|---|
| Fara1.5-9B | **63.4%** | **86.6%** |
| Fara1.5-27B | **72.3%** | — |

**Orchard-GUI**(2605.15040)另有 4B 档:WebVoyager/Online-Mind2Web/DeepShop 平均 68.4%。

- **唯一一个 4B 和 9B 都有明确同基座 SoTA 的 benchmark 组**,横向对比最干净。
- WebVoyager 86.6% 已接近饱和,**Online-Mind2Web(63.4%)更适合作主战场**。

### ★★ ALFWorld + WebShop + DBBench(当前主战场)

**Harness-R1**(2608.02276)一篇内给全三个,`Qwen3.5-9B` 分别 **40.6 / 31.2 / 61.0**,
且 ALFWorld 有 family 级拆分 + ReAct / Self-Refine / Reflection / Agent SFT 全套对照臂。

- **口径最完整、最省事**:一篇论文即可获得四条 baseline。
- 建议:**WebShop 和 DBBench 值得顺带引入** —— 它们与 ALFWorld 出自同一篇、同一基座、
  同一 harness 设置,加进来几乎零额外对照成本,却把结论从单环境扩到三环境。

### ★★ tau-bench / GAIA(4B 档)

- **tau-bench**:Multi-Turn RL for Tool-Calling(2604.02869,Qwen3.5-4B)。多轮工具调用 +
  用户模拟,与我们的多轮协议同构。
- **GAIA**:ExComm(2605.22102,Qwen3.5-4B)。

## 3. 与我们已集成资产的关系

据 CD-010 已完成的集成:**ALFWorld**(主)、**AppWorld**、**SpreadsheetBench verified-400**、
**Math(DAPO 六源)**。结合本次调研:

- **AppWorld** 的 Qwen3.5 baseline 只有 ACCORD 的 **27B-FP8**,4B/9B 档**没有**
  → 我们跑出来的数会是该档首批,有新颖性但**缺可比对照**。
- **SpreadsheetBench** 在本次 87 篇里**一次都没出现** → 同上,零 Qwen3.5 baseline。
- 若目标是"有 baseline 可比",**优先补 WebShop + DBBench(零成本蹭 Harness-R1 的表)**,
  其次 **OSWorld-Verified**(难度上限)或 **Online-Mind2Web**(4B/9B 双档 SoTA)。

## 4. 建议的引入优先级(供裁定)

| 优先级 | benchmark | 理由 | 主要成本 |
|---|---|---|---|
| P0 | **WebShop + DBBench** | 与现有 ALFWorld 同篇同基座同 harness,**四条对照臂现成** | 环境接入,低 |
| P1 | **OSWorld-Verified** | 9B 起点仅 27.0%,难度与长程属性最匹配;StructAgent 完整对照 | 需验纯文本接口;桌面沙箱,中高 |
| P1 | **Online-Mind2Web** | **4B/9B 双档同基座 SoTA**(Fara-1.5),横向最干净 | 需实时网页环境,中高 |
| P2 | **BrowseComp(-Plus)** | 9B 38.19%,多篇多档 baseline | 需检索环境,中 |
| P2 | tau-bench | 4B 档多轮工具调用,协议同构 | 低 |
| — | Terminal-Bench / SWE-bench | **4B/9B 档无 baseline**,不建议 | — |

## 补充来源

- [Fara-1.5 (arXiv 2606.20785)](https://arxiv.org/abs/2606.20785)
- [StructAgent (arXiv 2607.11388)](https://arxiv.org/abs/2607.11388)
- [SearchMaster (arXiv 2608.01822)](https://arxiv.org/abs/2608.01822)
- [Orchard (arXiv 2605.15040)](https://arxiv.org/abs/2605.15040)
- [ABSeeker (arXiv 2608.05102)](https://arxiv.org/abs/2608.05102)
- [SearchArt (arXiv 2607.24850)](https://arxiv.org/abs/2607.24850)
- [UNIBROWSE (arXiv 2607.10557)](https://arxiv.org/abs/2607.10557)
- [PaperPilot (arXiv 2607.00597)](https://arxiv.org/abs/2607.00597)
- [Nanbeige4.2-3B (arXiv 2607.22083)](https://arxiv.org/abs/2607.22083)
- [When Top-K Misses the Decision (arXiv 2607.07050)](https://arxiv.org/abs/2607.07050)
- [CoPES (arXiv 2608.02391)](https://arxiv.org/abs/2608.02391)

---

# 第三部分:SWE-bench 实况与"可训练"agentic 环境(2026-08-08 追加)

## 0. 更正上一部分的一个判断

第二部分说"SWE-bench 在 4B/9B 档没有 baseline"——**该判断不成立**,是检索 query 偏差所致
(第二轮 query 里 benchmark 名与规模词没有同时出现)。补检后小模型档工作很多。

## 1. SWE-bench 实况

### 1.1 小模型档的 baseline(补检结果)

| 论文 | 模型 | SWE-bench Verified | 备注 |
|---|---|---|---|
| **Dockerless**(2606.28436)| 超 **Qwen3.5-9B baseline** +2.4 → 62.0% | **9B baseline ≈ 59.6%** | 另 Multilingual 50.0(+8.7)、Pro 35.2(+2.9)|
| **SWE-MeM**(2606.28434)| 4B / 30B | **43.4%** / 60.2% | memory-aware GRPO |
| **DAgger**(2605.12913)| 4B / 8B | **27.3%** / **29.8%** | 4B 超过若干已发表 8B 系统 |
| FuseSearch(2601.19568)| 4B | 84.7% file-level F1 | 代码定位子任务 |
| Orchard-SWE(2605.15040)| Qwen3.5-35B-A3B | 69.7 / 73.0 | 开源 SoTA |
| OpenSWE(2603.13023)| Qwen2.5-32B / 72B | 62.4 / 66.0 | |
| Open-SWE-Traces(2606.16038)| Qwen3-30B-A3B | 61.7 | 数据集贡献 |

注意 **9B 的 SWE-bench Verified 基线约 59.6%,远高于其 ALFWorld 的 40.6%** ——
说明 SWE-bench Verified 对 9B 不算"更难",反而是 ALFWorld 那类具身长程更难。

### 1.2 训练成本:有直接的公开数字

**这是 coding 任务最真实的成本画像**:

- **OpenSWE / daVinci-Env**(2603.13023)公开了总账:
  **环境构建 $891K + 轨迹采样与筛选 $576K = 约 $1.47M**,
  换来 **45,320 个可执行 Docker 环境**(12.8k repos)与约 **13,000 条 curated 轨迹**。
  其 pipeline 部署在 **64 节点分布式集群**上。
- **Dockerless**(2606.28436)整篇的动机就是成本:「execution-based verification requires
  running unit tests inside per-repository environments such as Docker images,
  **incurring substantial environment setup costs**」——它做的是完全绕开 Docker 的验证器。
- **Rollout Pass-Rate Control**(2605.05112):「Agentic RL for software engineering
  **spends much of its compute on stateful trajectories**」,其方法带来
  **2.01× / 1.55× 端到端墙钟加速**(Qwen3-14B / 32B)——需要专门优化才跑得动。

**结论**:SWE-bench 是**训练数据极其充足、但单位环境成本高**的方向。成本主要不在 GPU,
而在**每个 task 要起 Docker 容器 + 跑单元测试**的环境侧。对我们的具体含义:
当前 ALFWorld 单步 18.1 分钟(16 任务 × 16 rollout);同规模换成 SWE-bench,
每条 rollout 都要容器内执行测试,墙钟会显著劣化,且需要额外的容器编排基础设施。

---

## 2. ★★★ "可训练"的 agentic 环境套件(正面回应"要有训练数据")

这些不是"只能评测的 benchmark",而是**自带大规模训练环境或可程序化生成**的资源:

| 资源 | 规模 | 训练数据形态 | 纯文本? | 备注 |
|---|---|---|---|---|
| **RLVE-Gym**(2511.07317)| **400 个可验证环境** | **程序化生成问题,难度随策略自适应** —— 训练数据本质无限 | ✓ | 环境数量 scaling 本身带来泛化;1.5B 上六个推理 benchmark +3.37%(对照:继续原 RL 用 3× 算力只 +0.49%)|
| **C-World**(2601.06328)| **5,571 个格式统一工具 / 204 个应用** | 任务分布引擎合成长程工作流;含失败注入与扰动 | ✓ | **仅 1,170 条 C-World 轨迹微调即超过用 119k 样本训练的 baseline**;World Engine 与真实执行的排序相关 ρ=0.883 |
| **Auto-ClawEval / ClawEnvKit**(2604.18543)| **1,040 个环境 / 24 类** | 自然语言描述 → 自动生成可验证环境;**同时是按需训练环境生成器** | ✓ | 比人工构建**低 13,800× 成本**;harness engineering 可带来最多 +15.7pp;**无模型饱和** |
| **RACES**(2606.12373)| 300 个基础环境 + 递归组合 | 组合算子(SEQUENTIAL/PARALLEL/SORT/SELECT)自动生成新环境 | ✓ | 用 50 个基础环境即达到 300 个的效果 |
| **LiteCoder-Terminal**(2605.29559)| **602 个 RL 可验证环境** + **11,255 条 SFT 轨迹**,10 个域 | 零依赖合成,直接从域规范生成 | ✓ | 32B 变体 Terminal-Bench 1.0/2.0/Pro 分别 29.06/18.54/34.00 |
| **SWE-Universe**(2602.02361)| **807,693 个**真实 SWE 环境 | 从 GitHub PR 自动构建 | ✓ | 百万级;Qwen3-Max-Thinking 用它到 SWE-bench Verified 75.3% |
| **OpenSWE**(2603.13023)| **45,320 个** Docker 环境 | 13,000 条 curated 轨迹 | ✓ | **全开源**(Dockerfile/评测脚本/基础设施);成本已公开 |
| Agent World Model(2602.10090)| 1,000 个合成环境 | 合成 | ✓ | |

### 与我们需求的匹配度

我们的硬约束:**纯文本 policy(非 VLM)、多轮长程、可验证 reward、能切出训练集**。

- **RLVE-Gym / RACES**:程序化生成 ⇒ **训练数据无上限**,且 RLVE 的"难度自适应"机制
  正好对治我们观察到的问题(step2 里 `pick_cool` bare 恒 0%,组内全错 ⇒ advantage 恒零)。
  ⚠ 但它们偏"可验证推理题",**长程 agentic 属性弱于 ALFWorld**。
- **C-World / Auto-ClawEval**:**最贴合"长程多工具 agentic + 有训练数据"**。
  C-World 的 5,571 工具 / 204 应用规模远超 AppWorld;Auto-ClawEval 1,040 环境且
  **无模型饱和**(说明难度足够)。两者都是**生成器**,想要多少任务就有多少。
  Claw-Eval 侧还有 Qwen3.5 baseline(Orchard-Claw 35B 59.6% pass@3;Nanbeige4.2-3B 对照 9B)。
- **LiteCoder-Terminal**:602 个 RL 环境 + 11K SFT 轨迹,是 Terminal 方向里**唯一
  明确给出 RL 环境数量**的;但 Terminal-Bench 在 4B/9B 档仍缺 baseline。
- **SWE-Universe / OpenSWE**:数据量碾压,但如上文,环境成本是主要负担。

---

## 3. 关于"测试集大到可切分"的现实数据

已知的评测集规模(用于判断能否切训练集):

| benchmark | 训练 | 验证 | 测试 | 来源 |
|---|---|---|---|---|
| **WebShop** | **10,480** | 100 | 500 | Harness-R1 表 10.1 |
| **DBBench** | **4,703** | 100 | 300 | 同上 |
| **ALFWorld** | **2,660** | 99 | 500 | 同上(我们现用 274 条 val)|
| OSWorld-Verified | — | — | 约 369 | 公开信息;**偏小,切分后训练量不足** |
| GAIA | — | — | 约 466 | 同上 |
| Auto-ClawEval | **按需生成** | | 1,040 | ClawEnvKit |
| C-World | **按需生成** | | — | 任务分布引擎 |
| RLVE-Gym | **程序化无限** | | — | 400 环境 |

**这条直接回答你的约束**:
- **WebShop(10,480 训练任务)是现成资源里训练量最大的**,且与 ALFWorld 同篇有完整 baseline
  —— 兼顾"有训练数据"与"有可比 baseline"两个要求。
- **DBBench(4,703)次之**,同样来自 Harness-R1 那一篇。
- **OSWorld / GAIA 虽然难,但总量仅数百条,切分后训练量不足**,不适合作 RL 主战场,
  更适合作 held-out 评测。
- 若要"训练数据无上限",**只能走生成器路线**(C-World / Auto-ClawEval / RLVE-Gym),
  代价是这些环境的 Qwen3.5 baseline 稀疏(Claw 系除外)。

---

## 4. 修订后的引入建议(供裁定)

| 优先级 | 目标 | 理由 | 主要成本 |
|---|---|---|---|
| **P0** | **WebShop + DBBench** | **训练集 10,480 / 4,703**,与 ALFWorld 同篇同基座同 harness,四条对照臂现成 | 环境接入,低 |
| **P1** | **C-World** 或 **Auto-ClawEval** | 长程多工具、**训练任务可按需生成**、难度未饱和 | 需接生成器与验证器,中 |
| **P2** | **RLVE-Gym** | 训练数据无限 + **难度自适应**(对治组内全错的 advantage 恒零)| 偏推理题,长程属性弱 |
| P3 | OSWorld-Verified / GAIA | 难度高,但**总量数百条不够训练** | 建议只作 held-out 评测 |
| — | SWE-bench 系 | 数据极充足,但**每条 rollout 要起容器跑测试**;且 9B 基线已 59.6% 不算难 | 环境侧成本高 |

## 补充来源(第三部分)

- [Dockerless (arXiv 2606.28436)](https://arxiv.org/abs/2606.28436)
- [SWE-MeM (arXiv 2606.28434)](https://arxiv.org/abs/2606.28434)
- [DAgger for LLM-Agents (arXiv 2605.12913)](https://arxiv.org/abs/2605.12913)
- [Rollout Pass-Rate Control (arXiv 2605.05112)](https://arxiv.org/abs/2605.05112)
- [OpenSWE / daVinci-Env (arXiv 2603.13023)](https://arxiv.org/abs/2603.13023)
- [SWE-Universe (arXiv 2602.02361)](https://arxiv.org/abs/2602.02361)
- [Open-SWE-Traces (arXiv 2606.16038)](https://arxiv.org/abs/2606.16038)
- [RLVE (arXiv 2511.07317)](https://arxiv.org/abs/2511.07317)
- [RACES (arXiv 2606.12373)](https://arxiv.org/abs/2606.12373)
- [C-World (arXiv 2601.06328)](https://arxiv.org/abs/2601.06328)
- [ClawEnvKit / Auto-ClawEval (arXiv 2604.18543)](https://arxiv.org/abs/2604.18543)
- [LiteCoder-Terminal (arXiv 2605.29559)](https://arxiv.org/abs/2605.29559)
- [TermiGen (arXiv 2602.07274)](https://arxiv.org/abs/2602.07274)
- [FuseSearch (arXiv 2601.19568)](https://arxiv.org/abs/2601.19568)
