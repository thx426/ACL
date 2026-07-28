# 多任务 RL 配方 / 课程 / 异质奖励混合 —— 七篇精读

读的是正文方法节 + 实验节 + 附录超参与消融,不是 abstract。所有结论标注节/式/表编号;论文没报告的项一律写"未报告",不补。

PDF 位置:`.conductor/branches/CD-002/papers-mtl/`(5 篇本地);DUMP 与 ToolSample 从 arXiv HTML 抓取,文本存
`/private/tmp/claude-501/.../scratchpad/txt/2504.09710-DUMP.txt`、`.../2509.14718-ToolSample.txt`。

---

## 结论先行

1. **"混批就有协同增益"在这批文献里没有一篇支持,反而有四条独立的反证。** Klear-AgentForge 明说多任务联合 RL "does not produce a synergistic performance boost";LUFFY 表 4 里朴素混合策略(44.4)**低于**纯 on-policy(45.5);BRIDGE 表 1/2 里固定 λ 同批 SFT+RL 在三个 backbone 上**全部低于**纯 RL;MAHALO 表 2 里把冲突偏好 pool 到单 head 会崩到远低于 base。混批的正确心态是"代价可控的工程简化",不是"能白拿收益"。
2. **对我们最可落地的一条,来自最不起眼的地方:用 pass-rate 窗口筛任务、并用 advantage 乘系数而不是改 dataloader 来实现。** Klear、Embodied-R1.5、ToolSample 三篇独立收敛到同一配方,且 ToolSample 的实现只需在 advantage 计算后加一行(Algorithm 1 第 8 行 $\hat{A}_b \leftarrow \hat{A}_b * Ratio$)。
3. **DUMP 的 UCB 不该用在"两个角色的配比"上(K=2 且 advantage 尺度不可比),但该用在"3553 个任务里哪些值得花 research 预算"上** —— 这一维度的信号同尺度、arm 数合适,详见判决 (a)。
4. **MAHALO 与团队预期有落差,必须说明**:它的主体是 DPO + PRM,不是 RL 里混二值/连续 reward。真正可移植的只有式(6)的 per-objective batch-size 归一化,以及一条负面警示。
5. **最强的单条警示**:ToolSample 表 3 —— 不 warmup 直接开动态采样,BFCL 从 56.96 掉到 47.34,几乎退回未训练模型(47.68)。任何动态采样都必须等 reward 稳定后再开。

---

## 一、逐篇

### 1. Klear-AgentForge(2511.05951)—— 多任务 agentic RL 的工业配方,结论偏负面

**RL 目标(式 5–6)**:GRPO 去掉 KL 项;DAPO 非对称 clip($\varepsilon_{low}/\varepsilon_{high}$,数值未报告);truncated importance sampling 修正 $\pi_{inference}$ 与 $\pi_{training}$ 的分布差;对触碰 max context / step 数 / timeout 的轨迹做 oversampling + masking。Advantage 是**除 std 的** $\hat A_{i,t} = (R_i - \text{mean}(\{R_i\}_{i=1}^G))/\text{std}(\{R_i\}_{i=1}^G)$(式 6)。

**turn-level 与 outcome reward 混用(§2.3.2)**:端到端多轮 RL 的 $R_i$ 是二值(式 7,format 正确 **且** 任务完成才给 1);因为稀疏,对"正确工具调用序列已知"的一部分 prompt 额外做 step-wise RL,turn-level 二值 reward(式 8,第 $T$ 次工具调用格式正确且与 ground truth 完全匹配才给 1)。**两种 reward 都是二值,不是连续/二值混合** —— 这一点与我们的关切不同,别误读。

**任务配比与调度(§3.4.3)**:多任务 RL 用 **step-level 固定交替**,4 步一周期 = 3 步 tool-use + 1 步 SWE,维持全量数据的 3:1 比例。作者明确写"Rather than mixing data randomly, we apply a step-level schedule"。**没有任何动态调度。**

**难度筛选(§3.4.1)—— 全文最可移植的部分**,三个域各有一条硬规则:
- tool-use Stage 2:按 SFT 模型实测成功率保留 $0 < p < 1$;
- code contest:8 次采样后只保留 $0.25 \le p \le 0.75$;
- SWE:只保留"至少被强 teacher 解出过一次"的样本。

**效果(§3.4.3)**:联合 RL 得 BFCL v3 **70.8** / SWE-bench Verified **39.6**;单任务分别 RL 后模型合并(式 9–11,Select–Calculate–Erase)得 SWE **39.4**(合并前 40.4,掉 1.0)、Aider-Polyglot **33.8**(合并前 35.1,掉 1.3)。作者结论原文:joint RL "can nearly match the gains of separate training, **it does not produce a synergistic performance boost where the tasks mutually enhance each other**";而 merge 也掉分,所以最终仍认为多任务 RL 是必要的。主表 Klear-AgentForge-8B:BFCL v3 71.5 / τ-bench Retail 56.7 / Airline 41.5(表 2),SWE 39.4 / Aider 33.8(表 3)。

**其他**:disaggregated 架构(4 训练节点 + 4 rollout 节点)相比 co-located 提效约 **32%**(§3.4.2,8 节点 64×H800)。§3.3.3 一条负面结果值得记:在 reasoning 模型(Deepseek-distill-qwen3-8B)上做 agentic SFT,所有 benchmark **几乎全降到 0**,原因是工具调用前 thinking 过长耗尽 64k 输出预算 —— "reasoning 与 agentic 能力不直接互相增强"。

**未报告**:RL 的 lr / batch size / group size $G$ / 训练步数 / clip 具体数值;各任务在 batch 中的 token 级占比。

---

### 2. Embodied-R1.5(2606.11324)—— 归一化配方清楚,课程是空白

**Advantage 归一化(§4.2.1 第(3)条,行内式未编号,PDF p.8)**:
$$\hat A_i = \frac{R_i - \mu_{\text{group}}}{\sigma_{\text{batch}} + \epsilon}$$
**去均值按组、除 std 按整个混合 batch**(跨任务、跨 group)。不按任务分别归一化,不需要任务标签、不需要历史状态。动机(§4.2.1 开头):intra-task 上 group-std 归一化偏向低方差样本;inter-task 上各能力奖励尺度差异导致高密度任务主导梯度。消融显示这优于 EMA-GRPO 的 per-task 移动平均(低资源任务统计不稳)。

**异质 reward 的处理(§4.2.1(4) + §4.2.2)**:策略是**先把所有 reward 构造进 [0,1],再靠 $\sigma_{\text{batch}}$ 统一梯度幅度**,没有 per-task 权重、没有额外 reward clip。统一分段线性衰减 $\varphi(d;\tau_p,\tau_z)=\text{clip}\!\left(\frac{\tau_z-d}{\tau_z-\tau_p},0,1\right)$(式 1)把连续误差映成 [0,1] 的 partial credit;五族 reward 混用:exact-match 二值、IoU、point-distance $\varphi(d_{nn};40,150)$、trajectory-RMSE($0.5R_{2D}+0.5R_{depth}$)、Skywork RM 语义分。Format reward 与 accuracy 按 $R=(1-\lambda)R_{acc}+\lambda R_{fmt}$、$\lambda=0.1$ 加权(式 2)。**二值与连续确实同批,且不做 reward 侧加权** —— 这是这篇对我们最直接的一条。

**零方差组(§4.2.1 第(2)条)**:判据是 "all rollouts in a group receive **identical rewards**"(比"全对/全错"更宽,连续 reward 恰好相等也算)→ **mask 掉该 group**,不重采样、不补采样。过滤比例未报告。

**任务采样**:**没有动态调度、没有课程。**唯一手段是训练前一次性用 SFT checkpoint 的 pass rate 筛出 ~200K 中等难度样本,静态、不随 policy 重算。各任务在 batch 中的占比未报告。

**消融(§7.5 Q3,图 10,三维度平均分 %)**:vanilla GRPO 70.4/64.8/68.9;EMA-GRPO 69.4/62.6/71.0;w/o Difficulty Filter 68.9/63.3/70.6;Full Recipe 70.2/65.3/72.8。三维均值 68.03 / 67.67 / 67.60 / **69.43**,**全配方比 vanilla GRPO 仅 +1.4pp**,且在 Cognition 维度上**输给** vanilla(70.2 vs 70.4)。增益几乎全部来自 Pointing(+3.9pp)。没有干净的单因子消融 —— global-batch-norm 单独贡献多少论文未隔离,也**没有**"uniform 采样 + 全局归一化"这一严格对照组。

**配置**:Qwen3-VL-8B-Instruct 全参;RFT 用改造版 EasyR1;clip 0.2/0.28;KL-in-reward + K1 无偏估计,$\beta=0.01$;$n=8$ rollout;lr $3\times10^{-6}$;grad clip 1.0;2 epochs;~200K 数据。RL 的 batch size / 总步数 / GPU 数未报告(无独立超参附录)。

---

### 3. MAHALO(2510.01167)—— 与预期有落差:主体是 DPO + PRM,不是 RL 混 reward

**必须先说清楚**:团队 brief 期待这篇给"可验证与不可验证奖励**同批 RL 优化**的精确机制"。实际上 MAHALO 的路线是:所有异质信号先**统一训成 PRM 的连续分**,再由 PRM 构造偏好对,最后用 **DPO** 训练。RL 里混二值/连续 reward 这件事,论文没做。

**PRM 统一(§4)**:可验证域用 rollout + hindsight relabeling(式 1):$\tilde r_t = r_t + \gamma^{n-t} z$、$V_t^{target}=\frac1M\sum_{m}\tilde r_t^{(m)}$、$L_{PRM}=\mathbb E[(p_t - V_t^{target})^2]$,$z\in\{0,1\}$ 是终局二值正确性。不可验证域分三种情形:Case A(流程清晰、rollout 便宜)用 $M$ 次 rollout 的多数投票(式 2);Case B(rollout 昂贵,如多轮对话)直接让 LLM-as-Judge 打前缀 $r_t=J(y_{1:t})$;Case C(流程无结构)用 Bradley-Terry reward model 打部分回复。

**MAH-DPO(§5.1)**:共享 backbone $\theta_b$ + $H$ 个独立线性头 $W_i \in \mathbb R^{d\times|V|}$(从 SFT head 初始化 + 小随机扰动)。每个目标的偏好数据 $D_i$ 路由到对应头算 DPO loss $L_i$(式 5)。**总 loss(式 6)是全文最可移植的一条**:
$$L_{\text{MAH-DPO}}(\theta_b,\{W_i\}) = \sum_{i=1}^{H}\alpha_i \cdot \frac{1}{|B_i|}\sum_{B_i} L_i(\theta_b, W_i),\quad \sum_i\alpha_i=1$$
mini-batch 按目标划分 $B=\bigsqcup_i B_i$,**每个目标先按自己的子批大小 $|B_i|$ 归一化,再用 $\alpha_i$ 加权** —— 而不是把所有样本 loss 除以总数。推理时按 $\pi_{\text{MAH}}=\text{Softmax}(\sum_i w_i z_i)$ 加权 logits(式 4)。

**配比是固定的**:§6 Implementation Details 明确 "we use **equal objective weights and balanced sampling** so no objective dominates training"。无动态调度。

**效果(表 1)**:Math 域(Acc / Eng):Base 0.7107/0.5007;SFT 0.7300/0.5920;Single-Head DPO 0.7253/0.7160;MODPO 0.7280/0.7367;DPO Soup 0.7260/0.7353;**MAH-DPO Ensemble 0.7247/0.8733**(Eng 比最好基线 +13.7pp,Acc 基本持平)。Human Values 三维 Ensemble 0.6389/0.3687/0.2478,全面超过 MODPO(0.6175/0.3477/0.2325)。

**表 2 是对我们最有价值的负面证据**:在 helpfulness 与 honesty 偏好**明确冲突**的子集上,Single-Head DPO(把所有维度数据 pool 到一个 head)崩到 Help **0.3397** / Honest **0.0734**,远低于 Base 的 0.5800/0.3042;MAH-DPO Help Head 0.6201/0.3503。**共享参数 + 异质信号 + 目标冲突 = 灾难性崩塌**,而分头是有效的隔离手段。

**Finding 5(图 4)**:跨 7 个维度训一个 unified PRM,能全面超过 base,但在任何单轴上都不如专用 PRM,始终位于 base 与专用之间。**Finding 7 值得记**:高可验证、确定性的 reward(如 Math accuracy)靠 **test-time PRM 引导**获益大得多(表 4,Acc 0.6853→0.7993),训练时对齐只有小幅收益;低可验证/主观的 reward 则主要靠训练时多头获益。开销(表 7):双头 ensemble 相比单头 DPO 延迟 +12.96%、显存 +7.14%、吞吐 -0.45%。

**未报告**:详细超参在 Appendix B(本次未展开读);RL 相关的一切(因为不做 RL)。

---

### 4. LUFFY(2504.14945)—— off-policy 混批的梯度病与修法

**目标(式 5)**:off-policy 轨迹**直接塞进 on-policy 的 group** 一起算 mean/std(式 4),$\hat A_i = (R(\tau_i)-\text{mean}(\mathcal G_{on}\cup\mathcal G_{off}))/\text{std}(\cdot)$;但 §4 "RL Practice" 又声明按 Dr.GRPO 去掉 std 与长度归一化 —— **论文内部不一致,以 §4 为准**。工程上取 $\pi_\phi\equiv 1$(避免 teacher/student 分词不一致),于是 off 分支的"重要性比"退化成 $\hat r_{j,t}=\pi_\theta(\tau_{j,t})$,**off 分支不 clip**,on 分支保留 clip(最终配置连 on 的 clip 也去掉)。

**Policy shaping(§3.2,式 6–8)**:off 分支用 $f(\hat r)\cdot \hat A$ 替换 clip,$f(x)=\frac{x}{x+\gamma}$,$\gamma=0.1$。logit 级梯度上界(式 8)在线性情形是 $\pi_\theta(1-\pi_\theta)$,$\pi_\theta\to 0$ 时**梯度消失**;shaping 后权重变成 $\frac{\gamma\,\pi_\theta(1-\pi_\theta)}{(\pi_\theta+\gamma)^2}$,相对朴素的放大倍数 $\gamma/(\pi_\theta+\gamma)^2$:$\pi_\theta\to0$ 放大 $1/\gamma=10\times$,$\pi_\theta\to1$ 压制到 $\approx 0.083$。图 2 右:权重峰值从 $p=0.5$ 移到 $p\approx0.1$。

**混合比例是固定的**:每 prompt 8 条 rollout = **1 条 off + 7 条 on(12.5%)**,全程不变。**论文未做混合比例消融**(附录 A 自陈"只用一条 off-policy 轨迹,且发现一条已足够")。唯一的自适应来自式 4 的组内联合归一:on 全错时 off 优势最大,on 全对时 $\hat A_{off}\to 0$ 自动退化为纯 on-policy。

**消融(表 4,Qwen2.5-Math-7B 六数学基准均分)**:On-Policy RL **45.5**;**Mixed-Policy(无 shaping)44.4** ← 比纯 on-policy **还低 1.1**;+Shaping 47.8;+Shaping+NoClip = LUFFY **50.1**。反向对照:On-Policy + Shaping **42.0**(-3.5)、On-Policy + NoClip 44.9(-0.6) —— shaping/NoClip 是 **off-policy 专用修补**,施加到纯 on-policy 上会掉分。表 1(ID/OOD 均分):LUFFY 50.1/57.8;SFT 44.1/47.5;RL w/ SFT Loss 40.1/46.0;SFT→RL 两阶段 48.2/**44.8**(OOD 差 13.0,且 LUFFY 只需其 59% GPU 时,表 2)。

**熵(图 2 左、图 5 右、图 6)**:mixed-policy 的熵**比纯 on-policy 塌得更快**(图 2 左标注 "Entropy Collapse"),机理是只强化那些在 $\pi_\theta$ 下本来就高概率的 off token;加 shaping 后熵稳定在 ~0.3–0.45,200–250 步间甚至回升。

**超参**:lr 1e-6 常数;rollout batch 128 prompts,update batch 64;8 rollouts/prompt;$\beta_{KL}=0$;entropy coef 0.01;rollout temperature 1.0;$\gamma=0.1$(§E.4 扫 {0.05,0.1,0.2,0.3,0.5},0.1 最优);500 步;8×A100,77×8 GPU·h。clip $\varepsilon$ 数值未报告。

**自陈局限(附录 A)**:仅限有 golden answer 的可验证奖励;强化推理会损害指令跟随;仅 ≤8B;每题只用 1 条 off 轨迹、单 teacher。

---

### 5. BRIDGE(2509.06948)—— cooperative gain 的定义值得抄,机制本身太贵

**cooperative gain(§3.2,式 5/9/10)**:不是梯度内积,是**两个模型在同一批 rollout 上 RL 目标值之差**。罚函数 $p(w,\theta)=\max_{\theta'}J_{RL}(\theta',w)-J_{RL}(\theta,w)$(式 5),用 Danskin 定理去掉二阶项(式 8)后得
$$J_{meta}(\theta,w)=(1-\lambda)J_{SFT}(\theta,w)+\lambda\big[\underbrace{J_{RL}(\theta,w)-J_{RL}(\hat\theta,w)}_{\text{cooperative gain}}\big]$$
$\hat\theta$ 是一份**纯 RL 影子模型**。估计方式:每步只采一批 rollout,用**同一批轨迹**同时给 $\theta$ 和 $\hat\theta$ 打分以降方差。开销(表 4):3B 时间 6.1→6.9h(+13%)、显存 52.2→59.3GB;8B 显存 50.7→67.4GB。

**调度(Algorithm 1)**:**双层优化 + 同批固定权重,不是 SFT/RL 交替**。每个 meta-step 三个更新全做,无阈值、无周期:$\theta$ 按 $(1-\lambda)J_{SFT}+\lambda J_{RL}$ 更新、$\hat\theta$ 按纯 $J_{RL}$ 更新、LoRA teacher $w$ 按 $J_{meta}$ 更新(策略是 $\pi_{\theta+w}$)。摘要里的 "alternates two updates" 指 $\theta$ 与 $w$ 交替,**不是 SFT 与 RL 交替**。

**$\lambda$ 固定不自适应**,表 12 扫描:0.0→37.6(退化纯 SFT)、0.3→47.2、**0.5→49.9(默认最优)**、0.7→49.0、1.0→42.9(退化纯 RL),[0.3,0.7] 平坦。自适应性全部来自 $w$ 重塑 loss landscape,不是缩放标量。

**理论**:**无编号定理/命题**。只有式(11)的噪声分析:二值 reward 以概率 $p$ 翻转时 $\mathbb E[r_{noisy}]=(1-2p)r_{true}+p$,加性偏置在**差值**中相消,cooperative gain 只被 $(1-2p)$ 缩放、符号不变,$p=0.5$ 才失效。

**实验(表 1,五数学 benchmark 平均)**——三组对照直接回答我们的问题:

| backbone | SFT | RL | SFT→RL | **SFT+RL(朴素同批)** | LUFFY | CHORD | **BRIDGE** |
|---|---|---|---|---|---|---|---|
| Qwen2.5-3B | 27.9 | 32.2 | 32.2 | **29.4** | 32.4 | 32.5 | **36.0** |
| Llama3.2-3B-Ins | 18.3 | 21.8 | 18.9 | **20.0** | 21.9 | 20.8 | **24.7** |
| Qwen3-8B-Base | 37.6 | 42.9 | 45.5 | **40.1** | 44.0 | 45.9 | **49.9** |

**朴素固定 λ 同批混合在三个 backbone 上全部低于纯 RL**(29.4<32.2、20.0<21.8、40.1<42.9)。交替更新同样无效(附录 A.1 表 10:SFT→RL 后每 5 步/10 步插入 SFT 更新得 31.5/31.4,**比不插的 32.2 还差**)。去掉上层更新退化成固定权重多任务:49.9→**40.3(-9.6)**(表 5),是全文最大消融幅度。表 9 噪声 reward:gap 从 $p=0$ 的 +3.8 涨到 $p=0.2$ 的 +19.0(纯 RL 崩到 13.9,BRIDGE 仍 32.9)。

**超参**:verl + GRPO;batch 128 / mini-batch 64;student lr 5e-7、teacher lr 5e-5;8 rollouts/prompt;2 epochs(≈130 步);KL 1e-4、entropy 1e-5;LoRA r/α=32/16;4×A100-80G(3B)。

**局限**:只在可自动验证任务上验证;只做到 3B–8B,70B+ 时影子模型 $\hat\theta$ 会主导显存;$p=0.5$ 时 cooperative gain 完全消失。

---

### 6. DUMP(2504.09710)—— learnability 驱动课程的完整算法,证据偏弱

**learnability 信号**:Theorem 1(证明在附录 A)—— 期望绝对 advantage $\mathbb E_{x\sim d}[\mathbb E_{o_i\sim\pi_\theta}[|\hat A_i|]]$ 是分布 $d$ 可学性的代理。取绝对值的理由(§3.1):正负 advantage 在 batch 内会抵消,掩盖真实的次优程度;$|\hat A|$ 小既可能是已掌握、也可能是太难/太噪。**它是 rollout 里已经算出来的量,额外开销近零。**

**算法(Algorithm 1)**:每个分布 $d_j$ 维护滑窗 $\mathcal A^w_{d_j}$(**默认 $k=300$**)、计数 $n_{d_j}$、初始 $P(d_j)=1/N$。每步:按 $P$ 采 batch → 算 advantage → 更新计数与滑窗(截断保留最近 $k$ 个)→
$$\hat L(d_j)=\frac{1}{|\mathcal A^w_{d_j}|}\sum_{a\in\mathcal A^w_{d_j}} a,\qquad \text{UCB}(d_j)=\hat L(d_j)+\sqrt{\frac{2\log(n_{total}+1)}{n_{d_j}+1}}\ \ (\text{式 3})$$
$$P(d_j)=\frac{\exp(\text{UCB}(d_j)/\tau)}{\sum_{j}\exp(\text{UCB}(d_j)/\tau)},\quad \tau=0.1\ \text{默认}$$
→ 用 GRPO 更新 $\theta$。**soft 选择而非贪心** —— 每步给所有 arm 一个概率,不是选一个。

**arm 数**:Setting 1 = 12 个(K&K 按人物数 3–14);Setting 2 = 15 个逻辑分布;Setting 3 = 5 个数学分布。**都是 5–15 量级。**

**效果**:Setting 2 平均 test answer reward 0.90→**1.17**;Setting 3 −0.59→**−0.43**。注意这个指标是 format(±1)+ answer(+2/−1.5/−2)的复合分,不是 accuracy,**换算不成百分点**。Setting 1 只有 12 张曲线图(图 1)无表,文中举例 9 字符分布 DUMP >0.5 而 baseline <0.0。基线只有 uniform sampling 一个。

**消融(§4.3)**:贪心($\tau=0$)显著劣于 soft($\tau=0.1$) —— 13/14 字符任务上 −0.91/−1.38 vs −0.66/−1.16。**贪心会早早锁死单个分布**,非零温度是必须的。

**图 2(自动课程形态)**:简单分布(3–5 字符)在早期被重采,复杂分布(10–14 字符)随训练推进逐步被优先 —— easy-to-hard 课程**自动涌现**,无需手工指定顺序。

**附录 C** 对静态启发式课程的论证:静态课程需要关于难度/可学性的强先验,而"看起来简单"的分布可能学习信号很低、"看起来难"的反而梯度效用高;且静态课程无法响应训练中模型能力的变化。

**超参**:GRPO + VeRL;Qwen2.5-7B-Instruct-1M / Qwen2.5-3B-Instruct;lr 1e-6;train batch 128;PPO mini-batch 32;KL 系数 0.001;每 prompt 16 responses;$k=300$、$\tau=0.1$;8×A100。

**局限(附录 D)**:只在 LLM 上验证未扩展到 MLLM;只到 7B 规模。**我补一条方法论局限**:没有与 reward 驱动的动态采样(如 DAPO / 按 pass-rate 筛)做对照,只比了 uniform。

---

### 7. ToolSample / DSCL(2509.14718)—— reward 驱动采样的精确规则,含最强的失败警示

**两个组件**。**RDS(Reward-Based Dynamic Sampling)** 对每条数据 $d_i$ 在第 $j$ epoch 算三个统计量:$M^{i,j}=\text{Mean}(\{R_g\})$(式 7)、$V^{i,j}_{sample}=\text{Var}(\{R_g\})$(式 8)、$V^{i,j}_{epoch}=\text{Var}(\{M^{i,1},\dots,M^{i,j}\})$(式 9,**跨 epoch 的均值方差**,衡量模型在这条数据上还有没有探索性变化)。据此分四类:

- **a. Easy**:$M^{i,j}=4$(满分)
- **b. Hard**:$M^{i,j}<t_{mean}$ — b.1:$V_{sample}>t_{var}$ **或** $V_{epoch}>t_{var}$;b.2:否则
- **c. Medium**:$M^{i,j}\in(t_{mean},4)$ — c.1:$V_{sample}>t_{var}$ **且** $V_{epoch}>t_{var}$;c.2:否则

$$\text{Ratio}_i=\begin{cases}0.0,& d_i\in\{a,\ b.2\}\\ 0.5,& d_i\in\{c.2\}\\ 1.0,& d_i\in\{b.1,\ c.1\}\end{cases}\quad(\text{式 10})$$

难样本用**宽松**判据(或)保留,中等样本用**严格**判据(且)保留。c.2 给 0.5 而非 0 的理由(§RDS 末段原文):"relying exclusively on highly informative or exploratory samples can lead to training instability"。

**实现方式是本篇最值钱的工程细节(Algorithm 1 第 8 行)**:$\hat A_b \leftarrow \hat A_b * \text{Ratio}$ —— **Ratio 乘在 advantage 上,不是过滤数据、不是改 dataloader**。这意味着 batch 形状不变、不需要重采样补齐,verl 里就是一行。

**TDCL(Task-Based Dynamic Curriculum)**:按子任务依赖关系分三阶段调 reward 权重(式 11–13):阶段 1 $\lambda_{format}\times2.5$、其余 $\times0.5$;阶段 2 $\lambda_{format}\times0.5$、name/key $\times1.5$、value 保持 0.5;阶段 3 $\lambda_{value}\times2.5$、name/key 降回 0.5。**阶段切换靠监控最近若干 batch 的各子任务 reward 动态判定**,不是固定步数。

**warmup 判据**:RDS 只在"连续 7 个 batch 的 mean reward 超过阈值 1.0"后才启动。

**效果(表 1,BFCL v3 Overall Acc)**:Qwen2.5-7B-Instruct 47.68;ToolRL 基线 56.96;+DAPO **47.35**;+SMV 57.03;+SMR 58.18;+RDS 59.55;+TDCL 59.95;**+DSCL 60.25(比 ToolRL +3.29)**。表 2(API-Bank Overall):ToolRL 61.81 → DSCL **64.99(+3.18)**。

**两条负面证据极其重要**:
1. **DAPO 掉到 47.35,比基线低 9.61,几乎等于未训练模型**。作者归因(§Main Results):DAPO 这类方法"mainly designed for **binary** reward problems",搬到多值细粒度 reward 的工具学习上不适用,且其 Soft Overlong Punishment 在短响应任务上有害。
2. **表 3:+RDS w/o CL = 47.34** vs +RDS = 60.48 vs ToolRL 56.96。不 warmup 直接开动态采样,**模型会因为大部分数据被丢弃而退回未训练水平**。

**超参(表 4)**:$t_{mean}=0.5$、$t_{var}=0.1$、8 rollouts;GRPO + veRL;8×H20-96GB;15 epochs;训练集 4000 条(ToolACE 2000 + Hammer 1000 + xLAM 1000)。作者注明 $t_{mean}$、$t_{var}$ 是"训练中调整"的超参,不是固定常数。

---

## 二、横向:三种配比范式的证据强度

| 范式 | 支持文献 | 证据强度 | 判断 |
|---|---|---|---|
| **固定配比** | Klear 3:1 step-level 交替(§3.4.3);MAHALO equal weights + balanced sampling(§6);LUFFY 固定 12.5% off(§4);BRIDGE 固定 λ=0.5(表 12) | **"够用"的证据最强,"最优"的证据为零** | 四篇都用固定配比拿到了主结果,但**没有一篇证明固定配比优于动态配比**——因为没人做这个对照。Klear 甚至明说 joint RL 无协同增益。 |
| **learnability 驱动** | DUMP(唯一一篇) | **弱**:单篇、单基线(只比 uniform)、7B 单尺度、指标是复合 reward 而非 accuracy、Setting 1 无数据表 | 机制清晰、开销近零、有 verl 实现、图 2 的自动 easy→hard 课程有说服力。但**从未与 reward 驱动方法正面对比过**。 |
| **reward 驱动** | ToolSample RDS(式 10,+3.29/+3.18);Klear 三个域各自的 pass-rate 窗口(§3.4.1);Embodied-R1.5 difficulty filter(§4.2.1)+ zero-variance mask | **最强且最一致**:三篇独立收敛到同一形态,且 Embodied 的消融显示 difficulty filter 是其配方里贡献最一致的单项 | 唯一被多篇独立复现的范式。**但有明确的适用边界**:ToolSample 表 1 证明为二值 reward 设计的过滤规则(DAPO)搬到多值 reward 上是 −9.61 的负收益。 |

一句话:**"该采哪些任务"(reward 驱动)证据充分;"该给哪个任务通道多少 batch"(配比)所有人都在用固定值,而且没人证明这是对的。**

---

## 三、适用性判决

### (a) research 样本占比:通道配比固定,任务级选择用 DUMP

**通道配比(executor vs researcher)固定,不做 UCB。** 两条理由,第二条是决定性的:

1. $K=2$ 时 UCB 的探索项 $\sqrt{2\log(n_{total}+1)/(n_{d_j}+1)}$ 很快在两臂间趋同,配比几乎完全由 $\hat L$ 的差决定;而 $\tau=0.1$ 的 softmax 会把 $\hat L$ 的微小差异放大成剧烈的配比抖动。DUMP 自己的 arm 数是 5–15。
2. **更根本:两通道的 $|\hat A|$ 尺度不可比。** executor 的 advantage 来自二值成功 reward 的组内去均值,researcher 的来自 $p_i\in[0,1]$ 下游成功率的组内去均值 —— 前者的 $|\hat A|$ 天然更大(二值分布的组内离差上界远大于连续 $p_i$ 的)。DUMP 的 Theorem 1 是在**同一 reward 函数**下比较不同 prompt 分布,它的三个 setting 里所有 arm 共用同一个 rule-based reward。跨异质 reward 通道直接比 $\hat L$ 没有意义,UCB 会系统性偏向 executor。

建议按 MAHALO §6 与 Klear §3.4.3 的做法:**固定配比 + step-level 交替**(比随机混更可控,Klear 原话是"Rather than mixing data randomly")。

**但另一维度 —— 3553 个训练任务里哪些值得花 research 预算 —— DUMP 的信号在这里同尺度、arm 数合适,应该用。** 具体方案:

- **arm 定义**:不要以单任务为 arm(3553 个 arm,每个的滑窗永远填不满 $k=300$,探索项将完全主导 UCB,退化成均匀采样)。按 **ALFWorld 任务族 × 难度分箱** 构造 arm,目标 $N\approx 12$–18,与 DUMP 的 12/15/5 同量级。任务族已有现成划分(pick_and_place / pick_clean_then_place / pick_heat_then_place / pick_cool_then_place / look_at_obj_in_light / pick_two_obj_and_place),每族按 executor 当前 pass-rate 分 2–3 箱。
- **信号必须取自 executor 的 rollout,不是 researcher 的。** 理由:research 的价值定义为"能否提升下游 executor 成功率";executor 在某族上 $|\hat A|$ 大 = 该族还有可学空间 = 值得投 research;executor 全对($|\hat A|\to 0$)= research 无从加分。**这同时天然解决了任务毕业问题**——毕业族的 UCB 利用项自动趋零。
- **参数照抄 DUMP**:滑窗 $k=300$、$\tau=0.1$、soft 采样(§4.3 已证明 $\tau=0$ 贪心会锁死单一分布)。
- **只动 research 侧的采样,executor 侧保持现状。** 两边同时动会让归因不可能 —— 这一条是为了保住实验的可解释性,不是保守。
- **必须先 warmup**(见判决 c 的警示)。

一句话:UCB 用在"研究预算投向哪一族任务",不用在"两个角色各占多少批"。

### (b) MAHALO 的可移植点(以及不可移植的部分)

**可移植 1 —— 式(6)的 per-objective batch-size 归一化,建议直接落地。**
$$L = \sum_i \alpha_i\cdot\frac{1}{|B_i|}\sum_{B_i} L_i$$
我们两通道的样本数结构完全不同:executor 每任务 8 条 rollout 产生 8 个 loss 项;researcher 每任务 8 份文档、每份 8 次下游 rollout,但只有 8 个 loss 项(reward 是聚合出来的 $p_i$)。若按 verl 默认的全局 token-sum 聚合,长轨迹通道会系统性主导梯度 —— 这与已记录的"loss 是 SUM 聚合、K2 放大机理"是同一个坑的两个面。**先按各自 $|B_i|$ 归一化再用 $\alpha_i$ 加权**,把"配比"这件事从"样本数的意外副产品"变回一个显式可控的超参。

**可移植 2 —— 表 2 的负面警示,当作监控项。** 目标冲突时,共享参数把异质信号 pool 在一起会灾难性崩塌(Help 0.5800→0.3397、Honest 0.3042→0.0734,双双低于 base)。我们的两角色输出空间不同(executor 出动作、researcher 出文档),先验上冲突弱于 helpfulness-vs-honesty,但**这正是需要监控而非假设的东西**:建议在 by-family 行为指标之外,加一条"两通道各自的单通道基线对照"(见 c 的建议)。

**可移植 3 —— Finding 7 的分工原则。** 高可验证 reward 靠 test-time 搜索获益最大(Math Acc 0.6853→0.7993),低可验证 reward 靠训练时通道分离获益。映射到我们:executor 的二值成功是高度可验证的 → 若要再挤性能,**best-of-N + verifier 的 test-time 路线可能比继续加训练技巧更划算**;researcher 的 $p_i$ 是噪声大的间接信号 → 收益主要来自训练时把它与 executor 分开(我们已经这么做了,这是一条支持现状的证据)。

**不可移植**:MAH-DPO 的 multi-action-head 与 PRM-guided decoding。前者是 DPO 框架下的机制(我们用 GRPO,没有偏好对);后者需要训 PRM,而式(1)的 PRM 训练要对每个中间步做 $M$ 次 rollout —— 在 ALFWorld 多轮交互上成本不可接受。**必须向团队说明:brief 期待的"可验证与不可验证奖励同批 RL 优化的精确机制",这篇论文没有提供,它把所有信号先统一成 PRM 连续分再走 DPO。**

### (c) 任务毕业后的采样调度

三篇独立文献收敛到同一形态,建议照做,但有两处必须偏离字面配方:

**第一层(免费,立刻做)—— 零方差组置零。** 判据用 Embodied-R1.5 §4.2.1(2) 的 **"组内 reward 全等"** 而不是"全对/全错" —— 对 researcher 通道的连续 $p_i$ 同样适用(8 份文档的下游成功率恰好全相等时同样无信号)。做法是 mask,不重采样。

**第二层 —— pass-rate 窗口筛选,但保留 0.5 而不是 0。** Klear §3.4.1 的三条硬规则($0<p<1$ / $0.25\le p\le 0.75$ / 至少解出一次)可直接借用;我建议 executor 侧起步用 $0 < p < 1$(最宽松、最安全),观察后再收紧到 $0.25\le p\le 0.75$。**但对已毕业任务不要用 Ratio=0**:ToolSample 式(10)对"中等但信号弱"的 c.2 类给 0.5 而非 0,理由是只喂高信息样本会训练不稳定。对我们而言还多一层理由——毕业任务若被永久移除,遗忘无法被察觉。建议毕业任务给一个小的非零 Ratio(0.2–0.5),既省算力又保留遗忘探针。

**第三层 —— 实现方式:advantage 乘系数,不改 dataloader。** 抄 ToolSample Algorithm 1 第 8 行 $\hat A_b \leftarrow \hat A_b * \text{Ratio}$。verl 里是一行,batch 形状不变,不需要处理"过滤后 batch 变小"的连锁问题。这条是全部七篇里工程性价比最高的单点。

**警示(比配方本身更重要)——必须先 warmup。** ToolSample 表 3:不 warmup 直接开动态采样,BFCL 56.96 → **47.34**,几乎退回未训练模型(47.68)。他们的启动判据是"连续 7 个 batch 的 mean reward 超过阈值"。我们的对应做法:等 executor 成功率曲线进入稳定上升段再开动态采样,并把启动步数记进 launch_manifest。**在崩塌史(CD-003)之后,这一条应当按硬闸门对待,不是建议。**

---

## 四、落地建议

### 必做(证据充分、成本近零、不改架构)

1. **零方差组置零**,判据"组内 reward 全等"(Embodied-R1.5 §4.2.1(2))。两个通道都做。
2. **executor 侧 pass-rate 窗口 $0<p<1$**(Klear §3.4.1),实现用 advantage 乘 Ratio(ToolSample Alg.1 L8),毕业任务给 0.2–0.5 而非 0。
3. **动态采样必须 warmup 后启动**,判据写成显式条件并记进 launch_manifest(ToolSample §RDS 脚注 + 表 3)。
4. **loss 聚合改成 per-channel batch-size 归一化再加权**(MAHALO 式 6),把两通道配比变成显式超参。
5. **通道配比固定 + step-level 交替**(Klear §3.4.3、MAHALO §6),不要随机混批。

### 症状驱动(先测症状,确认再上)

6. **DUMP-UCB 用于 research 预算的任务族选择**(方案见判决 a)。触发条件:观察到 research 预算被大量花在 executor 已经全对或完全无望的任务族上。先做一次离线统计(按族统计 executor 的 $\text{mean}|\hat A|$ 分布)确认存在错配,再上在线 UCB。
7. **cooperative gain 当纯监控指标**(BRIDGE 式 10 的方括号项),**不上 LoRA teacher**。做法:跑一个纯 GRPO 的影子分支,在同一批轨迹上给两个模型打分做差,量化"这一步的 OPD/SFT 混合到底帮了还是害了 reward"。触发条件:怀疑混批在拖累 reward 时。注意开销是一份完整模型副本,而 A800 上 4B 已经贴着显存天花板 —— 可以只在小规模对照实验里跑,不进主线。
8. **LUFFY policy shaping($f(x)=x/(x+0.1)$)用于 OPD 分支**。**但先别上**:LUFFY 的 shaping 是为 $\pi_\phi\equiv 1$ 的近似设计的,而我们的 OPD teacher 是真实模型、有真实概率;更关键的是 CD-003 已定论我们的崩塌病因是 moving self-teacher 正反馈,不是低概率 token 梯度消失。触发条件:实测到 OPD 分支上 teacher 高概率而 $\pi_\theta$ 低概率的 token 其梯度范数确实塌陷。
9. **executor 侧 test-time scaling(best-of-N + confidence 选择)**。MAHALO Finding 7 与 Klear §3.4.4 都指向"高可验证 reward 从 test-time 获益大";Klear 实测 Confidence Selection 最稳(BFCL 16 候选 0.7668)。触发条件:训练侧收益见顶时。

### 不采纳

10. **MAHALO 的 multi-action-head + PRM-guided decoding**。主机制在 DPO 框架内,我们用 GRPO 没有偏好对;PRM 训练(式 1)要对每个中间步做 $M$ 次 rollout,ALFWorld 多轮交互上成本不可接受。
11. **BRIDGE 的完整机制(LoRA teacher + 影子模型双层优化)**。需要多一份完整模型副本(3B 上显存 +13.6%,8B 上 +33%),而 A800 上 4B 已在显存天花板;且其增益来源(表 5 的 −9.6)全在上层更新,砍掉就退化成固定权重多任务,没有折中版本。只取 cooperative gain 当监控(第 7 条)。
12. **Klear 的 model merge(式 9–11)**。论文自己的实验就掉分(SWE 40.4→39.4、Aider 35.1→33.8);更重要的是,我们让两角色共享 backbone 的目的就是让 research 能力反哺 executor,merge 会正好破坏这一点。
13. **DUMP-UCB 用于两个角色通道的配比**。$K=2$ 且两通道 advantage 尺度不可比,详见判决 (a)。
14. **DAPO 式的纯 zero-variance 动态采样(对整批做过滤 + resample 补齐)**。ToolSample 表 1 实测在多值 reward 任务上是 **−9.61** 的负收益,作者归因于该设计针对二值 reward。我们的 researcher 通道恰恰是连续 $p_i$。用第 1 条的 mask 方案代替。
15. **静态启发式课程(手工排 easy→hard)**。DUMP 附录 C 的论证成立:需要关于难度的强先验,而"看起来简单"的分布可能学习信号很低;且静态课程无法响应 executor 能力的演化 —— 这正是我们关切③的核心。

---

## 附:与团队预期的两处落差(需知悉)

1. **MAHALO 不是 RL 混 reward 的论文**(详见 §一.3)。brief 里"最贴我们二值 vs 连续混合的关切"这一判断,读完正文后不成立。真正回答这个问题的是 **Embodied-R1.5 §4.2.1(3)(4)**:二值与连续同批、reward 端全部压进 [0,1]、advantage 端 group-mean / batch-std —— 这与我们现行的"组内去均值不除 std"的差别**只在分母**,是一个干净的可控对照实验。
2. **Klear-AgentForge 对"多任务联合 RL"的结论偏负面**(§3.4.3),它给的配方是固定 3:1 交替 + 大量难度筛选,而不是任何形式的动态配比。这篇是本组里最接近我们场景(agentic、多轮、二值 outcome reward、8B 规模)的工业报告,它的沉默本身是信息:**没有人在多任务 agentic RL 上做出过动态配比的正收益。**
