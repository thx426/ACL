# 多目标权重与奖励归一化六篇精读报告

> 精读范围:GAC / CHORD / CTWA / RDPO / HPT / SRFT 六篇正文(方法 + 实验),非摘要。
> PDF 基路径:`.conductor/branches/CD-002/papers-mtl/`
> 服务对象:TRACE-OPD 四项联合 loss `L = w_opd·OPD + w_rl·RL_exec + w_sft·SFT + w_res·RL_research`
> 所有结论标注论文节/式编号。日期:2026-07-28。

**一句话结论**:六篇里没有一篇的自适应机制直接适配我们的四项异质联合;第一版应当上**固定系数 + 三层零反传监控**,证据来自这些论文自己的消融(GAC 的 oracle-μ 上限只比固定 μ 高 5.7pp;HPT 更激进的自适应反而更差;CHORD 承认 token 级 φ 到位后 μ 调度"不再必要")。真正值得立刻移植的是**统计量与判据**,不是控制器。

---

## 1. GAC(2605.26184)噪声感知混合

### 算法精确公式

**核心闭式解(Theorem 1, Eq 3)**

```
μ* = (α_tgt·Δg² + σ_r²) / (Δg² + σ_s² + σ_r²)
```

由最小化 MSE 目标 `E(μ) = E[‖ĝ(μ) − g*‖²]`(Def 1, Eq 1)得到,展开为 Eq 2:

```
E(μ) = (μ − α_tgt)²·Δg² + μ²·σ_s² + (1−μ)²·σ_r²
```

符号解释:
- `σ_s²` / `σ_r²` —— SFT / RL 梯度噪声方差
- `Δg² = ‖g_s* − g_r*‖²` —— 两目标**无噪梯度的分歧度**
- `α_tgt` —— 用户指定的目标混合偏好(理论推导用固定值,实践中换成 KL 控制的 `α_ctrl`)

物理含义(§3.1):`Δg² → 0` 时退化为逆方差加权;`Δg² → ∞` 时 `μ* → α_tgt`。

**相关噪声扩展(Eq 5)**:`μ_c* = (α_tgt Δg² + σ_r² − c) / (Δg² + σ_s² + σ_r² − 2c)`,`c = tr Cov(ε_s, ε_r)`。
**作者自己弃用了它** —— 附录 B 实测 `c` 的变异系数 >0.8,收益 +0.2pp 且 paired t-test p=0.38 不显著,却把大跳变事件从 3% 抬到 9%。Eq 6 是偏置估计器版本(RL 有 clipping/IS/KL 正则导致的偏置 `b_s, b_r`)。

**三个在线代理(逐条抄录)**

| 代理 | 公式 | 与真值相关性(Table 11) |
|---|---|---|
| `σ_r²`(Eq 8) | `(1/\|B\|) Σ_{i∈B} (Ā_i − (1/\|B\|)Σ_j Ā_j)²`,`Ā_i` 为序列级归一化 advantage | 与 `Var(∇L_r)` r=0.82±0.04 |
| `σ_s²`(Eq 9) | `Var_trim(nll_i)`,`nll_i` 长度归一化 NLL,**上下各裁 10%** | 与 `Var(∇L_s)` r=0.76±0.05 |
| `Δĝ²`(Eq 10) | `E_{i∈B}[ mean_{t∈resp(i)} (g̃_{s,it} − g̃_{r,it})² ]`,`g̃` 是 **z-normalized 的每 token 损失系数**(RL 取 advantage,SFT 取权重) | 与 `‖g_s*−g_r*‖²` r=0.84±0.03 |

`Δĝ²` 的构造依据:两目标梯度都形如 `∇_θL = E[Σ_t c_t ∇_θ log π_θ(a_t\|s_t)]`,只有系数 `c_t` 不同,故**系数之差即分歧度**。Table 12 的破坏性检验很有说服力:打乱 `A_t` 后相关性掉到 0.12,打乱 `φ(p_t)` 掉到 0.09,常数系数掉到 0.05 —— 说明信号来自真实结构而非偶然。

**KL 控制的 α_ctrl(Eq 11 + 16)**:`α_ctrl,t+1 = clip(α_ctrl,t · exp(s_t), [α_min, α_max])`,步长 `s_t` 走迟滞规则:

```
s_t = { η↑(KL_t/KL_tgt − 1)      if KL_t/KL_tgt > 1+h
      { −η↓(1 − KL_t/KL_tgt)     if KL_t/KL_tgt < 1−h
      { 0                         otherwise
```

在线估计器 Eq 12:`μ_t = (α̂_ctrl Δĝ_t² + σ_{r,t}²) / (Δĝ_t² + σ_{s,t}² + σ_{r,t}²)`。

**守护控制器(Eq 13, Alg 1)**,三步:
1. EMA 平滑:`μ_ada = β·μ_{t−1} + (1−β)·μ*`
2. 调度先验混合:`μ_blend = (1−λ)·μ_prior(t) + λ·μ_ada`,`μ_prior` 是 warmup+cosine
3. 逐步幅度封顶:`δ_t = clip(μ_blend − μ_{t−1}, −c̄, +c̄)`,`μ_t = clip(μ_{t−1} + δ_t, [μ_min, μ_max])`

最终损失:`L = (1 − μ_t)·L_RL + μ_t·L_SFT-φ`,其中 `L_SFT-φ` 用的是 CHORD 的 token 级 `φ(p)=p(1−p)`(GAC 明确声明这部分是沿用,非自己贡献)。

### 超参默认值

β=0.99,c̄=0.01,λ=0.5,f_μ=10(每 10 步更新统计),NLL 尾部裁剪 10%,`(KL_tgt, α_min, α_max, η↑, η↓, h) = (0.02, 0.1, 0.95, 0.2, 0.3, 0.1)`,KL-EMA 系数 0.9,μ_init=0.5。
Base model Qwen2.5-7B-Instruct;SFT 5,000 条 + RL 20,000 prompts(OpenR1-Math-220k);AdamW cosine,batch 64;GRPO batch 32,K=8。
敏感性(Table 17):β∈[0.95,0.99]、c̄∈[0.003,0.02]、λ∈[0.3,0.7] 内波动 ≤0.5pp。

### 计算/显存开销

3 个 EMA 标量,**零额外 forward/backward**(附录 E 明确:只用已有张量 —— token log-probs、masks、advantages),唯一的分布式同步是 `σ_r²` 一阶/二阶矩的低成本 all-reduce。
实测(Table 14,Qwen2.5-7B FSDP 8 GPU):L=2048 → +0.6%,L=4096 → +0.7%,L=8192 → +0.8% wall-time;峰值显存无可测增加(22.4/31.8/42.6 GB 与基线同)。
**对照**:MGDA/PCGrad/CAGrad/Nash-MTL 需要 **2 次 backward**(附录 I:"We compute ∇L_SFT and ∇L_RL on shared mini-batches (2 backward passes)")。

### 证据幅度(固定 vs 自适应,这是我们最关心的)

**Table 7 因果归因(单一最重要的表)**:

| 配置 | AMC | 归因 |
|---|---|---|
| QCM(固定 μ,无平滑) | 62.1 | 基线 |
| 固定 μ + EMA | 62.3 | EMA 单独 **+0.2pp** |
| CHORD(token-φ,固定调度) | 62.5 | φ 单独 **+0.4pp** |
| GAC w/o φ(仅噪声引导) | 65.8 | 噪声估计器 **+3.7pp** |
| Full GAC | 67.2 | 合计 +5.1pp |
| **Oracle μ(上限)** | **67.8** | **最大可达** |

即"完美自适应"相对固定权重的**全部可得空间只有 5.7pp**,GAC 拿到 5.1。

其他:
- 稳定性(Table 9):KL-drift 面积 12.4→8.9(**−28%**),`|Δμ|>0.02` 大跳变事件 11%→3%(**−73%**),长度方差 284→189(−33%)
- 规模效应(Table 6,vs HPT):1.5B **+2.2pp** / 7B **+3.8pp** / 14B **+3.3pp**。作者解释小模型 `σ_s²/σ_r²` 动态范围小,自适应空间小 —— **我们是 4B/1.7B,落在增益偏小的一端**
- 代理退化消融(Table 8):Full 67.2(3% 大跳变)→ 常数 σ_r²=1 得 65.5(7%)→ 打乱 Δĝ² 得 65.1(9%)→ 随机 Δĝ² 得 64.8(11%)
- 组件消融(Table 22):去掉 cap −2.5pp,去掉 EMA −1.8pp,去掉先验混合 −1.1pp,不做 trim −0.7pp
- 弱先验鲁棒性(Table 15):即使 `μ_prior` 设成常数 0.1 或 0.9 的"坏先验",AMC 仅降 ≤0.4pp
- 跨域:MBPP 78.8(+3.4 vs CHORD)、HumanEval 83.5、GPQA 43.5(+3.9 vs CHORD)、SciBench 41.2、BBH 65.7

**方法学诚实度**:只有 3 seeds,作者自陈"显著性需 p<0.05 且 Cohen's d>0.8,3 seed 下应视为指示性而非定论";Proposition 2 明确标注为"motivating analysis, not convergence guarantees";Limitations 承认只在可验证 reward 任务上验证过。

### 适用性判决

**统计量直接适用,控制器不适用。**

- 闭式解 Eq 3 是**两路推导**(μ 与 1−μ),四项联合需要在单纯形上重推 Theorem 1,论文没做,我们也不该在第一版做。
- 三个代理量与我们的 verl 张量一一对应,是**最便宜的预警层**,第一版就该上。
- 另一个直接可用的失败样本:附录 K/L 记录 QCM 固定 μ 在 **step 280–320 有 2/3 seed 熵崩**(熵 0.035→0.008,40 步内),伴随 4.2pp 掉点;而同期 GAC 的 μ 从 0.52 升到 0.67。**"σ_r² 尖峰 → 熵悬崖"这一对症状,就是固定系数最典型的死法**,直接进我们的监控告警。

---

## 2. CHORD(2508.11408, ICLR 2026)双控:全局系数 + token 级权重

### 算法精确公式

**全局系数(Eq 3)**:`L_Hybrid(θ) = (1−μ)·L_GRPO(θ) + μ·L_SFT(θ)`,`μ ∈ [0,1]`。

μ 走衰减调度。**注意论文两处不一致**:正文 §4.3 写"前 **200** 步从 0.9 降到 0.05 后保持";附录 A.1 写"μ decay schedule is to decrease from 0.9 to 0.05 over the first **30** training steps"。**以 A.1 的实现参数为准**(附录是超参表)。B.4 医疗任务用 decay step=50。

**token 级权重(Eq 5–6)**

```
φ(y*_t; π_θ) = p_t·(1 − p_t),  其中 p_t = π_θ(y*_t | x, y*_<t)
L_SFT-φ(θ) = −E_{(x,y*)~D_SFT}[ Σ_t φ(y*_t; π_θ)·log π_θ(y*_t | x, y*_<t) ]
```

抛物线在 p=0.5 取峰、两端趋零:**同时压制模型已确信的 token(防熵崩)和极不可能的 token(防破坏既有 pattern)**。作者从信息论角度解释为"生成该 token 这一二元事件的策略不确定性",并称其创造了一个"learning sweet spot"。

**对照的 IS 版本(Eq 4)**:`L_SFT-IS = E[−Σ_t sg(π_θ(y*_t|·)/π_sample(y*_t|·))·log π_θ(y*_t|·)]`,`sg` 为 stop-gradient;因 expert 数据的 `π_sample` 未知,按惯例令分母为 1。

**为什么需要 φ 而不是 IS**(Fig 5,关键对照实验):不加 IS 直接混 off-policy 数据 → 熵**急升**(既有 pattern 被破坏);加 IS → 熵**急崩**(IS 下压低概率 token 的同时激进强化高概率 token,导致过度自信)。φ 是二者之间的解。

**"shift-readapt-overfit" 三阶段(§3.1, Fig 2)**:在 Qwen2.5-7B-Instruct 上用 DeepSeek-R1 数据做 SFT,MATH-500 准确率先跌(policy shift,expert pattern 与既有 pattern 冲突 + exposure bias)、再升(readapt)、最后跌(overfit)。这是全文的问题动机。

### 超参默认值

Adam(β1=0.9, β2=0.999),lr ∈ {1e-6, 5e-6, 1e-5},rollout/eval 温度 1.0,max response 16k,SFT 最多 3 epoch,K=8 严格 on-policy(`π_sample = π_θ`)。
数学任务:batch 64(SFT)/32(RL),最多 1500 RL 步;工具调用:batch 96 双侧,100 RL 步。clipping ε=0.2。
**CHORD-φ 用固定 μ=0.1**。硬件 8×A100 + 8×H20;SFT 用 LLaMA-Factory,RL 用 Trinity-RFT。
数学 reward 是分层的:正确 +1.0;格式对但答案错 0.0;格式与答案都错 −0.1;超长无答案 −1.0。工具调用为二值 reward。

### 计算/显存开销

φ 只用前向已有的 `p_t` 做逐元素乘,**零额外开销、零额外反传**。全局 μ 是标量调度,零成本。
(注:CHORD 本身仍需在 expert 数据上跑一次 SFT 前向,但那是 loss 项本身,不是额外开销。)

### 证据幅度(固定 vs 自适应,有反转)

主结果 Table 1:

| 方法 | AMC | AIME24 | AIME25 | MMLU-Pro | BFCL Overall |
|---|---|---|---|---|---|
| Original | 43.8 | 11.7 | 6.66 | 24.7 | 46.2 |
| GRPO(纯 RL) | 52.1 | 13.2 | 8.54 | 45.8 | 77.1 |
| LUFFY | 52.8 | 16.6 | 14.3 | 44.0 | 76.1 |
| SFT-best + RL | 58.4 | 17.1 | 16.3 | 51.3 | 76.1 |
| CHORD-μ | 60.8 | 18.1 | 17.9 | 43.3 | 77.6 |
| **CHORD-φ** | **62.5** | **18.2** | 17.2 | **56.2** | **78.5** |

**固定 μ 劣于动态 μ 的证据(Fig 7)**:固定 μ ∈ {0.5, 0.1, 0.02} 全部劣于动态 μ,且 μ=0.02 "相对纯 on-policy RL 没有显著改进"。

**但附录 B.7 明确退让**:φ 到位后"复杂的 μ 衰减调度不再必要"(a complex and decaying schedule for μ is no longer essential),CHORD-φ 用**固定 μ=0.1** 拿到 BFCL 78.5,反超走调度的 CHORD-μ 77.6。Fig 11 显示更精细的 warmup-then-decay(CHORD-tune-both)与 CHORD-φ 最终差距"不显著"。

**reward 感知自适应 μ(附录 B.1)**:`μ' = max(0, τ − reward_mean)`
- τ=0.5 → BFCL Overall **78.1**(可行,接近 CHORD-φ 的 78.5)
- τ=0.7 → **75.6**(**低于纯 GRPO 的 77.1**)
- 作者结论:"仍依赖任务特定超参调优,其实际实现未必更简单"

**φ 变体消融(Table 4/5,证明设计不是任意的)**:

| φ 变体 | BFCL Overall | AMC23 | AIME24 | AIME25 |
|---|---|---|---|---|
| GRPO(纯 RL) | 77.1 | 55.0 | 12.6 | 7.3 |
| **CHORD-φ = p(1−p)** | **78.5** | **59.7** | 14.0 | **14.2** |
| Entropy Top(仅 top5% 高熵 token) | 77.8 | 58.8 | **17.2** | 13.8 |
| Entropy Norm(φ ∝ H(t)) | 78.0 | 52.5 | 15.0 | 9.2 |
| IS Clip(clip p_t>0.4) | 75.9 | 55.0 | 13.9 | 12.0 |
| **Focal Loss φ=(1−p)^γ** | **73.4** | **34.4** | **3.8** | **4.2** |

**Focal Loss 直接崩塌** —— 上加权低概率(高 surprise)token 会摧毁已学推理能力并过拟合 expert pattern。这条负面结果对我们的 OPD 权重设计是**硬约束**。

其他:响应长度(Table 2)CHORD-φ 数学 2,444 token(expert 6,132、SFT-best 8,442、纯 RL 1,423),工具调用 120 token —— 说明 φ 能**按任务特性选择性吸收** expert pattern 而非整体照搬。非可验证域(B.4 RaR-Medicine)CHORD-φ 81.3 vs 纯 RL 76.8。弱模型(B.5,Qwen2.5-3B)朴素 SFT 崩塌、SFT+RL 不稳,只有 CHORD-φ 稳住。

### 适用性判决

**φ 值得立刻移植到 OPD 和 SFT 两项;μ 调度暂不需要。**

- 我们的 OPD 是 token 级 dense 蒸馏,`φ(p)=p(1−p)` 天然适配:对 policy 已经确信的 token 降权,正好抑制"self-teacher 正反馈"这条我们已确认的崩塌通路(CD-003 结论)。
- Focal Loss 的崩塌结果反向印证:**不要给低概率 token 加权**。
- μ 调度对我们价值低,因为 B.7 自己说 φ 到位后它可省;B.1 的 reward 感知版本又对阈值极敏感(τ 从 0.5 到 0.7 就跌破基线)。

---

## 3. CTWA(2602.06869)协方差定律 + 权重自适应

### 算法精确公式

**局部协方差定律(Theorem 4.2, Eq 3)**

```
r_m(p_θ⁺) − r_m(p_θ) = η · E_{x∼D}[ Cov_{y∼p_θ(·|x)}( r_m(x,y), s(x,y) ) ] + O(η²)
```

其中 `p_θ⁺` 是 KL 正则改进步(Eq 2)的最优解(Lemma 4.1:`p_θ⁺(y) ∝ p_θ(y)·exp(η·s(x,y))`),`s(x,y) = Ψ(r(x,y))` 是标量化分数,`r_m` 是第 m 个目标的**真实 reward**。

**推论**:若 `E_x[Cov(r_m, s)] > 0`,则对充分小的 η 有 `r_m(p_θ⁺) > r_m(p_θ)`。反之负协方差意味着一阶意义上**提升 s 必然降低 r_m**。

**Remark 4.3(cross-objective interference 的机理)**:线性标量化 `s_λ = Σ_j λ_j r_j` 下,`Cov(r_m, s_λ) = Σ_j λ_j Cov(r_m, r_j)`。**强调某个"容易"目标会翻转"困难"目标的协方差符号**;且因协方差是 on-policy 计算的,**符号会在训练中翻转** —— 早期改善某目标、后期开始损害它。

**两模式玩具例(§4.2, Eq 4–5)**:闭式协方差

```
Cov_{y∼p_θ}( r_m, s ) = p_t·(1 − p_t)·(r_good − r_bad)·(s_good − s_bad)
```

log-odds 更新 `log[p_{t+1}/(1−p_{t+1})] = log[p_t/(1−p_t)] + η(s_bad − s_good)`。当代理分数偏好 `y_bad`(`s_bad > s_good`)而目标 m 偏好 `y_good`(`r_good > r_bad`)时,`p_t → 1`,`E[r_m] = r_good − p_t(r_good − r_bad)` **单调下降**,而协方差(Eq 5)严格为负 —— 定律精确地标记了这个失败。

**裁剪鲁棒性(Corollary 4.4 / C.4)**:clipping 只是从更新中删去一部分加权 logit-gradient 项;若 Fisher 几何下的"被删梯度质量" `‖F(θ)^{-1/2}(G_unclip − G_clip)‖` 小于未裁剪的一阶改善裕度 `γ_m^unclip`,则 `r_m(θ⁺) ≥ r_m(θ)` 的充分条件仍成立。附录 C.2 还给出自然梯度版本:`Cov(r_m, w) ≥ 0 ∀m` 是"无目标退化"的充分一阶条件。

**CTWA 算法(§5 + Alg 1),逐条**:

1. token 级裁剪后 advantage 权重 `w^clip_{k,l}(θ) = A_k(x)·ρ_{k,l}(θ)·1_{k,l}(θ)`,其中 `1_{k,l}=0` 恰当该项被裁掉
2. completion 级聚合:`w(x, y^(k); θ) = (1/L_out)·Σ_l w^clip_{k,l}(θ)`
3. 组内经验协方差:`Ĉov_m(x) = Cov_{k=1..K}( r_m(x, y^(k)), w(x, y^(k); θ) )`
4. batch 平均:`c_m = E_{x in batch}[ Ĉov_m(x) ]`
5. EMA:`c̄_m ← (1−τ)·c̄_m + τ·c_m`
6. **单边**缺口:`δ_m = [ c*_m − c̄_m ]_+`(只在协方差低于目标时才动)
7. **对数空间**更新保正:`u_m ← u_m + η_λ·δ_m`,`λ_m = exp(u_m)`

注意执行顺序(Alg 1 第 11–16 行):**先做标准 GRPO 内层更新,再用同一批数据算 CTWA 统计并做外层权重更新**。

### 超参默认值

附录 F Table 1:`λ(0) = [0.333, 0.333, 0.334]`;权重学习率 `η_λ = 0.05`;协方差目标 `c* = [0.15, 0.08, 0.08]`(accuracy 这个"更难"的目标给更高的 0.15);EMA 率 `τ = 0.1`。
共享策略更新:lr 1e-6 常数,batch 32,K=16 rollouts,max prompt/response 1024,entropy coef 0,KL coef 0.001,clip ε=0.2,90 epochs,FSDP,4×L40 48GB,vLLM。
三个目标:accuracy、conciseness(响应长度)、clarity,均为规则可验证的 0/1 reward。

### 计算/显存开销

论文明确声明:比 dynamic weighting / MGDA 更省,因为**不需要 per-objective 梯度、不需要投影梯度下降**;协方差可与标准 RFT 流程同批计算,"negligible overhead"。**零额外反传**。

### "变负预示退化"的证据强度 —— 必须分级

| 级别 | 证据 | 强度 |
|---|---|---|
| 理论 | Theorem 4.2 | **一阶、on-policy、带 O(η²) 余项**,是充分条件不是充要条件;需要 KL 正则改进步或自然梯度这一理想化更新 |
| 相关性 | Fig 3 | 受 interference 的基线在所有目标上协方差**一致低于** CTWA。方法间对比,非 run 内因果 |
| **干预性** | **Fig 4a** | 把协方差目标设成**负值**(−0.08、−0.15)一致劣于正值,且幅度差越大差距越大。**这是最强的一条** |
| 稳健性 | Fig 4b | 正目标在 [0.08, 0.20] 区间稳健,0.15 最佳,**0.20 开始下降**(且实际难以达到) |
| **缺失** | —— | **全文没有任何"run 内 c_m 在 step t 转负、指标在 t+k 掉"的时间序列** |

**结论:足以支撑记录它,不足以支撑用它 gate 训练。**

**附录 B 的自我限制**:Theorem 6.4(μ-PL 全局收敛)"难以直接转成实用训练策略",因为 Assumption 6.3 含 token 级 Jacobian 项,计算代价过高;他们试过把 per-token logit 梯度投影到共享参考方向的锥内,"未取得预期收益"(did not yield desired gains)。

### 对我们最有价值的其实是一条反面结论

**附录 A Fig 5**:三个模型(Qwen2.5-1.5B-Base / -IFT / Qwen3-1.7B-Base)上,per-objective 梯度的 pairwise 余弦相似度**始终非负并随训练收敛到 0**,而 cross-objective interference 正在发生。原文:"objectives are weakly coupled, neither strongly synergistic nor persistently antagonistic, with **no conflicting behavior observed**"。

即 **PCGrad 式的梯度冲突检测在这里是假阴性**。如果我们的监控里含"per-objective 梯度余弦相似度",它不会报警 —— 而且算它需要 per-objective 反传(2× backward),是这六篇里唯一被明确规避的开销。

### 适用性判决

**定律的形式可用,算法本体不可直接用。** CTWA 结构上要求 M 个 reward 打在**同一批 rollout** 上、共享一个标量化 advantage;我们的 OPD/SFT 根本不是 rollout 上的 reward。改写方案见 §必答问题 (b)。

---

## 4. RDPO(2605.13641)MAQ 归一化 + Mahalanobis 白化

### 算法精确公式

**MAQ(Magnitude-Aware Quantile)归一化,三步(Eq 1–2)**。对每个 prompt i 与 reward k,组内排序 `r_1 ≤ r_2 ≤ … ≤ r_G`:

1. **对数压缩间距**
   ```
   gap_j = log( 1 + |r_{j+1} − r_j| / (β · σ_global) ),   j = 1..G−1
   ```
   `σ_global` 是该 reward 在**全局 batch 上的四分位距 IQR**(稳健尺度基准),`β > 0` 控制压缩强度。对数压缩是稳健性的关键:天然抑制离群点,同时对小的密集间距近似线性,保留组内细微差异。

2. **CDF 分配**:`norm_gap_j = gap_j / Σ_{j'} gap_{j'}`,CDF 位置 `u_(j)` 按归一化间距**成比例系统分配**

3. **逆正态映射**:`A_(j) = Φ^{-1}( u_(j) )`

MAQ 的作用**不是去相关**,而是让组内 advantage 分配稳定。Fig 2 用两个诊断量衡量(`p_j = |A_j| / Σ_ℓ |A_ℓ|`):
- **advantage 支配度** `max_j p_j` —— 最大 rollout 占据的绝对 advantage 质量份额
- **有效 rollout 参与度** `1/(G · Σ_j p_j²)` —— advantage 质量在 G 条 rollout 上的均匀度

MAQ 在四个任务子空间上一致降低前者、抬高后者。

**Mahalanobis 白化(Eq 3–5)**

```
W^(i,j) = Σ̂_t^{-1/2} · A^(i,j)                                    (Eq 3)
Σ̂_t^{-1/2} = U Λ^{-1/2} U^T,  由 Σ̂_t = U Λ U^T 特征分解得到
Σ̂_t = (1−α)·Σ̂_{t−1} + α·Σ̂_batch                                  (Eq 4)  EMA,α ∈ (0,1)
A_sum^(i,j) = Σ_k W_k^(i,j) = 1^T Σ̂_t^{-1/2} A^(i,j)              (Eq 5)
```

`T_warm` 预热期(实现取**前 5 步**)内不做白化,等协方差估计稳定。

**子空间白化**(异质任务的关键处理):单条 rollout 极少观测到全部 n 个 reward 维度。对活跃 reward 集合 `S ⊆ {1..n}`,只取主子阵 `Σ̂_S ∈ R^{|S|×|S|}` 单独求 `Σ̂_S^{-1/2}`,**避免给从不共现的维度造假相关**。最后再做 batch 归一化。

**诊断指标 η_eff = η_proj × η_corr(§2.2)**
```
η_proj(w) = cos²(w, 1) = (w^T 1)² / (n · ‖w‖²)      # 聚合方向与等权方向的对齐度
η_corr = n / (1^T |Σ_z| 1)                           # 求和后保留的独立信息量
两 reward 情形:η_corr = 2/(2+2|ρ|) = 1/(1+|ρ|)
```
关键洞察:裸 GRPO 直接求和 `Σ_k r_k = Σ_k μ_k + Σ_k σ_k z_k`,常数项被组归一化消去,故其**隐含权重是 `w_GRPO = (σ_1,…,σ_n)^T`** —— 方差大的 reward 自动获得更大权重,`η_proj(w_GRPO) = (Σσ_k)²/(n Σσ_k²)`。GDPO 的 z-score 使 `w_GDPO = 1`,完美对齐等权方向、消除方差缩放损失,但**在 prompt 层面仍不稳定**(二值/并列/离群导致 advantage 质量集中到少数 rollout),且**不处理 η_corr**。

### 超参默认值

β(压缩强度)、α(EMA 率)论文**未给具体数值**;`T_warm` = 前 5 步。Base 为 LongCat-Flash(MoE)与同族小模型。
四个任务混合的活跃 reward 子空间:数学 `{math, length}`、代码 `{code, length}`、指令跟随 `{ifeval, rubrics}`、通用写作 `{length, rm, rubrics}`。

**条件奖励门控(§3.2,值得单独抄的一条)**:合并多 reward 前先做门控 —— rubric reward <0.5 时 RM reward 截断为 `min(r_rubric, r_rm)`;IFEval <0.5 时 length reward 降权;主任务 reward <0.5 时 length reward 同样截断。**目的是防止辅助信号掩盖核心要求的失败**。

### 计算/显存开销

纯 reward 空间后处理(在算 advantage 之前),**零额外反传**;每组一次排序 O(G log G),每步每个活跃子空间一次 n×n 特征分解(他们 n≤3,可忽略);一个 n×n 的 EMA 协方差状态。
**论文未给任何 wall-time 数字**。

### 证据幅度(六篇里最弱)

技术报告,**无 seed、无误差棒、无显著性检验**。

小模型组件消融(Table 3):

| 方法 | IFEval | AIME24 | AH-Hard | **AH-Creative** | FullStack | HumanEval+ | MBPP+ |
|---|---|---|---|---|---|---|---|
| Base(GDPO) | 78.56 | 58.54 | 14.00 | 24.00 | 55.87 | 83.54 | 77.25 |
| Q(仅 MAQ) | 81.70 | 59.69 | 12.30 | **27.50** | 56.19 | 85.37 | 77.78 |
| M(仅白化) | 81.33 | 60.31 | 13.80 | 25.20 | 56.70 | 84.15 | 77.25 |
| **Q+M(RDPO)** | **83.55** | **60.42** | **14.40** | **21.50** | **56.91** | **85.37** | **78.51** |

**注意 AH-Creative 反向**:Init 是 30.10,所有方法都比它差,而完整 pipeline Q+M 掉到 **21.50**(最差)。即 MAQ+白化在创意写作上主动损害。

规模化(Table 4,LongCat-Flash)同样是混合结果:
- **赢**:IFEval 90.39(GRPO 89.46 / Init 86.14)、**AH-Hard 76.10(vs GRPO 65.80,幅度最大)**、AH-Creative 89.00、WritingBench 87.63、SOP-Maze 38.17、MBPP+ 79.63
- **输/平**:GuideBench(Init 87.81 > RDPO 87.04)、AIME2024(Init 86.3 > 85.73)、FullStack(GRPO 67.16 > 66.48)、HumanEval+(Init 92.68 > 91.46)、GPQA(GRPO 68.54 > 67.79)

Fig 1:`η_eff` 从 GDPO 的约 0.65 抬到 RDPO 的约 0.85–0.90 并稳定;Fig 3:四个子空间的平均 |Pearson r| 一致降低。

### 适用性判决

**不适用于当前设计**,详见 §必答问题 (a)。唯一值得立刻抄的是 §3.2 的**条件奖励门控**,以及 Fig 2 的两个组内诊断量(支配度、有效参与度)—— 后者对我们的 research 组特别有价值。

---

## 5. HPT(2509.04419)统一策略梯度估计器 + 混合后训练

### 算法精确公式

**统一策略梯度估计器 UPGE(Eq 6)**

```
grad_uni = E_{τ∼π_ref(·|q)}[ 1_stable(τ,q) · (1/π_ref(τ|q)) · Â_uni(τ,q) · ∇_θ π_θ(τ|q) ]
         = 1_stable · (1/π_ref) · Â · ∇π_θ
```

**四个可互换部件(§2.1)**:

| 部件 | 含义 | 各算法取法 |
|---|---|---|
| `1_stable` | 稳定化掩码 | PPO clipping 的推广;DAPO 改阈值、CISPO 做 token-wise、Clip-Cov 加协方差裁剪 |
| `π_ref` | 参考策略分母(token 级重加权系数) | SFT/REINFORCE 取 `π_θ`(无偏但高方差);PPO/GRPO 取 `π_θ_old`;offline 取 `≡1`;GSPO 取 `π_θ(π_θ_old/π_θ)^{1/\|τ\|}` |
| `Â` | advantage 估计 | 固定 ±1(REINFORCE、SFT 隐含 `Â≡1`);GRPO 组归一化;Dr.GRPO/RLOO 只做 recentering;token 级 `Â_{i,j,t}` |
| `∇π_θ` | 似然梯度 | **所有方法共享,不变** |

**推导起点(Eq 1–4),这是可借用性最高的部分**

```
J_μ(θ) = E_{τ∼π_θ(·|q)}[ r(τ|q) ] − μ·KL( π_β(·|q) ‖ π_θ(·|q) ),   μ ≥ 0        (Eq 1)
∇_θ J_μ(θ) = E_{τ∼π_θ}[ r(τ|q)·∇_θ log π_θ(τ|q) ] + μ·E_{τ∼π_β}[ ∇_θ log π_θ(τ|q) ]   (Eq 2)
Â_uni(τ,q) = r(τ|q)  +  μ · π_β(τ|q)/π_θ(τ|q)
             └ Â_RL ┘    └────── Â_SFT ──────┘                                    (Eq 4)
```

结论(§2.2 末):"SFT 和 RL 优化的是**同一个 Common Objective**,可以在一个 loss 内联合训练而**无内在冲突**",差别只在数据分布(`π_θ` vs `π_β`)与 bias-variance 取法。

**HPT 算法(Eq 7–13, Alg 1)**:对每个 question q 采 n 条 on-policy 轨迹,verifier `v(τ_i) = R(τ_i) ∈ {0,1}`,性能反馈

```
P = (1/n)·Σ_{i=1..n} v(τ_i)                                              (Eq 8)
α = f(P) = 1 if P > γ else 0 ;   β = g(P) = 1 if P ≤ γ else 0            (Eq 10)
L = α·L_RL + β·L_SFT                                                     (Eq 13)
```

**二值门控,同一步内不混合** —— P 高于阈值就纯 RL、低于就纯 SFT。设计原则(§2.4):`f` 应与 P 正相关(能力强时靠 RL 探索)、`g` 应与 P 负相关(能力弱时靠 SFT 给正确指引)。RL 用 Dr. GRPO(Eq 11,`A_{i,t} ≡ A_i` 组归一化),SFT 在监督轨迹 `τ*` 上(Eq 12)。

### 超参默认值

`γ = 0`(Qwen 全系,即"只有 8 条 rollout 全错时才做 SFT")/ `γ = 2/8`(LLaMA);lr 5e-6 **常数**,AdamW;8 rollouts,温度 1.0,max gen 8192 token;8×A800 80GB。
Base:Qwen2.5-Math-1.5B/7B、LLaMA3.1-8B。

### 计算/显存开销

`P` 就是 GRPO 本来就要算的 verifier 结果,**零额外成本、零额外反传**。且论文指出 HPT 比 SFT→GRPO **更省**:SFT 步不需要 rollout。

### 证据幅度

Table 2(Qwen2.5-Math-7B,in-dist 6 benchmark 平均 + OOD):

| 方法 | AIME24 | AIME25 | AMC | MATH-500 | Minerva | Olympiad | **Avg** | OOD Avg |
|---|---|---|---|---|---|---|---|---|
| SFT | 25.1 | 22.8 | 56.1 | 84.2 | 33.8 | 44.7 | 44.5 | 46.4 |
| GRPO | 19.4 | 13.8 | 59.1 | 81.8 | 38.2 | 46.2 | 43.1 | 58.8 |
| SFT→GRPO | 25.7 | 21.6 | 62.2 | 84.6 | 38.2 | 46.8 | 46.5 | 49.3 |
| LUFFY | 26.1 | 21.8 | 66.2 | 88.4 | 41.9 | 54.1 | 49.8 | 60.1 |
| SRFT | 18.4 | 15.5 | 55.9 | 83.8 | 42.6 | 48.9 | 44.2 | 58.7 |
| **HPT** | **33.0** | 21.9 | **69.4** | **89.2** | **46.0** | **56.9** | **52.7** | **62.3** |

跨模型(Table 3):LLaMA3.1-8B HPT 18.2 avg vs LUFFY 13.2 / GRPO 9.6 / SFT 5.9;**Qwen2.5-Math-1.5B HPT 41.9 vs LUFFY 34.7(+7.2pp)**。LUFFY 是**固定 1/8 混合比例**,所以这是全组"自适应 vs 固定比例"最干净的一个数。

**门控消融(Table 6,反直觉且对我们最重要)**:

| γ | AIME24 | AIME25 | AMC | MATH-500 | Minerva | Olympiad | Avg |
|---|---|---|---|---|---|---|---|
| **0** | 16.6 | **17.8** | **51.0** | **81.0** | **37.5** | **47.3** | **41.9** |
| 1/8 | **18.1** | 14.2 | 46.0 | 75.4 | 35.7 | 42.5 | 38.7 |
| 2/8 | 15.8 | 13.0 | 49.0 | 77.6 | 34.6 | 44.1 | 39.0 |

**引入更多 SFT 一致更差(−2.9 到 −3.2pp)**,而 γ=0 意味着"只有全错时才做 SFT",几乎等于纯 RL 加一个失败兜底。作者结论:"简单地增加 SFT 未必带来更好结果,关键是维持 RL exploration 与 SFT exploitation 的动态平衡。"

其他:
- Table 5:SFT/ON(HPT)41.9 > Mix/ON 40.3 > OFF/ON 38.1 —— **off-policy RL 不是必需的**
- Fig 2:Pass@k 到 k=1024,HPT 在**所有 k 上最高**,不仅 Pass@1 涨,探索边界也没被牺牲(反驳 Limit-of-RLVR)
- Table 4:MATH-500 双向独占解题,HPT 相对 GRPO +58/−15,相对 LUFFY +43/−16,且**红色增益随难度单调上升**(Level 5:+27/−8),绿色(遗忘)几乎不变 → 缓解灾难性遗忘
- Fig 6:offline data ratio 从约 0.25 快速降到 2–5% 并稳定;7B 比 1.5B 转折更早(强模型更早离开 SFT 主导期)
- Fig 7:HPT 熵始终高于 GRPO;响应长度快速上升后**不回退**,说明 long-form 推理被内化而非仅回声 teacher

### 适用性判决

**形式化直接可用于论文写作(见 §必答问题 (c));门控本体可作为 SFT 项的失败兜底。**

---

## 6. SRFT(2506.19767)熵感知权重

### 传闻公式核验:**属实,完全一致**

Eq 8:
```
w_SFT = 0.5 · stop_grad( exp( −H(π_θ) ) )
L_SFT^demo(θ) = w_SFT · E_{(x,y)∼D_demo}[ −log π_θ(y|x) ]
```

另有一个对称的、传闻里常被漏掉的第二个权重(Eq 12):
```
w_RL = 0.1 · stop_grad( exp( +H(π_θ) ) )
L_RL^self-rollout(θ) = w_RL · E_{x∼D, y⁺∼π_θ}[ −log π_θ(y⁺|x) ] + E_{x∼D, y⁻∼π_θ}[ log π_θ(y⁻|x) ]
```

**注意 w_RL 只加在正样本项上,负样本项不加权。**

机理(§4.2):二值 reward {1,−1} 下 on-policy RL 目标可分解为"正样本项 + 负样本项"(Eq 11),正样本项在结构上**等同于 SFT**,会快速降熵;用 `exp(H)` 加权使得熵一旦下降、该项自动减弱 —— 这是一个**熵崩塌的负反馈刹车**。
而 w_SFT 用 `exp(−H)`:熵高(策略不确定)时压低 SFT 影响,"缓解 demonstration 的行为策略与当前策略之间分布错配造成的性能退化"。

由于 H ≥ 0,`w_SFT ∈ (0, 0.5]`;实测 H ∈ [0, 0.5](Fig 6c),故 `w_RL ∈ [0.1, 0.165]`。

**总损失(Eq 13)**:`L_SRFT = L_SFT^demo + L_RL^demo + L_RL^self-rollout` —— 四部件,但只有两个带自适应权重。

**另外两个设计**:
- demonstration 被直接并入 on-policy rollout 组(Eq 6–7):`G_aug = G_roll ∪ G_demo`,advantage 在**增广组**上算 `Â_k = (r − mean{r|G_aug}) / std{r|G_aug}`。因 expert 响应 reward 更高,这**抬高整组 advantage**,论文称之为"promoting optimistic exploration"
- 行为策略取 `π_β ≡ 1`(Eq 10 的 IS 比率分母),并因此**省略 clipping**("π_β=1 时标准 clipping 会失衡且潜在不稳定")

### 超参默认值

Qwen2.5-Math-7B;OpenR1-Math-46k-8192(NuminaMath 1.5 子集,DeepSeek-R1 生成响应,Math-Verify 过滤);8 rollouts/prompt;max 8192 token;500 训练步;推理温度 0.6;验证器 Math-Verify(训练)/ OAT-Grader(评估)。

### 计算/显存开销

`H(π_θ)` 来自已有 logits,**零额外反传**,单个全局标量。省略 clipping 反而更省。

### 证据幅度

主结果 Table 2(Qwen2.5-Math-7B):SRFT in-dist avg **59.1**(AIME24 35.3 / AMC 74.3 / MATH500 89.8 / Minerva 39.7 / Olympiad 58.3),OOD avg **62.5**(ARC-C 85.3 / GPQA-D 46.4 / MMLU-Pro 55.9)。
对比:LUFFY 55.5 / 57.8,TAPO 56.4 / 55.9,ReLIFT 54.0 / 55.9,SFT→RL 54.5 / 54.6,RL_GRPO 49.3 / 49.7,SFT 54.3 / 49.2。
相对最佳 zero-RL 基线 **+9.0**(数学)/ **+10.9**(OOD);相对 SFT 方法 +4.8;相对 SFT+RL 方法 +3.4。

**固定 vs 自适应证据(Table 3),但严重混淆**:

| 配置 | AIME24 | AMC | MATH-500 | Minerva | Olympiad | Avg |
|---|---|---|---|---|---|---|
| **SRFT** | **35.3** | **72.2** | **89.8** | **39.7** | **58.3** | **59.1** |
| w/o w_SFT(置常数 1.0) | 30.1 | 65.8 | 87.0 | 36.8 | 55.8 | 55.1(**−4.0**) |
| w/o w_RL(置常数 1.0) | 32.6 | 67.2 | 87.5 | 37.4 | 56.5 | 56.2(**−2.9**) |

**批判**:自适应 `w_SFT` 上限是 0.5、`w_RL` 上限约 0.165,而"固定"基线一律取 1.0 —— 分别是 2× 和 6–10×。所以这 −4.0 / −2.9 里有相当部分只是"固定值取得太大",不是"自适应本身"的功劳。单一 base model,无 seed 方差报告。

训练动态(Fig 6):SRFT 熵稳定在 0.2–0.3 而纯 RL 快速崩到接近 0;SRFT 响应长度渐进上升(推理更详细),纯 RL 在难题上反而变短。

**前置分析部分(§3,与我们相关)**:
- Fig 2:SFT 对整条响应的 token 概率分布做**大幅全局改动**,RL 只对小部分 token 做**细粒度选择性修改**("Sledgehammer vs Scalpel")
- Table 1:RL→SFT 一致劣于 SFT→RL(37.4 vs 52.5 avg),即使加 KL 约束也救不回来;Fig 4b 给出熵解释 —— RL 后模型熵极低、**可塑性丧失**
- 局限性自陈:熵的利用"相对简单,只用了基本指数加权",未探索多时间尺度熵分析

### 适用性判决

**`w_RL` 的熵刹车值得移植到 OPD;`w_SFT` 方向对我们可疑。**

- 我们的 SFT 是"引入新动作的唯一通道"(已确认的能力边界结论),而 `exp(−H)` 随训练推进(H 下降)单调**上升** —— 训练后期反而加大模仿,方向与 CHORD/HPT/GAC(都是退火降 SFT)相反。SRFT 的自洽理由是防分布错配,但我们的场景更怕后期过拟合 demonstration。
- 相反,**`w = c·exp(H)` 形式的熵刹车对 OPD 项是高价值移植**:把它挂在 OPD 上(`w_opd,eff = w_opd · exp(H)/exp(H_0)`),熵一降 OPD 自动松手,正好对冲 moving self-teacher 的正反馈。

---

## 必答问题

### (a) research reward(0-1 成功率)与 exec reward(0/1 二值)组间各自 z-score 后,还需要 RDPO 式跨信号归一化吗?

**不需要,且不应该做。** 理由分三层:

**1. 结构不匹配 —— 这是决定性的。**
RDPO / GDPO / GRPO-sum 解决的全是"**同一条 rollout 收到 n 个 reward,必须求和成一个标量 advantage**"的问题。`η_corr = n/(1^T|Σ_z|1)`(RDPO §2.2)度量的正是**求和时的重复计数**;白化 `Σ̂^{-1/2}`(Eq 3)去的是**同一条 rollout 内各 reward 维度的相关**。
我们的 exec reward 与 research reward 打在**互不相交的两批 rollout** 上,从不相加,而是作为两个独立 PG 项各自带系数进入总 loss。每条 rollout 的 n=1,`η_corr ≡ 1`,白化的分母不存在。RDPO 的活跃子空间是 `{math,length}`、`{code,length}` 这种同轨迹共现,我们没有对应物。

**2. 组内归一化后真正剩下的危险不是"尺度",是"退化组"。**
二值 reward 组内 z-score 后 std 恒为 1(除非全对/全错组,此时 0/0)。真正危险的是**连续 [0,1] 的 research reward**:若一组四份文档的下游成功率是 0.41 / 0.42 / 0.43 / 0.44,组内 std ≈ 0.013,除以它会把纯噪声放大约 75×。这正是 HPT §2.3 转述的 Dr.GRPO 批评("除以标准差引入难度偏置,只做 recentering 就够了";Dr.GRPO / RLOO / REINFORCE++ 都持此立场)。

**具体建议**:
- RL_research **只做去均值、不除标准差**
- RL_exec 保持现有配置(二值下 z-score 无害)
- 两者共同加 `std < ε` 的**丢组规则**与 ε 下限
- 剩余的尺度差由 `w_res` 一次性吸收 —— 这是一个标量、可网格搜索,比 RDPO 整条 pipeline 简单一个量级,且规避了 RDPO 自己记录的副作用(Table 3 的 AH-Creative 30.10 → 21.50)

**3. 什么时候 RDPO 才变得相关。**
如果将来把 research reward 拆成**同一条 research 轨迹上的多个分量**(文档质量 + 下游成功率 + 长度),且分量间相关,那时**只移植 MAQ(Eq 1–2)** —— 它是纯组内变换,约 30 行,无协方差状态。白化(Eq 3–5)要 n≥3 且确实相关才划算,而他们自己的 Table 3 显示 M-alone 在 7 项指标里有 4 项弱于 Q-alone。

**现在就该抄的两件小事**:
- **RDPO §3.2 的条件奖励门控**:若 research 文档的下游成功率不高于 base policy 的成功率,则把该组的一切辅助 shaping 置零,防止辅助信号掩盖主目标失败
- **Fig 2 的两个组内诊断量**(advantage 支配度 `max_j p_j`、有效参与度 `1/(G Σ p_j²)`)直接算在 research 组上 —— research 组大概率组小、reward 密集,这两个数会最早暴露"整组更新被一条轨迹绑架"

### (b) GAC 与 CTWA 若只监控不控权,最小实现是什么

#### GAC 监控层(约 40 行,3–6 个 EMA 标量,零额外反传,实测 <1% wall-time)

| 量 | 公式来源 | 我们的取法 |
|---|---|---|
| `σ²_r,exec` / `σ²_r,res` | Eq 8,序列级归一化 advantage 的 batch 方差 | 两个 RL 项各算一个,`Ā_i` 已有 |
| `σ²_s,sft` / `σ²_s,opd` | Eq 9,长度归一化 NLL 的 trimmed 方差(上下各 10%) | SFT batch 用 NLL;OPD 用每样本长度归一化的蒸馏损失 |
| `Δĝ²(opd, rl_exec)` | Eq 10,z-normalize 每 token 损失系数后求逐 token 差的平方均值 | **这一对天然可算** —— OPD 打在 on-policy rollout 上,与 RL_exec 共享 token |
| `Δĝ²(sft, opd)` | 同上 | 当 OPD 也跑在 expert 序列上时可算 |

`Δĝ²` 只在两项共享 token 时有定义,这限制了可算的组合数,但恰好覆盖了我们最关心的一对。更新频率 `f_μ=10` 步,EMA β=0.99。GAC 附录 E 的"早期非平稳性"建议:前 100 步用 `f_μ=2`,之后恢复 10。

#### CTWA 监控层(约 30 行,M 个 EMA 标量)

需要一次**诚实的重新定义**,因为我们的四项不是同批 rollout 上的四个 reward:

- 在 **exec rollout 批**上,把 `w` 定义为该 completion 从**所有 loss 项合计**实际收到的更新权重(逐 token 求和其真实系数):
  `w(x, y^(k)) = (1/L)·Σ_l [ w_rl·A_k·ρ_{k,l}·1_{k,l} + w_opd·(OPD token 权重)_l + w_sft·1{demo token}_l ]`
- 对每个我们关心的**能力探针** m(任务成功率、tool-call 格式合法率、`look_at` 使用率、by-family 成功率),在同一批 completion 上算
  `c_m = E_{x in batch}[ Cov_{k=1..K}( r_m(x, y^(k)), w(x, y^(k)) ) ]`,EMA τ=0.1
- **只记录 `c̄_m` 与缺口 `δ_m = [c*_m − c̄_m]_+`,不做 `u_m ← u_m + η_λ δ_m` 的权重更新**

**这个改写忠于 Theorem 4.2**:定理讲的是 `Cov(r_m, s)` 的符号,`s` 是**任何诱导出更新方向的标量**,不必是"多个 reward 的求和"。参考目标值从 CTWA Table 1 起步:主目标 0.15、次目标 0.08(Fig 4b 显示 [0.08, 0.20] 稳健,**0.20 开始退化**)。

#### 必须同时记住的一条

CTWA 附录 A **Fig 5 证明:per-objective 梯度余弦相似度在 interference 发生时仍非负并收敛到 0**。它是假阴性,不能当判据 —— 而且算它需要 per-objective 反传(2× backward)。

**横向总结一句**:**这六篇方法无一需要第二次 backward;所有需要第二次 backward 的方法(MGDA / PCGrad / CAGrad / GradNorm / Nash-MTL)都在这个集合之外**,且在 CTWA Fig 1、GAC Table 3 中均非最优。这对我们 verl 栈是决定性的成本论据。

### (c) HPT 的形式化对我们论文写作的可借用点

**核心可借用点:把"四项 loss"重述为"同一个目标的四个估计器",而不是四个目标。**

直接套 UPGE 的四槽表格(仿其 Table 1):

| 项 | `π_ref` | `Â` | `1_stable` | 数据来源 |
|---|---|---|---|---|
| SFT | `π_θ` | `Â ≡ 1` | 1 | 离线 demo `D_sft` |
| OPD | `π_θ` | `Â_opd`(token 级,teacher/student 权重) | 1 | on-policy rollout |
| RL_exec | `π_θ_old` | `Â_GRPO`(二值 reward 组归一化) | `1_clip` | exec rollout |
| RL_research | `π_θ_old` | `Â_res`(连续下游成功率组归一化/去均值) | `1_clip` | research 会话 rollout |

**最有价值的一句可借论证**:HPT Eq 1–4 从单一目标 `J_μ(θ) = E_{τ∼π_θ}[r] − μ·KL(π_β‖π_θ)` 出发,梯度(Eq 2)天然分裂成"reward 项 + 数据贴合项",且 `Â_uni = Â_RL + μ·π_β/π_θ`。这直接回应审稿人最可能的质疑("四个损失不会互相打架吗"):**它们不是四个目标,是一个目标在不同数据分布下的不同偏差-方差取法。**

**两个必须诚实标注的扩展缺口**:

1. HPT 的推导只有**一个** `π_β` 和**一个** `r`。我们的干净推广是
   ```
   J = E_{τ∼π_θ, exec}[r_exec] + κ·E_{τ∼π_θ, res}[r_res]
       − μ_sft·KL(π_sft ‖ π_θ) − μ_opd·KL(π_teacher ‖ π_θ)
   ```
   两个 reward 通道 + 两个数据贴合项。

2. **这个形式化顺带给出 OPD 崩塌的一行形式陈述,是我们独有的贡献点**:
   当 teacher 是 moving self-teacher 时 `π_β = π_θ`,KL 项**恒等于 0**(目标值退化),但其梯度 —— Eq 2 的第二项 `μ·E_{τ∼π_β}[∇_θ log π_θ]` —— **并不为零**。即**目标已消失而更新仍在推进**。这正是"崩塌是目标的合法解"(CD-003 定论)的形式化说法,也直接解释了为什么冻结 teacher 是真修复:冻结使 `π_β ≠ π_θ`,KL 项重新良定义。
   HPT 本身没写这一点,但它完全落在其框架内,是我们可以拿走的推论。

---

## 横向矩阵:实现成本 × 证据强度

| 方法(可移植单元) | 实现成本 | 额外反传 | 证据强度 | 对我们四项的判决 |
|---|---|---|---|---|
| **CHORD `φ(p)=p(1−p)`** | 极低(~3 行) | 无 | 中高(ICLR26;φ 变体消融充分,Focal Loss 崩塌反证设计非任意) | **直接适用**,挂到 OPD 与 SFT |
| **SRFT `w·exp(H)` 熵刹车** | 极低(~5 行) | 无 | 中(消融被尺度混淆,单模型无 seed) | **值得移植**到 OPD;`exp(−H)` 方向对我们可疑 |
| **HPT UPGE 形式化** | 零(写作) | 无 | 高(跨 3 模型族/尺度,门控消融干净) | **写作直接借用** |
| **HPT 二值门控** | 极低(P 已有) | 无 | 高,但 Table 6 显示更多 SFT 更差 | 可作为 SFT 项的**失败兜底**,γ 从 0 起 |
| **GAC 三代理(仅监控)** | 低(~40 行) | 无 | 高(Table 8 破坏性检验 + Table 11/12 相关性验证,方法学最严谨) | **直接适用**,第一版就上 |
| **CTWA 协方差(仅监控)** | 低(~30 行) | 无 | 中(定理一阶;干预证据 Fig 4a 有,但无 run 内时间序列) | **改写后适用**,只记录不控权 |
| **RDPO 条件奖励门控** | 极低 | 无 | 低(技术报告,无 seed) | 值得抄这一条 |
| **CTWA 完整控制器** | 中(+10 行,但 `c*_m` 难定) | 无 | 中 | 缓,等监控数据说话 |
| **RDPO MAQ** | 中(~30 行,组内排序) | 无 | 低 | **暂不适用**(见 (a)) |
| **GAC 完整控制器** | 中高(~120 行;且**只有两路推导**) | 无 | 高但增益有上限(oracle 仅 +5.7pp) | 四项需重推 Theorem 1,**不建议第一版** |
| **RDPO Mahalanobis 白化** | 中(EMA 协方差 + 特征分解) | 无 | 低(AH-Creative 大幅退化) | **不适用** |
| (对照)MGDA / PCGrad / CAGrad / Nash-MTL | 高 | **需 2× backward** | 在 CTWA Fig 1 与 GAC Table 3 中均非最优 | **不适用**,开销不可接受 |

---

## 决策树:第一版固定系数 + 监控,什么阈值/症状触发升级到哪个方案

### 第一版定稿(有据可依,不是保守妥协)

- **四系数固定可配置,不上任何自适应控制器。** 依据:
  - GAC Table 7 的 oracle-μ 上限相对固定 μ **只有 +5.7pp**
  - HPT Table 6 更激进的自适应(γ 从 0 提到 1/8、2/8)反而 **−2.9 到 −3.2pp**
  - CHORD 附录 B.7 承认 token 级 φ 到位后 μ 调度不再必要,且 CHORD-φ(固定 μ=0.1)**反超** CHORD-μ(78.5 vs 77.6)
  - CHORD B.1 的 reward 感知自适应在 τ=0.7 时**跌破纯 RL 基线**(75.6 vs 77.1)
  - 我们项目的主效应量级是 −8 到 +28pp(CD-001 继承结论),权重自适应是**二阶旋钮**

- **但同时上三个 token 级/项级整形**(全部零额外反传,总成本 <1% wall-time):
  1. OPD 与 SFT 加 `φ(p) = p(1−p)`(CHORD Eq 5)
  2. OPD 额外乘熵刹车 `exp(H)/exp(H_0)`(SRFT Eq 12 形式)
  3. RL_research **只去均值不除标准差** + 退化组(`std < ε`)丢弃

- **监控三层**:
  - **Tier-0**(已有,零成本):by-family 成功率、**thought 逐字重复率**(我们已知最强判别指标,赢轨迹恒为 0)、policy 熵 H
  - **Tier-1**(GAC 三代理):`σ²_r,exec`、`σ²_r,res`、`σ²_s,sft`、`σ²_s,opd`、`Δĝ²(opd,rl_exec)`、`Δĝ²(sft,opd)`
  - **Tier-2**(CTWA 协方差):每个能力探针的 `c̄_m`
  - **Tier-3**(RDPO 组内诊断,仅 research 组):advantage 支配度、有效参与度、退化组率

### 预警的时间顺序(判断"先看到哪个")

```
σ_r² 尖峰(RL 噪声爆发)
   ↓ 约 40 步(GAC 附录 K 实测)
熵悬崖
   ↓
c̄_m 转负(一阶量,应领先指标下降)
   ↓
Δĝ² 上升而 reward 走平
   ↓
by-family 成功率下降 / thought 重复率上升(滞后指标)
```

### 升级触发条件

阈值来源已标注;未标注者为基于论文参数的外推,**需用我们前 200 步基线自校准**。

**症状 1 —— 某一 family 的 `c̄_m` 连续 ≥50 步为负,且该 family 成功率尚未下降**
→ CTWA 定律预言的退化前兆。**先不改控制器**,改**数据配比**(该 family 上采样)或临时下调与之竞争最强的那一项系数。
   理由:CTWA Fig 4a 只证明"负目标劣于正目标",没证明"闭环调权能救回来";CTWA 附录 B 自己说 Theorem 6.4 转不成实用策略。
→ 若数据配比无效且症状持续 → **升级到 CTWA 控制器**,仅对该目标开 `u_m ← u_m + η_λ·δ_m`(η_λ=0.05,τ=0.1,`c*` 从 0.08 起,**不要超过 0.20** —— Fig 4b 显示 0.20 开始退化)。

**症状 2 —— `σ²_r,res` 持续高于 `σ²_r,exec` 一个量级,或 research 组退化率(std<ε)>30%**
→ research 信号在用噪声主导更新。**先降 `w_res` 并增大 research 组大小 G**。
→ 若受算力限制无法增大 G → 上 **RDPO 组内诊断**确认是否"单条轨迹绑架整组"(`max_j p_j > 0.5`);若是 → **移植 MAQ**(仅 research 通道,β 从 1.0 起,`σ_global` 用全局 batch IQR)。

**症状 3 —— 熵在 <50 步内下降 >3×,或 `σ_r²` 出现 >2× 基线的尖峰**
→ 这是 GAC 附录 K/L 记录的固定系数标准死法(QCM 2/3 seed,step 280–320,熵 0.035→0.008,伴随 4.2pp 掉点)。
→ **立即降 OPD 与 SFT 的有效权重**;检查内建的 `exp(H)` 熵刹车是否生效;同时对照 thought 逐字重复率交叉验证。
→ 若熵刹车不足以稳住 → **升级到 GAC 控制器,但只在 {OPD, RL_exec} 这一对上开两路 μ**,其余两项保持固定 —— 这样规避了"四项需重推 Theorem 1"的问题。默认 β=0.99、c̄=0.01、λ=0.5、f_μ=10;把 `|Δμ|>0.02` 的步数占比当健康度看,GAC 认为 **3% 正常、11% 危险**(Table 8/9);c̄ 从 0.01 提到 0.02 会把大跳变从 3% 抬到 8%(§3.3)。

**症状 4 —— `Δĝ²(opd, rl_exec)` 持续上升而两项 reward 均走平**
→ 两项在同一批 token 上朝相反方向拉。**先查是不是 teacher 漂移**(我们已确认的病因),而不是先动权重。
→ 若 teacher 已冻结仍出现 → 与症状 3 同路,上两路 GAC。

**症状 5 —— SFT 引入的新动作在若干步后消失(能力回退),或零频动作(如 `look_at`)学不进来**
→ 这不是权重问题,是我们已确认的**能力边界**(SFT 是引入新动作的唯一通道)。
→ **上 HPT 门控**:对失败率高的 question 走 SFT、其余走 RL,γ **从 0 起**(Table 6:γ=0 → 41.9 明显优于 1/8 → 38.7 与 2/8 → 39.0,**不要把 γ 往上调**)。

**不触发升级的情况 —— by-family 成功率整体缓慢波动但 Tier-1/Tier-2 全部正常**
→ 改数据、改 teacher、改 prompt,**不要改权重**。六篇里没有任何一篇的证据支持"权重自适应能修复非权重来源的问题",而 CHORD B.1 与 CTWA 附录 B 都记录了自适应机制在超参没调好时反而跌破基线。

---

## 附:PDF 绝对路径

基路径 `/Users/shangwu/workspace/vibe_coding_projects/work/TRACE-OPD/.conductor/branches/CD-002/papers-mtl/`

- `2605.26184-GAC-NoiseAware-Mixing.pdf`(15 页,全文)
- `2508.11408-CHORD.pdf`(34 页,正文 1–16 + 附录 A/B 17–24)
- `2602.06869-Cross-Objective-Interference-CTWA.pdf`(30 页,正文 1–16 + 附录 C/D/E/F)
- `2605.13641-RDPO-MixedReward.pdf`(10 页,全文)
- `2509.04419-Unified-View-PostTraining-HPT.pdf`(22 页,正文 1–16 含全部消融)
- `2506.19767-SRFT.pdf`(22 页,正文 1–14 含消融)
