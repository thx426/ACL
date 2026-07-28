# 经典 MTL 优化器族正文级重审 —— 对 TRACE-OPD 双通道结构的逐个判决

日期 2026-07-28 · 分支 CD-002 · 精读 7 篇正文(方法+实验+开销),结论标注节/式编号。
PDF:`.conductor/branches/CD-002/papers-mtl/`;FAMO 下载至 `/Users/shangwu/workspace/vibe_coding_projects/work/refs/pdfs/2306.03792-FAMO.pdf`。

---

## 0. 结论摘要

1. **上一轮"用 CTWA 余弦非负否决整个梯度层"确属误伤,但修正后结论不变**:CTWA 的证据只证伪了方向手术类(PCGrad/CAGrad/MGDA);幅度均衡类(GradNorm/UW/FAMO)不受该证据约束,须另行判决。
2. **另行判决的结果仍是不采纳,但换了病因**。真正的否决理由不在文献里,在我们自己的栈里:`optimizer_step()` 强制梯度裁剪(`trace_opd/verl/verl/workers/engine/fsdp/transformer_impl.py:701` 的 `assert clip_grad is not None`),**只要某通道 pre-clip grad_norm > clip_grad,该通道的 loss 系数对实际施加的更新幅度完全无效**(裁剪把 w·g 重新缩放回范数 = clip_grad,w 精确抵消)。幅度均衡在我们这里已被裁剪器抢先执行。
3. **前提成立、疗法未验**:2510.19178 在 LLM RL post-training 实测通道间梯度失衡真实且严重(平方范数最大 33×,即 5.7× 有效学习率),且**无法用学习收益解释**(§3)——GradNorm 类方法的动机在我们这一档是成立的;但该文只做诊断,明确把"梯度层手术"列为 future work(§5),LLM RL 上无任何方法侧实证。
4. **真正可动的旋钮不是 `research_rl_weight`,是逐通道的 `clip_grad`**。若将来确需自动均衡,应调裁剪阈值而非 loss 系数——前者线性有效,后者在裁剪区饱和。
5. **GAC 的 5.7pp 上限论证覆盖不到幅度均衡类**(GAC 的 μ 是单批内 SFT/RL 的凸组合,总幅度守恒),但我们的 clip 论证覆盖得比它更彻底。7.3"固定+监控"裁定维持,依据升级。

---

## 1. GradNorm(1711.02257,ICML'18)

**算法(§3.1–3.2, Alg 1)**。共享权重子集 `W`(§3.1 定义,取**最后一层共享权重**,脚注 1:NYUv2 上训练时间仅 +~5%);`G_W^(i)(t)=‖∇_W w_i(t)L_i(t)‖₂`;`Ḡ_W(t)=E_task[G_W^(i)]`;逆训练速率 `L̃_i(t)=L_i(t)/L_i(0)`,相对逆训练速率 `r_i(t)=L̃_i/E_task[L̃_i]`。
目标(式 1):`G_W^(i)(t) ↦ Ḡ_W(t)×[r_i(t)]^α`。
梯度损失(式 2):`L_grad=Σ_i |G_W^(i)(t) − Ḡ_W(t)×[r_i(t)]^α|₁`,**目标项按常数处理(stop-grad),否则 w_i 会漂向零**;只对 w_i 求导;每步后重归一化使 `Σ_i w_i = T`(解耦全局学习率)。

**超参**:仅 α 一个。玩具对称任务 α=0.12;NYUv2 α=1.5;MTFL α=0.2–0.3。w_i 用 Adam,lr=0.025。补充材料 §7.1:0<α<3 几乎全部优于等权,α≥3 数值不稳;原文断言"任何形式的梯度均衡(即使在次优区)对多任务训练都是健康的"。

**开销**:**不需要额外 backward**。w_i 只出现在 `w_i·L_i` 里,`∂L_grad/∂w_i` 可由 `G_W^(i)` 解析得到;而 `G_W^(i)` 只在 `W` 这一层上计算,故成本 ≈ +5%。这是全族里最便宜的。

**效果幅度**(Table 1/2):NYUv2+seg,α=1.5 vs 等权:depth 0.944→0.925、seg 70.1→67.8、normals 0.192→0.174;NYUv2+kpts VGG:0.658→0.629 / 8.39→7.73 / 0.155→0.139(depth 测试误差 ~5%)。§5.3:100 次随机权重 grid search 仍不及 GradNorm 单次训练。

**对我们的两个致命点**。(a)`r_i` 依赖 `L_i(t)/L_i(0)` 的"损失越小学得越快"语义,GRPO 代理损失不满足(值可正可负、大小无意义)。(b)§7.2 记录了失效模式:MTFL 的 glasses/pose 任务卡在多数类分类器,GradNorm 会**持续抬高卡死任务的权重**,作者列为待解决问题。research 通道恰恰最可能卡住(研究行为不涨),风险同构。
**唯一可移植的部分**:令 **α=0** 时 `[r_i]^α≡1`,目标退化为纯范数对齐 `G_W^(i)→Ḡ_W`(§3.2 原文明言),**完全不需要损失比语义**——这是第 3 节最小实现的唯一合法出发点。

---

## 2. Uncertainty Weighting(1705.07115,CVPR'18)

**精确形式**。回归(式 7):`L = 1/(2σ₁²)·L₁ + 1/(2σ₂²)·L₂ + log σ₁σ₂`;分类(式 8 把 logits 除以 σ²,式 10 近似后):`L ≈ 1/(2σ₁²)·L₁ + 1/σ₂²·L₂ + log σ₁ + log σ₂`(**分类项系数是 1/σ² 不是 1/2σ²**)。实现上回归 `s := log σ²`(数值稳定、免除零)。σ 是普通可学习标量,随主 loss 同一次 backward 更新——**零额外反传**,比 GradNorm 还便宜。

**效果幅度**(Table 1, Tiny CityScapes):等权 3 任务 IoU 50.1 / inst 3.79 / depth 0.592;近似最优 grid 权重 (0.89,0.01,0.1) 62.8 / 3.61 / 0.549;3 任务 UW **63.4 / 3.50 / 0.522**——UW 优于细粒度 grid search 的最优点。训练末端学到的权重比为 43 : 1 : 0.16(seg : depth : instance)。

**对 RL loss 的理论障碍(硬)**。式 2/5 的推导前提是 `L_i` 为高斯/softmax **负对数似然**,`1/2σ²` 是观测噪声精度。GRPO 代理损失不是任何观测数据的 NLL,σ 无概率意义。更硬的是数值问题:对 `L=1/(2σ²)L₁+log σ` 求 σ 的驻点得 `σ²=L₁`;**`L₁<0` 时无驻点,σ→0 使目标 →−∞ 发散**。GRPO 代理损失均值近零且可正可负,直接套用会发散。原文自称"权重不会收敛到零"的良态性(§3.2 末)只在 `L_i≥0` 下成立。
**旁证**:GradNorm §4 与 Table 2/3 一致显示 UW 劣于 GradNorm(`w_i∼1/L_i` 无约束增长,推高全局学习率);FAMO Table 1/2 中 UW 在 NYU-v2(Δm% +4.05)与 QM-9(+108.0)明显劣于 FAMO,仅在 CityScapes(+5.89)胜出。

---

## 3. PCGrad(2001.06782,NeurIPS'20)

**算法(§2.3, Alg 1)**:`g_i^PC ← g_i − (g_i^PC·g_j)/‖g_j‖²·g_j`,**仅当 `g_i^PC·g_j < 0`**;随机顺序遍历 j≠i;更新 `Δθ=Σ_i g_i^PC`。
**理论(§2.4)**:Thm 1 凸情形收敛到最优或 cos φ₁₂=−1;Thm 2 给出 PCGrad 优于普通多任务梯度的充分条件,**条件 (a) 是 `cos φ₁₂ ≤ −Φ(g₁,g₂) < 0`**——严格要求负余弦。
**开销**:需 K 次 backward 得到 K 份完整梯度向量,O(K) 空间;投影本身只是点积(§3:预计算后不再反传)。

**判决(完整论证,不反驳)**。CTWA 2602.06869 Fig 5 在 Qwen2.5-1.5B-Base / 1.5B-IFT / Qwen3-1.7B-Base 三个模型上实测 per-objective 梯度两两余弦"**mostly non-negative 且随训练收敛到 0**",无冲突行为。代入 PCGrad 定义:余弦 ≥0 时算法**按定义不修改任何梯度**(§2.3 原文"when gradients do not conflict, we do not change the gradients"),即**恒等映射**;Thm 2 条件 (a) 同时被证伪,连"偶尔有用"的理论保证也不剩。因此这不是"假设不成立、效果打折",而是**付满 K 份梯度的存储代价换一个 no-op**。
补充负面证据:GCond §III 在两个数据集上实测 PCGrad/GradDrop/CAGrad **均劣于不做任何处理的 Baseline**;PCGrad 论文自身 §5.2 只在 MetaWorld MT10/MT50 这类任务间真冲突的 RL 上有效。**不采纳。**

---

## 4. CAGrad(2110.14048,NeurIPS'21)

**算法(§3.1, Alg 1)**:`max_d min_i ⟨g_i,d⟩ s.t. ‖d−g₀‖ ≤ c‖g₀‖`(式 3),`g₀` 为平均梯度。对偶只需解 K 维 `w* = argmin_{w∈W} g_w^T g₀ + √φ‖g_w‖`(`φ=c²‖g₀‖²`),更新 `θ ← θ − α(g₀ + φ^{1/2}/‖g_w‖·g_w)`。c=0 退化为 GD,c→∞ 退化为 MGDA。
**理论(Thm 3.2)**:`0≤c<1` 时收敛到 `L₀` 的驻点,`Σ‖∇L₀‖² ≤ 2(L₀(θ₀)−L₀*)/(α(1−c²))`。
**开销(§3.3)**:仍需 K 次反传;提出子采样 S⊆[K],复杂度 O(|S|N+T),K<10 时 T≪N。
**效果**:NYU-v2 Table 1 Δm% CAGrad 0.20 vs PCGrad 3.97 / MTAN 5.59;c 消融(附录表)c=0.2→1.55、0.4→0.20、0.6→−1.36、0.8→−2.29。

**判决**。CAGrad 比 PCGrad 稳健(它有 `g₀` 锚,余弦非负时退化为普通 GD 而非 no-op,不会白做),但同样只在有冲突时产生增量——CTWA 证据下增量≈0。且 c 是新的敏感超参(0.2→0.8 跨越 3.8pp Δm%),需要与 α 同量级的调参预算。**不采纳**,理由是"收益期望为零",非"有害"。

---

## 5. MGDA-Decoupled(2604.20685)

**改进点(§2.1)**。原始 MGDA(式 2)求凸包最小范数向量,对梯度尺度过敏,会过度偏袒小范数(已收敛)目标;MGDA-Normalised(式 4,GAPO 采用)把梯度归一化到单位长度,但**丢弃全部幅度信息**,把已收敛与未收敛目标同等对待。本文 MGDA-Decoupled(式 5)**解耦系数计算与聚合**:用 `∇L_i/L_i` 求系数 `c_i`,再把 `c_i` 施加到**原始**梯度上。式 6 给出 k=2 闭式:`‖∇L_i‖/L_i` 越小的目标得到越大系数;其倒数 `L_i/‖∇L_i‖` 正是 Polyak 步长,作为"距最优点距离"的一阶代理。

**LLM 实验(§3–4)**:Gemma-2-2b-it 与 Qwen2.5-0.5B-Instruct,UltraFeedback 4 目标,**DPO 不是 RL**,GPT-4o 判分 net win rate。Gemma:Decoupled +2.3%*(唯一显著)vs Uniform +0.9%;Qwen-0.5B:Decoupled +4.7%* vs Normalised +4.1%* vs **Uniform +4.0%***——**对固定等权的增量仅 0.7pp,且未做 Decoupled-vs-Uniform 显著性检验**。
**开销**:k 份 per-objective 梯度 + Gram 矩阵解 QP;作者称 k=4 时 QP 相对前反传可忽略(§4 末),但 k 次反传的成本被略过未计。

**判决**。三个不可移植点:(a)作者自陈成功前提是**目标函数同构**——四个目标用同一 DPO 公式,故 `L_i` 是"距收敛距离"的可校准代理,并明确对比"一般 MTL 中 MSE vs CE 这类异构损失"(§4);我们恰是异构的最坏情形(OPD KL 蒸馏 / GRPO 代理 / SFT 交叉熵);(b)作者指出 DPO 的 KL 项"**人为对齐了梯度**"、β 大时更甚,削弱冲突密度——我们的 OPD/KL 锚同机理,预期冲突更少;(c)增量 0.7pp 对固定权重,与 GAC 的 5.7pp oracle 上限同量级偏小。**不采纳。**

---

## 6. GCond(2509.07252)

**做法(§II.A–D)**。两阶段:估计阶段 K 步把 N 个 loss 的梯度累积进 N 个独立 buffer(式 1),`Var(ĝ_i)=Var(g_i)/K`(式 2);解析阶段对 N 个平均梯度做仲裁。仲裁不是硬开关而是把余弦 c 经分段函数(式 4,阈值 θ_weak/θ_main/θ_crit + remap_power p)映到有效冲突角 α_eff,再由 `s_w=sin(α_eff)`(式 5)、`s_l=sin(min(α_eff,π/2))`(式 6)决定 winner/loser 的投影强度;winner 由 `Score_i = w_stab·max(0,S_i) + w_str·N_i`(式 7–9,S=与上一步梯度的余弦,N=范数/其 EMA)选出,并有 dominance 窗口防止单任务长期获胜。

**2× 加速的适用条件(§III)**:该 2× 是 **GCond 随机模式 vs GCond 顺序模式**的自比,不是对 baseline 的加速。成立前提是**有效批足够大**(文中 256×24=6144),使稀疏采样的梯度仍能给出可靠的冲突估计;随机模式把 K 步切成 N 个 K/N 块,每任务只用 1/N 的数据。与 Baseline 比只是"达到 Baseline 吞吐"。

**可迁移性(强负面)**。实验全在视觉:MobileNetV3-Small / ConvNeXt-tiny / ConvNeXtV2-Base(~89M),任务是 MIM 重建 + 分类,无 LLM、无 RL、无 transformer LM。两个可直接搬用的经验事实反而更有价值:
- **PCGrad/CAGrad 在 ConvNeXtV2-Base + 16GB 上 batch size=1 都跑不起来**(需 retain_graph 且逐任务梯度向量,显存随任务数线性增长),GCond 用 `torch.func.functional_call` + 即时释放计算图 + bf16 buffer 才能跑到 70 张图(§III)。
- **§II.E:GCond 与 AdamW 的"集成式"(GCond 做投影+EMA,AdamW 只做 RMS 归一化、β₁=0)显著优于"分离式"**;原因是分离式下 AdamW 的动量去平滑一个已被修正但非平稳的梯度序列,导致方差高估、过早停滞。这是**梯度层方法与 Adam 矩估计相互干扰**的直接实证,见第 8 节。

**判决:不采纳**(非 LLM/非 RL、依赖 torch.func 与 verl+FSDP 的兼容性未验、O(N) buffer),但其 AdamW 交互结论与 PCGrad/CAGrad 显存证据**采信并引用**。

---

## 7. FAMO(2306.03792,NeurIPS'23)

**算法(Alg 1)**。定义改进率 `r_i(α,d)=(ℓ_{i,t}−ℓ_{i,t+1})/ℓ_{i,t}`(式 3),求最差改进率最大化(式 4/5),取对偶得 `z* = argmin_{z∈S_k} ½‖J_t z‖²`,`J_t` 行为 `∇log ℓ_i`(式 6,即 log 域的 MGDA);Prop 3.2:z* 为内点时各任务改进率相等。
**O(1) 的来源(§3.2)**:不每步求解式 6,只做一步梯度下降 `∇_z = J_t^T J_t z`,再用式 9 的关键近似 `(1/α)[log ℓ_{i,t} − log ℓ_{i,t+1}] ≈ J_t^T J_t z`——**用相邻两步的对数损失差替代逐任务梯度**,故无需存任何 per-task 梯度。
**实现(§3.3)**:`z=Softmax(ξ)`;重归一化系数 `c_t=(Σ z_i/ℓ_i)^{-1}`(式 11)防 ℓ 小时爆炸;衰减 `ξ_{t+1}=ξ_t−β(δ_t+γξ_t)`(式 12);Adam(lr α=0.025, weight_decay=γ)。**超参 γ∈{1e-4,1e-3,1e-2},默认 1e-3**;§5.4 消融显示 γ 选错代价明显(CityScapes 因某任务初始 loss 近 0 而敏感),作者在 §6 把 γ 依赖列为唯一 limitation。
**硬前提**:`ℓ_i ≥ ε > 0`,需已知每任务 loss 下界 `ℓ_i*`(Alg 1 输入 `min_losses`,伪代码 Alg 2 用 `losses - min_losses + 1e-8`)。

**效果与开销**:NYU-v2 Δm% −4.10(vs LS +5.59、UW +4.05、CAGrad +0.20、NashMTL −4.04);QM-9 58.5(vs LS 177.6);CelebA(40 任务)4.72;**MTRL MetaWorld MT10 成功率 0.83±0.05,与 CAGrad 持平,优于 PCGrad 0.72、LS 0.49、UW 0.77,低于 STL 0.90**。Fig 4:相对 LS 开销可忽略;附录 B 说明 CAGrad/NashMTL 若要同样摊销需 3 次前反传。

**判决**。FAMO 是本族唯一同时满足"O(1) 空间时间 + MTRL 有效"的方法,是幅度均衡类里最值得考虑的候选。但两个前提在我们这里不成立:(a)全部机制建立在 `ℓ_i>0` 且"loss 下降 = 进步",GRPO 代理损失两者皆不满足,且需要 `ℓ_i*` 下界;(b)其 MTRL 实证是 SAC + 10 个奖励函数不同的机器人任务,任务间真冲突,与 CTWA 观察到的弱耦合 LLM 场景不同构。**不采纳**;若将来 research 通道改为可监控的正值代理量(如 research reward 的 regret),FAMO 是第一顺位复评对象。

---

## 8. 三分类判决与双通道成本重估

| 类别 | 成员 | 单模型原始开销 | 我们双通道下的成本变化 |
|---|---|---|---|
| **方向手术类** | PCGrad, CAGrad, MGDA-Decoupled, GCond | K 次 backward + O(K) 完整梯度副本 | **由"贵"变"中等"**:执行/research 两次 backward 本已发生;FSDP 下每卡多存一份 4B/8 卡 ≈ 2GB(fp32 grad);投影所需点积只是 1 个标量 all-reduce。真实代价是**必须推迟两次 optimizer_step 到两份梯度都在手**,破坏现有"算完即 step"结构。**但收益期望≈0(CTWA)**,故成本重估不改判决。<br>注:执行批内部三项之间做手术仍是"贵"——需把 1 次 backward 拆成 3 次。 |
| **幅度均衡类** | GradNorm(α=0 变体), FAMO | GradNorm ≈ +5%(仅末层范数);FAMO O(1) | **由"便宜"变"免费"**:`optimizer_step()` 已返回 pre-clip 全局 grad_norm(`trace_opd/verl/verl/workers/engine/fsdp/transformer_impl.py:699,740`),FSDP 分片间已 all-reduce,连末层近似都不用做。**这是本次重审确认的唯一真实成本下降。** |
| **权重学习类** | Uncertainty Weighting, GroupDRO | 零额外反传(σ/c 随主 loss 更新) | 成本不变(本来就免费);但 UW 在 `L<0` 时发散、GroupDRO 需可比的 `L_i` 量纲,两者都卡在 RL loss 的语义上,与成本无关。 |

**跨类的两条共性风险(文献未覆盖、我们栈特有)**:
- **梯度裁剪抢先均衡**。`clip_grad_norm_(c)` 在 `N>c` 时把梯度整体缩放为 `c/N`;把 loss 乘 w 后 `g→wg, N→wN`,施加梯度 `= wg·c/(wN) = g·c/N`,**w 精确抵消**。即 `research_rl_weight` 只在 research 通道 pre-clip 范数**低于** `clip_grad` 时线性有效,越过阈值后硬饱和。任何以"调 loss 系数"实现的幅度均衡器都会在裁剪区变成空转,并可能积分饱和(windup)。
- **Adam 矩估计稀释**。Adam 的 `m/√v` 对梯度的全局常数缩放严格不变;两通道各自 step 但共享 Adam state 时,w 只改变 research 梯度在共享 m、v 中的相对占比,响应是两端饱和的 S 形而非线性。GCond §II.E 的"集成 vs 分离"实证同向。GradNorm §5.2 自称"对优化器不敏感"不构成反证——它是单次 backward 内改变合成梯度的**方向**,Adam 不会把方向归一化掉。

---

## 9. 核心问题:双通道 GradNorm 式自动均衡的最小实现

**唯一合法形式**:GradNorm α=0(§3.2 明言 α=0 即纯范数对齐),因为 `r_i` 所需的 `L_i(t)/L_i(0)` 语义在 GRPO 上不成立。目标是把 research 通道的更新幅度控到执行通道的固定比例 ρ。

```python
# 每个全局 step 内,两次 update 各自的 optimizer_step() 返回 pre-clip grad_norm
# 注:verl 中每个 mini-batch 一次 step,取该通道本 step 内所有 mini-batch 的均值
n_exec = mean(grad_norms_from_exec_update)        # 免费,无额外反传
n_res  = mean(grad_norms_from_research_update)    # 免费

ema_e = beta * ema_e + (1 - beta) * n_exec        # beta = 0.95,同 2510.19178 App C.1
ema_r = beta * ema_r + (1 - beta) * n_res

# 【前置门】裁剪区检测:越过阈值则 w 无效,不得调节
if ema_r > clip_grad * 0.8:
    log("research channel saturated by grad clip; w is a no-op")
    freeze(w)                                      # 或改调 clip_grad_research,见下
else:
    err   = log(ema_r / (rho * ema_e))             # rho = 目标范数比,如 0.1
    log_w = log_w - eta * err                      # eta ≈ 0.02,乘性控制器
    log_w = clip(log_w, log(w0 / 4), log(w0 * 4))  # 硬护栏:相对初值不超过 4×
    w     = exp(log_w)
    if step % update_every != 0:                   # update_every = 10,同 GAC 的 f_mu
        w = w_prev                                 # 降频,避免与 RL 噪声共振
```

**为什么是乘性控制器而非 GradNorm 的 L1 目标**:两任务时式 2 退化为单个标量误差,乘性形式在对数域是线性的,天然给出 GradNorm 的"重归一化"效果(此处由 `rho` 固定比例替代 `Σw_i=T`)。

**预期收益**。参照各文的"均衡 vs 不均衡"差幅:GradNorm NYUv2 相对等权约 2–5% 相对误差改善(Table 1/2);UW 在**极端失衡**时 IoU 50.1→63.4(Table 1)——但那是三个量纲差数十倍的损失被等权相加,我们的执行/research 已分批独立归一,不在该区;MGDA-Decoupled 在 LLM(DPO)上相对 Uniform 仅 +0.7~1.4pp;FAMO 在 MTRL 上与 CAGrad 持平。**综合估计:在我们结构上,自动均衡相对一个调对了的固定 `research_rl_weight`,期望收益 < 1pp,且被上节两条饱和机制进一步压缩。**

**风险**(按严重度):① 裁剪区空转 + 积分饱和(上节);② GradNorm §7.2 的卡死任务病理——research 通道若长期不涨,均衡器会持续抬权直至污染执行通道,与 CD-003 已定论的 moving self-teacher 正反馈同型;③ 与 OPD 系数、KL 系数构成多环耦合,故障归因成本剧增;④ 引入 ρ、η、beta、update_every 四个新超参去省掉一个 w。

**默认与升级门槛**。默认维持"固定 + 监控";**只有同时满足以下三条才升级**:
- (a) `ema_r / ema_e` 在连续 ≥100 step 内偏离设定值 **> 3×** 或 **< 1/3**(单次尖峰不算);
- (b) 同期 research 通道 pre-clip `grad_norm` **持续低于 `clip_grad`**(否则 w 是 no-op,升级无意义,应改调 `clip_grad_research`);
- (c) research 侧行为指标(按 CD-006 逐步轨迹协议)与执行侧指标出现**反向**移动——即确证是幅度问题而非数据/奖励问题。
任一不满足,先查数据与奖励口径,不动系数机制。

---

## 10. 与 §7.3 固定系数裁定的关系:GAC 的 5.7pp 是否覆盖幅度均衡

**GAC 的 μ 语义(2605.26184)**:`L = (1−μ)·L_GRPO + μ·L_SFT−φ`——**同一批数据上两个 loss 项的凸组合**,单次 backward。μ* 由最小化 `E(μ)=E‖ĝ(μ)−g*‖²` 得到,`E(μ)=(μ−α_tgt)²Δg² + μ²σ_s² + (1−μ)²σ_r²`(式 2),闭式解式 3;Δg²→0 时退化为逆方差加权,Δg²→∞ 时 μ*→α_tgt。Table 7:QCM(fixed μ)62.1±0.8 → Full GAC 67.2±0.4 → **Oracle μ 上界 67.8±0.3**,故"oracle 相对固定 μ 的可得空间 = 5.7pp"。

**覆盖性判定:部分覆盖,不完全覆盖。**
- **覆盖的部分**:μ 在 (0,1) 上遍历时,两项的相对幅度比 `μ‖ĝ_s‖ : (1−μ)‖ĝ_r‖` 从 0 扫到 ∞,**相对幅度轴被完整覆盖**。所以对"执行批内部三项(OPD/GRPO/SFT)的相对幅度是否值得自适应",GAC 的 5.7pp 上界**是适用的**——那正是同批凸组合的同构情形。
- **不覆盖的部分**:凸组合下 `‖ĝ(μ)‖ ≤ max(‖ĝ_s‖,‖ĝ_r‖)`,**总幅度守恒**,μ 无法改变有效步长。而 `research_rl_weight` 缩放的是一个**独立 batch、独立 backward、独立 optimizer_step** 的通道,它同时改变(i)相对幅度与(ii)该次更新的绝对步长——(ii) 是 GAC 的 μ 构造不到的自由度,5.7pp 未对其取上界。
- **但这个未覆盖的自由度在我们这里恰好被两个机制吃掉**:梯度裁剪把绝对步长钉死在 `clip_grad`(第 8 节),Adam 又对全局缩放不变。**结论:GAC 的 5.7pp 覆盖不到幅度均衡类,但我们不需要它来覆盖——clip + Adam 的论证更强,且是可直接在日志上验证的**(看 `grad_norm` 相对 `clip_grad` 的位置即可)。7.3 裁定维持,论据由"文献显示自适应空间小"升级为"我们栈里该旋钮在主要工作区内近似失效"。

---

## 11. 修正后的采纳/不采纳总表

| 方法 | 类别 | 原判 | 修正后判决 | 决定性依据 |
|---|---|---|---|---|
| PCGrad | 方向手术 | 不采纳(CTWA) | **不采纳(理由加强)** | 余弦 ≥0 时按定义恒等映射(§2.3)且 Thm 2 条件 (a) 失效;GCond §III 实测劣于 Baseline |
| CAGrad | 方向手术 | 不采纳(CTWA) | **不采纳(理由改为收益≈0)** | 有 g₀ 锚故无害,但 CTWA 下增量≈0;新增敏感超参 c(Δm% 跨 3.8pp) |
| MGDA-Decoupled | 方向手术 | 未单独评 | **不采纳** | 前提是目标同构(§4 作者自陈),我们异构;LLM 上相对 Uniform 仅 +0.7~1.4pp 且未检验显著性;DPO 非 RL |
| GCond | 方向手术 | 未单独评 | **不采纳,但采信其两条实证** | 纯视觉 ≤89M;2× 是自比且需超大有效批;采信 PCGrad/CAGrad 显存实证与 §II.E 的 AdamW 交互结论 |
| Uncertainty Weighting | 权重学习 | 被误伤否决 | **不采纳(理由更换)** | 非 CTWA 问题;是 `L<0` 时 σ→0 发散 + 代理损失非 NLL(式 7/10 推导前提) |
| GradNorm(完整版) | 幅度均衡 | 被误伤否决 | **不采纳(理由更换)** | 非 CTWA 问题;`r_i` 需 `L_i(t)/L_i(0)` 语义 + §7.2 卡死任务权重发散病理 |
| **GradNorm α=0(纯范数对齐)** | 幅度均衡 | — | **条件性备选,默认不启用** | 成本从"便宜"降为"免费"(grad_norm 已返回);但受 clip 饱和 + Adam 稀释双重压制,期望 <1pp;升级门槛见 §9 |
| FAMO | 幅度均衡 | 被误伤否决 | **不采纳(理由更换),列第一顺位复评** | 非 CTWA 问题;需 `ℓ_i>0` 与下界 `ℓ_i*`;MTRL 实证(MT10 0.83)在真冲突任务上,与我们弱耦合场景不同构 |
| 固定系数 + 监控 | — | 裁定 | **维持,依据升级** | 从"GAC oracle 仅 5.7pp"升级为"clip + Adam 使幅度旋钮在主要工作区内近似失效";真正可动的是逐通道 `clip_grad` |

**一条落地建议(超出原问题但成本为零)**:两通道的 `grad_norm` 已经免费返回,**无论是否启用均衡器,都应把 `n_exec`、`n_res`、`n_res/n_exec` 三条曲线接入 swanlab**,并在图上标出 `clip_grad` 水平线。这是 §9 三条升级门槛的唯一数据来源,也是判断"w 是否还是个有效旋钮"的唯一依据。
