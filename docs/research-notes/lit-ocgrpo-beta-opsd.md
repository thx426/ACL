# 文献对照:OC-GRPO 与 β-OPSD 如何处理特权引导的分布失配(2026-08-06)

> 语境:用户问"per_decision 前缀积 IS 在 B 大时不可用,这两篇怎么解决的"。
> 两篇均全文精读(arXiv HTML 版)。结论先行:**两篇都没有用无截断的
> per-decision 前缀积——一篇把 token 级校正塞进 PPO clip 截断,一篇通过
> 从学生自身采样让 IS 整体消失。**

## 1. OC-GRPO(arXiv:2607.19313,Meta AI,2026-07)

**结构:与 PGS 高度同构,特权进 prompt 端。**
hard 题(pass@64=0)在训练时给 solution prefix 等特权引导 g(x),rollout 从
π_old(·|g(x)) 采;目标 objective 是无引导的 J(θ)=E_{π_θ(·|x)}[r]。

**校正(核心答案):token 级双上下文比,塞进 PPO clip 内。**

    ρ^oc_t = π_θ(y_t|x,y_<t) / π_old(y_t|g(x),y_<t)
           = r_t(标准 GRPO drift) × γ_t(off-context correction)
    γ_t = π_old(y_t|x,y_<t) / π_old(y_t|g(x),y_<t)

ρ^oc 直接替换 GRPO surrogate 里的 ratio 位置,被 clip(1±ε) 整体截断。
**没有前缀积**:序列级 IS 按 token 分解后逐位独立进 clip——与我们的
per_token 同类,同样不校正"到达该状态的前缀测度"(状态分布偏差同在)。
论文 §5 的序列级校正因子 λ+=p/q(压成功 credit)、λ−=(1−p)/(1−q)
(放大失败罚)只是**期望层的解释定理,不是实现**。

**方差控制(App E.2,Thm 4)**:γ_t ∈ e^{±nη}(n=guidance token 数,
η=单 guidance token 对 next-token logprob 的扰动界),Var ≤ e^{2nη}−1
——**随 guidance 长度指数增长,与 rollout 长度无关** ⇒ 设计原则
"用能恢复学习信号的最短 guidance"。对照:我们的 W_beh 界 e^{±2ηc}
与 doc 长度无关(倾斜经 kernel 截断,doc 只经 p_T 影响方向)。

**实验要点**(Qwen2.5-7B/3B/1.5B LoRA,595 hard MATH 题):
- 未校正的 guided 基线(POPE*/BREAD*/PrefixRL*)在 1.5B 跌破 vanilla
  GRPO(PrefixRL* −7.2%),3B 也有跌;OC-GRPO 各尺度均正
  (+13.8%/+7.2%/+10.2% 相对)——**back-generalization 容量依赖,
  小模型吸收不了失配**(直接支持我们 v3.1 分臂的动机③)。
- P@16 几乎不动甚至跌:特权引导主要锐化 P@1,不太扩能力边界
  (与 [[opd-capability-boundary]] 一致)。
- guidance 是 5 级 solution prefix,训练前按"最小可产成功级"一次性选定
  (Fixed 优于每步自适应的 Adaptive)。

**与 PGS 的三点差异**:① 他们 γ 塞 clip 内(方差控制唯一手段;spec
NF02 拒绝此形态的理由——clip 会在最需校正处零梯度——他们用 clip 换稳定);
我们 W 自带硬界故可放 clip 外保持校正力度。② 他们无门/预算/保尾核,
全 token 都"被干预"(整个 context 不同);我们干预面由 D 门+B 控制。
③ 他们逐 token 校正比需在**原 prompt 下重新前向**(双 context 两次
forward);我们的 W 两个量在采样时锁定,零额外训练前向。

## 2. β-OPSD(arXiv:2607.28582,Maryland,2026-07)

**结构:特权进 loss 端(蒸馏目标),不进采样端——L1 路线的强化版。**
把 vanilla OPSD(序列 reverse-KL 到特权 teacher)重写为 KL 正则 RL:
reward=log(p_T/π_ref),β 加权 KL(π_θ‖π_ref);β=1 恰为 OPSD。
闭式最优解 = 几何插值 π*_β ∝ π_ref^{1−1/β}·p_T^{1/β}。

**实现:不做 RL、不做 IS——回到蒸馏。** 逐前缀把 reference/teacher
两组 **logits 线性插值** softmax 得局部目标 p̃_β;**从 π_θ 自身
on-policy 采样**,最小化 D_KL(π_θ‖p̃_β)。采样分布=学生自身 ⇒
**零分布失配 ⇒ 无需任何重要性采样**。1/β_k 线性 schedule
(先贴 reference 后贴 teacher 的分布课程,方向与我们讨论的
"教师退场退火"相反——他们冷启动保护,我们毕业撤出;可相容为钟形)。

**Return-to-go 不是 IS**:token 权重 G_t=Σ_{s≥t}γ^{s−t}ρ_s
(未来失配折扣和,stop-grad),γ=1 时是序列 reverse-KL 的无偏梯度。
它是 credit assignment(本 token 对未来失配负责),与测度校正无关。
**优美的对偶**:PDIS 前缀积 = 过去向乘积(测度:怎么到这的);
return-to-go = 未来向求和(功劳:这一步影响什么)。β-OPSD 靠 on-policy
采样把测度问题归零只剩 credit;我们/OC-GRPO 靠环境 reward 把 credit
交给 advantage 只剩测度。两族方法各解半边。

## 3. 对 PGS 的启示(记账,不改变 v3.1 计划)

1. **组内 λ+ 降权(v3.2 候选)**:OC-GRPO 的序列级校正 λ+=p/q 需要
   估计 p,q——**我们的分臂恰好免费给出组内 p̂(bare 4 条)与 q̂
   (steered 4 条)**。对 steered 成功轨迹的轨迹级 advantage 乘
   λ̂+=p̂/q̂(或其平滑),即用他们的 Lemma 1 原则性地修掉分臂已知的
   "credit 泄漏"次级效应。等 v3.1 跑出 δ 数据后评估。
2. **截断前缀积(Retrace 式 Π min(c̄,w))是两篇都没占的中间档**,
   若消融显示状态分布偏差主导且需要保留 steered 梯度,这是
   per_token 与不可用的裸 per_decision 之间的正解;我们的 vtrace
   模式(逐点 min(1,w),无连乘)不是它。
3. **定位确认**:OC-GRPO 用比我们更弱的控制(纯 clip,无门/预算/
   核截断)都在三个尺度拿到正收益;PGS 的 token 级细粒度干预 +
   有界 W + 分臂 bare 锚在该文献坐标系中站得住,且分臂的在线 δ
   读数是两篇都没有的。竞品重合度最高的是 OC-GRPO(同为
   "采样端特权 + IS 校正回原目标"),论文写作时需正面引用并对比:
   granularity(prompt 级 vs token 级)、控制面(clip vs 门/预算/核)、
   代理(math 单轮 vs 多轮 agentic)。
