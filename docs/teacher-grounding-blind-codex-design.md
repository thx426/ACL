> **双盲独立设计存档**(codex rescue:gpt-5.6-sol,effort=max(--effort 未传,落 config.toml),任务 task-msbj0h26-u0nfkr)。
> 防火墙:fresh 线程 + 空目录 blind_design_room + 自足任务书(仅含系统客观背景、won 膨胀实测表、用户原始想法逐字转述与目标约束),禁读任何文件,零现行方案泄露。
> 下方正文为 codex 输出逐字原样(27974 字符);与 CD-009 定稿 tg 方案的双盲对照与处置见 teacher-grounding-design.md(待入档)。

# TRACE-OPD：成功条件化的特权教师校准方案

以下设计完全基于任务书中的信息，未访问任何文件、代码仓库或外部资料。

## 0. 结论摘要

用户的直觉是对的：教师不能永远作为无梯度、自动正确的目标。尤其在 student 前缀上，`teacher(doc, h_student)` 并不是一个经过结果监督的“轨迹修复器”。

但不能简单地在成功轨迹上把原 KL 反向传播到 teacher。若直接最小化

\[
D_{\mathrm{KL}}\!\left(p_T^\theta(\cdot\mid d,h)\,\|\,\operatorname{sg}p_S(\cdot\mid h)\right),
\]

teacher 会被拉向无特权 student，可能通过降熵、模式收缩或忽略 doc 来降低 KL，最终抹掉特权能力。

建议采用：

> 成功条件化前缀校准 OPD，Verified Prefix-Calibrated OPD，简称 VPC-OPD。

核心是一个交替过程：

1. 用结果验证过的成功续写校准 teacher：
   \[
   D_{\mathrm{KL}}(Q^+_{\text{success}}\|p_T^\theta)
   \]
   而不是简单的 \(D_{\mathrm{KL}}(p_T\|p_S)\)。

2. \(Q^+_{\text{success}}\) 来自三类数据：
   - 成功 student 轨迹的后缀；
   - 从失败/偏航 student 前缀出发，由带 doc teacher 实际执行并成功的续写；
   - teacher 从头执行成功的轨迹，用于保留特权能力。

3. teacher 校准完成后冻结一个快照，再做 student-side OPD；只蒸馏经过成功续写验证的前缀，且以 action、结束 reasoning 的结构 token 为主，reasoning 文本只给很弱权重。

4. 使用固定健康期的 reasoning 结束概率、成功轨迹长度和 teacher 原生成功能力作为锚，避免“当前模型生成得越长，下一轮就越把长文本当成正样本”的自举循环。

这个方案保留“单份共享权重”的设定，但明确承认：teacher-side 更新也会溢出影响 student，student-side 更新也会改变 teacher。因此必须交替更新、使用冻结目标和 KL trust region，不能把两个 KL 同时无约束地堆在一起。

---

# 1. 病理机理分析

## 1.1 记号

设：

\[
p_S^\theta(y\mid h)=\pi_\theta(y\mid h)
\]

是裸 student，

\[
p_T^\theta(y\mid d,h)=\pi_\theta(y\mid d,h)
\]

是注入 doc 的 teacher。

\(h\) 包括任务、历史 observation、历史 reasoning/action，以及当前 assistant turn 内已经生成的 token。

现有 OPD 近似为：

\[
\mathcal L_{\mathrm{OPD}}
=
\mathbb E_{\tau\sim P_S^{\theta_b}}
\left[
\sum_t
D_{\mathrm{KL}}
\left(
\operatorname{sg}p_T^{\theta_b}(\cdot\mid d,h_t)
\,\Vert\,
p_S^\theta(\cdot\mid h_t)
\right)
\right],
\]

其中 \(\theta_b\) 是本轮 rollout/打分时的行为策略快照，`sg` 表示目标侧停止梯度。

## 1.2 teacher 自走成功不等于 student-prefix oracle

teacher 的高自走成功率只说明：

\[
h\sim P_T(\cdot\mid d)
\]

时它表现较好。

OPD 实际查询的是：

\[
h\sim P_S(\cdot),
\qquad
p_T(\cdot\mid d,h).
\]

这是另一个分布。student 前缀可能包含：

- teacher 自己不会走到的动作历史；
- 与 doc 冲突的错误推断；
- 大量自我说服式 reasoning；
- teacher 低概率甚至分布外的语言状态。

语言模型在这种 off-policy 前缀上的条件分布没有经过“能否修复环境轨迹”的训练。doc 让它知道目标答案，却不自动赋予它从任意错误历史恢复的能力。常见结果不是立即选择修复动作，而是先在文本上解释、调和或合理化冲突。

因此：

\[
\text{高 } SR(T\text{ from start})
\centernot\implies
\text{高 } SR(T\text{ continuation from }h_S).
\]

后者才是 OPD 所需要的教师质量。

## 1.3 特权信息产生不可蒸馏的条件信息

令 \(D\) 为 doc，\(H\) 为 student 可见历史，\(Y\) 为下一 token。

teacher 可以利用：

\[
I(Y;D\mid H)>0
\]

的信息。例如它可以断言一个尚未观察到的物体位置。student 在同一前缀下看不到 \(D\)，因此不可能逐任务复现所有 doc-specific token。

若固定理想 teacher 分布 \(q_T(y\mid h,d)\)，裸 student 在 forward KL 下的最优解是：

\[
p_S^*(y\mid h)
=
\mathbb E_{D\mid h}
\left[q_T(y\mid h,D)\right].
\]

最低可达损失对应 teacher token 对 doc 的剩余条件信息；在适当联合分布下，可解释为条件互信息项。也就是说，存在不可消除的“信息鸿沟”。

问题在于，reasoning 文本是承载这种 doc-specific 信息的巨大自由通道。teacher 可以生成：

- 具体但 student 无依据的断言；
- 反复解释为什么某个行动“必然正确”；
- 多种语义相近、措辞不同的反思；
- 对 doc 中已知路线的回顾式说明。

这些内容对 teacher 很自然，但裸 student 无法从 \(H\) 推导。它只能学到跨任务共有的表面形式，例如“重新分析”“这是唯一合理步骤”“我必须再去一次”。

因此真正被蒸馏的可能是特权推理的修辞外壳，而不是其信息来源。

## 1.4 reasoning 是低约束、高容量的 loss hacking 通道

环境只对 action 产生状态转移；reasoning 文本基本不受环境约束。它具有三个特点：

1. token 数远多于 action；
2. 可以无限产生语义近似但逐字不同的内容；
3. 终局奖励不进入当前损失，无法惩罚无效思考。

从优化角度看，改变语言风格、推理转折和自我确认，通常比改变少量关键 action token 更容易降低 KL。于是共享参数优先吸收 teacher 的文本风格，而非稀疏的决策差异。

更严重的是，student 控制下一轮训练数据的 occupancy。虽然通常不对 rollout 采样路径反向传播，但每轮更新后重新采样会形成慢时间尺度反馈：

\[
\theta_k
\rightarrow
P_S^{\theta_k}(h)
\rightarrow
\mathcal L_{\mathrm{OPD}}
\rightarrow
\theta_{k+1}.
\]

延长 reasoning 会让模型停留在“语言内部状态”，推迟进入 action 决策和新 observation。语言内部状态往往是 teacher/student 更容易局部一致的区域；真正的 action 决策反而具有最大的 doc-induced 分歧。因此系统可能逐渐进入“延迟决策”的低局部阻力区域。

这是一种 occupancy hacking：不是某一次梯度显式奖励长度，而是策略改变了以后被训练的状态分布。

## 1.5 reasoning 结束 hazard 的乘法效应

定义在 reasoning 内部开始输出 `</reasoning>` 的概率为：

\[
c(h)=
P_\theta(\text{开始关闭 reasoning}\mid h).
\]

在近似平稳条件下，reasoning 的期望长度与 \(1/c\) 同阶。即使每个位置只把关闭概率从 0.02 降到 0.01，期望长度也会近似翻倍。

forward KL 在大量位置上把 student 拉向 teacher 的“继续展开、补充、转折”概率，却只有极少数关闭标签。因此很小的局部 logit 偏移可以产生巨大的序列长度变化。

长度增加后，下一轮又得到更多 reasoning 前缀。如果损失按 token 求和，长轨迹贡献更多梯度；即使按 token 平均，训练 occupancy 仍更偏向 reasoning 状态。两者都可能形成正反馈。

## 1.6 共享权重使 stop-gradient teacher 仍然漂移

单步中 teacher logits 虽停止梯度，但参数是共享的。设 student/teacher logits 的 Jacobian 分别为 \(J_S,J_T\)。

student-side KL 的局部更新近似为：

\[
\Delta\theta
\approx
\eta J_S^\top(p_T-p_S).
\]

于是 teacher logits 同时发生：

\[
\Delta z_T
\approx
\eta J_TJ_S^\top(p_T-p_S).
\]

因此 teacher 不是一个静止靶标。若共享参数方向同时增强 doc-conditioned 叙述和裸 student 的叙述倾向，就会出现：

1. teacher 本轮偏好更长的解释；
2. student 被拉长；
3. student 更新又间接改变 teacher；
4. 下一轮 teacher 在更长、更异常的 student 前缀上继续打分；
5. teacher/student 一起进入长 reasoning 吸引域。

“teacher 侧 stop-gradient”只阻断单个计算图，并不阻断跨 step 的共享权重反馈。

## 1.7 对表格现象的统一解释

### 成功轨迹也膨胀

成功只证明动作序列最终完成任务，不证明 reasoning 有用、真实或最小。终局奖励无法给 reasoning 内容做信用分配，因此成功轨迹完全可以携带大量无因果作用的叙述。

### 环境步数下降但 token 增加

两者并不矛盾：

- 每个环境步之前的 reasoning 变得极长；
- 固定上下文预算导致能够执行的环境步减少；
- 在晚期成功样本中，只有动作路径较短、任务较容易的 episode 还能在上下文耗尽前成功，产生明显的成功条件选择偏差。

因此“成功样本环境步变少”不一定表示策略更高效，也可能表示长 reasoning 已经淘汰了需要更多环境交互的任务。

### reasoning/action 比率暴涨

action 的语法和长度由环境约束，扩张空间很小；reasoning 没有外部约束，所以所有长度漂移几乎都进入 reasoning。

### 相邻 turn 逐字重复率为 0

病理不是 token-level 复读，而是语义层面的反复论证。大模型可以不断生成新措辞：

- 改写同一结论；
- 增加新的过渡句；
- 更换容器罗列顺序；
- 重新构造“唯一合理步骤”的论证。

因此 exact-match 重复率为零不能排除语义循环。相反，它说明模型仍保持流畅生成能力，但决策信息密度不断下降。

---

# 2. 教师侧损失的精确设计

## 2.1 不选择“原 KL 直接反传到 teacher”

若成功轨迹上使用：

\[
D_{\mathrm{KL}}
\left(
p_T^\theta
\Vert
\operatorname{sg}p_S
\right)
\]

并更新 teacher，目标是让 teacher 接近完整 student 分布，而不是接近成功续写分布。它有三个问题：

- student 分布中仍含大量未采样的错误 token；
- 最优共识 \(p_T=p_S\) 会消除 doc 优势；
- 对第一参数求梯度包含 teacher 熵项，中间过程可能通过模式收缩降低损失。

若改为：

\[
D_{\mathrm{KL}}
\left(
\operatorname{sg}p_S
\Vert
p_T^\theta
\right),
\]

优化更稳定，但仍是在模仿未经过 outcome conditioning 的裸 student 分布。

因此推荐的 target 不是 \(p_S\)，而是：

> 在给定前缀下，经过环境结果验证的成功续写分布 \(Q^+\)。

方向选择为：

\[
D_{\mathrm{KL}}(Q^+\Vert p_T^\theta).
\]

即始终把“经过验证的目标”放在 KL 第一项，把“被更新的模型”放在第二项。这等价于最大似然，避免 teacher 通过自身降熵逃避目标。

## 2.2 成功条件化目标分布

对任务 \(i\)，冻结本轮行为策略 \(\theta_b\)。student 轨迹分布为 \(P_S^{\theta_b}(\tau\mid x_i)\)。

定义：

\[
Q_{i,k}^{S+}(\tau)
=
\frac{
P_S^{\theta_b}(\tau\mid x_i)
\mathbf 1[R(\tau)=1]
g(\tau)w(\tau)
}{
Z_{i,k}
}.
\]

其中：

- \(g(\tau)\in\{0,1\}\)：质量硬门控；
- \(w(\tau)\)：效率、回放可信度等软权重；
- \(Z_{i,k}\)：任务内归一化常数。

理想 teacher 校准目标是：

\[
\mathcal L_{T,S+}^{\rm seq}
=
\mathbb E_i
D_{\mathrm{KL}}
\left(
Q_{i,k}^{S+}(\tau)
\Vert
P_T^\theta(\tau\mid d_i,x_i)
\right).
\]

环境转移概率与 \(\theta\) 无关，因此与 teacher 有关的部分分解为：

\[
\mathcal L_{T,S+}^{\rm seq}
=
-\mathbb E_{\tau\sim Q^{S+}}
\sum_{t\in\text{assistant tokens}}
\log p_T^\theta(y_t\mid d,h_t)
+\text{const}.
\]

这正是“在 teacher 条件视图上，对成功 student 轨迹做 teacher-forced 最大似然”，但其正确解释是拟合成功条件化轨迹分布，而不是匹配整个 student 分布。

## 2.3 实际采用任务—轨迹—turn 平衡

直接对整条序列求和会让长 reasoning 再次拥有更多权重。实际使用重加权的成功 prefix occupancy：

\[
\ell_T(\tau)
=
\frac{1}{U_\tau}
\sum_{u=1}^{U_\tau}
\left[
\frac{\lambda_A}{|A_u|}
\sum_{t\in A_u}
-\log p_T^\theta(y_t\mid d,h_t)
+
\frac{\lambda_F}{|F_u|}
\sum_{t\in F_u}
-\log p_T^\theta(y_t\mid d,h_t)
+
m_u\frac{\lambda_R}{|G_u|}
\sum_{t\in G_u}
-\log p_T^\theta(y_t\mid d,h_t)
\right].
\]

其中：

- \(A_u\)：action 内容 token；
- \(F_u\)：`<reasoning>`、`</reasoning>`、`<action>` 等结构 token；
- \(G_u\)：reasoning 内容 token；
- \(m_u\)：该 reasoning 是否允许作为正标签；
- 每个任务先归一化成功 rollout 权重，再对任务求平均。

这样：

- 每个任务权重相同，容易任务不会因成功样本多而垄断训练；
- 每个 turn 权重相同，长 turn 不会自然获得更多梯度；
- action、格式和 reasoning 可以独立调节。

建议初始相对权重：

\[
\lambda_A=1,\qquad
\lambda_F=2,\qquad
\lambda_R=0.1.
\]

对于超过健康期固定阈值的 verbose reasoning，令 \(m_u=0\)。其 action 仍可作为成功标签，但 reasoning 文本不作为“正确推理”学习。

这些数值是起始点，不应被视为已知最优值。

## 2.4 reasoning 结束 hazard 损失

仅依赖序列 CE 对结束 token 的监督过稀。建立健康缓冲区 \(\mathcal B_{\rm safe}\)，来源为：

- 训练前/最早健康 checkpoint 的成功轨迹；
- 后续经过成功、长度和格式验证的紧凑轨迹。

令：

\[
c_\theta(h)=
P_\theta(\text{下一 token 开始关闭 reasoning}\mid h).
\]

对健康 reasoning，设结束位置为 \(n^*\)，加入：

\[
\mathcal L_{\rm close}
=
-\mathbb E
\left[
\log c_\theta(h_{n^*})
+
\frac{1}{n^*-1}
\sum_{n<n^*}\log(1-c_\theta(h_n))
\right].
\]

若关闭标签由多 token 构成，可用有限状态机定义“开始关闭”的首 token 概率，并在后续 token 上监督完整 delimiter。

还可把词表投影到宏类别：

\[
\Gamma(p)=
\big(
P(\text{continue reasoning}),
P(\text{close reasoning}),
P(\text{invalid transition})
\big),
\]

对固定健康快照加入：

\[
D_{\mathrm{KL}}
\left(
\Gamma(p_{\rm safe})
\Vert
\Gamma(p_\theta)
\right).
\]

这个损失只约束 reasoning 的状态转换，不要求复制健康模型的具体文本内容。

## 2.5 从 student 前缀出发的成功 teacher 续写

只训练成功 student 轨迹有一个根本缺陷：它教 teacher“不要破坏已经成功的路径”，却不能教它如何从失败路径恢复。

完整方案必须增加：

\[
Q_{i,h}^{C+}(\xi)
=
P_{T}^{\theta_b}
\left(
\xi\mid d_i,h,R(h\oplus\xi)=1
\right),
\]

即从 student 的真实环境前缀 \(h\) 出发，由冻结 teacher 采样多个 continuation，实际执行后仅保留成功续写。

相应损失：

\[
\mathcal L_{T,C+}
=
\mathbb E_{h}
D_{\mathrm{KL}}
\left(
Q_{i,h}^{C+}
\Vert
P_T^\theta(\cdot\mid d_i,h)
\right).
\]

工程上它仍是对成功 continuation 的 teacher-forced CE，不是 policy-gradient。

前缀应选在完整环境 observation 后、下一 assistant turn 开始前，而不是任意 reasoning 中间位置。建议固定分层抽样：

- episode 初期；
- 约 \(1/3\) 和 \(2/3\) 位置；
- invalid action 或明显无进展之后；
- teacher/student action KL 特别高的位置。

若环境可确定性回放，可通过重放 student action 前缀恢复状态；否则必须使用环境 snapshot。若不能可靠恢复状态，这一部分不能伪造，应退化为 success-only 版本，并明确其只能校准成功路径，不能声称学会恢复。

## 2.6 保留 teacher 的特权能力

增加第三类数据：

\[
Q_i^{N+}(\tau)
=
P_T^{\theta_b}
\left(
\tau\mid d_i,x_i,R(\tau)=1
\right),
\]

即 teacher 从头自走并成功的原生轨迹。

其作用不是证明 student-prefix 指导质量，而是保护 doc-conditioned 能力：

\[
\mathcal L_{T,N+}
=
D_{\mathrm{KL}}
\left(
Q^{N+}
\Vert
P_T^\theta
\right).
\]

同时维护一个“最佳安全 teacher”快照 \(\theta_{\rm anchor}\)。只有当它在固定 native rollout 与 prefix-continuation 评测上都不退化、且长度健康时才更新。

在该 teacher 已被证明成功的原生前缀上加入：

\[
\mathcal L_{T,\rm keep}
=
\mathbb E_{h\sim \mathcal B_N}
D_{\mathrm{KL}}
\left(
\Gamma_{AF}p_T^{\theta_{\rm anchor}}(\cdot\mid d,h)
\Vert
\Gamma_{AF}p_T^\theta(\cdot\mid d,h)
\right),
\]

其中 \(\Gamma_{AF}\) 只保留 action、结构和“其他”宏类别，避免把旧 teacher 的长篇 reasoning 逐字冻结。

关键点是：retention 只锚定 teacher 原生成功区域，不能在所有 student 前缀上锚定旧 teacher，否则会把当前病理一起保留。

## 2.7 控制共享权重对 student 的溢出

teacher-side 更新虽然从 doc 分支前向，但仍改变共享 \(\theta\)。因此在 teacher 校准阶段加入裸 student trust region：

\[
\mathcal L_{S,\rm anchor}
=
\mathbb E_{h\sim\mathcal B_S}
D_{\mathrm{KL}}
\left(
\Gamma p_S^{\theta_b}(\cdot\mid h)
\Vert
\Gamma p_S^\theta(\cdot\mid h)
\right).
\]

更推荐把它实现为约束：

\[
\mathbb E D_{\mathrm{KL}}
\left(
\Gamma p_S^{\theta_b}\Vert\Gamma p_S^\theta
\right)
\le \epsilon_S,
\]

而不是依赖一个固定 penalty 系数。

初始可试：

\[
\epsilon_S=0.01\sim0.02
\]

nats/加权 token/teacher 子阶段。该区间只是工程起点，必须先通过短 pilot 校准。

## 2.8 teacher 阶段总损失

建议起始形式：

\[
\begin{aligned}
\mathcal L_T
={}&
1.0\,\mathcal L_{T,S+}
+
1.0\,\mathcal L_{T,C+}
+
0.25\,\mathcal L_{T,N+}\\
&+
0.5\,\mathcal L_{\rm close}
+
\lambda_{\rm keep}\mathcal L_{T,\rm keep}
+
\lambda_{\rm share}\mathcal L_{S,\rm anchor}.
\end{aligned}
\]

其中：

- 前三项的当前模型梯度都只经过 teacher/doc 前向；
- 成功轨迹、成功选择、奖励、行为策略和 anchor target 全部停止梯度；
- \(\mathcal L_{S,\rm anchor}\) 是有意加入的共享权重保护项；
- \(\lambda_{\rm keep},\lambda_{\rm share}\) 最好通过 KL budget 的对偶变量自适应，而不是长期固定。

可选增加“历史 reasoning 扰动”：

- 保持 task、action、observation、doc 不变；
- 随机将一部分过去 reasoning 替换为占位符；
- 仍监督相同的已验证成功下一 action。

因为旧 reasoning 不改变环境状态，这能降低 teacher 对 student 自我说服文本的依赖。但 reasoning 可能承担记忆功能，所以只应对 action token 使用、低比例启用，不能强制完全不变性。

---

# 3. 与 student-side OPD 的组合

## 3.1 必须交替，不能同时使用未校准 teacher

若同一 minibatch 同时计算 teacher 校准和 student OPD，student 看到的仍是校准前 teacher logits。正确顺序是：

1. 冻结行为策略 \(\theta_b\) 并收集数据；
2. 更新 teacher/doc 视图，得到校准参数；
3. 冻结校准 teacher 快照 \(\theta_c\)；
4. 用 \(p_T^{\theta_c}\) 作为 student 的无梯度目标；
5. 更新裸 student 视图。

这形成：

\[
\theta_k
\xrightarrow{\text{teacher calibration}}
\theta_c
\xrightarrow{\text{student projection}}
\theta_{k+1}.
\]

## 3.2 可信前缀门控

student OPD 不再覆盖所有 student 前缀。可信集合 \(\mathcal M\) 包含：

1. 已验证成功 student 轨迹上的前缀；
2. 从该前缀存在成功 teacher continuation 的前缀；
3. 上述成功 continuation 内部的 token 前缀。

以下前缀不做 OPD：

- failed trajectory 上没有成功 teacher continuation 的位置；
- teacher 只从头成功、但没有在该 student 状态验证过的位置；
- 已撞上下文、格式异常或环境状态无法可靠重放的位置。

可以按 teacher continuation 的实际成功率赋权。若从某前缀采样 \(M\) 条，成功 \(s\) 条，可使用：

\[
w_{\rm conf}(h)=\frac{s+1}{M+2}
\]

作为简单 Beta 平滑置信度，或使用 Wilson/Beta 下置信界做硬门控。样本很少时不应宣称 teacher “显著优于” student。

## 3.3 student 目标

冻结校准 teacher：

\[
q_t(\cdot)
=
\operatorname{sg}
p_T^{\theta_c}(\cdot\mid d,h_t).
\]

student 损失：

\[
\mathcal L_{S,\rm OPD}
=
\mathbb E_{h_t\sim\mathcal M}
w_{\rm conf}(h_t)
\sum_{c\in\{A,F,R\}}
\lambda_c
D_{\mathrm{KL}}
\left(
q_{t,c}\Vert p_{S,c}^\theta
\right).
\]

建议仍采用：

\[
\lambda_A=1,\qquad
\lambda_F=2,\qquad
\lambda_R=0.05\sim0.1.
\]

reasoning 只在紧凑、成功、格式正常的续写中启用；对 verbose 或未验证 reasoning，\(\lambda_R=0\)。

同时加入 student 行为 trust region：

\[
\mathcal L_{S,\rm beh}
=
D_{\mathrm{KL}}
\left(
\Gamma p_S^{\theta_b}
\Vert
\Gamma p_S^\theta
\right).
\]

最终 student 阶段：

\[
\mathcal L_S
=
\lambda_{\rm OPD}\mathcal L_{S,\rm OPD}
+
\lambda_{\rm beh}\mathcal L_{S,\rm beh}
+
\lambda_{\rm close}\mathcal L_{\rm close}^{S}.
\]

## 3.4 top-64 与尾部处理

不能把 teacher top-64 简单重归一化后当作完整分布。这样会：

- 人为放大 top-64；
- 丢失尾部质量；
- 若关闭 delimiter 或合法 action 不在 top-64，结构信号会消失。

构造支持集：

\[
U=
\operatorname{Top64}(p_T)
\cup
\operatorname{Top64}(p_S)
\cup
\{\text{实际标签、delimiter、合法 action 关键 token}\}.
\]

定义聚合分布：

\[
\widetilde p_U
=
\left(
\{p(v)\}_{v\in U},
1-\sum_{v\in U}p(v)
\right),
\]

最后一项为 `OTHER` 桶。使用：

\[
D_{\mathrm{KL}}
\left(
\widetilde p_{T,U}
\Vert
\widetilde p_{S,U}
\right).
\]

这相当于在词表聚合后的 KL，是完整 KL 的数据处理下界，但不会凭空丢掉 tail mass。

对 teacher-side 成功 CE，实际成功 token 必须显式查询其 logprob；不能因为它不在 top-64 就丢弃该样本。

---

# 4. 数据选择与标签质量

## 4.1 什么算成功轨迹

硬条件：

- 环境明确返回终局成功；
- 未依赖上下文截断；
- action 能被正常解析；
- 没有协议漏洞或非法环境状态；
- 若环境存在随机性，成功应通过 snapshot/replay 或重复执行确认。

对于确定性环境，成功轨迹的每个决策前缀都有一个已知成功后缀，因此可作为 teacher 校准数据。

## 4.2 成功不等于每个 token 都正确

终局成功只支持以下结论：

- 整体 action 序列是一个可行解；
- 从其中每个前缀存在一个成功后缀。

它不支持：

- 每段 reasoning 都真实；
- 每个中间动作都必要；
- 该路径是最短或最优；
- 无依据的位置断言是可迁移知识。

因此：

- 所有合格成功轨迹可贡献 action 标签；
- reasoning 只作为低权重、经过长度门控的标签；
- 不能把失败轨迹中的所有 token 都当负样本；
- 不能用“成功”给长篇 reasoning 整体背书。

## 4.3 选择高质量成功路径

同一任务 8 条 rollout 中若有多条成功路径，优先使用 action 路径的 Pareto 前沿：

- 成功；
- 环境步数较少；
- 无 invalid/no-op action；
- reasoning 长度未超过固定健康期阈值。

建议：

- action 标签：使用 best action steps \(+2\) 范围内的所有成功路径；
- reasoning 标签：只使用健康期成功 reasoning 长度分布的固定 p95 以下样本；
- 阈值固定于训练前健康期，不能随当前模型滚动上升，否则阈值本身会跟着病理漂移。

## 4.4 失败轨迹如何使用

失败轨迹不直接提供正/负 token 标签，只用于：

- 提取需要恢复的 student 前缀；
- 评估 teacher continuation success；
- 监控 invalid action、循环和上下文耗尽；
- 构造 teacher/student 差异分层。

若从失败前缀采样 teacher continuation 后得到成功，成功 continuation 才进入 \(Q^{C+}\)。

若没有任何成功 continuation，则跳过该前缀。不能因为 teacher“看起来合理”就将其作为指导。

## 4.5 何时引入 teacher-side 训练

推荐调度：

### Phase 0：健康基线

用初始 checkpoint 收集固定指标和缓冲区：

- student 成功轨迹长度；
- reasoning/turn 分布；
- close hazard；
- teacher native success；
- teacher 在固定 student 前缀上的 continuation success。

### Phase 1：teacher 预校准

先进行 1–2 个 teacher-only 校准周期，同时用 student trust region 限制裸视图漂移。此时：

\[
\lambda_{\rm OPD}=0.
\]

只有当 teacher prefix-continuation 指标不恶化、reasoning hazard 健康时才进入 Phase 2。

### Phase 2：交替更新

每个训练 step：

- teacher 校准 1 个受 KL budget 限制的子阶段；
- 冻结 calibrated teacher；
- student OPD 1 个子阶段。

\(\lambda_{\rm OPD}\) 可在四个合格更新中按：

\[
0.25,\;0.5,\;0.75,\;1.0
\]

逐步升高。

若当前 batch 没有成功 student 轨迹，也没有成功 teacher continuation：

- 不生成伪标签；
- 可从固定成功 replay buffer 进行有限 teacher 保持训练；
- 本轮跳过新的 OPD，或只做 trust-region/结构保持。

---

# 5. 训练动力学与不动点

## 5.1 期望的不动点

令 \(Q^+(y\mid d,h)\) 是经过环境验证的成功续写分布。理想状态为：

\[
p_T^\theta(y\mid d,h)
\approx
Q^+(y\mid d,h)
\]

在可信 student-prefix 与 native-prefix occupancy 上成立。

裸 student 则逼近 teacher 对不可见 doc 的投影：

\[
p_S^\theta(y\mid h)
\approx
\mathbb E_{D\mid h}
\left[
Q^+(y\mid D,h)
\right].
\]

两者不要求处处相等。teacher 保留 doc 条件优势，student 只吸收其中可以由任务、观察和跨任务规律预测的部分。

## 5.2 纯双向 KL 的坏不动点

若直接使用：

\[
D_{\mathrm{KL}}(p_T\Vert p_S)
+
\lambda D_{\mathrm{KL}}(p_S\Vert p_T),
\]

则任意共识分布 \(p_T=p_S\) 都是零损失不动点，包括：

- 都很冗长；
- 都忽略 doc；
- 都很少 action；
- 都成功率很低。

因此“双向一致”本身不能定义正确性。VPC-OPD 用 outcome-conditioned \(Q^+\) 打破这一退化：teacher 不是被要求和 student 一致，而是被要求对成功续写负责。

## 5.3 特权能力与可蒸馏性的张力

如果 doc 包含学生在当前历史下绝不可能知道的任务特定事实，则某些 teacher 行为不能无损蒸馏。强行把这部分 KL 压到零有两种结果：

1. student 学会无依据猜测；
2. teacher 为了和 student 一致而不再使用 doc。

本方案的解决方式是：

- teacher 在 native-success 和 recovery-success 区域保留 doc 优势；
- student 只在 outcome-verified 区域接受指导；
- action 和结构信号优先；
- 不要求 student 模仿 doc-specific reasoning 文本；
- 接受一个非零、不可约的 teacher–student KL，而不是把 KL=0 当目标。

## 5.4 共享权重导致的两阶段循环

整体更新可写为：

\[
\theta_{k+1}
=
\mathcal D\big(\mathcal C(\theta_k)\big),
\]

其中 \(\mathcal C\) 为 teacher 校准算子，\(\mathcal D\) 为 student 投影算子。

若两阶段步长过大，可能形成二周期：

- teacher 阶段恢复 doc-sensitive action；
- student 阶段共享参数把 teacher 拉回 generic distribution；
- 下一轮 teacher 再恢复；
- 参数和 KL 指标交替震荡。

局部稳定要求组合算子的 Jacobian 谱半径小于 1。工程上无法直接求谱半径，但可以通过以下方式降低它：

- 每阶段只做少量更新；
- 冻结目标快照；
- 使用 KL trust region；
- 监控两个阶段对另一视图造成的 KL 漂移；
- 避免 teacher/student 两侧都训练到收敛；
- 必要时使用 EMA calibrated teacher 作为 student target。

## 5.5 崩塌吸收态

若 reasoning 已经长到几乎无成功轨迹：

- \(Q^{S+}\) 消失；
- teacher 校准恢复力变为零；
- 当前长轨迹仍继续占据训练 occupancy；
- 成功率进一步下降。

这是近似吸收态，也是历史上“不可逆”的来源之一。

固定健康 replay buffer、close-hazard anchor 和 teacher native-success 数据的作用，是在当前 on-policy 成功样本消失时仍保留恢复力。

---

# 6. 训练伪代码

```text
initialize shared parameters θ
initialize safe snapshot θ_safe = θ
initialize buffers B_safe, B_native, B_prefix

# Phase 0: no parameter update
collect healthy student and teacher rollouts
store verified compact successes in B_safe
build a fixed stratified student-prefix evaluation bank

for training step k:

    θ_b = frozen_copy(θ)

    # 1. Naked student on-policy collection
    for each task i in batch:
        obtain current researcher doc d_i
        collect K=8 student rollouts with pS(θ_b, no_doc)
        record:
            histories, observations, assistant tokens,
            turn boundaries, actions, reward, token counts

    # 2. Build successful-student dataset
    D_Splus = []
    for each student trajectory τ:
        if verified_terminal_success(τ):
            add action labels from τ
            add format labels
            add reasoning labels only if compact_under_fixed_safe_threshold(τ)
            assign task-normalized quality weight

    # 3. Build prefix-recovery dataset
    D_Cplus = []
    select fixed/stratified decision-boundary prefixes from:
        successful student trajectories
        failed student trajectories
        invalid/no-progress regions

    for each selected prefix h:
        reconstruct exact environment state
        sample M teacher continuations with pT(θ_b | doc, h)
        execute continuations in environment
        retain only verified successful continuations
        estimate confidence weight from success count
        add retained continuations to D_Cplus

    # 4. Build native privileged retention data
    D_Nplus = successful teacher-from-start rollouts
    update B_native only with verified successful native trajectories

    # 5. Teacher-side calibration
    for a small fixed number of minibatches:
        L_success = balanced_teacher_CE(D_Splus)
        L_recovery = balanced_teacher_CE(D_Cplus)
        L_native = balanced_teacher_CE(D_Nplus)
        L_close = close_hazard_loss(B_safe)
        L_keep = projected_teacher_KL(θ_safe, θ, B_native)
        L_share = projected_student_KL(θ_b, θ, current_student_prefixes)

        L_T =
            1.0 * L_success
          + 1.0 * L_recovery
          + 0.25 * L_native
          + 0.5 * L_close
          + dual_keep * L_keep
          + dual_share * L_share

        update θ
        enforce teacher-stage KL budgets
        # success/reward/snapshots/targets are all stop-gradient

    θ_c = frozen_copy_or_EMA(θ)

    # 6. Construct trusted OPD token set
    trusted = prefixes and suffixes that belong to:
        verified successful student trajectories, or
        verified successful teacher continuations

    # 7. Student-side OPD using calibrated frozen teacher
    for a small fixed number of minibatches:
        qT = pT(θ_c | doc, trusted_prefix)
        pS = pS(θ | trusted_prefix)

        U = top64(qT) union top64(pS) union mandatory_tokens
        L_OPD = segment_balanced_KL_with_OTHER_bucket(qT, pS, U)
        L_beh = projected_student_KL(θ_b, θ, broad_on_policy_prefixes)
        L_close_S = close_hazard_loss_student(B_safe)

        L_S =
            ramp(k) * L_OPD
          + dual_beh * L_beh
          + close_weight * L_close_S

        update θ
        enforce student-stage KL budget

    # 8. Evaluation and checkpoint policy
    measure:
        naked student val success
        teacher native success
        teacher success from fixed student-prefix bank
        reasoning length and close hazard
        action/format validity
        context truncation
        action/reasoning/boundary KL separately
        cross-view drift after each substage

    if all teacher and stability criteria improve:
        optionally update θ_safe

    if stop criteria fire:
        mark scientific run as failed
        for production: roll back to last safe snapshot
```

---

# 7. 稳定性与失效模式

| 失效模式 | 监控信号 | 缓解手段 |
|---|---|---|
| 成功样本 verbosity ratchet | 成功轨迹 reasoning p90 连续上升；compact 样本占比下降 | 固定健康阈值；verbose 样本只用 action；safe replay；close-hazard loss |
| teacher 被拉成 student，特权消失 | teacher correct-doc native success 下降；correct-doc 与 no-doc 差距消失 | native-success CE；成功原生前缀 KL retention；降低 \(L_{T,S+}\) |
| teacher 只在成功路径上校准，不会恢复 | native success 高，但固定 student-prefix continuation success 不升 | 增加 \(Q^{C+}\)；从失败前缀实际执行 teacher continuation |
| 成功轨迹是偶然或含无关 detour | 回放失败；同任务成功动作差异极大；action steps 高 | 重放验证；多 rollout；task 内 Pareto 选择；置信权重 |
| terminal success 给错误 reasoning 背书 | action 成功但 observation-grounding 降低；自我确认句密度上升 | reasoning 默认低权重；verbose reasoning 设为 0；只监督 action/结构 |
| teacher best-of-\(M\) 过拟合训练任务 | train-prefix continuation 上升，unseen 不升；doc shuffle 后仍自信执行具体路径 | 固定 held-out prefix bank；correct/shuffled/no-doc 对照；增加任务多样性 |
| teacher/student 两阶段震荡 | 两子阶段后 KL、成功率、close hazard 呈交替锯齿；梯度余弦长期为负 | 减小步长；KL budget；EMA target；减少每阶段 update 次数 |
| 共享权重 teacher 更新破坏 naked student | teacher 子阶段后裸 student action KL 或成功率突降 | student-view anchor；更紧 \(\epsilon_S\)；梯度裁剪 |
| student OPD 又破坏 teacher | student 子阶段后 native teacher success 或 doc 利用率下降 | 每轮重新校准；teacher retention；限制 student 子阶段 KL |
| top-64 截断隐藏 delimiter/action | `</reasoning>` 或合法 action 不在 top-64；tail mass 大 | mandatory token union；OTHER 桶；成功标签单独查询 |
| 成功数据稀疏导致模式收缩 | teacher action entropy骤降；少数模板占比暴涨 | 多任务 replay；native success 数据；reference KL；无正样本则跳过更新 |
| reasoning 权重过低损害真实规划 | 长任务成功率先下降；action invalid 或短视决策增加 | 在“紧凑成功 reasoning”上逐步将 \(\lambda_R\) 从 0 提到 0.1/0.2 |
| close loss 导致过早结束 | reasoning 长度下降但 action success、复杂任务成功率同步下降 | 软 hazard 而非硬长度截断；按任务类型设健康分布 |
| doc 本身错误或不稳定 | teacher correct-doc continuation 失败；不同 step doc 指导互相矛盾 | 只有环境成功续写进入目标；doc 与轨迹绑定保存；不以 doc 文本本身作标签 |
| 健康 anchor 过旧，阻止真正改进 | 新 teacher prefix success 上升但 retention penalty持续增大 | anchor 只在成功 action/结构上使用；满足非退化条件后更新 anchor |
| 当前成功消失后无法恢复 | \(Q^{S+}\) 为空；context hit 增加；只剩长失败轨迹 | 固定 B_safe/B_native；跳过未验证 OPD；回滚安全 checkpoint |

---

# 8. 实验验证设计

## 8.1 最小可解释消融集

建议至少 7 个臂；少于此数量会把“teacher 校准”“长度控制”“恢复能力”“特权保留”混在一起。

1. **A：原始 OPD**
   - 当前 teacher top-64 forward KL；
   - 仅 student-side 梯度。

2. **B：原始想法的朴素版本**
   - 在成功 student 轨迹上增加 full-token teacher-side CE/KL；
   - 不做 token 平衡、恢复 rollout 或特权 retention。

3. **C：稳定化控制**
   - 无 teacher-side 成功校准；
   - 只加入 task/turn/span 平衡、reasoning 低权重和 close-hazard anchor。
   - 用于判断收益是否只是长度正则造成。

4. **D：成功 student 前缀校准**
   - C + \(Q^{S+}\) teacher loss；
   - 无失败前缀 recovery，无 native retention。

5. **E：增加 recovery**
   - D + 从 student 前缀出发的成功 teacher continuation \(Q^{C+}\)；
   - 无 native retention。

6. **F：完整 VPC-OPD**
   - E + native-success 数据、teacher retention、shared-weight trust region。

7. **G：完整方案但恢复 full-token reasoning 权重**
   - 令 \(\lambda_R=1\)，按 token 聚合；
   - 用来验证 reasoning 通道是否是长度病理的关键因果因素。

至少 3 个随机种子，5 个更可靠。所有臂使用相同任务顺序、batch 结构、解码参数和评测频率。若 researcher doc 无法在不同策略间完全配对，应记录该不一致，并依赖多种子而非单次结果。

## 8.2 预注册主要指标

### 最终任务指标

- 裸 student seen/unseen val success；
- step 1–30 的成功率曲线面积；
- 峰值后最大回撤；
- collapse step。

不能只报告最终 checkpoint，因为原系统的关键问题正是训练动力学。

### 长度和结构指标

成功/失败轨迹分别报告：

- episode token p50/p90/p99；
- reasoning token 或字符/turn；
- reasoning/action 比；
- 环境步数；
- context-hit 比例；
- delimiter/格式错误率；
- close hazard 与预期 reasoning 长度。

### teacher 质量必须拆成两类

1. native teacher success：
   \[
   SR_T^{\rm native}(d)
   \]

2. 固定 student-prefix bank 上的 continuation success：
   \[
   SR_T^{\rm prefix}(d,h_S)
   \]

第二项才是 teacher 是否学会“基于 student 轨迹前缀正确调整”的直接指标。

另外报告：

- correct-doc；
- no-doc；
- shuffled-doc。

希望看到 correct-doc 相比 no-doc 的优势被保留，同时 shuffled-doc 不应造成大规模无依据行动。

### 优化指标

- action/reasoning/structure 三类 KL；
- teacher/student entropy；
- `</reasoning>` 概率；
- teacher 阶段后裸 student 漂移；
- student 阶段后 teacher 漂移；
- 各损失梯度范数和梯度余弦；
- 每任务正样本数量及 recovery 成功率。

语义循环可作为次级指标：

- sentence-level 语义相似度；
- 自我确认句比例；
- 每 100 reasoning token 引入的新观察实体数；
- reasoning 中未出现在任务或 observation 中的实体/位置断言比例。

这些自动语义指标可能有判断误差，不应用作唯一判停依据。

## 8.3 预注册预测

若机理假设正确，应观察到：

- A 在约 10–19 step 复现长度失控；
- B 可能降低成功轨迹上的 teacher/student KL，但仍可能膨胀或损害 teacher 特权；
- C 显著抑制长度，却未必提高 teacher 的失败前缀恢复率；
- D 提高 teacher 在已成功 student 路径上的 likelihood/稳定性，但对失败前缀提升有限；
- E 明显提高固定 student-prefix continuation success；
- F 在 E 的基础上维持 native teacher success 和 correct-doc advantage；
- G 重新出现 reasoning 增长趋势，即使 action 指标短期改善。

若 C 与 F 的 student 成功率相当，而 F 的 prefix continuation 没提升，则 teacher-side 校准机制没有获得独立支持，主要收益只是长度控制。

## 8.4 预注册成功标准

相对于 Phase 0 健康基线：

- 训练至少运行到 step 30，不出现不可逆 collapse；
- 成功轨迹 reasoning/turn p90 不超过健康基线的 2 倍；
- 最近 5 次评测中 log reasoning length 的斜率不持续为正；
- context-hit 低于 2%；
- native teacher success 相对 anchor 下降不超过 5 个百分点；
- fixed-prefix teacher continuation success 有统计上可信的正提升；
- 裸 student unseen AUC 高于原始 OPD；
- 最终 naked success 不以明显缩短 reasoning、但同时损失复杂任务成功率为代价。

具体 2%、2 倍、5 个百分点属于预注册工程阈值，并非理论常数；可以在不查看实验结果前，根据 Phase 0 方差统一调整一次。

## 8.5 判停条件

科学比较中，触发后应将该臂记为失败，而不是在线改超参数继续美化结果：

- context-hit 连续两次评测超过 5%；
- reasoning p95 连续两次超过健康基线 4 倍；
- val success 连续三次低于历史峰值的 50%；
- teacher native success 下降超过 15 个百分点；
- delimiter/解析错误超过 5%；
- action-token KL 单次出现预先定义的异常尖峰；
- NaN、梯度爆炸或无法恢复环境状态。

生产式调试可在触发后回滚最后安全 checkpoint、减半学习率或 OPD 权重，但这种干预不能混入预注册主结果。

---

# 9. 备选方案谱系

## 9.1 朴素对称 KL

\[
D_{\mathrm{KL}}(p_T\Vert p_S)
+
D_{\mathrm{KL}}(p_S\Vert p_T).
\]

优点是实现简单。缺点是只定义一致性，不定义正确性；任何低成功共识都是不动点，而且极易消除 doc 特权。未选。

## 9.2 只在成功轨迹上做 \(D_{\mathrm{KL}}(p_S\Vert p_T)\)

比对 teacher 第一参数求梯度稳定，近似让 teacher 模仿 student。但完整 \(p_S\) 并没有经过 success conditioning，仍包含错误备选 token。它也不能教失败前缀恢复。适合作为消融，不作为主方案。

## 9.3 完全取消 reasoning OPD，只蒸馏 action

这是强稳定基线，几乎关闭最明显的 hacking 通道。缺点是可能失去可迁移的计划结构，也没有解决 teacher 在 student 前缀上的质量问题。

主方案采用“action 强、reasoning 弱且门控”，是两者之间的折中。

## 9.4 硬限制 reasoning 长度

例如每 turn 最多固定 token，然后强制 `</reasoning>`。它能快速防止上下文耗尽，但只处理症状：

- 可能截断真正需要的规划；
- teacher 仍可能在有限预算内生成无依据叙述；
- 无法提高失败前缀的动作恢复能力。

可作为运行安全阀，不应作为机制有效性的主要证据。

## 9.5 独立 teacher 参数、EMA teacher 或 teacher adapter

优点是切断共享权重正反馈，并能更好保留特权能力。缺点是改变“一份 policy 权重两个条件视图”的核心研究设定，还会产生 teacher stale、额外存储和部署差异。

若共享权重下的 KL 约束始终不可满足，这是最合理的后续方案，但不是当前首选。

## 9.6 对成功/失败 continuation 做 DPO 或 ranking

可以在同一前缀下构造：

\[
\xi^+\succ\xi^-.
\]

优点是直接利用 outcome 对比；缺点是需要大量 matched-prefix continuation、终局奖励信用仍很粗，而且容易对 best-of-\(M\) 样本过拟合。可在 VPC-OPD 验证成功后追加，不宜第一版同时引入。

## 9.7 PPO/GRPO 等终局 RL

它直接优化任务成功率，理论上能惩罚无效长 reasoning。现实问题是：

- 奖励极稀疏；
- 30 步多轮环境信用分配困难；
- 会改变“纯 OPD self-evolve”研究问题；
- 可能引入新的长度和策略退化。

因此不作为本方案的主机制，但可作为外部上界。

## 9.8 由 researcher 生成压缩 rationale 作为标签

可能直接提供简洁推理监督，但任务书只保证 doc 是经验/指导文档，不保证它是逐前缀正确续写，也不保证与 student 当前环境状态一致。这样会把问题从 teacher hallucination 转移到 researcher-label hallucination。除非经过环境执行验证，否则不应作为主标签。

---

# 10. 最关键的不确定性

1. **终局成功不能证明 reasoning 正确。**  
   因此方案最可信的监督对象是 action 和 reasoning 结束结构，不是完整自然语言推理。

2. **success-only teacher loss不足以实现“纠偏”。**  
   它只能保证 teacher 不破坏成功 student 路径。真正的失败恢复需要从 student 前缀实际执行 teacher continuation。

3. **共享权重约束可能不可同时满足。**  
   某些参数方向可能既是 teacher 使用 doc 所必需，又会让 naked student 产生无依据断言。若 tight trust region 下 teacher 无法提高，应考虑独立 adapter，而不是继续加大相互 KL。

4. **teacher continuation 采样成本可能较高。**  
   可先每条轨迹只选 2–3 个决策边界、每前缀采样少量 continuation；但样本少时必须使用软置信权重，不能过度声称质量提升。

5. **数值权重不是理论结论。**  
   推荐值是用于启动 pilot 的量级。最终应以 action/structure KL budget、close hazard 和固定 prefix success 为控制目标，而不是追求某个总 loss 数值。

最终判断标准不应是 teacher/student KL 是否更低，而应同时满足：

\[
\boxed{
\text{prefix continuation success 上升}
\;\land\;
\text{native privileged success 保留}
\;\land\;
\text{naked student val success 上升}
\;\land\;
\text{reasoning 长度稳定}
}
\]

只有这四项同时成立，才能说明 teacher 真正从“知道答案并善于叙述”转变成了“能在 student 真实前缀上给出可执行、可迁移、结果对齐的指导”。