# PGS v3.3 — 分臂混合策略训练的修订设计(asym-advantage / W 档位 / 门控组件化)

状态:设计定稿待实施(用户裁定 2026-08-07;模块 M1–M3 已确认,M4 门控族与
消融矩阵按本文档定型)。真源关系:本文件是 spec v3.0-final(§2.6 分离式校正、
§2.3 门控)与 v3.1 分臂裁定的**版本演进**,差异清单见 §6;实施后
`docs/pgs-impl-design.md` 随代码更新。

诊断依据:`docs/pgs-s3-diagnosis-cd010.md`(s3 双病根因,数据可溯)。
文献资产:`related_work/INDEX.md`(论文 PDF + MENTOR/LUFFY 代码库,commit 已钉)。

---

## 1. 背景:为什么是"修分臂"而不是"回退全臂"

v3.1 引入 bare+steered 分臂的五条目标全部成立且不可放弃:
(a) bare 臂恢复真实能力读数(全臂 score=E_b[R] 系统性高估);
(b) 教师强时全臂组内方差坍缩,bare 供给天然负样本;
(c) per-token IS 的状态分布偏差随 B 放大,bare 是零校正的 on-policy 锚;
(d) δ gap 是教师退场(退火)的天然判据;
(e) 部署分布是 bare 的 π,优化目标必须含直接在部署分布上的成分。

s3 实测的两个病(bare 吸收慢/δ gap 拉大;thinking 弥散化膨胀)的根因**不在
分臂本身**,在两个实现选择:①16 条同组共享 LOO 基线——"结构性更强的臂持续
抬高基线",bare 失败轨迹被无差别深罚(A_bare fail-only 实测 −0.31),主模式
被侵蚀、生成分布弥散化;②教师改写位的 W_beh=π/b 忠实打折——教师信息最密处
学习率被压得最狠(远 rank 硬知识 W≈0.01–0.1)。

2025–2026 文献(LUFFY / MENTOR / SDAR / QuestA / MiGrATe)集体确认
"on-policy + 引导 rollout 混合训练"正是前沿共识形态,且对两个病因都有
成熟答案(§2)。162 全臂 run(150 步 val 68.25%、零膨胀、六族全开)是
"轻度干预下 b≈π"的特例佐证,不构成回退全臂的理由。

## 2. 文献坐标系(机制矩阵)

| 工作 | 场景 | 教师形态 | 混合结构 | advantage 口径 | off-IS | 干预选位 | 退火 |
|---|---|---|---|---|---|---|---|
| **MENTOR**(ICLR26, 2510.04140) | math 单轮 | 外部 7B distill | 4 on + 4 mixed(50%) | **on 组自比;mixed=[R−mean_on]₊×α** | **无**(ratio 分母统一 π_θold) | 学生熵软插值,P95 反馈环+1.5 地板 | **α cosine 120 步→0,停产引导** |
| **LUFFY**(NeurIPS25, 2504.14945) | math 单轮 | R1 轨迹(离线) | 1 off + 7 on 混组共享基线 | 共享基线(有意做自动课程) | **f(π)=π/(π+0.1) 反向加权**(π_φ=1) | token 级 prefix_mask | 无显式(on 自然接管) |
| **SDAR**(2605.15155) | **agentic 多轮**(ALFWorld+9.4%) | **π+特权上下文(自条件化,与我们同)** | loss 端辅助项(λ=0.01) | RL 主干不动 | — | **σ(βh)/σ(βΔ)/soft-OR 三门**,detach | 门自适应 |
| QuestA(2507.13266) | math | prompt 层 hint | 数据增强 | 原生 GRPO | — | — | hint 比例 50%→25% |
| MiGrATe(2508.08641) | TTT 搜索 | 历史高分解(纯正样本) | 混组共享基线 | 共享(off 全正样本,无臂间惩罚语境) | 无 | — | — |
| **PGS v3.3(本设计)** | agentic 多轮 | **π+特权 doc(researcher 自进化)** | 8 bare + 8 steered 分臂 | **M1 不对称口径** | **M3 off 默认/shaped_reject 消融** | **M4 组件化三门** | **M2 α cosine 配置期** |

三个共同判词:①两篇 SOTA(MENTOR/LUFFY)都拒绝"忠实 per-token IS 打折"
(MENTOR 免 IS、LUFFY 反向 shaping);②混合样本的失败不惩罚(MENTOR [·]₊)
或低配比稀释(LUFFY 1:7);③干预选位用平滑软调制,不用硬门+恒强度。
SDAR 的补充证词:自条件化教师在 agentic 有效;**平均 gap 恒为负**(教师在
多数位置不比学生强),无门控蒸馏主动有害(1.7B 上 GRPO+OPSD 32.0 vs 纯
GRPO 46.1)——门控是把"教师真正有增量的少数位置"筛出来的开关,与我们
"仅 0.75–4% 位置换 token、29–34% 拒绝是纯降噪"的测量同构。

我们保留的独有点(论文差异化主张):教师=π+特权文档自条件化(无词表约束、
教师随 policy 复利变强);steered 轨迹**文本层信息状态干净**(倾斜只动 logit,
无特权文字落轨迹——SDAR 实证了 prompt 注入 skill 的对照方法测试时移除即大跌,
我们结构上免疫);doc 由 researcher 在线自进化(v4 主线);agentic 多轮语境
的常驻引擎 tail-only verify(41×)。

## 3. v3.3 机制定义(六模块)

### M1 分臂结构不变 + 不对称 advantage(`pgs_asym_grpo`)【已裁定】

分臂 8 bare + 8 steered、同批同任务照旧;advantage 改为:

- **bare 行**:组内 LOO,组=同任务的 8 条 bare(现行 grpo_loo 的 no-std 口径
  不变):`A_i = r_i − mean_{j≠i, j∈bare}(r_j)`。
- **steered 行**:`A = clamp_min(r − mean(bare 组), 0) × α_t`——相对 bare 均值
  的**正超额**;失败或不超均值 → advantage 恰 0(零梯度,不惩罚:引导轨迹的
  失败可能是干预所致,归因不明的样本不给梯度)。
- 防御闸:`steered_adv_cap`(组内 steered advantage 总额上限,默认不激活,
  照 MENTOR expert_cap 先例)。

性质:①bare 失败不再被臂间基线深罚(s3 病②的根被拆);②bare 全败时
mean_bare=0,steered 任何成功都是满额信号——**教师信号强度随 policy 能力
自动退坡**(policy 最弱时最强);③"教师能成/policy 不能"的对比信号保留,
但作用对象从"惩罚 bare"换成"奖励 steered 超额"。
配置:`steering.adv_mode: asym`(进 objective_fingerprint;`legacy_shared`
保留供既有 run 语义)。

### M2 α 调度(cosine 有限期,简单版先行)【已裁定】

`α_t = α_0 · 0.5·(1+cos(π·t/T_decay))`,t ≥ T_decay 后恒 0,**且 α=0 后
steered 臂停产**(rollout 全 bare,教师是有限期脚手架;MENTOR 同款)。
配置:`steering.steered_adv_weight`(α_0,默认 1.0)、
`steering.steered_adv_decay_steps`(T_decay,**按总步数比例配置,建议 2/3——
为副动作族慢热留窗,162 实测该族 80–120 步起飞**)、
`steering.steered_adv_decay_mode: cosine_steps`(枚举留位;后续自适应版
如"bare val 两点不涨则冻结衰减"只加枚举值不动结构)。均进指纹。

### M3 W_beh 档位(spec §2.6 版本演进)【已裁定】

- `w_beh_mode: off`(**新默认**):steered 行不做行为分布 IS,PPO ratio 分母
  统一 π_θold(MENTOR 口径)。偏差由 M1 的 [·]₊(只学正例,正例的教师偏向
  恰是想要的转移)与 α(总量旋钮)承接。
- `w_beh_mode: shaped_reject`(消融):仅拒绝位(rejected=1 的 token 级掩码,
  既有记账直接可用)乘 f(π)=π/(π+γ_s),γ_s=0.1(LUFFY 口径)——低概率教学位
  梯度反向加权;接受位不动。
- `w_beh_mode: per_token`(v3.0 口径,保留供对照)。
依据:s3 实测拒绝位 W 中位 0.42–0.65、远 rank 硬知识 W≈0.01–0.1,教学位
打折与"吸收慢"直接相关;两篇 SOTA 的反方向选择见 §2。三档构成论文关键消融。

### M4 门控组件化 + 三实例(本轮定型)

**结构重构**:门从"开/关判定 + 恒定 η_max"改为**强度函数 w_t ∈ [0,1]**,
`η_t = η_max · w_t`(硬门=阶跃特例)。统一接口 `IntensityFn`(输入
per-position 信号包:学生熵 h_t(draft 侧免费)、D_t(需教师 verify)),
配置 `steering.gate_mode` 一键切换。ε 探针语义平移:5% 随机 w→0,因果读数不变。
`w_t < w_min`(配置,默认 0.05)的位置**跳过教师 verify**(bypass,直通 draft)。

- **G-E 熵门(默认首发)**:`w = min(1, h_t / H_q)`。H_q = 上一步全 batch
  学生熵的 P_q 分位(反馈环,滞后一步)+ **硬地板 H_floor**(防阈值追熵坍缩,
  MENTOR 的 max(q80,1.5) 先例)。分位默认 q=0.95;**H_q 初值与 H_floor 从
  s3 已落盘的 pi_top16 计算我们语境的逐位熵分布定标**(零额外算力)。
  工程红利:关门位零教师成本(D 门做不到——为决定"要不要问老师"必须先问老师)。
- **G-D 门(改良保留,对照臂)**:`w = min(1, D_t / θ_D)`(软斜坡化,弃阶跃),
  D = Var_{a~π|S}[ℓ̃(a)] 与 θ_D 自适应照旧。保留理由:D 携带教师异议信息,
  熵门的盲区恰是"confidently wrong"(学生低熵地走向错误);对照实验干净回答
  "学生不确定性 vs 教师异议,哪个选位更准"。
- **G-SOR 软或(二版/消融)**:`w = 1 − (1−w_E)(1−w_D)`——两盲区互补
  (confidently-wrong ∪ 学生高熵),理论上限候选(SDAR soft-OR 同构)。
- Gap 门(SDAR σ(βΔ))不进采样端:干预发生在 token 选出之前,无"已采 token"
  可评有符号 gap;其思想由 M3 的 shaped_reject 在 loss 端承接(拒绝位=教师
  实际否定 draft 处)。

### M5 无进展截断:不做【用户裁定 2026-08-07】

替代:监控保留 respLen max 距 32768 余量告警(<10% 红)。判断依据:膨胀的
驱动力(臂间深负 advantage)已被 M1 拆除,162(无此结构)全程无膨胀;预期
自愈。若 v3.3 臂仍膨胀,即构成干预实验级新证据,届时重议。

### M6 记账、监控与预注册消融矩阵

新曲线:w_t 分布(分位)、有效干预密度(η>0 且被采纳)、steered 超额率
([r−mean_bare]₊>0 占比)、α_t、吸收进度三曲线(拒绝位 Δlogπ 中位 /
final=π-top1 率 / 拒绝密度,`s3_deep.py` 口径)、**fork-token occurrence
双曲线(§4,redundancy 带 vs essence 带)**。
消融矩阵(预注册):`gate_mode {entropy, d_var, soft_or}` ×
`w_beh_mode {off, shaped_reject, per_token}` × α 调度(主对照:有/无衰减);
主臂 = entropy × off × cosine(2/3)。

## 4. 推理模式与长度动态:MENTOR 的观察 × 我们的机理(论文素材)

### 4.1 MENTOR 的度量与发现(§4.3/§4.4/Fig.2-3/App.G)

度量:高频推理 token 的 **occurrence rate**(500 条 MATH500 轨迹中该词至少
出现一次的轨迹占比),四个体系对照(Expert / On-policy RL / LUFFY / MENTOR)。
词表分两带:**Redundancy 带**(but / wait / okay / alternatively——无谓犹疑与
反复重审)与 **Essence 带**(check / verify——有目的的校验)。发现:

- **On-policy RL**:两带 token 几乎全零——模型自身分布里没有反思模式,推理
  短平快,但能力上限受限(pass@32 停滞甚至低于 base);
- **Expert 本身**:wait/okay 也高——专家轨迹天然携带冗余;
- **LUFFY(全轨迹模仿)**:无差别吸收,okay/wait 采纳率过高 → "overly
  redundant reasoning"。App G 同题样例:一次得到答案后仍 "But wait, maybe
  there's a better way"→"Hmm, okay"→"Wait, let me double-check"→"But let me
  make sure" 连续多轮重审;
- **MENTOR**:verify/check 吸收至接近 expert 水平、wait/okay 显著低于 LUFFY
  ——**选择性吸收**(样例:单次 "To verify" 校验后即收束)。

长度动态(§4.3):MENTOR 早期长度**上升**,归因为 "adopting expert-style
reasoning forks such as verify and wait, **the occurrence of which extends the
reasoning chain**"——每个分叉词开启一段新推理,分叉词占有率直接决定链长;
后期随 α 衰减长度回落,"model starts to distinguish useful tokens (verify)
from redundant ones (wait)"——**筛选由 outcome RL 完成:两阶段 = 先(有限期)
吸收、后(健康 advantage)筛选**。

### 4.2 与 s3 观察的统一:同一表象、两条流入通道、同一收束机制

s3 的 thinking 膨胀在 token 层的表象与 MENTOR/LUFFY 描述一致:fork 词
(however / but / wait / alternatively)反复出现、推理链不收束。但**流入通道
不同**,这是关键对照(也是论文的分析点):

| | MENTOR/LUFFY(模仿通道) | s3(侵蚀通道) |
|---|---|---|
| fork 词从哪来 | 从 expert 分布**注入**(教师风格被模仿) | **非注入**——bare 臂零教师接触同样膨胀;混合组把失败轨迹的主模式(果断推进+收尾)持续压低,概率质量外溢 |
| 为什么流向 fork 词 | 教师本来就说 | fork 词在 reasoning 语境是"永远合法的延续"——当具体的果断路径被压低,分叉成为外溢概率质量的默认去处 |
| 微观证据 | Fig.3 occurrence 对照 | top-1 概率 p25 从 ≈0 钝化到 −0.16~−0.38;`</` 收尾预备信号衰减;轮长分布右尾化;轮内 5-gram 重复率 ≈0(新内容不是复读) |
| 收束机制 | α 退坡 + 健康 advantage 筛选(有用 fork 提升成功率→保留,冗余 fork→淘汰) | s3 的病态 advantage(bare 失败恒负、无差别压制)**只有压制没有筛选**,且压的方向恰好相反(压掉的是果断收束模式) |

统一结论:**fork 词占有率上升 = 生成分布弥散化在词层的具体形态**;我们 deep2
的三个测量与"fork 词反复"是同一件事的四个切面。修 M1 不只是治臂间不公平,
更是**恢复 fork 筛选机制的前提**——只有当成功/失败的对比落在正确对象上,
outcome 信号才能区分"有用的 verify"与"冗余的 wait"。162(健康 advantage)
respLen 全程平稳、s2(高胜率)长度不升反降,均为佐证。此为 M5 "预期自愈"
判断的机理基础。

### 4.3 落地为监控(M6)

把 occurrence rate 移植为常规观察量:redundancy / essence 两条曲线。ALFWorld
语境词表不照抄 math——**从赢/输轨迹的差异词数据驱动定标**(标定项:对既有
dump 统计两类轨迹的高频差异词,人工归带;初始候选 redundancy: wait / however /
but / actually / let me re-;essence: check / verify / confirm / locate)。
该曲线比 respLen 更早、更可解释地暴露"冗余分叉膨胀",也直接产出论文 Fig.3
同款对照图的数据面。

## 5. 预注册要点(细节归验证设计文档)

主判据仍是 val(274 任务 greedy,10 步一点);逐步分臂曲线只作观察。
触发器:①λ±/W-floor 升级判据="连续两个 val 点不涨且 Δlogπ 停止缩小";
②膨胀复发判据=respLen 连续 5 步斜率>+150 tok/step 或 max 余量<10%(触发即
重议 M5);③α 冻结讨论判据=衰减进入后半程且 bare val 两点不涨。

## 6. 与 spec v3.0-final 的差异清单(版本演进留痕)

| spec 条目 | v3.0-final | v3.3 | 性质 |
|---|---|---|---|
| §2.6 分离式校正 W_beh | per-token π/b 必做 | 三档,默认 off | **方向修订**(文献+实测依据 §2/§3-M3) |
| §2.3 D 门+θ_D | 硬门,恒 η_max | 组件化 w_t,G-D 为其软化特例 | 泛化(D 门保留为对照) |
| v3.1 分臂口径 | 16 条同组共享 LOO | 不对称口径 M1 | **修订**(s3 实证) |
| §2.2 η | 恒 η_max | η_t = η_max·w_t | 泛化 |
| 预算 B/pacing | v3.2 maxopen 档 | 不变(b_max 语义不动) | — |
| ε 探针 | 门翻转 | w→0 翻转 | 语义平移 |

## 7. 参考文献(PDF 与代码见 related_work/INDEX.md)

MENTOR arXiv:2510.04140(ICLR 2026;code commit b1cc4c6)· LUFFY
arXiv:2504.14945(NeurIPS 2025;code commit 01f4ae0)· SDAR arXiv:2605.15155 ·
QuestA arXiv:2507.13266 · MiGrATe arXiv:2508.08641 · GSPO arXiv:2507.18071 ·
Group-Relative REINFORCE is Secretly Off-Policy arXiv:2509.24203 · CRAFT
arXiv:2606.29476 · OPID arXiv:2606.26790 · SD Reward Shaping arXiv:2608.03223。
