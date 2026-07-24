# 分布分析套件(distillation microscope)— 设计文档

> 用户裁定(2026-07-24):本套件是**论文 analysis 章的关键部分**,同时服务训练
> 期诊断(内化慢的病因、崩塌预警)。绘图必须是**出版级学术标准**。已裁定:
> 对齐指纹双轨(稳健版进主文、指纹版做亮点)、E1 冻结臂升格为 C3 核心消融、
> probe 64 条按建议构造、重注入消融列 P-d。开发在 `distribution-analysis` 分支。

## 0. 本质框架(一段话)

机制 = 条件信息的「注入—消费—补给」循环:teacher 与 student 是**同一权重**在
有/无文档两个条件下的分布,故 `KL(π_θ(·|x⊕d) ‖ π_θ(·|x))` 恰是"doc 携带的、
尚未内化的可用信息量"(nats)。OPD 是把该上下文信息压缩进参数的算子;动态文档
= 当信息被消费(内化)后由 researcher 重新注入(对抗 h3 式枯竭崩塌)。分析的
任务:把 C1 通道有信息 / C2 内化真的发生 / C3 补给对抗枯竭 / C4 时间结构与
边界 / C5 稳定性机理 五个 claim 各配上分布层证据。

## 1. 指标(正式定义与估计器)

| # | 量 | 定义 | 估计器/口径 |
|---|---|---|---|
| 1 | 残余特权信息 G_k | E_{y~π_k}[KL(π_k(·\|x⊕d_k,y_<t) ‖ π_k(·\|x,y_<t))] | 训练内 = topk 近似(与 loss 同源,免费,趋势用);probe 上全 vocab **精确**(论文用) |
| 2 | 冻结参照 G^frozen_k | 同上但 d 固定为 d_fix(step-1 文档快照) | probe 前向;**回答"G 降是内化还是 doc 变了"**;其渐近平台 = 经验不可内化残量(RLSD irreducible-gap 的可测化) |
| 3 | 分布位移 d_init(k) | KL(π_k ‖ π_0) | probe 全 vocab;π_0 logits 缓存一次 |
| 4 | 内化效率 η_k | −ΔG^frozen/Δd_init | 每单位移动换来的残余信息减少;走低=漂移前兆,<0=崩塌几何特征 |
| 5 | 对齐指纹 align_k | cos(Δ_doc, Δ_train),Δ_doc=teacher−student 分布差,Δ_train=π_{k+1}−π_k | probe 决策点、相邻 checkpoint logits 差分;**双轨**:稳健版=doc 抬升 token 集与训练抬升 token 集的重叠率 |
| 6 | 深度剖面 G_τ | G 按 assistant turn τ 分桶 | 训练内 per-turn 聚合(全量);**熵门控版**只计 H^S_t>h₀ 的 token(剔除"需求衰减"混杂);配 overlap_τ、ΔH_τ(Prune-OPD 同款,可对话文献);重注入恢复率 ρ_τ 判可达性衰减(P-d) |
| 7 | 类型分解 | KL 质量份额 by {action-verb, action-arg, reasoning-connective, format} + top-1% 份额/Gini | token 级抽样落盘 token_id,类型离线规则判(块结构+小词表,插件级) |
| 8 | 信息—行为换算 | 任务级散点 G vs uplift(特权−裸成功率) | analysis 章开场图素材 |

统计纪律:probe 集一次冻结全程不换;曲线带 task-bootstrap CI;单 run 局限入
limitations(444 step 时间分辨率 + E1 对照臂补强)。

## 2. 数据层契约(落盘 schema,P-a/P-b 的实现依据)

**训练内(verl patch,P-a)**——蒸馏 loss 聚合前采样落盘,两级:
- per-turn 聚合(**全量**,~40 turn×128 条×6 float/step,零负担):
  `{step, traj_id, turn, g_mean, g_gated, overlap, h_teacher, h_student, n_tokens}`
- token 级(**抽样** 4-8 条/step,~5-10MB/step):
  `{step, traj_id, t, turn, token_id, kl, logratio, h_t, h_s, top1_t, top1_s, in_overlap}`
- **turn 边界**:verl 侧没有 turn 概念——由 trace 侧在 rollout dump 里记
  `turn_token_spans: [[start,end),...]`(agent loop 逐轮 append 时天然知道),
  离线 join。**这是 P-a 的 trace 侧半件。**

**probe 工具(evalm 侧,P-b)**——每 val 点对 64 条冻结前缀做四条件前向
{π_k, π_k⊕d_k, π_k⊕d_fix, π_0(缓存)},全 vocab logits 落盘(供 align 差分),
输出 `ProbeSweep` 行:`{step, d_init, d_gain, d_frozen, align, align_robust, ci_*}`。
probe 构造:valid_seen/unseen 各 24 + step-0 训练轨迹 16;fork 位置
(top-1<0.25)为主,任务族覆盖;文本冻结。

## 3. 论文图规格(六张,`trace_opd.analysis.figures` 实现)

| 图 | 内容 | 编码 |
|---|---|---|
| **A ledger** | 上:score 与特权成功率双线+uplift 带;下:G 与 G^frozen | 双面板共享 x=step;事件竖线 |
| **B trajectory** | 内化轨迹:x=d_init, y=G^frozen,时间渐变着色;多臂(main vs E1)同图 | 散点连线+viridis colorbar;健康=右下,枯竭漂移=回勾右上 |
| **C terrain** | G_τ 地形:x=turn, y=step, 色=G;右侧早/晚期剖面边际 | imshow(viridis)+边际线;熵门控版并置调用 |
| **D microscope** | 上:单轨迹 per-token log-ratio 热条(turn 边界+动作位标记);下:类型份额堆叠随 step | RdBu_r 中心 0;Okabe-Ito 堆叠 |
| **E action-flow** | 上:probe 决策点动作组概率堆叠面积随 step;下:doc uplift 按动作组演化(补给重心迁移) | stackplot+多线 |
| **F fingerprint** | 上:align_k 双轨曲线+CI;下:单决策点案例 Δ_doc vs Δ_train 成对条形 | 双色成对条形 |
| (附)fork | fork/lock 率随 step + fork 点 doc 抬升类别 | 双面板 |

**绘图标准**(`analysis.style`):ICLR 栏宽(单栏 3.35in/双栏 6.75in)、字号
7-8pt、Okabe-Ito 色盲安全色板、发散量用 RdBu_r 中心 0、顺序量用 viridis、
top/right spine 关、PDF(矢量,投稿)+PNG(300dpi,预览)双输出、单位入轴标签
(nats/token)。**合成 demo 数据的图一律带 "(synthetic demo)" 标注**,严禁与
真实数据图混淆(仓库铁律:数字必须来自真实运行)。

**分工**:论文静态图 = 本包(matplotlib);token 文本着色的交互显微镜与浏览
= viz P3(同一数据层);viz 的 figure-spec 管线(P1)后续接本包做"点击导出
论文图"。

## 4. 实施分期(本分支)

1. **绘图包**(本轮):style/schema/figures/demo + 渲染测试——先把图的形态、
   输入契约定死,数据层照契约实现;
2. **P-a**:verl patch(per-token 采样落盘)+ trace 侧 turn_token_spans;
3. **P-b**:probe 工具(evalm 侧,四条件前向 + logits 缓存/差分);
4. **P-c**:viz P3 接数据层(交互显微镜);
5. **P-d**:重注入消融 teacher 变体(三分解的 b/a 判别);
6. **P-e**(第二期):反事实对照(错位/反向 doc)。

E1 冻结臂注意事项:probe 与落盘 patch 必须在 E1 run 之前就位(两臂同测才可比)。
