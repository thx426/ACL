# Teacher Grounding(教师接地)— 反向蒸馏臂设计(v2,2026-08-02)

> 状态:**方案整体经用户最终确认(2026-08-02)**;v2 = 定稿陈述重写(§0–§4)
> + 解耦落地(§5)+ 判据件(§4.3/§7)+ 双盲验证台账(§12)。实施真源。
> 分支:`mixed-objective-r2`(基点 b4e97a5,机制链至 8c83246);verl 真源
> worktree `../verl-mixob`(trace-opd 分支,链至 0a4bb30)。
> T1 底座:**reverse_kl_topk(rkl_seqmean 基线复用)**,用户拍板。

## 0. 总纲

**教师的理想形态 = 用学生的健康说话方式,说出 doc 才知道的内容。**

要治的不是学生,是教师的条件分布本身:让教师学会基于学生轨迹前缀做正确轨迹
分布的续写式指导;保留特权信息带来的能力,但把它对齐到"正确调整轨迹"上,
而不是"叙述自己知道答案"。形态对齐学生健康样本(tg 标签承载);内容正确性由
doc 承载于失败区(OPD)与教师输入上下文。

## 1. 动机与病灶(用户诊断,三方证据锚定)

纯 OPD 的病:teacher 条件在 (doc, h)、student 条件在 h——在 student 前缀上,
teacher 因"知道答案"其分布系统性放大**复述正确推理、反复反思转折**的叙述
token(它无法替学生行动,只能"说该怎么做");这些 token 满足 KL 梯度但不提升
正确率;student 被拉向叙述分布 → 上下文膨胀 → 撞上界 → 成功率崩。证据:A0
step-17 长度失控必崩指纹;v2.2 的 23.6% SFT 样本教"零观察依据断言";
2606.04703 trajectory inflation 与 WhySD 的 epistemic 通道机理。**病根:
teacher 的指导分布不在"裸上下文可执行"的流形上。**

此前一切臂(含 SFT-doc)训练的都是 q 的**执行者角色**(在自己的状态分布 d^q 上
从头走成);q 在 OPD 里的真实角色是**教师**(在 student 的状态分布 d^p 上被评
估、被消费)——教师角色此前从未被直接训练,其在学生前缀上的可靠性只靠跨分布
外推,而 CD-003 实证外推恰好在此断裂(自走 87.5% vs 错误前缀零抓地力)。

**成功轨迹实证(2026-08-02,`trace_opd_4b_rkl_seqmean_rl` won 分层复盘)**:
won 分割的动机由推断升级为实证——该 run(rkl+RL,崩于 step-17)的**成功轨迹
本身**在 OPD 下呈单调加速膨胀:

| 训练段 | won tok 均值 | thought 字符/turn | thought/action 比 | env_steps |
|---|---|---|---|---|
| step 5–7 | 1216–1453 | 284–315 | 14–17 | 10–13 |
| step 9(相变) | 3300 | 481 | 26 | 20 |
| step 13–15 | 2598–2822 | 568–604 | 30–31 | 14–15 |
| step 17–20(崩塌窗) | 3539–5199 | 1164–1986 | 59–110 | 8–10 |

总 token ×3–4、每 turn reasoning ×6、thought/action 比 ×6;同期 env_steps
**下降**(10–13→8)——膨胀纯来自叙述而非行为。相邻逐字重复率恒 0:不是复
读,是**新写的合法叙述**(KL 的合法最优解形态)。定性三形态:无依据断言物体
位置、循环自我说服("This is the only logical step … I must go there
again")、receptacle 罗列复述——特权措辞已遗传进成功轨迹。数据:162
`runs/trace_opd_4b_rkl_seqmean_rl/rollouts/step_*`;分层脚本已入库
`scripts/won_traj_evolution.py`;代表性轨迹样本 `analysis/won-traj-samples/`。

## 2. 机制定义:双向教学结构(won 分割)

每 step 的裸 rollout(16×8)按终局成败分流:

- **lost 行(失败轨迹)→ OPD 照旧,doc 教学生。** teacher(带 doc)在学生前缀
  打分 top-64,学生裸前向,KL 梯度经学生侧。失败区正是 doc 内容价值最大的地
  方——学生不会做、教师知道答案。
- **won 行(成功轨迹)→ 转 tg 行,学生教教师。** 裸学生的成功轨迹是"健康说
  话方式"的实证样本——它证明了无 doc 条件下这样的 reasoning 密度与行为分布
  足以完成任务。打分:**裸前缀** top-64 重归一 detach → 标签 p̄;训练前向:
  prompt 换 **doc 前缀**(= q_θ);损失只经教师前向回传:

```
L_tg(行) = mean_t Σ_v p̄_t(v)·( log p̄_t(v) − log q_θ(v | doc, h_t) )
```

**方向与形式论证**:

- **标签在第一槽(前向/MLE 形)**:mode-covering 最大似然,教师必须把质量铺到
  学生健康分布支持的每个地方;反向(q 在第一槽)则教师可以靠降熵/模式收缩
  逃逸。标签 detach,教师不能移动目标。q 侧不重归一是**正确**的(保留把质量
  拉进标签支持集的梯度压力);标签重归一在打分器一次完成(截断质量实测
  ~5e-4,忽略)。
- **梯度 = q − p̄(logits 层)**:thinking token 上,健康标签 p̄ 在叙述类 token
  质量低、推进/关闭类 token 质量高;膨胀教师 q 恰相反——梯度逐位压叙述、抬
  推进,即"用学生的健康说话方式"的机械实现。per-logit 有界 ≤1、行 L1 ≤2
  (§11 P0-2 极端复核)。
- **全 token(thinking+action),权重均匀**:thinking 是主修面——病灶恰在教师
  对 thinking 的叙述分布,任何绕开 thinking 的设计(action-only 等)都是绕开
  问题本身(用户驳回定案)。action 位学生熵低、实现动作占标签主质量,同样受
  监督,无需特殊化。
- **内容正确性从哪来**(不从 won 标签来——标签只承载形态):①lost 行 OPD
  (失败区学生真正需要 doc);②tg 训练前向的输入里有 doc——赢局动作与 doc
  推荐一致,doc **帮助**教师预测标签,tg 因此强化教师对 doc 的 attention 利
  用:学会"说出 doc 才知道的内容",而非"复述自己知道答案"。
- **不追 KL=0**:doc 含学生原理上推不出的信息,师生存在不可约残差;学生的
  极限是教师对不可见 doc 的投影。tg 的目标是把教师的可蒸馏部分变健康。

**机理预测**:①指导分布拉回裸可执行流形 → 长度相变推迟/消失;②成功区 PMI
推力显式归零 → "信号耗尽→噪声传递→崩"改写为"耗尽→归零→平台";③reward 经
won 过滤(RFT 式数据选择)进入蒸馏环——最稳的锚形式,规避 AntiSD"裸加 RL 锚
仍崩"的坑;④teacher 的 in-context 文档使用能力被主动维护(直接回应
2606.04703"global 注入损毁经验使用能力")。

## 3. 聚合:频率加权(用户裁定,seq-mean-token-mean)

```
L = (N_lost/N)·mean_lost[OPD] + w_tg·(N_won/N)·mean_won[tg],
N = N_lost + N_won,  w_tg = 1.0
```

codex P1-1 指出独立条件均值下"won 率上升 ⇒ OPD 自动退火"不成立。裁定频率
加权:两项同除 batch 全局**总**信号行数,每项有效权重 = 批内份额;won 率上升
真正退火 OPD、lost→0 连续归零、单条 lost 行只携 1/N(方差防护内生)。实现:
`_stamped_global_seq_info` 在 tg 戳(`batch_num_tg_seqs`,引擎 all-reduce 打
戳)存在时并入总数,两项同一 helper 同一分母;tg 关(无戳)时逐字节回历史
lost-only 分母。行内 token-mean、行间 seq-mean;被 drop 的 won 占位行两个计数
都不进。tg 行不产生 OPD token 损失(response_mask 全零),但**照常计入**行为
曲线 critic/score/mean(裸策略真实赢局,与 rollout 全量胜率一致——P0-1)。

## 4. 训练动力学:枢纽变量与三道防线

**枢纽变量 = won 标签的 thinking 健康度。** 良性循环:tg 治教师 → lost 行
OPD 的 top-64 目标变健康 → 学生学到健康行为 → 下一批 won 轨迹更健康 → 标签
更健康。恶性分支客观存在:病理若抢先污染 won 轨迹,tg 会把膨胀形态固化进教
师(正反馈)。三道防线:

1. **step-0 开臂(硬约束)**:T1 必须从第 0 步开 tg,绝不中途接入——中途接入
   = 把已膨胀的成功形态当健康标签固化进教师。实证支撑:基线 step 5–7 标签仍
   健康(thought/turn 284–315),step-9 才起相变;tg 必须抢在相变前建立压制。
2. **标签质量曲线(必带监控,已实施)**:`tg_label_thought_chars` /
   `tg_label_thought_turns` 两条在线曲线(仅真正成为标签的完整 tg 行非零;
   cpt = 两曲线相除 = 标签行的 turn 加权 per-turn 均值)。口径与
   `scripts/won_traj_evolution.py` 逐字节同源(Σ steps[].thought 字符 / step
   数)。这是枢纽变量的直接观测,同时是预注册锐判据的在线版。
3. **标签健康门(已实施,默认关)**:`mix_tg_health_gate`(字符/turn 阈值),
   超阈 won 行不转换(`tg_dropped_unhealthy` 占位,曲线可见)。T1 首跑保持
   null(单变量纪律);凭曲线 2 决定启用;阈值锚定健康期(基线 ~315 的 2–3
   倍),**严禁随当前模型滚动上调**;非 null 条件式进 fingerprint。

**4.3 机理判据:thought/action 段分解(已实施,离线,双镜像模式)**。
`scripts/opd_kl_span_decomp.py` 读 TRACE_OBS_DIR 观察包,按 `</reasoning>`
闭合分 thought/action 段,输出 per-token 均值对、总量份额与 token 份额对照:

- `--role bare`(环2 读数):lost bare 行的 OPD per-token KL——教师在输局前
  缀上给学生的目标是否变健康。覆盖本方案的诚实软肋:tg 在赢局前缀训教师、
  教师的打分功能消费在输局前缀,靠"叙述病是全局风格属性"泛化——此即对该
  泛化假设的检验。
- `--role tg`(环1 显微,2026-08-02 补全):tg 行的 per-token soft-CE(q 离
  健康标签的距离)——**教师被纠正的梯度花在叙述 token 还是行动 token**。预
  测:初期 thought 段占大头(教师叙述倾向与学生健康标签的分歧集中区),随
  训练收敛下降。数据层:tg 行以专属观察包落盘(teacher_*=标签 p̄、
  student_support_lps=q-on-support、token_loss=soft-CE;q 自身 top-k 即行内
  student_*),占位行无标签不落。

判据数据前提:run 开 TRACE_OBS_DIR(runbook 注明)。

## 5. 解耦(用户裁定 2026-08-02,已实施)

学生 OPD 损失与教师 tg 损失**完全独立**,不因工程实现折中:

1. **独立标签通道**:打分器把 tg 标签发到 `tg_label_ids`/`tg_label_logprobs`
   (batch 含 tg 行才存在;零字段零传输)。tg 行的 teacher_* 变零占位(sft 行
   同款,保持 OPD 公式的全批 packed 对齐)——OPD 公式永远见不到裸策略分布
   穿着 teacher 字段名。
2. **独立公式** `compute_tg_soft_ce`(fsdp/losses.py):mask 切子集 → chunked
   gather(logit−logsumexp,无 [.,V] softmax buffer)→ soft-CE → scatter 回
   全批形状。**结构性无任何 OPD clamp**(P0-2 豁免由行级打洞变结构;
   `compute_forward_kl_topk` 整段还原纯 OPD 形态,与历史臂逐字节可比);
   loss_max_clamp 不作用于 tg(大 loss 是合法监督);聚合层 clamp_min(0) 仅
   fp 兜底,`tg/loss_min_preclamp` telemetry 为不变量监控。SP 切法与 OPD 公式
   同构(标签/mask 同 slice;bool pad 为 False 安全)。
3. **配置解绑**:`opd_kl_mode` 白名单 **{forward_kl_topk, reverse_kl_topk}**。
   这不是折中:k3/gated 族用 teacher_topk=0 单列(realized token)打分,打分
   器强制 batch 内统一列宽、nested 末维同宽,与 tg 的 top-64 标签**数据布局层
   结构不兼容**,fail-loud。三个红利:P0-2 结构化;OPD 散度族内自由;**T1 可
   直接叠 rkl_seqmean 基线**(省一条 forward 配对 run,且与最丰富的崩塌实证
   直接单变量对照)。

## 6. 数据流:与纯 OPD 的镜像对称(实现骨架)

| | 打分前向(no-grad,fsdp 后端) | 训练前向(grad) | label 通道 | loss |
|---|---|---|---|---|
| OPD 行(won=0 bare) | **带 doc** 前缀 → teacher top-64 | **裸**序列(student) | teacher_ids/logprobs | forward/reverse_kl_topk(现状不动) |
| tg 行(won=1 bare 转换) | **裸**前缀 → p̄ top-64 重归一 | **带 doc** 前缀序列(q) | tg_label_ids/logprobs | compute_tg_soft_ce(独立) |

每行一次打分 + 一次训练前向,**总算力与纯 OPD 持平**。tg 行序列构造复用 SFT
臂"doc 前缀 + 轨迹重放"构造器(同构性质单测已锁)。行容器形态:tg 行
`response_mask` 全零(不进 OPD/RL 分子分母、显微镜 obs 跳过)、`tg_mask` 标
生成 token、`sample_role="tg"`;打分侧 `teacher_mode="tg"`(裸前缀丢失
fail-loud,不许静默退化)、占位行 `"tg_skip"` 零前向。doc 配对:同 step 现产
同 step 消费,无陈旧配对。

## 7. 配置与开关(trace_opd 侧)

```yaml
train:
  mix_tg_weight: 0.0          # tg 项权重;T1 = 1.0
  opd_lost_only: false        # won 分割;T1 = true(T1' 归因臂单开)
  mix_tg_health_gate: null    # 标签健康门(默认关;见 §4 防线 3)
```

校验(config 拦):tg 要求 opd_lost_only / 与 mix_rl/mix_sft 互斥 /
opd_kl_mode ∈ 白名单 / teacher_scoring_backend=fsdp / research.enabled /
loss_agg_mode=seq-mean-token-mean;health_gate 非 null 要求 tg 开、正数。
fingerprint:三键条件式(默认不进 hash,存量指纹逐字节不变)。关闭即零开销
契约:mix_tg_weight=0 不做裸打分、不构造 tg 行、无 tg_label_* 字段、无 tg/*
曲线键。

## 8. 改动清单(实施完成,commit 链)

verl 真源 trace-opd:f783026 → 608b6aa(tg 臂首版)→ 0d653b0+037c3cf(两轮
审阅修复)→ 01a9a06(频率加权)→ 0a4bb30(解耦:双通道/独立公式/白名单/
forward 还原/obs 跳过)→ **189d789(二轮实施审阅修复批,§11.5)**。
vendored = 189d789 导出(VENDORED.md 对账)。

trace_opd mixed-objective-r2:b4e97a5 → … → cff8a45(runbook v1)→ cc419b0
(解耦落地 + 测试 278→284)→ 442200b(判据件,284→286)→ 8c83246(B3 段分解
脚本,286→289)→ 5beecec(文档批)→ **二轮审阅修复批(289→298 全绿)**。

## 9. 验证实验(T1,预注册判读;rkl 底座)

口径:16×8、同 caps、`save_freq: 2`、≥30 步;底座 = 既有 rkl_seqmean 配置
(与崩塌实证直接单变量对照);跑单经 cd009-cd006 交 CD-006,细则见 runbook。

- **主判据(病的终点)**:response length / p95 / resp_clip——step-9 相变是否
  推迟/压平;崩塌步 > 基线×1.5 或预算内不崩 ⇒ 正向。
- **锐判据(枢纽变量)**:won/tg 标签 `thought_chars_per_turn`——基线 ×6
  (315→1986);T1 预测平坦或显著压缓。在线曲线 + 离线 won_traj_evolution 双
  口径。
- **效果判据**:裸 val 三件套(峰值高度/时刻/峰后形态;±5pp 误差棒)。
- **tg 曲线族**:tg/loss 收敛、tg/rows 与 tg_role 随胜率、
  tg/loss_min_preclamp ≥ −1e-6、tg/loss_max 回落、dropped 计数。
- **机理判据**:lost 区 KL 段分解(§4.3)——thought 段份额应下降/持平;
  teacher 自走成功率(K=8)不降(判停表:骤降=溢出候选,记录报回)。
- 追加臂:T1′(归因)、T3(标签 one-hot CE 退化版:分布匹配 vs token 匹配净
  贡献;亦是双盲对照中 codex 硬标签路线的消融)按 T1 结果决定。

## 10. 已知边界与风险(对策已配)

- **P1 OPD 数据集中到 teacher 最不可靠区**(失败前缀 OOD):tg 接地经泛化改善
  ;OPD 健康族曲线监控。**训练(won 前缀)/服务(lost 前缀)分布错位是本方案
  最大泛化假设,§4.3 段分解是其直接检验。**
- **P2 学生在成功轨迹零直接梯度**:保持率哨兵;后备 = 小 bare-won CE(RFT)。
- **P3 contextual collapse 溢出 / 特权保持**:tg 的标签在 action 位与 doc 推荐
  一致 ⇒ doc 帮助教师预测标签,attention 利用被强化而非削弱(结构论证);
  won 限定 + doc 持续刷新是结构对冲;teacher 自走监控为哨兵;后备 = 降 w_tg /
  间歇施加。
- **P4 次优固化**(won≠每步都优):探索期接受;后备 per-step 加权。
- **P5 失败条件化选择偏差**:T1′ 归因臂即对照;后备 = P2 的 RFT anchor。

## 11. Codex 深审处置台账(gpt-5.6-sol,2026-08-02;全文见会话记录)

| # | 裁决 | 处置 |
|---|---|---|
| P0-1 行为曲线口径(tg 排除反了) | 属实——0d653b0 过度修复 | revert:tg 行回行为曲线;GRPO 隔离与 padding 修复保留 |
| P0-2 log_prob_min_clamp 破坏 tg soft-CE | 属实(真数学缺陷) | 首修 = 行级豁免;**解耦后升级为结构性**(独立公式无任何 OPD clamp,forward 公式整段还原)。极端复核:无 1/q 项,per-logit 梯度 ≤1(q=6e-16 实测 1.0/2.0),fp32 无溢出,grad_clip=1.0 兜底;回归测试锁定 |
| P0-3 tg×research 曲线键集异构 | 属实 | research replica 补传 mixed_tg |
| P1-1 渐进停机不成立 | 属实 | 用户裁定频率加权(已实施) |
| P1-2 min_signal_rows 漏 tg 信号 | 属实 | 计数并入 tg_mask |
| P1-3 GRPO 隔离门控不可达 | 属实 | _mixed_objective_enabled 纳入 tg_loss_coef |
| P1-4 tg 监控盲区 | 属实 | telemetry 首批 + 标签质量曲线/段分解已实施;context 探针经用户裁定**先不管** |
| P1-5 token-mean 回退非同一 estimand | 属实 | config 拒绝 tg×token-mean |
| P2-1 sentinel 对 renorm 失效 | 属实 | 挂账 |
| P2-2 tg 关闭仍分配零 tg_mask | 属实(无数值影响) | 挂账 |
| 总判"不可进 T1" | 接受 | P0 全修 + 解耦后就绪 |

## 11.5 Codex 二轮实施审阅台账(gpt-5.6-sol max,fresh 线程,2026-08-02;全文见会话记录)

对解耦实施批(0a4bb30/cc419b0→8c83246)的对抗审阅:3 P0 + 6 P1 + 2 P2,
**全部核实属实、全部修复**(verl 189d789 + trace_opd 对应批,测试 289→298):

| # | 缺陷 | 处置 |
|---|---|---|
| P0-1 | tg 谓词生产/消费不一致:占位行(sample_role="tg"+teacher_mode="tg_skip")触发消费端索要不存在的标签通道 ⇒ 占位-only 分片前向 RuntimeError/collective 挂死 | 消费端谓词统一 `teacher_mode=="tg"`(与打分器 has_tg 同源);占位-only/混排两用例入测试 |
| P0-2 | `_chunked_topk_log_probs` 回写 bf16:~1e-3 log-prob 误差把近收敛 KL(~1e-5)翻负,clamp_min(0) 恰在 tg 精修区杀 loss+梯度 | helper 增 out_dtype(默认 None=历史行为,OPD 基线逐字节不动),tg 全程 fp32;bf16 近邻分布测试(新路径非负对拍 fp64、旧回写复现负值) |
| P0-3 | 条件发射的 MIN/MAX Metric 在动态装箱下各 rank 计数不同,aggregate_dp 等长断言崩训练(tg telemetry 与全 tg 微批的 OPD 极值双向触发) | 等长断言只留 SUM/MEAN(矩阵均值真需要);MIN/MAX ragged flatten 取极值(数学等价,既有等长场景行为不变);ragged 测试 |
| P1-1 | 全 tg 微批 overlap 空集 `.mean()`=NaN 污染 OPD 诊断 | 空集缺席而非零(与 mass 诊断同契约) |
| P1-2 | 段分解脚本解析不了真实目录名 `step_00004`(测试用了假名) | 严格正则 `^step_?0*(\d+)$`;测试改真名 |
| P1-3 | resp_hash 当行主键:相同短回答跨样本合并覆盖 | 观察包新增稳定 `row_uid`(微批指纹+行号,SP 组内一致);reader 优先 uid、旧包回退 (task,hash) 复合键 + 元数据冲突 fail-loud |
| P1-4 | 脚本忽略 response_mask、SP 覆盖残缺静默出完整表 | 统计条件 = mask ∧ covered ∧ finite;残缺行记 partial 剔除并计数入表 |
| P1-5 | 在线 cpt(chars÷turns=turn 加权)与离线基准(逐轨迹均值)非同一 estimand,284–315 健康区判读会错位 | 新增行级 `tg_label_thought_cpt` 键:mean(cpt)/mean(tg_role) = 标签行 trajectory-mean,与离线基线同 estimand 直用;chars/turns 留作总量视图 |
| P1-6 | TG-only(opd 0、tg>0)合法消融被两层 objective-presence 判断误拒 | 两处判断纳入 tg 系数 |
| P2-1 | health_gate=inf 过校验但永不拦截,fingerprint 却记录门已配置 | isfinite 校验 |
| P2-2 | verl 字段注释仍写公式复用旧语义 | 注释同步解耦 |

审阅同时独立确认:vendored 与真源逐字节一致、forward 公式还原、gather 链通、
rkl 零占位有限、关断零回归、rkl 底座 top-64 映射——解耦核心正确性无恙。

## 12. 双盲验证台账(2026-08-02)

用户令 codex(gpt-5.6-sol,max effort)在完全防火墙下(fresh 线程、空目录、
自足任务书、零现行方案泄露)从最早观察与原始想法独立设计完整方案。产出
27974 字符(VPC-OPD),全文存档 `teacher-grounding-blind-codex-design.md`。
对照结论:**病理机理与机制内核双盲收敛**(教师上损失、学生成功数据、梯度走
教师侧、验证目标放 KL 第一槽 MLE 形、赢局限定、标签 ratchet 为头号风险、
"教师自走 vs 前缀指导"两量分离、接受不可约 KL);**主要分歧 = 同一内核的两种
标签实现**(codex:实现 token 硬 CE + 分段降权,执行器观;我方:学生 top-64
soft-CE 全 token 均匀,打分器观——形态载体、thinking 主修面)。**用户裁定:
codex 增量不采纳,方案维持本文件定稿**;其硬标签路线保留为 T3 消融定义(§9)。
