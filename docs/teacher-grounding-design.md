# Teacher Grounding(教师接地)— 反向蒸馏臂设计(v1,2026-08-02)

> 状态:用户三轮探讨定稿(2026-08-02):机制方向、won 分割、seq-mean-token-mean
> 聚合、w_tg=1.0 对称分工均已裁定。本文件是实施真源。
> 分支:`mixed-objective-r2`(基点 b4e97a5);verl 真源 worktree `../verl-mixob`
> (trace-opd 分支,基点 f783026)。

## 1. 动机与病灶(用户诊断,三方证据锚定)

纯 OPD 的病:teacher 条件在 (doc, h)、student 条件在 h——在 student 前缀上,
teacher 因"知道答案"其分布系统性偏向**叙述正确推理**的 token(它无法替学生行
动,只能"说该怎么做");KL 把 student 拉向叙述分布,student 学会生成看似正确
的推理文本而非做出正确动作;叙述不推进环境状态 → 轨迹膨胀 → 撞上下文上界 →
成功率崩。证据:A0 step-17 长度失控必崩指纹;v2.2 的 23.6% SFT 样本教"零观
察依据断言";2606.04703 trajectory inflation(2.5→21.9 turns)与 WhySD 的
epistemic 通道机理。**病根:teacher 的指导分布不在"裸上下文可执行"的流形上。**

此前一切臂(含 SFT-doc)训练的都是 q 的**执行者角色**(在自己的状态分布 d^q 上
从头走成);q 在 OPD 里的真实角色是**教师**(在 student 的状态分布 d^p 上被评
估、被消费)——教师角色此前从未被直接训练,其在学生前缀上的可靠性只靠跨分布
外推,而 CD-003 实证外推恰好在此断裂(自走 87.5% vs 错误前缀零抓地力)。

## 2. 机制定义

**Teacher Grounding(tg)**:首次按 q 的真实角色训练 q——在 bare student 的
**成功轨迹**上,以裸分布为软目标,训练带 doc 的 teacher 前向:

```
L_tg = mean_{τ~p, won=1} [ (1/|gen|) Σ_{t∈gen} KL( sg[p_θ̄(·|h_t)] ‖ q_θ(·|doc,h_t) ) ]
```

soft-CE 形式:梯度 −Σ_v p̄(v)∇log q_θ(v|doc,h_t),只经 teacher(带 doc)前向
回传;p̄ 取 rollout 权重的裸分布 top-64 重归一化(student 在自己轨迹上 top-64
质量实测 ~0.9995,截断偏差可忽略),detach。

**won 分割**(用户裁定):成功轨迹退出 OPD,失败轨迹退出 tg——

```
L = mean_{won=0}[ L_OPD ] + w_tg · mean_{won=1}[ L_tg ],   w_tg = 1.0
```

两项均 **seq-mean-token-mean**(行内 token-mean → 行间 batch 全局 mean),各带
自己的 batch 全局行数分母(见 §4)。分工:**失败区 doc 出手教学生;成功区学生
校准老师**。每状态区单向信息流,无同状态对冲;won 率上升 ⇒ OPD 数据自动减少、
tg 数据自动增多——**渐进停机由数据分布内生实现**。

**机理预测**(实验判据的来源):①指导分布被拉回裸可执行流形 → 长度相变推迟/
消失;②成功区 PMI 推力显式归零 → "信号耗尽→噪声传递→崩"改写为"耗尽→归
零→平台";③reward 经 won 过滤(RFT 式数据选择)进入蒸馏环——最稳的锚形式,
规避 AntiSD"裸加 RL 锚仍崩"的坑;④teacher 的 in-context 文档使用能力被主动
维护(直接回应 2606.04703"global 注入损毁经验使用能力")。

## 3. 数据流:与纯 OPD 的镜像对称(实现的骨架)

| | 打分前向(no-grad,fsdp 打分后端) | 训练前向(grad) | label | loss |
|---|---|---|---|---|
| OPD 行(won=0 bare) | **带 doc** 前缀 → teacher top-64 | **裸**序列(student) | teacher top-64 | forward_kl_topk(现状不动) |
| tg 行(won=1 bare 转换) | **裸**前缀 → p̄ top-64 | **带 doc** 前缀序列(q) | p̄ top-64(重归一化) | tg soft-CE(新) |

每行一次打分 + 一次训练前向,**总算力与纯 OPD 持平**。tg 行的序列构造完全复用
SFT 臂的"doc 前缀 + 轨迹重放"构造器(response 流构造不依赖 system prompt 有无
doc,单测已锁的同构性质);label 传输复用 teacher_ids/teacher_logprobs 通道的
机械(独立字段名 `tg_label_ids`/`tg_label_logprobs`,防语义混淆)。

**行的容器形态**(照 expert 行先例):tg 行 `response_mask` 全零(不进 OPD/RL
的任何分子分母、microscope 自动跳过)、新 `tg_mask` 标生成 token、
`sample_role="tg"`。won=1 的 bare 行**不再以裸形态 put**(opd_lost_only 语义,
见 §5);won 率等行为指标一律从 rollout 侧全量统计(既有 critic/score/mean 口
径),不受 put 侧分流影响。

**doc 配对**:tg 行用该轨迹 rollout 当 step 现场产出的同一份 doc——机制本就
per-step 现产、同 step 消费,无陈旧配对。

## 4. seq-mean-token-mean 聚合与分母(沿用 f783026 纪律)

- OPD 项:现状不动——f783026 的 `batch_num_seqs`(非空 response_mask 行的
  batch 全局计数)+ dp_size 补偿。tg 行 response_mask 全零,**天然不进该分母**。
- tg 项:`agg_loss(tg_losses, tg_mask, "seq-mean-token-mean",
  global_batch_size=batch_num_tg_seqs, dp_size=dp)`——引擎在
  forward_backward_batch 处新打戳 `batch_num_tg_seqs`(非空 tg_mask 行的
  all-reduce 全局计数),fail-loud 契约与 batch_num_seqs/research_num_tokens
  同款。微批边界自动正确:无 tg 行的微批贡献 0(分子零/全局分母非零);全 tg
  行微批的 OPD 项走全局分母(f783026 已修)。
- 任一区行数为 0 的 step:该项恰为 0(数据层面;不需要 min_rows 特判——但
  won 全无时 tg 打分/构造整体跳过,见零开销契约)。

## 5. 配置与开关(trace_opd 侧)

```yaml
train:
  mix_tg_weight: 0.0        # tg 项权重;>0 时启用 tg 行构造+裸打分。裁定值 1.0
  opd_lost_only: false      # true: won=1 的 bare 行不进训练 batch(OPD 只训失败轨迹)
```

- 语义正交,消融矩阵干净:T0 = 全默认;**T1 = {mix_tg_weight: 1.0,
  opd_lost_only: true}**(主验证臂);T1′ = {mix_tg_weight: 0,
  opd_lost_only: true}(归因臂:分离"切成功区 OPD"与"加 tg")。
- 校验:`mix_tg_weight > 0` 要求 `opd_lost_only: true`(won 行已转换为 tg 行,
  不允许同一轨迹既进 OPD 又进 tg——首版不支持"全轨迹 OPD + tg 附加"形态);
  与 `mix_rl_weight>0` / `mix_sft_weight>0` **互斥**(won 行退出 response_mask
  会改变 RL 组语义/SFT 数据语义,组合形态待后续裁定);research_rl 正交不受限。
- fingerprint:两键**条件式**进 hash(非默认才进),旧臂 resume 零影响。
- **关闭即零开销契约**:mix_tg_weight=0 时不做裸打分、不构造 tg 行、不传
  label 字段;opd_lost_only=false 时 won 行照旧,逐字节现状。
- verl 侧新键:`distillation_loss.tg_loss_coef`(launch_train 由
  mix_tg_weight 传入)+ `opd_lost_only` 不进 verl(put 侧行为,trace_opd 实现)。

## 6. verl 真源改动清单(trace-opd 分支,worktree ../verl-mixob)

1. `trainer/distillation/fsdp/losses.py`:新增 `compute_tg_soft_ce`(与
   compute_forward_kl_topk 同构:label top-64 **重归一化** + soft-CE/KL 形式
   `Σ p̄_norm(log p̄_norm − log q)`;输出 per-token 矩阵 + label_mass 诊断)。
2. `trainer/distillation/losses.py`:
   - logits processor 路由:微批内按行类型分发(response_mask 非空行走现行
     OPD 公式;tg_mask 非空行走 tg soft-CE;两类行相斥,合并进同一 per-token
     矩阵的各自行);
   - `distillation_ppo_loss` 加第四项(照 sft/research 模式):
     `tg_loss_coef > 0` 时 `tg_loss = agg_loss(tg_losses, tg_mask,
     loss_agg_mode, global_batch_size=batch_num_tg_seqs, dp_size)`,fail-loud
     缺字段;Metric:`tg/loss`、`tg/rows`、`tg/label_mass`。
3. `workers/config/distillation.py`:`DistillationLossConfig.tg_loss_coef`。
4. `workers/engine/fsdp/transformer_impl.py`:forward_backward_batch 打戳
   `batch_num_tg_seqs`(照 batch_num_seqs 同址同款 all-reduce)。
5. 打分后端:裸前缀打分模式(现有带 doc 打分函数的 prefix 参数化;R11 fsdp
   打分路径复用)。

## 7. trace_opd 侧改动清单

1. `common/config.py`:两键 + 校验(互斥/依赖)+ fingerprint 条件式。
2. `train/agent_loop.py`:put 分流——won=0 照旧;won=1 且 tg 开:裸打分取
   p̄ top-64 → SFT 构造器造 doc 前缀序列 → 装配 tg 行(tg_mask/label/
   sample_role,response_mask 清零)→ put;won=1 且仅 opd_lost_only:不 put。
3. `scripts/launch_train.py`:tg_loss_coef 传参。
4. 曲线:`tg/loss`、`tg/rows`、won 率(rollout 侧)、OPD-KL 按 won/lost 分层
   (分割后 OPD 全为 lost,该曲线退化为存在性核验)、context-sensitivity 探针
   (KL(q(·|doc)‖p(·)) probe 集,won/lost 分层——溢出报警哨兵)留二期接
   microscope。

## 8. 验证实验(T0 vs T1,预注册判读)

口径:16×8、同 caps、同 val 三件套,≥30 步(覆盖 A0 崩塌窗口 + 延后观察);
GPU 等 162 空窗,跑单经 cd009-cd006 交 CD-006。

- **主判据(病的终点)**:response length / p95 / resp_clip——T1 的 step-9 长
  度相变是否推迟/压平;崩塌步数 > T0×1.5 或预算内不崩 ⇒ 正向。
- **效果判据**:裸 val 峰值高度/时刻/峰后形态(平台化 vs 不可逆崩)。
- **机理判据**:①tg/loss 下降且 won 区(即 tg 行)soft-CE 收敛;②探针 KL
  lost 区不被动收缩(塌缩溢出报警 → 降 w_tg 或转间歇);③teacher 自走成功率
  (K=8 doc 注入)不降;④won 任务保持率(P2 哨兵:学生成功行为不流失)。
- **失败预案**:长度无差异 ⇒ 查 tg 行数量/权重;lost 区探针塌 ⇒ 溢出主导。
- 追加臂:T1′(归因)、T3(CE 退化版:label 换实际 token one-hot——分布匹配
  vs token 匹配的净贡献)按 T1 结果决定。

## 9. 已知边界与风险(对策已配)

- **P1 OPD 数据集中到 teacher 最不可靠区**(失败前缀 OOD):tg 的接地经泛化
  改善之;OPD 健康族曲线(overlap/grad_norm)监控;既有护栏正交可加。
- **P2 学生在成功轨迹零直接梯度**:保持率哨兵;后备 = 小 bare-won CE(标准
  RFT)项,探索期不加。
- **P3 contextual collapse 溢出**(W022 teacher-only 警告):won 限定数据 +
  doc 持续刷新是结构对冲;探针分层监控;后备 = 降 w_tg / 间歇施加。
- **P4 次优固化**(won≠每步都优):探索期接受;后备 per-step 加权。
- 曲线语义变化入档:分割后训练侧熵/OPD 曲线只覆盖 lost 行(won 行为指标一律
  rollout 侧)。
