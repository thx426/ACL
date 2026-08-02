# Teacher Grounding 解耦与判据件实施方案(impl-plan v1,2026-08-02)

> 方案定稿(用户确认 2026-08-02):双向教学结构 + 频率加权 + 全 token soft-CE +
> 解耦 + 三判据件;T1 底座 = reverse_kl_topk(rkl_seqmean 基线复用)。
> 机制真源 `teacher-grounding-design.md`;本文件只讲怎么落地。

## A. 解耦(学生 OPD loss 与教师 tg loss 完全独立)

### A1 数据通道:独立 tg_label 字段

- 打分器 `teacher_topk_fsdp.py`:mode=="tg" 行的裸打分+renorm 结果写入
  **`tg_label_ids` / `tg_label_logprobs`**(新字段,nested jagged,与
  teacher_* 同布局同宽:`(P-1) zeros | R valid | 1 dummy`);该行的
  `teacher_ids`/`teacher_logprobs` 改为零占位(与 sft/research/tg_skip 行同款,
  保持 OPD 公式的全批 packed 对齐假设零改动)。
- 非 tg 行的 tg_label_* 为零占位行(同宽);**字段仅在 batch 含 tg 行时存在**
  (tg 关断 ⇒ 零字段零传输,逐字节现状)。
- 打分发生在模板收缩之后 ⇒ padding_utils 无需处理新字段(核验:padding_utils
  不含 teacher_* 处理;tg_mask 在 TQ put 时生成所以才需要收缩)。
- batch 内 `teacher_topk` 强制一致(打分器既有校验)⇒ 两字段同宽自动成立。

### A2 独立公式 `compute_tg_soft_ce`(fsdp/losses.py 新函数)

```
输入: student_logits [1, T/sp, V](SP 切片后)、tg_label_log_probs / tg_label_ids
      (nested → packed [1, T, K])、tg_token_mask [1, T](packed 全批)
SP:   标签与 mask 都按 dim=1 slice_input_tensor(与 OPD 公式同构)
子集: flat = 张量[0][mask[0]] 三方切子集(布尔索引 = packed 序,天然对齐)
公式: q_lp = chunked(logit.gather(ids) − logsumexp(logit))   # 无 [.,.,V] buffer
      ce_t = Σ_k exp(l̄_k) · (l̄_k − q_lp_k)                  # fp32
输出: scatter 回 [1, T/sp] 全批形状(非 tg 位置 0),键 "tg_losses"
```

- **无 log_prob_min_clamp、无 loss_max_clamp**(P0-2 豁免结构化);renorm 标签
  下逐位非负有数学保证,聚合层 clamp_min(0) 仅作 fp 兜底,telemetry
  `tg/loss_min_preclamp` 保留为不变量监控。
- 梯度 = q − p̄,per-logit 上界 2(P0-2 极端验证结论继承)。

### A3 聚合层与还原

- `compute_topk_loss`:tg 项读 `outputs["tg_losses"]`(不再读 OPD 公式矩阵);
  OPD 公式按 loss_mode 正常调度(forward/reverse_kl_topk),tg 行零占位标签的
  输出是垃圾值但有限(既有 clamp 链),response_mask 零 ⇒ 0×有限=0 无毒。
- `compute_forward_kl_topk` 的 `tg_token_mask` 参数与 torch.where 打洞**整段
  还原**(恢复纯 OPD 形态,与历史臂逐字节可比)。
- 诊断(mass/overlap/entropy)已按 response_mask_bool 聚合 ⇒ tg 行天然排除
  (核验过);microscope obs collect 对 tg 行跳过(其 OPD 矩阵值在解耦后无意义)。

### A4 配置解绑(白名单而非锁死)

- tg 兼容白名单 = **{forward_kl_topk, reverse_kl_topk}**。这不是折中:k3/gated
  族 teacher_topk=0(单列 realized 布局),tg 标签需要 top-64,而打分器强制
  batch 内 topk 一致、nested 最后一维必须同宽 ⇒ **数据布局层结构不兼容**,
  fail-loud。verl `distillation.py` 校验 + trace_opd `config.py` 校验同步改,
  错误文案写明结构原因。
- fingerprint 不变:opd_kl_mode 非默认已进 hash,mix_tg_weight/opd_lost_only
  已进 hash。
- launch_train.py:无改动(loss_mode 映射与 topk 参数既有逻辑已覆盖)。

### A5 兼容性红线

tg 关断(mix_tg_weight=0)时全链路逐字节现状:无新字段、不调新公式、
`_stamped_global_seq_info` 回退 lost-only 分母、无 tg/* 曲线键。

## B. 判据件

### B1 标签质量曲线(必带,预注册锐判据的在线版)

- tg 转换点(agent_loop.py won 分流处)对该轨迹算 `thought_chars_per_turn`
  (口径与 `scripts/won_traj_evolution.py` 完全同源:steps[].thought 字符和 /
  assistant turn 数);曲线键 `tg_label_thought_cpt`(_MIXED_TG 族,批均值 =
  当步 tg 标签健康度);占位行(dropped)不计入。
- 基线锚:rkl 基线 won 行 315(step5-7)→ 1986(step17-20);T1 预测平坦。

### B2 标签健康门(实现、默认关)

- config 键 `mix_tg_health_gate: float | null = null`(null=关;数值 = 该轨迹
  thought_chars_per_turn 超阈则**不转换**,以占位行退出,计数
  `tg_dropped_unhealthy` 入曲线)。
- 进 fingerprint(条件式:非 null 才入 hash——它改变"优化什么")。
- 校验:health_gate 非 null 要求 mix_tg_weight>0。

### B3 lost 区 OPD-KL thought/action 段分解(离线脚本)

- `scripts/opd_kl_span_decomp.py`:基于 microscope obs dump(per-token 师生
  分布量)+ rollout 的 turn/segment 结构,输出 lost 行 OPD-KL 的 thought 段 /
  action 段均值与份额随训练步曲线——良性循环第二环(教师在输局前缀上的目标
  是否变健康)的直接读数。实施时先核对 obs dump 与 span 数据的实际契约,
  数据不足则回报缺口(不造数)。

## C. 实施顺序与验收

1. verl 真源(worktree `../verl-mixob`,trace-opd 分支):A1 打分器 → A2 公式
   → A3 聚合/还原 → A4 verl 校验;
2. verl 侧测试改造+新增,全绿;commit;`git archive`(绝对路径)重导出
   vendored + VENDORED.md;
3. trace_opd 侧:A4 config 校验/文案 → B1 曲线 → B2 健康门 → 测试全绿;
4. B3 脚本;
5. 设计文档 §1–4 重写(定稿陈述)+ 双盲对照台账;runbook v2(T1 底座改 rkl
   复用、判据/冒烟表更新);changelog v1.3 发 CD-006;
6. 分两个 commit:机制解耦+判据件;文档批。

测试要求(在既有 278 上净增):双通道布局(tg 行 teacher_* 零占位/tg_label_*
有值,OPD 行相反)、rkl_topk×tg 组合分流正确、k3×tg 拒绝、soft-CE 手算对拍新
公式、tg 行垃圾值不入 OPD 损失(0×有限)、SP 切片对齐(可行则加)、健康门
开/关行为与 fingerprint 条件式、曲线键批一致性(P0-3 纪律)。
