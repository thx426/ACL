# PGS 机制说明(面向实验运行方 CD-006)

CD-010,2026-08-04。**这不是规范**(规范=`docs/pgs-final-spec.md` + 其末尾实现裁定
注记);这是给运行/监控实验的人的自足说明:训练里到底发生了什么、每个观测量是
什么意思、异常长什么样。

## 1. 一句话

**PGS = 特权文档只进采样端**:rollout 时,带 GT 文档的"教师视角"对学生正在生成的
token 流做**有界的分布倾斜**(教师实质异议处轻推几下),训练损失是**纯 GRPO**
(reward = 环境 outcome,损失里没有教师、没有蒸馏项、没有 critic);倾斜造成的
采样偏差由逐 token 权重 W_beh 精确纠偏。与 L1(特权进损失端)/L2(特权进价值端)
正交。

## 2. 一条轨迹里发生了什么(六件套流水)

每条训练轨迹(validate 不干预)、每个生成轮:

1. **DRAFT**:学生策略 π 正常全速生成一段(vLLM 引擎照旧,顺带带出每步 top-256
   候选明细)。
2. **VERIFY**:把「教师前缀(同 scaffold + `## Task guidance` + GT doc)+ 已生成
   响应流 + 这段 draft」发给同一引擎做一次打分前向,拿到教师分布 p_T 的逐位置
   top-256。**教师就是同一个模型加了文档**——不是外部大模型。
3. **逐位置判定**(对 draft 段每个 token):
   - 构造候选集 S = π 的 top-64 ∪ p_T 的 top-64(固定,不看 draft token——因果红线);
   - 算异议度 **D = Var_{π|S}[ℓ̃]**,ℓ̃ = clip(log p_T − log π, ±c):教师与学生在
     该位置候选上的分歧有多实质;
   - **D 门**:D > θ_D 才干预(θ_D 标定 run 的 P85,≈15% 位置开门);**ε=5% 探针**
     随机翻转门的决定(为因果读数留随机对照);
   - 门开 ⇒ 从 **D_∞ 预算**(每轨迹 B_∞=10 nats)领强度 η(默认整额 0.5,余额
     不足自动缩水,耗尽归零);
   - **speculative 判定**:按保尾核 b_S(S 内几何倾斜、S 外恒 π)对 draft token
     做接受/拒绝;拒绝 ⇒ 在 S 上按残差分布重采一个 token(**这就是"教师把
     policy 想说的改成了别的"**),丢弃 draft 后缀、从断点续 DRAFT;
   - 每轮拒绝到 **R_max=4** 次后该轮余下不再干预(防抖)。
4. **记账**:每个门参与位置落一行账(log π/log b/η/D/门状态/探针/draft→final
   token/预算余额),随轨迹 dump 落盘并进训练批。
5. **训练**:纯 GRPO(LOO baseline,Dr.GRPO 无 std 归一化);采样偏差由
   **W_beh = π/b** 逐 token 乘在 PPO clip **之外**(verl 原生 rollout_is_weights
   通道)——倾斜过的 token 的梯度被精确折算回 on-policy 期望。

## 3. 关键概念速查

| 概念 | 是什么 | 为什么 |
|---|---|---|
| 保尾核 b_S | 实际采样分布:S 内按 e^{ηℓ̃} 倾斜重分配(保 S 总质量),S 外恒 π | 全词表密度比 ≤ e^{2ηc}≈12 倍 ⇒ W_beh 有界,训练端安全 |
| q-kernel 口径 | 核构造于引擎采样分布 q(=draft 返回的 logprobs) | speculative 精确性要求接受分母=真实提案分布;q≈π_old 残差与普通 GRPO 忽略的失配同类(P4 实测) |
| D 门 | 只在教师**实质异议**处干预 | 处处干预=回到蒸馏覆写;门率 15% 是"教师偶尔插话" |
| ε 探针 | 5% 概率随机翻转门决定,倾向得分记账 | 天然随机对照 ⇒ 干预的因果效应可估(ITT 读数) |
| D_∞ 预算 | 每轨迹 Σ(2ηc) ≤ B_∞=10,先算后花 | 轨迹级 IS 权重严格 ∈ e^{±10};干预总量可控可审计 |
| W_beh | exp(log π − log b),稀疏账展开成稠密权重 | 把 b 采样的轨迹折算回 π 的期望(分离式:PPO clip 只管 π_θ/π_old) |
| 标定三件套 | calib run:θ_D=1e9+ε=0+率界[0,1] ⇒ 行为纯 bare 只观测 | 在目标语境测 D 分布(定 θ_D)与 ℓ 分布(定 c) |

## 4. 观测量怎么读(swanlab 实时 + 报表离线)

**swanlab(step 级,`rollout_extra/*` 与 `actor/*`)**:
- `steer_count`(每轨迹倾斜次数均值):预期 ~2-4(B_∞=10 下最多 4 次整额);恒 0=没干预(查 doc/门率),暴涨=预算异常
- 门率(报表算;swanlab 看 `steer_n_base_open`/`steer_n_candidates` 均值之比):带 [5%,25%],滑出=标定失效告警
- `steer_gate_alarm`:>0 的 step 有率界告警
- `actor/rollout_is_weight_off1_frac`:**W_beh 通路对账**——有 steer 的 step 必须 >0;=0 即权重静默丢失,立即停查
- `steer_reject_count`/`steer_n_pt_window_miss`:拒绝与窗外代入,趋势观察
- 崩塌四件套:`entropy`、响应长度、val won、截断率(既有口径)

**pgs_report.py(分布层,日常一跑)**:红绿灯头部(不变量违例必须 0)+ 四区
(详见 `docs/pgs-observability-design.md`);最有信息量的一张表是 **top
「draft→final」重采对**——教师把 policy 的什么 token 改成了什么,直接肉眼判断
干预语义是否合理。

**pgs_traj.py(个案层)**:异常轨迹逐 turn 检视,steer 行内联标注。

## 5. 已知边界(判读时须知)

- **OPD 能力边界**:采样端倾斜只能放大 π 已有质量的行为(π 极冷 token 抬 12 倍
  仍是冷的)——若任务需要全新动作,PGS 不解决(SFT 冷启动的既有结论)。
- 训练 rollout won 曲线**判不了改善**(MDE 23pp),只盯崩塌;改善看 val 臂间差。
- val greedy 单点噪声 ±5pp;臂间差 <8pp 首轮无法判定(升级路径见
  `docs/pgs-validation-design.md` §4)。
- 探针 ITT 首轮只作方向读数,不作判决(功效不足)。
