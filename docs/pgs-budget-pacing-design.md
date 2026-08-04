# PGS 预算均衡化设计(干预位置坍缩的修复方案,待用户裁定)

CD-010,2026-08-05。触发:s1 首查实测干预位置坍缩(下)。配套:θ_D 自适应
分片死锁修复(gate.py,同日已 commit,独立于本文方案)。

## 1. 问题(s1 step 1–17 实测,2176 轨迹)

- 1158 次拒绝重采(唯一真改 token 的干预)中 **98.7% 落在 turn 0**,位置中位
  数在轨迹**前 0.4%**(P99 也只到 6%)。机制实际形态 = 只修饰第一段 reasoning
  开头,其后 97% 的轨迹(全部中后段决策分叉点)零覆盖。
- 因果链:θ_D 冻结在初值(已修)⇒ verify 窗口内门率恒 ~40% × **预算贪心顺序
  花费** ⇒ 8 次额度在前 ~50–100 token 花光 ⇒ 其后整轨迹 bypass(candidates
  仅 ~20/轨迹,即截断窗口)。
- 数字可溯:`/data4/ws/PGS/pgs_s1/rollouts/step_000*`,审计脚本
  `/tmp/rewrite_audit.py`、`/tmp/pos_audit.py`(162),改写明细
  `/tmp/pgs_rewrites.json`。

## 2. 结构约束(方案必须尊重)

1. **流式因果**:token 顺序生成、verify 顺序进行,不存在"看完全轨迹再挑最大
   D"的选项(那要求全轨迹教师 verify + 事后置换,破坏 speculative 恒等)。
   任何方案都是**在线决策**。
2. **IS 有界**:Σδ ≤ B_∞ 的总预算约束不可破(W_beh 界的根据)。
3. 预算耗尽即 bypass(不 verify)是性能设计(教师调用成本);方案不应要求
   全程 verify(≈100× verify 量)。
4. θ_D 率控与预算分配是**正交旋钮**:门率管"哪些位置有资格花",预算分配管
   "额度何时可用"。坍缩的直接根因是后者(贪心顺序)。

## 3. 方案空间

| 方案 | 机理 | 评价 |
|---|---|---|
| A. per-turn 硬帽 | 每 turn 花费上限(如 ≤1 次整额) | 简单;但 B=40 仍在前 ~16 turn 花光,长轨迹后段裸奔、短轨迹不省;单帽=均匀假设,turn 间无侧重 |
| **B. 匀速解锁(pacing,推荐)** | unlocked(turn)=min(B_∞, U0+rate·turn),只允许花已解锁额度;rate=(B_∞−U0)/T_max,T_max=`max_assistant_turns`(环境硬配置,非先验估计);U0=初始解锁量 | 零分布假设;**短(赢)轨迹自动少花、长(困难)轨迹覆盖到尾**;token-bucket 允许跨 turn 攒(单 turn 解锁 <δ_full 也能隔几 turn 花一次整额);U0 保住头部纠错(s1 实测头部改写含真纠错) |
| C. 头部 θ 加严(区段整形) | 前 k turn 用 κ·θ | 超参丑(κ,k),治标;头部纠错被一刀切压制 |
| D. 概率稀释 | 开门后以 p<1 概率真花 | 窗口只拉长 1/p 倍,不改变"先到先得"本质 |
| E. 预算匹配率控 | q_eff≈(B/δ)/E[候选数]≈0.4–0.8%,θ 逼到只有全轨迹极端异议过门 ⇒ 近似 top-B 选位 | 原理最优雅(额度只花在最大异议处);但依赖"极端 D 位置不堆头部"假设——**现无数据可验**(calib 旧 schema 无逐位置 D;s1 只见头部窗口),且观测口径需配稀疏 verify 探针。留二版/消融 |
| F. receding-horizon | 按剩余长度/价值动态分配 | spec §5 挂账组件,正式但重;二版 |

## 4. 推荐方案:B(pacing),参数与预期

- `budget_pacing: true`(新键,进指纹;false=现行为,兼容消融)
- `budget_u0: 5.0`(nats,=2×δ_full:头部允许 2 次整额,保任务理解期纠错)
- rate = (B_∞−U0)/T_max = (40−5)/40 ≈ 0.875 nats/turn(T_max 取自
  `episode.max_assistant_turns`,不新增键)
- 语义:turn t 开始时可用额度 = min(B_∞, U0+rate·t) − 已花。开门位置若可用
  额度 < δ 则 bypass(记 `n_budget_blocked`,与耗尽 bypass 区分)。
- 预期效应量(B=40):16 次整额机会全轨迹摊开;θ_D 修复后爬升 ⇒ 开门位置 D
  更大 ⇒ 单次拒绝率 7%→10–15%,换 token ≈ **1.6–2.4 次/轨迹**,分布全程。
- 与率控的耦合:pacing 下 verify 窗口散布全轨迹,tracker 观测口径 ≈ 全轨迹
  分布 ⇒ 标定初值 0.0338 即近均衡(口径断裂自动弥合,无需稀疏观测探针)。

## 5. 实现要点(裁定后 ~半天)

- `TrajectoryBudget`:加 `unlocked(turn_idx)` 阶梯与 `advance_turn()`;
  `remaining` 语义拆为 `total_remaining`/`available_now`。
- `hook.SpsGenerate`:turn 入口推进 turn 序号;`steerable` 判据改用
  `available_now`;记账加 `steer_unlocked_at`(审计解锁曲线)。
- 计数器:`n_budget_blocked` 区分"额度未解锁"与"总额耗尽"。
- 测试:解锁阶梯/跨 turn 攒额/短轨迹欠花/坍缩回归(mock 头部高 D 分布,断言
  花费位置分布的 P90 > 0.5 轨迹长度)。
- fingerprint:`budget_pacing`/`budget_u0` 进 objective_fingerprint。

## 6. 待裁定

1. 方案取舍:B(推荐)/ A / E / 组合(如 B+turn 内帽 ≤1 次整额);
2. U0 取值(默认 2δ=5);
3. s2 是否等本方案落地后一并起跑(建议:是——B=40 不配均衡化,只是把头部
   窗口从 0.4% 拉到 ~1%,坍缩不解)。
