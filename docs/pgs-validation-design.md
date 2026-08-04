# PGS 验证层设计(臂 · 种子 · 功效 · 判决口径)

CD-010,2026-08-04。spec §5 留白项的补齐(spec 明确"验证层另文");执行方 = CD-006
(用户裁定),162 四卡。**预注册文档:判据在开跑前定死,跑完不改口径。**

## 1. 共同设置

| 项 | 值 | 依据 |
|---|---|---|
| 机器/卡 | 162,4 卡(n_gpus=4, rollout_tp=2 ⇒ 2 个 TP2 replica) | 用户裁定 |
| 基座 ckpt | **SFT coldstart `sft_4b_coldstart_v1/hf/step_6`**(推荐) | bare 47.4% 起点;裸 4B 起点过低全输组过多(GRPO 零梯度);与 L2 C3 严格同起点 ⇒ 三线可比 |
| 训练数据 | `trace_opd-pgc/data/rl_train_v2.parquet`(2875 任务) | doc 表全覆盖(gt_docs_rl_train_v2 与之同键);与 C3 同数据 |
| val | valid_seen(140)+ valid_unseen(134),greedy | pilot 可比口径(PG 约束) |
| batch/rollout | 16 任务 × 8 rollouts,每 step 一次全批更新 | 用户裁定;命题 C 单快照单步 |
| advantage | grpo_loo + norm_adv_by_std=False(Dr.GRPO 形态) | 审阅裁定方案 A(条件 iv 严格) |
| 文档源 | GT walkthrough doc 表(2875 条) | 用户裁定(效率优先,排除文档质量噪声) |
| 步数 | 首轮 **100 step**(≈0.56 epoch),`trainer.total_training_steps=100` 经 passthrough?否——走 `--set train.total_epochs=1` + 手动停或 total_training_steps 由 CD-006 按 C3 惯例传 | 与 C3(97 步)同量级可比 |
| val/save 频率 | `test_freq=10, save_freq=10`(--set 传) | 100 步内 10 个 val 点,拐点分辨率优先 |

## 2. 臂设计(三个 run,顺序执行)

| run | 配置 | 目的 | 步数 |
|---|---|---|---|
| **calib** | `162-pgs-calib.yaml`(steering on,θ_D=1e9 门恒关+ε=0+率界豁免 ⇒ 行为=纯 bare,只记 D/ℓ 观测) | θ_D 标定(P85)+ c 标定(截断饱和率表)+ 预算耗尽位置观察 | 3 step |
| **base** | `162-pgs-base.yaml`(steering off,纯 GRPO) | 对照臂 | 100 |
| **s1** | `162-pgs-s1.yaml`(steering on,θ_D/c 填标定值) | 处理臂 | 100 |

- base 与 s1 **同 seed、同数据序、同起点 ckpt** 配对(verl data seed 默认一致即可;两臂 run_name 不同目录隔离)。
- calib 的 D/ℓ 观测在**同一起点 ckpt 的真实目标语境**(GT doc 前缀 + 该 ckpt 的 π)采集——spec 的"禁沿用 G0 语境"由此满足。

## 3. 预注册判据(主/次分层)

**主终点**:s1 − base 的 **val won 率臂间差**(seen+unseen 合并 274 任务),取**末段
(最后 3 个 val 点)均值**。方向:s1 > base 支持机制;量级解读见 §4 功效。

**机制健康门(gating,不满足则主终点无效)**:
1. s1 的门触发率落预注册带 [5%, 25%](逐 step,允许 ≤10% 的 step 越界);
2. 报表 L1 不变量违例 = 0;
3. 训练端 W 对账:有 steer 的 step,`actor/rollout_is_weight_off1_frac` > 0。

**次级读数(方向性,不作判决)**:探针 ITT 粗读数、won-lost 分层 steering 强度、
预算耗尽位置分布(早花晚饿)、拒绝率(经验 TV(b,π))。

**安全门(任一触发 ⇒ 停机保 ckpt,按崩塌处置)**:thought 逐字重复率 >55%、
熵持续跌破 0.2、val 单点跌 >10pp。

## 4. 功效与种子(诚实边界)

- **单次 greedy val 的 MDE ≈ 7.8pp**(274 任务,既有实测底数);同 seed 配对可略降
  但不改量级。**臂间差 <8pp ⇒ 首轮无法判定**,升级动作:对末段 ckpt 追加
  val K=8 重复采样(把单点噪声 ±5pp 压到可分辨 3-4pp)。
- 训练 rollout 曲线 MDE ≈ 23pp(零任务重复)——**只用于崩塌监控,不判改善**。
- 首轮 1 seed(资源约束);若主终点为正且 >8pp,第二轮加 2 seed 确认后才升级
  为线路判决;若为负且机制健康门全过,同样是有效结论(采样端干预不足以传导)。
- 探针 ITT 的功效:ε=5% × 门率~15% × 每轨迹候选数 ⇒ 每 step 约 10-20 条含翻转
  轨迹;100 step 累积 ~1-2k 条,won 率差的 CI 约 ±3-5pp——**方向读数可用,
  效应量判决需累积到第二轮**。

## 5. 与 L1/L2 的可比口径

- 同起点(SFT coldstart step_6)、同数据(rl_train_v2)、同 val(274 greedy)、
  同 val 频率 ⇒ 与 C3(L2)逐 val 点可比。
- 差异声明:PGS 用 grpo_loo+Dr.GRPO(L2 用 GAE+critic)——**优化器口径不同,
  跨线比较以"各自臂间差"为准,不比绝对值**。

## 6. 升级路径(首轮之后,按结果分支)

- 主终点正 + 健康 ⇒ 第二轮:+2 seed、researcher doc 源替换 GT、B_∞/η_max 敏感性;
- 主终点平 ⇒ 查探针 ITT 与 won-lost 分层定位(干预太稀?→ 调 B_∞/θ_D;
  干预无效?→ 检视重采对表的语义质量);
- 主终点负 + 健康 ⇒ 机制结论成立(负向传导),检视 W_beh 校正后的有效样本量
  与重采内容,写线路判决。
