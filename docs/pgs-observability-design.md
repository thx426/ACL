# PGS 观察体系设计(记录 · 分析 · 可视化)

CD-010,2026-08-04。设计原则:**从「要回答的问题」倒推「要的数据」再倒推「呈现形式」**。
所有数据源均已在机制实现中落盘(spec §4 + 审阅修复 + 观察补齐 commit 9032910);
本文定义其组织与消费方式。

## 0. 数据源清单(单一真源:代码侧字段)

| 通道 | 内容 | 落点 | 粒度 |
|---|---|---|---|
| steering 稀疏行 | position/log_π/log_b/η/δ/D/门状态/探针/propensity/拒绝/cap/预算余额/**draft→final token** | rollout dump `steering.*` + 训练批 extra_fields | 逐「门参与」位置 |
| D 直方图 | Var_{π\|S}[ℓ̃] 分布(64 桶,上界=c² 随账落盘) | 同上 | 全候选位置 |
| ℓ 直方图 | 未截断 log p_T−log π(±12,96 桶,窗内逐 token 等权) | 同上 | 全候选位置×S 窗内 token |
| 计数器组 | 候选/开门/探针翻转/预算阻断/旁路/窗外代入 | 同上 | 轨迹级 |
| 轨迹汇总 | budget_spent、turns_capped、steer/reject/record 计数 | 同上 | 轨迹级 |
| 轨迹本体 | 全部 thought/action/obs 文本、turn_token_spans、outcome/won/truncated | rollout dump 顶层 | 轨迹级 |
| swanlab 标量 | won/steer_count/budget_spent/d_mean/reject/probe/window_miss/doc_missing 等 per-step 均值 | 云端曲线 | step 级实时 |
| 训练端对账 | `actor/rollout_is_weight_mean` / `off1_frac`(W_beh 真实抵达 loss 的证明) | swanlab | step 级实时 |
| 审计三件套 | launch manifest / effective_config / objective fingerprint | run_dir | run 级 |

## 1. 四层问题分层(报告分区即按此组织)

### L1 机制正确性 —— 「按设计运行了吗」
- **不变量核验**(报表逐轨迹重算,违例=红灯):逐 token |log π−log b| ≤ 2·η_max·c;
  逐轨迹 Σδ = budget_spent ≤ B_∞;|Σ(log π−log b)| ≤ B_∞(命题 B 端到端);
  拒绝行必有 final≠draft;接受行必有 final=draft。
- **门率健康**:开门率(n_base_open/n_candidates)落预注册带 [5%,25%](带阴影图)。
- **数值前提**:p_T 窗外代入率(单侧鲁棒的经验前提=多数饱和,率应低);
  W 分布(exp(logπ−logb))无越界;截断饱和率 P(|ℓ|>c)。
- **通路对账**(swanlab 侧):rollout 有 steer 的 step ⇔ 训练端 off1_frac>0。

### L2 干预行为 —— 「干预长什么样」
- **何时何地**:steer 位置在轨迹内的相对位置分布(早花晚饿检测);预算耗尽点
  (budget_after≈0 的最早位置)分布;每轨迹 R_max 触顶轮数。
- **内容**(机制语义最直接的可视):top「draft→final」重采对表(教师把 policy
  的什么 token 改成了什么,带频次与 decode 文本);ℓ 方向占比(抬升 vs 压制)。
- **强度**:η 分布(整额/缩水占比)、D 分布随 step 漂移、拒绝率曲线
  (经验 TV(b,π))、每轮拒绝数分布。

### L3 效果信号 —— 「干预有没有用」
- **ε 探针 ITT**(机制因果读数的设计初衷):含「base_open 被翻关」位置的轨迹
  vs 未翻转对照轨迹的 won 配对差(正向翻开同理);逐 step 累积,报样本量与
  粗 CI(功效低,只作方向读数,不作判决)。
- **won-lost 分层**:赢/输轨迹的 steer_count、budget_spent、拒绝率差
  (教师在哪类轨迹上说话多)。
- **doc_missing 天然对照**:缺 doc 任务(纯 bare)与有 doc 任务的 won 率。
- **主终点**(判决级,验证层文档定义):base vs s1 的 val 曲线(swanlab);
  训练 rollout won 曲线只盯崩塌不判改善(MDE 23pp,既有结论)。

### L4 崩塌监控 —— 「会不会重演 G0」
响应长度分布与漂移、截断率、thought 逐字重复率(最强判别指标,轨迹内
相邻 thought exact-dup)、val 拐点(swanlab)。熵曲线在 swanlab(训练端)。

## 2. 工具形态

### `scripts/pgs_report.py` — 机制健康报表(离线聚合,可增量)
```
python scripts/pgs_report.py --run-dir <run> [--out report.html]
       [--max-steps N] [--stride K] [--tokenizer <hf-path>]
```
- 输入 rollout dumps(train;val 只取 won 供 L3/L4),输出**自包含单页 HTML**
  (matplotlib 图 base64 内联;无 matplotlib 时降级为表格版)。
- 页面结构:红绿灯头部(不变量违例数/门率在带内/告警计数/覆盖 step 范围)
  → L1 → L2 → L3 → L4 四区 → 单轨迹检视入口(每 step won/lost 抽样的
  pgs_traj 命令行)。
- 增量:重跑即重渲(分钟级);训练中远程使用:在 162 跑完 scp 回,或本机
  拉 dumps 后跑。实时标量层由 swanlab 云端覆盖,本报表管分布/内容层。

### `scripts/pgs_traj.py` — 单轨迹检视器(人工逐例抽查)
```
python scripts/pgs_traj.py <dump.json> [--tokenizer <hf-path>] [--out traj.html]
```
- 逐 turn 渲染 thought/action/obs;有 tokenizer 时在 response 文本内**内联
  高亮 steer token**(hover:draft→final、η/D/W、accept/reject、预算余额、
  探针标记),无 tokenizer 时按 turn 给 steer 行表(token 显 id)。
- 用途:机制正确性人工抽查(重采是否语义合理、位置是否对齐)、案例采集。

## 3. 与标定/验证流程的衔接

- **标定 run**(theta_d=1e9+epsilon=0+rate_bounds=[0,1],3 step)产出即含
  D/ℓ 双直方图 ⇒ `calibrate_theta_d.py` 一次给出 θ_D 建议(P85)与候选 c
  饱和率表 ⇒ 人工定 c 后填 s1 配置。
- 正式 run 期间:swanlab 盯实时标量(门率/won/熵/W 对账),每日跑一次
  pgs_report 看分布层,异常时 pgs_traj 抽查个案。
- 验证判决数据(臂间差/探针读数)以报表 L3 区 + swanlab val 曲线为准,
  判决口径见 `docs/pgs-validation-design.md`。

## 4. 刻意不做的(边界)

- 不做训练端逐 token 显微(旧 microscope 形态):PGS 的机制信号全在采样端,
  训练端只需 W 通路对账(off1_frac);loss 级显微是旧蒸馏机制的需求。
- 不做在线自动干预(告警降级等):v1 冻结语义,观察体系只观察、不反馈。
- ITT 读数不作首轮判决终点:ε=5%×门率 15% 下功效有限,累积样本够了才升级。
