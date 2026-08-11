# 干预剂量控制通用化(v3.4 dosage;2026-08-11 用户裁定)

**裁定**:参照 MENTOR 修复 ssb engine run 双降的机制病因,但**必须 benchmark
无关**——后续引入多种 benchmark,改进不得锚定 Spreadsheet。
**上游**:诊断 `docs/diagnosis/2026-08-11-ssb-engine-run-degradation.md`;
MENTOR 取证 `docs/notes/2026-08-11-mentor-dosage-comparison.md`。

## 0. 通用化原则

老方案的病根本身就是 benchmark 锚定:`entropy_theta_init: 1.28` /
`entropy_floor: 0.6` 是**绝对 nats 常数**(ALFWorld-4B 语境标定),换任务/换
模型熵分布整体平移即全错——本次事故正是占位值沿用 + EMA 慢收敛引爆
(门率全程 0.25–0.37,3× 于目标 0.10)。通用化手法:**一切阈值改为无量纲量**
——分位数(相对当前分布)、比率、以及"以本 run 自己的首次标定为锚"。
不引入任何需要 per-benchmark 标定的新常数。

## 1. 改动四件套

### 1.1 熵阈值分位自标定(gate.py / config.py / agent_loop.py)

| 键(steering.*) | 新默认 | 语义 |
|---|---|---|
| `entropy_theta_update` | `direct` | θ ← 每 step 边界分位直标(quantile_{1−rate},MENTOR 每步 set 同款,无 EMA 时延);`ema` = 历史行为完整复现档 |
| `entropy_theta_init` | `0.0` | **0 = 自标定**:θ 以 `THETA_WARMUP=1e9` 哨兵起步 → 首个 step 边界从真实熵分布落地(MENTOR 首步 γ_p=999 同款);>0 = 显式起点(已标定场景) |
| `entropy_theta_ratchet` | `true` | θ 不低于首次标定值:策略熵坍缩时 w=H/θ 全体下降、教师自动淡出,而非阈值追着坍缩把门变宽。**取代绝对值 floor**(仅 direct 档生效) |
| `entropy_floor` | `0.0` | 仅 ema 档有语义;direct 档配 >0 直接拒(校验) |

**warmup 哨兵的安全性**(tests/steering/test_dosage_control.py 锁定):
θ=1e9 时 `w0 = H/θ ≈ 1e-9 < w_min` ⇒ 全 bypass、**零 rng draw**(软档
short-circuit:effective=0 不调 rng)、零干预——warmup 期 steered 臂行为与
bare 逐位一致,且 G-E 段级 bypass 跳过教师 verify(零开销)。ent_hist 照常
积累(执行位熵与门开关无关),攒够 `min_obs` 即落定。哨兵取有限大数:
JSON/wire 可传输;dump 落账口径:`steer_entropy_theta = -1.0`(哨兵期,
防 1e9 拉爆曲线量程)。

**观测源注**:熵样本来自 steered 行执行位的 draft 熵(=π 的熵,无偏;轨迹
前缀为 b 诱导分布,修复后干预率 ~5% 时二阶小)。MENTOR 从 on-policy 组
统计,同有滞后一步偏差;若后续要求更严,可换 bare 行统计(需 bare 行加
熵记账,本轮不做)。

### 1.2 预注册率带告警接通熵门(防线缺口修补)

spec 的 [5%,25%] 干预率防线此前只接 `gate_mode=adaptive`(agent_loop.py
rate_bounds 三元条件),熵门不在环上——事故超标 24 步零告警。现
`GateThresholdTracker` 每次 step 更新时用本步直方图估计**旧 θ 生效期的
非零干预率** P(H ≥ w_min·θ),超 `gate_rate_bounds` 即 `logger.warning` +
`last_info["alarm"]`(warmup 落地步抑制下界告警——零干预是设计语义)。
**只告警不降级**:自动降级是影响效果的干预,须另行裁定
(effect-limiters-need-user-ruling 纪律)。

### 1.3 行为漂移体检脚本(scripts/rollout_health_check.py)

诊断指纹固化,benchmark 无关(只吃统一 dump schema,停止 token 从 turn 末位
众数自举,不依赖 tokenizer):turns 分布 + quick-submit 率(≤2 turns 占比,
本次事故最早指纹:val00 31 条→val10 236 条)/ mid-EOS 行(engine P0-1 零
假阳性指纹)/ 分臂 won/len/turns 对照 / θ 与 open_rate 均值(warmup 哨兵行
单列)。与 `engine_sps_parity_check.py` 分工:parity 是上线判定门,本脚本是
趋势观察(恒零退出码)。

### 1.4 α 消融臂(验证方案项,无代码)

MENTOR Table 4:α 固定 1.0 在其最温和语境已翻负(AIME24 13.9 < GRPO 14.2),
0.5 优于 1.0。重跑验证矩阵加 `steered_adv_weight: 0.5` 对照臂。

## 2. 核实后取消的项(防重复怀疑)

**won 门控 excess**:初判"SSB soft 分让 [·]₊ 失去 MENTOR A.3 的自动过滤
护栏"。核实推翻:训练 reward 自 Initial commit 起就是二值
(`reward_score = 1.0 if traj.won else 0.0`;soft score 仅进 dump 审计,
README 双轨裁定)。护栏在场,won 门控恒等于现状——不实施。护栏挡不住的
"**成功**特权轨迹的行为克隆"(本次 546 正推行全为 won 行)由剂量控制承担
第一道防线;结构性方案(干预位/自由位分治 advantage、doc 答案密度)留作
后续裁定项。

## 3. 兼容与指纹

- **新默认 = 新方案**:`gate_mode: entropy` 裸配即自标定(旧默认下会被
  "未标定拒斥"校验拦;该校验移入 ema 档)。
- **旧 run 完整复现 / 续跑**:显式三键
  `entropy_theta_update: ema` + `entropy_theta_init: <旧值>` +
  `entropy_floor: <旧值>` ——material 不追加新段,**指纹逐字节不变**,
  resume 守卫放行(测试锁定)。
- direct 档进指纹:`|entropy_theta_update=direct:{ratchet}`(改变行为分布 b)。
- 仓内全部 16 个实验/模板 yaml(experiments/*/config.yaml + configs/**)已
  迁移 direct 自标定;服务器侧副本迁移指引走 changelog(CD-006 操作面)。

## 4. 重跑验证要点(交 CD-006 时展开)

1. 部署含本批次的 sha;engine 档仍须先过 V1 对拍(runbook 红线不变);
2. 主臂:迁移后的 ssb-4b-selfres 配置(direct 自标定);观察:
   `steer_entropy_theta` 曲线(-1.0 warmup → 落定值)、`steer_gate_open_rate`
   应落 [0.05,0.25] 带内、tracker 告警日志应零;
3. 每 2–3 step 跑一次 `rollout_health_check.py`(train + val dump):
   quick-submit 率不得复现单调上升;steered/bare turns 差收窄;
4. 消融臂(GPU 允许时):α0=0.5;
5. 判据仍以 val 曲线为主终点(val 三点法);行为分布是领先指标。

## 5. 实施状态(2026-08-11)

代码 + 测试 + yaml 迁移全部落地,507 passed / 3 skipped。改动面:
`gate.py`(tracker 双模式/warmup/ratchet/告警 + THETA_WARMUP)、
`config.py`(三键/校验重构/指纹)、`agent_loop.py`(tracker 构造/哨兵落账)、
`scripts/rollout_health_check.py`(新)、
`tests/steering/test_dosage_control.py`(新,9 用例)、
`tests/test_v33_mechanism.py` 与 `tests/test_162_v33_config.py`(随新语义
更新)、16 个 yaml。advantage/trainer/estimator/wire/engine 路径零改动。
