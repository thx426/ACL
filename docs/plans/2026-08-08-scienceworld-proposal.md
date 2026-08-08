# ScienceWorld 引入协商文档(M3,CD-010,2026-08-08)

**状态:调研完成,待与用户探讨确认(需求 §4 流程;确认前不实现)。**
按需求七项组织;文末集中列出六个待裁定问题。调研方式:官方 repo/PyPI 包
源码逐行(scienceworld 1.2.3,本地拉包审阅)+ 文献检索 + 两台服务器
基础设施实查。

## 0. 结论先行

ScienceWorld 与 ALFWorld **高度同构**(文本命令环境、单命令/步、engine 判
分),引入成本在四个新环境中最低:adapter 可复刻 ALFWorld 的 worker 子进程
骨架 + 单工具 `execute(command)` FC 协议(已裁定形态);**gold path 是引擎
原生 API**(比 ALFWorld 的 walkthrough 通道成熟);官方自带 train/dev/test
变体划分 API。三个实质差异需要裁定:**部分分 reward(0-100 且可为负)、
步预算(任务链长于 ALFWorld)、JVM 基础设施(A800 无 java)**。

## 1. 环境本体(调研发现)

- 30 个任务类型 / 10 个主题(Matter/Measurement/Electricity/Classification/
  Biology/Chemistry/Genetics 等),**7,207 个参数化变体**;引擎 Scala(编译
  JAR),Python API 经 py4j(`pip install scienceworld`,当前 1.2.3)。
- 每个 `ScienceWorldEnv` 实例 = 一个 JVM 子进程(`launch_gateway`,
  die_on_exit);**同一实例可反复 `load(taskName, variationIdx)` 换任务**
  ——JVM 池化复用可行(见 §7)。
- `step(cmd) -> (obs, reward, isCompleted, infos)`:score = 0-100 整数
  (引擎 0-1 转换),**reward = score 增量**;`score < 0` 时引擎强制终局
  (不可逆失败,如 focus 错对象);构造参数 `envStepLimit`(默认 100)超限
  终局。免费动作(look/inventory/taskdesc)不耗步。
- 动作空间 ~25 个模板(open/activate/pick up/pour/mix/go to/read/
  use X on Y/**focus on**/wait 等);`focus on` 是计分关键动作且是负分
  主要来源。5 项 simplifications,`"easy"` 全开(teleport 动作、容器默认
  开等)。

## 2. 训练数据(裁定 Q1)

两个候选口径:

| 口径 | 内容 | 依据 |
|---|---|---|
| **A · ETO 口径(推荐)** | train 1,483 变体实例;dev=seen 194;test=unseen 211 | ETO/IPR/PGPO/RLVMR/SkillPyramid 等社区 RL 工作通用,可比性最好;量级与我们 ALFWorld(2,875/274)一致 |
| B · 官方全量 | 7,207 变体按 50/25/25 | 引擎 API 直接给(`get_variations_train/dev/test`);量大但无社区直接对照 |

seen/unseen 语义:**变体级**划分(所有任务类型都见过,变体参数不同)——
与 ALFWorld 的场景级 seen/unseen 口径不同,报告时需注明。数据构建同
ALFWorld:build_datasets 产 parquet(task_id=任务类型+变体号、initial_obs、
gt_text 预提取),family=任务类型(30 类,family 分层指标现成适用)。

## 3. 测试数据

随 Q1:ETO 口径 dev 194(seen)+ test 211(unseen),val 双 split 与现有
评测器口径(valid_seen/valid_unseen)直接对齐;greedy 单次 + ±5pp 误差棒
纪律沿用。

## 4. 格式协议与默认系统提示

- **已裁定形态**:统一 FC、单工具 `execute(command)`(与 ALFWorld 完全
  同构;M0/M1 基座零新增)。工具描述写行为契约:一步一命令、免费动作
  说明、focus 的计分语义与不可逆风险、负分终局、步预算。
- 系统提示:role + 动作模板表(引擎 `get_possible_actions` 捕获,逐字
  引用不转写——沿 ALFWorld COMMAND_REFERENCE 纪律)+ 实体引用规则 +
  终局语义(达成即结束/负分即失败)。
- 观察回注经 `tool_response_wrap`(公共件);preflight FC 三门自动覆盖。

## 5. 评估方案(裁定 Q2,最实质的一项)

ScienceWorld 是**部分分环境**(vs ALFWorld 二元):
- **val 指标**:主 = avg score/100(社区口径,RLVMR 等报 avg score);
  副 = won 率(isCompleted 且 score=100)。双指标都报,family 分层照旧。
- **训练 reward(GRPO)**,两案:
  - **R1(推荐)**:episode 终局 score/100 直接作 reward(SSB soft-score
    先例;部分分信号密,组内方差更有效)。负分处理:`max(score, 0)/100`
    (负分 clamp 到 0——负值幅度可到 -100,一条 -1.0 会主导组内
    advantage,且"失败=0 分"与 ALFWorld 语义对齐);
  - R2:保留负值 `score/100 ∈ [-1, 1]`(信号最忠实,但负样本权重放大,
    与既有环境 reward∈[0,1] 口径不一致)。
- step 级 delta reward 不用(机制是 outcome GRPO,不做 step shaping)。

## 6. Ground Truth 反馈信息

`load(generateGoldPath=True)` → `get_gold_action_sequence()`:引擎原生
gold 命令序列(达 win 态,不保证最优)。**per-variation、可执行、与动作
空间同语法**——比 ALFWorld 的 gt_text(需自建 walkthrough)成熟一代。
build_datasets 时预生成入 parquet(gt 防火墙沿用:只进 researcher/teacher,
永不进 executor);gold path 生成失败的变体记录缺失率(ALFWorld 47/2875
缺失的教训:先量化再决定是否剔除)。

## 7. 任务特定信息与工程实施(裁定 Q3-Q6)

- **simplifications(Q3)**:社区各工作是否开 "easy" 不统一,直接影响
  可比性——实现前先核对 ETO/RLVMR 原文设置再定(倾向跟随社区主流口径)。
- **步预算(Q4)**:任务链长于 ALFWorld(grow-plant 等含 wait 等待),
  gold path 长度分布是定值依据——build 期统计后定 `max_env_steps`
  (预计 50 档;引擎 envStepLimit=100 兜底)。per-run yaml 配置,零代码。
- **JVM 内存与并发(Q5/Q6)**:每 JVM 预估 300-500MB(需实测),128 并发
  = 40-60GB RAM;沿 AlfworldCfg.spawn_concurrency 模式设门。**推荐 JVM
  池化**(worker 常驻、load 换任务,省 JVM 冷启;官方示例即循环 load),
  带定期回收防泄漏;备选每 episode 冷启(ALFWorld 现模式,最保守)。
- **A800 无 java**(实查:`java: command not found`),162 有系统
  OpenJDK 11(满足 ≥1.8)。A800 方案:conda-forge `openjdk` 装入环境
  (无 root、符合"GPU 生产栈以 conda-forge 为底"纪律),装哪个 env 待定。
- adapter 骨架:复刻 `envs/alfworld` 的 `_Worker` 线协议子进程(JVM 藏
  worker 内),FC react 直接套单工具模式;预计新增代码 ~400 行 + 测试。

## 8. 基线对照(调研发现)

- **Qwen3.5 同基座公开基线:未检索到**(CD-006 的 baseline survey 亦未
  覆盖 SciWorld)→ 引入后先跑 9B/4B raw zero-shot 自建基线。
- 社区参照(非同基座,定性锚):RLVMR avg ~47.5%(7B 档);ETO/IPR/PGPO
  系有完整 seen/unseen 对照表可引。

## 9. 待裁定问题汇总(探讨清单)

| # | 问题 | 推荐 |
|---|---|---|
| Q1 | 数据口径:ETO(1,483/194/211)vs 官方全量 | ETO(社区可比) |
| Q2 | GRPO reward:clamp 部分分 vs 保留负值 vs 二元 | R1:max(score,0)/100 |
| Q3 | simplifications 开不开 | 核对 ETO/RLVMR 原文后跟随社区 |
| Q4 | max_env_steps | build 期按 gold path 分布定(预计 50) |
| Q5 | A800 java:conda-forge openjdk 装入哪个 env | 随 q35 生产环境 |
| Q6 | JVM 池化 vs 每 episode 冷启 | 池化 + 定期回收 |

---

# 增补(2026-08-08 二轮):社区代码实测答案与修订后的裁定项

按用户要求逐仓读了社区实现(资产与 commit 见 `related_work/INDEX.md`
benchmark 章节;ETO/IPR/PGPO/RLVMR/官方引擎五库 + 五篇 PDF 已归档
`related_work/benchmarks/scienceworld/`)。四个原裁定项中两个被社区实测
**直接定案**,一个被**改写**,并新增一个此前未识别的抉择:

## 已定案(社区口径一致,不再作为裁定项)

- **Q3 simplifications → "easy" 全开**:ETO/PGPO 在 `reset()` 里硬编码
  `simplificationStr="easy"`,RLVMR `env_manager` 默认同值——两个谱系
  (SFT+DPO 系与 GRPO 系)完全一致,跟随即可,无可比性歧义。
- **Q4 步预算 → per-task 表**(修订原"全局 P95"设想):ETO 的
  `max_steps.json` 按任务给 10–120(lifespan 10 vs melting-point 120,
  差 12 倍)——全局单值要么截断长任务要么让短任务失败轨迹烧 12 倍预算。
  实施采用 ETO 表(24 任务全覆盖);机制侧小增量:adapter 提供 per-task
  `max_env_steps` 覆盖(默认回落全局)。RLVMR 的全局 30 作为对照口径披露。

## 修订后的裁定项

- **Q2 reward:新增 R4「轨迹峰值分」为推荐项**。社区两家都不用"终局
  score":ETO 系取 **max-over-time raw_score**(只增不减——天然免负分、
  对进度回退鲁棒);RLVMR 用**二元 10×won**(其 dense 信号来自
  meta-reasoning 标签,非环境分)。R4 = `max_t score_t / 100`:既是 ETO 系
  语义(可比),又天然解决负分(峰值恒 ≥0),又保留部分分的组内梯度密度
  (我们没有 RLVMR 那种 meta-reward 补密度,部分分对 GRPO 价值更大)。
  val 双指标:avg peak score(对 ETO 系)+ won 率(对 RLVMR;won 定义
  采严格版 = engine completed,与其宽松版差异披露)。
- **Q1 数据:ETO 口径构成已实锤为「24 任务」**(taskname2id 29 但
  indices 只用 24;train/dev/test 同任务集,seen/unseen 纯变体级)。
  推荐不变(主 ETO 口径,数字直接进其对照表);RLVMR 的 **L2 任务级
  held-out**(19/10 任务不相交)是更强的泛化口径,列为后续可选第二评测
  (泛化叙事用),首轮不做。

## 新增裁定项

- **Q7 观察增强(admissible actions 注入)**:RLVMR 每步把 valid_actions
  全表+对象表+示例注入观察;ETO 系不注入(代码里注释掉了)。**推荐:不
  注入**——动作知识正是 researcher 文档要填的信息面(与 ALFWorld 的
  "No admissible-command list" 纪律同源);与 RLVMR 比数字时披露该差异
  (它有动作先验、我们没有,方向对我们有利)。

## 其它读码收获

- RLVMR 是 verl-agent step-level 架构(每步独立 prompt、
  max_response=1024/步),与我们整轨迹 token 流不同型——数字可比、
  训练形态不可比,论文里如对照需注明。
- RLVMR 附 SciWorld 专家轨迹 SFT 冷启动资产(`sciworld_expert_traj`),
  与我们 4B SFT 冷启动先例同构,若 SciWorld 上需要冷启动可参照其构成。
- ETO 步数口径"计对话步含格式失败步"≈ 我们的 max_assistant_turns 语义
  (而非 max_env_steps);实施映射时对齐这一层。
