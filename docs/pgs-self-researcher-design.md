# PGS v4.0:自体研究员双目标 RL(self-researcher)机制设计

> **状态:v4.0-draft 定稿(2026-08-06,用户裁定;分支 `pgs-selfres`)。**
> 在 v3.2(分臂 + maxopen)之上把 researcher 角色交还 policy 本体,并以
> **前缀续写 reward** 让"产出更有效特权文档"的能力随训练自进化——
> 单一 policy 的完全自体 compute-scaling 自进化闭环的最后一环。
> 历史前身:mixed-objective 时代 self-research v3(2026-07-28 定稿实施,
> P1.5 出清;裁定与教训经考古继承,见 §8)。
> 本文档只管机制定义与正确性;实验/对照/判据另文。

---

## 0. 总形态

单一参数 π_θ 扮演全部角色(bare executor / steered executor / researcher /
narrator——narrator 已内化)。每训练 step、每任务 **24 行**(16 executor +
8 research,行数均可配):

```
session_id [0, 8)    steered 臂   PGS maxopen 倾斜(v3.2 全套),await「argmax doc」
session_id [8, 16)   bare 臂      纯 π 采样,立即起跑(零依赖)
session_id [16, 24)  research 行  8 个独立研究会话(n_exp=16 实验 × k=4 episodes)
# 段序沿 v3.1 is_steered_arm 前缀约定(steered 前段);行角色真值以
# role/steer_arm 记账字段为准,段序为确定性标签细节(副本先验 iid)
                                  → doc → await「bare 前缀集」
                                  → 自评:5 分位 × K_c=4 续写 → r_i = 有效续写成功比例
                                  → 登记 r_i;自身即训练行,reward_score = r_i
```

训练目标(joint,一次 backward):

    L = L_exec(16 行,v3.1 混合组 GRPO + W_beh) + λ_r · L_research(8 行,独立组 GRPO)

两目标共享参数、数据面严格隔离;唯一耦合是数据流(argmax doc 拿教师位、
bare 轨迹供前缀)。

## 1. 组件精确定义

### 1.1 research 会话(每任务 `research_n=8` 个,独立)

会话形态沿现行(run_experiment / fetch_raw / finalize 三工具、n_exp=16 实验
预算、每实验 k=4 episodes、max_turns=30、叙述 + fact card 返回),全部沿
`research:` 段既有配置键(均可配)。变化:

- researcher chat 走**本机 rollout 引擎**(policy 自体),TextProtocol
  (research/toolproto.py 现成,注释本为 policy-as-researcher 预留);
- researcher 采样温度 **1.0**(用户裁定 2026-08-06;`research_rl.researcher_temperature`
  可配。external 35B 模式保留 roles.researcher.temperature 旧语义);
- 组内多样性 = 采样温度 × 实验结果随机性,一期不加显式扰动;
- researcher 不可见:GT(gt: off)、bare 臂轨迹(**盲写**——诊断学生要靠
  自主实验,这正是被训练的能力;bare 轨迹作可选工具为二期留白)。

### 1.2 前缀集(每任务一份,组内 8 个 doc 共享)

- 来源:本步 8 条 bare 轨迹,**失败优先、跨轨迹分散**(5 个分位点尽量取自
  5 条不同失败轨迹,不足以成功轨迹补位);
- 切分:按 **policy 生成 token(response_mask=1)的累计比例**取分位
  `prefix_quantiles = [0.05, 0.15, 0.30, 0.50, 0.75]`(用户裁定;可配),
  **不对齐 turn 边界**——切点可落在一轮生成的中间(半轮);
- 前缀内容 = 轨迹 token 流从头到切点的完整流(含 observation 结构 token)。

### 1.3 续写评估(新组件:环境重放 + 半轮承接)

对每个 (doc_i, prefix_j) 做 `continuation_samples = K_c = 4` 次独立续写
(用户裁定默认 4,可配):

1. **教师前缀构造**:`scaffold + "## Task guidance" + doc_i + 初始 obs +
   前缀流`——与 PGS verify 教师序列同一构造路径(token 级拼接);
2. **环境重放**:`adapter.open_episode()` 开新 episode,把前缀中**已完成
   轮次**的 action 逐个 `env.step` 重放至前缀点状态;重放期间逐步校验
   observation 与前缀记录一致——不一致 = 环境漂移,该次续写作废(ENV_ERROR
   类;确定性引擎契约下不应发生,作免费一致性哨兵)。半轮截断轮的 action
   尚未产生,不重放;
3. **续写**:从教师前缀 + 前缀流的 token 断点直接继续生成(引擎 token 级
   续写,与 SPS draft 调用同构)。教师**承接半轮**(用户裁定确认):补完
   该轮剩余 reasoning/action;action 解析作用于**半轮前缀文本 + 续写文本
   的拼接**;随后进入标准 episode 循环直至终局。**纯 p_T 采样、温度
   `continuation_temperature=1.0`、无 SPS、无倾斜、无 W 记账**——reward
   测教师条件分布本身,与 PGS 倾斜方向源同一对象;
4. **预算继承**(用户裁定确认):环境步预算 = `max_env_steps − 前缀已用
   步数`;token 预算 = `response_length − 前缀 token 数`。教师与学生同规,
   评估语义 = "从这个状态、剩余预算内能否完成",不给教师加时。

### 1.4 reward 与有效性规则

    r_i = #{续写终局 won} / #{有效续写}     (名义 5 × K_c = 20 次)

- 单次续写 ENV_ERROR → 重试一次,仍败剔除分母;
- 有效次数 < 名义半数 ⇒ 该 research 行降级:不进训练组、不参 argmax;
- no_finalize(未产 doc)⇒ r_i = 0(finalize 门:产文档是能力的一部分);
- doc 超教师上下文 ⇒ r_i = 0(续写不可构造)——**结构性长度压力**,
  替代一切显式长度惩罚(不加 shaping 的裁定自洽)。

### 1.5 research 组 GRPO

- 独立组(uid `:research` 后缀);A_i = r_i − mean(r)(LOO 均值口径复用
  grpo_loo 估计器;**不除 std**——K_c 级样本噪声下除 std 放大纯噪声,
  历史 75× 实证;norm_adv_by_std=false 全局已是);
- r ∈ [0,1] ⇒ A ∈ [−1,1] **天然有界,无需额外 clip**(较历史设计简化);
- 全 0 / 全 1 组:advantage 全零自然无梯度,分别计数(无能力 / 毕业读数);
- 轨迹级 advantage 乘到会话全部 researcher 生成 token(工具返回/叙述
  mask=0);一期不做 turn 级 shaping(留白)。

### 1.6 训练会计(joint)

- **seq-mean-token-mean(现默认)下的简化**:每行先 token-mean 再行间等权
  ⇒ 长 research 行不按长度抢权重,历史"分角色分母"的动机消解;λ_r 实现为
  **research 行的 advantage 标量乘 λ_r**(梯度线性等价,最小侵入);
- `research_warmup_steps` 内 research 行 advantage 置 0(doc/r 照产,不反传);
- research 行超长(> `research_max_sample_tokens`)⇒ **response_mask 置全 0**:
  零梯度但保留在组内 baseline(其 r 有效),doc 照用,计数;
- 全零 advantage 的 research 子批无梯度贡献,optimizer step 由 executor 行
  驱动(无 Adam 空转问题);
- research 行不进 executor 任何行为指标/分母(隔离纪律)。

### 1.7 steered 臂教师选择

d* = argmax_i r_i(降级行除外;tie-break 取更短 doc)。**准入门**:r* = 0
(最好的 doc 续写也全败)⇒ 该任务本步无教师,steered 臂退化 bare(现有
doc 缺失降级路径,记账可见)。steered 臂拿 d* 走 v3.2 maxopen 全套不变。

## 2. 正确性论证(九条)

1. **reward 环境可验证**:won 由环境判定,无模型自评——同参数自我串谋
   必须真做对任务才有分,RLVR 纪律封死 reward hacking 主面;
2. **评分器漂移免疫**:组内 8 doc 共享同一 executor 快照 + 同一前缀集,
   组间归一化精确消去 executor 能力与任务/前缀难度公共项(SEAL 绝对
   reward 中招的病理,组间设计天然免疫);
3. **归因纯度与信用链**:组间差异 = doc 差 ← 观察质量差 ← 实验设计+运气;
   期望上 advantage 强化"提高关键信息暴露概率的实验策略"——探索能力的
   信用路径成立(方案甲的独有价值);
4. **评估分布正确**:r_i = Ê_{s~学生前缀分布}[V_T^{d_i}(s)]——在学生真实
   到达的状态上评教师(DAgger 式),与 PGS 用途(在学生轨迹上给倾斜方向)
   对齐;policy-token 半轮切点把状态分布细化到"思路写到一半"的中间态;
5. **续写对象一致性**:续写采样自纯 p_T,与 PGS 倾斜方向源同一条件分布;
6. **续写轨迹不进训练**:reward 测量工具,属教师条件分布样本,进 executor
   训练需 OC-GRPO 式跨上下文校正——明确排除;
7. **信息状态一致性**:researcher 无 GT、盲写(不见 bare);doc 只进教师
   前缀不进学生上下文;research 训练行 = researcher 自身会话;
8. **重放确定性前提**:挂在 TaskAdapter 既有确定性契约上 + 逐步 obs 校验
   兜底;随机环境需 seed 契约(通用性边界,与 initial_obs 检查同一条);
9. **双目标干涉受控**:λ_r 小起步 + warmup + 超长零 mask + 隔离指标;
   共享参数正向面(实验即执行练习)保留,负向面由 executor invalid_format
   曲线监测。

## 3. 并发与吞吐架构(用户裁定:依赖同步等待,其余全并发,不设机制性节流)

worker 内(同任务 24 行同 worker 同 event loop,dedupe invariant):

```
t=0  同时 spawn 24 个 task
 ├─ bare×8      并发跑 → 各自终态登记 → 第 8 行终态 resolve「前缀集 future」
 ├─ research×8  并发起会话(内部:chat 逐轮;run_experiment 的 k=4 episodes
 │              gather 并发)——会话为墙钟主导,bare 其间早已完成 ⇒
 │              doc 产出时前缀集几乎必然 ready,await 零等待
 │              → 评估 20 次续写 asyncio.gather 全并发 → 登记 r_i(或降级标志)
 │              → 第 8 个登记 resolve「argmax future」
 └─ steered×8   await「argmax future」→ 并发起跑(PGS maxopen)
```

- **每行必须登记**(r 或降级标志),否则 argmax future 悬挂——硬规则;
- 任一行两次 ENV_ERROR fail-loud ⇒ verl 整 prompt failure(24 行同弃,
  与现行语义一致);
- 并发量账(worker 峰值→全局 16 worker):实验期 ~8×(1+4)=40 → ~640;
  评估期 8×20=160 → ~2560;A800 8×TP1×max_num_seqs 256 = 2048 在途槽,
  评估峰值靠 SGLang 排队消化(degrade 不崩)。机制侧并发上限全部走配置
  (_per_worker 折算)默认拉满,唯一执行层旋钮 = 引擎 max_num_seqs。

## 4. 边界与降级规则总表

| 情形 | 处置 | 记账 |
|---|---|---|
| no_finalize | r=0,进组承受梯度 | finalize 率 |
| doc 超教师上下文 | r=0(结构性长度压力) | doc_overflow |
| 续写 ENV_ERROR | 重试 1 次,仍败剔除分母 | 有效续写数 |
| 有效续写 < 名义半数 | 行降级:不进组、不参 argmax、必须登记 | 降级计数 |
| 重放 obs 漂移 | 该次续写作废(同 ENV_ERROR) | 漂移计数 |
| r*=0 | 该任务无教师,steered 退化 bare | 准入门计数 |
| research 行超长 | mask 全 0(零梯度),r 留组内 baseline,doc 照用 | 超长率 |
| 组全 0 / 全 1 | 零梯度自然跳过 | 无能力/毕业 frac |
| bare 失败轨迹不足 5 | 成功轨迹补位 | 前缀来源构成 |
| bare 行 fail-loud | verl prompt failure(24 行同弃) | 现行语义 |

## 5. 配置面(全参数化;research_rl.* 与 1.4 规则全部进 objective_fingerprint)

```yaml
research_rl:
  enabled: false                # false = v3.2 现状逐字节不变
  research_n: 8                 # 每任务研究会话数 = GRPO 组大小(用户裁定默认 8)
  prefix_quantiles: [0.05, 0.15, 0.30, 0.50, 0.75]   # policy-token 比例
  continuation_samples: 4       # 每 (doc,prefix) 续写次数(用户裁定默认 4)
  continuation_temperature: 1.0
  researcher_temperature: 1.0   # 自体 researcher chat 采样温度(用户裁定默认 1.0)
  research_rl_weight: 0.1       # λ_r(A×λ_r 实现;seq-mean 下行等权语义)
  research_warmup_steps: 0
  research_max_sample_tokens: 24576   # 超长零 mask 阈
# 会话形态沿 research: 段(n_exp=16、k_default=4、max_turns=30 等,均已可配;
# 用户裁定 2026-08-06:n_exp 默认改 16 与生产配置对齐)
```

依赖校验:research_rl.enabled ⇒ steering.enabled 且 doc_source=researcher 且
research.enabled;行布局 rollout.n = train.rollout_n + research_n(launcher
推导);arms 三段 = v3.1 前缀切分的推广。

## 5.5 SFT 冷启动(用户裁定 2026-08-06 第二批;scripts/sftgen_researcher.py)

依据「OPD/RL 只能调已有行为,SFT 是唯一引入新行为的通道」(CD-004/CD-006
实证):v4 起跑前须以强模型会话把 policy 的 researcher 角色能力从 ≈0 拉起。

**数据采样**(裁定):固定 seed(默认 42)按 benchmark 类别**分层均匀**抽
100 任务(桶内确定性排序+shuffle、round-robin 取满,桶间差 ≤1);每任务
**8 个独立 research 会话**(与 research_n=8 的 RL 组形态分布匹配);
researcher = Qwen3.6-35B-A3B 端点,executor/narrator = 目标 policy 的
推理 server。**机制参数与 v4 训练器同一配置真源**(load_config 同一份
yaml:text 协议、researcher system prompt、n_exp=16、max_turns=30、k=4、
researcher_temperature=1.0、episode caps 全部同源)——35B 以 policy 将要
使用的同一协议/同一提示词采样,SFT 学到的行为形态与 RL 推理形态一致。

**分层声明**(用户要求显式划分):
| 层 | 内容 | 换 benchmark 时 |
|---|---|---|
| benchmark 特定 | 任务分层抽样的 family 推导(adapter.task_type)、executor scaffold / task brief(经 adapter) | 换任务抽样函数 + adapter 工厂两处([BENCHMARK-SPECIFIC] 标注) |
| 通用机制 | research 会话循环(session/toolproto/tools/narration)、断点续跑/落盘/筛选/转换、verl multiturn SFT 训练 | 原样复用 |

**转换与训练**:messages(text 协议,system/user/assistant)原样落
parquet ⇒ verl `MultiTurnSFTDataset` 对 assistant 轮自动 loss=1、其余=0
——零格式改造。质量筛选:finalized & doc 非空 & 无 error;报告给出
family 分布/finalize 率/轮数与长度分位(SFT max_length 依据)。训练走
vendored verl `sft_trainer`(通用机制,与 benchmark 无关)。

**信息状态**:35B 会话无 GT(gt: off 同源),数据 = researcher 可见
信息,无特权泄漏。**M0 探针**:SFT 后把 policy server 当 researcher
端点,用同一 sample 子命令小规模复采(如 10 任务×2),convert 报告的
finalize 率/长度与 35B 报告直接对比 = Go/No-Go 读数(同脚本兼做探针)。

## 6. 留白(记账不实施)

随机环境 seed 契约;bare 轨迹作 researcher 可选工具;turn 级 credit;
doc 跨步复用/轮转省算;research_n / n_exp scaling 轴;separate update
消融档;组内同质化时的显式扰动;乙方案(共享探索+多次 finalize)为
远期救援档。

## 7. 用户裁定记录(2026-08-06,本设计的约束)

① 方案甲(N 个独立完整会话;乙否决——同质化+训不到实验设计能力);
② research_n=8 默认可配;③ 前缀分位 {5,15,30,50,75}%、按 policy 生成
token 比例切、不对齐 turn(半轮承接确认为特性);④ K_c=4 默认可配;
⑤ 前缀失败优先跨轨迹分散;⑥ 预算继承(教师不加时);⑦ researcher
温度 1.0 可配;⑧ n_exp 维持 16、max_turns 30(均可配);⑨ 并发原则:
依赖同步等待、其余全并发、worker 内并发度拉满;⑩ joint + warmup +
超长降级结构认可,λ_r/warmup 数值实施时定。

## 8. 历史继承清单(self-research v3 考古,mixed-objective 时代)

组间对比免疫评分器漂移(SEAL 教训);不除 std(75× 噪声放大实证);
finalize 门 r=0;argmax + 准入门;joint 默认(v2→v3 推翻"零先例不可同批");
research 行隔离纪律(mask/指标/超长);全零组自然跳过;warmup;
会话 token 面 append-only 构流(训练 token = 引擎实际采样 token,
messages 重渲染有 BPE roundtrip 风险,不可用);"narrator 不内化"裁定
已被演进推翻(现 narration.backend=engine)。

## 9. 通用性契约(2026-08-07 定稿;引入新 benchmark 的完整改动面)

> 操作指南与参考实现:**`src/trace_opd/envs/_template/`**(完整可运行的
> 独立玩具 benchmark + README 交互点总表);契约唯一真源 `core/adapter.py`。
> 分离性由 `tests/test_template_env.py` 机器证明:非 ALFWorld 环境上真实
> 跑通 episode 运行器与 v4 前缀续写,零机制改动。

机制核心对任务协议**零假定**(用户裁定 2026-08-07):

1. **交互协议抽象**:`TaskAdapter.react(env, assistant_text) → ActionOutcome`
   ——动作语法(文本标签/JSON 工具调用/代码补丁)、执行接口、无效判定、
   格式提醒全部封装在任务实现内;core 循环只消费 ActionOutcome(token 流、
   预算、防 spiral、记账)。文本单动作协议作为可复用件下沉
   `protocol.text_action_react`(ALFWorld/模板一行委托)。
2. **评估反馈抽象**:`ActionOutcome.score: float ∈ [0,1]`——二元 won 是
   `score∈{0,1}` 特例;部分分保真进 `Trajectory.score` 与
   `ContinuationOutcome.score`。当前 GRPO/续写口径的 won 由 `score>=1.0`
   派生(把连续分接进 reward 是训练口径决策,非协议改动)。
3. **确定性重放钩子**:`TaskAdapter.replay_action(env, action) →
   (feedback, ended)`——v4 前缀续写的硬前提显式化。
4. **环境句柄缩面**:`EnvHandle = {initial_obs, close}`;`step` 从 core
   契约移除(字符串命令族环境的实现细节)。
5. **装配注册表**:`envs/registry.py` + 配置 `environment.name`(默认
   alfworld,默认值不进 objective_fingerprint——s4 指纹逐字节不变,有测试
   锁)。装配点(agent_loop/run_eval/sftgen)一律 `make_env_factory(config)`,
   资源 per-worker 折算归各 builder。工厂契约:`adapter_from_item(extra_info)`,
   parquet 行的 extra_info 必须自足(字段契约见 `_template/build_dataset.py`)。

引入新 benchmark = 复制 `_template/` 实现 adapter + 数据集脚本 + registry
一行 + 配置一行 + 照 `test_template_env.py` 写四组测试;机制侧(SPS/保尾核、
W_beh、v4 三段/TaskCoord/前缀续写、SFT 转换)零改动。researcher 模板与
文档规则已中性化(交互形态描述由 task_brief 承担)。

**首个按此契约引入的真实第二 benchmark:SpreadsheetBench(verified 400,
`envs/spreadsheetbench/`,原生 function-calling executor + 部分分评估)——
方案与四项裁定见 `docs/spreadsheetbench-integration-plan.md`;core 仅新增
`protocol.parse_tool_calls` 通用件,机制层零改动,实证了本契约。
