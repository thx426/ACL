# Self-Research:researcher 内化与联合训练 — 设计探讨稿

> **状态:v3 已定稿,M1 机制实施已完成(2026-07-28,trace_opd `f9abbda` + vendored verl `af6a8d3`;206 测试全绿)。** 实施与设计的偏差见 §13。CD-005。
> v0→v1:并入第一轮六路深读(skill/self-evolve 邻域源码 + 三个论文组;§7 重写为稳定性定稿;§4-4 改双通道)。
> v1→v2:按用户要求补**通用**多任务/多目标线第二轮四路深读(AgentRL 论文+代码专项、经典 MTL 优化器族重审、多任务 RL 配方与课程、工业多域 RL 实践)——修正一处误伤(幅度均衡类不受余弦证据株连,但各有硬阻断)、发现 research_rl_weight 的 clip 饱和语义、新增 §7.7 配比与任务级课程。证据文件 `docs/research-notes/deepread-*.md` 共十份。
> v2→v3(用户质疑触发的结构修正):"零先例=不可同批"经复审属证据过度解读(UCOB 分开的动机是 padding 效率,我们栈无此问题;同批 dense+稀疏 PG 我们生产三项即正面先例)——结构裁定改为 **`research_update_mode: joint`(一次联合 backward、分角色分母)为默认**,完美 on-policy、w 语义在 clip 下保持、工程侵入面最小;separate(第二次 update)降为可配备选与消融臂。
> 本稿是对用户构想("researcher 角色换成 policy 自身 + research 能力 SFT 冷启动 + 训练循环内 research RL 与 OPD 联合反传")的理解、形式化、修正建议与实施设计。
> 输入:CD-002 专项调研 `.conductor/branches/CD-002/multi-task-joint-training.md`(9 条硬结论)与主报告 novelty 分析、`experiments/h1_pilot/docs/mechanism-playbook.md`、当前生产代码 `src/trace_opd/train/agent_loop.py`。
>
> **用户裁定(2026-07-28)**:①N=8、K=8 为默认,作可配置参数(快速实验可调小);②先专注机制设计实现,冷启动数据问题后议,SFT 用 verl 栈(最稳健方式);③narrator 不参与 self-evolve;④消融/实验安排暂缓,先做机制;⑤**模块正交性(关键澄清)**:研究侧 RL 与执行侧混合目标(mix_opd/mix_rl/mix_sft)是两个模块——执行侧三通道优化"任务执行更好",研究侧优化"researcher 更好",不得混淆;问题核心 = 双能力联合优化的稳定性(如任务难度差异导致梯度不均衡、部分任务无有效训练信号等);⑥设计前必须对关键相关工作做**论文正文级 + 源码逐行级**深读(6 路子 agent 深读进行中,结论将并入 §7)。

---

## 1. 目标与定位

现行机制(v2.x)的 researcher 是外部 35B:policy 的"经验来源"仍寄生于一个更强的外部模型,self-evolve 不完整。本机制(内部代号 **self-research**,机制版本 v3)把 researcher 角色交还 policy 本体,并让 research 能力随训练进化:

```
                    ┌──────────────────────────────────────────┐
                    │              π_θ(同一份权重)             │
                    │   role=R(researcher)   role=X(executor)  │
                    └──────────────────────────────────────────┘
   research 会话(R):N 份文档 d_1..d_N   ──►  doc-conditioned rollout(X)×K
        ▲    │                                     │
        │    │ 组间 reward = 成功率 p_i             │ 成功轨迹 → SFT 臂数据源
        │    ▼                                     ▼
   research RL(PG)                d* = argmax p_i → teacher 特权前缀 → OPD(bare rollout)
        └────────────────  L = w_opd·OPD + w_rl·RL_exec + w_sft·SFT + w_res·RL_research ──┘
```

飞轮(双向咬合):research 能力↑ → doc 质量↑ → teacher 优势↑ → OPD/SFT 增益↑ → executor 强 →
(a) doc reward 评测更准;(b) research 会话内 run_experiment 的实验 episodes 更有效(executor 是研究的仪器)→ research 更强。
诚实的另一面:executor 变强 → bare 成功率升 → doc 边际增益空间缩小 → 组间方差衰减 → research 信号自耗尽。这是"任务毕业"而非故障,但必须有曲线(§10)。

**scope 声明**:brief 将"元能力 RL(H4)"默认排除在第一轮外,用户本轮显式行使分支内调整权启动它;此调整将记入完成报告 Proposed Global Decisions 报备 master。

## 2. 用户构想的形式化

每训练 step、每任务 τ:

1. **Research 组**:π_θ^R 独立跑 N(默认 8)个 research 会话 → N 份经验文档 {d_i}(会话按 H1 定版形态:run_experiment / fetch_raw / finalize 三工具、预算制、文本协议)。
2. **Doc 评测**:每份 d_i 作 guidance,π_θ^X 跑 K(默认 8)次 doc-conditioned rollout,成功率 p_i = wins_i / K。
3. **组间 reward**:A_i = normalize({p_i}) 作为第 i 个 research 会话轨迹的 advantage(GRPO 式组间对比)。
4. **Teacher 选择**:d* = argmax_i p_i 作为本 (step, task) 的特权文档,teacher = π_θ(d* 前缀),对 bare rollout 打分做 OPD(现行机制不变)。
5. **联合反传**:`L = w_opd·L_OPD + w_rl·L_RL_exec + w_sft·L_SFT + w_res·L_RL_research`,一次 backward。

**模块正交性(用户裁定 ⑤)**:前三项(mix_opd_weight / mix_rl_weight / mix_sft_weight)是**执行侧**三通道——目标都是"任务执行更好";第四项是**研究侧**——目标是"researcher 更好"。二者是两个正交模块,配置命名刻意区分:研究侧系数命名 **`research_rl_weight`**(不进 mix_ 前缀家族),避免被误读为执行目标的第四通道。两侧唯一的耦合点是数据流(argmax 文档拿 teacher 位、doc-cond 池喂 SFT),不在 loss 语义上。

冷启动:离线收集 research 会话 SFT 数据(§5),先把 π_θ 的 R 角色能力从 ≈0 拉到可用,再进联合循环(依据:OPD/RL 只能调已有行为,SFT 是唯一引入新行为的通道——CD-004/CD-006 实证)。

## 3. 对构想的分析与修正建议

### 3.1 组间成功率 reward:方向正确,且有一个用户构想里隐含的结构性优点

- 它就是 **PG-7 已确认的文档质量信号**(doc 注入 K=8 rollout 成功率)升格为训练信号,与 H1 的 `evalm/m1_replay.measure_guidance` 同口径——信号定义零新发明,实证有判别力。
- **对"评分器漂移"的结构性免疫(应写进论文)**:doc reward 的评分器是 executor 自身,而 executor 每 step 都在变——这正是 SEAL 报告 GRPO/PPO 不稳、退回 ReST^EM 的场景(reward 依赖被更新的参数)。但**组内 N 份文档共享同一个 executor 快照**(step 内权重冻结),组间归一化把 executor 能力的全局分量整体消去,只留文档间相对质量。SEAL 用绝对 reward 才中招;组间对比设计对该病理天然免疫。同理,它也天然中心化了任务难度(同任务 N 份文档共享难度)。
- 与 UCOB 的 reward(δ_m = SR(用 m) − SR(no-skill),跨时间 EMA + pool 归一化)相比:我们的同 step 组间对比不吃 EMA 的漂移污染,且 credit 直接落在产出该 doc 的完整 research 轨迹上(而非 reflection 重放)。

### 3.2 四条修正建议(按重要性)

1. **归一化不除组内 std(Dr.GRPO 式),或给 std 设下限**。K=8 时 p̂ 的采样噪声 σ=√(p(1−p)/8)≈0.17(p=0.5 处);组内各 doc 真实质量接近时 std→小,除 std 会把纯噪声放大成大 advantage。建议 `A_i = p_i − mean(p)`(定稿细则见 §7.2:UCOB σ 下限 0.05 与 weighting 报告 75× 噪声放大实证三方收敛)。
2. **退化组不硬凑梯度**:全 0 组(research 无能为力)与全 1 组(任务毕业)advantage 全 0,research 项自然无梯度——保持 as-is,但分别计数落曲线(`research/group_all_fail_frac`、`group_all_pass_frac`)。这两条曲线本身是机制健康的一级读数。
3. **finalize 门进 reward**:未 finalize(无文档)的会话不跑 K 次评测,直接记 p_i = 0。finalize 是 research 能力的一部分,必须承受梯度压力;但这要求冷启动先把 finalize 率打到高位(27B 实测 100%,policy 冷启动后的 finalize 率是 M0 探针的必测项),否则全组 p=0 → 无梯度,飞轮点不着。
4. **argmax teacher 的噪声与准入门**:K=8 的 p̂ 方差大,argmax 容易选中运气好的 doc。第一版仍用 argmax(简单、无新旋钮),tie-break 取更短的 doc;但加**teacher 准入门**:p* = 0(最好的文档也全败)时本 (step,task) 不设 teacher(OPD 中性样本,现行 doc=None 降级路径复用)。这是 PG-5③"输入过滤"的内生实现——坏文档在成为 teacher 之前就被下游成功率过滤掉。

### 3.3 明确不做(第一版)

- reward 不加长度惩罚/格式 shaping——保持单一 grounded 信号,长度与格式退化走观察面(§10),真护栏是 unseen 评测。
- 不移植 UCOB 的双向蒸馏/anchor-state 局部 credit(它们解决的是"检索技能不可靠"的问题;我们的 doc 是现场为该任务定制的,且已有 teacher 准入门)。
- narrator 不内化(见 §12 裁定点 3):research 会话内实验 episodes 的叙述仍由外部 strong narrator 提供(H1 消融:27B QA 0.685 vs 4B 0.544,policy 叙述有实体编号丢失+低频虚构两类功能性缺陷,训练它是另一个课题,PG-8 暂缓)。论文定位:narrator 是环境仪器(观察压缩器),不是被进化的智能。

## 4. 机制详细设计(数据面)

现有基础设施盘点(好消息:骨架已铺约七成):

| 需求 | 现状 |
|---|---|
| policy-as-researcher 推理面 | `train/backends.py::engine_researcher_chat` 已实现(文本协议,注释明言"for the self-evolve form"),缺 token 面 |
| doc-conditioned rollout | `agent_loop.py::_expert_rollout` 就是(带 doc 的 episode,同 runner 同协议) |
| 按 replica index 分角色 | `_next_replica_index` 三段先例:[0,rollout_n) bare、[+sft_rollouts) expert |
| 独立 GRPO 组 | trainer `":sft"` 组隔离先例(uid 后缀) |
| reward-weighted PG loss | `task_reward_pg`(混合目标 RL 臂),每微批 token 平均口径 |
| research 会话编排/归档 | `research/` 包 + `ResearchArchive`(会话、实验 episodes、doc、meta 全落盘) |

新增五件(按依赖序):

1. **research 会话 token 面**:改造为 **append-only 增量构流**(与 `core/episode.py` 同构),严格保证训练 token = 引擎实际采样 token(PG 语义要求;messages 重渲染有 BPE roundtrip 不一致风险,不可用):
   - 会话 token 流:system+kickoff 渲染 ids(= 样本 prompt_ids)→ 每轮 researcher 生成 ids 原样 append(mask=1)→ 工具返回/wrapup 提醒经 TemplateDelta 渲染成 user 轮 delta append(mask=0)→ …;response_ids = prompt 之后全部,与 executor 多轮样本同构(response 含环境轮,mask 区分)。
   - `ResearchSession` 保持 backend-agnostic:external(35B,fncall,messages 模式)与 engine(policy,text 协议,token 模式)并存;token 模式同步维护 messages 作归档/SFT 真源(现有 SessionResult.messages 契约不变)。
   - 会话内实验 episodes 的 token **不**进 research 样本(工具调用内部,researcher 只见叙述);也不作为独立训练样本(partial-guidance 分布特殊,第一版不训)。
   - token 预算:轮数上限之外加 token 数触发的 wrapup(现有 `_wrapup_at` 按轮数,token 模式补按累计长度),超硬限则截断——未 finalize 自然落入 p=0 规则(§3.2-3)。
2. **replica 第三段**:每任务副本布局 `[0,rollout_n) bare | [+mix_sft_rollouts) expert | [+research_n) research`。research 副本的 run():跑一个独立 research 会话(不再 shared_task 去重)→ d_i → 起 K 个 doc-conditioned rollout(复用 `_expert_rollout`)→ p_i → 返回 AgentLoopOutput(response=research 会话 token,reward_score=p_i,sample_role="research")。K 个评测 rollout 全部 dump(观察面 + SFT 数据池),但不进训练 batch。
3. **teacher 选择跨副本同步**:bare 副本 await N 个 research future(gather,shared dict 同款不变式:同任务副本在同一 event loop),取 argmax → `teacher_prompt_ids`。现行 research-与-rollout overlap 优化保留(bare episode 先跑,teacher prefix 在 rollout 结束后才需要)。
4. **research loss 通道(结构定稿见 §7.1,v3 修正)**:research 样本 uid 加 `":research"` 后缀成独立 GRPO 组,组内 `A_i = p_i − mean`(不除 std,§7.2)。**默认 `research_update_mode: joint`**——research 行留在同一 batch,loss 端按 sample_role 分流出第四项 `L_res`(**独立分母 = 全 batch research token 总数,切微批前 all-reduce**,verl 现成设施),`L_total = L_exec + research_rl_weight·L_res` 一次 backward——与生产在跑的三项分流完全同构,完美 on-policy。**备选 `separate`**:research 行 select 出走同一全局 step 内第二次 update(独立打包、原生 GRPO ratio+clip)。`research_rl_weight` 默认 0.0 = 通道关断、逐字节回退现状(不产 research 副本、不算第四项,与三系数同一纪律)。
5. **SFT 数据源切换**:mix_sft 臂的数据源从"额外 expert rollout"切为"N×K doc-conditioned 评测池中择优"(池子免费自带,省掉单独的 expert rollouts;择优规则沿用现行 build_expert_sample 的接受逻辑)。保留旧路径作配置开关,消融可比。

**长度风险(A800 教训直接适用)**:research 样本 = researcher 多轮 + 叙述返回,天然长于 executor 样本,而显存峰值∝最长序列(CD-006 实证)。必须:research 样本总长预算(独立于 executor 的 response_length)+ 超限降级(doc 仍用、轨迹不进训练面,计数 `research/traj_dropped_overlen`)。

## 5. SFT 冷启动

- **数据源(按优先序)**:①**当前生产 run 的 research/ 归档**——35B researcher 的真实会话每 step 都在落盘(`ResearchArchive`,meta 含 researcher_model / final_success / doc_words / error),同域同任务分布,量级 = 16 任务/step × 累计步数 × 3 臂;②H1 pilot 274 任务会话(27B,ALFWorld,`/data3/.../h1_pilot/runs/pilot/`)作补充。
- **协议转换**:生产归档是 fncall 协议(35B client),policy-as-researcher 用文本协议 → 机械转换(tool_call JSON → 文本协议轮;`research/toolproto.py` 两种协议的对偶已在)。
- **质量筛选**:final_success 高 + doc 非空 + 无 error 的会话;可再按"该 doc 的下游增益"筛(归档 meta + 当步 doc-cond 成功率可回查)。
- **训练**:独立离线 SFT 阶段(非训练循环内)。栈:verl `fsdp_sft_trainer` 或 LLaMA-Factory(裁定点 2)。loss 只落 researcher 轮(工具返回 mask=0),与 §4-1 同一 mask 口径。
- **M0 可行性探针(Go/No-Go 门,最小投入,先于一切机制开发)**:冷启动后的 policy-as-researcher 在 held-out 任务上产 doc,K=8 增益探针(evalm 口径),对照外部 35B doc 增益与 bare 基线。三个必测数:finalize 率、doc 增益(p_doc − p_bare)、增益相对 35B 的比例。**探针不达标(如 finalize 率过低或增益≈0)则整个 v3 缓行**——此时钱花在改冷启动(数据/量/协议)而不是机制开发上。

## 6. 调度与算力账(真实配置数字)

现行生产(A0 臂,`configs/experiments/162-4b-a0.yaml`):batch 16 任务 × [1 research 会话(n_exp 预算 12,H1 实测中位 4-5 实验 × k≈4 episodes ≈ 16-20 eps)+ 8 bare] ≈ **28 eps/任务**。

| 档位 | 配置 | eps/任务(估) | 相对成本 |
|---|---|---|---|
| **默认(用户裁定)** | N=8, K=8, n_exp=12 | 8×~18 + 64 + 8 ≈ 216 | ~7-8× |
| 快速实验(调小示例) | N=4, K=4, n_exp=6 | 4×~12 + 16 + 8 ≈ 72 | ~2.5-3× |
| 轮转再省(可选机制,默认关) | 每 step 对 batch 子集做 research(其余任务沿用上次 d*,只跑 OPD) | — | 按子集比例 |

N、K、n_exp 全部进配置(`research_rl.n_docs` / `research_rl.eval_k` 等),默认即全量档;快速实验由实验分支按需下调。

三点说明:①成本估算基于 H1 实测行为中位数,首跑后用真实 wall-clock 校正;②doc-cond 评测池兼职 SFT 数据源(§4-5)已把 expert rollouts 省掉;③bare rollout 组兼职 p_bare 对照(观察面用,不进 reward——组间归一化下它是组常数,进了也会被消去)。轮转档的取舍:信号密度减半 vs 文档新鲜度降半步,倾向算力紧张时才启用。

## 7. 联合优化稳定性设计(定稿 v2,2026-07-28 两轮共十路深读综合)

> 证据基础:十份逐行/逐节深读报告,`docs/research-notes/deepread-{ucob,skill-family,seal-evolver,imbalance-papers,weighting-papers,forget-seed-sdar,agentrl,mtl-optimizers,mt-recipes,industrial-recipes}.md`(全部带 file:line / 论文式编号 / 一手 URL 出处)。第二轮(后四份)按用户要求覆盖**通用**多任务/多目标联合优化:AgentRL 专项(论文+官方代码)、经典 MTL 优化器族重审、多任务 RL 配方与课程、工业大规模多域 RL 实践。本节所有数值与断言可回溯至这些文件。

### 7.1 结构裁定(v3 修正):一次联合更新为默认,双通道为可配备选

> v2→v3 修正说明(用户质疑触发,2026-07-28):v2 曾定稿"双通道两次 update",其核心论据"零先例"经复审属**证据过度解读**——UCOB 分开更新的代码注释动机是 **padding 效率**(writer 2048 vs 执行 512,混批 padding 失控),不是稳定性;我们的 verl 用 seqlen balancing 做 token-budget 打包,**无 padding 问题,该动机不成立**。SEED 的独立更新同样无稳定性证据。且**同批混合 dense+稀疏 PG 我们自己就是正面先例**(生产在跑的 OPD+RL_exec+SFT 三项同批)。

**默认形态 `research_update_mode: joint`——一次联合 backward,归一化按角色解耦**:

```
L_total = L_exec + research_rl_weight · L_res
L_exec  = mix_opd·OPD + mix_rl·RL + mix_sft·SFT          # 现行三项、现行口径,逐字节不动
L_res   = −(1/T_res_global) Σ_{research 行 token} A_i · logπ_θ(y_t)
          # T_res_global = 全 batch research 行 token 总数(切微批前 all-reduce)
```

- **归一化污染的根源是共享分母,不是共享 backward**:L_res 用全局 research token 数作分母,每微批只算分子、跨微批累加自动得到正确全局平均——**数值与打包方式无关**(形式保证)。所需设施 verl 现成(切微批前 all-reduce 全局 token 数,transformer_impl.py:643-657,原生 agg_loss 同款路径)。
- joint 的三个硬优点:①**完美 on-policy**(所有梯度都在采样权重 θ_old 处计算,ratio≡1,无双通道的每步系统性一步 off-policy;clip 保留作安全网但不承担校正);②**w 语义在梯度裁剪下保持**(clip_grad_norm_ 作用于联合梯度 g_exec+w·g_res,只缩总范数不改两项比例——反转下述 separate 模式的 clip 饱和问题);③**工程侵入面最小**(loss 端按 sample_role 加一项分流,与生产在跑的三项分流完全同构;不动 trainer 更新循环)。
- 梯度干扰风险的证据面:CTWA 附录 A 实证 LLM 上目标间梯度余弦≈0(近正交)——近正交时联合一步与序贯两步几乎等价,联合更省;干扰若真发生由监控组捕获(cov、分角色行为指标),一键切 separate。

**备选形态 `research_update_mode: separate`——同一全局 step 内第二次 update**(v2 方案全文保留):research 行 select 出,独立 seqlen 打包、verl 原生 GRPO(ratio+clip,第二次 update 带一步 off-policy 需 clip 保护)、独立归一。启用场景:joint 出现干扰症状(§7.6),或作为消融臂(Jacobi vs Gauss-Seidel 的受控对比——工业界零家做过,本身是学术空白)。UCOB/SEED 的先例形态归位于此。

**两形态共同不变式**:研究侧 loss 独立分母(joint 靠 T_res_global,separate 靠独立批);reward/advantage 细则(§7.2)、系数策略(§7.3)、监控组(§7.4)全部同款;research 样本长度预算(§4)与更新形态无关,两形态同款。

**第二轮通用线的三重加固**(deepread-{industrial-recipes,mt-recipes,agentrl}):
- **工业拆/合判据的定位(v3 校准)**:GLM-4.5 §3.4 与 NVIDIA Nemotron §4.3 独立给出"输出长度与验证耗时相近就联合,差异大就拆(expert 再蒸馏)"。该判据的语境是**训练阶段级**的拆分(单独 expert + 蒸馏回补),其动机主要是 rollout 木桶效应等效率因素;对我们"同 step 内一次 vs 两次 backward"的选择,它约束的是"研究侧必须有独立的归一/预算边界"——joint 模式的分角色分母与长度预算同样满足,separate 是其更彻底的结构化形态。
- **"混批协同增益"零支持、四条独立反证**(Klear 明说无协同;LUFFY 表 4 朴素混合 44.4<45.5;BRIDGE 表 1 固定 λ 同批全线低于纯 RL;MAHALO 表 2 冲突 pool 崩穿 base)。论文叙事按此校准:我们主张的是**数据流耦合的飞轮**(doc→teacher/SFT),不主张 loss 层协同增益。
- **通道调度频率作可配旋钮**:`research_rl.every_n_steps`(默认 1=每 step;Klear 工业配方为固定 3:1 step 级交替、SKILL0 为每 8 步间歇——两者为调大该值提供背书,算力与稳定双赢选项)。2026 工业主流已倒向联合/混合(MiniMax M2 明写 sequential 引发负迁移、Olmo 3 试过 warm-up 判 suboptimal),但**零家做过联合 vs 分阶段受控消融**——我们的消融矩阵(§9)因此是真实学术空白。

**`research_rl_weight` 与梯度裁剪的交互(mtl-optimizers 发现,按模式区分)**:vendored verl 的 fsdp `optimizer_step()` 强制 `clip_grad_norm_`(transformer_impl.py:701)。
- **joint 模式(默认)**:clip 作用于联合梯度,只缩总范数、不改 `g_exec` 与 `w·g_res` 的相对比例——**w 全程语义有效**。这是 joint 的硬优点之一(§7.1-②)。
- **separate 模式**:research 通道 pre-clip 范数 N > clip_grad 时,`wg·c/(wN)=g·c/N`——**w 被精确抵消**,仅欠饱和区线性有效(Adam 的 m/√v 尺度不变性是第二重稀释);该模式下通道强度的硬控制是逐通道 clip_grad(与 lr)。
- 监控(两模式都做):分项 grad_norm 与比值曲线 + clip_grad 水平线。separate 下两通道范数由 `optimizer_step()` 免费返回(:699,740);joint 下分项范数不免费——画像期(前 N step)用"交替只开单项"的画像 step 或 2510.19178 App C.1 交叉积估计器取一次,生产期监控联合范数 + 各项 loss 值 + 分角色行为指标。

### 7.2 research reward 与 advantage 细则(三方证据收敛)

- **组间去均值、不除 std**(或 σ 下限 0.05):UCOB 实证(z-score+σ 下限 0.05)× weighting 报告(连续 [0,1] reward 退化组 std≈0.013 时除 std 放大噪声 ~75×)× Binomial 推导(K=8 噪声底 σ≈0.177)。
- **advantage 硬 clip ±2.0**;组内有效候选 <2 整组放弃;零优势行丢弃(UCOB 三件套)。
- **全对/全错组分角色处理**(imbalance 报告):researcher 侧自然零梯度 + 计数落曲线,**不做丢弃重采样**(一条 research 轨迹的评分要 K 次下游 rollout,重采样代价是 executor 的 8 倍以上);executor 侧维持现状。
- **warmup**:research RL 延迟 ~30 步再开(UCOB `:1012-1016` 实证做法);SKILL0 的"间歇施加"(每 M step 一次)作为算力/稳定双赢的可配选项,默认关。
- **SNR 前置诊断(零成本,开训前必做)**:SEAL 的真实病因可能是评分信噪比 <1 而非漂移(其代码实测组间散度 0.124 < 组内散度 0.159)。用现有生产 run 的 doc 评测数据先估一次"组间方差 vs Binomial(K) 噪声底"——在写任何机制代码之前回答"K=8 够不够"。观察面固化为 `research/reward_snr`。
- 我们"组间同 step 对比"的辩护边界据此修正:**免疫的是评分器漂移,不是低 SNR**(论文表述也按此口径)。
- **通道内各自 adv_norm、不跨通道**(AgentRL 实证背书:其 task advantage normalization 即"GRPO 组内 z-score 之上按任务桶再归一",消融 −5.6;映射到我们,"任务桶"=角色通道)。executor 通道维持组内 z-score;research 通道维持去均值不除 std;跨通道尺度差不做归一(由通道分离天然吸收)。备选对照口径:Embodied-R1.5 §4.2.1 的 group-mean/**batch**-std(分母用 batch 级 std,组退化时不塌)。
- **reward 有界同尺度**:两通道 reward 天然都在 [0,1](executor 二值、research 成功率)——工业实践的实际做法正是"全域压进同一有界尺度"(k1.5 全 {0,1}、Olmo 3 全 [0,1]);R1 的反面教材(二值+连续等权相加吃 reward hacking)对我们不成立,因为两通道 reward 从不相加。
- **researcher 侧不做 DAPO 式零方差过滤的第三证**:纯 zero-variance 过滤在多值/连续 reward 上实测 −9.61(ToolSample 作者自述该设计针对二值 reward)。
- **warmup 硬闸门推广**:不止 research RL 延迟开启——任何动态采样/筛选机制也必须 warmup 后才开(ToolSample 表 3:不 warmup 从 56.96 掉到 47.34)。

### 7.3 系数策略:固定 + 监控,不上自适应

- **固定系数有论文自证**:GAC Table 7——oracle 自适应相对固定权重的全部可得空间仅 5.7pp;HPT Table 6——激进自适应 −2.9~−3.2pp;CHORD 附录 B.7——token 级机制到位后 μ 调度"不再必要"。我们的主效应量级 ±8~28pp,权重自适应是二阶旋钮。**mtl-optimizers 重审后维持且论据升级**:GAC 的 5.7pp 正覆盖 joint 模式的相对幅度轴(同批凸组合语义,与我们默认形态同型);separate 模式的绝对步长自由度则被 clip+Adam 吃掉(§7.1)——两种模式下固定裁定都成立且可在日志上直接验证。
- **经典 MTL 优化器族判决(重审定稿)**:方向手术类(PCGrad/CAGrad/MGDA/GCond)不采纳——CTWA 余弦证据下"付满 K 份梯度换 no-op";幅度均衡类(GradNorm/UW/FAMO)**不受余弦证据株连(上一版误伤,已修正)但各有硬阻断**——UW 的 1/2σ² 推导前提是 NLL(GRPO 代理损失可正可负,σ→0 发散)、GradNorm 有"卡死任务权重发散"病理(与 moving self-teacher 正反馈同型)、FAMO 需正损失与下界。若将来要自动均衡,**调逐通道 clip_grad 而非 loss 系数**(§7.1)。
- **`research_rl_weight` 初值定法**:画像期(首跑前 N step)取分项梯度范数——joint 模式用"交替只开单项"的画像 step(或 App C.1 估计器),separate 模式直接读 `optimizer_step()` 返回值;初值从范数比出发,参照邻居量级(SEED/SDAR λ_opd≈0.01;Skill 家族辅助信号为主信号 1–3%——语义差异:它们是同轨迹辅助项且不带独立分母,可比性有限,以画像为准)。
- **梯度幅度只做诊断与静态对齐,禁止动态课程**(2510.19178 硬结论:梯度正比采样全面劣于均匀)。
- **任何系数调度必须有端到端生效测试**:SDAR 的 λ warmdown 在 trainer 算了却只进 metrics、实际系数是启动时静态拷贝——warmdown 全程无效(反面教材)。测试形式:改系数→断言 loss 数值变化。**同一测试须覆盖 clip 饱和区**(w 在超饱和区无效是机制现实而非 bug,测试要能区分两者)。

### 7.4 监控组(放什么、明确不放什么)

**放**:两通道 pre-clip grad_norm 与比值 + clip_grad 水平线(optimizer_step 免费返回,§7.1)| `cov(research_reward, 总信号)`(CTWA:变负 = 能力退化先行指标)| EPPO 三诊断量(熵 crossover 计数 / 熵-log t 拟合 R² / 尾段最大跳变——N=2 角色数下不退化)| `research/reward_snr` | 组退化率(all_pass/all_fail_frac,AgentRL 的 `effective_uids` 同款只监控不过滤)| finalize 率 / doc_words / doc 跨任务相似度 | 分角色熵曲线。
**不放:梯度余弦**——CTWA 附录 A 实证 per-objective 梯度余弦始终非负收敛到 0 而干扰正在发生(假阴性),且要 2× backward。PCGrad/CAGrad/MGDA 族全部不采纳(同一理由 + 开销)。

### 7.5 防角色遗忘

- research SFT replay:mix_sft 通道天然可载 research 会话样本(同 mask 机制),比例作配置——第一道防线。
- 顺序背书:MGS §3.1 的干扰非对称性支持"research SFT 冷启动在前、联合在后"的现有顺序;SEAL 实测顺序 self-edit 灾难性遗忘(0.3825→0.1894 掉穿 base)作外部警示。
- 参数冻结降级为备选:MIFO 附录 E.4 自证 random 冻结 ≈ TopK 冻结(收益来自限幅非选参)——若需要,先试便宜的 random-50%。

### 7.6 症状 → 升级决策树(预置开关,不预置机制)

| 症状(监控信号) | 升级动作 |
|---|---|
| 组间信号长期零(all_pass/all_fail 高企) | K 增大 / SKILL0 式间歇施加 / 任务课程 |
| cov(research_reward, 总信号) 持续为负 | 调 research_rl_weight / 提 replay 比例 |
| executor 熵崩(OPD 正反馈前兆) | SRFT 熵刹车挂 OPD 项:`w=0.1·sg(exp(+H))`(熵一降自动松手——对冲 moving self-teacher 正反馈的现成负反馈器;注意 SRFT 的 `exp(−H)` 方向与我们需求相反,不抄) |
| research 轨迹质量崩(finalize 掉 / doc 相似度收敛 = 万金油坍缩) | 停 research RL、查冷启动与 reward |
| GRPO 真不稳 | **ReST^EM flag**:只取 argmax_i p_i 做筛选 SFT,不需要 research 梯度,OPD/执行 RL/SFT 三项一行不改(SEAL 备选臂,成本≈0) |
| executor 轨迹上 OPD 与 RL_exec token 级符号打架 | GD²PO 的 SNR 公式移植(两个量现有 loss 里都已算出,零额外开销);其原始过滤器不照搬(会挖掉 OPD 在 RL 无信号区的覆盖) |
| joint 模式出现跨角色干扰(cov 变负 + 分角色行为指标反向移动) | 切 `research_update_mode: separate`(一键,§7.1) |
| separate 模式下调 research_rl_weight 无响应(范数持续压在 clip_grad 上) | 不是机制不敏感,是 w 进了 clip 饱和区(§7.1)——改调逐通道 clip_grad / lr |
| research 信号噪声大 / 算力紧张 | 调大 `research_rl.every_n_steps`(Klear 3:1 交替、SKILL0 每 8 步间歇双背书) |
| research 预算浪费在已毕业/无望任务上 | 开启 §7.7 任务级 UCB 调度(默认关) |

### 7.7 配比与任务级课程(第二轮新增)

- **通道配比:固定,不动态**(AgentRL 固定 1:1 轮转、Klear 固定 3:1 零动态配比双背书;且两通道 advantage 尺度不可比,UCB 会系统性偏向 executor)。executor:research 样本占比由 rollout_n:research_n 决定,起点建议 8:8=1:1 组数(即用户裁定的 N=8;若按样本行数计 8 bare:8 research,token 量 research 侧更大——配比语义在实施时以"组数"为准声明)。
- **任务级 research 预算调度(可选模块,默认关)**:回答"3553 个训练任务里哪些值得花 research 预算"。DUMP 式 UCB:任务族×难度构造 12–18 个 arm(单任务作 arm 会因滑窗填不满退化成均匀采样)、learnability 信号取自 **executor rollout 的 expected |advantage|**(天然覆盖任务毕业:全对任务 learnability→0)、k=300/τ=0.1 照抄、**只动 research 侧采样**以保住归因(executor 侧任务分布不变,消融可比)。实现技巧:pass-rate 窗口(0<p<1)筛选按 ToolSample Algorithm 1 的"Ratio 乘在 advantage 上"落地,不改 dataloader。开启前置条件:warmup 硬闸门(§7.2)。
- 后续深读挂账:**MOPD(2606.30406)**——唯一把 Mix-RL / Cascade-RL / Param-Merge / 蒸馏放同一张表对照的工作,与我们双角色同型,补上工业界缺的受控消融;实施期若要引用其数字,先深读核验。

### 7.8 测试与自检(机制实施随附)

- verl 自检减负结论:两类框架 bug(DeepSpeed 丢 micro-batch / loss 聚合加权错)在 vendored verl 0.9.0.dev 结构上不存在(agg_loss 全局归一、all_reduce 全局 token 数、无 DeepSpeed 路径)。自检重点 = 我们自定义路径:forget-seed 报告 C1–C7 清单,**C2(world_size=1/2/4 跑同一 step 对比 loss/grad_norm)为最强判据**。
- token 面铁律:research 轨迹存原始 token id(UCOB 反面教材:纯文本再分词 + skip_special_tokens 剥 EOS → on-policy 假设破坏且训不到停止符)。
- 系数端到端生效测试(7.3)。
- 论文可用的形式化礼物(记录):HPT Eq 1–2 的目标分裂中代入 π_β=π_θ ⇒ KL 项恒 0 但梯度非零——"崩塌是蒸馏目标的合法解"(CD-003 定论)的一行形式化;冻结 teacher 为真修复的直接解释。

## 8. 文献对刀(novelty 立足点)

| 最近邻 | 已占 | 未占(我们的立足点) |
|---|---|---|
| **UCOB**(2606.29502,最强撞车) | 同参数双视图 OPD + 技能写手下游效用 RL + 进化记忆 + 四项联合 loss | 写手是**事后反思**自身 rollout;无主动实验、无独立 research 会话、无同 step 组间 doc 对比 |
| **Evolving-RL**(2605.10663,新发现,CD-002 未覆盖) | 经验提取器+solver 分开优化、协同共进化,ALFWorld unseen +98.7% rel | 提取仍是 **hindsight**("from past interactions");无主动实验、无特权 teacher OPD、无组间 doc reward。**必引必切割** |
| **ARTS/ARTS\***(2606.21891) | 解耦 scientist 主动实验 + RL 训 scientist(TTT) | scientist 是**另一个模型**、内化进 scientist 自身、executor 冻结;无文档→executor 的 OPD |
| **SEAL**(2506.10943) | self-edit 生成 + 下游表现 reward + 内化 | 内化=SFT 非 OPD;reward 要内层更新(贵且不稳,退 ReST^EM);非 agentic |
| **SEED / OPD-Evolver / Skill0.5** | 内化半壁(hindsight skill/经验 + OPD/蒸馏) | 经验生产者不做主动实验;OPD-Evolver 的"写作"元能力用蒸馏非 RL |

组合立足点:**同一份权重的 researcher–executor–teacher 三位一体 × 主动实验式 research(假设-实验-验证会话)× 同 step 组间 doc reward × OPD 显式内化 × 撤文档裸能力口径**。其中"主动实验被 RL 训练"是 CD-002 判定的 v1 仅存干净空白,本机制正是把系统推向该空白的一步。
**随之而来的义务(CD-002 判为"必须自证的核心消融")**:同一 OPD 管线下"主动实验经验 vs hindsight 反思经验"的对照——机制上我们两种都能跑(反思臂 = research 会话换成"读自己 bare 轨迹写 doc",不耗实验预算,还便宜),这一消融同时回应 UCOB/Evolving-RL 两个最强邻居。

## 9. 消融矩阵与评测

| 臂 | 回答什么 |
|---|---|
| 主线:冷启动 + 联合(OPD+research RL) | 机制主张 |
| −research RL(w_res=0,doc 仍由 policy 产) | research RL 的增量(回应 Agent-R"迭代 SFT 就够"质疑) |
| 外部 35B researcher(现状机制) | self vs external 的 gap 上限 |
| 主动实验 vs hindsight 反思(同管线) | **核心消融**(§8) |
| −冷启动 | 能力边界预期验证(小成本) |
| ReST^EM 替代 GRPO | 稳定性备选(症状触发才跑) |
| 评测 | 现行三件套(训前/特权上限/训后,seen/unseen)+ **research 能力评测**:held-out 任务 policy 产 doc 的 K=8 增益曲线随训练步的变化(这是"research 能力在进化"的直接证据,论文主图候选) |

## 10. 观察面(新增曲线)

- `research/reward_mean|std`(组间 p 分布)、`group_all_fail_frac`、`group_all_pass_frac`(§3.2-2)
- `research/finalize_rate`、`doc_words`、`experiments_used`、`traj_dropped_overlen`
- `research/best_doc_gain`(p* − p_bare,teacher 质量)、`teacher_admitted_frac`(准入门通过率)
- doc 跨任务相似度抽检(防"万金油文档"坍缩,离线)
- `research/reward_snr`(组间方差 vs Binomial(K) 噪声底,§7.2)+ EPPO 三诊断量(熵 crossover / 熵-log t R² / 尾段最大跳变,§7.4)
- 分角色熵、分项 grad_norm(App C.1 便宜估计器)、cov(research_reward, total)(§7.4);**明确不放梯度余弦**(假阴性,§7.4)
- 继承现有全部 OPD/崩塌观察面(thought 重复率、response 长度等)

## 11. 分期

- **M1 机制实施:✅ 完成(f9abbda)**——§4 五件中四件全量落地 + 测试 18 项(数值/接线/回退契约);SFT 数据源切换(§4-5)降为 P2,见 §13。
- **M0 冷启动探针**(数据问题后议,用户裁定):归档转换 → verl SFT → doc 增益探针,Go/No-Go 门保留。
- **M2 联合首跑**:梯度画像 → 定 research_rl_weight 初值 → 小规模全流程;实验安排由用户后续裁定。

## 12. 裁定记录与余留问题

**已裁定(2026-07-28,见文档头)**:N8K8 默认可配;SFT 用 verl 栈;narrator 不参与 self-evolve;消融与实验安排暂缓(§9 保留为设计存档);模块正交性命名(`research_rl_weight` 不入 mix_ 家族)。

**已收口(2026-07-28,两轮共十路深读综合)**:
1. **联合优化稳定性方案**:§7 定稿 v2——结构裁定(双通道,7.1,工业拆/合判据背书)、reward/advantage 细则(7.2,通道内各自归一)、固定系数+免费画像(7.3,含 MTL 优化器族重审判决)、监控组(7.4)、防遗忘(7.5)、症状→升级决策树(7.6)、配比与任务级课程(7.7)、测试自检(7.8)。证据文件:`docs/research-notes/deepread-*.md` 十份。
2. **口径隐性放大问题**:被 §7.1 双通道结构整体消解(research 批独立打包独立归一)。
3. **research_rl_weight 与 clip 的交互按模式区分**(§7.1):joint 下 w 全程有效(clip 不改两项比例);separate 下 w 有饱和区,硬控制是逐通道 clip_grad/lr;分项范数画像与监控为必做。
4. **后续深读挂账**:MOPD(2606.30406,四方案同表对照,与双角色同型)——实施期引用前先核验。
5. **结构裁定 v3(用户质疑触发)**:`research_update_mode: joint`(一次联合 backward、分角色分母)为默认;separate 为可配备选与消融臂。修正过程与理由见 §7.1 头注。

**转入实施(M1)**:§4 五件 + §7.8 测试自检;实验安排与冷启动数据(M0/M2)按用户裁定后议。

## 13. 实施记录与设计偏差(2026-07-28,M1 完成)

**落地物**:verl 真源 3 commits(`6e524ed` 通道本体 / `e3bd3bf` 行为指标隔离 / `af6a8d3` 子批 mini-batch 修复),vendored 重导出;trace_opd `f9abbda`(config `research_rl` 节 / token 构流 `EngineResearcherTokenChat` / agent loop 三段 replica 与组编排 / launcher / 模板 / 18 个新测试,合计 206 全绿)。

**如实记录的设计偏差(P2 挂账)**:
1. **`research_rl.every_n_steps`(间歇施加)未实现**——第一版只有 `warmup_steps` 梯度门控(trainer 端 select drop,文档生产照常);间歇施加需 rollout.n 动态化或非施加步降级逻辑,§7.6 决策树中该升级项标注"待实现"。
2. **SFT 数据源切换(§4-5)未实现**——expert rollout 现行路径保持;评测池兼职 SFT 数据源留 P2(减少一次交付的活动部件;evals 摘要已进 dump,数据不丢)。
3. **doc-cond 评测 rollouts 不逐条 dump**(每 step 最多 16×8×8=1024 条,文件量不可接受)——每条的 won/outcome/env_steps 摘要记入 research 行 dump payload 与归档 meta。
4. **`research/reward_snr` 在线曲线未实现**——组间方差 vs Binomial 噪声底的比值离线从 dump/归档算(字段齐备);开训前 SNR 前置诊断(§7.2)照做,用现有生产 doc 评测数据。
5. **joint 模式分项 grad_norm** 按 §7.3 画像期方案(交替单项)执行,未做常驻曲线;separate 模式天然免费。
6. **separate 模式首跑必验项**:TQ 对拆分子批的行→dp 分配均匀性(launcher 已校验行数整除 n_gpus,分配序未验);`research_update/` 指标组出现。

**机制不变式(实现层已测试钉死)**:`research_rl.weight=0` ⇒ 不产副本、不加字段、不进 fingerprint、loss 逐字节现状;research 行 response_mask 清零 ⇒ 执行侧三项及其分母永不见 research token;teacher 打分对 research 行零 forward;p*=0 不设 teacher;未 finalize p=0;组注册走 finally(无死锁);token 样本=引擎采样 ids 原样(重分词禁绝)。
