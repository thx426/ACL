# OPD 训练过程逐 token 观察分析 — 自主探索任务书(对象:trace_opd_4b_rkl)

> 2026-07-29 装配(CD-005),07-29 修订收窄范围。用途:分派给一个或多个**自主分析
> agent**,**只**基于已完成的 4B 纯 OPD reverse-KL 实验 **`trace_opd_4b_rkl`**
> 记录下的全部产物数据(核心是显微镜系统承载的逐 token 观察包),做**穷尽式**
> ——逐个 step、逐条轨迹、逐个 token——的数据驱动观察、分析、推断。本文档只提供:
> 问题、已确认的观察事实、数据资产、工具用法、产出与纪律要求。**分析路线、指标
> 构造、假设生成、判决设计全部由执行 agent 自主规划** —— 这正是本任务的核心要求:
> 结论必须从数据里长出来,而不是从任何人(包括本文档作者与前人文档)的预设里
> 长出来。

---

## 1. 使命与铁律

**使命**:基于 `trace_opd_4b_rkl` 全部 17 步的真实轨迹与逐 token 的 teacher/student
分布数据,回答 §4 的四个关键问题,产出可复核的关键信息、证据与结论,并给出对
方案机制设计与实验设计有建设性的改进输入。**覆盖方式是穷尽式**:统计层处理必须
覆盖全部 step × 全部轨迹 × 全部 token;深读/下钻层做不到全量时,选取规则与覆盖
范围必须显式声明(见铁律 8)。

**铁律(违反任何一条即产出作废)**:

1. **一切结论必须来自实际数据观察**。严禁编造、估算、脑补数字;算不出来、查不到
   的写"未确认"并说明缺什么。
2. **每个数字必须带可追溯出处**(文件路径 / API 调用参数 / 脚本路径 + 运行方式),
   使他人能独立复算。
3. **观察(数据里看到了什么)与推断(你认为它意味着什么)在行文中严格分层**,
   不得混写。推断必须标注置信程度与依赖的前提。
4. **前人材料不是证据**。本仓已有若干分析文档与机理理论(§5.3、§5.4),它们是
   "待检验的前人假设/初步分析":可以作为线索,可以被你的数据支持或反驳,但
   **不得未经独立复核就引用为结论**。引用时必须标注"已复核(附你的复核路径)"
   或"未复核,仅作线索"。
5. **结论需要双层证据**:统计层(跨样本/跨 step 的量化)与个例层(具体 step、
   具体轨迹、具体 token 位置,给出显微镜可直接打开复现的定位)。只有聚合数字
   没有个例、或只有个例没有总量,都不算完成。
6. **主动构造替代解释**:对每个核心结论,至少提出一个合理的替代解释,并说明数据
   为何支持你的解释而非替代解释(或承认无法区分)。
7. **否定结果也是结果**:查了没查到、假设被数据否掉、指标无判别力,都要写入报告,
   同样带路径。
8. **覆盖透明(穷尽式要求的落地形式)**:默认要求全量——17 步全部、每步全部轨迹、
   轨迹内全部 token 都进入统计处理;任何抽样、截断、跳过(包括"只细读了 N 条
   轨迹全文"这类深读层取舍)都必须显式声明选取规则、数量与理由。每份报告附
   **覆盖账单**:哪些 step/轨迹/token 范围被以何种深度处理过(全量统计 / 逐条
   细读 / 未触及),不允许静默抽样冒充全量。

---

## 2. 研究对象与最小背景

### 2.1 训练机制(被观察对象)

- **OPD(on-policy distillation)**:student(Qwen3-4B-Instruct-2507,**无**任务
  指导文档)在 ALFWorld 任务上自主 rollout;teacher = **同一权重的同一模型 +
  任务指导文档(特权信息,以 `## Task guidance` 前缀注入)**,在 student 自己
  rollout 出的 token 序列上逐位置计算 top-64 分布;训练目标 = 两分布间的 KL 蒸馏
  (全 response token,response_mask 内)。
- **A0 语义(重要)**:teacher 与 student 权重相同且随训练同步更新(moving
  self-teacher),两者在任一位置的**唯一条件差异是文档**。
- 交互协议:多轮对话式,`<reasoning>...</reasoning>` + `<action>...</action>` 契约,
  环境返回 observation,append-only 构流;won/lost 由环境判定。
- 两臂消融:`opd_kl_mode = forward_kl_topk`(前向)与 `reverse_kl_topk`(反向),
  除此单一变量外配置相同。**本任务对象 `trace_opd_4b_rkl` 即其中的 reverse 臂**。

### 2.2 分析对象(唯一)与背景 run

**本任务的分析对象只有一个 run**:

- **`trace_opd_4b_rkl`**(zkgy 162 四卡,`opd_kl_mode=reverse_kl_topk`,sp=2):
  从头训练至 **step 17 崩溃**(崩因见 F2),2026-07-29 已停机。**唯一带完整逐
  token 观察包(obs npz,17 步,5.0G)的 4B run** —— 显微镜的全部能力都作用在
  这份数据上。

同期另有三条 4B run(`trace_opd_4b_a800_ddp8`/又称 v1、`trace_opd_4b_narpar`、
forward 臂 `trace_opd_4b_sp2`)——**它们的数据不在本任务范围内**,仅在 §3 以
背景事实的身份出现(其数字不得移植到 rkl 头上,见 §3 注意事项)。

服务器与访问方式:`docs/infra/README.md` + `docs/infra/servers-zkgy.md`
(工作区 `/data3/wushang/workspace/TRACE-OPD`)。

---

## 3. 已确认的观察事实(这些是**现象**,不是解释;解释是你的任务)

⚠ **注意事项**:下列 F1/F3 的具体数字测自**其他 run**(背景地位)。rkl 自身在
这些维度上的形态——相变是否同样发生在 step-9、幅度多少、KL 空间分布如何——
**必须由你在 rkl 数据上独立确立**,不得移植背景数字。F2 中 reverse 列直接测自
rkl,是本任务范围内的硬事实。

**硬测量**(有数据与出处,可直接复核):

- **F1 step-9 行为相变(背景,测自 v1 与 sp2 臂)**:v1 臂 step 8→9:`go` 动作
  占比 9.8%→41.6%,训练任务成功率 0.8%→35.9%(`docs/experiments/
  cd006-run-observations.md` 条目 (66));sp2 臂在 step-9 复现同一相变(同文条目
  (80)⑥ 与后续确认条目):score 0.008→0.359、`go to` 41.7%、无效动作(`nothing`)
  归零。rkl 的长度轨迹亦在 step-9 跳变(F2 表),行为层是否同步相变待你确立。
- **F2 长度增长与 step-17 崩溃(双臂逐点对照,单位=字符)** —— 出处
  `docs/collab/cd005-cd006/requirements.md` §3-W(数据源为两 run 的 rollouts dump):

  | step | reverse p50 | reverse p90 | reverse max | forward p50 | forward p90 | forward max |
  |---|---|---|---|---|---|---|
  | 1 | 10017 | 12232 | 14041 | 10016 | 12399 | 15360 |
  | 8 | 16201 | 21197 | 27028 | 16743 | 21461 | 33106 |
  | 9 | 20135 | 35059 | 54840 | 22746 | 40329 | 54396 |
  | 16 | 26826 | 51064 | 58841 | 38651 | 57302 | 72768 |
  | 17 | 38008 | 66267 | 85737✗ | 9258† | 69466 | 100680✗ |

  †forward step-17 只跑出 63/128 条即崩。两臂在**同一步、同一类错误**(CUDA illegal
  memory access 等,崩溃日志 `logs/crash_rkl_step17_0729_094827/`)倒下;长度轨迹
  逐点重合(step-16 reverse 反而更短)→ 长度失控与 KL 方向无关(§3-W 观察 2)。
  step-9 同时是长度起飞点(p90 +65%)。每步墙钟 880s→1923s 与长度同步。
- **F3 KL 空间分布的初步观察(背景,测自 narpar 臂小样本)**:narpar 臂 668 条
  旧版 token 级记录的分析结论——"信号 96.5% 落在 reasoning 段;action 段 KL 在
  step 8→9 断崖;最强纠正是压制 `examine`;`cool/heat` 失败是 teacher 也不会,
  `look_at` 失败是样本不足"(observations 条目 (73))。**此分析基于旧版落盘
  (字段仅 `ids/mask/kl/s_mass/t_mass/overlap/t_ent`)、forward 类蒸馏、668 条
  小样本;rkl(reverse、全量、四元组)上这些结论是否成立,由你独立确立——复核
  与推翻同样有价值。**
- **F4 thought 逐字重复率的强判别性**:赢轨迹重复率恒为 ~0,输轨迹 26–70%
  (observations 条目 (112)–(116),脚本 `scratchpad/repeat_watch.sh`)。
- **F5 val 评测噪声底**:val 用 greedy 但**非确定性**——同 checkpoint 重跑 won 差
  可达 3pp;CD-006 已定线:**<2pp 的 val 差异不作结论依据**(requirements.md
  517 行附近)。训练分数曲线(每步 128 条训练 rollout)另有采样噪声,量级自查。

**印象级(人工看轨迹形成的印象,真伪与准确形态本身是待查对象)**:

- **F6** 训练后期(约 step 10+)轨迹"越来越长但像是在不停说一些没有意义的话",
  长度增长未伴随成功率提升。
- **F7** teacher 接续 student 轨迹越深,其 token 分布与 student 的分布"看起来越
  趋同,似乎提供不了有效信号"。
- **F8** KL 信号的空间分布随训练发生迁移(与 F3 同源的人工印象版本)。

---

## 4. 关键问题(按问题分派,每问附验收标准)

**总则**:所有问题都在 `trace_opd_4b_rkl` 的数据上回答;所有验收标准都默认叠加
铁律 8 的穷尽覆盖要求(统计层全量 17 步 × 全轨迹 × 全 token,深读层覆盖账单
透明)。F1/F3 类背景事实只可用作对照参照,不可替代 rkl 自身的测量。

### Q1 — step-9 跳变前后,OPD 训练到底改变了 policy 的什么?

step 1–8 分数几乎不动,step 9 跳变(F1)。这段时间里蒸馏训练每步都在发生——
它在改变什么?改变发生在哪里(哪些 token、哪类位置、哪层行为)?为什么效果以
"跳变"而非渐变的形式兑现?跳变兑现的到底是什么(哪些任务、靠什么新行为赢的、
这些行为从哪来)?

**验收**:给出 token 分布层 + 行为层互相咬合的机制解释;解释必须能说明"为何前
8 步分数平、第 9 步跳";必须定位到具体证据(哪些任务在 step 8→9 翻转、翻在哪个
决策点、该点分布在此前各步如何演变);必须检验 F3 的小样本结论在全量 obs 上是否
成立。

### Q2 — 长度失控的本质成因是什么?

F2 显示长度从 step-9 起持续暴涨直到物理崩溃,且与 KL 方向无关。这是训练过程中
什么样的累积效应?在逐 token 数据上,"变长"具体是什么在变长(哪个段、新增轮次
还是每轮变长)?新增的内容是什么性质(F6 的"没有意义"是否属实,如何量化)?
teacher 分布在与"继续生成 vs 收尾"相关的位置上呈现什么形态,它随训练如何变化?
由此可导出哪些有数据依据的改善方向?

**验收**:长度增长的构成分解(带数字);"无意义"印象的量化判定(证实或证伪);
一条从逐 token 证据出发、能解释"为何蒸馏训练会系统性推动变长"的机理链条(每环
带证据),或明确报告数据不足以定因并指出缺什么;改善方向必须逐条注明依据的观察。

### Q3 — teacher 信号的真实形态:趋同是否属实?对"开局就走错"的轨迹,数据里存在哪些可利用的早期信号?

OPD 的结构约束:teacher 只能在 student 自己的轨迹上逐位给分布。F7 的印象是轨迹
展开越深 teacher/student 越趋同、信号消失。请用数据回答:趋同是否属实、发生在
什么位置/深度/token 类型上、随训练步如何演变;lost 轨迹上 teacher 是否还持有
student 没有的信息(在哪、什么形态);对于"student 一开始思路就错、走上错误路径"
的轨迹,序列早期的 token 分布/统计量/其他层面,是否存在能把它与最终成功轨迹
区分开的信号,最早在何处可分、判别力多强。

**验收**:趋同命题的量化判决(在什么口径下成立/不成立);lost 轨迹残余信号的
定性与个例;"早期信号"的候选清单——每个候选给出判别力的量化评估(含最早可判别
的轮次/位置)与失败候选的否定记录;如果发现有潜力支撑"及时发现走错 + 高效供
信号"的机制改进,说明数据依据与适用边界。

### Q4 — 开放巡查:上述问题未覆盖的一切有价值发现

数据里任何未被 Q1–Q3 覆盖的模式、异常、反直觉现象、机制洞察、数据质量问题
(采集缺陷、口径陷阱)都在范围内。宁可报告"看过没发现"也不要不看。

**验收**:每条发现独立成条,同样满足铁律 1–7;数据质量问题单列一节(这直接影响
其他任务卡结论的可信度)。

---

## 5. 数据资产清单

### 5.1 主数据源:obs 观察包(npz,逐 token 四元组)

- **位置**:162 `/data3/wushang/workspace/TRACE-OPD/trace_opd/runs/trace_opd_4b_rkl/obs/`
  (`step_00001` … `step_00017`,共 5.0G)。每步目录 `rank*.npz`。
- **内容**(经 `trace_opd.microscope.reader.StepPack` 重组后,每样本一行):
  - `response_ids`(R,)student 实际生成的 token id 序列(R=response 长度);
  - `teacher_ids/teacher_lps`(R,64)teacher 在每个位置的 top-64 候选 id 与 logprob;
  - `student_ids/student_lps`(R,64)student 自身 top-64;
  - `student_support_lps`(R,64)student 在 **teacher 支撑集**上的 logprob(与
    teacher_ids 对齐);
  - `realized_student_lp`(R,)实际采样 token 的 student logprob;
  - `token_loss`(R,)训练端实收的逐 token loss(**真值锚**,直接来自训练存档);
  - `response_mask`(R,)1=参与训练的生成 token,0=observation/框架 token;
  - `covered`(R,)该位置是否有 teacher 数据(SP 段重组覆盖);
  - `turn_spans` 每轮的 [起,止) token 区间;`task_id`/`sample_role`/`won`/`reward`/
    `resp_hash`;pack 级 `loss_mode`/`log_prob_min_clamp`。
- 每步约 128 条训练 rollout(样本数以实际盘点为准)。

### 5.2 轨迹与评测数据(文本层,均为 rkl 自身产物)

- 训练 rollout dump:`runs/trace_opd_4b_rkl/rollouts/step_NNNNN/`(每条含完整
  多轮轨迹文本、meta);val dump:`runs/trace_opd_4b_rkl/rollouts/step_NNNNN_val/`
  (含 `split` 字段区分 seen/unseen;val 执行频率与覆盖以 T0 盘点为准)。
- 训练日志与崩溃存档:`logs/`(每步标量打印)、
  `logs/crash_rkl_step17_0729_094827/`(step-17 崩溃完整日志)。
- ⚠ `trace_opd_4b_rkl` 曾于 `2026-07-29T09:50:14` 从 ckpt_15 恢复重跑 16–17,
  这两步目录混有新旧两批文件,**统计必须按文件 mtime 切分**(方法见
  mixed-objective 分支 `docs/mixed-objective-runbook.md` §5.3)。
- 训练标量曲线:swanlab 在线(凭据 CD-006 持有,不在本文档);本地可从训练日志
  `logs/` 提取。**严禁运行 `swanlab sync`**(已实证会终结在跑 run 的云端记录)。

### 5.3 前人分析材料(线索,非证据;引用须复核标注)

- `docs/experiments/cd006-run-observations.md`(6850 行,编号条目底账;与本任务
  最相关:(64)–(80) 4B 各 run 过程记录、(73) KL 空间分布、(112)–(116) 重复率)。
- `docs/experiments/cd006-opd-signal-decay-analysis.md`、
  `cd006-opd-ineffectiveness-diagnosis.md`、`cd006-doc-value-probe.md`。
- `docs/collab/cd005-cd006/requirements.md` §3-V(audit 判据背景)、§3-W(长度/
  崩溃)、§3-Z(工作台性能实测)。
- narpar 臂旧版 token 显微镜 `runs/trace_opd_4b_narpar/microscope/rank*.jsonl`
  (668 条,F3 的数据源;**非 rkl 产物**,仅作背景对照,不进入本任务分析范围)。

### 5.4 机理理论文档(**待检验假设**,地位同 5.3)

`experiments/h3_pilot/notes/stability_survey_v2/`(崩塌机理统一理论与预测 P1–P5)、
`experiments/h3_pilot/notes/key_conclusions_and_roadmap.md`。这些形成于本 run 之前
的 1.7B/静态文档时代;你的数据可以支持、细化或反驳它们——三种都是有价值产出。

---

## 6. 显微镜系统使用手册

代码:`trace_opd` 仓 `src/trace_opd/microscope/`(reader/derive/segments/server)。
两种用法,**批量统计走库,个例下钻走工作台**。

**分支与版本说明**:被观察实验跑在 `mixed-objective` 分支,显微镜子系统在
`mixed-objective` 与 `intervention-opd` 两分支**同步维护且已验证逐字节一致**
(`git diff intervention-opd mixed-objective -- src/trace_opd/microscope/` 为空,
截至 mixed-objective `c9e9fa9` / intervention-opd `5f0aa94`),本手册对两分支同样
成立;数据格式兼容为**实证**(CD-006 已用本工具在 rkl 全 17 步数据上跑通,见
requirements §3-Z)。执行前自查:所用代码树的 `git log -1 -- src/trace_opd/microscope/`
须不早于上述 commit(旧版本缺 audit 新判据、字符串 hash 等修复);162 生产树若旧,
从任一分支拉取显微镜目录即可,**不得顺手同步其他目录**(162 树有生产改造,同步
覆盖有前科)。

### 6.1 库接口(批量分析推荐;CPU-only,直接读 npz)

```python
# PYTHONPATH=src,工作目录 trace_opd/
from trace_opd.microscope.reader import list_steps, load_step
from trace_opd.microscope import derive, segments

steps = list_steps(OBS_DIR)                 # [1..17]
pack  = load_step(str(OBS_DIR), 9)          # lru_cache(8);大步首载约十秒级
rows  = pack.rows()                         # 每行 = §5.1 的字段 dict
tr    = derive.tracks(row, pack.log_prob_min_clamp)
# tr 的 key(逐 token,均为 (R,) float):
#   kl_forward        前向 KL(teacher‖student),teacher top-64 支撑集上
#   kl_reverse        反向 KL(student‖teacher),student top-64 支撑 + 尾桶修正,
#                     与训练端 reverse_kl_topk 逐项同构
#   token_loss        训练实收 loss(真值锚;mask 外/未覆盖为 NaN)
#   gap               realized token 的 teacher_lp − student_lp;NaN = 该 token
#                     不在 teacher top-64(teacher"弃权",本身是信号)
#   teacher_entropy / student_entropy   各自 top-64 分布熵
#   overlap_count     两侧 top-64 候选集交集大小(0–64)
#   teacher_mass / student_mass_on_support / student_mass_own   支撑集概率质量
#   realized_teacher_lp / realized_student_lp / student_rank / teacher_rank
#   mask / covered
```

段级仪器(reasoning/action 切分已内置,无需自写解析):

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/data3/wushang/model/Qwen/Qwen3-4B-Instruct-2507")
pieces = tok.convert_ids_to_tokens(row["response_ids"].tolist())
seg = segments.segment_spans(pieces)          # <reasoning>/<action> 半开区间
m   = segments.segment_mask(seg, "reasoning", len(pieces))
agg = segments.aggregate_segments(row, tr, seg)   # 每段核心轨道均值 + KL 预算份额
cts = segments.closing_token_stats(row, tr, seg, pieces)
#   ^ 每个闭合标记(</reasoning> 等)首 token 处双方 realized logprob
```

单点分布对照:`row["teacher_ids"][pos]`×`row["teacher_lps"][pos]` 对
`row["student_ids"][pos]`×`row["student_lps"][pos]` 直接比即是 teacher/student 在
该前缀下的完整 top-64 视图。

### 6.2 工作台(server + 浏览器,个例下钻与取证截图)

```bash
cd /data3/wushang/workspace/TRACE-OPD/trace_opd
PYTHONPATH=src python3 -m trace_opd.microscope serve \
    --obs-dir runs/trace_opd_4b_rkl/obs \
    --tokenizer /data3/wushang/model/Qwen/Qwen3-4B-Instruct-2507 \
    --port 8643 --preload 4        # 端口勿撞 8301(vLLM)/18888/18889(隧道)
PYTHONPATH=src python3 -m trace_opd.microscope audit --obs-dir runs/trace_opd_4b_rkl/obs
```

四个视图:**样本**(每步样本表,按 kl_mean/kl_max/gap/overlap/熵等排序筛选)→
点行进**轨迹**(五轨道概览带 + 悬停/点击固定的局部放大镜 + 逐 token 换行细节热图
+ token 文本 + 单点 teacher/student 分布 inspector;页内"热力图图例与指标说明"
面板有全部轨道语义)→ **检索**(全库位点级检索,四模式:指标 Top /
verdict_conflict(两侧 top-1 不同的高分歧位)/ repeat(重复 n-gram 起点)/
teacher_abstains(realized 不在 teacher top-64))→ **演化**(单 task 的
step × turn 指标网格,跨步追踪同一任务)。

HTTP API(脚本可直接调,返回 JSON;`hash` 一律按**字符串**处理,64 位整数会在
JS/JSON 层失真):

| 端点 | 参数 | 内容 |
|---|---|---|
| `/api/meta` | – | step 列表、体积 |
| `/api/samples` | `step` | 步内全样本摘要表 |
| `/api/sample` | `step,hash` | 单轨迹骨架(summary/turn_spans/loss_mode) |
| `/api/sample_tokens` | `step,hash,lo,hi` | token 文本窗口 |
| `/api/sample_tracks` | `step,hash,lo,hi` | 轨道窗口(lo=0 时附 audit+segments) |
| `/api/position` | `step,hash,pos` | 单点双侧 top-K 分布 + 标量 + 上下文 |
| `/api/leaderboard` | `metric,lo,hi,mode,limit` | 全库位点检索(四模式同上) |
| `/api/segment_report` | `lo,hi,role` | **段级一揽子扫描**:逐 step、won/lost 分层的各段 KL 预算、熵、closing-token 概率、长度、重复率 |
| `/api/evolution` | `task,metric` | task × step × turn 网格 |
| `/api/tasks` `/api/status` | – | 任务列表 / 加载与预热状态 |

### 6.3 已知口径与坑(全部为实测事实)

- **audit 先行**:开始分析前对数据跑 `microscope audit`。reverse_kl 数据存在 fp16
  尾桶效应,判据为 `exceed_frac ≤ 0.5%`(工具已内置;背景见 requirements §3-V)。
  audit 不过的步,分布级数字须注明保留意见。
- **性能**:step-9+ 的大步,`/api/samples`/tracks 首次派生曾实测 502s,持久缓存
  (`obs/step_*/_cache/`,纯派生物可随意删除重建)命中后 0s;`--preload N` 自动
  后台预热。批量脚本首次全量扫描请预留时间或复用 `_cache`。
- **val ±2pp 结论线**(F5):任何 <2pp 的 val 差异不作结论;跨 run 比较建议按
  ±5pp 心算误差棒。
- **rkl step 16–17 双批混存**,mtime 切分(§5.2)。
- `token_loss` 是训练真值锚:你的任何派生量若与它矛盾,先怀疑你的口径。
- loss 曲线聚合口径为 **SUM**(非 mean),读训练曲线时注意。

---

## 7. 产出要求

- 交付位置:`trace_opd/docs/analysis/`,每张任务卡一份报告
  `T<N>-<slug>.md`,外加 `INDEX.md` 汇总(结论先行,每条结论一行 + 指向报告小节)。
- 报告结构:**结论先行** → 证据(观察层,带路径)→ 推断(解释层,带置信与前提)
  → 替代解释及其排除 → 对机制/实验设计的建议(若有,逐条注明依据)→ 复核指引
  (他人如何用最少步骤验证你的核心数字)→ 未决与否定结果。
- 个例取证:给出工作台定位(step + task_id + resp_hash + pos),必要时附截图。
- 使用中间脚本一律落盘(建议 `docs/analysis/scripts/`),报告引用脚本路径 +
  调用命令,保证可复算。

## 8. 边界与安全纪律

1. **162 上可能有在跑训练(干预点 V1)。本任务全程 CPU-only:禁止启动任何占用
   GPU 的进程**;若某分析确需推理/续跑类实验,写入报告"待批实验需求"一节,
   不得自行执行。
2. 一切数据目录**只读**(唯一例外:`obs/*/_cache/` 派生缓存)。不改任何 `runs/`
   内容、不删不动 checkpoint、不碰任何在跑 run 的目录。
3. 严禁 `swanlab sync`(§5.2)。
4. 大体积拷贝(如 rsync obs 到本地分析)允许,注意磁盘与带宽,报告记录副本路径
   与同步时刻。
5. 工作台端口避开 8301/18888/18889;用完关闭进程。

## 9. 任务卡(可并行分派;T0 是所有卡的前置)

- **T0 数据盘点与质量门(rkl 专属)**:盘点 rkl 的实际数据覆盖(obs 每步样本数/
  体积/轨道完整性、rollouts 与 val dump 的步覆盖、resume 混批的精确范围与切分
  账),对全 17 步跑 audit,产出"数据事实表 + 可信度分级",供 T1–T4 引用。
  发现的口径问题立即通报。
- **T1 = Q1**(step-9 跳变解剖)。输入:§3 F1/F3/F5、§5、§6。
- **T2 = Q2**(长度失控成因)。输入:§3 F2/F6、§5、§6。
- **T3 = Q3**(teacher 信号形态与早期信号)。输入:§3 F7/F8/F4、§5、§6。
- **T4 = Q4**(开放巡查 + 数据质量汇总)。输入:全文档。

T1–T3 各自独立可跑;若单 agent 承接多卡,T0 只做一次。跨卡发现归属模糊时就近
写入并在 INDEX 交叉引用,不要重复分析。多 agent 分派也可按 step 区间 / 轨迹子集
切分同一问题,但**合并后的覆盖必须达到铁律 8 的全量要求**——切分方案与各自
覆盖范围写入各报告的覆盖账单,INDEX 汇总核对无缝隙、无重叠计数。
