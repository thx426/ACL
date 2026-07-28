# 混合目标实验操作手册(CD-006 用)

> 面向"拿到代码后怎么把实验跑起来"。机制原理与设计取舍见
> [`mixed-objective-design.md`](mixed-objective-design.md);本文只讲**操作**。
> 2026-07-26,CD-005 交付。

## 0. 先确认三件事(30 秒)

```bash
cd trace_opd
git log --oneline -1                      # 期望 main 含 "mixed objective OPD x RL x SFT"
grep -n "^  mix_" configs/trace.yaml      # 期望 mix_rl_weight/mix_sft_weight 均为 0.0
python3 -m pytest tests/ -q | tail -2     # 期望 158 passed
```

**默认全关 = A0 = 纯 OPD,与你们现在跑的 4B 主线逐字节一致。**
不改任何配置直接续跑现有 run 是安全的——这是刻意的设计约束,不是巧合。

## 1. 怎么开臂

**`configs/trace.yaml` 是这三个值的真源**(`mix_rl_weight` / `mix_sft_weight` /
`mix_sft_rollouts`)。臂是实验设计参数——它定义这个 run 在优化什么目标,所以它跟
`n_gpus`/`rollout_tp` 那类机器参数不同:**启动脚本不会 sed 改写它们**。

两种改法,任选:

**(a) 改配置文件**——这个臂要长期跑时用:
```yaml
train:
  mix_rl_weight: 1.0
  mix_sft_weight: 0.5
```

**(b) 启动时显式覆盖**——同一份 yaml 轮流跑几个臂时用(推荐):
```bash
# A800(4B 主线所在机器)
bash scripts/run_a800_4b.sh                                                        # A0 纯 OPD(=文件里的值)
bash scripts/run_a800_4b.sh --set train.mix_rl_weight=1.0                          # A1 +RL
bash scripts/run_a800_4b.sh --set train.mix_sft_weight=0.5                         # A2 +SFT
bash scripts/run_a800_4b.sh --set train.mix_rl_weight=1.0 --set train.mix_sft_weight=0.5   # A3 全混

# 162(1.7B 线,同一套写法)
RUN_NAME=trace_opd_1p7b_a3 bash scripts/run_experiment_1p7b.sh \
    --set train.mix_rl_weight=1.0 --set train.mix_sft_weight=0.5
```

`--set` 的三条保证:①走**与配置文件完全相同的校验**(负权重、缺 expert rollout、
research 未启用都会拦);②**打字错误会报错而不是静默失效**(`mix_rl_wieght` 会被
未知键检查抓住);③写进 **launch manifest 的 `config_overrides` 字段**,且 manifest
里的 config 快照是**实际生效的值**——半年后能查出某个 run 是哪个臂。

启动日志会打印 `[launch] config overrides: …`——**开跑前对一眼**,最便宜的防误配。

`mix_sft_rollouts` 默认 4(每任务额外的带文档 rollout 数);SFT 臂开启时 launcher
**自动**把 `rollout.n` 从 8 抬到 12(bare 组仍是 8,expert 是加出来的)。

**`mix_sft_prefix` 默认 `doc`**(2026-07-26 裁定):expert 样本以**它生成时的带文档 prompt**
进 CE ⇒ SFT 优化的是 `policy(·|裸+文档)` = **OPD 的 teacher**,分工是"SFT 修老师、OPD 做
传导"。`bare`(重放到裸 prompt)是对照臂——A2 首跑实测它有 **23.6% 的已接受样本在教
"零观察依据下断言物体位置"**,且现有引用过滤结构性地抓不到那一类,**要用必须先换过滤器**。
详见设计文档 §2.5。

## 2. 建议的实验序列

用户的动机是 1.7B 起不来,所以**优先级从 1.7B 出发**:

| 序 | 臂 | 机器 | 目的 | 停止条件 |
|---|---|---|---|---|
| ① | **A3**(1.0/0.5) | 1.7B 线 | 直取痛点:混合目标能否让 1.7B 上正轨 | 跑到第一个 val 点(test_freq=10)出 seen/unseen 读数 |
| ② | A1(1.0/0) | 1.7B 线 | 若 ① 有效,拆解:RL 锚单独够不够 | 同上,与 ① 同 step 数对照 |
| ③ | A2(0/0.5) | 1.7B 线 | 拆解:SFT 锚单独够不够 | 同上 |
| ⓪ | **A0** | 1.7B 线 | **配对基线,必须与其他臂同代码同数据顺序重跑**——v2 线的历史数据之间隔着代码变更、数据顺序与 researcher 换代(27B→35B),拿它当对照会把这些差异算进目标函数效应 | 同其他臂 |
| ④ | A4 纯 RL / A5 RL+SFT 无 OPD | 1.7B 线 | 回答"混合是否优于单纯 RL""OPD 在弱模型上还有无净贡献" | 同上 |
| 后 | A0 | 4B 主线 | **保持不动**,继续现有 run | — |

**⚠ 更正(2026-07-26 外部审阅)**:此前写"①失败则②③大概率也失败"——**该推断不成立**。
A3 完全可能因 RL 与 SFT 梯度冲突而失败,而 A1 或 A2 单独成功。**不得以 A3 的成败决定是否
跑拆解臂**;先跑 A3 只是因为它最贴近痛点、信息量最大,不是因为它能代表其他臂。
另:**A0 必须配对重跑**(与其他臂同代码、同数据顺序、同 researcher),不能用 1.7B v2 的
历史数据当对照;**A4 纯 RL 与 A5(RL+SFT 无 OPD)也是必跑臂**——没有它们就答不了
"混合是否优于单纯 RL"和"OPD 在弱模型上还有没有净贡献"。完整矩阵见设计文档 §6。

**新开 run,不要 mid-run 切臂**——切臂会改变梯度目标,step 前后不可比,污染唯一一条完整曲线
(与你们当年 mid-run 换 researcher 的判断同理)。

## 3. 首验清单(A3 第一个 step 就能查,请回报)

按重要性排序,**前三条不过就别看 loss 了,先查配置**:

| # | 查什么 | 期望 | 不符说明什么 |
|---|---|---|---|
| 1 | `rollout_extra/sft_role/mean` | **≈0.333**(=4/12) | 副本发号错了。查 `rollout.n` 是否真的是 12、`train_batch_size % agent_num_workers == 0` 是否仍成立 |
| 2 | `rollout_extra/sft_won/mean ÷ rollout_extra/sft_role/mean` | **0.9 量级** | <0.5 说明 expert rollout 本身有问题(doc 没注进去?抓 `runs/*/rollouts/step_*/` 里 `sample_role=="sft"` 的 dump 看轨迹) |
| 3 | `rollout_extra/sft_dropped_ref/mean ÷ rollout_extra/sft_won/mean` | 报数即可 | **这是引用过滤的误杀率,本地无真实数据可校准,我故意做宽了**。>30% 我们收紧词面;≈0 则考虑加强(检测对文档内容的隐式复述) |
| 4 | `sft/expert_tokens` | >0 | =0 说明没有任何专家样本进 loss,SFT 臂等于没开 |
| 5 | `sft/loss` | 有限 | NaN/inf 立即停并回报。**`doc` 前缀下开局就低、几乎不降是预期**——expert 轨迹本就采自 `policy(·│doc)`,只对成功轨迹做 CE 是 filtered BC,teacher 自走成功率越高(实测 87.5%)可转移的概率质量越少。**若确认弱到接近 no-op,那是"该走二期(从 student 错误前缀出发、带 doc、成功恢复的轨迹)"的信号**,不是 bug |
| 5b | guided 成功率 vs bare 成功率的差 | 不应被 SFT 拉大 | teacher 与 student 是同一组权重,SFT(doc) 存在"有 doc 就照做、没 doc 就摆烂"的失败模式:**guided 涨而 bare 不涨**即是它 |
| 6 | `critic/score/mean` | **不因开臂而跳变** | 跳变说明 expert 行漏进了行为指标(隔离-3 失效),此时曲线不可信 |
| 7 | `rl/pg_loss`、`rl/nonzero_advantage_frac` | 有值;后者 ≈ 0.34 @ 5% 成功率 | **组内全败或全成功 ⇒ advantage 恒 0 ⇒ 该组无 RL 梯度,这是预期不是 bug**。`nonzero_advantage_frac` 就是直接量:成功率 5% 时约 34% 的组带梯度,50% 时约 99% ⇒ **RL 的有效强度会随训练自动长大约 3 倍**(隐式 curriculum)。注:metric 前缀是 `rl/` 而非 `actor/`——RL 项不走上游 `ppo_loss`(设计文档 §2.6);`pg_clipfrac` 恒 ≈ 0 是预期(每步一次 optimizer step ⇒ ratio ≡ 1),该指标无信息量 |
| 8 | 显微镜 `mask_mismatch` | 仍为 **0** | 非 0 说明隔离-4 失效,分析链路会读到脏数据 |

## 4. 会变化的运行形态(别当成故障)

- **轨迹数 128 → 192/step**(SFT 臂开时),rollout 墙钟相应上升。
  expert 副本**必须等 doc 就绪才能开跑**(不像 bare 副本能与 research 重叠),
  这是新增的串行段——**若 step 时间超预期,这是第一嫌疑**,回报后我们评估流水化。
- **打分算力不增**:expert 行在 fsdp 后端跳过打分前向。
  (vLLM 回退后端会给它们打一次无用的分,浪费但正确——回退路径零风险优先。)
- **显存**:actor 微批预算受控,但 **rollout 并发从 8 升到 12(KV 占用与 preemption 会变)**、
  开臂关掉 `distillation_only` 后要完整算 log-probs、FSDP 打分虽跳过 expert 前向仍为其分配
  零填充 top-k 张量 ⇒ **峰值未知,以你们的 smoke 实测为准**(此前写"峰值不变"没有实跑支持)。
  但 SFT/RL 任一臂开启会关掉 `distillation_only` 优化(verl 因此计算 `log_probs`),
  这是两个臂的必要成本,**留意训练峰值是否逼近你们的余量**。
- **viz 索引会自动重建一次**(schema 加了 `sample_role` 列 + `PRAGMA user_version` 迁移),
  全量 rescan 几分钟,属预期行为。
- **expert 轨迹在 viz 的可见性**:step 下钻页能看到,带 `expert` 标签;
  但**不进任何聚合**(score/曲线/完整性审计/失败分类)。step 页的 KPI 会多一个
  "expert rows (excluded)" 计数。

## 5. 出问题怎么办

| 症状 | 先查 | 回退 |
|---|---|---|
| 起不来 / config 校验报错 | 校验会明确拦三种误配:负权重、`mix_sft_weight>0` 但 `mix_sft_rollouts=0`、`mix_sft_weight>0` 但 research 未启用 | 按报错改 |
| `sft_mask ... does not match the response layout` | 这是我们加的 fail-loud 断言:某行的 `sft_mask` 与其 responses 长度不一致 | **立即停并回报完整 traceback**——这是结构性问题,不要绕过 |
| `sft_loss_coef > 0 but the batch carries no 'sft_mask' field` | rollout 侧没盖 `sample_role`(agent loop 与 verl 版本不匹配?) | 确认 vendored verl 是本次导出物(`verl/VENDORED.md` HEAD 应为 `c7c2194` 之后) |
| loss 尺度与历史 run 对不上 | 只在开 RL 臂时才可能——我们已把 OPD 项归一化钉死在纯 OPD 口径,若仍不符请回报 | 去掉 `--set train.mix_rl_weight=…` 复现 |
| 想立刻回到已知良好状态 | — | 不带任何 `--set` 重启(配置文件里两个权重为 0)= A0 = 与历史 run 同口径 |

## 6. 分析链路的适配(已就位,无需你们改)

- **蒸馏显微镜**:expert 行不落盘(它们没有蒸馏),所以地形/token 显微镜不受污染;
  `collect` 也不索引 `sample_role=="sft"` 的 dump(避免哈希碰撞抢走真实行的 provenance)。
- **dump 新增两个字段**:`sample_role`("bare"/"sft")与 `sft_accepted`,
  你们做自定义分析时可据此分层。
- **probe 工具链**不受影响(它读 checkpoint,与 batch 组成无关)。

## 7. KL 模式(`opd_kl_mode`,2026-07-27 新增;设计文档 §2.7)

OPD 项内部的散度可换,与三系数正交。**默认 `forward_kl_topk` = 现状,不改配置就没有任何变化**(存量 run 的 fingerprint 也不变,resume 不受影响)。

```bash
# 纯方向单变量(接 CD-003 结论):dense 反向 KL,同一份 top-64 数据
--set train.opd_kl_mode=reverse_kl_topk
# CD-003 消融同款(历史曲线可比):采样点 k3,teacher 打分自动降为单点
--set train.opd_kl_mode=reverse_kl_k3
# SDAR/SEED 文献配方(通常配 RL 臂;λ 用 mix_opd_weight 表达,SDAR 代码默认 0.1)
--set train.opd_kl_mode=reverse_kl_gated --set train.mix_rl_weight=1.0 \
  --set train.mix_opd_weight=0.1   # 可选:--set train.opd_gate_beta=5.0(默认即 5.0)
```

**须知**:
- **换模式 = 换目标**,fingerprint 会拒绝在旧 run 目录上 resume——新模式请换 `run.name` 或新实验配置副本(`config_tool new`)。
- **loss 数值尺度跨模式不可比**(k3/gated 与 forward 完全不同量纲;gated 可为负),判停与对比只看成功率/entropy/response 长度。`sft/loss`、`rl/*` 不受影响。
- estimator 两模式(`reverse_kl_k3`/`reverse_kl_gated`)下,`distillation/student_mass`、`teacher_mass`、`overlap_ratio`、`teacher_topk_entropy` 曲线**自动缺席**(数据上不存在,不是故障);gated 新增 `distillation/gate_mean`、`gate_active_ratio`、`teacher_gap_mean`。显微镜照常落盘(per-token loss;gated 多一列 gate)。
- **reverse 系重点盯 response 长度**:CD-003 记录过 reverse-KL 的 length inflation 伴生症(7186→9193 逼近上限)。
- 打分开销:estimator 模式**低于** forward(无 topk kernel、teacher 传输从 64 列降为 1 列);top-k 两模式开销与现状相同。

## 8. 方向消融三定点:怎么跑(v2.3 实验指导,2026-07-27)

### 8.0 前置:代码同步与在跑 run 的安全性

- 拉 `mixed-objective` 最新 `dd3de2f`(vendored verl 已重导出到 `48c8743`)。同步到服务器后两个必做:
  ①`python3 -m pytest tests/test_vendored_verl.py -q`(vendored 完整性,秒级;有 torch 的机器可加 `tests/test_kl_modes_numeric.py` 跑数值真值);
  ②工作树 status 核对你们的生产热修与活配置没被覆盖(你们四个未提交文件——三份实验配置 + `run_a800_4b.sh`——我这边全程未碰)。
- **在跑的 A0/A1/A2-doc 零动作**:新代码对 `forward_kl_topk` 路径行为等价(唯一差异是打分对齐哨兵从"看首行"放宽为"看前 8 行",只会更少误报),fingerprint 逐字节不变 ⇒ **任何时候重启/resume 都安全,不需要等臂跑完才换代码**。

### 8.1 三定点与优先级(卡位由你们/用户定,建议接各臂出结论后的空位,不抢占)

| 优先 | 定点 | 唯一变量 | 回答什么 |
|---|---|---|---|
| P1 | `A1×gated`:OPD 1.0(gated)+ RL 1.0 | opd_kl_mode(对照 A1×forward) | 同 RL 锚同权重下,OPD 项形式(dense forward vs 采样点 gated)的贡献差——SDAR/SEED 配方核心元件在我们机制下的真对照,也直指 1.7B 原始动机 |
| P2 | `A0×reverse_kl_topk`:纯 OPD,只换方向 | KL 方向本身 | CD-003"方向是二阶因素"在 1.7B + 动态 doc 语境下的复验;这是唯一严格单变量的方向臂 |
| P3 | `A0×reverse_kl_k3` | 形式(方向+单点+估计器) | 主流估计器形式参照。⚠ CD-003 的 `h3_2507_full_rkl3_from10` 是 **4B** 从 step-10 起,与 1.7B 只做**定性**对照,不做数值对照 |
| 备选 | `A1×gated` 文献复刻版:`mix_opd_weight=0.1` | (相对 P1)λ | SDAR 原配方(RL 主干 + 小系数 gated;其代码默认 0.1、论文 0.01)。P1 显示形式有差后再上 |

### 8.2 建副本(以 P1 为例;沿用你们"cp 最近邻副本"的实践)

```bash
cd trace_opd
cp configs/experiments/a800-1p7b-a1-rl.yaml configs/experiments/a800-1p7b-a1-gated.yaml
# 编辑新文件,三处:
#   1) run.name → trace_opd_a800_1p7b_a1_gated(必须换:换模式=换目标,旧目录
#      resume 会被 fingerprint 拒,这是保护不是故障)
#   2) 头部注释写明:派生自 a800-1p7b-a1-rl,唯一变量 opd_kl_mode
#   3) train: 段加两键(你们的副本尚无它们;推荐从 configs/trace.yaml 把
#      「OPD 项的散度形式」整个注释块 paste 过来,最少则两行):
#        opd_kl_mode: reverse_kl_gated
#        opd_gate_beta: 5.0
PYTHONPATH=src python3 -m trace_opd.tools.config_tool diff a800-1p7b-a1-gated  # 顺手核漂移
```

P2 同法:`cp a800-1p7b-a0-pure.yaml → a800-1p7b-a0-rtopk.yaml`,`opd_kl_mode: reverse_kl_topk`(此模式**不需要** `opd_gate_beta`,留默认即可);P3:`opd_kl_mode: reverse_kl_k3`。启动用各机现行方式(162 `run_experiment_1p7b.sh` / A800 你们的现行 env 覆盖),把 config 指向新副本即可。

### 8.3 首验清单(第一个 step 内,7 条)

1. `<run_dir>/effective_config.yaml` + 启动命令行:`loss_mode=reverse_kl_gated`(gated/k3 模式**不应出现** `topk=`/`use_chunked_topk` 行——出现了说明映射没生效;rtopk 应出现 `topk=64`)。
2. trainer stdout 出现 **`teacher_topk_fsdp ACTIVE: backend=fsdp topk=0`**(gated/k3;rtopk 为 `topk=64`)——这行是单点打分真正生效的唯一凭证(判据纪律:看产物不看配置)。
3. 曲线面:gated 应出现 `distillation/gate_mean`、`gate_active_ratio`、`teacher_gap_mean`;estimator 两模式下 `student/teacher_mass`、`overlap_ratio`、`teacher_topk_entropy` **缺席是预期**,不是故障。
4. **gate 健康判读**:`gate_mean` 预期 0.5~0.8 区间(glue token gap≈0 → gate≈0.5;doc 增益 token → 1)。**恒≈1.0 或恒≈0.0 = teacher 前缀/对齐出了结构性问题,立即停并回报**。`teacher_gap_mean` 应 >0——这条就是 v2.2 补充监控要的「teacher−student gap」的 token 级直接度量,gated 模式免费自带。
5. `distillation/loss` 语义:k3 非负(bare 降级行 ≈0);**gated 可为负(合法)**;各模式与 forward 历史曲线不同量纲,只比任务指标不比 loss 数值。
6. 显微镜:`runs/<run>/microscope/rank*.jsonl` 照常落盘,estimator 模式记录含 `kl` 列(gated 另有 `gate` 列);`collect` join 口径不变。
7. 哨兵:若见 `first response rows look like dummy rows` / `looks like the dummy row` 报错 = 打分对齐真出错,带完整 traceback 回报,**不要绕过**。

### 8.4 运行观察与判停

- 三主线照旧:成功率(`critic/score/mean`,只算 on-policy 行)、entropy、**response 长度**——reverse 系的重点(CD-003 记录 length inflation 7186→9193 逼近上限);若 5 步内 p50 涨 >30% 且成功率不涨,按崩塌前兆处理。thought 逐字重复率判别照用(测模型状态,与 KL 模式无关)。
- gated 特有读数:`gate_active_ratio` 的趋势 = 信号自耗尽的直接读数——student 逼近 teacher 时它向 0.5 回落(类比 forward 模式 KL 收敛=文档增益被吃完)。
- 判停沿用现行口径;跨模式误 resume 由 fingerprint 自动拦,无需人肉防。

### 8.5 回退

任何异常:停新臂;原臂配置照旧重启即回已知良好状态(默认模式行为等价、指纹不变)。新臂的 run 目录保留供事后分析,不复用。

## 9. Self-research 通道(policy-as-researcher RL;docs/self-research-design.md)

> 机制 v3(2026-07-28,trace_opd `f9abbda`,vendored verl `af6a8d3`)。与执行侧
> mix_* 三系数**正交**:那边优化"执行更好",这边优化"researcher 更好"。
> ⚠️ **冷启动前置声明**:policy 未做 research SFT 冷启动(M0)之前,开启本通道
> 只有调试/探针意义——1.7B/4B 裸模型的 research 能力≈0,全组 p=0 ⇒ 通道安全
> 空转(无梯度)但纯烧算力(N×会话 + N×K×评测)。正式开启以 M0 探针达标为门。

### 9.0 同步与在跑 run 安全

- **同步物**:`mixed-objective` 分支 HEAD **`52524bb`**(机制 `f9abbda` → codex 修复 `604ddbc` → 自查加固 `52524bb`;vendored verl **`92f8ac9`**)。同一仓库直接 `git log` 核对即可,无需跨机拷贝。
- **在跑臂零动作**:`research_rl.weight` 默认 0 ⇒ 不产副本、不加字段、不加 loss 项、fingerprint 逐字节不变(有钉死测试)。A0/A1/A2-doc/rtopk 各臂随时重启/resume 安全,不需要等臂跑完。
  唯一的既有行为变化(与 research 无关、且只在 **vLLM 打分后端**):expert(SFT)行不再被 vLLM 白白打分,改发对齐 dummy——loss 从不读这两列,数值逐位不变,纯省时间;A2-doc 臂重启后 rollout 段会略变快,不是异常。
- **同步后两必做**:`python3 -m pytest tests/ -q`(应 210 全绿)+ `git status` 核对你们的四个活文件与 rtopk 配置未被覆盖。

### 9.1 开法

```yaml
research:
  backend: engine           # 硬前提(config 校验拦):research 轨迹必须是
  tool_protocol: text       # policy 自己的 on-policy 采样
roles:
  researcher:
    sidecar_sleep: false    # engine 形态没有外部服务可驱动
research_rl:
  weight: 0.02              # >0 开通道。量级参照 §7.3:邻居 λ≈0.01,画像后定
  update_mode: joint        # joint(默认)| separate(消融/退路,一键切)
  n_docs: 8                 # N(用户裁定默认;快速实验 4)
  eval_k: 8                 # K(同上)
  warmup_steps: 30          # 文档生产从第 1 步照常,梯度从第 31 步进
```

算力账:全量 N8K8 每任务 ≈216 episodes ≈ 现行 7-8×;快速档 N4K4+n_exp=6 ≈ 3×。
rollout.n 自动扩为 `rollout_n + (sft_rollouts) + n_docs`,launcher 校验角色行数
整除 n_gpus(不满足直接 FATAL)。

### 9.2 首验清单(第一个 step 内)

1. trainer stdout 出现 `research replica <task>: doc=... p=...`(每任务 N 行)——
   会话真跑在 engine 上的凭证;
2. `self-research warmup: step N <= 30, ... skipped` 行(warmup 生效凭证;
   第 31 步起消失且 `research/pg_loss` 曲线开始非零);
3. 曲线:`rollout_extra/research_rl_role/mean ≈ n_docs/(rollout_n+n_docs)`
   (副本布局正确);`research_rl_reward`、`research_rl_finalized`、
   `research_rl_group_all_fail|all_pass`、`research_rl_best_p` 出现;
4. `research_final_success` 曲线语义升级提示:组模式下它 = **胜出文档的
   K-rollout 实测成功率**(不再是会话内末次实验成功率),与
   `research_rl_best_p` 同值互证;
5. resume 保护:开启通道后 fingerprint 变化,旧目录 resume 被拒是设计行为
   (换 run.name);
6. separate 模式加验:`research_update/` 指标组出现;首跑核对各 dp rank 的
   research 行数均匀(launcher 只校验总数整除)。

### 9.2b 曲线读法与超时要求(codex 审阅补充)

- **曲线是全副本均值,读比值不读裸 mean**:`research_rl_reward` 只在 research 行
  非零 ⇒ 真实 researcher 均值 = `research_rl_reward / research_rl_role`;组统计
  (all_fail/all_pass/best_p)只在 bare/expert 行非零,同理按占比换算。
- **超时层级(config 校验强制)**:`research_rl.weight>0` 要求
  `train.agent_task_timeout_s = 0` 或 `>= research.session_timeout_s + 1800`——
  否则 R3 hang guard 会先于会话自身的超时降级杀掉整个 prompt 组。

### 9.3 判停与回退

- `research_rl_group_all_fail` 长期≈1:飞轮没点着(冷启动不足/任务过难)——
  停通道查 M0,不要调 weight 硬拉;
- `research_rl_finalized` 均值下滑:research 行为退化前兆,对照 doc_words 与
  归档抽读;
- 调 weight 无响应:joint 模式下 w 全程有效(clip 只缩总范数),真无响应先查
  warmup 是否还没过;separate 模式下看 grad_norm 是否压在 clip_grad 上
  (§7.1 饱和语义);
- 回退 = `research_rl.weight: 0`:副本、字段、loss 项、fingerprint 全部消失,
  逐字节回到现行机制(测试钉死)。

### 9.4 机制冒烟方案(M0 冷启动之前唯一该跑的形态)

目的是**验证管线而非效果**:会话在 engine 上真跑、reward 真算、双模式更新真发生、
曲线齐、显存不爆。未冷启动的 policy 大概率产出低质文档(p 低甚至全 0)——
**全 0 是预期读数不是故障**(零信号=零梯度,机制安全空转)。同时它会给出两个
真实数字:裸 policy 在 text 协议下的 finalize 率、doc 增益基线——这两个数就是
M0 冷启动探针的达标线原料。

**冒烟配置(快速档,≈现行 2× 算力)**——从最近臂配置 cp 副本、换 `run.name`、
paste 以下段(全部键显式写,勿依赖默认):

```yaml
research:
  backend: engine
  tool_protocol: text
  n_exp: 4                  # 会话实验预算压缩(冒烟不需要 12)
  max_turns: 12
roles:
  researcher:
    sidecar_sleep: false
research_rl:
  weight: 0.02
  update_mode: joint        # 先 joint;separate 冒烟第二轮
  n_docs: 4
  eval_k: 2
  warmup_steps: 2           # 冒烟把 warmup 压短,第 3 步就能看到 loss 通道
  max_sample_tokens: 10240  # + prompt 1024 < log_prob 打包预算 12288,避开
                            # 10GiB 级 fp32 logits 峰值(§9.2b 警告的来源)
train:
  agent_task_timeout_s: 0   # 或 >= session_timeout_s+1800(config 校验强制)
```

**跑多少**:5-8 个 step 足够回答全部管线问题。卡位不抢生产臂(A800 空闲组
或 162 空窗自行安排)。

**逐步判读**:
- step 1-2(warmup 内):stdout 每任务 4 行 `research replica ...` + 每 step 一行
  `self-research warmup: step N <= 2`;曲线 `research_rl_role/mean ≈ 4/12`;
  `research/pg_loss` 不存在或为 0;**显存峰值记录一次**(rollout / old-log-prob /
  update 三段,对照该臂 research 关断时的基线);
- step 3+:warmup 行消失;若组内 p 有差异则 `research/pg_loss` 非零、
  `research/tokens`>0;若全 0/全同,`research_rl_group_all_fail` ≈1 且 loss=0——
  预期,记录 finalize 率(`research_rl_finalized/research_rl_role` 比值)与
  doc_words 供 M0 参考;
- 任一 step:`research_rl_best_p`>0 时核 rollouts dump 里该任务 teacher 是否
  真用了胜出文档(`doc_missing=0`);
- 结束后:`ls research/` 归档目录应有 policy(engine) 会话(meta 里
  `researcher_model: policy(engine)`、`tool_protocol: text`)。

**separate 冒烟(第二轮,同配置只改一键)**:`update_mode: separate`。加验三条:
`research_update/` 指标组出现;插桩确认 optimizer step=2 / scheduler step=1
(改 constant→cosine 临时跑 2 步比 lr 曲线即可);故意做一个全同 p 组
(eval_k=1 且任务全败)看 `research_update/skipped_zero_signal=1`。

**服务器侧三必做(codex 验证清单精选,上正式实验前完成一次)**:
1. **world-size 梯度等价**:同一固定 batch 在 n_gpus=2 与 4 下各跑 1 step,
   对比 `research/pg_loss` 与 grad_norm——修复后应一致(P0-3 的回归;修复前
   会差 2×);
2. **打包不变性**:同一 batch 改 `ppo_max_token_len_per_gpu`(8192 vs 16384)
   各跑 1 step,`research/pg_loss` 累计值应一致(全局分母的直接验证);
3. **separate 插桩计数**(上段)。

### 9.5 给 M0 冷启动的衔接

冒烟产出的三个数(finalize 率 / doc 增益 / p 分布)直接构成 M0 的 Go/No-Go
基线:冷启动后同口径重测,finalize 率应显著上移、doc 增益应转正。冷启动数据
与训练方案由 CD-005 后续同步,实验侧无需先动。
