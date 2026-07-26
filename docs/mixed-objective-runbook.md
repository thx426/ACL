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
