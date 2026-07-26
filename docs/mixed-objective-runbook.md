# 混合目标实验操作手册(CD-006 用)

> 面向"拿到代码后怎么把实验跑起来"。机制原理与设计取舍见
> [`mixed-objective-design.md`](mixed-objective-design.md);本文只讲**操作**。
> 2026-07-26,CD-005 交付。

## 0. 先确认三件事(30 秒)

```bash
cd trace_opd
git log --oneline -1                      # 期望 main 含 "mixed objective OPD x RL x SFT"
grep -n "^  mix_" configs/trace.yaml      # 期望 mix_rl_weight/mix_sft_weight 均为 0.0
python3 -m pytest tests/ -q | tail -2     # 期望 154 passed
```

**默认全关 = A0 = 纯 OPD,与你们现在跑的 4B 主线逐字节一致。**
不改任何配置直接续跑现有 run 是安全的——这是刻意的设计约束,不是巧合。

## 1. 怎么开臂

两个启动脚本都已接通,**只用环境变量,不要手改 yaml**(手改会与脚本的 sed 派生打架):

```bash
# A800(4B 主线所在机器)
MIX_RL=0.0 MIX_SFT=0.0  bash scripts/run_a800_4b.sh     # A0 纯 OPD(默认,可省略)
MIX_RL=1.0 MIX_SFT=0.0  bash scripts/run_a800_4b.sh     # A1 +RL
MIX_RL=0.0 MIX_SFT=0.5  bash scripts/run_a800_4b.sh     # A2 +SFT
MIX_RL=1.0 MIX_SFT=0.5  bash scripts/run_a800_4b.sh     # A3 全混

# 162(1.7B 线,同样两个变量)
MIX_RL=1.0 MIX_SFT=0.5 RUN_NAME=trace_opd_1p7b_a3 bash scripts/run_experiment_1p7b.sh
```

启动日志会回显 `混合目标: RL=… SFT=…`,派生 yaml 里也能看到三个 `mix_*` 键——
**开跑前先在日志里对一眼**,这是最便宜的防误配手段。

`mix_sft_rollouts` 默认 4(每任务额外的带文档 rollout 数),需要调就改 yaml;
SFT 臂开启时 launcher **自动**把 `rollout.n` 从 8 抬到 12(bare 组仍是 8,expert 是加出来的)。

## 2. 建议的实验序列

用户的动机是 1.7B 起不来,所以**优先级从 1.7B 出发**:

| 序 | 臂 | 机器 | 目的 | 停止条件 |
|---|---|---|---|---|
| ① | **A3**(1.0/0.5) | 1.7B 线 | 直取痛点:混合目标能否让 1.7B 上正轨 | 跑到第一个 val 点(test_freq=10)出 seen/unseen 读数 |
| ② | A1(1.0/0) | 1.7B 线 | 若 ① 有效,拆解:RL 锚单独够不够 | 同上,与 ① 同 step 数对照 |
| ③ | A2(0/0.5) | 1.7B 线 | 拆解:SFT 锚单独够不够 | 同上 |
| — | A0 | 1.7B 线 | 已有历史数据(v2 线 step-29),**不必重跑** | — |
| 后 | A0 | 4B 主线 | **保持不动**,继续现有 run | — |

**先 ① 后 ②③ 而不是并行三臂**的理由:①失败则②③大概率也失败(省两次 run);
①成功才有归因价值。若机器允许并行,顺序无所谓。

**新开 run,不要 mid-run 切臂**——切臂会改变梯度目标,step 前后不可比,污染唯一一条完整曲线
(与你们当年 mid-run 换 researcher 的判断同理)。

## 3. 首验清单(A3 第一个 step 就能查,请回报)

按重要性排序,**前三条不过就别看 loss 了,先查配置**:

| # | 查什么 | 期望 | 不符说明什么 |
|---|---|---|---|
| 1 | `rollout_extra/sft_role/mean` | **≈0.333**(=4/12) | 副本发号错了。查 `rollout.n` 是否真的是 12、`train_batch_size % agent_num_workers == 0` 是否仍成立 |
| 2 | `sft_won/mean ÷ sft_role/mean` | **0.9 量级** | <0.5 说明 expert rollout 本身有问题(doc 没注进去?抓 `runs/*/rollouts/step_*/` 里 `sample_role=="sft"` 的 dump 看轨迹) |
| 3 | `sft_dropped_ref/mean ÷ sft_won/mean` | 报数即可 | **这是引用过滤的误杀率,本地无真实数据可校准,我故意做宽了**。>30% 我们收紧词面;≈0 则考虑加强(检测对文档内容的隐式复述) |
| 4 | `sft/expert_tokens` | >0 | =0 说明没有任何专家样本进 loss,SFT 臂等于没开 |
| 5 | `sft/loss` | 有限、随训练下降 | NaN/inf 立即停并回报(有 clamp 保护,不该出现) |
| 6 | `critic/score/mean` | **不因开臂而跳变** | 跳变说明 expert 行漏进了行为指标(隔离-3 失效),此时曲线不可信 |
| 7 | `actor/pg_loss`、advantage 分布、clip 率 | 有值 | **低成功率早期组内全败 ⇒ advantage=0 ⇒ RL 信号稀疏,这是预期不是 bug** |
| 8 | 显微镜 `mask_mismatch` | 仍为 **0** | 非 0 说明隔离-4 失效,分析链路会读到脏数据 |

## 4. 会变化的运行形态(别当成故障)

- **轨迹数 128 → 192/step**(SFT 臂开时),rollout 墙钟相应上升。
  expert 副本**必须等 doc 就绪才能开跑**(不像 bare 副本能与 research 重叠),
  这是新增的串行段——**若 step 时间超预期,这是第一嫌疑**,回报后我们评估流水化。
- **打分算力不增**:expert 行在 fsdp 后端跳过打分前向。
  (vLLM 回退后端会给它们打一次无用的分,浪费但正确——回退路径零风险优先。)
- **显存**:多 4/12 的行参与前向;峰值仍由**单条最长序列**决定,不变。
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
| loss 尺度与历史 run 对不上 | 只在开 RL 臂时才可能——我们已把 OPD 项归一化钉死在纯 OPD 口径,若仍不符请回报 | `MIX_RL=0.0` 复现 |
| 想立刻回到已知良好状态 | — | `MIX_RL=0.0 MIX_SFT=0.0` 重启 = A0 = 与历史 run 同口径 |

## 6. 分析链路的适配(已就位,无需你们改)

- **蒸馏显微镜**:expert 行不落盘(它们没有蒸馏),所以地形/token 显微镜不受污染;
  `collect` 也不索引 `sample_role=="sft"` 的 dump(避免哈希碰撞抢走真实行的 provenance)。
- **dump 新增两个字段**:`sample_role`("bare"/"sft")与 `sft_accepted`,
  你们做自定义分析时可据此分层。
- **probe 工具链**不受影响(它读 checkpoint,与 batch 组成无关)。
