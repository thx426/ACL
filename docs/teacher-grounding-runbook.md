# Teacher Grounding(tg)T0/T1 运行手册(v1,2026-08-02)

> 面向 CD-006。机制设计真源 `docs/teacher-grounding-design.md`(含 codex 深审
> 处置台账 §10);本手册只讲怎么跑、怎么看、怎么停、报回什么。
> 代码:trace_opd 分支 **`mixed-objective-r2`** HEAD `4592873`,vendored verl
> = 真源 trace-opd `01a9a06` 导出;**278 测试全绿**。

## 0. 同步安全(老纪律,逐条执行)

1. tar-over-ssh 推送 `mixed-objective-r2`(GitHub 不通;macOS rsync 段错误);
   同步前备份现场(`backup_presync_<date>/`),同步后全量 md5 清单比对。
2. 服务器侧 `python -m pytest tests/ -q` 须 **278 passed**(b035732 时代是
   251——数字不对说明同步不完整)。
3. 生产改造保护:同步后 `git status` 核对,勿 stash、勿覆盖你们的工作件。
4. 本手册的一切以 **effective_config.yaml(run 目录内落盘)** 为真值;配置
   副本与模板 diff(`config_tool diff`)启动前跑一遍。

## 1. 机制速览(30 秒版;详见设计文档)

won 分割 + 频率加权(N = 批内 lost 行 + tg 行):

```
L = (N_lost/N)·mean_lost[ OPD(forward_kl_topk) ] + w_tg·(N_tg/N)·mean_won[ tg soft-CE ]
```

- **失败轨迹**照旧 OPD(doc 教学生);**成功轨迹转换为 tg 行**:训练 prompt 换
  doc 前缀(前向即 teacher q)、标签 = 裸前缀打分的 top-64 重归一分布——教
  q "在学生状态上给出可执行指导",治叙述通胀病根(won 轨迹实证:基线上
  thought/turn ×6)。
- 数据流与纯 OPD **镜像对称**(OPD 行:打分带 doc/训练裸;tg 行:打分裸/训练
  带 doc),**每行仍一次打分+一次训练前向,总算力与纯 OPD 持平**。
- won 率上升 ⇒ OPD 份额真退火、tg 份额增长(频率加权);lost→0 连续归零。
- 行为曲线口径:**tg 行是裸策略真实赢局,照常计入 critic/score/mean**(与
  rollout 全量胜率一致);它们只从训练分母/诊断里隐身(response_mask 零)。

## 2. 实验身份与单变量链条

| 臂 | run.name 建议 | 相对上一环的 diff | 作用 |
|---|---|---|---|
| (既有)rkl 纯 OPD | trace_opd_4b_rkl_seqmean | — | 已崩 step-19,历史锚 |
| **T0** | trace_opd_4b_fkl_seqmean_t0 | 仅 `opd_kl_mode: reverse_kl_topk → forward_kl_topk` | tg 绑定 forward 公式 ⇒ 配对基线必须 forward;顺带拿到 kl 方向单变量读数 |
| **T1** | trace_opd_4b_fkl_seqmean_t1 | 仅加 `mix_tg_weight: 1.0` + `opd_lost_only: true` | 主验证臂;**必须 step-0 起开臂**(中途接入会把已膨胀的成功形态固化进 teacher,设计 §1 时机约束) |

- 口径:16×8、同 caps、`save_freq: 2`(崩塌窗密存档)、`test_freq: 10`、
  **≥30 步**(T0 预期在 forward 语境重现 step-9 相变/±数步崩塌;T1 观察延后)。
- T1′(仅 `opd_lost_only: true`,归因臂)按 T1 结果决定,不首发。
- GPU:等 162 空窗(用户裁定);沿用 4 卡配置保持与既有基线可比。

## 3. 配置制备(diff 规范)

从你们站点母本 `configs/experiments/162-4b-rkl-seqmean.yaml` 复制两份:

- **T0**:改 `run.name` + `train.opd_kl_mode: forward_kl_topk`,其余逐字节同。
  注意母本无 tg 两键 ⇒ `config_tool diff` 会报缺 `mix_tg_weight`/
  `opd_lost_only`(用代码默认 0/false=关断,**预期**,记录即可)。
- **T1**:在 T0 副本上加两行 `mix_tg_weight: 1.0`、`opd_lost_only: true`。
- 启动前核验(fail-loud 都已内建,这里是预检):
  `python scripts/config_tool.py diff configs/experiments/<file>` +
  两臂互 diff 应恰为上表所列键。T1 的 config 校验会强制:
  forward_kl_topk / teacher_scoring_backend=fsdp / research.enabled /
  loss_agg_mode=seq-mean-token-mean / 与 rl/sft 互斥——任一不满足启动即拒。

## 4. 冒烟(全量前必做,SMOKE 快速档数 step)

七项首验(T1 冒烟;T0 冒烟只看 1/6/7):

1. **参数实收**:run 目录 `effective_config.yaml` 两键正确;启动日志
   build_command 含 `+distillation.distillation_loss.tg_loss_coef=1.0`;
   `launch_manifest` 的 objective fingerprint 相对 T0 变化(两键条件式进 hash)。
2. **分流活着**:出现 won 轨迹的 step,曲线 `rollout_extra/tg_role/mean > 0`
   (won 转换率),`tg/rows > 0`、`tg/loss` 有值;`tg_dropped_overflow/nodoc`
   记录损耗(少量正常,持续高比例→查 doc 生成/长度)。
3. **打分后端**:日志有 `teacher_topk_fsdp ACTIVE` 探针;无任何
   `teacher_mode ... provenance` / `'tg' but ... missing` fail-loud。
4. **P0-2 不变量**:`tg/loss_min_preclamp ≥ -1e-6`(soft-CE 非负;显著负值
   = clamp 豁免失效,停下报机制侧)。
5. **行为曲线口径**:`critic/score/mean` ≈ 该 step rollout 全量胜率(won 行
   分流后仍计入——对不上说明 P0-1 口径破了)。
6. **算力持平**:step 耗时/显存与 T0 同量级(镜像对称设计;偏差 >30% 报回)。
7. **零开销对照**:T0 曲线**不得**出现任何 tg/* 键(关断即零传输)。

## 5. 全量监控(在跑期)

- 主健康:既有组合签名照旧(score/entropy/resp_len/重复率)。
- tg 专属:`tg/loss`(应下降收敛)、`tg/rows` 与 `tg_role`(随胜率上升)、
  `tg/loss_min_preclamp`(恒 ≥ -1e-6)、`tg/loss_max`(大值合法=q 支持集外
  质量,应随训练回落)。
- **预注册主判据(病的终点)**:response length / p95 / resp_clip——T1 的
  step-9 相变是否推迟/压平;崩塌步数 > T0×1.5 或预算内不崩 ⇒ 正向。
- **won 分层锐判据**:`python3 scripts/won_traj_evolution.py --rollouts
  <run>/rollouts`——基线 thought/turn ×6(315→1986);T1 预测平坦/显著压缓。
  跑完(或崩后)两臂各出一份表报回。
- 效果判据:裸 val 三件套(峰值高度/时刻/峰后形态:平台化 vs 不可逆崩)。

## 6. 判停表

| 信号 | 判定 | 动作 |
|---|---|---|
| 既有组合签名触发(长度钉顶+熵尖峰+score 归零) | 崩塌 | 停;存峰值 ckpt;两臂对照点已达即为有效读数 |
| `tg/loss_min_preclamp < -0.01` 持续 | P0-2 豁免失效 | 停,报机制侧 |
| won 率 >0 而 `tg/rows` 连续多 step 为 0 | 分流断裂 | 停,查 doc 生成与 overflow 计数 |
| `critic/score/mean` 与 rollout 胜率背离 >5pp | 口径破裂 | 停,报机制侧 |
| teacher 自走成功率(K=8 doc 注入)骤降 | q 执行功能受损 | 记录继续,报回(P3 溢出候选证据) |

## 7. 报回清单(experiment-feedback.md)

1. 两臂 run 目录 + effective_config + fingerprint;互 diff 确认单变量。
2. 长度三曲线(mean/p95/resp_clip)两臂对照 + 崩塌步(如崩)。
3. `won_traj_evolution` 两臂输出表(json 一并留档)。
4. 裸 val 三件套逐 val 点。
5. tg 曲线族截图/数值(loss/rows/role/dropped/min_preclamp)。
6. 冒烟七项的逐项结果。
7. 任何 fail-loud 文本原样。
