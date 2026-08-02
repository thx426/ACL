# Teacher Grounding(tg)T1 运行手册(v3,2026-08-02,正式跑单版)

> 面向 CD-006。机制设计真源 `docs/teacher-grounding-design.md`(v2,含解耦
> §5、二轮实施审阅台账 §11.5 与双盲台账 §12);本手册只讲怎么跑、怎么看、
> 怎么停、报回什么。代码:trace_opd 分支 **`mixed-objective-r2`**(HEAD
> `ec4f66f`),vendored verl = 真源 trace-opd **`d6ac87e`** 导出;
> **301 测试全绿**。
> v3 相对 v2(用户确认的运行形态定稿):①**不设独立 SMOKE run——起跑核验
> 并入正式跑头 2 step,fail-fast**;②`test_freq: 5`、30 步延长规则与判读
> 阈值全部预注册(§6b);③段分解升双镜像模式,tg 行观察包已补全;④起跑前
> gate 四件(§3b)。

## 0. 同步安全(老纪律,逐条执行)

1. tar-over-ssh 推送 `mixed-objective-r2`(GitHub 不通;macOS rsync 段错误);
   同步前备份现场(`backup_presync_<date>/`),同步后全量 md5 清单比对。
2. 服务器侧 `python -m pytest tests/ -q` 须 **301 passed**(298/289/278 均为
   旧代码——数字不对说明同步不完整)。
3. 生产改造保护:同步后 `git status` 核对,勿 stash、勿覆盖你们的工作件。
4. 一切以 **effective_config.yaml(run 目录落盘)** 为真值;启动前
   `config_tool diff` 过一遍。

## 1. 机制速览(30 秒版;详见设计 v2)

won 分割 + 频率加权(N = 批内 lost 行 + tg 行):

```
L = (N_lost/N)·mean_lost[ OPD(rkl_topk) ] + w_tg·(N_tg/N)·mean_won[ tg soft-CE ]
```

- **失败轨迹**照旧 OPD(doc 教学生);**成功轨迹转 tg 行**(学生教教师):
  训练 prompt 换 doc 前缀(前向即 q),标签 = 裸前缀打分 top-64 重归一——教
  q "用学生的健康说话方式,说出 doc 才知道的内容"。
- **解耦(0a4bb30)**:tg 标签走独立 `tg_label_ids/logprobs` 通道、独立
  `compute_tg_soft_ce` 公式(无任何 OPD clamp)——OPD 散度可自由取
  reverse_kl_topk,故 T1 直接叠既有 rkl 基线,单变量干净。
- 数据流镜像对称(OPD 行:打分带 doc/训练裸;tg 行:打分裸/训练带 doc),
  **总算力与纯 OPD 持平**。
- 行为曲线口径:tg 行是裸策略真实赢局,照常计入 critic/score/mean;只从训练
  分母/OPD 诊断/显微镜里隐身。

## 2. 实验身份与单变量链条

| 臂 | run.name 建议 | 相对基线的 diff | 作用 |
|---|---|---|---|
| (既有)rkl 纯 OPD | trace_opd_4b_rkl_seqmean | — | 历史锚(已崩;won 膨胀实证同族) |
| **T1** | trace_opd_4b_rkl_seqmean_t1 | 仅加 `mix_tg_weight: 1.0` + `opd_lost_only: true`(+ 环境变量 `TRACE_OBS_DIR` 开观察包,见 §5) | 主验证臂;**必须 step-0 开臂**(中途接入会把已膨胀成功形态固化进教师,设计 §4 防线 1) |

- 口径:16×8、同 caps、`save_freq: 2`(崩塌窗密存档)、**`test_freq: 5`**
  (val 纯评测不改权重,加密只花墙钟;与基线在共同点 10/20/30 照常对比)、
  **≥30 步**;**预注册延长规则(一次性)**:step-30 未触发判停且 val 未
  衰退 ⇒ 延至 **step-40** 终评,不再继续延。
- `mix_tg_health_gate` **保持 null**(默认关,单变量纪律;启用与阈值由标签
  质量曲线走势定,须报回裁定)。
- **T1′(仅 `opd_lost_only: true`,归因臂)在 T1 出结果后默认排跑**(预算
  许可下):T1 正向时分离"切成功区 OPD"与"加 tg"两成分的贡献,T1 失败时
  定位失败成分。T3(标签 one-hot CE 退化版)列 T2 池按 T1 结果定。
- GPU:等 162 空窗(用户裁定);沿用与既有基线可比的卡数配置。

## 3. 配置制备(diff 规范)

从你们站点母本 `configs/experiments/162-4b-rkl-seqmean.yaml` 复制一份:

- **T1**:改 `run.name` + 加两行 `mix_tg_weight: 1.0`、`opd_lost_only: true`,
  其余逐字节同。母本无 tg 三键 ⇒ `config_tool diff` 报缺
  `mix_tg_weight`/`opd_lost_only`/`mix_tg_health_gate`(代码默认 0/false/null
  =关断,**预期**,记录即可)。
- config 校验会强制:opd_kl_mode ∈ {forward_kl_topk, **reverse_kl_topk**}(白
  名单,rkl 底座合法)/ teacher_scoring_backend=fsdp / research.enabled /
  loss_agg_mode=seq-mean-token-mean / 与 rl/sft 互斥——任一不满足启动即拒。
- 环境变量:启动脚本导出 `TRACE_OBS_DIR=<run>/obs`(两条段分解判据的数据
  源;`TRACE_OBS_EVERY` 默认 1 = **每步全存**,保持默认——相变窗覆盖靠它)。
  **忘导出 = 环1/环2 机理判据整条数据不存在,不可事后补。**

## 3b. 起跑前 gate(四件,分钟级)

1. **同步校验**:服务器侧 `python -m pytest tests/ -q` 须 **301 passed**。
2. **基线 won 分层补算**:`python3 scripts/won_traj_evolution.py --rollouts
   runs/trace_opd_4b_rkl_seqmean/rollouts` ——锐判据的**同臂配对数**(现有
   ×6 实证表来自 rkl+RL 臂,纯 rkl 臂必须有自己的对照表);产出随首次报回。
3. **续跑预检**:读 launcher 代码确认 T1 崩溃后可从 ckpt resume(不信文档
   承诺,代码级确认)。
4. **磁盘余量**:obs 全存 40 步约 12–19GB + `save_freq: 2` 的 ckpt 体量。

## 4. 起跑核验(并入正式跑头 2 step,fail-fast——不设独立 SMOKE run)

正式跑起跑后,前 1–2 个 step 内就地核验八项;**任一不过立即停跑**、保留
现场与日志原样报回;全过即继续,该跑就是正式 run:

1. **参数实收**:`effective_config.yaml` 两键正确;启动日志 build_command 含
   `+distillation.distillation_loss.tg_loss_coef=1.0`;`launch_manifest` 的
   objective fingerprint 相对基线变化(两键条件式进 hash)。
2. **分流活着**:出现 won 轨迹的 step,`rollout_extra/tg_role/mean > 0`、
   `tg/rows > 0`、`tg/loss` 有值;`tg_dropped_overflow/nodoc` 记损耗(少量
   正常;`tg_dropped_unhealthy` 必须恒 0——门是关的,非零=配置错)。
3. **打分后端**:日志有 `teacher_topk_fsdp ACTIVE` 探针;无任何
   `teacher_mode ... provenance` / `'tg' but ... missing` / `tg_label_ids
   ... missing` fail-loud 文本。
4. **P0-2 不变量(解耦后仍监控)**:`tg/loss_min_preclamp ≥ -1e-6`(独立
   soft-CE 数学非负;显著负值 = 公式破,停下报机制侧)。
5. **标签质量曲线活着**:`tg_label_thought_cpt`/`_chars`/`_turns` 三键有值;
   **判读口径 = mean(tg_label_thought_cpt) ÷ mean(tg_role)**(标签行的逐轨迹
   均值,与离线 won_traj_evolution 同 estimand)——应落健康区(基线 step5-7
   实测 284–315)。chars÷turns 是 turn 加权总量视图,只作辅助,勿与 284–315
   直比。
6. **行为曲线口径**:`critic/score/mean` ≈ 该 step rollout 全量胜率(won 行
   分流后仍计入——对不上说明 P0-1 口径破了)。
7. **算力持平**:step 耗时/显存与基线同量级(镜像对称;偏差 >30% 报回)。
8. **观察包落盘**:`<run>/obs/step_*/rank*.npz` 出现;头 2 step 内各跑一次
   `python3 scripts/opd_kl_span_decomp.py --obs-dir <run>/obs --tokenizer
   <Qwen3-4B 路径>`(`--role bare` 与 `--role tg`)出表即通(`part` 列应
   恒 0)。

## 5. 全量监控(在跑期)

- 主健康:既有组合签名照旧(score/entropy/resp_len/重复率)。
- tg 专属:`tg/loss`(应下降收敛)、`tg/rows` 与 `tg_role`(随胜率上升)、
  `tg/loss_min_preclamp`(恒 ≥ -1e-6)、`tg/loss_max`(大值合法=q 支持集外
  质量,应随训练回落)、dropped 三计数。
- **枢纽变量(标签质量)**:cpt 判读 = mean(tg_label_thought_cpt)÷mean(tg_role)
  ——若相对健康期(284–315)持续上行 >2×,报回并附曲线,等待健康门启用裁定
  (阈值锚健康期,严禁滚动上调)。
- **预注册主判据(病的终点)**:response length / p95 / resp_clip——step-9
  相变是否推迟/压平;崩塌步 > 基线×1.5 或预算内不崩 ⇒ 正向。
- **won 分层锐判据**:`python3 scripts/won_traj_evolution.py --rollouts
  <run>/rollouts`——基线 thought/turn ×6(315→1986);T1 预测平坦/显著压缓。
  跑完(或崩后)T1 与基线各出一份表报回。
- **机理判据(段分解,双模式)**:`scripts/opd_kl_span_decomp.py`
  ①`--role bare`:lost 区 OPD-KL thought 份额随步——T1 预测下降/持平;
  ②`--role tg`:tg 行 soft-CE 的 thought 份额随步——预测初期高、随训练
  收敛下降(教师叙述被纠回健康形态的逐 token 证据)。注:基线 run 未开
  obs,无对照数据,两个模式均看 **T1 自身趋势**;若你们重跑基线可顺手开
  obs 补对照(非必须)。
- 效果判据:裸 val 三件套(峰值高度/时刻/峰后形态,±5pp 误差棒)。

## 6. 判停表

| 信号 | 判定 | 动作 |
|---|---|---|
| 既有组合签名触发(长度钉顶+熵尖峰+score 归零) | 崩塌 | 停;存峰值 ckpt;对照点已达即有效读数 |
| `tg/loss_min_preclamp < -0.01` 持续 | 独立公式不变量破 | 停,报机制侧 |
| won 率 >0 而 `tg/rows` 连续多 step 为 0 | 分流断裂 | 停,查 doc 生成与 overflow 计数 |
| `tg_dropped_unhealthy > 0`(门关着) | 配置错 | 停,查 effective_config |
| `critic/score/mean` 与 rollout 胜率背离 >5pp | 口径破裂 | 停,报机制侧 |
| 标签 cpt 持续 >2× 健康期 | ratchet 前兆 | 记录继续 + 报回(健康门启用裁定) |
| teacher 自走成功率(K=8 doc 注入)骤降 | q 执行功能受损 | 记录继续,报回(P3 溢出候选) |

## 6b. 预注册判读与结果分型(2026-08-02 定稿;终评时对号入座,防事后合理化)

阈值(预注册,基线 = `trace_opd_4b_rkl_seqmean`,崩于 ~step-19):

- **主判据**:崩塌步 ≥ 基线×1.5(≈29)或预算内不崩 ⇒ 正向;
- **锐判据(cpt,`tg_label_thought_cpt` 口径,健康期 284–315)**:全程 <2×
  (~630)= 压平;2–4× = 压缓;>4× 且持续上行 = 失败;
- **效果判据**:val 峰值 ≥ 基线峰值 −5pp(greedy 非确定性误差棒内不劣)且
  峰后平台化(不出现基线式不可逆崩形态);
- **机理判据**:段分解两模式的 thought 份额趋势(T1 自身趋势,无基线对照)。

结果四型(终评归入其一,连同证据链报回):

| 型 | 观测 | 解读 → 下一步 |
|---|---|---|
| A 强正向 | 不崩 + cpt 压平 + val 不劣且平台 | 机制成立 → T1′ 归因 |
| B 正向 | 崩塌显著推迟 + cpt 压缓 | 方向对、力度不足 → 看环2 段分解定加强点(健康门启用走裁定) |
| C 中介成立终点未达 | cpt 压平但 val 无改善 | 教师治了、学生未受益 → 查环2/环3 断点(段分解 + val 截断率口径) |
| D 失败 | 同窗崩 + cpt 同样飙升 | 恶性分支抢跑 → 健康门(阈值锚 ~630,严禁滚动上调)优先,余走 T2 池裁定 |

D 型的预埋提醒:tg 份额 = won 率(频率加权内生),早期 won 率低则推力弱——
若曲线显示"tg 未起力病已发",按上行处置,勿改 w_tg(裁定值)。

## 7. 报回清单(experiment-feedback.md)

1. run 目录 + effective_config + fingerprint;与基线互 diff 确认单变量。
2. 长度三曲线(mean/p95/resp_clip)对基线对照 + 崩塌步(如崩)。
3. `won_traj_evolution` 两份表:T1 输出 + **基线补算表(gate 2 产出,纯 rkl
   臂同口径对照)**(json 一并留档)。
4. 标签质量曲线(chars/turns/cpt)全程截图或数值导出。
5. `opd_kl_span_decomp` 两模式表 + json(`--role bare` 环2 与 `--role tg`
   环1 的 thought 份额随步)。
6. 裸 val 三件套逐 val 点。
7. tg 曲线族数值(loss/rows/role/dropped×3/min_preclamp/loss_max)。
8. 起跑核验八项逐项结果;任何 fail-loud 文本原样。
9. **终评归型**:§6b 四型对号入座 + 逐条判据的数值证据链(数字必须带
   可追溯路径)。
