# ssb_4b_selfres_engine 双降诊断(2026-08-11,CD-010)

**现象**:A800 run `ssb_4b_selfres_engine`(engine executor,K=64,asym,selfres,24 步)
train/val 分数随训练单调下降:val 31.74% → 25.35% → 24.11%(564 条,-7.6pp,超
val greedy ±5pp 误差棒)。远端源码 = 本地 `18752cd`(缺 codex 修复批 `684c6f4`)。

**结论一句话**:主因**不是** engine 执行层,而是机制层的三因子链——
**未标定熵门导致干预率失控(实测 0.25–0.37,设计目标 0.10)× SSB 答案密度型
selfres 文档 × asym advantage 对特权轨迹的单向正推(W_beh=off、cap=0、失败零惩罚)**
——三者复合等效于对"拿着答案快速提交"的行为做加权行为克隆;bare/val 语境下模型
只学到形态(秒交)学不到内容(答案),中间探索段产能被摧毁。P0-1(窗尾 EOS)实锤
在场且量化确认,但量级为次因(污染 token ≈ 训练量 0.11%)。

数据源:A800 `$R/runs/ssb_4b_selfres_engine/rollouts/step_*`(R=/fs/fast/u2021000902/
wushang/workspace/TRACE-OPD-ENGINE);对照 `TRACE-OPD-PGS/runs/ssb_9b_grpo_baseline`、
`TRACE-OPD-SSB/runs/ssb_4b_selfres_a800.ARCHIVED-20260809_2153`。统计口径见 §5。

---

## 1. 证据链(全部实测)

### 1.1 val 的退化形态是"行为塌缩到秒交",不是长度失控

| 读数 | val00 | val10 | val20 |
|---|---|---|---|
| won(名义) | 31.7% | 25.4% | 24.1% |
| won(未截断口径) | 35.7% | 26.3% | 24.7% |
| mean len(token) | 13549 | 6052 | 5907 |
| mean turns | 11.5 | 7.4 | 8.4 |
| **turns=2 行数** | **31** | **236** | **153** |
| turns=2 的 won | 64.5% | 34.7% | 32.7% |
| truncated | 63 | 20 | 13 |
| invalid_format | 85 | 30 | 26 |

- val00 的 turns=2 won 64.5% = 模型原本**只在有把握时**快交(校准的);
  val10 起变成无差别秒交(236 条,won 掉到 34.7%)。
- val00 里 4–13 turns 段贡献 121/179 个 won;val20 该段只剩 61——中等难度任务
  的探索产能被摧毁。turns=20(打满)双峰依旧(113 条,won ~2%)。
- 毒性注入在前 10 步内完成(val10 已塌缩);对应 α cosine(123 步)前 10 步 ≈0.97–1.0。

### 1.2 决定性对照:纯 GRPO 同任务在涨

`ssb_9b_grpo_baseline`(无 steering,同 SSB reward、同 20-turn 上限):
val00 41.2% → val10 **45.8%**(+4.6pp),且同样在变快(turns 10.3→8.7)。
⇒ "GRPO×SSB 必然退化"“推短本身有毒”两个假设**均被证伪**;bare LOO 的效率化
是良性学习。毒性特异于特权正推通道。(9B/4B 有模型差,方向性对照有效;
4B 纯 GRPO 对照不存在。)

### 1.3 正梯度构成:61.4% 来自特权轨迹,且推的正是"短平快"

用 trajectory.score(soft 分)精确计算,24 步累计:
- steered 正 excess 行(score>同任务 bare 均值,拿 `excess×α` 正推)= **524 行**;
- bare LOO 正行 = 330 行;⇒ **正梯度 61.4% 来自 steered 特权轨迹**;
- 正 excess 行的 mean turns 几乎每步比 bare 臂短 3–7 轮(如 s01 8.5 vs 13.4、
  s10 4.6 vs 9.3)——被克隆的正是"文档直达答案"形态;
- asym 负向为零(失败 steered 恒 0 advantage)+ W_beh=off + steered_cap=0
  ⇒ 该通道**只进不出**(单向棘轮)。
- train 侧行为塌缩与 val 同步:s10 steered 臂 33/64 条 ≤3 turns(s01 仅 6/64),
  bare 臂同步 20/64——policy 本体已转向。

### 1.4 干预率失控:未标定门 + 防线不在环上

| step | theta(mean/max) | open_rate | 干预 token/轨迹(count) | budget_spent(mean/max)* |
|---|---|---|---|---|
| s01 | 1.12 / 1.28 | 0.338 | 2529 | 3559 / 17568 |
| s08 | 1.22 / 1.46 | 0.365 | 2827 | 4013 / 20621 |
| s10 | 1.26 / 1.51 | 0.349 | 922 | 1209 / 10546 |
| s24 | 1.15 / 1.57 | 0.248 | 818 | 1061 / 6849 |

- `entropy_theta_init: 1.28` 是 **ALFWorld-4B 占位值**(config 内自带 ⚠ 注释
  "step-1 后须按实测重标"——未执行);tracker EMA(β=0.1/步)是活的、方向正确,
  但 24 步才从 1.28 爬到 ~1.57,**关键的前 10 步门率一直 0.30–0.37**
  (目标 0.10 的 3 倍+,预注册上界 0.25 之上);
- 每条 steered 轨迹 800–3600 个 token 被倾斜、δ 累计数千 nats ⇒ steered 轨迹
  深度 off-policy(接近教师行为)——v3.3 "干预稀疏所以可省 IS 校正"的前提
  在本语境不成立;
- **防线缺口(实现层)**:spec 预注册的 [5%,25%] 超限告警降级,实现里
  `rate_bounds` 只在 `gate_mode=="adaptive"` 时传入(agent_loop.py:1042-1043);
  v3.3 熵门(`gate_mode: entropy`)**不在该防线环上**,超限无告警无降级。
- *budget 为纯记账(用户裁定无预算墙),且该字段受 P0-3 影响(零扣费 turn
  会清零总账),表中数值按近似/下界读。

### 1.5 P0-1 实锤、量化,判为次因

- steered 臂 mid-EOS 行率 3.1–23.4%(均 ~9%),turn 级 ~0.5–1.6%/turn——与
  codex 预估 1.6%/turn@K64 吻合;**bare 臂与 val 恒为 0**(指纹特异性:span 中段
  出现 stop token 248046 只能来自窗尾越过);
- 24 步累计垃圾 token 33,935 ≈ 训练 token 量 **0.11%**,且无逐步增长趋势
  (无自我强化失控);
- 其中 40 行"污染且 excess>0"被正推(占正推行 7.6%)——确认的毒性注入,但
  解释不了 -7.6pp;
- 判定:**必修的正确性红线(684c6f4 已修),但非本次退化主因**。

### 1.6 其余排除项

- P0-2(seeded×K):effective_config 无 seed/full_determinism,未触发;
- P0-3:仅账面(budget 读数失真),不改采样;
- research 行(64 行/步,占训练 1/3):r_raw 0.19–0.59 波动无趋势、doc_words
  530–670 稳定、degraded/error 个位数——selfres 反馈回路未崩(s13/s20 steered
  won 倒挂更像小样本噪声 + doc lift 波动);
- val 测量:564 条三点单调 + 未截断口径更陡(-11pp),真实退化;
- python 档 executor 对照:ARCHIVED run 仅 2 个 train step(val0 29.4%,与
  engine val0 31.7% 同水位)——**engine 执行层无法被数据完全洗清**,但主因
  证据链(§1.2–1.4)不依赖该对照。

## 2. 因果链(定案)

```
theta 未标定 + EMA 慢 + 防线不在环 ──> 门率 0.25–0.37(3× 设计)
        │                                        │
        ▼                                        ▼
SSB selfres doc 答案密度高 ──────> steered 轨迹深度 off-policy:
(r_raw 0.2–0.6 有真 lift)          短 3–7 轮、快提交、高 soft 分
                                             │
              asym: clamp_min(r−bare_mean,0)×α,α≈0.91–1.0
              W_beh=off、cap=0、失败零惩罚(单向棘轮)
                                             │
                                             ▼
              61.4% 正梯度 = 特权轨迹加权行为克隆("拿着答案快交")
                                             │
              信息状态不一致(bare 无 doc,行为分布不可达)
                                             ▼
              val/bare:无差别秒交(turns=2 31→236)、探索段产能摧毁
                                             ▼
                       val 31.7→24.1(未截断口径 35.7→24.7)
```

## 3. 流程层事实(决定本 run 读数的法律地位)

1. 远端源码 = `18752cd`(13:17 同步的工作树快照),**非** changelog 指定部署
   sha `684c6f4`(14:30 提交);起跑 13:46 早于部署通知;
2. runbook 红线 "`executor: engine` 在 V1 对拍 PASS 前禁止进正式 run"——V1 未执行;
3. config ⚠ "entropy_theta step-1 后须按实测重标"——未执行,连跑 24 步。

⇒ 本 run **不能作为机制有效性判决依据**;但诊断价值已榨出。
**重要**:即使三条全合规重跑同配置,大概率仍降——主因在机制层参数与
advantage 结构,不在部署 sha。

## 4. 处置建议(待用户裁定,均未实施)

- **立即**:停当前 run(24 步诊断价值已尽,继续烧卡无意义);dump/ckpt 保留。
- **率控修复(必做级,机制侧)**:
  a. theta 起点按 SSB 实测标定(冻结校准批,或首步标定后再放开训练);
  b. 把 rate_bounds 告警降级接到熵门(防线缺口修补);
  c. 视需要加快 EMA 或提高 min_obs 前的保守性。
- **off-policy 防线(核心探讨点,均进指纹、需裁定)**:
  A. `w_beh_mode` off→shaped/full(现成通道零新代码,对特权正推按 π/b 降权);
  B. `steered_adv_cap` 开启(限总量,治标);
  C. α0 降低 / decay 提前(治标);
  D. 结构候选:干预位/自由位分治 advantage(干预 token 不作克隆目标)、
     doc 模板降答案密度(动 selfres 侧)。
- **engine 侧不变**:`684c6f4` 部署 + V0/V1 对拍红线照走;V2 吞吐读数与机制
  修复正交(用修复后的机制配置跑)。

## 5. 统计口径备注

- mid-EOS 指纹:`turn_token_spans` 各 span 内 stop token(248046)出现在非末位;
  正常单发生成在 stop 即停,该指纹仅 P0-1 窗尾越过可产生(EOS 被拒绝重采的
  设计内路径不落 stop token,无假阳性);
- 正梯度构成:excess = trajectory.score − 同任务 bare 臂 score 均值(**soft
  口径近似**;后续核实训练 reward 实为二值 won——`agent_loop.py`
  `reward_score = 1.0 if traj.won else 0.0`,soft score 仅进 dump 审计。
  won 口径重算 ≈546 正推行,与 soft 口径 524 同量级,§1.3 的构成结论不变;
  修正记录见 `docs/notes/2026-08-11-mentor-dosage-comparison.md` §2);
  bare LOO 正行 = score > 组内留一均值;
- 未截断口径 won = won / (n − truncated);
- 所有逐步表:steered=steer_arm==1,bare=steer_arm==0,val 目录全体。
