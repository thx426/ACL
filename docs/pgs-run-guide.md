# PGS 实验运行指导(CD-006 执行手册)

CD-010,2026-08-04。配套阅读:`pgs-mechanism-explainer.md`(机制在干什么)、
`pgs-validation-design.md`(臂/判据/功效,**预注册,开跑前读一遍**)、
`pgs-observability-design.md`(观测体系)。本文只管"怎么跑"。

## 0. 前提与环境(162)

- 代码 = worktree `trace_opd-pgs`,分支 `pgs-sps`(**不要**用主树/其他 worktree 的代码)。
  同步到 162:tar-over-ssh,**独立父目录**(现行部署
  `/data3/wushang/workspace/TRACE-OPD-PGS/trace_opd/`,入口
  `env_pgs.sh`/`run_pgs.sh`);launcher 已改**自定位**(commit 后目录任意命名
  均正确),但沿用现行部署即可。**勿覆盖** `TRACE-OPD` 树(CD-008/9/5)。
- **LD_PRELOAD 是 162 启动必要条件**(非可选):
  `export LD_PRELOAD=/data3/wushang/libcxx_x/un/lib/libstdc++.so.6.0.34`
  ——vLLM worker 无条件 import flash_attn(即使 PGS 用 sdpa),系统 libstdc++
  只到 GLIBCXX_3.4.28 会当场 ImportError(env_pgs.sh 已含)。
- venv:`venv_train`(162 训练环境;sp=2 档要求 flash-attn——LD_PRELOAD 后 2.8.3 实测可用)。
- vendored verl 就在 worktree 内(`trace_opd-pgs/verl`),launcher 自动
  prepend 到 PYTHONPATH——**不需要**另装/另切 verl。
- 环境变量(launcher 经 yaml `${VAR}` 消费):
  ```bash
  export TRACE_MODEL_PATH=/data3/.../trace_opd-mixob/runs/sft_4b_coldstart_v1/hf/step_6
  export TRACE_TRAIN_FILE=/data3/.../trace_opd-pgc/data/rl_train_v2.parquet
  export TRACE_VAL_SEEN_FILE=/data3/.../trace_opd/data/out/valid_seen.parquet
  export TRACE_VAL_UNSEEN_FILE=/data3/.../trace_opd/data/out/valid_unseen.parquet
  export ALFWORLD_DATA=...; export ALFWORLD_WORKER_PYTHON=...
  export SWANLAB_API_KEY=...        # 缺省自动降级 offline
  ```
- GPU 冲突纪律:起跑前确认 PGC r2/C3 已结束、四卡空闲(`nvidia-smi` 功耗判据)。

## 1. 运行序列(五步,顺序执行)

### 步骤 ① 底座冒烟(steering off,验证纯 GRPO 通路)

```bash
python trace_opd/scripts/launch_train.py \
  --config trace_opd/configs/experiments/162-pgs-base.yaml \
  --run-name pgs_smoke_base trainer.total_training_steps=2 trainer.val_before_train=False
```
预期:2 step 正常完成;swanlab 有 `critic/score/mean`(即 won)与 entropy 曲线;
rollouts/ 下有 dump。**任何 import/字段错误在这里暴露,别带病进下一步。**

### 步骤 ② SPS 引擎冒烟(P4 六项,一次 1-step run 全验)

```bash
python trace_opd/scripts/launch_train.py \
  --config trace_opd/configs/experiments/162-pgs-calib.yaml \
  --run-name pgs_smoke_sps trainer.total_training_steps=1 trainer.val_before_train=False
```
核查单(全过才继续;任何一条不过 → 记录现象反馈 CD-010,勿自行改机制代码):
1. 启动日志无 `max_logprobs` 相关报错(launcher 已联动 vLLM `max_logprobs=256`);
2. run 正常出 dump 且 `steering.steer_n_candidates > 0`(verify 通路活着);
3. **prefix cache × prompt_logprobs**(头号风险):若 verify 请求报错或
   `prompt_logprobs` 空 → 现象反馈(预案:教师路由禁用 prefix cache,由 CD-010 出 patch);
4. dump 里 `steer_d_count`、`steer_ell_count` 均 >0(标定数据面活着);
5. 墙钟:对比步骤①的单 step 时长,记录 SPS 开销倍数(预期 <2×;>3× 反馈);
6. `python trace_opd/scripts/pgs_report.py --run-dir <run>` 出报告且**不变量违例=0**。

### 步骤 ③ 标定 run(3 step)→ 定 θ_D 与 c

```bash
python trace_opd/scripts/launch_train.py \
  --config trace_opd/configs/experiments/162-pgs-calib.yaml \
  --run-name pgs_calib trainer.total_training_steps=3 trainer.val_before_train=False
python trace_opd/scripts/calibrate_theta_d.py --rollouts <run_dir>/rollouts
```
输出两组标定量:
- `suggested steering.theta_d`:**率控自适应下语义 = EMA 初值**(取 P(1−q),
  q=gate_target_rate,即 `--q 0.90` 对应 q=10%)→ 填入 s1 配置;训练中 θ_D
  每 step 自动追踪目标门率(swanlab `steer_theta_d` 曲线可审计);
- **候选 c 饱和率表**(P(|ℓ|>c)):选 c 使饱和率落 10–25% 区间(spec c∈[2,3] 内取;
  若默认 2.5 的饱和率已在区间,不动)。若脚本 FATAL(分位落溢出桶)→ 反馈。

### 步骤 ④ 回填 s1 配置

编辑 `configs/experiments/162-pgs-s1.yaml`:
- `theta_d: <标定值>`(占位 0 会被启动校验拒绝——防呆是故意的);
- 若 c 改了:同步改 `c:` 并**重跑一次步骤③**(D=Var[ℓ̃] 依赖 c,θ_D 须在同一 c 下标定)。

### 步骤 ⑤ 正式对照(base 先行或并行按卡量)

```bash
# 对照臂
python trace_opd/scripts/launch_train.py \
  --config trace_opd/configs/experiments/162-pgs-base.yaml --run-name pgs_r1_base \
  trainer.total_training_steps=100 trainer.test_freq=10 trainer.save_freq=10
# 处理臂(标定回填后)
python trace_opd/scripts/launch_train.py \
  --config trace_opd/configs/experiments/162-pgs-s1.yaml --run-name pgs_r1_s1 \
  trainer.total_training_steps=100 trainer.test_freq=10 trainer.save_freq=10
```
**s1 首查项(起跑后第一个有 steer 的 step)**:swanlab
`actor/rollout_is_weight_off1_frac` 必须 >0 —— 这是 sp=2/rmpad 路径下 W_beh
真实抵达 loss 的运行时终验(静态核查见 impl-design §8);=0 立即停查。
长跑纪律照旧:setsid/nohup + 新连接确认进程存活;进度以文件系统产物为准。
注:`trainer.*` 键属 verl 原生训练调度(非优化目标),launcher 的 passthrough
守卫放行;**`algorithm.*` 等 objective 键会被 launcher 拒绝**——那类改动必须
走 `--set`(进指纹)。

## 2. 监控(跑起来之后盯什么)

**swanlab 实时(每天至少看两次)**:

| 曲线 | 健康 | 异常动作 |
|---|---|---|
| `actor/rollout_is_weight_off1_frac` | s1 有 steer 的 step >0 | =0 ⇒ **W 静默丢失,立即停**反馈 |
| 门率(`steer_n_base_open`/`steer_n_candidates`) | [5%,25%] | 滑出 ⇒ 记录,连续 10 step 出带 ⇒ 反馈 |
| `steer_count` 均值 | ~4-8/轨迹(B_∞=20) | 恒 0 或暴涨 ⇒ 反馈 |
| `steer_gate_alarm` | 0 | >0 持续 ⇒ 反馈 |
| entropy | 不持续跌破 0.2 | 触发安全门(§3) |
| val won(seen/unseen) | 臂间对比 | 单点跌 >10pp ⇒ 安全门 |
| 响应长度/截断率 | 平稳 | 长度失控是既有崩塌前兆 |

**日常离线(每天一次)**:
```bash
python trace_opd/scripts/pgs_report.py --run-dir <run> --tokenizer $TRACE_MODEL_PATH
```
看:红绿灯全绿、重采对表语义(教师改的 token 合不合理——最直觉的机制检查)、
thought 重复率(>55% 警戒)、预算耗尽位置(全堆前 20% = 早花晚饿,反馈)。

**个案抽查(异常时)**:`pgs_traj.py <dump.json> --tokenizer ...`。

## 3. 安全门(预注册,任一触发 ⇒ 停机保 ckpt + 反馈)

thought 逐字重复率 >55% / entropy 持续 <0.2 / val 单点跌 >10pp / 不变量违例 >0。

## 4. 配置键速查(steering 段;完整注释见 configs/pgs.yaml)

| 键 | 默认 | 含义 | 什么时候动 |
|---|---|---|---|
| `enabled` | false | 总开关(false=纯 GRPO 基线) | 臂定义 |
| `doc_table` | (162 路径) | task_id→doc jsonl(GT 文档) | 换文档源时 |
| `eta_max` | 0.5 | 单次倾斜强度上限(密度比 ≤e^{2ηc}≈12×) | 敏感性消融 |
| `c` | 2.5 | 单 token 话语权限幅 | **标定后定**(步骤③) |
| `b_max` | 20.0 | 每轨迹预算(≈8 次整额干预,1-2% 作用面) | 用户裁定 v2;消融 {10,20,40} |
| `topk` | 64 | 候选集 S 两侧宽度 | 不动 |
| `obs_topk` | 256 | 引擎观测窗(联动 vLLM max_logprobs) | 不动 |
| `theta_d` | 0(防呆) | θ_D 的 **EMA 初值**(自适应关闭时即冻结阈值) | **标定后填**(步骤③);1e9=标定哨兵 |
| `gate_adaptive` | true | 率控自适应开关(θ_D 每 step EMA 追目标门率) | 标定 run 必须 false |
| `gate_target_rate` | 0.10 | 目标门率 q(干预密度旋钮) | 消融 {5,10,20}%(第二轮) |
| `gate_ema_beta` | 0.1 | EMA 慢速率 β | 不动 |
| `epsilon` | 0.05 | 探针翻转率 | 不动(0=标定模式) |
| `gate_rate_bounds` | [0.05,0.25] | 门率告警带 | 不动([0,1]=标定豁免) |
| `r_max` | 4 | 每轮拒绝上限 | 不动 |
| `w_beh_mode` | per_token | 训练纠偏口径 | 消融时(per_decision/vtrace) |

train 段关键:`adv_estimator: grpo_loo` + `norm_adv_by_std: false`(Dr.GRPO,
审阅裁定主档,**两臂一致勿动**);其余训练键与 4B 主线同源,注释在模板。

## 5. 反馈通道

问题/观察 → `docs/collab/cd010-cd006/experiment-feedback.md`(CD-006 维护,
append-only、带日期、结论先行、标优先级);机制侧变更由 CD-010 记
`mechanism-changelog.md`。**机制代码(steering/verl 面)勿在实验侧改**——现象
反馈,修复走 CD-010 出 commit。
