# SSB 第四域(经典主线)RUNBOOK — 从零到发车

> 2026-08-17 起。用户三裁定:①探针先行、正式 run 排 P0 后;②协议直接
> text 化 + one-shot;③终评 400 全集报数。
> 规划真源:`experiments/bench_env/docs/2026-08-17-ssb-classic-integration-plan.md`
> 上游资产:SRL 引入 `experiments/bench_env/docs/2026-08-11-spreadsheet-srl-env.md`

## 0. 关键事实(2026-08-17 实测锚)

| 项 | 值 |
|---|---|
| A800 单根 $SSB_DATA | `/fs/fast/u2021000902/wushang/data/ssb_raw/spreadsheetbench_verified_400`(其下 `spreadsheet/`=SSB 400 xlsx、`excelforum/`=SRL 树、`dataset.json`、`srl_manifests/`) |
| A800 LibreOffice | `/fs/fast/u2021000902/wushang/opt/libreoffice/opt/libreoffice7.6/program/soffice`(**7.6.7.2**;162 判定用 6.4.7.2 ⇒ oracle 必须 A800 重判) |
| A800 判定/构建 python | `/fs/fast/u2021000902/wushang/workspace/TRACE-OPD/env_sglang_train_q35/bin/python`(openpyxl **3.1.5** / py 3.12.13,与 162 判定环境同版) |
| 代码树 | `/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM`(本地 trace_opd-bench 为真源,scp 同步;2026-08-17 已同步 text 协议 adapter/config/pytool 三件,md5 对拍一致) |
| 162 数据源 | `162env:~/srl_oracle/`(task-data 876M + dataset + 清单;user=haoyang) |
| verified-400 | **全库单 case**(soft≡hard 二值,与 SRL 论文 Pass@1 同口径);42930 golden 文件名 typo 已 symlink 修 |
| 训练 parquet 目标 | `/fs/fast/u2021000902/wushang/data/classic/classic_ssb_data/{train_srl,val_srl123,test_ssb400}.parquet` |
| 配置 | `configs/classic/ssb_7b_{pgs,grpo}.yaml`(fp d62e5b59cc63 / bab172aa2c89;protocol=text 进指纹) |

## 1. 数据部署(已完成 2026-08-17)

```bash
# A800 上执行;从 162 拉 excelforum(876M)与清单(~2min 内网)
ROOT=/fs/fast/u2021000902/wushang/data/ssb_raw/spreadsheetbench_verified_400
ssh 162env "tar -C ~/srl_oracle/task-data -cf - excelforum" | tar -C $ROOT -xf -
ssh 162env "tar -C ~/srl_oracle -cf - exclusions.json golden-selfcheck-*.json \
  out/oracle-full-*.json dataset/split_manifest.json" | tar -C $ROOT/srl_manifests -xf -
```
验收:`excelforum` 下 5,928 目录;`srl_manifests/` 含 exclusions/golden-selfcheck/out/。

## 2. oracle 全量重判 @ A800(LO 版本差异 ⇒ unreliable 清单必须本机重出)

启动器 `$ROOT/../run_oracle_a800.sh`(setsid+nice 32 workers,162 同款):
```bash
bash /fs/fast/u2021000902/wushang/data/ssb_raw/run_oracle_a800.sh
# 产物 $ROOT/srl_manifests/out_a800/oracle-full-<ts>.json;日志 srl_manifests/oracle_a800.log
```
162 版读数锚:fail 410/5,924=6.9%(A800 LO 7.6 预期同量级;显著偏离要停下诊断)。
⚠ probe 清单(G1/G2)是 openpyxl 层判定,openpyxl 同版 ⇒ **直接复用 162 版**
(`srl_manifests/golden-selfcheck-20260811-141156.json`);exclusions(人工裁定)同复用。

## 3. parquet 构建 @ A800(openpyxl 红线:initial_obs 必须生产环境重建)

```bash
WS=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM
PY=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD/env_sglang_train_q35/bin/python
ROOT=/fs/fast/u2021000902/wushang/data/ssb_raw/spreadsheetbench_verified_400
OUT=/fs/fast/u2021000902/wushang/data/classic/classic_ssb_data
# 3a. SRL train + valid(oracle 用 A800 重判版)
PYTHONPATH=$WS/src $PY -m trace_opd.envs.spreadsheetbench.build_dataset_srl \
  --data-root $ROOT --out $OUT \
  --probe $ROOT/srl_manifests/golden-selfcheck-20260811-141156.json \
  --oracle $ROOT/srl_manifests/out_a800/oracle-full-<ts>.json \
  --exclusions $ROOT/srl_manifests/exclusions.json
# 3b. SSB 400 全集测试 parquet(终评口径=400 全集,用户裁定 2026-08-17)
#     build_dataset.py 的 verified-400 入口;split 参数按脚本 --help
```
验收:train ≈5,37x(A800 oracle 数与 162 的 410 有出入属预期)/val 123±/
test 400;逐行含 `agent_name='trace_opd'` + `reward_model.style='env'`。

## 4. Phase 0 探针(工具链三件全现成,零新代码)

前置:A800 起 Qwen2.5-7B sglang 探针服务(单卡即可,与 P0 复跑错峰);
`probe/research_gen_docs.py` 发车前从本地 classic_bench 覆盖同步(本地为真源)。

```bash
# 4a. bare 起点探针(text 协议;GO 线 won ≥ ~8%,<5% 走 SFT 冷启动预案)
PYTHONPATH=$WS/src $PY -m trace_opd.evalm.run_eval \
  --config $WS/configs/classic/ssb_7b_pgs.yaml --arm bare \
  --parquet $OUT/test_ssb400.parquet --n 2 --temperature 0.6 \
  --model-path <7B> --api-base http://127.0.0.1:<port>/v1 --served-model qwen25-7b \
  --out-dir runs/ssb_probe/bare400
# 4b. C-R 产 doc(纯实验型,--no-privileged:SSB golden 红线,researcher 无特权素材
#     ——与训练态同构;四域先例是强特权口径,SSB 这点刻意不同)
python probe/research_gen_docs.py --config $WS/configs/classic/ssb_7b_pgs.yaml \
  --no-privileged --tasks-parquet $OUT/test_ssb400.parquet --n-tasks 24 \
  --model-path <7B> --executor-api-base http://127.0.0.1:<port>/v1 \
  --executor-model qwen25-7b \
  --researcher-api-base http://10.77.110.162:8880/v1,http://10.77.110.162:8881/v1 \
  --researcher-model qwen3.6-35b-a3b --out-dir runs/ssb_probe/cr24
# 4c. doc lift:同 4a 但 --arm doc --doc-dir runs/ssb_probe/cr24/docs(GO 线:lift 显著为正)
```

## 5. 正式双臂(等 P0 复跑让位;两硬门全过后)

```bash
# 冒烟 2 步定墙钟 → 裁定 batch(64 vs 128)与 decay_steps(总步数×1.05)→ 发车
# 环境注入:SSB_DATA=$ROOT SSB_SOFFICE=<soffice> TRACE_35B_KEY=<key>
# pgs: CVD 0-3 / grpo: CVD 4-7(ALFWorld 双臂同款布局)
```
观测面(SSB 特有,预注册):快提交率(turns≤2)分臂 / val 校准差 / doc 答案
密度审计 / 干预词数曲线;停手判据 = val 连续 3 点下行 且 快提交率分臂裂开。

## 5.5 smoke 验收定案(2026-08-17 夜,ssb3b_smoke,Qwen2.5-3B×4 卡,两次发车)

**全链 100% 跑通,机制面全部验证**(轨迹在 runs/ssb3b_smoke/,已停 run 保留数据):
- **bare 起点**:val 400 全集 7.0% / SRL 9.1%(avg@3 temp0.6;总 7.5%,GO 线贴边;
  SRL 论文锚 Qwen3-4B-Thinking base 12.0%);训练分布 temp1.0 bare ~11%。
- **text 协议(one-shot)实证**:格式违规 0.3% 步级(19/1283 episode),对照 FC
  档 SQA 11.8%;86% 显式 ```submit``` 终局;同现防线真实触发并正确处置。
- **v3.4 熵门自标定逐字验证**:step1 warmup 哨兵 θ=-1.0/零干预/26 万候选全
  bypass;step2 θ 首标 0.95-1.72 nats(16 worker 分位直标),剂量 p50 250→514
  词/条(早期→全批,约 ALF7B 首步 2 倍,与 episode 长度成比例)。
- **分臂/asym**:64+64 精确;step1(零干预)两臂同分布 ✓;step2 全批
  bare 6.2% vs steered 10.9%(+4.7pp,方向正,单步不下结论)。
- **researcher(35B外部)**:16 会话/步全 finalize,doc 均 430 词;会话内改进
  first 0.06→last 0.25(best 0.50),8/16 任务做出成功实验;doc 为方法论/排错型
  (HIGH 样本=七步 verified approach+陷阱;LOW 样本=诚实的已排除方向清单),
  未见答案值抄写——答案密度审计继续挂观测面。
- **ckpt**:save_freq=2 首存 global_step_2=35G(3B+optimizer,fsdp)。
- **吞吐**(120s/4096 旧档):step1≈20min(researcher12+rollout15 重叠+update);
  step2≈15min;val(1566eps)≈13-17min,p50 90 秒完成、其后全在收 9% 长尾。
  两次修正已落 yaml:max_tokens_per_turn 2048(裁定)/python_timeout_s 30
  (裁定;离线重放 409 段实证 30s 新杀率 0.00%——分布双峰,10-120s 真空)。
- **两个发车前修正**(smoke 抓到):①Qwen2.5 全系 max_position_embeddings=
  32768,rollout_max_model_len 40960 超限装配即死 ⇒ 全线 prompt8192+
  response24576=32768 恰贴上限(SRL 论文 response 亦 24576);②LXC 容器
  ps etime 列不可信(恒 00:00),判进程行为用累积 CPU TIME。

## 5.6 正式 run 与长期看护(2026-08-17 23:2x 发车)

**run**:`classic_ssb3b_pgs`(Qwen2.5-3B,CVD 4-7,指纹 `2eab6f438b1f`)。
用户裁定终值:batch **32 × n16**(组内 8 bare + 8 steered,512 eps/步)、
**334 步**(167/epoch × 2)、decay 350、**探针 epsilon=0**(全线默认已改
`SteeringCfg.epsilon` 默认值)、并发上调(`agent_num_workers` 32 /
`exec_concurrency` 64 / `rollout_max_num_seqs` 512 / researcher 96)。
**35B researcher 迁 A800 本机卡 1/2/3**(`serve_35b_a800.sh`,TP1×3 端点、
max_num_seqs **64**、max-model-len **65536** —— 后者是 researcher 会话峰值
8k prompt + 50k 流预算的硬需求,32768 会中途被端点拒;起服必须先 source
`env_pgs_sglang_a800.sh` 否则 inductor 找不到 ptxas 装配即死)。

**看护体系(三层)**:
1. **A800 watchdog**`probe/watchdog_ssb.sh`(自 watchdog_single.sh 派生,
   三处改造:fire 注入 SSB env 而非 ALFWORLD_*、35B 探活改本机三端点、
   重启日志进 run 目录);启动 `setsid env RUN_PREFIX=classic_ssb3b
   CFG_STEM=ssb_3b TOTAL_STEPS=334 ARMS=pgs CVD_PGS=4,5,6,7 PORT_PGS=34000
   RAY_TAG=ssb3b bash probe/watchdog_ssb.sh &`;停 `touch runs/watchdog_classic_ssb3b.stop`;
   日志 `runs/watchdog_classic_ssb3b.log`(5 分钟一轮:崩溃自动重启续跑、
   卡死 45 分钟只告警、researcher 异常率 >50% 告警防机制静默降级)。
2. **本会话 cron 巡查**(每 2 小时):进度/错误/watchdog 尾部/val 曲线/
   分臂 gap/35B 存活;判断纪律见 cron prompt(连续两次不动才算停滞等)。
3. **显微镜** 8791(指向本 run,实时增量)。

**step0 基线(1565 条,avg@3 temp0.6)**:won **5.2%**(SSB400 5.3 / SRL 5.2),
20 轮打满 22.1%,格式违规 0.6% 步级。

**★ 两刀无害的同口径对拍(2026-08-17,推翻一次误报)**:正式(2048/30s)
vs smoke(4096/120s)同脚本对拍 —— 每轮生成 token **p50 309 vs 307、
p90 706 vs 710(分布几乎重合,2048 只在 p99 收窄 1840→1442)**;timeout
**两侧同为 0.1% eps / 2 条**(30s 零新杀,与离线重放预测吻合);syntax/EOF
错误正式**更少**(1.60% vs 1.97%);20 轮打满 22.1% vs 23.1%。won 5.2% vs
6.1% 的差 = 0.9pp ≈ 1.1σ(n=1564,σ_diff 0.82pp),**采样噪声非配置代价**。
> **方法论教训(致误报的根因)**:**中期快照会系统性高估 won** —— 先完成的
> episode 偏简单偏赢(smoke 中期 1283 条时 7.5%,完整 1565 条时 6.1%)。
> **跨 run 比较必须用完整批同口径同脚本**,否则会把采样顺序效应误读成配置代价。

## 5.7 两次中断与最终配置(2026-08-18)

正式 run 两次中断,均非机制问题,教训已固化:

**① 端点 OOM(23:37-23:55)**:我在 val 阶段把 35B 从 32→64 seqs 重启,首次真实
负载即 CUDA OOM(79.32G 卡装 67G 权重后只剩 ~4G 给激活)。后果:step1-2 的
researcher 全灭 → doc 缺失 → steered 臂静默退化成 bare。**放行门(强制)**:
端点起来后必须跑 `probe_35b_load.py 32 6000`(A800:`~/probe_35b_load.py`),
`ok=32/32` 才准发车 —— `/v1/models` 返回 200 不代表引擎可用。
**并发终值:3 × TP1 × 32 seqs**(96 服务位,batch32 的 32 会话/步用不满;
要真 64+ 并发只能 TP2)。

**② 驱动楔死(03:52)**:跑到 step5 复发已知故障(产物停滞 68 分钟 +
`nvidia-smi` hang,SSH/proc 正常)。处置=停 watchdog → 清自己的 415 个进程
(uid 校验,绝不碰卡 0 他人进程)→ **等 5-10 分钟驱动自愈**(★ 不要立刻判
"只能重启",详见 [[a800-driver-wedge-diagnosis]] 的 2026-08-18 修正)。
恢复后按 infra 纪律先跑真实张量运算验证再发车。

**吞吐实测与由此定的配置**:每步 800 episodes(512 rollout + 288 researcher
实验)⇒ **28-46 分钟/步**;334 步需 8-11 天 ⇒ **`total_epochs` 2→1(167 步,
decay 176)**,仍是 SRL 论文训练量(60 步×1024=61k eps)的 2 倍;save_freq=20
保证可按 val 曲线随时提前停。恢复链一键脚本:`recover_and_launch.sh`
(等端点→压测放行→发车→起 watchdog,TOTAL_STEPS=167)。

## 5.8 ★ 意外收获:warmup 步天然分离出"doc 的 prompt 级效应"(2026-08-18)

正式 run 的 **step1 是 θ warmup 步(token 级干预恒 0)**,但 steered 臂的 prompt
里**已带 researcher 文档** —— 于是这一步干净地测出了"只给文档、不做 token
倾斜"的独立效应(每臂 256 条):

| step1(零干预) | bare | steered | 差 |
|---|---|---|---|
| 平均轮次 | 8.6 | **7.1** | −1.5 轮 |
| 平均 token | 4,741 | **3,824** | −19% |
| 预算耗尽率(20 轮打满) | 20.3% | **12.5%** | −7.8pp |
| won | 6.6% | 7.4% | +0.8pp(噪声内) |

⇒ **文档本身就让 3B 少走弯路**(更少轮次/更少迷失),而 won 未显著动 ——
即"效率先改善、正确率后跟上"。这是论文消融里 **doc-in-prompt vs
token-level 倾斜**两种效应的天然分解点,**零额外实验成本**(每个 PGS run 的
step1 都自带这组对照)。后续 val 若出现 won 提升,可回头用这组数据论证
"提升不只是把文档抄进 prompt"。

**同期克隆风险监测(SSB 预注册指标)健康**:快提交率(≤2 轮)bare vs steered
= step2 19.5/22.3、step3 13.7/14.3(+0.6~2.8pp),对照 engine 双降事故的指纹
(turns==2 从 31 条暴增 236 条 = 7.6×)**完全不在一个量级**,判无克隆信号。
执行质量:语法错 4.5-7.8%(两臂相当,3B 固有)、空输出 ~2%、**超时几乎归零**
(30s 裁定生效)、格式违规 episode 0.8-2.7%。

## 6. 已知坑(截至 2026-08-17)

- **pytool 已重写**(孤儿防线):朴素 subprocess.run 两坑——超时孙进程逃逸、
  **父正常退出但孙进程持管道 ⇒ communicate 挂满 timeout 误判超时**(本机复现,
  tests/test_pytool.py 锁行为)。A800 已同步。
- LO 首跑建 profile 慢属正常;SAL_DISABLE_OPENCL/SKIA 已在 recalc.py 内置。
- oracle 日志大量 openpyxl `Unknown extension` UserWarning = 正常噪声。
- A800 GPU 卡 0 有他人进程(2026-08-17),探针/正式 run 避开或先确认释放。
- 本地 trace_opd-bench 工作树在 terminal-9b-run 分支且 5 天未提交(46 文件,
  含整个经典主线开发)——git 落账待用户裁定;跨机同步以工作副本为真源。
