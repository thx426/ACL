# 论文主实验结果台账(6 格 PGS × SEED 对拍)

> **单一真源**:6 格主实验的终值数字全部登记于此,每条含可追溯路径。
> 口径:主表 = 我方 **avg@3 @ temp0.6**;baseline 全部引用 SEED(arXiv 2607.14777)Table 1;
> 差异见总控 `docs/classic-main-matrix.md` §2 的 6 条脚注。
> **监控 val 读数一律不进论文**(SQA 尤其:训练内用 val_eq320,论文数字出自 test 全量终评)。

## 状态总览

| 格 | 状态 | 终值(Succ / Score 或 seen / unseen) | SEED 本尊 | 差 |
|---|---|---|---|---|
| **WS7B** | ✅ **完成** 2026-08-25 | **83.78 / 89.49** | 78.1 / 89.7 | **Succ +5.68** / Score −0.21 |
| WS3B | 待发 | — | 78.9 / 88.5 | — |
| ALF3B | 待发 | — | 91.8(seen macro) | — |
| ALF7B | 待发 | — | 96.1(seen macro) | — |
| SQA3B | 待发 | — | 45.7(7 集 macro) | — |
| SQA7B | **在跑** 2026-08-27 11:10 @A800(重发;1325 步) | — | 48.6(7 集 macro) | — |

---

## WS7B — WebShop × Qwen2.5-7B-Instruct × PGS ✅

**身份**:run `classic_ws7b_pgs` @162 卡 0-3 · 指纹 `f2bb626e6f2d`(= A800 语境 `463b32932ef9`,仅宿主路径异)
**训练**:2026-08-23 13:49 发车 → **2026-08-25 12:53 完跑 200/200 步**(1 epoch,约 47 小时)
**协议**:batch 32×n16 = 512 eps/步 · lr 1e-6 constant · Dr.GRPO(无 KL、norm_adv_by_std=false)· pgs_asym_grpo · 熵门 v3.4 自标定 · steered_fraction 0.5 · decay 210 · epsilon 0.0 · **零 SFT**

### 终值(测试集 `classic_webshop_data_v4/test.parquet` 全量 500 goals)

| 指标 | mean | std | min | max | n |
|---|---|---|---|---|---|
| **Succ**(won/reward mean@3) | **83.78%** | 0.29 | 83.47 | 84.27 | 10 |
| **Score**(score mean@3) | **89.49%** | 0.24 | 89.12 | 89.86 | 10 |

**n=10 的来源**:训练完成后 watchdog 完成盲区导致 10 次空转重启,每次对**同一 ckpt `global_step_200`** 完整跑了一遍 avg@3@0.6 全量评测(详见下"运行异常")。即 **同一权重的 10 次独立评测**——极差仅 0.8pp / 0.74pp,是**评测重复性的直接实证**(可回应"单次评测不可靠"的审稿质疑)。

其他终值指标(末次评测):env_steps mean@3 = 5.44 · invalid_format = 0.27% · truncated = 0.067% · num_turns mean 11.89(min 7 / max 31)。

### SEED(2607.14777)Table 1 对拍 — Qwen2.5-7B-Instruct × WebShop

| 方法 | Score | Succ |
|---|---|---|
| GRPO(SEED 报告) | 80.9 | 72.6 |
| SEED(本尊) | 89.7 | 78.1 |
| **本工作 PGS** | **89.49** | **83.78** |

⇒ **Succ 超 SEED 本尊 +5.68pp、超其 GRPO +11.18pp;Score 与 SEED 持平(−0.21pp)、超其 GRPO +8.59pp。** 且本工作**零 SFT 冷启动**(SEED 用了 1440 条 GLM-5.2 标注轨迹)。口径差异按总控 §2 脚注 1/2/3/4/5/6 处理(测试集我方 500 goals vs SEED 128 tasks;评测 avg@3@0.6 vs temp0.4×1;预算 200 步 32×16 vs 150 步 16×8)。

### 训练曲线(val Score,test_freq=20)

| step | 0 | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Score | 34.98 | 76.26 | 80.11 | 81.51 | 82.68 | 84.38 | 85.74 | 87.07 | **82.54** | 88.88 | 88.95 |

step0 bare 基线 Succ = **15.33%**(3B 同点 4.0%)。**step160 单点回落 −4.5pp 后回升**,待分析(候选:评测采样波动 / α 退火阶段 / 熵门 ratchet 事件)——分析归 evidence-map 挂账。

### 数据路径(162,`/data4/wushang/classic_tree/`)

- ckpt(**唯一保留**,86G 四 rank 完整):`runs/classic_ws7b_pgs/checkpoints/global_step_200/`(`latest_checkpointed_iteration.txt`=200;其余 9 个 ckpt 按 `max_ckpt_to_keep=1` 已清空,只余 12K `data.pt`)
- 训练日志(24,099 行,含全部 11 次启动):`runs/classic_ws7b_pgs.launch.log`
- watchdog 日志:`runs/watchdog_ws7b.log`
- rollouts / researcher 会话:`runs/classic_ws7b_pgs/{rollouts,research}/`(research 6,400 目录)
- effective_config:`runs/classic_ws7b_pgs/effective_config.yaml` · manifest ×13
- swanlab:project `trace_opd` / run `classic_ws7b_pgs`(末次 id `zi1ixz1x`)

### 运行异常(不影响结果,但记录在案)

1. **watchdog 完成盲区**(2026-08-25 13:03–15:41):训练正常完成退出后,watchdog 仅凭 `procs=0` 判为崩溃,连环重启 10 次(每次 resume→跑一遍全量 val→见 `global_step≥total` 即退出),触顶上限才停。代价 ≈2.6h×4 卡空转;**副产品 = 上表 n=10 的重复评测**。后续 5 格发车前须修完成门。
2. **发车早期 429**(2026-08-23,已知):日志前 2,355 行为前两发的 429 fail-loud 中止(残次归档于 `rollouts.aborted_429_1349/`);第三发(13:49)起至完成**全程零错误**。

---

## SQA7B — SearchQA × Qwen2.5-7B-Instruct × PGS ⏳ 在跑

**身份**:run `classic_sqa7b_pgs` @**A800 卡 0-3** · 指纹 **`b28e18c19eba`**(A800 语境 decay=1400 版;
2ep 旧版 47c6734f849b,decay 进指纹——asym α 三参数 weight:decay:cap:mode 同批,config.py ~1701)
**发车**:**2026-08-27 11:10(A800 重发,正式 run)**。前史:162 首日四发(piecewise×2 / val schema×1 /
成功 1)后,因 A800 GPU 被抢占致 8880 端点 hang、37.5% 会话降级 bare,用户裁定停杀重构
(162 17 步脏数据归档 `classic_sqa7b_pgs_aborted8880_20260827`)。
**新架构(2026-08-27 用户裁定)**:A800 全机自有——卡 0-3 训练 + 卡 4-7 四端点 35B researcher
(TP=1,8880-8883,serve_vllm.sh 环境序言);retriever 留 162 独占四卡(A800→162 HTTP 直连,
8-13 原架构);**client 端点熔断上线**(PooledClient:连败 3 摘除/半开自愈/全灭兜底,commit 440ccaf)
——单端点故障不再拖垮训练。
**协议**:batch 128×n8 = 1024 eps/步 · **1 epoch = 1325 步**(2026-08-26 用户裁定方案 B;decay 1400)· lr 1e-6 constant · Dr.GRPO · pgs_asym_grpo · 熵门 v3.4 · steered_fraction 0.5 · epsilon 0.0 · 零 SFT · 文本标签协议 + one-shot · retriever wiki-18 e5 topk3 · 交互 4 轮
**监控 val**:`val_eq320.parquet`(七源等额 6×320 + bamboogle 125 = 2045,md5 `1abbb5155ee8e6ef56147ab93c0adfd8`,两机一致);**论文数字出自 test 全量 51,713 终评,监控读数不进论文**

### A800 重发首验(2026-08-27,机制五面全过)

step0 七源(A800 自算,与 162 run Δ<2.2pp=评测随机性,macro 33.56):同集同权重同判分实证。
机制:researcher 128/128 零超时零错(docW 中位 ~420)/ 分臂恰 50% 行受引导(step1 warmup=0 ✓)/
doc_missing 全程 0 / won 41.0→45.3%(step6)。**墙钟 531.9 s/it ≈ 8.9 min/步**(vs 162 健康步 13.1:
本机四端点零隧道 + SXM NVLink + retriever 独占),投影 **9 月 4 日收官**。
观测面:A800 driver 日志无指标行(原因未深究),监控走 rollouts/*.json 自算 + tqdm。

### step0 基线(训练前,EM mean@3 @0.6,val_eq320)

| nq | triviaqa | popqa | hotpotqa | 2wiki | musique | bamboogle | **macro(7 源等权)** |
|---|---|---|---|---|---|---|---|
| 37.19 | 57.19 | 39.48 | 31.25 | 20.73 | 13.54 | 31.47 | **32.98** |

对照 SEED Table 1(7B,test 全量 macro):GRPO 42.0 / SEED 本尊 48.6 ⇒ 起点 headroom ≈ +9 / +15.6pp。
(口径警告:本行是 val_eq320 训练内读数,与 SEED 的 test 全量不可直接对比,仅作 headroom 量级参考。)
旧 8-13 run 起点(val_shuf2048 micro)≈35.1%——micro 被大源(popqa/2wiki/triviaqa 占 test 73%)拉高,macro 口径更严。

### 数据路径(162,`/data4/wushang/classic_tree/`)
- 日志 `runs/classic_sqa7b_pgs.launch.log`(第四发自第 3637 行起)· watchdog `runs/watchdog_classic_sqa7b.log`
- 时间预算:**162 实测 step1 = 788 s(13.1 min;gen 485 + update_actor 209)**,快于旧 A800 run 的 16.1 ⇒ 1325 步 ≈ 12.1 天 + val 67 次 × 16 min ≈ 18 h ⇒ **约 12.8 天,预计 9 月 8 日收官**;第一训练步 GPU 峰值 66.9G/卡(共卡余 ~13G)
