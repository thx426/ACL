# 162 · Qwen3-1.7B 原始基座 · PGS S2 / GT 正式运行卡

> 状态真值日期：2026-08-05。基础设施纪律见 `docs/infra/README.md`；本卡只记录
> `pgs_s2_sgl_1p7b_162_gt` 的环境、门禁、运行与端点。

## 运行身份

| 项 | 值 |
|---|---|
| 机器 | `ssh -p 5102 haoyang@10.77.110.162` |
| run | `pgs_s2_sgl_1p7b_162_gt` |
| run dir | `/data4/ws/PGS/pgs_s2_sgl_1p7b_162_gt` |
| 代码 bundle | `/data4/ws/PGS/trace_opd_1p7b_s2_bundle` |
| 统一环境 | `/data4/ws/PGS/env_sglang_train_1p7b` |
| 主日志 | `<bundle>/logs/pgs_s2_sgl_1p7b_162_gt.log` |
| 模型 | `/data3/wushang/model/Qwen/Qwen3-1.7B`（原始base权重，从头训练） |
| 逻辑终点 | 358（2875条、batch16、drop_last、2 epochs） |
| objective fingerprint | `e9ef2835e0af`（含绝对路径，跨机不可直接比较） |
| launcher PID / PGID | `3707792 / 3707792`（已按entropy安全门受控停止；新连接复核进程组为空） |
| 启动时间 | 2026-08-05 10:42 +08 |
| SwanLab | online，run `obxifdy4`（运行中禁止sync） |
| Ray原始指标 | `/tmp/ray/session_2026-08-05_10-42-49_762350_3707793/logs/worker-b23a5724087c10b3b76f4aef49ec0b615316e7034e47200c76e81ba2-01000000-3713363.out` |

## 配置边界

配置以A800 4B正式臂`pgs_s2_sgl_a800/effective_config.yaml`为模板；机制和训练目标保持
一致：SGLang 0.5.8、rollout TP=1、actor SP=2、16×8、`grpo_loo`、
`seq-mean-token-mean`、adaptive θ_D、目标门率10%、`B∞=400`、budget pacing
`U0=5/p=2`、per-token W_beh、每10步val/save、训前固定274-task val。

必要执行差异只有：原始1.7B模型、162四卡、`/data4` run根、162 ALFWorld/数据路径与本机
spawn并发。GT表为2875个唯一key、2828非空/47空；不回填缺失，按ITT + `doc_present` + family
报告。

## 环境与代码身份

- 新环境独立构建，不修改旧`venv_train`或`venv_sglang`；
- `torch 2.9.1 / CUDA 12.9 / flash-attn 2.8.3 / SGLang 0.5.8 / Transformers 5.11.0 /
  Ray 2.56.1 / TensorDict 0.13.0`；
- FlashAttention与libtorch/libstdc++/cudart均解析到新conda前缀，不需旧`LD_PRELOAD`；
- 代码bundle从162生产树剔除历史runs后构建，并精确同步A800正式臂八个S2/SGLang补丁文件；
- objective fingerprint含绝对路径，跨机不可直接比值，只能做语义diff。

## 启动门禁

已通过：

1. 单卡真实矩阵乘与FlashAttention kernel；
2. 双卡NCCL all-reduce（rank0/1均SUM=3.0）；
3. 单卡SGLang memory-saver真实生成；
4. tail-only top-256：145-row full 0.215s、8-row tail 0.0397s（5.41×），首行None、后7行每行256项；
5. conda GCC14 + `NVCC_PREPEND_FLAGS=-ccbin=<conda g++>`，JIT不再降级；
6. adapter/steering `27 passed`；
7. 正式dry-run逐项确认原始模型、SGLang、TP1、SP2、16×8、2 epochs、十步val/save、
   `grpo_loo`与无std归一化；fingerprint=`e9ef2835e0af`。

首次1-step smoke在Ray TaskRunner导入期因统一环境缺`cachetools`退出，未起GPU更新/rollout；
失败目录与日志保留。随后按A800完整Python分发差集补齐闭包并验证TaskRunner导入。

第二次smoke已完成端到端门：step1为128/128 JSON-valid、16/128 won、invalid/truncated均0；
actor update=24.96s、SGLang weight update=3.51s、grad norm=0.466、entropy=0.0217、W_beh
mean=1.00003/off-1=2.67%。steering候选141,434、全局base-open门率2.74%、实际干预10,522、
θ_D=0.0338、pacing干预位置P50=42.82%，doc_missing=8/128。四卡checkpoint共19文件/
20,660,399,665 bytes（4 model+4 optimizer+4 extra+`data.pt`），marker=1。全部产物完成后
DataLoader worker在atexit被signal Killed，故准确判为“端到端门通过 / clean-exit失败”；不以该尾错
否定链路，也不复用smoke checkpoint。

原始1.7B smoke首步相邻thought exact-dup为轨迹均值36.10%、pooled39.19%、45/128条>55%，
与既有1.7B base早期约45.7%的高基线同量级；不能机械套用4B的55%绝对阈值。正式run于10:42
从干净目录启动，manifest `overrides=[]`且无任何smoke限制泄漏。

## 正式训前基线与启动状态

固定训前val已274/274：**14/274=5.11%**，seen 6/140=4.29%、unseen 8/134=5.97%；
invalid_format=6、truncated=2、平均响应2026 token。by-family仅simple 10/59与cool 4/46非零，
look_at/clean/heat/pick_two均0。相邻thought exact-dup为轨迹均值41.08%、pooled42.72%、
118/274条>55%，thought字符中位98；这是本基座正式零点，而不是训练后崩塌信号。

SwanLab已确认online（未执行sync）；正式step1已128/128并完成actor更新：22/128 won、
invalid_format=9、truncated=3、entropy=0.02406、grad norm=0.5923、W_beh mean=1.00012/
off-1=2.55%，actor update=22.54s、SGLang weight update=3.37s、整步113.81s。artifact全局
base-open门率2.63%、实际干预9394、θ_D=0.0338、pacing位置P50=42.68%、doc_missing=8/128；
相邻thought重复率均值36.56%、pooled40.87%、44/128条>55%，thought字符中位99。错误面全零。
step2也完成更新：2/128 won、entropy0.01759、grad norm0.153、W_beh off-1=2.05%；artifact门率
2.09%、干预10,153、pacing P50=42.49%、重复率均值40.58%/pooled40.83%/53条>55%。step3为
17/128 won、entropy0.02216、grad norm0.487、W_beh off-1=2.72%；artifact门率2.66%、干预10,994、
pacing P50=42.65%、重复率均值38.84%/pooled42.30%/49条>55%。adaptive θ_D三步为
0.0338→0.03090→0.02809，证明率控没有冻结，但门率仍低于10%目标。三步actor与SGLang
weight update均完成、grad均非零、错误面全零；高重复与低熵均在训前零点存在，不能按4B绝对值
直接定性。首个可恢复安全端点为step10，届时结合固定val与连续低熵按预注册门裁定。

## step10安全端点与受控停机

**结论：本run已按预注册entropy安全门停机，唯一固定评测/可恢复端点为step10。**
`docs/pgs-validation-design.md:48-49`规定thought逐字重复率>55%、entropy持续<0.2、
val单点下降>10pp任一触发即停机保checkpoint。step1–10的actor entropy依次为：

`0.02406 / 0.01759 / 0.02216 / 0.02092 / 0.01770 / 0.02135 / 0.01800 /
0.02216 / 0.02062 / 0.02294`

十步均值0.02075，范围0.01759–0.02406，连续十步远低于0.2，故门已明确触发；不能因val
上涨而事后覆盖预注册的“任一触发”规则。2026-08-05约11:12 +08向精确训练PGID `3707792`
发送TERM，并停止只读watchdog PGID `3838700`。新连接复核两个进程组均为空，四卡均回到
14MiB/0%利用率；无Traceback、OOM、NCCL、EngineDead、RayTaskError、DataLoader worker或
写盘失败。

### 固定274-task val

| 端点 | overall | seen | unseen | invalid | truncated | 平均响应 |
|---|---:|---:|---:|---:|---:|---:|
| step0 | 14/274 = 5.11% | 6/140 | 8/134 | 6 | 2 | 2026 token |
| **step10** | **23/274 = 8.39%** | **13/140** | **10/134** | **6** | **2** | **2004 token** |

同任务为12升/3降，净+9题=+3.28pp，exact paired binomial `p=0.0352`。by-family从step0到
step10为：simple 10→14/59、clean 0→3/58、cool 4→6/46；look_at、heat、pick_two仍为0。
这是方向为正的早期信号，但274×单次greedy对小效应功效有限，且greedy本身非逐字确定，不能用
单端点把机制效果写成定论。更重要的是，val上升不取消独立entropy安全门。

step10固定val相邻thought exact-dup从训前轨迹均值41.08%降至35.14%，pooled从42.72%降至
37.26%，>55%轨迹从118降至99，thought字符中位98→95；没有“复读恶化”指纹。因此准确表述是
**entropy安全门触发**，不是已证实训练诱发的复读型崩塌。原始1.7B训前本来就高重复、极低熵，
说明0.2绝对门对该基座可能不校准；本run仍严格执行现行规则，后续是否按模型族定标必须另行预注册，
不能在本run中途改口径。

### step10训练与机制读数

step10 train为14/128 won、invalid=3、truncated=1；entropy=0.02294、grad norm=0.5481，W_beh
mean=1.000108/off-1=3.274%，证明per-token行为权重持续抵达原生loss。actor update=22.16s、
SGLang weight update=3.12s、整步111.14s。artifact全局base-open门率3.287%、干预12,032次，
θ_D均值0.014843（worker范围0.014833–0.014854），相对step1的0.0338持续下降，证明adaptive
tracker工作；门率仍显著低于10%目标。pacing干预位置P50=42.29%，无头部速烧；step10批
相邻thought重复率均值26.60%、pooled28.50%、32/128条>55%，thought字符中位94。

### checkpoint、停机尾部与恢复边界

`global_step_10`共19文件/20,660,399,665 bytes：4 model + 4 optimizer + 4 extra，含
`data.pt`与HF/FSDP metadata；marker=10，`data.pt` SHA-256为
`85a92d66881a57a4dedc4d3eb0e788ef7886f2d02cd4ed3eeb1ae8fe9a110632`。这是唯一完整checkpoint。
但adaptive `θ_D` worker状态尚不进checkpoint；未来若resume，会从配置初值0.0338而非step10实收
0.014843重收敛。因此它是模型/optimizer/RNG/数据游标可恢复锚，不是机制状态逐位连续锚；本轮不
自行恢复。

安全审计和停机命令落地存在流水线延迟：Ray原始metrics证明step11–14各自均完成actor与SGLang
weight update，entropy仍为0.02069/0.02515/0.02744/0.02298；它们各有128条train dump，但无
固定val、无checkpoint。step15仅9/128条，为未完成尾部。故最新完成参数更新是step14，但唯一可
恢复且有固定评测的科学端点仍只认step10；不得把11–14写成仅生成，也不得把其目录伪装为恢复锚。
该约4步停机越界是P0运维缺口：后续应把安全门计算与停止动作内置训练循环，而不是等外部十步审计。

## 2026-08-05 纠正：旧run为操作性误停；完整实验另起干净run

旧run停止不是机器、训练链路或SGLang异常，而是实验执行侧在step10形成完整端点后，错误地把4B
预注册的`entropy<0.2`绝对门跨模型套到原始1.7B，并主动向PGID3707792发送TERM。原始1.7B
训前entropy本来约0.02，说明该阈值没有跨模型校准；用户目标是推进完整358-step实验，故这次停机
属于操作性误停，不应写成run自身故障。旧目录完整保留，不删除、不复用。

不能从旧step10直接续跑来声称科学连续：adaptive θ_D worker状态未进checkpoint，会从配置初值
0.0338而非step10的0.014843重启；同时TERM落地前step11–14已完成参数更新但没有checkpoint，
resume会回到step10并重放后续数据。纠正方案是使用相同原始模型、GT表、代码和机制配置另起干净
run `pgs_s2_sgl_1p7b_162_gt_full`，从step0运行至358。新run目录：
`/data4/ws/PGS/pgs_s2_sgl_1p7b_162_gt_full`；主日志：
`<bundle>/logs/pgs_s2_sgl_1p7b_162_gt_full.log`；PID/PGID=`102239/102239`。

新run配置SHA-256仍为`7f3ed56c5305a435f4e69e2580a713e57446d4a71ee0cc7cd793b54e710c7d4c`，
GT表SHA-256仍为`25367813d29477f9f5b4695016bb032a1b72bb88c2d84f4117129ae96a1a0dbd`，
objective fingerprint仍为`e9ef2835e0af`，manifest `overrides=[]/config_overrides=[]`。dry-run创建的
3文件空目录已原子重命名为`pgs_s2_sgl_1p7b_162_gt_full_prelaunch_dryrun_20260805T115058`
留证，无训练产物被覆盖或删除。

新run训前固定val已274/274且SwanLab online：13/274=4.74%（seen5/140、unseen8/134），
invalid/truncated均0；相邻thought exact-dup均值38.08%、pooled39.22%、105/274条>55%。这与旧run
训前14/274和高重复基线同量级，差1题符合greedy非逐字确定性。后续会记录低entropy，但**不再用
未校准的4B绝对0.2阈值中止本1.7B full run**；只有真实基础设施fatal、写盘失败、产物停滞等异常
进入诊断/恢复流程。

## 长跑维护（已结束）

- 完成判据只认128条train JSON、274条val JSON、完整四卡checkpoint与latest marker；
- 运行中未执行`swanlab sync`；
- 本run因预注册entropy门有意停止，不得由watchdog或人工当故障自动恢复；
- 若未来另行批准resume，须先处理adaptive θ_D checkpoint缺口，并明确step11–14不可恢复尾部语义。

远端只读watchdog曾以PID/PGID=`3838700/3838700`运行，周期600秒、stale门1800秒、目标358；
安全停机后已受控停止，避免将有意停机误报为故障。
