# 162 base110 → step358 最终运行报告

> 状态真值：2026-08-05。基础设施纪律见 `docs/infra/README.md`；本报告只记录
> `pgs_base_162_from110` 的执行、终点与恢复口径。

## 结论

162 四卡 base 续跑已完成逻辑 step 358：最终 train 128/128、固定 val 274/274、四卡 FSDP
checkpoint 与 `data.pt` 全部完整。最终裸 policy won 为 **265/274=96.72%**，相对迁移锚
step110 的 **240/274=87.59%** 上升 **9.12pp**；同任务31升/6降，配对二项检验
`p=4.13e-5`。训练产物与终点评测有效。

进程不是 clean exit：所有终点产物落盘后，SwanLab atexit `finish()` 等待期间报告一个
DataLoader worker 被 signal Killed，留下1个 Traceback/RuntimeError。无 OOM、NCCL、
EngineDead、RayTaskError或写盘失败。故准确判定为 **实验完成 / clean-exit失败**，不得写成
干净退出，也无需重启或补跑。

## 运行身份与迁移口径

| 项 | 真值 |
|---|---|
| 机器 | `ssh -p 5102 haoyang@10.77.110.162` |
| run | `pgs_base_162_from110` |
| run dir | `/data4/ws/PGS/pgs_base_162_from110` |
| 主日志 | `/data3/wushang/workspace/TRACE-OPD-PGS/logs/pgs_base_162_from110.log` |
| 起点 | A800 `global_step_110` 合并后的 HF 权重 + 原 `data.pt` |
| 终点 | 358（2875条、batch16、drop_last、2 epochs） |
| 训练拓扑 | 4 GPU、rollout TP=1、actor Ulysses SP=2 |
| 目标 | bare GRPO base，steering off |
| objective fingerprint | `104e5294da04` |
| SwanLab run | `7edvezi0`（运行中未执行 `swanlab sync`） |

8→4卡无法直接恢复原FSDP optimizer shard，因此模型从合并HF权重加载；StatefulDataLoader
通过跨机迁移的 `global_step_110/data.pt` 恢复数据游标。两机next-batch索引回放与162实际
step111 task集合已验证一致。Adam moments、optimizer RNG/extra state因world-size变化重置；LR
为constant，scheduler重置无效果。这是与原A800连续训练相比必须披露的迁移差异。

## 终点产物

证据根：`/data4/ws/PGS/pgs_base_162_from110/`。

- `rollouts/step_00358/`：128个非空、JSON-valid train dump；
- `rollouts/step_00358_val/`：274个非空、JSON-valid固定评测dump；
- `checkpoints/global_step_358/`：19文件、48,283,717,276 bytes；
- shard：4 model（每份4,022,682,075 bytes）+ 4 optimizer（每份8,045,373,322 bytes）+
  4 extra；
- 含 `data.pt`（7,316 bytes）、`fsdp_config.json`、HF config/tokenizer；
- `latest_checkpointed_iteration.txt=358`；
- `data.pt` SHA-256：`82f463108b484bfeb014b4f25ece47d6a5458f4088aad2e1e260c17b16c0525c`。

最终Ray指标来源：
`/tmp/ray/session_2026-08-04_21-21-31_928236_875612/logs/worker-9b928b34635cfb1eabf331e059e6fbfac8fa673d638e8e3fe3bdda96-01000000-882793.out`。

## 固定评测

| 端点 | overall | seen | unseen | invalid | truncated |
|---|---:|---:|---:|---:|---:|
| step110 | 240/274 = 87.59% | 123/140 | 117/134 | 2 | 2 |
| step330 | 262/274 = 95.62% | 135/140 | 127/134 | 1 | 1 |
| step340 | 264/274 = 96.35% | 135/140 | 129/134 | 1 | 1 |
| step350 | 262/274 = 95.62% | 136/140 | 126/134 | 0 | 0 |
| **step358** | **265/274 = 96.72%** | **136/140** | **129/134** | **1** | **1** |

step110→358：31个task由输转赢、6个由赢转输，净+25/274=+9.12pp，配对
`p=0.00004126`。step330→358和step350→358均为7升/4降、+1.09pp、`p=0.549`，表明
末段已在95.6%–96.7%高位平台，而非单点性能跳变。

### step358 by-family

| family | won |
|---|---:|
| look_at_obj_in_light | 30/31 = 96.77% |
| pick_and_place_simple | 58/59 = 98.31% |
| pick_clean_then_place_in_recep | 57/58 = 98.28% |
| pick_cool_then_place_in_recep | 45/46 = 97.83% |
| pick_heat_then_place_in_recep | 37/39 = 94.87% |
| pick_two_obj_and_place | 38/41 = 92.68% |

## 训练与崩塌健康

step358训练批：127/128 won（99.22%）、invalid=1、truncated=0；平均响应948 token。Ray实收：

- actor entropy：0.2671；
- grad norm：0.02054；
- actor max allocated：56.72GiB；
- actor update：29.11s；weight update：6.06s；
- checkpoint：47.80s；整步：240.76s；最终评测：79.88s。

最终val thought相邻逐字重复率：轨迹均值1.23%、pooled3.70%、3条>55%；最终train为
均值0.029%、pooled0.082%、0条>55%。entropy仍高于0.2且grad非零，固定val没有>10pp下跌，
没有A800处理臂的低熵安全门信号。

## 终止状态与后续边界

日志顺序明确为：`358/358`完成 → step358 checkpoint落盘 → step358 val 274/274与最终metrics
打印 → SwanLab atexit期间DataLoader worker signal。主训练进程和四卡服务随后退出，GPU显存约
14MiB/卡、利用率0。该错误只影响clean-exit声明，不改变终点数字和checkpoint有效性。

旧轮转点110–320均只遗留`data.pt`空壳；完整可恢复点为330/340/350/358，marker仍为358。
这些空壳不影响恢复真值，但任何自动枚举工具都必须按shard与`data.pt`完整性判定，不能只看目录名。
终点checkpoint可用于后续固定评测；任何继续训练都应视为新实验决策，而不是本run的故障恢复。

## 2026-08-05 维护复核

终点不变量无漂移：step358 train 128/128、val 274/274均非空且JSON-valid，checkpoint仍为
19文件/48,283,717,276 bytes、marker358，`data.pt`哈希不变；完整点仅330/340/350/358，
110–320仍为单个7,316-byte `data.pt`空壳。最终val仍265/274，train仍127/128 won。

错误面重扫仍只有业务产物全部完成后的1个SwanLab atexit/DataLoader worker Traceback；OOM、
EngineDead、RayTaskError、PytorchStreamWriter与写盘失败均0，NCCL命中仅是配置/版本文本。按真实
argv/PGID复核无training/verl/rollout进程；四卡各14MiB/0%且无compute app。`/data4`可用
5,629,623,984,128 bytes；`/data3`使用率97%、可用512,398,995,456 bytes，列为后续写重任务前的
容量预警，但没有证据支持清理。本run继续维持**已完成 / clean-exit失败 / 不重启、不补跑**。

口径澄清：step358 val的`invalid_format`是事件计数，唯一非零轨迹的字段值为3；因此scalar为
3/274，但“含invalid的轨迹数”为1/274，与报告表中的invalid=1一致。
