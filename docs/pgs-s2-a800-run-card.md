# PGS S2 / SGLang · A800 正式运行卡

> 状态真值日期：2026-08-05。基础设施与重建方式见
> `docs/infra/a800-sglang-training.md`；本卡只记录本次 run 的接管信息。

## 运行身份

| 项 | 值 |
|---|---|
| 机器 | `ssh -p 23653 u2021000902@10.10.252.11` |
| run | `pgs_s2_sgl_a800` |
| run dir | `/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/runs/pgs_s2_sgl_a800` |
| 主日志 | `/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/logs/pgs_s2_sgl_a800.log` |
| Ray 原始指标 | 首次：`/tmp/ray/session_2026-08-04_23-21-22_877834_28315/logs/worker-a6c0962c178b39c9848ac7da8667bdd39f09e1e4f75a0b7d05e0bada-01000000-32614.out`；step-50 恢复后：`/tmp/ray/session_2026-08-05_03-15-45_331199_71521/logs/worker-ba645f407159d5e9be12d9877f592c91080ef48a79f4c682c0beafd0-01000000-75924.out` |
| launcher PID / PGID | 首次 `28303 / 28303`；2026-08-05 step-50 恢复后 `71516 / 71516`（cmdline 必须含 `pgs_s2_sgl_a800`） |
| TaskRunner PID | 首次 `32614`；恢复后 `75924` |
| 启动时间 | 2026-08-04 23:21 +08 |
| 逻辑终点 | 358（2875 条、batch 16、drop_last、2 epochs） |
| objective fingerprint | `52431755eb79`（含绝对路径，跨机不可直接比较） |

配置：SGLang 0.5.8、8 GPU、rollout TP=1、actor SP=2、自适应 θ_D、budget pacing、
B∞=400、共同 `sft_step6`、save/test 每 10 步、训前 val。启动时 manifest 的 manual
overrides 为空，不含 2-step smoke 覆盖。

## 初始验收

- 训前 val：274/274；seen 23.57%、unseen 29.85%、整体 26.64%。
- Step 1–3：均 128/128 JSON，actor update 与 SGLang weight update 均完成。
- `rollout_is_weight_off1_frac`：3.876% / 3.654% / 3.907%。
- Entropy：0.669 / 0.641 / 0.680。
- Actor max allocated：24.88 / 28.76 / 31.11 GiB。
- Step wall：207.2 / 188.0 / 191.9s。
- θ_D：0.0338 → 0.03249 → 0.03077；pacing/steering 均非零。
- 前三步没有 Traceback、OOM、EngineDead、fatal NCCL、DataLoader worker 或 RayTaskError。
- Step 1/2 各一组 `doc_missing=0.0625`、step 3=0。已定位为 doc table 中 exact-match key
  的空字符串，不是 task-id/SGLang bug：2875 key 仅 2828 非空，47 空（1.635%），其中
  pick_two 38/674（5.638%）。本 run 不回填数据；终点按全量 intention-to-treat +
  `doc_present` + family 三层报告。
- Step 1 pacing 位置实证：13,545 次实际干预中仅 1.77% 落全局前 16 token，98.23%
  位于后续 turn；最后一次干预相对位置中位 98.19%，旧 s1 头部预算速烧已消除。
- Step 1 thought 相邻逐字重复率 pooled 0.786%，0 条超过 55% 崩塌阈。

训练 step 的 won 任务组成每步不同，不是性能曲线；改善只看固定 274-task 的每 10 步 val。

## 2026-08-05 step-60 写盘故障与恢复

- 第一次 attempt 的 `step_00060` train 128/128 已完成，但 checkpoint 保存时 8 rank 均报
  `PytorchStreamWriter ... file write failed`；残缺目录只有 8 model shard，latest marker 仍为50。
  无 OOM/NCCL/EngineDead/DataLoader worker；八卡释放且 `nvidia-smi` 正常，故恢复点只认完整
  step 50。
- `/fs/fast` 全局约余39T、inode使用6%、128MiB写探针通过；不能武断写成全盘耗尽，更符合
  Lustre并发大文件瞬时失败或配额/目录容量压力。用户裁定在线最多保留一个 checkpoint；恢复
  override `train.max_ckpt_to_keep=1` 不进入 objective fingerprint，目标语义不变。
- step 50 在任何清理前复制到
  `/disk/trace-opd-recovery/pgs_s2_sgl_a800/global_step_50`：31文件、48,286,310,244 bytes，
  源/副本逐文件 SHA-256 相等。第一次 attempt 的 step 51–60（10×128 JSON）和故障日志归档到
  `/disk/trace-opd-recovery/pgs_s2_sgl_a800/attempt1_failed_step60_20260805T024918+0800/`。
- 活跃 run 回滚至 step 50 后以 `resume_mode=auto` 重启；03:19 明确加载8卡 model/optimizer，
  step 51 任务多重集与失败前完全一致，证明 `data.pt` 游标连续。恢复后 launcher/TaskRunner =
  `71516/75924`；watchdog PID `101534`，仍为600秒。
- 恢复时 `val_before_train=true` 使 step-50 val 重跑一次；第二次274条（221 won）单独移至
  `/disk/trace-opd-recovery/pgs_s2_sgl_a800/resume_20260805T0315/val_before_train_step_00050_val/`，
  活跃目录保留原正式端点274条（216 won），避免548条污染标准口径。
- 已知缺口：Adaptive `GateThresholdTracker` worker状态未进 checkpoint，θ_D 从配置初值0.0338
  重启（step50故障前均值0.04924）；恢复首步门率仍健康（artifact全局10.94%，Ray轨迹均值
  9.25%）。须反馈机制侧补 checkpoint 状态，当前不伪造恢复值。
- 最终恢复闭环条件已于03:31满足：重新生成的 step60 train=128、val=274、checkpoint=31文件/
  48,286,310,244 bytes且 marker=60；step61随后128/128。step60错误面全零，save checkpoint
  8.48s；W_beh off-1=11.37%、entropy=0.0649、θ_D=0.02873、artifact全局门率10.25%、
  pacing位置P50=49.98%、thought逐字重复率0。确认step60完整且step50有`/disk`灾备后，在线
  step50目录按用户裁定删除；Lustre只保留`global_step_60`（约45GB），后续由同一进程按
  `max_actor_ckpt_to_keep=1`安全滚动（新点成功注册后再删旧点）。

## 2026-08-05 step-70 entropy 安全停点

- 预注册安全门规定 entropy 持续低于0.2须停机保ckpt。step30/40/50/60/70依次为
  0.1791/0.1078/0.0620/0.0649/0.0389，故03:45在完整`global_step_70`处向精确PGID71516
  发TERM；17秒内训练进程组退出，八卡归零，无新基础设施fatal。watchdog101534已停止以免
  误报；显微镜50055继续只读服务。
- step70固定val=255/274=93.07%（seen132/140，unseen123/134）；invalid/truncated均0，
  thought重复率pooled0.64%、0条>55%。这不像既有复读崩塌，但执行上已触发预注册门；科学
  解释保留为“能力收敛/任务饱和 vs 过度确定化前置期”待多次val与base裁定。
- step70 checkpoint为31文件/48,286,310,244 bytes，marker=70，在线唯一；W_beh off-1=8.32%、
  门率约7.2%、θ_D=0.02678、pacing位置P50=47.8%。训练批128/128全赢、grad norm=0，符合
  该批GRPO组内零advantage。停机命令落地前流水线又完成step71–74：四步均各128条轨迹，且Ray
  原始指标均有actor update和SGLang weight update（step74 entropy=0.0507、grad norm=0.799、
  W_beh off-1=9.03%）；但未形成step71–74 checkpoint或固定val，故最新完成更新为74，**唯一可恢复且
  有固定评测的锚仍只认70**。
- **不要把当前A800停机当运行故障自动重启**；后续是否从70继续属于安全门后的实验裁定。

## 2026-08-05 维护复核

产物级复核未发现漂移：现有train step1–74均为128个非空、JSON-valid文件，现有固定val
step0–70均为274个非空、JSON-valid文件；最新完成actor/SGLang更新仍为step74，最新固定val、
checkpoint与marker仍只到step70。`global_step_70`保持31文件/48,286,310,244 bytes（8 model+
8 optimizer+8 extra+`data.pt`+HF/FSDP metadata），`data.pt`为7,316 bytes，SHA-256=
`6bd547bc04260670cd3308d777e1d659afa590103be70301fd1c27cfd436f682`。

step74训练批115/128 won、invalid/truncated均0；Ray实收entropy=0.0507、grad norm=0.799、
W_beh off-1=9.03%。artifact全局门率1505/16772=8.97%、实际干预2143、θ_D均值0.02196
（worker范围0.00733–0.03545）、pacing位置P50=50.0%；相邻thought exact-dup均值0.108%、
pooled0.346%、0/128条>55%。机制链路仍活着且无复读恶化，但连续低熵安全门未解除；step74仍
无val/checkpoint，恢复锚只认70。

即时`/proc`按argv/PGID复核：训练PGID71516与watchdog PGID101534成员均为0，对应PID文件是
陈旧记录；仅只读显微镜PID/PGID50055监听`127.0.0.1:8793`并精确绑定本run。八卡均0MiB/
0%利用率，空闲功耗57.81–62.14W；`/fs/fast`可用42,185,083,326,464 bytes（约38.37TiB）。
恢复attempt未新增Traceback/OOM/NCCL/EngineDead/DataLoader/RayTask/写盘失败；历史首次step60
写盘故障仍单列。结论保持**有意安全停机 / 无需恢复或清理**。

## 实时显微镜

- 远端：`127.0.0.1:8793`，PID `50055`；绑定当前 run，加载 2875 条 doc。
- 本地：`http://127.0.0.1:8793/`，SSH tunnel PID `52215`。
- 已验收 `/`、`/api/run`、`/api/step/step_00001`、`/api/traj/step_00001/0`。
- 旧 8792 仍绑定 `pgs_base_a800`，不要混淆。

## 长期监控

远端 watchdog：

```text
script:  <PGS>/trace_opd/scripts/pgs_run_watchdog.py
PID:     <PGS>/logs/watch_pgs_s2_sgl_a800.pid   # step-50 恢复后 101534
log:     <PGS>/logs/watch_pgs_s2_sgl_a800.log
period:  600s
stale:   1800s
```

watchdog 只读、只告警，不自行 kill/restart。会话内另有每小时深审和 step-10
train/checkpoint/val 完整性唤醒。运行中的 SwanLab 禁止 `swanlab sync`。

## 每 10 步验收

1. `step_N` 恰 128 个非空、JSON-valid 文件；`step_N_val` 恰 274 个。
2. checkpoint marker=N；8 model + 8 optimizer + 8 extra shard，含 `data.pt` 与 HF metadata。
3. Ray 原始 step 指标含 actor update、weight update、IS、entropy、grad、显存、θ_D、pacing。
4. 检查 fatal、主进程 cmdline、八卡显存/温度、可用 RAM、`/fs/fast` 余量。
5. 分 family 报 val；greedy 仍有随机性，跨点使用误差意识。
6. 行为层检查 thought 重复率、长度/截断、polarization、干预位置分布。

## 故障处理边界

- 先按 run 产物 + Ray 原始日志 + `/proc/<pid>/cmdline` 诊断，不凭进程名或 console 尾部。
- 只在证据支持时恢复；不要因一条长尾轨迹或 console 缓冲盲重启。
- 如退出，先验 checkpoint 是否完整，再根据 launcher/checkpoint manager 的真实恢复语义续跑。
- 进程清理必须按精确 PID/PGID，检查端口、GPU 显存和多级 SGLang/Ray 子进程。
- 2-step smoke 曾在全部产物完成后 DataLoader worker 被 signal 杀死；正式 run 对该签名单列告警。
