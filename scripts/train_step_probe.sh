#!/usr/bin/env bash
# =============================================================================
# 训练步进探针(162)— CD-006。供监控轮询的单行状态输出,跨 run 复用:
# 不带参数 = 自动锁定 logs/ 下最新的 train_*.log(新实验开跑自动跟随);
# 也可显式传日志路径:train_step_probe.sh /path/to/train_xxx.log
#
# 输出一行:RUN=<run名> LOG=<日志名> STEP=<已完成步(取进度条与metrics行的max)>
#          ERR=<错误计数> ALIVE=<main_ppo进程数> VAL=<最近val行(可空)>
# 步数口径:tqdm 进度条 "N/total" 的 N = 已完成 N 步(实时);
#          "training/global_step:N" metrics 行(权威但受 stdout 缓冲延迟)。
# =============================================================================
LOG_DIR=${TRACE_LOG_DIR:-/data3/wushang/workspace/TRACE-OPD/logs}
L=${1:-$(ls -t "$LOG_DIR"/train_*.log 2>/dev/null | head -1)}
[ -f "$L" ] || { echo "RUN=? LOG=none STEP=-1 ERR=0 ALIVE=0"; exit 0; }
run=$(grep -aoE "run_dir=[^ ]*" "$L" | head -1 | sed "s|.*/||")
step_m=$(grep -aoE "training/global_step:[0-9]+" "$L" | tail -1 | cut -d: -f2)
# 只认 Training Progress 的 tqdm 条(初始化阶段的其他进度条如 shard 加载会误报)
step_b=$(grep -aoE "Training Progress:[^|]*\| *[0-9]+/[0-9]+" "$L" | tail -1 | grep -oE "[0-9]+/[0-9]+$" | cut -d/ -f1)
step=$(( ${step_m:-0} > ${step_b:-0} ? ${step_m:-0} : ${step_b:-0} ))
err=$(grep -acE "FATAL|Traceback \(most recent|OutOfMemoryError" "$L")
alive=$(pgrep -fc "[m]ain_ppo")
val=$(grep -aoE "val-core/alfworld/valid_seen/reward/mean@1[^ ]*" "$L" | tail -1)
run=${run:-$(basename "$L" .log)}   # launcher 行未刷缓冲时回退用日志名
echo "RUN=$run LOG=$(basename "$L") STEP=$step ERR=$err ALIVE=$alive VAL=${val:-none}"
