#!/usr/bin/env bash
# =============================================================================
# TRACE-OPD 一键实验启动(162/zkgy 服务器版)
#
#   bash trace_opd/scripts/run_experiment.sh            # 全量实验
#   SMOKE=1 bash trace_opd/scripts/run_experiment.sh    # 冒烟(32 任务、3 step)
#
# 自动完成:环境/GPU 预检 → 数据 parquet 构建(缺则建)→ 27B researcher
# 起服或复用(DP2×TP2、sleep-mode)→ 训练 nohup 后台启动 + 存活确认。
# 幂等:再次运行同一命令 = 断点恢复(27B 复用、parquet 跳过、训练
# resume_mode=auto 从最近 checkpoint 续)。
#
# 所有路径默认取 162 部署事实(docs/infra/servers-zkgy.md),均可用同名
# 环境变量覆盖。超参数一律来自 configs/trace.yaml(单一真源,本脚本不带
# 任何训练超参;SMOKE 的步数上限经 verl override 传入并记入 manifest)。
# =============================================================================
set -euo pipefail

# ---- 部署事实(env 可覆盖)--------------------------------------------------
TRACE_REPO=${TRACE_REPO:-/data3/wushang/workspace/TRACE-OPD}
VENV_TRAIN=${VENV_TRAIN:-$TRACE_REPO/venv_train}
VENV_INFER=${VENV_INFER:-$TRACE_REPO/venv_infer}
VENV_ALFWORLD=${VENV_ALFWORLD:-$TRACE_REPO/venv_alfworld}
MODEL_4B=${MODEL_4B:-/data3/wushang/model/Qwen/Qwen3-4B-Instruct-2507}
MODEL_27B=${MODEL_27B:-/data3/wushang/model/Qwen/Qwen3.6-27B}
export ALFWORLD_DATA=${ALFWORLD_DATA:-/data3/wushang/data/alfworld_data}
PORT_27B=${PORT_27B:-8302}
SERVED_27B=${SERVED_27B:-Qwen3.6-27B}
CONFIG=${CONFIG:-$TRACE_REPO/trace_opd/configs/trace.yaml}
DATA_OUT=${DATA_OUT:-$TRACE_REPO/trace_opd/data/out}
LOG_DIR=${LOG_DIR:-$TRACE_REPO/logs}
SMOKE=${SMOKE:-0}                 # 1 = 32 任务小集 + 3 step 上限 + 独立 run 目录
SMOKE_N=${SMOKE_N:-32}
SMOKE_STEPS=${SMOKE_STEPS:-3}
FORCE=${FORCE:-0}                 # 1 = 跳过"GPU 被他人占用"防护(明知故犯时用)
TS=$(date +%m%d_%H%M%S)

say()  { echo "[run_experiment] $*"; }
die()  { echo "[run_experiment] FATAL: $*" >&2; exit 1; }

researcher_alive() {
    curl -sf --max-time 5 "http://127.0.0.1:$PORT_27B/v1/models" 2>/dev/null \
        | grep -q "\"$SERVED_27B\""
}

# ---- [0] 预检 ---------------------------------------------------------------
say "== [0/4] 预检 =="
[[ -d $TRACE_REPO/trace_opd ]]            || die "repo 不在 $TRACE_REPO(TRACE_REPO 覆盖)"
[[ -x $VENV_TRAIN/bin/python ]]           || die "训练 venv 缺失: $VENV_TRAIN"
[[ -x $VENV_INFER/bin/vllm ]]             || die "推理 venv 缺失: $VENV_INFER"
[[ -x $VENV_ALFWORLD/bin/python ]]        || die "alfworld venv 缺失: $VENV_ALFWORLD"
[[ -d $MODEL_4B ]]                        || die "基座模型缺失: $MODEL_4B"
[[ -d $MODEL_27B ]]                       || die "researcher 模型缺失: $MODEL_27B"
[[ -d $ALFWORLD_DATA/json_2.1.1 ]]        || die "ALFWorld 数据缺失: $ALFWORLD_DATA"
[[ -f $TRACE_REPO/trace_opd/verl/verl/__init__.py ]] \
    || die "vendored verl 缺失(trace_opd/verl/,见 VENDORED.md)"
mkdir -p "$LOG_DIR" "$DATA_OUT"

# GPU 抢占防护:卡上有大额显存占用、又不是我们的 27B 在服务 → 多半是别人
# 的实验(infra 纪律:起服前探活,勿抢占)。FORCE=1 才放行。
if [[ "$FORCE" != "1" ]] && ! researcher_alive; then
    max_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
               | sort -rn | head -1)
    if (( max_used > 10240 )); then
        die "GPU 有占用(峰值 ${max_used}MiB)且 :$PORT_27B 无本实验服务——疑似他人实验(CD-004?)。协调后 FORCE=1 重跑"
    fi
fi

# ---- [1] 数据 ---------------------------------------------------------------
say "== [1/4] 数据 parquet =="
if [[ "$SMOKE" == "1" ]]; then TRAIN_PARQUET=$DATA_OUT/train_${SMOKE_N}.parquet
else                           TRAIN_PARQUET=$DATA_OUT/train_3553.parquet; fi
if [[ -f $TRAIN_PARQUET && -f $DATA_OUT/valid_seen.parquet && -f $DATA_OUT/valid_unseen.parquet ]]; then
    say "已存在,跳过构建: $TRAIN_PARQUET"
else
    say "构建中(引擎逐任务 boot 取 initial obs + GT 预提取;冒烟档 32+274 实测约 1 分钟,全量 3553 耗时待 CD-006 实测补记)..."
    build_args=(--data-root "$ALFWORLD_DATA" --worker-python "$VENV_ALFWORLD/bin/python"
                --out-dir "$DATA_OUT" --config "$CONFIG")
    [[ "$SMOKE" == "1" ]] && build_args+=(--n-train "$SMOKE_N")
    PYTHONPATH="$TRACE_REPO/trace_opd/src" \
        "$VENV_TRAIN/bin/python" -m trace_opd.train.build_datasets "${build_args[@]}" \
        2>&1 | tee "$LOG_DIR/build_datasets_$TS.log"
    [[ -f $TRAIN_PARQUET ]] || die "构建结束但 $TRAIN_PARQUET 不存在(看日志)"
fi

# ---- [2] 27B researcher 服务 ------------------------------------------------
say "== [2/4] 27B researcher(:$PORT_27B,DP2×TP2,sleep-mode)=="
if researcher_alive; then
    say "已在服务,复用"
else
    say "起服中(nohup,日志 $LOG_DIR/researcher_27b_$TS.log)..."
    ( source "$VENV_INFER/bin/activate"
      MODEL=$MODEL_27B SERVED_NAME=$SERVED_27B PORT=$PORT_27B \
      nohup bash "$TRACE_REPO/trace_opd/scripts/serve_researcher.sh" \
          > "$LOG_DIR/researcher_27b_$TS.log" 2>&1 < /dev/null & )
    # 就绪判据 = HTTP 真实往返(infra 铁律:不信 pgrep/日志尾)
    for i in $(seq 1 120); do
        sleep 10
        researcher_alive && break
        (( i % 6 == 0 )) && say "等待 27B 就绪... ${i}0s"
        [[ $i == 120 ]] && die "27B 1200s 未就绪,看 $LOG_DIR/researcher_27b_$TS.log"
    done
    say "27B 就绪"
fi
curl -sf --max-time 5 "http://127.0.0.1:$PORT_27B/is_sleeping" >/dev/null \
    || die "/is_sleeping 不可达——现存服务未开 sleep-mode(旧进程?)。按 infra 铁律清理后重跑(pkill -9 -f 'vllm serve' 并核显存归零)"

# ---- [3] 训练环境变量 -------------------------------------------------------
export ALFWORLD_WORKER_PYTHON=$VENV_ALFWORLD/bin/python
export TRACE_MODEL_PATH=$MODEL_4B
export TRACE_TRAIN_FILE=$TRAIN_PARQUET
export TRACE_VAL_SEEN_FILE=$DATA_OUT/valid_seen.parquet
export TRACE_VAL_UNSEEN_FILE=$DATA_OUT/valid_unseen.parquet
export TRACE_RESEARCHER_ENDPOINT=http://127.0.0.1:$PORT_27B/v1
export TRACE_RESEARCHER_MODEL=$SERVED_27B
[[ -z "${SWANLAB_API_KEY:-}" ]] && say "提示:SWANLAB_API_KEY 未设,曲线将走 offline 模式"

# ---- [4] 启动训练(nohup + 存活确认)---------------------------------------
say "== [3/4] launcher(preflight → verl)=="
launch_args=(--config "$CONFIG")
extra_overrides=()
RUN_TAG=full
if [[ "$SMOKE" == "1" ]]; then
    RUN_TAG=smoke
    launch_args+=(--run-name "trace_opd_smoke")
    extra_overrides+=("trainer.total_training_steps=$SMOKE_STEPS")
fi
TRAIN_LOG=$LOG_DIR/train_${RUN_TAG}_$TS.log
say "训练日志: $TRAIN_LOG"
nohup "$VENV_TRAIN/bin/python" "$TRACE_REPO/trace_opd/scripts/launch_train.py" \
    "${launch_args[@]}" "${extra_overrides[@]}" \
    > "$TRAIN_LOG" 2>&1 < /dev/null &
TRAIN_PID=$!

say "== [4/4] 存活确认 =="
sleep 15
kill -0 "$TRAIN_PID" 2>/dev/null || { tail -30 "$TRAIN_LOG"; die "训练进程 15s 内退出(以上为日志尾)"; }
grep -q "\[preflight\] researcher OK" "$TRAIN_LOG" \
    || say "警告:日志尚未见 preflight OK(可能仍在启动),自行确认"
say "训练已启动 PID=$TRAIN_PID"
echo
echo "===================== 监控面板 ====================="
echo "  训练日志   : tail -f $TRAIN_LOG"
echo "  27B 日志   : ls -t $LOG_DIR/researcher_27b_*.log | head -1"
echo "  产物目录   : $TRACE_REPO/trace_opd/runs/   (research/ rollouts/ tape/ checkpoints/)"
echo "  研究概览   : grep 'research ' $TRAIN_LOG | tail"
echo "  显存锯齿   : watch -n 5 nvidia-smi   (阶段切换 27B 显存应升降)"
echo "  停止训练   : kill $TRAIN_PID          (27B 可留用)"
echo "  断点恢复   : 原样重跑本脚本(resume_mode=auto)"
echo "===================================================="
