#!/usr/bin/env bash
# =============================================================================
# TRACE-OPD 1.7B 实验启动(162/zkgy)— 由 run_experiment.sh 派生(CD-006)
#
#   SWANLAB_API_KEY=... bash trace_opd/scripts/run_experiment_1p7b.sh          # 全量 1.7B
#   SMOKE=1 SWANLAB_API_KEY=... bash trace_opd/scripts/run_experiment_1p7b.sh  # 冒烟
#
# 与 run_experiment.sh 的差异(仅 3 处):
#   ① policy 基座 = Qwen3-1.7B(POLICY_MODEL,原脚本 4B);
#   ② full/smoke 都显式传 --run-name(默认 trace_opd_1p7b;原脚本 full 用 config
#      run.name=trace_opd_v1,会起错 run);
#   ③ full 带 trainer.save_freq=5(前 20 步任意崩溃可从 step-5 密集续,防从头)。
# 其余(预检/数据/27B researcher 复用/sleep-mode/env/nohup+存活确认/幂等 resume)
# 完全一致。蒸馏显微镜由 configs/trace.yaml 的 distill_microscope:true 自动生效。
# =============================================================================
set -euo pipefail

# ---- 部署事实(env 可覆盖)--------------------------------------------------
TRACE_REPO=${TRACE_REPO:-/data3/wushang/workspace/TRACE-OPD}
VENV_TRAIN=${VENV_TRAIN:-$TRACE_REPO/venv_train}
VENV_INFER=${VENV_INFER:-$TRACE_REPO/venv_infer}
VENV_ALFWORLD=${VENV_ALFWORLD:-$TRACE_REPO/venv_alfworld}
POLICY_MODEL=${POLICY_MODEL:-/data3/wushang/model/Qwen/Qwen3-1.7B}   # ← 1.7B(原脚本 4B)
MODEL_27B=${MODEL_27B:-/data3/wushang/model/Qwen/Qwen3.6-27B}
export ALFWORLD_DATA=${ALFWORLD_DATA:-/data3/wushang/data/alfworld_data}
PORT_27B=${PORT_27B:-8302}
SERVED_27B=${SERVED_27B:-Qwen3.6-27B}
CONFIG=${CONFIG:-$TRACE_REPO/trace_opd/configs/trace.yaml}
DATA_OUT=${DATA_OUT:-$TRACE_REPO/trace_opd/data/out}
LOG_DIR=${LOG_DIR:-$TRACE_REPO/logs}
RUN_NAME=${RUN_NAME:-trace_opd_1p7b}   # ← full run 名(原脚本用 config run.name)
SMOKE=${SMOKE:-0}
SMOKE_N=${SMOKE_N:-32}
SMOKE_STEPS=${SMOKE_STEPS:-3}
FORCE=${FORCE:-0}
TS=$(date +%m%d_%H%M%S)

say()  { echo "[run_1p7b] $*"; }
die()  { echo "[run_1p7b] FATAL: $*" >&2; exit 1; }

researcher_alive() {
    curl -sf --max-time 5 "http://127.0.0.1:$PORT_27B/v1/models" 2>/dev/null \
        | grep -q "\"$SERVED_27B\""
}

# ---- [0] 预检 ---------------------------------------------------------------
say "== [0/4] 预检(policy=$POLICY_MODEL) =="
[[ -d $TRACE_REPO/trace_opd ]]            || die "repo 不在 $TRACE_REPO"
[[ -x $VENV_TRAIN/bin/python ]]           || die "训练 venv 缺失: $VENV_TRAIN"
[[ -x $VENV_INFER/bin/vllm ]]             || die "推理 venv 缺失: $VENV_INFER"
[[ -x $VENV_ALFWORLD/bin/python ]]        || die "alfworld venv 缺失: $VENV_ALFWORLD"
[[ -d $POLICY_MODEL ]]                    || die "policy 基座缺失: $POLICY_MODEL"
[[ -d $MODEL_27B ]]                       || die "researcher 模型缺失: $MODEL_27B"
[[ -d $ALFWORLD_DATA/json_2.1.1 ]]        || die "ALFWorld 数据缺失: $ALFWORLD_DATA"
[[ -f $TRACE_REPO/trace_opd/verl/verl/__init__.py ]] \
    || die "vendored verl 缺失(trace_opd/verl/)"
mkdir -p "$LOG_DIR" "$DATA_OUT"

# GPU 抢占防护(infra 纪律):有大额占用又不是我们的 27B → 疑他人实验。FORCE=1 放行。
if [[ "$FORCE" != "1" ]] && ! researcher_alive; then
    max_used=$(nvidia-smi -i 0,1 --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)   # 共享期只查我们的卡0/1
    if (( max_used > 10240 )); then
        die "GPU 有占用(峰值 ${max_used}MiB)且 :$PORT_27B 无本实验服务——疑他人实验。协调后 FORCE=1 重跑"
    fi
fi

# ---- [1] 数据 ---------------------------------------------------------------
say "== [1/4] 数据 parquet =="
if [[ "$SMOKE" == "1" ]]; then TRAIN_PARQUET=$DATA_OUT/train_${SMOKE_N}.parquet
else                           TRAIN_PARQUET=$DATA_OUT/train_3553.parquet; fi
if [[ -f $TRAIN_PARQUET && -f $DATA_OUT/valid_seen.parquet && -f $DATA_OUT/valid_unseen.parquet ]]; then
    say "已存在,跳过构建: $TRAIN_PARQUET"
else
    say "构建中(引擎逐任务 boot 取 initial obs + GT 预提取)..."
    build_args=(--data-root "$ALFWORLD_DATA" --worker-python "$VENV_ALFWORLD/bin/python"
                --out-dir "$DATA_OUT" --config "$CONFIG")
    [[ "$SMOKE" == "1" ]] && build_args+=(--n-train "$SMOKE_N")
    PYTHONPATH="$TRACE_REPO/trace_opd/src" \
        "$VENV_TRAIN/bin/python" -m trace_opd.train.build_datasets "${build_args[@]}" \
        2>&1 | tee "$LOG_DIR/build_datasets_$TS.log"
    [[ -f $TRAIN_PARQUET ]] || die "构建结束但 $TRAIN_PARQUET 不存在(看日志)"
fi

# ---- [2] researcher 服务 ----------------------------------------------------
# REMOTE_RESEARCHER=1 → 走外部 35B-A3B 双端点(经隧道,watchdog 守护),
# 不起 27B、四卡归 policy 独占;默认 0 → 本地 27B(原路径,回切预案)。
REMOTE_RESEARCHER=${REMOTE_RESEARCHER:-0}
if [[ "$REMOTE_RESEARCHER" == "1" ]]; then
    say "== [2/4] 远程 researcher(35B-A3B 双端点 18888/18889,经隧道)=="
    bash "$TRACE_REPO/trace_opd/scripts/tunnel_watchdog.sh" ensure || true
    RKEY=${TRACE_RESEARCHER_API_KEY:-token-abc123}
    for lp in 18888 18889; do
        curl -sf --max-time 10 -H "Authorization: Bearer $RKEY" \
            "http://127.0.0.1:$lp/v1/models" | grep -q "qwen3.6-35b-a3b" \
            || die "远程端点 :$lp 不可用(隧道/远端服务)。看 $LOG_DIR/tunnel_watchdog.log"
    done
    say "双端点 OK(launcher 预检会再验一遍;单端点亦可跑,池自动故障切换)"
else
say "== [2/4] 27B researcher(:$PORT_27B,DP2×TP2,sleep-mode)=="
if researcher_alive; then
    say "已在服务,复用"
else
    say "起服中(nohup,日志 $LOG_DIR/researcher_27b_$TS.log)..."
    ( source "$VENV_INFER/bin/activate"
      MODEL=$MODEL_27B SERVED_NAME=$SERVED_27B PORT=$PORT_27B \
      nohup bash "$TRACE_REPO/trace_opd/scripts/serve_researcher.sh" \
          > "$LOG_DIR/researcher_27b_$TS.log" 2>&1 < /dev/null & )
    for i in $(seq 1 120); do
        sleep 10
        researcher_alive && break
        (( i % 6 == 0 )) && say "等待 27B 就绪... ${i}0s"
        [[ $i == 120 ]] && die "27B 1200s 未就绪,看 $LOG_DIR/researcher_27b_$TS.log"
    done
    say "27B 就绪"
fi
curl -sf --max-time 5 "http://127.0.0.1:$PORT_27B/is_sleeping" >/dev/null \
    || die "/is_sleeping 不可达——现存服务未开 sleep-mode。清理后重跑"
fi   # REMOTE_RESEARCHER

# ---- [3] 训练环境变量 -------------------------------------------------------
export ALFWORLD_WORKER_PYTHON=$VENV_ALFWORLD/bin/python
export TRACE_MODEL_PATH=$POLICY_MODEL                              # ← policy 基座 = 1.7B
export TRACE_TRAIN_FILE=$TRAIN_PARQUET
export TRACE_VAL_SEEN_FILE=$DATA_OUT/valid_seen.parquet
export TRACE_VAL_UNSEEN_FILE=$DATA_OUT/valid_unseen.parquet
if [[ "$REMOTE_RESEARCHER" == "1" ]]; then
    # 池模式:逗号分隔端点(PooledClient 会话粘性+均衡+故障切换,llm.py)
    export TRACE_RESEARCHER_ENDPOINT="http://127.0.0.1:18888/v1,http://127.0.0.1:18889/v1"
    export TRACE_RESEARCHER_MODEL="qwen3.6-35b-a3b"
    export TRACE_RESEARCHER_API_KEY="${TRACE_RESEARCHER_API_KEY:-token-abc123}"
    # 从主 yaml 现场派生远程配置(防两份 yaml 漂移):researcher 段
    # sidecar_sleep→false、dp_size→0、插入 api_key_env(config 校验强制这套组合)
    GEN_CONFIG="${CONFIG%.yaml}_remote35b.gen.yaml"
    sed -e '/^  researcher:/,/max_concurrency:/{
        s/sidecar_sleep: true/sidecar_sleep: false/
        s/dp_size: 2/dp_size: 0/
        s|^    model: "\${TRACE_RESEARCHER_MODEL}".*|&\n    api_key_env: TRACE_RESEARCHER_API_KEY|
    }' "$CONFIG" > "$GEN_CONFIG"
    CONFIG=$GEN_CONFIG
    say "远程 researcher 派生配置: $CONFIG(sidecar off / dp 0 / api_key_env)"
else
    export TRACE_RESEARCHER_ENDPOINT=http://127.0.0.1:$PORT_27B/v1
    export TRACE_RESEARCHER_MODEL=$SERVED_27B
fi
[[ -z "${SWANLAB_API_KEY:-}" ]] && say "提示:SWANLAB_API_KEY 未设,曲线走 offline 模式"

# ---- [4] 启动训练(nohup + 存活确认)---------------------------------------
say "== [3/4] launcher(preflight → verl)=="
if [[ "$SMOKE" == "1" ]]; then
    RUN_TAG=smoke; EFF_RUN="${RUN_NAME}_smoke"
    launch_args=(--config "$CONFIG" --run-name "$EFF_RUN")
    extra_overrides=("trainer.total_training_steps=$SMOKE_STEPS")
else
    RUN_TAG=full; EFF_RUN="$RUN_NAME"
    launch_args=(--config "$CONFIG" --run-name "$EFF_RUN")
    extra_overrides=("trainer.save_freq=5")                        # ← 密集存档防从头
    [[ -n "${EXTRA_OVERRIDES:-}" ]] && extra_overrides+=($EXTRA_OVERRIDES)   # 额外 verl override 通道
fi
TRAIN_LOG=$LOG_DIR/train_1p7b_${RUN_TAG}_$TS.log
say "run=$EFF_RUN  训练日志: $TRAIN_LOG"
nohup "$VENV_TRAIN/bin/python" "$TRACE_REPO/trace_opd/scripts/launch_train.py" \
    "${launch_args[@]}" "${extra_overrides[@]}" \
    > "$TRAIN_LOG" 2>&1 < /dev/null &
TRAIN_PID=$!

say "== [4/4] 存活确认 =="
sleep 15
kill -0 "$TRAIN_PID" 2>/dev/null || { tail -30 "$TRAIN_LOG"; die "训练进程 15s 内退出(以上为日志尾)"; }
grep -q "\[preflight\] researcher OK" "$TRAIN_LOG" \
    || say "警告:日志尚未见 preflight OK(可能仍在启动),自行确认"
say "训练已启动 PID=$TRAIN_PID  run=$EFF_RUN"
echo
echo "===================== 监控面板 ====================="
echo "  训练日志   : tail -f $TRAIN_LOG"
echo "  产物目录   : $TRACE_REPO/trace_opd/runs/$EFF_RUN/"
echo "  显微镜落盘 : $TRACE_REPO/trace_opd/runs/$EFF_RUN/microscope/rank*.jsonl"
echo "  停止训练   : kill $TRAIN_PID   (27B 可留用)"
echo "  断点恢复   : 原样重跑本脚本(resume_mode=auto)"
echo "===================================================="
