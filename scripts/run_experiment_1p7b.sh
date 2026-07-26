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
# swanlab 云端同步:未设则走 offline,曲线只落本地。
# run_experiment.sh 与 run_a800_4b.sh 早有此行,本脚本派生时漏带(2026-07-26 补)。
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-58KT7T652vKUKobNhHb50}
# flash-attn 来自 conda-forge 预编译包(手动提取到 venv),它需要 GCC15 的 libstdc++
# 才能满足 GLIBCXX_3.4.29 与 CXXABI_1.3.15;系统库只到 3.4.28,故在此预载。
# ★ 去掉这一行,ulysses_sp>1 会因 flash_attn 导入失败而起不来(2026-07-26 A2 首启实证:
#   ImportError: libstdc++.so.6: version `GLIBCXX_3.4.29' not found)。
#   run_experiment.sh 早有此行,本脚本派生时漏带。
_LDP=${LD_PRELOAD:-/data3/wushang/libcxx_x/un/lib/libstdc++.so.6.0.34}
# 仅当该库真实存在时才预载:它是 flash-attn(ulysses_sp>1)的前提,
# 路径为 162 专有;A800 关 SP 时不需要,硬设会让每个子进程刷 ld.so 报错。
[ -f "$_LDP" ] && export LD_PRELOAD="$_LDP" || unset LD_PRELOAD
PORT_27B=${PORT_27B:-8302}
SERVED_27B=${SERVED_27B:-Qwen3.6-27B}
CONFIG=${CONFIG:-$TRACE_REPO/trace_opd/configs/trace.yaml}
DATA_OUT=${DATA_OUT:-$TRACE_REPO/trace_opd/data/out}
LOG_DIR=${LOG_DIR:-$TRACE_REPO/logs}
# run 名的单一真源是实验配置的 run.name;RUN_NAME 只作显式覆盖(留空即用配置值)。
# ★ 原默认值 trace_opd_1p7b 会经 --run-name 盖掉配置,而那是个已存在的历史 run
#   (ckpt=20)——配合 resume_mode=auto 会静默从别的实验续跑。2026-07-26 修。
CFG_RUN_NAME=$("$VENV_TRAIN/bin/python" -c 'import sys,yaml; d=yaml.safe_load(open(sys.argv[1])) or {}; print((d.get("run") or {}).get("name") or "")' "$CONFIG")
RUN_NAME=${RUN_NAME:-$CFG_RUN_NAME}
[[ -n "$RUN_NAME" ]] || die "run 名为空:实验配置缺 run.name,且未给 RUN_NAME"
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
# policy 基座的真源是**实验配置的 train.model_path**;POLICY_MODEL 只作后备。
# 上面那行默认值是 162 专有路径,在 A800 上必然不存在 —— 2026-07-26 A0 首启即被它拦下。
# 兼容两种写法:配置里写 "${TRACE_MODEL_PATH}" 插值时保持原有的 env 行为(A1 等在跑的
# run 不受影响);写绝对路径时以配置为准。
_CFG_MODEL=$("$VENV_TRAIN/bin/python" -c 'import sys,yaml; d=yaml.safe_load(open(sys.argv[1])) or {}; print(((d.get("train") or {}).get("model_path") or ""))' "$CONFIG" 2>/dev/null)
case "$_CFG_MODEL" in
    ""|'${'*) : ;;                                  # 空 / 插值 ⇒ 沿用 POLICY_MODEL
    *) POLICY_MODEL=$_CFG_MODEL ;;
esac

say "== [0/4] 预检(policy=$POLICY_MODEL) =="
[[ -d $TRACE_REPO/trace_opd ]]            || die "repo 不在 $TRACE_REPO"
[[ -x $VENV_TRAIN/bin/python ]]           || die "训练 venv 缺失: $VENV_TRAIN"
[[ -x $VENV_INFER/bin/vllm ]]             || die "推理 venv 缺失: $VENV_INFER"
[[ -x $VENV_ALFWORLD/bin/python ]]        || die "alfworld venv 缺失: $VENV_ALFWORLD"
[[ -d $POLICY_MODEL ]]                    || die "policy 基座缺失: $POLICY_MODEL"
# ★ 只有 local_sidecar 才需要本地 27B;external_pool 用外部端点池,不该因它缺失而拒启。
#   (2026-07-26:A800 上用外部池启动时被这条无条件检查拦下)
# researcher 形态由**实验配置**决定(roles.researcher.mode),不再靠环境变量:
#   external_pool = 已有外部端点池(隧道),不起本地 27B,卡全留给训练
#   local_sidecar = 本机起 27B(原路径,回切预案)
# 2026-07-26 用户裁定:凡需配置的项一律进配置文件。端点地址也从配置读,
# endpoint 是唯一真源,端口由它解析(不另设 EXT_PORTS 之类的第二真源)。
_researcher_mode() {
    "$VENV_TRAIN/bin/python" -c '
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])) or {}
r = ((d.get("roles") or {}).get("researcher") or {})
print(r.get("mode") or "local_sidecar")
print(r.get("endpoint") or "")
print(r.get("model") or "")
' "$1"
}
_rm="$(_researcher_mode "$CONFIG")" || die "读取 $CONFIG 的 roles.researcher 失败"
RESEARCHER_MODE=$(echo "$_rm" | sed -n 1p)
CFG_ENDPOINT=$(echo "$_rm" | sed -n 2p)
CFG_MODEL=$(echo "$_rm" | sed -n 3p)
say "researcher 形态(来自配置): $RESEARCHER_MODE"
[[ "$RESEARCHER_MODE" == "external_pool" ]] || [[ -d $MODEL_27B ]] \
    || die "researcher 模型缺失: $MODEL_27B(local_sidecar 模式需要它)"
[[ -d $ALFWORLD_DATA/json_2.1.1 ]]        || die "ALFWorld 数据缺失: $ALFWORLD_DATA"
[[ -f $TRACE_REPO/trace_opd/verl/verl/__init__.py ]] \
    || die "vendored verl 缺失(trace_opd/verl/)"
mkdir -p "$LOG_DIR" "$DATA_OUT"

# GPU 抢占防护(infra 纪律):有大额占用又不是我们的 27B → 疑他人实验。FORCE=1 放行。
if [[ "$FORCE" != "1" ]] && ! researcher_alive; then
    # 只查本 run 实际要用的卡(跟随 CUDA_VISIBLE_DEVICES),否则同机并行跑另一个
    # 实验时会把对方占用的卡误判成"他人实验"而拒绝启动。
    _CARDS=${CUDA_VISIBLE_DEVICES:-0,1}
    max_used=$(nvidia-smi -i "$_CARDS" --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
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
if [[ "$RESEARCHER_MODE" == "external_pool" ]]; then
    say "== [2/4] 远程 researcher(35B-A3B 双端点 18888/18889,经隧道)=="
    bash "$TRACE_REPO/trace_opd/scripts/tunnel_watchdog.sh" ensure || true
    RKEY=${TRACE_RESEARCHER_API_KEY:-token-abc123}
    IFS=',' read -ra _eps <<< "$CFG_ENDPOINT"
    for _ep in "${_eps[@]}"; do
        _ep="${_ep// /}"
        curl -sf --max-time 10 -H "Authorization: Bearer $RKEY" \
            "${_ep%/}/models" | grep -q "$CFG_MODEL" \
            || die "端点 $_ep 不可用或未服务 $CFG_MODEL(隧道/远端服务)。看 $LOG_DIR/tunnel_watchdog.log"
        say "  端点 $_ep OK"
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
fi   # RESEARCHER_MODE(来自实验配置)

# ---- [3] 训练环境变量 -------------------------------------------------------
# 同机并行:每个 run 起**独立** Ray 集群。verl 的 ray.init() 不带 address,
# Ray 默认会连上本机已有集群 ⇒ 第二个实验的 actor 会被调度到第一个的集群上,
# 两边抢同一批 GPU/CPU。RAY_ADDRESS=local 强制新建;temp dir 也须分开。
export RAY_ADDRESS=${RAY_ADDRESS:-local}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_${RUN_NAME}}
mkdir -p "$RAY_TMPDIR"
export ALFWORLD_WORKER_PYTHON=$VENV_ALFWORLD/bin/python
export TRACE_MODEL_PATH=$POLICY_MODEL                              # ← policy 基座 = 1.7B
export TRACE_TRAIN_FILE=$TRAIN_PARQUET
export TRACE_VAL_SEEN_FILE=$DATA_OUT/valid_seen.parquet
export TRACE_VAL_UNSEEN_FILE=$DATA_OUT/valid_unseen.parquet
if [[ "$RESEARCHER_MODE" == "external_pool" ]]; then
    # 端点/模型名都来自配置(唯一真源);仍导出 env 是因为模板型配置可能用
    # ${TRACE_RESEARCHER_ENDPOINT} 插值,写死端点的实验配置不受影响。
    export TRACE_RESEARCHER_ENDPOINT="$CFG_ENDPOINT"
    export TRACE_RESEARCHER_MODEL="$CFG_MODEL"
    export TRACE_RESEARCHER_API_KEY="${TRACE_RESEARCHER_API_KEY:-token-abc123}"
    # ★ 不再 sed 派生:实验配置是完整副本,researcher 段已自足。
    #   再改写一次正是 v2.1-fix 要消灭的"配置真源被悄悄丢弃"。
else
    export TRACE_RESEARCHER_ENDPOINT=http://127.0.0.1:$PORT_27B/v1
    export TRACE_RESEARCHER_MODEL=$SERVED_27B
fi
# 配置:每个实验在 configs/experiments/ 下有一份【完整副本】(config_tool new 从
# configs/trace.yaml 模板复制),它就是该实验的全部定义。用 CONFIG 指向它:
#   CONFIG=$TRACE_REPO/trace_opd/configs/experiments/162-1p7b-a3.yaml \
#   RUN_NAME=trace_opd_1p7b_a3 bash scripts/run_experiment_1p7b.sh
# 不同臂务必配不同 run 名(臂进 objective fingerprint,同名 resume 会被 launcher 拦下)。
# 上面 REMOTE_RESEARCHER 那段 sed 派生可由实验配置里的 roles.researcher.mode 取代;
# 保留是为了不打断 CD-006 正在用的启动方式,下轮统一。

[[ -z "${SWANLAB_API_KEY:-}" ]] && say "提示:SWANLAB_API_KEY 未设,曲线走 offline 模式"

# ---- [4] 启动训练(nohup + 存活确认)---------------------------------------
say "== [3/4] launcher(preflight → verl)=="
if [[ "$SMOKE" == "1" ]]; then
    RUN_TAG=smoke; EFF_RUN="${RUN_NAME}_smoke"
    launch_args=(--config "$CONFIG" --run-name "$EFF_RUN" "$@")
    extra_overrides=("trainer.total_training_steps=$SMOKE_STEPS")
else
    RUN_TAG=full; EFF_RUN="$RUN_NAME"
    # "$@" 透传:--set section.key=value(配置覆盖,可重复)走这里
    launch_args=(--config "$CONFIG" --run-name "$EFF_RUN" "$@")
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
