#!/usr/bin/env bash
# =============================================================================
# 实验:Qwen3.5-4B · v3.3 机制 × v4 自体研究员 × FC 协议 · 162 四卡
# 实验 ID: q35-4b-v33-selfres-think-162
# =============================================================================
# 本脚本自足:source 环境 → 九门 fail-closed 门禁 → setsid 启动。
# 手工复现只需:
#     bash experiments/q35-4b-v33-selfres-think-162/run.sh dry      # 看门禁与将执行的命令
#     bash experiments/q35-4b-v33-selfres-think-162/run.sh smoke    # 1 步冒烟(独立 run 目录)
#     bash experiments/q35-4b-v33-selfres-think-162/run.sh formal   # 正式:2 epochs = 358 步
#     bash experiments/q35-4b-v33-selfres-think-162/run.sh resume   # 断点续跑(目录必须已存在)
#
# 配置真源:与本脚本同目录的 config.yaml(实验自足,勿引用 configs/experiments/)
# =============================================================================
set -uo pipefail

EXP_ID=q35-4b-v33-selfres-think-162
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-dry}"
case "$MODE" in dry|smoke|formal|resume) ;;
  *) echo "FATAL: mode 必须是 dry|smoke|formal|resume"; exit 2;; esac

# ---- 环境入口(Qwen3.5 三层编译前置 + 本 run 身份)----
Q4B=/data4/ws/PGS/trace_opd_q35_4b
set +u; source "$Q4B/env_q35_4b.sh" >/dev/null 2>&1; set -u
[ -n "${PGS_ROOT:-}" ] || { echo "FATAL: env 入口未生效"; exit 2; }
cd "$PGS_ROOT" || { echo "FATAL: PGS_ROOT 不存在: $PGS_ROOT"; exit 2; }

CFG="$EXP_DIR/config.yaml"
[ -f "$CFG" ] || { echo "FATAL: 配置不存在: $CFG"; exit 2; }

# 2 epochs = 358 步(train 2875 条 ÷ train_batch_size 16 = 179 步/epoch)
FORMAL_STEPS=358
# run 名以 config.yaml 的 run.name 为唯一真源(下面只用于定位目录与日志)
RUN_NAME=$("$PY" scripts/cfg_probe.py "$CFG" runname 2>/dev/null)
[ -n "$RUN_NAME" ] || { echo "FATAL: 无法从配置读取 run.name"; exit 2; }
if [ "$MODE" = smoke ]; then
  # smoke 与 formal 共用 config.yaml 的 run.name ⇒ 同一 run 目录。冒烟只用于
  # 验链路/采标定数据,跑完必须清理该目录,否则 formal 的门7(目录不存在)会拦。
  STEPS=1
else
  STEPS=$FORMAL_STEPS
fi
RUN_DIR=/data4/ws/PGS/runs/$RUN_NAME
LOG="$EXP_DIR/logs/${RUN_NAME}.log"
mkdir -p "$EXP_DIR/logs"

fail=0
chk() { if [ "$2" = 1 ]; then echo "  [OK]   $1"; else echo "  [FAIL] $1"; fail=1; fi; }

echo "═══ 实验 $EXP_ID · mode=$MODE · steps=$STEPS ═══"
echo
echo "=== 配置提示(config.py 的预算主导判词)==="
"$PY" scripts/cfg_probe.py "$CFG" fingerprint 2>&1 >/dev/null | sed 's/^/  /'
echo
echo "=== 门禁 ==="

# 门1 配置可加载 + 机制指纹
FP=$("$PY" scripts/cfg_probe.py "$CFG" fingerprint 2>/dev/null)
chk "配置校验通过, objective_fingerprint=${FP:-<失败>}" "$([ -n "$FP" ] && echo 1 || echo 0)"

# 门2 起点必须是原始 Qwen3.5-4B(非任何 SFT/RL ckpt)——resume 除外
IS_RAW=0
if [ "$MODE" = resume ]; then IS_RAW=1
elif [ -f "$TRACE_MODEL_PATH/config.json" ] && [ "$(basename "$TRACE_MODEL_PATH")" = "Qwen3.5-4B" ] &&
     ! echo "$TRACE_MODEL_PATH" | grep -qE "global_step|checkpoint|huggingface$"; then IS_RAW=1; fi
chk "起点 = 原始 Qwen3.5-4B ($TRACE_MODEL_PATH)" "$IS_RAW"

# 门3 数据齐
DATA_OK=1
for f in "$TRACE_TRAIN_FILE" "$TRACE_VAL_SEEN_FILE" "$TRACE_VAL_UNSEEN_FILE" "$ALFWORLD_DATA"; do
  [ -e "$f" ] || DATA_OK=0
done
chk "训练/验证数据与 ALFWorld 数据就位" "$DATA_OK"

# 门4 GPU 空闲
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
chk "四卡无其他计算进程 (当前 $BUSY 个)" "$([ "$BUSY" -eq 0 ] && echo 1 || echo 0)"

# 门5 磁盘
AVAIL=$(df -BG --output=avail /data4 | tail -1 | tr -dc "0-9")
chk "磁盘余量 ${AVAIL}G >= 500G" "$([ "${AVAIL:-0}" -ge 500 ] && echo 1 || echo 0)"

# 门6 SwanLab 必须 online。key 就地从 .netrc 解出,只进本进程环境:
# 不落盘、不进 argv、不打印值。
if [ -z "${SWANLAB_API_KEY:-}" ] && [ -f "$HOME/.swanlab/.netrc" ]; then
  SWANLAB_API_KEY=$(awk '/password/{print $2}' "$HOME/.swanlab/.netrc" | head -1)
  export SWANLAB_API_KEY
fi
SW=0
if [ -n "${SWANLAB_API_KEY:-}" ] && [ "${SWANLAB_MODE:-online}" != offline ] &&
   [ "${SWANLAB_MODE:-online}" != disabled ]; then SW=1; fi
chk "SWANLAB_API_KEY 已就位(len=${#SWANLAB_API_KEY}) 且模式非 offline/disabled" "$SW"

# 门7 run 目录:formal 拒绝写已存在目录(防覆盖);resume 反过来要求它存在
DIR_OK=1
case "$MODE" in
  formal) [ -d "$RUN_DIR" ] && DIR_OK=0 ;;
  resume) [ -d "$RUN_DIR" ] || DIR_OK=0 ;;
esac
chk "run 目录状态符合 mode=$MODE ($RUN_DIR)" "$DIR_OK"

# 门8 单轮上限真实注入(P0 回归哨兵:断言注入后行为,不是断言配置读到了值)
PT=$("$PY" scripts/cfg_probe.py "$CFG" perturn 2>/dev/null)
chk "episode.max_tokens_per_turn 真实注入 protocol (P0 哨兵)" "${PT:-0}"

# 门9 运行时依赖文件存在(2026-08-08 事故:同步漏 train/ ⇒ 起跑 80 秒后
# FileNotFoundError: train/agent_loop_config.yaml。配置校验查不到这类
# 仓库内相对路径依赖,必须单列一门)
DEP_OK=1; MISSING=""
for f in train/agent_loop_config.yaml scripts/launch_train.py scripts/cfg_probe.py; do
  [ -f "$PGS_ROOT/$f" ] || { DEP_OK=0; MISSING="$MISSING $f"; }
done
chk "运行时依赖文件齐${MISSING:+ (缺:$MISSING)}" "$DEP_OK"

# 门10 预算链自洽(§7.5 控制器/保险丝解耦)
BUD=$("$PY" scripts/cfg_probe.py "$CFG" budget 2>/dev/null)
chk "预算链自洽 ${BUD:-<失败>}" "$([ "${BUD:-}" = "OK" ] && echo 1 || echo 0)"

# 只覆盖总步数。experiment_name / default_local_dir 一律由 config.yaml 的
# run.name 决定 —— 在此覆盖会造成身份分裂:launcher 的 run_dir 取自 run.name,
# 而 verl 侧取覆盖值,产物会散到两个目录(2026-08-08 实测)。
CMD=("$PY" scripts/launch_train.py --config "$CFG"
     "trainer.total_training_steps=$STEPS")

echo
echo "=== 将执行 ==="
printf "  %q" "${CMD[@]}"; echo
echo "  日志: $LOG"

if [ "$fail" -ne 0 ]; then echo; echo "=> 门禁未全过,拒绝启动。"; exit 1; fi
if [ "$MODE" = dry ]; then echo; echo "=> 干跑模式,不启动。门禁全过。"; exit 0; fi

setsid nohup "${CMD[@]}" > "$LOG" 2>&1 < /dev/null &
NEWPID=$!
echo
echo "=> 已启动 PID=$NEWPID  (setsid,已脱离本连接)"
echo "   新开连接确认: ps -o pid,pgid,etime,args -p $NEWPID"
echo "   跟日志:       tail -f $LOG"
