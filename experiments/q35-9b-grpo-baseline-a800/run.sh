#!/usr/bin/env bash
# =============================================================================
# 实验:Qwen3.5-9B · 原生 GRPO 对照臂 · A800 八卡
# 实验 ID: q35-9b-grpo-baseline-a800
# =============================================================================
#     bash run.sh dry               只打印门禁与将执行的命令
#     bash run.sh formal [STEPS]    正式(STEPS 可选,默认配置的 total_epochs)
#     bash run.sh resume            断点续跑(run 目录须已存在)
#
# 对照对象:q35_9b_formal(v3.3 × external 35B researcher)
# 配置真源:同目录 config.yaml —— 与 PGS 臂逐键相同,只差三处机制开关。
# =============================================================================
set -uo pipefail

EXP_ID=q35-9b-grpo-baseline-a800
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-dry}"
case "$MODE" in dry|formal|resume) ;;
  *) echo "FATAL: mode 必须是 dry|formal|resume"; exit 2;; esac

set +u; source /fs/fast/u2021000902/wushang/workspace/TRACE-OPD-Q35/env_q35_train.sh >/dev/null 2>&1; set -u
[ -n "${PGS_ROOT:-}" ] || { echo "FATAL: env_q35_train.sh 未生效"; exit 2; }
cd "$PGS_ROOT" || { echo "FATAL: PGS_ROOT 不存在"; exit 2; }

CFG="$EXP_DIR/config.yaml"
[ -f "$CFG" ] || { echo "FATAL: 配置不存在 $CFG"; exit 2; }
RUN_NAME=q35_9b_grpo_baseline
RUN_DIR="$Q35_REPO/runs/$RUN_NAME"
LOG="$EXP_DIR/logs/${RUN_NAME}.log"
mkdir -p "$EXP_DIR/logs"

fail=0
chk() { if [ "$2" = 1 ]; then echo "  [OK]   $1"; else echo "  [FAIL] $1"; fail=1; fi; }

echo "═══ $EXP_ID · mode=$MODE ═══"
echo "=== 门禁 ==="

# 门1 配置可加载 + 指纹
FP=$("$PY" - "$CFG" <<'PY' 2>/dev/null
import sys
from trace_opd.common import config as C
print(C.objective_fingerprint(C.load_config(sys.argv[1])))
PY
)
chk "配置校验通过, fingerprint=${FP:-<失败>}" "$([ -n "$FP" ] && echo 1 || echo 0)"

# 门2 机制必须是纯 GRPO —— 断言"三处全关",防止配成半截
#     (config.py:546 已有守卫拒绝 steering=off × asym 的组合,这里再断言一次
#      是因为对照臂的科学性完全建立在"机制确实关掉"上,值得双保险)
MECH=$("$PY" - "$CFG" <<'PY' 2>/dev/null
import sys
from trace_opd.common import config as C
c = C.load_config(sys.argv[1])
ok = (not c.steering.enabled
      and c.train.adv_estimator == "grpo_loo"
      and c.steering.adv_mode in ("", "legacy_shared")
      and not c.research.enabled)
print(int(ok))
PY
)
chk "机制为纯 GRPO(steering off / grpo_loo / legacy_shared / research off)" "${MECH:-0}"

# 门3 起点必须是原始 Qwen3.5-9B
IS_RAW=0
if [ "$MODE" = resume ]; then IS_RAW=1
elif [ -f "$TRACE_MODEL_PATH/config.json" ] &&
     "$PY" -c "import json,sys;sys.exit(0 if json.load(open('$TRACE_MODEL_PATH/config.json')).get('model_type')=='qwen3_5' else 1)" &&
     ! echo "$TRACE_MODEL_PATH" | grep -qE "global_step|checkpoint|huggingface$"; then IS_RAW=1; fi
chk "起点 = 原始 Qwen3.5-9B ($TRACE_MODEL_PATH)" "$IS_RAW"

# 门4 八卡空闲
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
chk "八卡无其他计算进程 (当前 $BUSY 个)" "$([ "$BUSY" -eq 0 ] && echo 1 || echo 0)"

# 门5 磁盘
AVAIL=$(df -BG --output=avail "$Q35_REPO" 2>/dev/null | tail -1 | tr -dc "0-9")
chk "磁盘余量 ${AVAIL}G >= 300G" "$([ "${AVAIL:-0}" -ge 300 ] && echo 1 || echo 0)"

# 门6 SwanLab 在线(只验存在性,不打印值)
SW=0
[ -n "${SWANLAB_API_KEY:-}" ] && [ "${SWANLAB_MODE:-online}" != offline ] &&
  [ "${SWANLAB_MODE:-online}" != disabled ] && SW=1
chk "SWANLAB_API_KEY 已就位 且模式非 offline/disabled" "$SW"

# 门7 C++20 编译器(Qwen3.5 三层编译前置之一)
chk "CXX 指向可执行的 C++20 编译器" "$([ -x "${CXX:-}" ] && echo 1 || echo 0)"

# 门8 run 目录隔离
DIR_OK=1
case "$MODE" in
  formal) { [ -d "$RUN_DIR" ] || [ -f "$LOG" ]; } && DIR_OK=0 ;;
  resume) [ -d "$RUN_DIR" ] || DIR_OK=0 ;;
esac
chk "run 目录/日志状态符合 mode=$MODE" "$DIR_OK"

# 注:不检 researcher 端点 —— 纯 GRPO 不消费文档,35B 不参与本臂。

CMD=("$PY" "$PGS_ROOT/scripts/launch_train.py" --config "$CFG" --run-name "$RUN_NAME")
[ -n "${2:-}" ] && CMD+=("trainer.total_training_steps=$2")

echo
echo "=== 将执行 ==="
printf "  %q" "${CMD[@]}"; echo
echo "  日志: $LOG"

if [ "$fail" -ne 0 ]; then echo; echo "=> 门禁未全过,拒绝启动。"; exit 1; fi
if [ "$MODE" = dry ]; then echo; echo "=> 干跑,不启动。门禁全过。"; exit 0; fi

setsid nohup "${CMD[@]}" > "$LOG" 2>&1 < /dev/null &
NEWPID=$!
echo
echo "=> 已启动 PID=$NEWPID  (setsid)"
echo "   新连接确认: ps -o pid,pgid,etime,args -p $NEWPID"
echo "   跟日志:     tail -f $LOG"
