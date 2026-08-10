#!/usr/bin/env bash
# =============================================================================
# 实验:Qwen3.5-9B · SpreadsheetBench(verified 400)· v3.3 × v4 自体研究员(完整自进化机制) · A800 八卡
# 实验 ID: ssb-9b-selfres-a800
# =============================================================================
#     bash run.sh dry               只打印门禁与将执行的命令
#     bash run.sh formal [STEPS]    正式(STEPS 可选,默认由 total_epochs 决定)
#     bash run.sh resume            断点续跑(run 目录须已存在)
#
# 对照臂:experiments/ssb-9b-grpo-baseline-a800(纯 GRPO,逐键相同只关机制)
# 本臂 = 完整机制:researcher 是 9B policy 本体(engine 后端),steering 走 v3.3 六模块。
# 配置真源:同目录 config.yaml
#
# 所有判定一律走 scripts/cfg_probe.py 探针,不在本脚本内联 python ——
# 内联 heredoc 经 ssh/引号多层传递极易被静默改写(2026-08-09 实测连踩三次)。
# =============================================================================
set -uo pipefail

EXP_ID=ssb-9b-selfres-a800
WANT_MECH=V33_SELFRES
MECH_DESC="v3.3 全套 + v4 自体(M1 asym / M2 α 有限期 / M3 W off / M4 熵门 + engine researcher)"

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-dry}"
STEPS="${2:-}"
case "$MODE" in dry|formal|resume) ;;
  *) echo "FATAL: mode 必须是 dry|formal|resume"; exit 2;; esac

set +u
source /fs/fast/u2021000902/wushang/workspace/TRACE-OPD-SSB/env_ssb.sh >/dev/null 2>&1
set -u
[ -n "${PGS_ROOT:-}" ] || { echo "FATAL: env_ssb.sh 未生效(PGS_ROOT 空)"; exit 2; }
cd "$PGS_ROOT" || { echo "FATAL: PGS_ROOT 不存在: $PGS_ROOT"; exit 2; }

CFG="$EXP_DIR/config.yaml"
[ -f "$CFG" ] || { echo "FATAL: 配置不存在: $CFG"; exit 2; }

RUN_NAME="$("$PY" scripts/cfg_probe.py "$CFG" runname 2>/dev/null)"
[ -n "$RUN_NAME" ] || { echo "FATAL: 无法从配置读取 run.name"; exit 2; }
RUN_DIR="$SSB_REPO/runs/$RUN_NAME"
LOG="$EXP_DIR/logs/${RUN_NAME}.log"
mkdir -p "$EXP_DIR/logs"

fail=0
G=0
chk() { if [ "$2" = 1 ]; then echo "  [OK]   $1"; else echo "  [FAIL] $1"; fail=1; fi; }

echo "=== $EXP_ID · mode=$MODE ==="
echo
echo "--- 配置提示 ---"
"$PY" scripts/cfg_probe.py "$CFG" fingerprint 2>&1 >/dev/null | sed 's/^/  /'
echo
echo "--- 门禁 ---"

# 门1 配置可加载 + 机制指纹
FP="$("$PY" scripts/cfg_probe.py "$CFG" fingerprint 2>/dev/null)"
if [ -n "$FP" ]; then G=1; else G=0; fi
chk "配置校验通过, objective_fingerprint=${FP:-<失败>}" "$G"

# 门2 臂身份 —— 对照实验的科学性完全建立在"机制确实按预期开/关"上
MECH="$("$PY" scripts/cfg_probe.py "$CFG" mech 2>/dev/null)"
if [ "$MECH" = "$WANT_MECH" ]; then G=1; else G=0; fi
chk "机制为 ${MECH_DESC} [期望 $WANT_MECH,实得 ${MECH:-<失败>}]" "$G"

# 门3 起点必须是原始 Qwen3.5-9B(resume 除外)
G=0
if [ "$MODE" = resume ]; then
  G=1
elif [ -f "${TRACE_MODEL_PATH:-}/config.json" ]; then
  G=1
  case "$TRACE_MODEL_PATH" in *global_step*|*checkpoint*|*huggingface) G=0 ;; esac
fi
chk "起点 = 原始 Qwen3.5-9B (${TRACE_MODEL_PATH:-<未设>})" "$G"

# 门4 数据就位 —— 断言**配置实收路径**(变量名可能有别名,路径存在才可信)
DF="$("$PY" scripts/cfg_probe.py "$CFG" datafiles 2>/dev/null)"
if [ "$DF" = OK ]; then G=1; else G=0; fi
chk "配置实收的 model/train/val 路径均存在 ${DF:-<失败>}" "$G"

# 门4b SSB 环境自有资产(xlsx 根;不经 train 段,故单列)
G=1
for f in "${SSB_DATA:-}/dataset.json" "${SSB_DATA:-}/spreadsheet"; do
  if [ -z "$f" ] || [ ! -e "$f" ]; then G=0; fi
done
chk "SSB xlsx 根就位 (${SSB_DATA:-<未设>})" "$G"

# 门5 LibreOffice —— README:评分前公式重算刚需,soffice 缺失装配期即死
G=0
if [ -n "${SSB_SOFFICE:-}" ] && [ -x "$SSB_SOFFICE" ]; then
  if "$SSB_SOFFICE" --version >/dev/null 2>&1; then G=1; fi
fi
chk "LibreOffice 可执行 (${SSB_SOFFICE:-<未设>})" "$G"

# 门6 八卡空闲
BUSY="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
if [ "${BUSY:-1}" -eq 0 ]; then G=1; else G=0; fi
chk "八卡无其他计算进程 (当前 ${BUSY:-?} 个)" "$G"

# 门7 磁盘
AVAIL="$(df -BG --output=avail "$SSB_REPO" 2>/dev/null | tail -1 | tr -dc '0-9')"
if [ "${AVAIL:-0}" -ge 300 ]; then G=1; else G=0; fi
chk "磁盘余量 ${AVAIL:-?}G >= 300G" "$G"

# 门8 SwanLab 在线(只验存在性,绝不打印值)
G=0
if [ -n "${SWANLAB_API_KEY:-}" ] &&
   [ "${SWANLAB_MODE:-online}" != offline ] &&
   [ "${SWANLAB_MODE:-online}" != disabled ]; then G=1; fi
chk "SWANLAB_API_KEY 已就位 且模式非 offline/disabled" "$G"

# 门9 运行时依赖文件(2026-08-08 事故:同步漏 train/ ⇒ 起跑 80 秒后 FileNotFound)
G=1; MISSING=""
for f in train/agent_loop_config.yaml scripts/launch_train.py scripts/cfg_probe.py; do
  if [ ! -f "$PGS_ROOT/$f" ]; then G=0; MISSING="$MISSING $f"; fi
done
chk "运行时依赖文件齐${MISSING:+ (缺:$MISSING)}" "$G"

# 门10 预算链自洽(§7.5/§7.6 口径)
BUD="$("$PY" scripts/cfg_probe.py "$CFG" budget 2>/dev/null)"
if [ "$BUD" = OK ]; then G=1; else G=0; fi
chk "预算链自洽 ${BUD:-<失败>}" "$G"

# 门11 P0 回归哨兵:断言 episode.max_tokens_per_turn 注入后的行为,
#      而非断言配置读到了值(该键曾是 dead config,当时测试只断言后者故未拦住)
PT="$("$PY" scripts/cfg_probe.py "$CFG" perturn 2>/dev/null)"
if [ "$PT" = 1 ]; then G=1; else G=0; fi
chk "episode.max_tokens_per_turn 真实注入 protocol" "$G"

# 门12 run 目录隔离(formal 拒绝覆盖;resume 反过来要求存在)
G=1
case "$MODE" in
  formal) if [ -d "$RUN_DIR" ] || [ -f "$LOG" ]; then G=0; fi ;;
  resume) if [ ! -d "$RUN_DIR" ]; then G=0; fi ;;
esac
chk "run 目录/日志状态符合 mode=$MODE ($RUN_DIR)" "$G"

# 注:不检 external researcher 端点 —— 本臂 researcher 是 policy 本体(engine)。

CMD=("$PY" scripts/launch_train.py --config "$CFG")
if [ -n "$STEPS" ]; then CMD+=("trainer.total_training_steps=$STEPS"); fi

echo
echo "--- 将执行 ---"
printf "  %q" "${CMD[@]}"; echo
echo "  日志: $LOG"

if [ "$fail" -ne 0 ]; then echo; echo "=> 门禁未全过,拒绝启动。"; exit 1; fi
if [ "$MODE" = dry ]; then echo; echo "=> 干跑模式,不启动。门禁全过。"; exit 0; fi

setsid nohup "${CMD[@]}" > "$LOG" 2>&1 < /dev/null &
NEWPID=$!
echo
echo "=> 已启动 PID=$NEWPID  (setsid,已脱离本连接)"
echo "   新连接确认: ps -o pid,pgid,etime,args -p $NEWPID"
echo "   跟日志:     tail -f $LOG"
