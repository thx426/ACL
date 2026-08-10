#!/usr/bin/env bash
# =============================================================================
# 实验:Qwen3.5-4B · ScienceWorld(ETO 口径)· 原生 GRPO 基线 · A800 八卡
# 实验 ID: sw-4b-grpo-a800
# =============================================================================
#     bash run.sh dry               只打印门禁与将执行的命令
#     bash run.sh formal [STEPS]    正式(STEPS 可选,默认由 total_epochs 决定)
#     bash run.sh resume            断点续跑(run 目录须已存在)
#
# 第二个 benchmark 的基线(第一个是 162 的 ssb-4b-grpo-162,同模型同为纯 GRPO)。
# 配置真源:同目录 config.yaml。所有判定走 scripts/cfg_probe.py,
#          不在本脚本内联 python(heredoc 经 ssh 多层传递会被静默改写)。
# =============================================================================
set -uo pipefail

EXP_ID=sw-4b-grpo-a800
WANT_MECH=GRPO
MECH_DESC="纯 GRPO(steering/research/research_rl 全关 + grpo_loo)"

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-dry}"
STEPS="${2:-}"
case "$MODE" in dry|formal|resume) ;;
  *) echo "FATAL: mode 必须是 dry|formal|resume"; exit 2;; esac

set +u
source /fs/fast/u2021000902/wushang/workspace/TRACE-OPD-SSB/env_sw_4b.sh >/dev/null 2>&1
set -u
[ -n "${PGS_ROOT:-}" ] || { echo "FATAL: env_sw_4b.sh 未生效(PGS_ROOT 空)"; exit 2; }
cd "$PGS_ROOT" || { echo "FATAL: PGS_ROOT 不存在: $PGS_ROOT"; exit 2; }

CFG="$EXP_DIR/config.yaml"
[ -f "$CFG" ] || { echo "FATAL: 配置不存在: $CFG"; exit 2; }

RUN_NAME="$("$PY" scripts/cfg_probe.py "$CFG" runname 2>/dev/null)"
[ -n "$RUN_NAME" ] || { echo "FATAL: 无法从配置读取 run.name"; exit 2; }
# run 目录取**配置实收的 runs_root**,不在脚本里另拼
RUNS_ROOT="$("$PY" scripts/cfg_probe.py "$CFG" runsroot 2>/dev/null)"
RUN_DIR="$RUNS_ROOT/$RUN_NAME"
LOG="$EXP_DIR/logs/${RUN_NAME}.log"
mkdir -p "$EXP_DIR/logs"

fail=0
G=0
chk() { if [ "$2" = 1 ]; then echo "  [OK]   $1"; else echo "  [FAIL] $1"; fail=1; fi; }
note() { echo "  [注意] $1"; }

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

# 门2 臂身份
MECH="$("$PY" scripts/cfg_probe.py "$CFG" mech 2>/dev/null)"
if [ "$MECH" = "$WANT_MECH" ]; then G=1; else G=0; fi
chk "机制为 ${MECH_DESC} [期望 $WANT_MECH,实得 ${MECH:-<失败>}]" "$G"

# 门3 起点必须是原始 Qwen3.5-4B(resume 除外)
G=0
if [ "$MODE" = resume ]; then G=1
elif [ -f "${TRACE_MODEL_PATH:-}/config.json" ]; then
  G=1; case "$TRACE_MODEL_PATH" in *global_step*|*checkpoint*|*huggingface) G=0 ;; esac
fi
chk "起点 = 原始 Qwen3.5-4B (${TRACE_MODEL_PATH:-<未设>})" "$G"

# 门4 数据就位 —— 断言配置**实收路径**(变量名可能有别名)
DF="$("$PY" scripts/cfg_probe.py "$CFG" datafiles 2>/dev/null)"
if [ "$DF" = OK ]; then G=1; else G=0; fi
chk "配置实收的 model/train/val 路径均存在 ${DF:-<失败>}" "$G"

# 门4b ETO 口径行数(train 1483 / seen 194 / unseen 211)—— 数据身份而非仅存在性
ROWS="$("$PY" scripts/sw_probe.py rows "$SCIWORLD_DATA" 2>/dev/null)"
if [ "${ROWS%% *}" = OK ]; then G=1; else G=0; fi
chk "ETO 口径行数 ${ROWS:-<失败>}" "$G"

# 门5 java —— A800 系统无 java(实查),必须由 env 前置独立 JDK 前缀。
#     worker 经 py4j launch_gateway 起 JVM,找不到 java 会在 episode 装配期才失败。
G=0; JV=""
if command -v java >/dev/null 2>&1; then
  JV="$(java -version 2>&1 | head -1)"
  case "$JV" in *\"1.8*|*\"9.*|*\"1[0-9].*|*\"2[0-9].*) G=1 ;; *) G=0 ;; esac
fi
chk "java ≥1.8 在 PATH (${JV:-<PATH 无 java>})" "$G"

# 门6 worker venv:scienceworld + py4j 可导入
G=0
if [ -x "${SCIWORLD_WORKER_PYTHON:-}" ]; then
  if "$SCIWORLD_WORKER_PYTHON" -c "import scienceworld,py4j" >/dev/null 2>&1; then G=1; fi
fi
SWV="$("${SCIWORLD_WORKER_PYTHON:-/bin/false}" -c "import scienceworld as s;print(getattr(s,'__version__','?'))" 2>/dev/null)"
chk "worker venv 可导入 scienceworld(${SWV:-?})+py4j (${SCIWORLD_WORKER_PYTHON:-<未设>})" "$G"

# 门7 真引擎冒烟 —— import 成功不是可用性判据。
#     实起一个 JVM 走生产 worker 协议,确认 ready 事件与真实 step。
SM="$("$PY" scripts/sw_probe.py smoke 2>/dev/null)"
if [ "${SM%% *}" = OK ]; then G=1; else G=0; fi
chk "真引擎冒烟(起 JVM → load → step) ${SM:-<失败>}" "$G"

# 门8 八卡空闲
BUSY="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
if [ "${BUSY:-1}" -eq 0 ]; then G=1; else G=0; fi
chk "八卡无其他计算进程 (当前 ${BUSY:-?} 个)" "$G"

# 门9 磁盘(runs_root 首次运行尚不存在,回退到最近存在的父目录)
DFDIR="$RUNS_ROOT"
while [ -n "$DFDIR" ] && [ ! -d "$DFDIR" ]; do DFDIR="$(dirname "$DFDIR")"; done
AVAIL="$(df -BG --output=avail "${DFDIR:-/}" 2>/dev/null | tail -1 | tr -dc '0-9')"
if [ "${AVAIL:-0}" -ge 300 ]; then G=1; else G=0; fi
chk "磁盘余量 ${AVAIL:-?}G >= 300G" "$G"

# 门10 SwanLab 在线(只验存在性,绝不打印值)
# key 真源 = ~/.swanlab/.netrc(既有登录态),现取现用,不写入任何新文件。
if [ -z "${SWANLAB_API_KEY:-}" ] && [ -f "$HOME/.swanlab/.netrc" ]; then
  SWANLAB_API_KEY=$(awk '/password/{print $2}' "$HOME/.swanlab/.netrc" | head -1)
  export SWANLAB_API_KEY
fi
G=0
if [ -n "${SWANLAB_API_KEY:-}" ] &&
   [ "${SWANLAB_MODE:-online}" != offline ] &&
   [ "${SWANLAB_MODE:-online}" != disabled ]; then G=1; fi
chk "SWANLAB_API_KEY 已就位 且模式非 offline/disabled" "$G"

# 门11 运行时依赖文件
G=1; MISSING=""
for f in train/agent_loop_config.yaml scripts/launch_train.py scripts/cfg_probe.py \
         src/trace_opd/envs/scienceworld/worker.py src/trace_opd/envs/scienceworld/adapter.py; do
  if [ ! -f "$PGS_ROOT/$f" ]; then G=0; MISSING="$MISSING $f"; fi
done
chk "运行时依赖文件齐${MISSING:+ (缺:$MISSING)}" "$G"

# 门12 预算链自洽(§7.5/§7.6 口径)
BUD="$("$PY" scripts/cfg_probe.py "$CFG" budget 2>/dev/null)"
if [ "$BUD" = OK ]; then G=1; else G=0; fi
chk "预算链自洽 ${BUD:-<失败>}" "$G"

# 门13 P0 回归哨兵:断言 episode.max_tokens_per_turn 注入后的**行为**
#      探针返回布尔:1 = protocol 里实际生效的 per-turn cap 等于配置值。
PT="$("$PY" scripts/cfg_probe.py "$CFG" perturn 2>/dev/null)"
if [ "$PT" = 1 ]; then G=1; else G=0; fi
chk "episode.max_tokens_per_turn 真实注入 protocol(注入值==配置值)" "$G"

# 门14 run 目录隔离
G=1
case "$MODE" in
  formal) if [ -d "$RUN_DIR" ] || [ -f "$LOG" ]; then G=0; fi ;;
  resume) if [ ! -d "$RUN_DIR" ]; then G=0; fi ;;
esac
chk "run 目录/日志状态符合 mode=$MODE ($RUN_DIR)" "$G"

echo
echo "--- 容量实测(A800 64 核,已测)---"
note "JVM 并发:64 路即达吞吐饱和(287 step/s);128→292、256→308。单 JVM 常驻 231MB"
note "本配置每步 16×16=256 episode × 30 env step ≈ 7680 step ⇒ 环境侧约 26 秒,不是瓶颈"
note "spawn_concurrency 保持 64 —— 调到 256 只多 7% 吞吐却让 load 翻 10 倍(CPU 饥饿曾饿死推理引擎)"

CMD=("$PY" scripts/launch_train.py --config "$CFG")
if [ -n "$STEPS" ]; then CMD+=("trainer.total_training_steps=$STEPS"); fi

echo
echo "--- 将执行 ---"
printf "  %q" "${CMD[@]}"; echo
echo "  run 目录: $RUN_DIR"
echo "  日志:     $LOG"

if [ "$fail" -ne 0 ]; then echo; echo "=> 门禁未全过,拒绝启动。"; exit 1; fi
if [ "$MODE" = dry ]; then echo; echo "=> 干跑模式,不启动。门禁全过。"; exit 0; fi

setsid nohup "${CMD[@]}" > "$LOG" 2>&1 < /dev/null &
NEWPID=$!
echo
echo "=> 已启动 PID=$NEWPID  (setsid,已脱离本连接)"
echo "   新连接确认: ps -o pid,pgid,etime,args -p $NEWPID"
echo "   跟日志:     tail -f $LOG"
