#!/usr/bin/env bash
# =============================================================================
# 实验:Qwen3.5-4B · SpreadsheetBench(reliable-372)· 原生 GRPO 对照臂 · 162 四卡
# 实验 ID: ssb-4b-grpo-162
# =============================================================================
#     bash run.sh dry               只打印门禁与将执行的命令
#     bash run.sh formal [STEPS]    正式(STEPS 可选,默认由 total_epochs 决定)
#     bash run.sh resume            断点续跑(run 目录须已存在)
#
# 主臂:A800 的 ssb-4b-selfres-a800 —— 同模型 4B、同数据集 372、executor 侧
#      逐键相同,差异只有「机制」与「机器档 4 卡/8 卡」。
# 配置真源:同目录 config.yaml。所有判定走 scripts/cfg_probe.py,
#          不在本脚本内联 python(heredoc 经 ssh 多层传递会被静默改写)。
# =============================================================================
set -uo pipefail

EXP_ID=ssb-4b-grpo-162
WANT_MECH=GRPO
MECH_DESC="纯 GRPO(steering/research/research_rl 全关 + grpo_loo)"

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-dry}"
STEPS="${2:-}"
case "$MODE" in dry|formal|resume) ;;
  *) echo "FATAL: mode 必须是 dry|formal|resume"; exit 2;; esac

set +u
source /data4/ws/PGS/env_ssb_4b.sh >/dev/null 2>&1
set -u
[ -n "${PGS_ROOT:-}" ] || { echo "FATAL: env_ssb_4b.sh 未生效(PGS_ROOT 空)"; exit 2; }
cd "$PGS_ROOT" || { echo "FATAL: PGS_ROOT 不存在: $PGS_ROOT"; exit 2; }

CFG="$EXP_DIR/config.yaml"
[ -f "$CFG" ] || { echo "FATAL: 配置不存在: $CFG"; exit 2; }

RUN_NAME="$("$PY" scripts/cfg_probe.py "$CFG" runname 2>/dev/null)"
[ -n "$RUN_NAME" ] || { echo "FATAL: 无法从配置读取 run.name"; exit 2; }
# run 目录取**配置实收的 runs_root**,不在脚本里另拼 —— 2026-08-09 事故:
# A800 那份把 RUN_DIR 拼成 $SSB_REPO/runs 而配置里 runs_root 指向另一棵树,
# 导致"formal 拒绝覆盖"这道门恒真、形同虚设。
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

# 门4b SSB xlsx 根 + 与 A800 同源(md5 对拍;跨机可比性的前提)
G=1
for f in "${SSB_DATA:-}/dataset.json" "${SSB_DATA:-}/spreadsheet"; do
  if [ -z "$f" ] || [ ! -e "$f" ]; then G=0; fi
done
chk "SSB xlsx 根就位 (${SSB_DATA:-<未设>})" "$G"
if [ -f "${SSB_DATA:-}/dataset.json" ]; then
  MD5="$(md5sum "${SSB_DATA}/dataset.json" 2>/dev/null | cut -c1-32)"
  if [ "$MD5" = "fb563c69587e45faaa69f3a917decbda" ]; then G=1; else G=0; fi
  chk "dataset.json 与 A800 同源 (md5 ${MD5:0:12}…)" "$G"
fi

# 门5 LibreOffice —— 官方要求 7.5+(README:LibreOffice 7.5+ 或 win32com)。
#     162 系统自带 6.4.7.2 低于该线,故指定用 SSB_SOFFICE 显式覆盖。
G=0; LOVER=""
if [ -n "${SSB_SOFFICE:-}" ] && [ -x "$SSB_SOFFICE" ]; then
  LOVER="$("$SSB_SOFFICE" --version 2>/dev/null | head -1)"
  case "$LOVER" in
    *" 7."*|*" 8."*|*" 9."*|*"2"[4-9]"."*) G=1 ;;
    *) G=0 ;;
  esac
fi
chk "LibreOffice ≥7.5 (${LOVER:-<未找到 SSB_SOFFICE>})" "$G"
# find_soffice() 走 PATH,PATH 上第一个 soffice 才是真正生效的那个。
PATHV="$(command -v soffice >/dev/null 2>&1 && soffice --version 2>/dev/null | head -1)"
if [ -n "$PATHV" ] && [ "$PATHV" != "$LOVER" ]; then
  note "PATH 首个 soffice 是 ${PATHV},与 SSB_SOFFICE(${LOVER}) 不同 —— 实际生效的是前者,须把 SSB_SOFFICE 目录前置到 PATH"
fi

# 门5b 数据集必须是**已剔除 reward 恒零任务**的过滤版(用户裁定 2026-08-09)。
#      判据取 split_manifest.json 的 unreliable 标记,不看目录名也不看行数。
DS="$("$PY" scripts/cfg_probe.py "$CFG" ssbdata 2>/dev/null)"
if [ "${DS%% *}" = OK ]; then G=1; else G=0; fi
chk "数据集已剔除 reward 恒零任务 ${DS:-<失败>}" "$G"

# 门5c 剔除清单仍然有效:清单里的任务在当前环境下**仍然**不可靠。
#      若某个变可靠了(换了 LibreOffice / 修了解析),说明清单过期须重审 ——
#      否则会白白少训一批本可用的任务。
# 复核必须用**六门**口径:清单是六门定的,其中 2 个任务是被反向门(不做/做错
# 也满分)剔的、正向门本来就过 —— 用只验正向的脚本复核会把它们误判成"已变可靠"。
# 底本喂全 400 未剔除版,因为清单里的任务已不在 372 里。
AU="$("$PY" scripts/ssb_verify_reward_signal.py --data "${SSB_DATA:-}" \
      --dataset "${SSB_AUDIT_BASE:-/data3/wushang/data/ssb_all400_rebuilt}" \
      --verify-list --workers 16 2>/dev/null | tail -1)"
if [ "${AU%% *}" = OK ]; then G=1; else G=0; fi
chk "剔除清单仍有效 ${AU:-<失败>}" "$G"

# 门6 四卡空闲(162 常驻 ALFWorld 线,起 SSB 前必须先让位)
BUSY="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
if [ "${BUSY:-1}" -eq 0 ]; then G=1; else G=0; fi
chk "四卡无其他计算进程 (当前 ${BUSY:-?} 个)" "$G"

# 门7 磁盘
# runs_root 首次运行时尚不存在,df 会拿不到值 —— 回退到最近存在的父目录。
DFDIR="$RUNS_ROOT"
while [ -n "$DFDIR" ] && [ ! -d "$DFDIR" ]; do DFDIR="$(dirname "$DFDIR")"; done
AVAIL="$(df -BG --output=avail "${DFDIR:-/}" 2>/dev/null | tail -1 | tr -dc '0-9')"
if [ "${AVAIL:-0}" -ge 300 ]; then G=1; else G=0; fi
chk "磁盘余量 ${AVAIL:-?}G >= 300G" "$G"

# 门8 SwanLab 在线(只验存在性,绝不打印值)
# key 真源 = ~/.swanlab/.netrc(swanlab 既有登录态),与 run_q35_4b.sh 同法现取,
# 不复制进 env 脚本、不写入任何新文件。
if [ -z "${SWANLAB_API_KEY:-}" ] && [ -f "$HOME/.swanlab/.netrc" ]; then
  SWANLAB_API_KEY=$(awk '/password/{print $2}' "$HOME/.swanlab/.netrc" | head -1)
  export SWANLAB_API_KEY
fi
G=0
if [ -n "${SWANLAB_API_KEY:-}" ] &&
   [ "${SWANLAB_MODE:-online}" != offline ] &&
   [ "${SWANLAB_MODE:-online}" != disabled ]; then G=1; fi
chk "SWANLAB_API_KEY 已就位 且模式非 offline/disabled" "$G"

# 门9 运行时依赖文件 + SSB python 依赖
G=1; MISSING=""
for f in train/agent_loop_config.yaml scripts/launch_train.py scripts/cfg_probe.py; do
  if [ ! -f "$PGS_ROOT/$f" ]; then G=0; MISSING="$MISSING $f"; fi
done
chk "运行时依赖文件齐${MISSING:+ (缺:$MISSING)}" "$G"
if "$PY" -c "import openpyxl" >/dev/null 2>&1; then G=1; else G=0; fi
chk "openpyxl 可导入 (SSB 评分/run_python 硬依赖)" "$G"

# 门10 预算链自洽(§7.5/§7.6 口径)
BUD="$("$PY" scripts/cfg_probe.py "$CFG" budget 2>/dev/null)"
if [ "$BUD" = OK ]; then G=1; else G=0; fi
chk "预算链自洽 ${BUD:-<失败>}" "$G"

# 门11 P0 回归哨兵:断言 episode.max_tokens_per_turn 注入后的**行为**
# 探针返回布尔:1 = protocol 里实际生效的 per-turn cap 等于配置值。
# (该键曾是 dead config,配 4096 实跑 2048;当时的测试只断言"配置读到了值"
#  故没拦住,所以这里断言的是**注入后的行为**。)
PT="$("$PY" scripts/cfg_probe.py "$CFG" perturn 2>/dev/null)"
if [ "$PT" = 1 ]; then G=1; else G=0; fi
chk "episode.max_tokens_per_turn 真实注入 protocol(注入值==配置值)" "$G"

# 门12 run 目录隔离
G=1
case "$MODE" in
  formal) if [ -d "$RUN_DIR" ] || [ -f "$LOG" ]; then G=0; fi ;;
  resume) if [ ! -d "$RUN_DIR" ]; then G=0; fi ;;
esac
chk "run 目录/日志状态符合 mode=$MODE ($RUN_DIR)" "$G"

# 注:自体研究员档不检 researcher 端点 —— config.py:836 强制 backend=engine,
#    roles.researcher 被校验豁免(agent_loop.py:161 明确不得求值),162 无需 35B 隧道。

# 注:本臂 steering 全关,熵门/c 等未标定项与它无关,故不打印。

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
