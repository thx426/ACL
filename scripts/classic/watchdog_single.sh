#!/usr/bin/env bash
# ALFWorld 7B 双臂正式 run 看护 + 自愈。
#   启动: setsid bash watchdog_dual.sh > /dev/null 2>&1 &
#   停止: touch $W/runs/watchdog.stop
#
# 纪律(记忆 longrun-ops-discipline / monitoring-signal-discipline):
#   - 进程名不是存活判据 ⇒ 死亡判据=按 experiment_name 匹配 /proc cmdline 计数为 0
#   - 卡死判据用产物推进(rollouts 目录 mtime),不用 CPU/GPU 瞬时值
#   - 清进程只按精确 cmdline 匹配,用 python 读 /proc 避免 pkill 自匹配
#   - 重启依赖 resume_mode=auto + objective_fingerprint 守门(指纹不符会 FATAL 拒绝,
#     那是保护不是故障 —— 此时 watchdog 停手并告警,绝不 --set 绕过)
set -u
# ---- 宿主档案注入(2026-08-23 宿主无关化;字面量只在 host_*.env)----
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -z "${TRACE_HOST:-}" ]; then
  if   [ -d /fs/fast/u2021000902 ]; then TRACE_HOST=a800
  elif [ -d /data4/wushang ];       then TRACE_HOST=162
  else echo "FATAL 未知宿主:export TRACE_HOST=a800|162" >&2; exit 1; fi
fi
source "$HERE/host_${TRACE_HOST}.env"
: "${TRACE_TREE:?}" "${TRACE_PYX:?}"
W=$TRACE_TREE
# 任务对由 RUN_PREFIX 指定(classic_alf7b / classic_sqa7b …),看护逻辑不变
RUN_PREFIX=${RUN_PREFIX:?need RUN_PREFIX, e.g. classic_sqa7b}
RAY_TAG=${RAY_TAG:-$RUN_PREFIX}
CFG_STEM=${CFG_STEM:?need CFG_STEM, e.g. searchqa_7b}
TOTAL_STEPS=${TOTAL_STEPS:?need TOTAL_STEPS, e.g. 2650}
PYX=$TRACE_PYX
LOG=$W/runs/watchdog_${RUN_PREFIX}.log
STOP=$W/runs/watchdog_${RUN_PREFIX}.stop
INTERVAL=300           # 检查周期 5 分钟
STALL_MIN=45           # 无产物推进多少分钟判卡死(整步最慢约 8-15 分钟,留 3× 余量)
MAX_RESTART=20         # 单臂重启次数上限,超了停手告警(防重启风暴)

declare -A RESTARTS=([pgs]=0 [grpo]=0)   # 两键都留,ARMS 子集安全

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# 按 experiment_name 精确统计存活进程数(读 /proc,排除自身与子 shell)
count_proc() {
  $PYX - "$1" <<'PYEOF'
import os, sys
name = sys.argv[1]
n = 0
me = {os.getpid(), os.getppid()}
for p in os.listdir("/proc"):
    if not p.isdigit(): continue
    pid = int(p)
    if pid in me: continue
    try:
        c = open(f"/proc/{pid}/cmdline","rb").read().replace(b"\0",b" ").decode("utf8","replace")
    except Exception:
        continue
    if not c or "watchdog_dual" in c: continue
    # --run-name <name> 或 experiment_name=<name>,且必须是精确词(pgs 不能匹配 pgs_smoke)
    if f"--run-name {name} " in c + " " or f"experiment_name={name} " in c + " ":
        n += 1
print(n)
PYEOF
}

kill_arm() {
  $PYX - "$1" <<'PYEOF'
import os, sys, signal, time
name = sys.argv[1]
def hits():
    out=[]
    me={os.getpid(), os.getppid()}
    for p in os.listdir("/proc"):
        if not p.isdigit(): continue
        pid=int(p)
        if pid in me: continue
        try: c=open(f"/proc/{pid}/cmdline","rb").read().replace(b"\0",b" ").decode("utf8","replace")
        except Exception: continue
        if not c or "watchdog_dual" in c: continue
        if f"--run-name {name} " in c+" " or f"experiment_name={name} " in c+" ":
            out.append(pid)
    return out
for pid in hits():
    try: os.kill(pid, signal.SIGTERM)
    except Exception: pass
time.sleep(10)
for pid in hits():
    try: os.kill(pid, signal.SIGKILL)
    except Exception: pass
time.sleep(3)
print(len(hits()))
PYEOF
}

# 最近推进时间(秒)。产物(rollouts/ckpt/research)优先;启动阶段尚无产物时
# 回落到 launch.log 的 mtime —— 否则"启动即卡死"会因 age=-1 永不触发 stall 告警。
last_progress() {
  local run=$1
  local t=0
  for d in "$W/runs/$run/rollouts" "$W/runs/$run/checkpoints" "$W/runs/$run/research"; do
    [ -d "$d" ] || continue
    local m
    m=$(find "$d" -mindepth 1 -maxdepth 2 -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
    [ -n "${m:-}" ] && [ "$m" -gt "$t" ] && t=$m
  done
  if [ "$t" -eq 0 ]; then
    local lm
    lm=$(stat -c %Y "$W/runs/${run}.launch.log" 2>/dev/null)
    [ -n "${lm:-}" ] && t=$lm
  fi
  echo "$t"
}

# researcher 健康(仅机制臂):产物层判据优先——最近若干 research 会话里
# error / doc_missing 的比例。35B 服务挂掉时进程与产物照常推进,watchdog 若只看
# 进程会完全漏报,机制臂会静默降级成普通 GRPO。附带端点探活作为直接佐证。
research_health() {
  local run=$1
  $PYX - "$W/runs/$run/research" <<'PYEOF2'
import os, sys, json, glob
root = sys.argv[1]
if not os.path.isdir(root):
    print("NA 0 0"); raise SystemExit
metas = sorted(glob.glob(os.path.join(root, "*", "*", "meta.json")), key=os.path.getmtime)[-24:]
if not metas:
    print("NA 0 0"); raise SystemExit
bad = 0
for f in metas:
    try:
        m = json.load(open(f))
    except Exception:
        bad += 1; continue
    if m.get("error") or not m.get("finalized"):
        bad += 1
print(f"{bad}/{len(metas)}", bad, len(metas))
PYEOF2
}

probe_35b() {
  local ok=0
  for port in 8880 8881 8882 8883; do
    code=$(curl -s -m 12 -o /dev/null -w "%{http_code}" \
      -H "Authorization: Bearer ${TRACE_35B_KEY:-token-abc123}" \
      "http://127.0.0.1:${port}/v1/models" 2>/dev/null || echo 000)
    [ "$code" = "200" ] && ok=$((ok+1))
  done
  echo "$ok/4"
}

fire() {  # fire <arm> <cvd> <raydir> <sglport> —— 统一走 launch_classic.sh
  # (发车逻辑单一真源:env 入口/预检/GPU 清场门都在发射器里,watchdog 不再复制;
  #  raydir 参数保留签名兼容,发射器按 TAG_ARM 自建同名目录)
  local arm=$1 cvd=$2 raydir=$3 sglport=$4
  local name="${RUN_PREFIX}_${arm}"
  local out
  out=$(bash "$HERE/launch_classic.sh" "$name" "$CFG_STEM" "$arm" "$cvd" "$sglport" "$RAY_TAG" 2>&1)
  log "  → launch_classic: $out"
  echo "$out" | grep -q "^FIRED" || log "  ?? 重发未确认(预检拦截?)上行输出已录"
}

# 2026-08-14 单臂化:臂集合与 CVD/端口由环境变量给(SearchQA 只剩 pgs 用 0-3,
# Math pgs 用空出的 4-7)。ARMS 默认沿用双臂,兼容旧用法。
ARMS=${ARMS:-"pgs grpo"}
CVD_PGS=${CVD_PGS:-0,1,2,3}
CVD_GRPO=${CVD_GRPO:-4,5,6,7}
PORT_PGS=${PORT_PGS:-30000}
PORT_GRPO=${PORT_GRPO:-32000}
arm_cfg() { case $1 in
  pgs)  echo "$CVD_PGS /tmp/ray_${RAY_TAG}_pgs $PORT_PGS" ;;
  grpo) echo "$CVD_GRPO /tmp/ray_${RAY_TAG}_grpo $PORT_GRPO" ;;
esac }

# --- 单实例锁 ---
# 2026-08-13 事故:stop 检查在循环顶部而 sleep 在底部,touch stop 后 8s 就删标志,
# 旧实例最长 5 分钟才轮到检查 ⇒ 连开三个实例并存。若此时某臂崩溃,三个实例会各自
# 重启一份,抢 GPU + 端口冲突。故加锁 + 让 stop 在 sleep 期间也能及时生效。
PIDFILE=$W/runs/watchdog_${RUN_PREFIX}.pid
if [ -f "$PIDFILE" ]; then
  OLD=$(cat "$PIDFILE" 2>/dev/null || echo "")
  if [ -n "$OLD" ] && [ -d "/proc/$OLD" ] && grep -q watchdog_single "/proc/$OLD/cmdline" 2>/dev/null; then
    log "拒绝启动:已有活实例 pid=$OLD(先 touch $STOP 停它,或 kill -9 $OLD)"
    echo "watchdog 已在运行 pid=$OLD,本次不启动" >&2
    exit 1
  fi
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

log "watchdog 启动 pid=$$ (interval=${INTERVAL}s stall=${STALL_MIN}min max_restart=${MAX_RESTART})"

while true; do
  [ -f "$STOP" ] && { log "检测到 stop 文件,退出"; exit 0; }

  for arm in $ARMS; do
    name="${RUN_PREFIX}_${arm}"
    n=$(count_proc "$name")
    now=$(date +%s)
    lp=$(last_progress "$name")
    age_min=$(( lp > 0 ? (now - lp) / 60 : -1 ))

    # 完成判据:最后 ckpt 逼近 TOTAL_STEPS(留 10 步容差,save_freq 未必整除)
    done_marker=$(ls -d "$W/runs/$name/checkpoints/global_step_"* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)

    if [ "$n" -eq 0 ]; then
      # 进程全没了 —— 正常结束还是崩溃?看步数是否到位
      if [ -n "${done_marker:-}" ] && [ "$done_marker" -ge "$((TOTAL_STEPS - 10))" ]; then
        # 完成后只播报一次,之后静默(每 5 分钟重复刷屏会淹没真正的告警)
        if [ ! -f "$W/runs/${name}.done" ]; then
          log "$name: 已完成(最后 ckpt=step_$done_marker),不再看护"
          touch "$W/runs/${name}.done"
        fi
        continue
      fi
      r=${RESTARTS[$arm]}
      if [ "$r" -ge "$MAX_RESTART" ]; then
        log "$name: !! 已重启 $r 次达上限,停手等人工介入"
        continue
      fi
      log "$name: !! 进程为 0(最后 ckpt=${done_marker:-无}),判定崩溃 → 清理并重启(第 $((r+1)) 次)"
      kill_arm "$name" > /dev/null
      read -r cvd raydir port <<< "$(arm_cfg "$arm")"
      rm -rf "$raydir" 2>/dev/null
      fire "$arm" "$cvd" "$raydir" "$port"
      RESTARTS[$arm]=$((r+1))
      sleep 60
    elif [ "$age_min" -ge "$STALL_MIN" ]; then
      # 进程活着但长时间无产物推进 —— 只告警,不自动重启(误杀代价高于等待)
      log "$name: ?? 进程 $n 个存活,但产物已 ${age_min} 分钟无推进(阈值 ${STALL_MIN})——疑似卡死,需人工确认"
    else
      extra=""
      if [ "$arm" = "pgs" ]; then
        read -r rh rbad rtot <<< "$(research_health "$name")"
        extra=" research_bad=$rh"
        if [ "$rtot" -ge 8 ] && [ "$rbad" -gt $(( rtot / 2 )) ]; then
          up=$(probe_35b)
          log "$name: !! researcher 异常率 $rh(最近 $rtot 个会话),35B 端点存活 $up/2 \
—— 机制可能已静默降级为普通 GRPO,需人工确认(不自动重启:训练本身仍健康)"
        fi
      fi
      log "$name: ok proc=$n 最近推进 ${age_min}min 前 ckpt=${done_marker:-无} 重启数=${RESTARTS[$arm]}$extra"
    fi
  done

  # 分片 sleep:stop 最迟 10s 生效(整段 sleep 会让 stop 拖到 5 分钟后才响应)
  waited=0
  while [ "$waited" -lt "$INTERVAL" ]; do
    [ -f "$STOP" ] && break
    sleep 10
    waited=$((waited + 10))
  done
done
