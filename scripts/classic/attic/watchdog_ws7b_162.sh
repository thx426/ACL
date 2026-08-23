#!/bin/bash
# WS7B PGS @162 看护(2026-08-23)。5min 周期,纪律同 A800 watchdog:
#   死亡 = experiment_name 匹配 /proc cmdline 计数 0 → 精确清残 → launcher 重启
#     (resume_mode=auto 续跑;objective_fingerprint 守门,不符 FATAL 是保护)
#   卡死 = rollouts 45min 无新产物 → 只告警不重启
# 停: touch /data4/wushang/classic_tree/runs/watchdog_ws7b.stop
set -u
W=/data4/wushang/classic_tree
LOG=$W/runs/watchdog_ws7b.log
STOP=$W/runs/watchdog_ws7b.stop
RESTARTS=0
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

count_procs() {
python3 - <<'PY'
import os
n = 0
for p in os.listdir('/proc'):
    if not p.isdigit(): continue
    try:
        c = open(f'/proc/{p}/cmdline', 'rb').read()
    except OSError:
        continue
    if b'experiment_name=classic_ws7b_pgs' in c:
        n += 1
print(n)
PY
}

purge_residuals() {
python3 - <<'PY'
import os, signal, time
pats = (b'classic_ws7b_pgs', b'port=34000')
victims = []
for p in os.listdir('/proc'):
    if not p.isdigit(): continue
    try:
        c = open(f'/proc/{p}/cmdline', 'rb').read()
    except OSError:
        continue
    if any(x in c for x in pats) and b'watchdog' not in c:
        victims.append(int(p))
for v in victims:
    try: os.kill(v, signal.SIGKILL)
    except OSError: pass
print(f"purged {len(victims)}")
PY
}

log "watchdog start pid=$$"
while true; do
  [ -f "$STOP" ] && { log "stop flag, exit"; exit 0; }
  n=$(count_procs)
  if [ "$n" = "0" ]; then
    RESTARTS=$((RESTARTS + 1))
    if [ "$RESTARTS" -gt 10 ]; then log "!! restart limit (10) reached, giving up"; exit 1; fi
    log "!! procs=0, purge + restart #$RESTARTS"
    purge_residuals >> "$LOG" 2>&1
    rm -rf /tmp/ray_ws7bp
    sleep 20
    bash $W/probe/launch_ws7b_pgs_162.sh go >> "$LOG" 2>&1
    sleep 600
  else
    fresh=$(find $W/runs/classic_ws7b_pgs/rollouts -maxdepth 1 -type d -newermt '-45 minutes' 2>/dev/null | wc -l)
    if [ "$fresh" = "0" ]; then log "?? no rollout progress in 45min (procs=$n) — alert only"; fi
  fi
  sleep 300
done
