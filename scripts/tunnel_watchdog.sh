#!/usr/bin/env bash
# =============================================================================
# TRACE-OPD 隧道守护(162/zkgy)— CD-006
#
# 守护 162 →(跳板 183.174.61.200)→ 173.0.69.2:{8888,8889} 的 SSH 本地转发
# (35B-A3B 远程 LLM 端点)。隧道会因网络波动/空闲超时自断,本脚本周期做
# **端到端 HTTP 探活**(curl /v1/models 经隧道真实往返,而非只看端口监听),
# 失败即清旧进程并 setsid 重建,带瞬时抖动容忍 + 连续失败退避 + 事件日志。
#
#   bash tunnel_watchdog.sh start      # 后台启动守护(幂等,已在跑则不重复)
#   bash tunnel_watchdog.sh status     # 三层健康:ssh 进程 / 端口监听 / HTTP
#   bash tunnel_watchdog.sh stop       # 停守护(隧道保留);`stop all` 连隧道一起停
#   bash tunnel_watchdog.sh ensure     # 供 cron 兜底:不在跑才拉起
#
# 教训(2026-07-24 实测):嵌套 SSH 里 pkill -f "<含端口的模式>" 会匹配到承载
# 命令的远端 shell 自身 → 自杀 → exit 255。本脚本一律用 [s]sh 括号模式规避。
# =============================================================================
set -u

BASE=${TUNNEL_BASE:-/data3/wushang/workspace/TRACE-OPD}
LOG_DIR=$BASE/logs
STATE=$BASE/.tunnel_watchdog
WLOG=$LOG_DIR/tunnel_watchdog.log
JUMP_USER=${JUMP_USER:-yaozhiming}
JUMP_HOST=${JUMP_HOST:-183.174.61.200}
JUMP_PORT=${JUMP_PORT:-22}
TARGET=${TUNNEL_TARGET:-173.0.69.2}
TUNNELS=${TUNNELS:-"18888:8888 18889:8889"}    # 本地端口:远端端口,空格分隔
API_KEY=${TUNNEL_API_KEY:-token-abc123}
INTERVAL=${TUNNEL_INTERVAL:-20}                # 探活周期(秒)
mkdir -p "$LOG_DIR" "$STATE"

ts()  { date "+%F %T"; }
log() { echo "[$(ts)] $*" >> "$WLOG"; }

# 端到端探活:经隧道真实请求 /v1/models(唯一可信健康判据)
http_code() {
    curl -s -o /dev/null --max-time 10 -w '%{http_code}' \
         -H "Authorization: Bearer $API_KEY" \
         "http://127.0.0.1:$1/v1/models" 2>/dev/null || echo 000
}
listening()   { (ss -tln 2>/dev/null || netstat -tln 2>/dev/null) | grep -q ":$1 "; }
# [s]sh 括号模式:匹配真实 ssh 隧道进程,不会匹配含此模式字符串的 shell/pgrep 自身。
# 模式只锚 "-L <lport>:<target>:<rport>",兼容任意选项顺序/风格(实测 162 上并存
# 过 "-N -f -p 22 -o StrictHostKeyChecking=accept-new" 的手工隧道,老模式会漏)。
tunnel_pids() { pgrep -f "[s]sh .*-L .*$1:$TARGET:$2" 2>/dev/null || true; }

launch_tunnel() {  # $1=lport $2=rport;清掉同端口**所有风格**旧 ssh → 等端口真释放 → setsid 重建
    local lp=$1 rp=$2 p w=0
    for p in $(tunnel_pids "$lp" "$rp"); do kill -9 "$p" 2>/dev/null; done
    while listening "$lp" && [ $w -lt 6 ]; do sleep 1; w=$((w+1)); done   # 防 v4/v6 残留占位
    if listening "$lp"; then
        log "PORT_BUSY lport=$lp 杀净 ssh 后仍被占用(非 ssh 持有者?),跳过本轮重建"
        return 1
    fi
    setsid nohup ssh -p "$JUMP_PORT" -N \
        -o BatchMode=yes -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -o ConnectTimeout=10 \
        -L "$lp:$TARGET:$rp" "$JUMP_USER@$JUMP_HOST" \
        >> "$LOG_DIR/tunnel_$lp.log" 2>&1 < /dev/null &
    log "LAUNCH lport=$lp rport=$rp pid=$!"
}

heal() {  # $1=lport $2=rport;健康返 0,恢复失败返 1
    local lp=$1 rp=$2 code alive=no lsn=no
    code=$(http_code "$lp"); [ "$code" = 200 ] && return 0
    sleep 3                                        # 瞬时抖动容忍:3s 后复测
    code=$(http_code "$lp"); [ "$code" = 200 ] && return 0
    [ -n "$(tunnel_pids "$lp" "$rp")" ] && alive=yes
    listening "$lp" && lsn=yes
    log "UNHEALTHY lport=$lp http=$code ssh=$alive listen=$lsn -> relaunch"
    launch_tunnel "$lp" "$rp"
    sleep 6
    code=$(http_code "$lp")
    if [ "$code" = 200 ]; then log "RECOVERED lport=$lp"; return 0; fi
    log "STILL_DOWN lport=$lp http=$code(隧道已重建仍不通:跳板不可达或远端服务挂)"
    return 1
}

run() {
    if [ -f "$STATE/watchdog.pid" ]; then
        local old; old=$(cat "$STATE/watchdog.pid" 2>/dev/null || echo "")
        if [ -n "$old" ] && [ "$old" != "$$" ] && kill -0 "$old" 2>/dev/null; then
            log "DUP run 拒绝(已有 watchdog pid=$old)"; exit 1
        fi
    fi
    echo $$ > "$STATE/watchdog.pid"
    log "WATCHDOG START pid=$$ tunnels=[$TUNNELS] interval=${INTERVAL}s"
    local n=0 consec=0 bad t lp rp extra
    while :; do
        bad=0
        for t in $TUNNELS; do
            lp=${t%%:*}; rp=${t##*:}
            heal "$lp" "$rp" || bad=$((bad+1))
        done
        if [ "$bad" -gt 0 ]; then consec=$((consec+1)); else consec=0; fi
        n=$((n+1))
        [ $((n % 90)) -eq 0 ] && log "HEARTBEAT checks=$n consec_bad=$consec"   # ~每 30min
        extra=$(( consec > 0 ? ( consec*15 > 120 ? 120 : consec*15 ) : 0 ))     # 退避封顶 120s
        sleep $((INTERVAL + extra))
    done
}

status() {
    local wp="-" walive=no t lp rp pids code lsn
    [ -f "$STATE/watchdog.pid" ] && wp=$(cat "$STATE/watchdog.pid")
    [ "$wp" != "-" ] && kill -0 "$wp" 2>/dev/null && walive=yes
    echo "watchdog: pid=$wp alive=$walive  (log: $WLOG)"
    for t in $TUNNELS; do
        lp=${t%%:*}; rp=${t##*:}
        pids=$(tunnel_pids "$lp" "$rp" | tr '\n' ',' | sed 's/,$//')
        lsn=no; listening "$lp" && lsn=yes
        code=$(http_code "$lp")
        echo "tunnel $lp -> $TARGET:$rp  ssh_pid=${pids:-none} listen=$lsn http=$code"
    done
}

case "${1:-}" in
    start)
        if [ -f "$STATE/watchdog.pid" ] && kill -0 "$(cat "$STATE/watchdog.pid" 2>/dev/null)" 2>/dev/null; then
            echo "already running pid=$(cat "$STATE/watchdog.pid")"; exit 0
        fi
        setsid nohup bash "$0" run >> "$LOG_DIR/tunnel_watchdog.boot.log" 2>&1 < /dev/null &
        echo "watchdog starting (boot pid=$!)"
        sleep 3; status ;;
    run)    run ;;
    status) status ;;
    ensure)
        if [ -f "$STATE/watchdog.pid" ] && kill -0 "$(cat "$STATE/watchdog.pid" 2>/dev/null)" 2>/dev/null; then exit 0; fi
        setsid nohup bash "$0" run >> "$LOG_DIR/tunnel_watchdog.boot.log" 2>&1 < /dev/null & ;;
    stop)
        if [ -f "$STATE/watchdog.pid" ]; then
            kill "$(cat "$STATE/watchdog.pid")" 2>/dev/null
            rm -f "$STATE/watchdog.pid"; echo "watchdog stopped"
        else echo "watchdog not running"; fi
        if [ "${2:-}" = all ]; then
            for t in $TUNNELS; do
                lp=${t%%:*}; rp=${t##*:}
                for p in $(tunnel_pids "$lp" "$rp"); do kill -9 "$p" 2>/dev/null; done
            done; echo "tunnels stopped"
        fi ;;
    *) echo "usage: $0 {start|status|stop [all]|ensure}"; exit 1 ;;
esac
