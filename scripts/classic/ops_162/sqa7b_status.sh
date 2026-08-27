#!/usr/bin/env bash
# 162:SQA7B 一行状态 + 增量告警。2026-08-27 降噪版:已知的 8880 GPU 抢占导致的
# researcher 降级(稳态 ~1/3 超时)静音,只在"真恶化"时报——doc_missing 突破半数
# (=不止 8880 一个端点挂)或进程/隧道/retriever 真故障。
R=/data4/wushang/classic_tree/runs; RUN=classic_sqa7b_pgs
L=$R/$RUN.launch.log; W=$R/watchdog_classic_sqa7b.log; ST=/tmp/sqa7b_status.state
[ -f $ST ] && . $ST; pE=${pE:-0}; pWDH=${pWDH:-0}; pVAL=${pVAL:-}; pDMcrit=${pDMcrit:-0}
S=$(ls $R/$RUN/rollouts 2>/dev/null | grep -v val | tail -1); S=${S:-none}
C=$(cat $R/$RUN/checkpoints/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)
E=$(grep -cE "Traceback|CUDA out of memory|EngineDead|FATAL" $L 2>/dev/null || echo 0)
# 硬故障 !! = watchdog 里"非 researcher 异常率"的告警(进程崩溃/stall 等)
WDH=$(grep '!!' $W 2>/dev/null | grep -vc '异常率' || echo 0)
P=0; for f in /proc/[0-9]*/cmdline; do tr '\0' ' ' < $f 2>/dev/null | grep -qE -- "(--run-name|experiment_name=)classic_sqa7b_pg[s] " && P=$((P+1)); done
now=$(date +%s); lp=$(find $R/$RUN/rollouts $R/$RUN/checkpoints -mindepth 1 -maxdepth 1 -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1); lp=${lp:-$(stat -c %Y $L 2>/dev/null || echo $now)}; FRESH=$(( (now-lp)/60 ))
TUN=$(curl -s -m 8 -o /dev/null -w '%{http_code}' -H 'Authorization: Bearer token-abc123' http://127.0.0.1:8880/v1/models 2>/dev/null || echo 000)
RET=$(curl -s -m 15 -XPOST http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' -d '{"queries":["ping"],"topk":1,"return_scores":true}' 2>/dev/null | grep -q '"result"' && echo up || echo DOWN)
G=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
VAL=$(grep -oE "step:[0-9]+ - val-core/[^ ]*mean@3:np.float64\([0-9.]+\)" $L 2>/dev/null | tail -1)
# 最近一步 doc_missing(机制健康:8880-only 稳态≈0.33;>0.55=真恶化)
DM=$(grep -oE "doc_missing/mean:np.float64\([0-9.]+\)" $L 2>/dev/null | tail -1 | grep -oE "[0-9.]+" | head -1); DM=${DM:-0}
DMcrit=$(awk "BEGIN{print ($DM>0.55)?1:0}" 2>/dev/null || echo 0)
echo "HEARTBEAT S=$S C=$C P=$P E=$E WDh=$WDH FRESH=${FRESH}m TUN=$TUN RET=$RET doc_miss=$DM GPUmax=${G}MiB"
[ "$E" -gt "$pE" ] && { echo "ALERT ERR-SURGE $pE→$E:"; grep -E "Traceback|CUDA out of memory|EngineDead|FATAL" $L | tail -2 | cut -c1-160; }
[ "$WDH" -gt "$pWDH" ] && { echo "ALERT WATCHDOG-HARD(非researcher):"; grep '!!' $W | grep -v '异常率' | tail -2; }
[ "$DMcrit" = 1 ] && [ "$pDMcrit" = 0 ] && echo "ALERT RESEARCHER-CRITICAL doc_missing=$DM(>0.55,疑似不止8880一端点挂,接近全灭)"
[ "$P" -eq 0 ] && [ "$C" -lt 1325 ] && echo "ALERT NO-PROCS(ckpt=$C)"
[ "$TUN" != 200 ] && echo "ALERT TUNNEL-DOWN($TUN)"
[ "$RET" = DOWN ] && echo "ALERT RETRIEVER-DOWN"
[ "$FRESH" -ge 60 ] && [ "$C" -lt 1325 ] && echo "ALERT STALL ${FRESH}min"
[ -n "$VAL" ] && [ "$VAL" != "$pVAL" ] && echo "VAL $VAL"
printf 'pE=%s\npWDH=%s\npVAL="%s"\npDMcrit=%s\n' "$E" "$WDH" "$VAL" "$DMcrit" > $ST
