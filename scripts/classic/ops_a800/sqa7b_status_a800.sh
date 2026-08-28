#!/usr/bin/env bash
# A800:SQA7B 一行状态 + 增量告警(降噪版:doc_missing>0.55 才报 researcher 危急)。
R=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM/runs; RUN=classic_sqa7b_pgs
L=$R/$RUN.launch.log; W=$R/watchdog_classic_sqa7b.log; ST=/fs/fast/u2021000902/wushang/ops/sqa7b_status.state
[ -f $ST ] && . $ST; pE=${pE:-0}; pWDH=${pWDH:-0}; pVAL=${pVAL:-}; pDMcrit=${pDMcrit:-0}; pEP=${pEP:-4}
OFF=$(cat /fs/fast/u2021000902/wushang/ops/sqa7b_launch_offset 2>/dev/null || echo 1)
LL() { tail -n +"$OFF" "$L" 2>/dev/null; }
S=$(ls $R/$RUN/rollouts 2>/dev/null | grep -v val | tail -1); S=${S:-none}
C=$(cat $R/$RUN/checkpoints/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)
E=$(LL | grep -cE "Traceback|CUDA out of memory|EngineDead|FATAL" || true); E=${E:-0}   # ★ grep -c 计数 0 时退出码 1,旧版 || echo 0 会写成 "0\n0" 炸掉状态文件
WDH=$(grep '!!' $W 2>/dev/null | grep -vc '异常率' || true); WDH=${WDH:-0}
P=0; for f in /proc/[0-9]*/cmdline; do tr '\0' ' ' < $f 2>/dev/null | grep -qE -- "(--run-name|experiment_name=)classic_sqa7b_pg[s] " && P=$((P+1)); done
now=$(date +%s); lp=$(find $R/$RUN/rollouts $R/$RUN/checkpoints -mindepth 1 -maxdepth 1 -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1); lp=${lp:-$(stat -c %Y $L 2>/dev/null || echo $now)}; FRESH=$(( (now-lp)/60 ))
EP=0; for p in 8880 8881 8882 8883; do code=$(curl -s -m 8 -o /dev/null -w '%{http_code}' -H 'Authorization: Bearer token-abc123' http://127.0.0.1:$p/v1/models 2>/dev/null); [ "$code" = 200 ] && EP=$((EP+1)); done
RET=$(curl -s -m 15 -XPOST http://10.77.110.162:8000/retrieve -H 'Content-Type: application/json' -d '{"queries":["ping"],"topk":1,"return_scores":true}' 2>/dev/null | grep -q '"result"' && echo up || echo DOWN)
G=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
VAL=$(LL | grep -oE "step:[0-9]+ - val-core/[^ ]*mean@3:np.float64\([0-9.]+\)" | tail -1)
DM=$(LL | grep -oE "doc_missing/mean:np.float64\([0-9.]+\)" | tail -1 | grep -oE "[0-9.]+" | head -1); DM=${DM:-0}
DMcrit=$(awk "BEGIN{print ($DM>0.55)?1:0}" 2>/dev/null || echo 0)
echo "HEARTBEAT S=$S C=$C P=$P E=$E WDh=$WDH FRESH=${FRESH}m EP35B=$EP/4 RET=$RET doc_miss=$DM GPUmax=${G}MiB"
[ "$E" -gt "$pE" ] && { echo "ALERT ERR-SURGE $pE→$E:"; LL | grep -E "Traceback|CUDA out of memory|EngineDead|FATAL" | tail -2 | cut -c1-160; }
[ "$WDH" -gt "$pWDH" ] && { echo "ALERT WATCHDOG-HARD:"; grep '!!' $W | grep -v '异常率' | tail -2; }
[ "$DMcrit" = 1 ] && [ "$pDMcrit" = 0 ] && echo "ALERT RESEARCHER-CRITICAL doc_missing=$DM(>0.55)"
[ "$EP" -lt 4 ] && [ "$EP" != "$pEP" ] && echo "ALERT 35B-ENDPOINT-DOWN $EP/4(熔断会自动绕开,关注是否继续掉)"
[ "$P" -eq 0 ] && [ "$C" -lt 1325 ] && echo "ALERT NO-PROCS(ckpt=$C)"
[ "$RET" = DOWN ] && echo "ALERT RETRIEVER-DOWN"
[ "$FRESH" -ge 60 ] && [ "$C" -lt 1325 ] && echo "ALERT STALL ${FRESH}min"
[ -n "$VAL" ] && [ "$VAL" != "$pVAL" ] && echo "VAL $VAL"
printf 'pE=%s\npWDH=%s\npVAL="%s"\npDMcrit=%s\npEP=%s\n' "$E" "$WDH" "$VAL" "$DMcrit" "$EP" > $ST
