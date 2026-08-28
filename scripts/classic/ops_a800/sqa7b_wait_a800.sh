#!/usr/bin/env bash
# A800:等目标训练步产物出现(rollouts 目录判据——driver 日志无指标行),失败指纹早报;最长 60min。
# 用法:STEP=21 bash sqa7b_wait_a800.sh   (默认 STEP=1;续跑场景传 ckpt 步+1)
R=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM/runs; RUN=classic_sqa7b_pgs; L=$R/$RUN.launch.log
STEP=${STEP:-1}; TGT=$(printf "step_%05d" "$STEP")
OFF=$(cat /fs/fast/u2021000902/wushang/ops/sqa7b_launch_offset 2>/dev/null || echo 1)
for i in $(seq 1 120); do
  sleep 30
  ERR=$(tail -n +"$OFF" "$L" 2>/dev/null | grep -cE "Traceback|FATAL|EngineDead|CUDA out of memory|illegal memory" || true); ERR=${ERR:-0}
  S=$([ -d $R/$RUN/rollouts/$TGT ] && echo Y || echo n)
  P=0; for f in /proc/[0-9]*/cmdline; do tr '\0' ' ' < $f 2>/dev/null | grep -qE -- "(--run-name|experiment_name=)classic_sqa7b_pg[s] " && P=$((P+1)); done
  RES=$(tail -n +"$OFF" "$L" | grep -oE "(Load|load|Resum|resum)[a-z]* .*global_step_[0-9]+|latest_checkpointed_iteration[^\n]{0,40}" | tail -1 | cut -c1-90)
  echo "t=$((i*30))s $TGT=$S procs=$P err=$ERR resume='$RES' $(tail -n +"$OFF" "$L" | grep -o 'Training Progress.*it\]' | tail -1 | cut -c1-70)"
  [ "$ERR" -gt 0 ] && { echo "WAIT-FAIL errors:"; tail -n +"$OFF" "$L" | grep -E "Traceback|FATAL|EngineDead|CUDA out of memory|illegal memory" | tail -3 | cut -c1-200; exit 1; }
  [ "$P" -eq 0 ] && [ "$i" -gt 4 ] && { echo "WAIT-FAIL no procs"; tail -5 "$L" | cut -c1-160; exit 1; }
  [ "$S" = Y ] && { echo "WAIT-OK $TGT produced $(date +%T)"; exit 0; }
done; echo "WAIT-TIMEOUT"; exit 2
