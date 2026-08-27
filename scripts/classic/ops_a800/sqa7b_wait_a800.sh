#!/usr/bin/env bash
# A800:等 step0 val 与 step1 产物(rollouts 目录判据——driver 日志无指标行),失败指纹早报;最长 60min。
R=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM/runs; RUN=classic_sqa7b_pgs; L=$R/$RUN.launch.log
OFF=$(cat /fs/fast/u2021000902/wushang/ops/sqa7b_launch_offset 2>/dev/null || echo 1)
for i in $(seq 1 120); do
  sleep 30
  ERR=$(tail -n +"$OFF" "$L" 2>/dev/null | grep -cE "Traceback|FATAL|EngineDead|CUDA out of memory|illegal memory")
  V=$([ -d $R/$RUN/rollouts/step_00000_val ] && echo Y || echo n); S1=$([ -d $R/$RUN/rollouts/step_00001 ] && echo Y || echo n)
  P=0; for f in /proc/[0-9]*/cmdline; do tr '\0' ' ' < $f 2>/dev/null | grep -qE -- "(--run-name|experiment_name=)classic_sqa7b_pg[s] " && P=$((P+1)); done
  echo "t=$((i*30))s val0=$V step1=$S1 procs=$P err=$ERR $(grep -o 'Training Progress.*it\]' "$L" 2>/dev/null | tail -1 | cut -c1-70)"
  [ "$ERR" -gt 0 ] && { echo "WAIT-FAIL errors:"; tail -n +"$OFF" "$L" | grep -E "Traceback|FATAL|EngineDead|CUDA out of memory|illegal memory" | tail -3 | cut -c1-200; exit 1; }
  [ "$P" -eq 0 ] && [ "$i" -gt 4 ] && { echo "WAIT-FAIL no procs"; tail -5 "$L" | cut -c1-160; exit 1; }
  [ "$S1" = Y ] && { echo "WAIT-OK step1 produced $(date +%T)"; exit 0; }
done; echo "WAIT-TIMEOUT"; exit 2
