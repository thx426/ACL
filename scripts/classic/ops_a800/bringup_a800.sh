#!/usr/bin/env bash
# A800 重启后一条龙(2026-08-27):八卡冒烟 → 发车 dry-run → 四端点 35B 拉起。任一失败即停。
# 用法:nohup setsid bash bringup_a800.sh > bringup_<ts>.log 2>&1 &
set -uo pipefail
OPS=/fs/fast/u2021000902/wushang/ops; cd "$OPS"
echo "== [A] 八卡冒烟(训练栈 python,子 shell 内 source 入口不外泄)$(date +%T)"
( source /fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/env_pgs_sglang_a800.sh >/dev/null 2>&1
  /fs/fast/u2021000902/wushang/workspace/TRACE-OPD/env_sglang_train/bin/python gpu_smoke_a800.py --all ) || { echo "BRINGUP-FAIL smoke $(date +%T)"; exit 1; }
echo "== [B] 发车 dry-run $(date +%T)"
bash "$OPS/sqa7b_launch_a800.sh" dry || { echo "BRINGUP-FAIL dry $(date +%T)"; exit 1; }
echo "== [C] 四端点 35B 拉起(干净父环境,序言在脚本内)$(date +%T)"
bash "$OPS/start_35b_x4_a800.sh" || { echo "BRINGUP-FAIL 35b $(date +%T)"; exit 1; }
echo "BRINGUP-OK $(date +%T)"
