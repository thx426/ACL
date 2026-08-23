#!/bin/bash
# 单臂发射(2026-08-14 用户裁定:SearchQA 只留 PGS 机制臂;Math PGS 用空出的 4 卡)
#   用法: launch_single.sh <run_name> <cfg_stem> <arm> <cvd> <sglport> <raytag>
#   ⚠ run_name 必须显式给(不可从 cfg_stem 推导:searchqa_7b 的 run 名是
#     classic_sqa7b_pgs,推导会得到 classic_searchqa7b_pgs 而 resume 找不到目录)
#   例: launch_single.sh classic_sqa7b_pgs   searchqa_7b pgs 0,1,2,3 30000 sqa
#       launch_single.sh classic_math7b_pgs math_7b     pgs 4,5,6,7 32000 math
NAME=${1:?run_name}; CFG=${2:?cfg_stem}; ARM=${3:?arm}; CVD=${4:?cvd}; PORT=${5:?sglport}; TAG=${6:?raytag}
W=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM
source /fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/env_pgs_sglang_a800.sh
export PYTHONPATH=$W/src:$W/verl      # ★ 必须在 source 之后
PYX=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD/env_sglang_train/bin/python
export TRACE_35B_KEY=token-abc123
export SWANLAB_API_KEY=${SWANLAB_API_KEY:?export 真值后再发车(密钥不入库;A800 部署副本内联真值)}
cd $W
# ★ 发车前 GPU 清场校验(2026-08-14 事故:上一批残留 50G/卡 未释放就发车,
# 新进程抢不到显存静默死亡——日志无 Traceback,只表现为主进程消失而 ray
# worker 成僵尸)。只检查本臂要用的卡。
for g in $(echo $CVD | tr "," " "); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  if [ "$used" -gt 500 ]; then
    echo "FATAL: GPU $g 仍占用 ${used}MiB —— 先清场再发车(勿覆盖启动)"; exit 1
  fi
done
RAYDIR=/tmp/ray_${TAG}_${ARM}; mkdir -p $RAYDIR $W/runs
setsid env CUDA_VISIBLE_DEVICES=$CVD RAY_TMPDIR=$RAYDIR \
  $PYX scripts/launch_train.py --config $W/configs/classic/${CFG}_${ARM}.yaml \
    --run-name $NAME "+actor_rollout_ref.rollout.engine_kwargs.sglang.port=${PORT}" \
  >> $W/runs/${NAME}.launch.log 2>&1 < /dev/null &
echo "FIRED $NAME cfg=${CFG}_${ARM} cvd=$CVD port=$PORT pid=$!"
