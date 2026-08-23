#!/bin/bash
# ALFWorld 单臂发射(2026-08-16)。= launch_single.sh + ALFWorld 的两个资产 env。
#   用法: launch_single_alf.sh <run_name> <cfg_stem> <arm> <cvd> <sglport> <raytag>
#   例:   launch_single_alf.sh classic_alf3b_pgs alfworld_3b pgs 4,5,6,7 34000 alf3b
#   干跑: DRY_RUN=1 launch_single_alf.sh ...(不写盘、不联网,只算指纹与命令)
#   ⚠ run_name 必须显式给(不可从 cfg_stem 推导)——resume 靠它找目录。
NAME=${1:?run_name}; CFG=${2:?cfg_stem}; ARM=${3:?arm}; CVD=${4:?cvd}; PORT=${5:?sglport}; TAG=${6:?raytag}
W=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM
source /fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/env_pgs_sglang_a800.sh
export PYTHONPATH=$W/src:$W/verl      # ★ 必须在 source 之后(入口脚本会覆写 PYTHONPATH)
PYX=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD/env_sglang_train/bin/python
export TRACE_35B_KEY=token-abc123
export SWANLAB_API_KEY=${SWANLAB_API_KEY:?export 真值后再发车(密钥不入库;A800 部署副本内联真值)}
# ★ ALFWorld 专属:配置里 alfworld.data_root / worker_python 是 ${} 插值,
#   launch_single.sh(为 SearchQA/Math 写的)没有这两个 export ⇒ 必须在此补。
export ALFWORLD_DATA=/fs/fast/u2021000902/wushang/workspace/data/alfworld_data
export ALFWORLD_WORKER_PYTHON=/fs/fast/u2021000902/wushang/workspace/alfworld-venv/bin/python
[ -d "$ALFWORLD_DATA/json_2.1.1" ] || { echo "FATAL alfworld 数据缺失: $ALFWORLD_DATA"; exit 1; }
[ -x "$ALFWORLD_WORKER_PYTHON" ]   || { echo "FATAL alfworld worker python 缺失"; exit 1; }
cd $W

if [ "${DRY_RUN:-0}" = "1" ]; then
  exec env CUDA_VISIBLE_DEVICES=$CVD $PYX scripts/launch_train.py \
    --config $W/configs/classic/${CFG}_${ARM}.yaml --run-name $NAME --dry-run
fi

# ★ 发车前 GPU 清场校验(2026-08-14 事故:残留 50G/卡 未释放就发车,新进程
# 抢不到显存静默死亡——日志无 Traceback,只表现为主进程消失、ray worker 成僵尸)。
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
