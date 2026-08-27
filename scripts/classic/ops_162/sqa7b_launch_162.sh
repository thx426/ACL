#!/usr/bin/env bash
# 162:SQA7B PGS 发车序列。用法:bash sqa7b_launch_162.sh dry|go
# 前置:retriever 已在 :8000 真实可服务(预检会真查一次,不通即 FATAL)。
set -uo pipefail
MODE=${1:?dry|go}
cd /data4/wushang/classic_tree
# 密钥只在远端取值,不回显(库内脚本已脱敏为 fail-loud)
eval "$(grep -m1 'export SWANLAB_API_KEY=' probe/launch_single.sh)"
export SWANLAB_API_KEY TRACE_35B_KEY=token-abc123
export GPU_BASELINE_MIB=12000   # retriever 合法常驻 ~10G/卡(先于训练启动)

echo "== [0] retriever 真实查询 =="
curl -s -m 20 -XPOST http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' \
  -d '{"queries":["capital of France"],"topk":1,"return_scores":true}' | head -c 160; echo
echo "== [0] GPU 基线 =="; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

if [ "$MODE" = dry ]; then
  echo "== [1] DRY_RUN 三层预检 =="
  DRY_RUN=1 bash scripts/classic/launch_classic.sh classic_sqa7b_pgs searchqa_7b pgs 0,1,2,3 34000 sqa7b 2>&1 | tail -25
  exit 0
fi

L=runs/classic_sqa7b_pgs.launch.log; OFF=$(( $(wc -l < "$L" 2>/dev/null || echo 0) + 1 )); echo "$OFF" > /tmp/sqa7b_launch_offset
printf 'pE=%s\npWD=%s\npVAL=""\n' "$(grep -cE 'Traceback|CUDA out of memory|EngineDead|FATAL' "$L" 2>/dev/null || echo 0)" "$(grep -c '!!' runs/watchdog_classic_sqa7b.log 2>/dev/null || echo 0)" > /tmp/sqa7b_status.state
echo "== [1] 发车(日志偏移 $OFF)=="
bash scripts/classic/launch_classic.sh classic_sqa7b_pgs searchqa_7b pgs 0,1,2,3 34000 sqa7b 2>&1 | tail -8
echo "== [2] watchdog 布防(完成门 TOTAL_STEPS=1325)=="
RUN_PREFIX=classic_sqa7b CFG_STEM=searchqa_7b TOTAL_STEPS=1325 ARMS=pgs \
  CVD_PGS=0,1,2,3 PORT_PGS=34000 RAY_TAG=sqa7b \
  setsid bash scripts/classic/watchdog_single.sh > /dev/null 2>&1 < /dev/null &
sleep 3; echo "watchdog pid=$(cat runs/watchdog_classic_sqa7b.pid 2>/dev/null || echo '?')"
tail -3 runs/watchdog_classic_sqa7b.log 2>/dev/null
