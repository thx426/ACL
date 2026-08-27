#!/usr/bin/env bash
# A800:SQA7B PGS 发车序列(2026-08-27 新架构:卡0-3 训练 + 卡4-7 四端点 35B + 162 retriever)。
# 用法:bash sqa7b_launch_a800.sh dry|go
set -uo pipefail
MODE=${1:?dry|go}
cd /fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM
eval "$(grep -m1 'export SWANLAB_API_KEY=' probe/launch_dual_sqa7b.sh)"   # 密钥只在远端取值
export SWANLAB_API_KEY TRACE_35B_KEY=token-abc123

echo "== [0] 跨机 retriever 真实查询(A800→162 直连)=="
curl -s -m 20 -XPOST http://10.77.110.162:8000/retrieve -H 'Content-Type: application/json' \
  -d '{"queries":["capital of France"],"topk":1,"return_scores":true}' | head -c 120; echo
echo "== [0] 本机四端点 35B =="
for p in 8880 8881 8882 8883; do printf "  %s:%s" $p "$(curl -s -m 8 -o /dev/null -w '%{http_code}' -H 'Authorization: Bearer token-abc123' http://127.0.0.1:$p/v1/models)"; done; echo
echo "== [0] 训练卡 0-3 清场 =="
nvidia-smi -i 0,1,2,3 --query-gpu=index,memory.used --format=csv,noheader

if [ "$MODE" = dry ]; then
  DRY_RUN=1 bash scripts/classic/launch_classic.sh classic_sqa7b_pgs searchqa_7b pgs 0,1,2,3 34000 sqa7b 2>&1 | tail -22
  exit ${PIPESTATUS[0]}   # ★ 透传发射器退出码(旧版恒 exit 0 会让 bringup 链误判 dry 通过)
fi
echo "== [1] 发车 =="
bash scripts/classic/launch_classic.sh classic_sqa7b_pgs searchqa_7b pgs 0,1,2,3 34000 sqa7b 2>&1 | tail -8
L=runs/classic_sqa7b_pgs.launch.log; OFF=$(( $(wc -l < "$L" 2>/dev/null || echo 0) + 1 )); echo "$OFF" > /fs/fast/u2021000902/wushang/ops/sqa7b_launch_offset
echo "== [2] watchdog 布防(完成门 1325)=="
RUN_PREFIX=classic_sqa7b CFG_STEM=searchqa_7b TOTAL_STEPS=1325 ARMS=pgs \
  CVD_PGS=0,1,2,3 PORT_PGS=34000 RAY_TAG=sqa7b \
  setsid bash scripts/classic/watchdog_single.sh > /dev/null 2>&1 < /dev/null &
sleep 3; echo "watchdog pid=$(cat runs/watchdog_classic_sqa7b.pid 2>/dev/null || echo '?')"
tail -2 runs/watchdog_classic_sqa7b.log 2>/dev/null
