#!/usr/bin/env bash
# A800:卡 4-7 各起一个 35B researcher(TP=1,8880-8883)。
# ★ 环境序言逐字对齐 serve_vllm.sh(conda env 不激活:PATH/ptxas/CPATH 手工摊开,
#   缺 ptxas ⇒ EngineCore 启动即死;memory a800-vllm-serve-pitfalls 坑1)
# ★ 两条超时链路都设(坑2:VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS 默认仅 300s 会杀服务)
set -uo pipefail
ROOT=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD
E=$ROOT/env_vllm
MODEL=/fs/fast/u2021000902/wushang/model/Qwen/Qwen3.6-35B-A3B
LOGD=$ROOT/logs_35b; mkdir -p "$LOGD"
declare -A CARD=( [8880]=4 [8881]=5 [8882]=6 [8883]=7 )
NEWLOGS=()
for port in 8880 8881 8882 8883; do
  c=${CARD[$port]}
  if curl -s -m 4 -o /dev/null http://127.0.0.1:$port/health 2>/dev/null; then echo "port $port 已在,跳过"; continue; fi
  LOG=$LOGD/vllm_35b_${port}_$(date +%m%d_%H%M).log
  setsid env \
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$c \
    PATH="$E/bin:$PATH" \
    TRITON_PTXAS_PATH="$E/bin/ptxas" \
    CUDA_HOME= \
    CPATH="$E/targets/x86_64-linux/include" \
    VLLM_RPC_TIMEOUT=1800000 \
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
    nohup "$E/bin/vllm" serve "$MODEL" \
      --served-model-name qwen3.6-35b-a3b --host 127.0.0.1 --port $port \
      --tensor-parallel-size 1 --reasoning-parser qwen3 \
      --enable-auto-tool-choice --tool-call-parser qwen3_xml \
      --language-model-only --enable-prefix-caching \
      --max-model-len 65536 --gpu-memory-utilization 0.93 \
      --max-num-seqs 32 --max-num-batched-tokens 16384 \
      --api-key token-abc123 > "$LOG" 2>&1 < /dev/null &
  NEWLOGS+=("$LOG"); echo "port $port → 卡$c pid=$! log=$(basename $LOG)"
done
echo "== 等待就绪(判据=真实 chat 往返;最长 15 分钟)=="
for i in $(seq 1 30); do
  sleep 30; ok=0; st=""
  for port in 8880 8881 8882 8883; do
    code=$(curl -s -m 10 -o /dev/null -w "%{http_code}" -H "Authorization: Bearer token-abc123" -H "Content-Type: application/json" -X POST http://127.0.0.1:$port/v1/chat/completions -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' 2>/dev/null)
    [ "$code" = 200 ] && ok=$((ok+1)); st="$st $port:$code"
  done
  echo "  t=$((i*30))s ready=$ok/4 ($st)"
  if [ "$ok" = 4 ]; then
    echo "ALL-4-READY"
    echo "== environ 实证(写在脚本里≠生效)=="
    P=""; for f in /proc/[0-9]*/cmdline; do tr "\0" " " < $f 2>/dev/null | grep -q "vllm serve.*--port 8880 " && { P=${f#/proc/}; P=${P%/cmdline}; break; }; done
    tr '\0' '\n' < /proc/$P/environ | grep -E "TRITON_PTXAS_PATH|VLLM_RPC_TIMEOUT|VLLM_EXECUTE_MODEL" | sed 's/^/    /'
    nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/    /'
    exit 0
  fi
  # 早期失败快报:只看本次拉起的日志(★ 旧版 glob 同日旧文件会误报,2026-08-27 修)
  for F in "${NEWLOGS[@]}"; do
    if grep -q "Engine core initialization failed" "$F" 2>/dev/null; then
      echo "INIT-FAILED: $F"; grep -B2 -m1 -E "Error[a-zA-Z]*:|RuntimeError|Cannot find" "$F" | tail -5 | cut -c1-160; exit 1
    fi
  done
done
echo "TIMEOUT"; exit 1
