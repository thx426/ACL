#!/usr/bin/env bash
# 162:从 seed-venv 拉起 SearchQA retriever(e5 + wiki-18,faiss fp16 四卡分片)并等真正可服务。
# 环境真源 = /data3/wushang/seed-venv(2026-08-26 排查定案;conda env `retriever` 是失败首建,勿用)。
set -uo pipefail
S=/data4/wushang/searchqa
PY=/data3/wushang/seed-venv/bin/python

probe() { curl -s -m 20 -XPOST http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' \
  -d '{"queries":["capital of France"],"topk":1,"return_scores":true}' 2>/dev/null; }
if probe | grep -q '"result"'; then echo "ALREADY-SERVING"; exit 0; fi

# 起服务前硬验:GPU faiss 可见(CPU 版/驱动不兼容会静默退化到 CPU flat 搜)
"$PY" -c "import faiss; n=faiss.get_num_gpus(); print(f'[pre] seed-venv faiss {faiss.__version__} num_gpus={n}'); exit(0 if n>=4 else 1)" || { echo "FATAL faiss 看不到 4 张 GPU,拒绝启动"; exit 1; }

LOG=$S/logs/retrieval_server_$(date +%Y%m%d_%H%M%S).log
cd "$S"
setsid nohup "$PY" retrieval_server.py \
  --index_path "$S/index/e5_Flat.index" --corpus_path "$S/corpus/wiki-18.jsonl" \
  --retriever_model "$S/model" --retriever_name e5 --topk 3 --faiss_gpu \
  > "$LOG" 2>&1 < /dev/null &
echo $! > "$S/logs/retriever.pid"
echo "[1] 已拉起 pid=$(cat $S/logs/retriever.pid) log=$LOG"

echo "[2] 等待索引加载(64.5G,最长 40 分钟)…"
for i in $(seq 1 80); do
  sleep 30
  kill -0 "$(cat $S/logs/retriever.pid)" 2>/dev/null || { echo "FATAL 进程已退出,日志尾部:"; tail -15 "$LOG"; exit 1; }
  R=$(probe)
  if echo "$R" | grep -q '"result"'; then
    echo "[3] SERVING(等待 $((i*30))s)"
    echo "    样本返回:$(echo "$R" | head -c 220)"
    echo "[4] GPU 占用:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/    /'
    echo "RETRIEVER-READY"; exit 0
  fi
  [ $((i % 4)) -eq 0 ] && echo "    …$((i*30))s: $(tail -1 "$LOG" | tr -d '\r' | cut -c1-100)"
done
echo "FATAL 40 分钟未就绪"; tail -10 "$LOG"; exit 1
