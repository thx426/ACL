#!/usr/bin/env bash
# Researcher (strong model) vLLM server — 162/zkgy single-machine default.
#
# Topology (design.md §9, user rulings 2026-07-23): the ONLY reliable machine
# is zkgy (10.77.110.162, 4x80G mixed A800/A100). The 27B spans ALL 4 cards as
# vLLM-native DP2 x TP2 (one engine instance per same-type pair: A800=GPU0,1,
# A100=GPU2,3 — the mixed-card "TP within same-type pairs only" discipline
# holds), single port, internal load balancing. Versus TP2-on-2-cards this
# symmetrizes the per-card load and turns GPU2,3's idle headroom into
# researcher KV (~88GB pool, back to the h1 dedicated-server level); versus
# TP4 it keeps the discipline and the better aggregate decode throughput.
#
# TIME-SLICING with training:
#   agent-loop phase : 27B awake (researcher sessions run)
#   training phase   : verl POSTs /sleep?level=1 -> weights offload to CPU RAM
#                      (DP2 keeps TWO weight copies: ~110GB of the 503G host
#                      RAM), KV freed, FSDP gets the full cards
# Sleep endpoints require BOTH --enable-sleep-mode and VLLM_SERVER_DEV_MODE=1;
# the launcher pre-flights /is_sleeping and refuses to start without them.
# dp x sleep-mode x qwen3_5: VALIDATED on zkgy 2026-07-23 (CD-006) — /sleep
# is a broadcast across all DP ranks (4 cards 49.35->2.01GB together), wake_up
# 1.9s, idempotent. No fallback needed (design.md §9/§10 item 1).
#
# START ORDER CONTRACT: start THIS server first, verl second — both pre-allocate
# GPU memory at boot, and the per-card split below assumes this order:
#   27B GPU_MEM=0.52 (41.6GB pre-alloc + ~1.4GB measured activation overshoot)
#   verl rollout_gpu_mem=0.35 (27.3GB measured) + ~1.3GB ctx/ray  => 71.6GB,
#   leaving ~7.5GB headroom for the self-teacher scoring logits transient
#   (rollout_max_num_batched_tokens x vocab x 2B; the 0.60-era account missed
#   this term and OOMed the smoke run — feedback P0 2026-07-23, design.md §9).
#   0.55 is the audited fallback if the 27B KV pool preempts; 0.60 has no margin.
# Raise one side only by lowering the other (configs/trace.yaml comments).
#
# Cross-machine deployment (topology stays flexible): run on any host with
# SLEEP_MODE=0, keep --host 127.0.0.1, reach it via an SSH tunnel
# (infra discipline: vLLM never binds a public interface), and set
# roles.researcher.sidecar_sleep=false in trace.yaml.
#
#   zkgy: source /data3/wushang/workspace/TRACE-OPD/venv_infer/bin/activate
#         MODEL=/data3/wushang/model/Qwen/Qwen3.6-27B bash serve_researcher.sh
set -euo pipefail

MODEL=${MODEL:?path to Qwen3.6-27B}
SERVED_NAME=${SERVED_NAME:-Qwen3.6-27B}
PORT=${PORT:-8302}
TP=${TP:-2}                       # TP within a same-type pair
DP=${DP:-2}                       # one engine instance per pair (vLLM-native DP)
GPUS=${GPUS:-0,1,2,3}             # A800 pair + A100 pair (CUDA_DEVICE_ORDER=PCI_BUS_ID)
GPU_MEM=${GPU_MEM:-0.52}          # colocated default (post-OOM account, §9);
                                  # 0.85 when the machine is dedicated
SLEEP_MODE=${SLEEP_MODE:-1}       # 1 = enable /sleep /wake_up /is_sleeping (single-machine)

EXTRA_ARGS=()
if [[ "$SLEEP_MODE" == "1" ]]; then
    export VLLM_SERVER_DEV_MODE=1
    EXTRA_ARGS+=(--enable-sleep-mode)
fi
if [[ "$DP" != "1" ]]; then
    EXTRA_ARGS+=(--data-parallel-size "$DP")
fi

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUS \
vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len 65536 \
    --max-num-seqs 128 \
    --gpu-memory-utilization "$GPU_MEM" \
    --enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    "${EXTRA_ARGS[@]}"
