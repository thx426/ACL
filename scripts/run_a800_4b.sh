#!/usr/bin/env bash
# A800-SXM 多卡 4B(FSDP1 NO_SHARD + 全 gloo,零 NVLink 流量) —— 对齐 trace_opd_v1(三件套全关),硬约束:零卡间通信
# memory: a800-no-inter-gpu-comm
set -euo pipefail
W=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD
T=$W/trace_opd
PY=$W/env_vllm/bin/python
CARDS=${CARDS:-0,1,2,3}
NGPU=$(echo $CARDS | tr "," "\n" | wc -l)
RUN_NAME=${RUN_NAME:-trace_opd_4b_a800}
say(){ echo "[a800] $*"; }

# ---- 资产门 ----
export ALFWORLD_DATA=/fs/fast/u2021000902/wushang/workspace/data/alfworld_data
export TRACE_MODEL_PATH=/fs/fast/u2021000902/wushang/model/Qwen/Qwen3-4B-Instruct-2507
[[ -d $ALFWORLD_DATA/json_2.1.1 ]] || { echo "FATAL alfworld 数据缺失"; exit 1; }
[[ -d $TRACE_MODEL_PATH ]]        || { echo "FATAL 模型缺失"; exit 1; }

# ---- 数据集与 alfworld worker ----
DATA_OUT=$T/data/out
SMOKE=${SMOKE:-0}
if [[ "$SMOKE" == "1" ]]; then export TRACE_TRAIN_FILE=$DATA_OUT/train_32.parquet
else export TRACE_TRAIN_FILE=$DATA_OUT/train_3553.parquet; fi
export TRACE_VAL_SEEN_FILE=$DATA_OUT/valid_seen.parquet
export TRACE_VAL_UNSEEN_FILE=$DATA_OUT/valid_unseen.parquet
export ALFWORLD_WORKER_PYTHON=/fs/fast/u2021000902/wushang/workspace/alfworld-venv/bin/python
for f in "$TRACE_TRAIN_FILE" "$TRACE_VAL_SEEN_FILE" "$TRACE_VAL_UNSEEN_FILE" "$ALFWORLD_WORKER_PYTHON"; do
  [[ -e "$f" ]] || { echo "FATAL 缺失: $f"; exit 1; }
done
say "数据集: $(basename $TRACE_TRAIN_FILE)"

# ---- 单卡硬约束 ----
export CUDA_VISIBLE_DEVICES=$CARDS
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# CD-006 2026-07-25:vLLM EngineCore↔worker 的 RPC 超时默认仅 10s,rollout 并发~160 +
# 序列已膨胀到 ~8k token + chunked prefill 每次仅 2048 ⇒ GPU 队列深时一次 sample_tokens
# 可等超 10s,EngineCore 便判定引擎已死并自杀(step-9 事故:GPU5 实例死亡→任务永久 hang→
# 主循环死锁)。该机 nvidia-smi 正常、无 Xid、显存干净归 0 ⇒ 非硬件楔死,是超时误杀。
export VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT:-60000}
# CD-006 R11a(2026-07-25):self-teacher 打分后端。fsdp = 在训练引擎上打分,
# 彻底绕开 vLLM prompt_logprobs 的预算外分配缺陷(官方 issue #5907,rollout 尾部
# 76-81GB 尖峰与基线爬升的共同根因);vllm = 原路径。等价性已验证后默认 fsdp。
export TRACE_TEACHER_SCORING_BACKEND=${TRACE_TEACHER_SCORING_BACKEND:-vllm}
# torch allocator 超 85% 主动回收 unused blocks(纵深防御,主要护 vLLM 进程侧驻留;
# 不可用 expandable_segments——与 vLLM CuMemAllocator 冲突为实证事故)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.85}
MINFREE=$(nvidia-smi -i $CARDS --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
[[ $MINFREE -lt 70000 ]] && { echo "FATAL 某卡剩余仅 ${MINFREE}MiB,疑有残留进程"; exit 1; }
# 通信后端强制 gloo:绕开该机不可靠的 NVLink/NCCL(memory a800-no-inter-gpu-comm)
# ── 训练拓扑:MODE=noshard(默认,零 NCCL)| MODE=shard2(fsdp_size=2,最小化 NCCL)──
MODE=${MODE:-noshard}
SKIP_VAL=${SKIP_VAL:-0}   # 1=跳过训前/周期 val(测试迭代用,省 5-13 分钟)
if [[ "$MODE" == "noshard" ]]; then
  export VERL_DIST_BACKEND="cpu:gloo,cuda:gloo"   # 全程 gloo,不碰 NVLink
  export VERL_FSDP_NO_SHARD=1                     # 不分片,每步仅 1 次梯度 all_reduce
  TOKLEN=${TOKLEN:-1024}                          # NO_SHARD 每卡 64G 固定开销,激活预算极小
  SHARD_OVR="actor_rollout_ref.actor.fsdp_config.param_offload=True"
  say "拓扑=NO_SHARD + 全 gloo(零 NCCL);token 预算 $TOKLEN"
else
  # CD-006 2026-07-25:param_offload 由 False 改回 True。当初关它是为躲 MODE=noshard 下的
  # CPU OOM(NO_SHARD 每卡持完整参数,8x56GB=448GB);切到 fsdp_size=2 后参数已分片,
  # offload 量降到每卡约 4GB、8 卡共 ~32GB(机器余 792GB),前提已变而开关未复评 —— 这是
  # 显存基线高出十几 GB 的主因,也是 verl `resume weights 前不清缓存` 那个 bug 被撞穿的前提。
  # v1(跑过 step-13 起飞)正是 param_offload=True。offload 走 PCIe,不增加卡间通信。
  unset VERL_DIST_BACKEND VERL_FSDP_NO_SHARD      # 走默认 cuda:nccl
  TOKLEN=${TOKLEN:-8192}
  # fsdp_size=2 ⇒ mesh(ddp=4, fsdp=2):只在 2 卡组内分片,显存降到 ~1/2
  # CD-006 2026-07-25 用户裁定 reshard_after_forward False→True:
  #   verl 把该值直接当 zero3_enable(transformer_impl.py:397),2维 mesh 下
  #   False ⇒ _HYBRID_SHARD_ZERO2、True ⇒ HYBRID_SHARD(ZeRO-3)。
  #   ZeRO-2 的 forward 后【不 reshard】,完整 8GB 参数常驻到 backward 结束;
  #   ZeRO-3 用完即 reshard,那 8GB 消失 ⇒ 预估省 ~7GB(fsdp_size=4 只省 ~2GB,
  #   因为它切不动这个常驻 full)。代价:backward 需重新 all-gather,
  #   通信【次数】+50%(~72→~108/步),远低于 ulysses 的 80 倍。数学等价。
  # reshard_after_forward=False ⇒ ZeRO-2 式,前向 all_gather 后保留,**反向不再重复 all_gather**
  #   (这是把 NCCL 通信量砍半的关键旋钮)
  # 更大的 TOKLEN ⇒ 微批更少 ⇒ **通信轮数成比例减少**(FSDP 每微批反向都要通信)
  SHARD_OVR="actor_rollout_ref.actor.fsdp_config.fsdp_size=2 actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.entropy_from_logits_chunk_size=512"
  say "拓扑=fsdp_size2 + NCCL(仅 2 卡组内分片,ZeRO-2 免重复 all_gather);token 预算 $TOKLEN"
fi
say "用 GPU $CARDS ($NGPU 卡),后端=gloo(零 NCCL),最小剩余 ${MINFREE}MiB"

# ---- 远程 researcher(A800→zkgy→35B 二级隧道) ----
export TRACE_RESEARCHER_ENDPOINT="http://127.0.0.1:18888/v1,http://127.0.0.1:18889/v1"
export TRACE_RESEARCHER_MODEL="qwen3.6-35b-a3b"
export TRACE_RESEARCHER_API_KEY="token-abc123"
for p in 18888 18889; do
  code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $TRACE_RESEARCHER_API_KEY" \
         http://127.0.0.1:$p/v1/models || echo 000)
  [[ "$code" == "200" ]] && say "researcher 端点 $p OK" || { echo "FATAL 端点 $p 不通($code)"; exit 1; }
done

# ---- chunked-prefill 预算与打分后端联锁 ----
# 打分在 vLLM 侧时,每 chunk 物化全 vocab logits(本值 x 151936 x 2B)是 feedback P0 的 OOM
# 根因,必须降半到 2048;R11 把打分移到训练引擎后该成因消失,恢复 h3 铁稳值 4096(长序列
# prefill 的 chunk 数减半,缓解 (72) 实测的 rollout 吞吐下降)。联锁写在这里而不是靠人记住:
# 回退 backend 时预算自动跟随,漏改就会当场 OOM。
if [[ "${TRACE_TEACHER_SCORING_BACKEND:-vllm}" == "fsdp" ]]; then
  NBT=4096; NBT_WHY="R11 fsdp 打分:vLLM 不再物化打分 logits,恢复 h3 铁稳值"
else
  NBT=2048; NBT_WHY="vLLM 侧打分:必须降半压制 logits 缓冲(feedback P0 OOM 根因)"
fi
say "chunked-prefill 预算 rollout_max_num_batched_tokens=$NBT  ($NBT_WHY)"

# ---- 派生配置:单卡 + 三件套全关(= v1 基线) ----
CONFIG=$T/configs/trace.yaml
GEN=$T/configs/trace_a800_4b.gen.yaml
sed -e "s|^  n_gpus: .*|  n_gpus: $NGPU                 # A800 ${NGPU}卡 + gloo 后端(零 NCCL)|" \
    -e 's|^  rollout_tp: .*|  rollout_tp: 1             # 单卡 ⇒ TP1|' \
    -e 's|^  rollout_gpu_mem: .*|  rollout_gpu_mem: 0.40     # CD-006 2026-07-25 二次上调(0.30→0.40):step-5 崩因是训练后 torch 未归还足够物理显存,vLLM wake_up 拿不到 KV 池(cumem_allocator OOM)。降 4GB 需求换余量|' \
    -e "s|^  rollout_max_num_batched_tokens: .*|  rollout_max_num_batched_tokens: $NBT   # $NBT_WHY|" \
    -e "s|^  ppo_max_token_len_per_gpu: .*|  ppo_max_token_len_per_gpu: $TOKLEN      # 执行层,不改训练数学|" \
    -e "s|^  log_prob_max_token_len_per_gpu: .*|  log_prob_max_token_len_per_gpu: $TOKLEN  # 同上|" \
    -e 's|^  distill_block_turns: .*|  distill_block_turns: 0    # K3 关(对齐 v1)|' \
    -e 's|^  distill_loss_soft_cap: .*|  distill_loss_soft_cap: false  # K1 关(对齐 v1,回退 hard clamp)|' \
    -e 's|^  distill_kl_weight_alpha: .*|  distill_kl_weight_alpha: 0.0  # K2 关(对齐 v1;已实证 no-op)|' \
    -e 's|^  test_freq: .*|  test_freq: 10             # CD-006 2026-07-25 用户裁定开评测:每 10 步一次 val(seen+unseen 两 split, greedy)|' \
    -e 's|^  save_freq: .*|  save_freq: 5|' \
    -e '/^  researcher:/,/max_concurrency:/{
        s|^    sidecar_sleep: .*|    sidecar_sleep: false|
        s|^    dp_size: .*|    dp_size: 0|
        s|^    model: "${TRACE_RESEARCHER_MODEL}".*|&\n    api_key_env: TRACE_RESEARCHER_API_KEY|
    }' "$CONFIG" > "$GEN"
say "派生配置: $GEN"
grep -E "^  n_gpus:|^  rollout_tp:|^  rollout_gpu_mem:|^  ppo_max_token_len|^  log_prob_max_token|^  distill_block_turns:|^  distill_loss_soft_cap:|^  distill_kl_weight_alpha:|^  lr:|^  test_freq:|sidecar_sleep:|dp_size:|api_key_env:" "$GEN" | sed 's/^/    /'

# ---- 单卡显存:optimizer 状态 offload 到 CPU(468G RAM) ----
# 单卡不再有 FSDP 4 路分片:模型状态从 12GB/卡 涨到 48GB,不 offload 必 OOM
# fsdp_size=1 ⇒ device_mesh=(ddp=N, fsdp=1):参数不分片、每步仅 1 次梯度 all_reduce
# 数学等价标准 DDP;配 gloo 后端后全程零 NCCL
# fused AdamW:消除 _foreach_sqrt 的 16GB 临时张量(三次 OOM 的真凶),
# 数学与默认 foreach 实现完全等价(同一 AdamW,融合进单个 CUDA kernel),且更快。
FUSED_OVR="actor_rollout_ref.actor.optim.override_optimizer_config={fused:true}"
VAL_OVR=""
[[ "$SKIP_VAL" == "1" ]] && VAL_OVR="trainer.val_before_train=False trainer.test_freq=-1" && say "跳过 val(测试模式)"
# vLLM 引擎统计(纯观测,不改计算路径):verl 默认 disable_log_stats=True,把吞吐/KV 占用/
# preemption 全部关掉了,(72) 的 decode 机理只能靠推断。打开后可直接实测,并量化 R11 的收益。
STATS_OVR="actor_rollout_ref.rollout.disable_log_stats=False"
export EXTRA_OVERRIDES="actor_rollout_ref.actor.strategy=fsdp actor_rollout_ref.ref.strategy=fsdp actor_rollout_ref.actor.fsdp_config.optimizer_offload=True $SHARD_OVR $FUSED_OVR $VAL_OVR $STATS_OVR"

export PYTHONPATH=$T/src:$T/verl:${PYTHONPATH:-}
export SWANLAB_API_KEY=58KT7T652vKUKobNhHb50
export TRITON_PTXAS_PATH=$W/env_vllm/bin/ptxas
export PATH=$W/env_vllm/bin:$PATH
export TORCH_NCCL_BLOCKING_WAIT=1

TS=$(date +%m%d_%H%M%S)
LOG=$W/logs/train_4b_a800_${TS}.log
mkdir -p $W/logs
say "启动 → $LOG"
cd $T
nohup $PY $T/scripts/launch_train.py --config "$GEN" --run-name "$RUN_NAME" $EXTRA_OVERRIDES > "$LOG" 2>&1 &
echo "PID=$! LOG=$LOG"
