#!/usr/bin/env bash
# A800-SXM 多卡 4B(FSDP1 NO_SHARD + 全 gloo,零 NVLink 流量) —— 对齐 trace_opd_v1,硬约束:零卡间通信
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
# EngineCore↔worker 的 RPC 超时。默认 10s 会在 GPU 队列深时把一次 sample_tokens 误判为
# "引擎已死",该实例后续请求全灭、任务永久 hang(2026-07-25 第二次事故)。当时提到 60s,
# 但 2026-07-26 02:36 在 22k 序列 + chunked-prefill 4096 下**连 60s 也被突破**
# (`RPC call to sample_tokens timed out`),疑为 step-17 那次 45 分钟卡死的真凶。
# 提到 300s:正常 sample_tokens 远快于此,阈值只防误杀、不影响性能。
# VLLM_RPC_TIMEOUT / 打分后端 / FSDP 拓扑均已迁入配置(train.*),由 launcher 写入 env。
# CD-006 R11a(2026-07-25):self-teacher 打分后端。fsdp = 在训练引擎上打分,
# 彻底绕开 vLLM prompt_logprobs 的预算外分配缺陷(官方 issue #5907,rollout 尾部
# 76-81GB 尖峰与基线爬升的共同根因);vllm = 原路径(回退用)。
# CD-005 2026-07-26 收编时切默认 fsdp:生产冒烟已过(overlap 0.850 在 vLLM 基线带内、
# teacher_mass 0.99880、rollout 尾部尖峰 81.2→47.2GB 根除),requirements §1 定版主线。
# torch allocator 超 85% 主动回收 unused blocks(纵深防御,主要护 vLLM 进程侧驻留;
# 不可用 expandable_segments——与 vLLM CuMemAllocator 冲突为实证事故)
# ⚠ 不要在这里设 expandable_segments:True —— 2026-07-26 实测它与 vLLM 的 memory pool 互斥,
# 引擎直接起不来:`AssertionError: Expandable segments are not compatible with memory pool`
# (pytorch#147851)。本 run 依赖 vLLM sleep/wake 的 CuMemAllocator,两者不可共存。
# 抗 OOM 由「跳过 temperature 恒等复制」(617817c,省 13.5GiB@22k 序列)承担;
# ulysses_sp 在该机不可用(硬约束一,见派生配置段注释)。
# 分配器保持 PyTorch 默认。garbage_collection_threshold 亦不再设(推迟回收,未验证有益)。
MINFREE=$(nvidia-smi -i $CARDS --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
[[ $MINFREE -lt 70000 ]] && { echo "FATAL 某卡剩余仅 ${MINFREE}MiB,疑有残留进程"; exit 1; }
# 训练拓扑(fsdp_strategy / fsdp_size / param_offload / reshard_after_forward /
# fsdp_mode / dist_backend)已全部迁入 configs/profiles/machine-a800.yaml —— 它们是
# 机制参数,过去只存在于本脚本的 shell 变量里,换一条启动路径就换一套行为。
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

# ---- 配置:模板 + 机器 profile(不再 sed 派生)----
# 机器差异全部写在 configs/profiles/machine-a800.yaml(卡数/并行/显存/端点/拓扑),
# 硬约束一与二的完整说明也在那里。启动脚本不再改写配置——sed 派生曾无条件覆盖
# yaml 里的值,谁在配置文件里设了什么都会被悄悄改回默认(MIX_RL、TOKLEN 两次同型事故)。
# 消融臂用臂 profile 或 --set,并务必配不同的 --run-name(臂进 objective fingerprint,
# 同名 resume 会被 launcher 拦下):
#   bash scripts/run_a800_4b.sh --profile $T/configs/profiles/arm-a3-mixed.yaml \
#        --run-name trace_opd_4b_a3
CONFIG=$T/configs/trace.yaml
PROFILE=$T/configs/profiles/machine-a800.yaml
say "配置: 模板 $CONFIG + profile $PROFILE(解析后写入 <run_dir>/effective_config.yaml)"

# ---- 单卡显存:optimizer 状态 offload 到 CPU(468G RAM) ----
# 单卡不再有 FSDP 4 路分片:模型状态从 12GB/卡 涨到 48GB,不 offload 必 OOM
# fsdp_size=1 ⇒ device_mesh=(ddp=N, fsdp=1):参数不分片、每步仅 1 次梯度 all_reduce
# 数学等价标准 DDP;配 gloo 后端后全程零 NCCL
# fused AdamW:消除 _foreach_sqrt 的 16GB 临时张量(三次 OOM 的真凶),
# 数学与默认 foreach 实现完全等价(同一 AdamW,融合进单个 CUDA kernel),且更快。
# fused AdamW:消除 _foreach_sqrt 的 16GB 临时张量(三次 OOM 的真凶),数学等价且更快。
FUSED_OVR="actor_rollout_ref.actor.optim.override_optimizer_config={fused:true}"
# 跳过 val 用配置覆盖,不再另设 SKIP_VAL —— 它与 train.val_before_train 是两套真源:
#   bash scripts/run_a800_4b.sh --set train.val_before_train=false --set train.test_freq=-1
# vLLM 引擎统计(纯观测,不改计算路径):verl 默认 disable_log_stats=True,把吞吐/KV 占用/
# preemption 全部关掉了,(72) 的 decode 机理只能靠推断。打开后可直接实测,并量化 R11 的收益。
STATS_OVR="actor_rollout_ref.rollout.disable_log_stats=False"
export EXTRA_OVERRIDES="$FUSED_OVR $STATS_OVR"

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
# "$@" 透传给 launcher:--set section.key=value(配置覆盖,可重复)与 verl override 都走这里
nohup $PY $T/scripts/launch_train.py --config "$CONFIG" --profile "$PROFILE" \
      --run-name "$RUN_NAME" $EXTRA_OVERRIDES "$@" > "$LOG" 2>&1 &
echo "PID=$! LOG=$LOG"
