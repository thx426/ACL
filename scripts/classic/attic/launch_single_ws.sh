#!/bin/bash
# WebShop 单臂发射(2026-08-16)。= launch_single.sh + WebShop 的环境前置检查。
#   用法: launch_single_ws.sh <run_name> <cfg_stem> <arm> <cvd> <sglport> <raytag>
#   例:   launch_single_ws.sh classic_ws3b_pgs webshop_3b pgs 4,5,6,7 34000 ws3b
#   干跑: DRY_RUN=1 launch_single_ws.sh ...
#
# ★ 为什么要专属脚本:WebShop 的环境是 162 上的 32 个 HTTP 分片,任何一个没起
#   来、或 option_fix 与配置不一致,都会在 factory 构造时才 fail loud —— 那时
#   已经占了 GPU、起了 ray、加载了模型,白白浪费好几分钟。这里提前拦。
NAME=${1:?run_name}; CFG=${2:?cfg_stem}; ARM=${3:?arm}; CVD=${4:?cvd}; PORT=${5:?sglport}; TAG=${6:?raytag}
W=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM
YAML=$W/configs/classic/${CFG}_${ARM}.yaml
[ -f "$YAML" ] || { echo "FATAL 配置不存在: $YAML"; exit 1; }

source /fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/env_pgs_sglang_a800.sh
export PYTHONPATH=$W/src:$W/verl      # ★ 必须在 source 之后(入口脚本会覆写)
PYX=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD/env_sglang_train/bin/python
export TRACE_35B_KEY=token-abc123
export SWANLAB_API_KEY=${SWANLAB_API_KEY:?export 真值后再发车(密钥不入库;A800 部署副本内联真值)}
cd $W

# ---- WebShop 环境前置(逐分片,fail loud)-------------------------------
echo "[pre] 校验 WebShop 分片与 reward 口径…"
$PYX - "$YAML" <<'PYEOF' || exit 1
import sys, json, urllib.request, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
ws = cfg.get("webshop") or {}
urls = [u.strip().rstrip("/") for u in str(ws.get("service_url", "")).split(",") if u.strip()]
want_fix = bool(ws.get("fix_option_matching", True))
if not urls:
    print("FATAL webshop.service_url 为空"); sys.exit(1)
metas, bad = {}, []
for u in urls:
    try:
        with urllib.request.urlopen(u + "/meta", timeout=8) as r:
            metas[u] = json.load(r)
    except Exception as e:
        bad.append(f"{u}: {type(e).__name__}")
if bad:
    print(f"FATAL {len(bad)}/{len(urls)} 个分片不可达: {bad[:4]}"); sys.exit(1)
ng = {m["num_goals"] for m in metas.values()}
if len(ng) != 1:
    print(f"FATAL 分片 num_goals 不一致 {ng} —— goal_idx 语义跨分片错位"); sys.exit(1)
fixes = {u: m.get("option_fix") for u, m in metas.items()}
if any(v is None for v in fixes.values()):
    print("FATAL 分片 /meta 无 option_fix 字段 —— 旧版服务端,请重新部署"); sys.exit(1)
mismatch = {u: v for u, v in fixes.items() if bool(v) is not want_fix}
if mismatch:
    print(f"FATAL fix_option_matching={want_fix} 与服务端不一致: "
          f"{list(mismatch)[:4]} —— reward 口径分歧,拒绝发车"); sys.exit(1)
slots = sum(m["pool"] for m in metas.values())
need = int((cfg.get("train") or {}).get("train_batch_size", 0)) * \
       int((cfg.get("train") or {}).get("rollout_n", 0))
print(f"[pre] OK 分片 {len(urls)} 个 / 总槽位 {slots} / 需求 {need} / "
      f"goals {ng.pop()} / option_fix={want_fix}")
if need and slots < need:
    print(f"[pre] ⚠ 槽位 {slots} < 并发 episode {need} —— episode 会分批排队,"
          f"环境侧墙钟显著变长(实测 4 分片跑 128 并发是 +323%)")
PYEOF

# ---- 数据存在性 --------------------------------------------------------
$PYX - "$YAML" <<'PYEOF' || exit 1
import sys, yaml
from pathlib import Path
t = (yaml.safe_load(open(sys.argv[1], encoding="utf-8")).get("train") or {})
miss = [p for p in [t.get("train_file")] + list(t.get("val_files") or [])
        if p and not Path(p).exists()]
if miss:
    print(f"FATAL 数据缺失: {miss}"); sys.exit(1)
print("[pre] OK 数据文件齐")
PYEOF

if [ "${DRY_RUN:-0}" = "1" ]; then
  exec env CUDA_VISIBLE_DEVICES=$CVD $PYX scripts/launch_train.py \
    --config $YAML --run-name $NAME --dry-run
fi

# ---- GPU 清场(2026-08-14 事故:残留显存未释放就发车 ⇒ 新进程静默死亡)----
for g in $(echo $CVD | tr "," " "); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  if [ "$used" -gt 500 ]; then
    echo "FATAL: GPU $g 仍占用 ${used}MiB —— 先清场再发车(勿覆盖启动)"; exit 1
  fi
done

RAYDIR=/tmp/ray_${TAG}_${ARM}; mkdir -p $RAYDIR $W/runs
setsid env CUDA_VISIBLE_DEVICES=$CVD RAY_TMPDIR=$RAYDIR \
  $PYX scripts/launch_train.py --config $YAML \
    --run-name $NAME "+actor_rollout_ref.rollout.engine_kwargs.sglang.port=${PORT}" \
  >> $W/runs/${NAME}.launch.log 2>&1 < /dev/null &
echo "FIRED $NAME cfg=${CFG}_${ARM} cvd=$CVD port=$PORT pid=$!"
