#!/bin/bash
# WS7B PGS 机制臂 @162 发射器(2026-08-23 用户裁定:162 四卡跑机制臂,
# 35B researcher 经 A800 反向隧道 127.0.0.1:8880-8882)。
# = A800 probe/launch_single_ws.sh 的 162 适配(路径/venv 固化为单用途):
#   venv = /data4/ws/PGS/env_sglang_train_q35(162 唯一验证过 verl 训练的栈)
#   树   = /data4/wushang/classic_tree(2026-08-23 自 A800 TERM 全量同步)
# 用法: launch_ws7b_grpo_162.sh dry|gate3|go
set -u
MODE=${1:?dry|go}
W=/data4/wushang/classic_tree
NAME=classic_ws7b_pgs
YAML=$W/configs/classic/webshop_7b_pgs_162.yaml
CVD=0,1,2,3
PORT=34000
source /data4/ws/PGS/env_q35_162.sh
export PYTHONPATH=$W/src:$W/verl      # ★ 必须在 source 之后(入口脚本会覆写)
PYX=/data4/ws/PGS/env_sglang_train_q35/bin/python
export SWANLAB_API_KEY=${SWANLAB_API_KEY:?export 真值后再发车(密钥不入库;A800 部署副本内联真值)}
export TRACE_35B_KEY=token-abc123   # researcher api_key_env
cd $W
[ -f "$YAML" ] || { echo "FATAL 配置不存在: $YAML"; exit 1; }

# ---- 35B researcher 端点预检(经隧道真实往返;/v1/models 200 不作数)----
echo "[pre] 校验 35B 隧道端点…"
for p in 8880 8881 8882; do
  RESP=$(timeout 30 curl -s -m 25 http://127.0.0.1:$p/v1/chat/completions \
    -H "Authorization: Bearer token-abc123" -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"say ok"}],"max_tokens":8}')
  echo "$RESP" | grep -q '"content"' || { echo "FATAL 35B 端点 $p 经隧道不可用: $(echo "$RESP" | head -c 120)"; exit 1; }
done
echo "[pre] OK 35B 三端点隧道真实往返通过"

# ---- WebShop 分片预检(逐分片 /meta + option_fix + 槽位;A800 版同源)----
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
    print(f"FATAL 分片 num_goals 不一致 {ng}"); sys.exit(1)
fixes = {u: m.get("option_fix") for u, m in metas.items()}
if any(v is None for v in fixes.values()):
    print("FATAL 分片 /meta 无 option_fix 字段"); sys.exit(1)
mismatch = {u: v for u, v in fixes.items() if bool(v) is not want_fix}
if mismatch:
    print(f"FATAL fix_option_matching={want_fix} 与服务端不一致: {list(mismatch)[:4]}"); sys.exit(1)
slots = sum(m["pool"] for m in metas.values())
need = int((cfg.get("train") or {}).get("train_batch_size", 0)) * int((cfg.get("train") or {}).get("rollout_n", 0))
print(f"[pre] OK 分片 {len(urls)} / 槽位 {slots} / 需求 {need} / goals {ng.pop()} / option_fix={want_fix}")
PYEOF

$PYX - "$YAML" <<'PYEOF' || exit 1
import sys, yaml
from pathlib import Path
t = (yaml.safe_load(open(sys.argv[1], encoding="utf-8")).get("train") or {})
miss = [p for p in [t.get("train_file"), t.get("model_path")] + list(t.get("val_files") or []) if p and not Path(p).exists()]
if miss:
    print(f"FATAL 路径缺失: {miss}"); sys.exit(1)
print("[pre] OK 模型与数据路径齐")
PYEOF

if [ "$MODE" = dry ]; then
  exec env CUDA_VISIBLE_DEVICES=$CVD $PYX scripts/launch_train.py \
    --config $YAML --run-name $NAME --dry-run
fi

# ---- GPU 清场门(>500MiB 拒发)----
for g in 0 1 2 3; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  if [ "$used" -gt 500 ]; then
    echo "FATAL: GPU $g 仍占用 ${used}MiB"; exit 1
  fi
done

mkdir -p /tmp/ray_ws7bp $W/runs
setsid env CUDA_VISIBLE_DEVICES=$CVD RAY_TMPDIR=/tmp/ray_ws7bp \
  $PYX scripts/launch_train.py --config $YAML \
    --run-name $NAME "+actor_rollout_ref.rollout.engine_kwargs.sglang.port=${PORT}" \
  >> $W/runs/${NAME}.launch.log 2>&1 < /dev/null &
echo "FIRED $NAME @162 cvd=$CVD port=$PORT pid=$!"
