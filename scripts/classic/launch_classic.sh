#!/bin/bash
# 经典三域统一发射器(宿主无关;2026-08-23)。同一份配置在 A800 / 162 发车
# 零改动:宿主差异全部收敛到同目录 host_<host>.env(路径/venv/env 入口),
# yaml 内路径为 ${TRACE_MODELS}/${TRACE_DATA} 插值(config._expand 于 load 时展开)。
#
#   用法: launch_classic.sh <run_name> <cfg_stem> <arm> <cvd> <sglport> <raytag>
#   例:   launch_classic.sh classic_alf3b_pgs alfworld_3b pgs 4,5,6,7 34000 alf3b
#   干跑: DRY_RUN=1 launch_classic.sh ...(不写盘;算指纹与展开后配置)
#   宿主: 自动探测(/fs/fast → a800;/data4/wushang → 162);TRACE_HOST 可显式覆盖
#   ⚠ run_name 必须显式给(resume 靠它找目录,不可从 cfg_stem 推导)。
#   ⚠ 同格跨宿主 objective_fingerprint 不同(路径展开进指纹)= 预期;
#     指纹守门是 per-run 目录的,单个 run 一生只在一台宿主 resume。
set -u
NAME=${1:?run_name}; CFG=${2:?cfg_stem}; ARM=${3:?arm}; CVD=${4:?cvd}; PORT=${5:?sglport}; TAG=${6:?raytag}
HERE=$(cd "$(dirname "$0")" && pwd)

# ---- 宿主探测与档案注入 ----
if [ -z "${TRACE_HOST:-}" ]; then
  if   [ -d /fs/fast/u2021000902 ]; then TRACE_HOST=a800
  elif [ -d /data4/wushang ];       then TRACE_HOST=162
  else echo "FATAL 未知宿主:export TRACE_HOST=a800|162 后重试"; exit 1; fi
fi
[ -f "$HERE/host_${TRACE_HOST}.env" ] || { echo "FATAL 缺宿主档案 $HERE/host_${TRACE_HOST}.env"; exit 1; }
source "$HERE/host_${TRACE_HOST}.env"
: "${TRACE_TREE:?}" "${TRACE_ENV_ENTRY:?}" "${TRACE_PYX:?}" "${TRACE_MODELS:?}" "${TRACE_DATA:?}"

source "$TRACE_ENV_ENTRY"
export PYTHONPATH=$TRACE_TREE/src:$TRACE_TREE/verl   # ★ 必须在 source 之后(入口会覆写)
export TRACE_35B_KEY=${TRACE_35B_KEY:-token-abc123}
export SWANLAB_API_KEY=${SWANLAB_API_KEY:?export 真值后再发车(密钥不入库)}
YAML=$TRACE_TREE/configs/classic/${CFG}_${ARM}.yaml
[ -f "$YAML" ] || { echo "FATAL 配置不存在: $YAML"; exit 1; }
[ -x "$TRACE_PYX" ] || { echo "FATAL python 缺失: $TRACE_PYX"; exit 1; }
cd "$TRACE_TREE"

# ---- 通用预检 1:模型/数据路径(经 ${} 展开后查存在;防 host env 漏配变空串)----
"$TRACE_PYX" - "$YAML" <<'PY1' || exit 1
import os, sys, yaml
from pathlib import Path
t = (yaml.safe_load(open(sys.argv[1], encoding="utf-8")).get("train") or {})
paths = [t.get("train_file"), t.get("model_path")] + list(t.get("val_files") or [])
miss = [q for q in (os.path.expandvars(p) for p in paths if p) if not Path(q).exists()]
if miss:
    print(f"FATAL 路径缺失(host env 是否 source?): {miss}"); sys.exit(1)
print("[pre] OK 模型与数据路径齐")
PY1

# ---- 通用预检 2:35B 三端点真实往返(researcher 全域启用;/v1/models 200 不作数;
#      两宿主同地址 127.0.0.1 —— A800 本机直连,162 经反向隧道)----
for p in 8880 8881 8882; do
  RESP=$(timeout 30 curl -s -m 25 http://127.0.0.1:$p/v1/chat/completions \
    -H "Authorization: Bearer $TRACE_35B_KEY" -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"say ok"}],"max_tokens":8}')
  echo "$RESP" | grep -q '"content"' || { echo "FATAL 35B 端点 $p 真实往返失败: $(echo "$RESP" | head -c 120)"; exit 1; }
done
echo "[pre] OK 35B 三端点真实往返"

# ---- 域预检(按配置族分派)----
case "$CFG" in
  alfworld*)
    : "${ALFWORLD_DATA:?162 宿主 ALF 资产未部署?见 host_162.env 注释}" "${ALFWORLD_WORKER_PYTHON:?}"
    [ -d "$ALFWORLD_DATA/json_2.1.1" ] || { echo "FATAL alfworld 数据缺失: $ALFWORLD_DATA"; exit 1; }
    [ -x "$ALFWORLD_WORKER_PYTHON" ]   || { echo "FATAL alfworld worker python 缺失"; exit 1; }
    echo "[pre] OK alfworld 资产" ;;
  webshop*)
    "$TRACE_PYX" - "$YAML" <<'PY2' || exit 1
import sys, json, urllib.request, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
ws = cfg.get("webshop") or {}
urls = [u.strip().rstrip("/") for u in str(ws.get("service_url", "")).split(",") if u.strip()]
want_fix = bool(ws.get("fix_option_matching", True))
if not urls: print("FATAL webshop.service_url 为空"); sys.exit(1)
metas, bad = {}, []
for u in urls:
    try:
        with urllib.request.urlopen(u + "/meta", timeout=8) as r: metas[u] = json.load(r)
    except Exception as e: bad.append(f"{u}: {type(e).__name__}")
if bad: print(f"FATAL {len(bad)}/{len(urls)} 分片不可达: {bad[:4]}"); sys.exit(1)
ng = {m["num_goals"] for m in metas.values()}
if len(ng) != 1: print(f"FATAL 分片 num_goals 不一致 {ng}"); sys.exit(1)
mism = {u for u, m in metas.items() if bool(m.get("option_fix")) is not want_fix}
if mism: print(f"FATAL option_fix 与配置不一致: {list(mism)[:4]}"); sys.exit(1)
slots = sum(m["pool"] for m in metas.values())
need = int((cfg.get("train") or {}).get("train_batch_size", 0)) * int((cfg.get("train") or {}).get("rollout_n", 0))
print(f"[pre] OK webshop 分片 {len(urls)} / 槽位 {slots} / 单步需求 {need} / goals {ng.pop()}")
PY2
    ;;
  searchqa*)
    "$TRACE_PYX" - "$YAML" <<'PY3' || exit 1
import sys, json, urllib.request, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
url = str((cfg.get("searchqa") or {}).get("retriever_url", "")).rstrip("/")
if not url: print("FATAL searchqa.retriever_url 为空"); sys.exit(1)
req = urllib.request.Request(url + "/retrieve",
    data=json.dumps({"queries": ["capital of France"], "topk": 1}).encode(),
    headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=30) as r: body = json.load(r)
except Exception as e:
    print(f"FATAL retriever 真实查询失败 {url}: {type(e).__name__}: {e}"); sys.exit(1)
res = body.get("result") or body.get("results") or []
if not res: print(f"FATAL retriever 返回空结果: {str(body)[:150]}"); sys.exit(1)
print(f"[pre] OK retriever 真实查询往返 {url}")
PY3
    ;;
esac

# ---- 干跑出口(不写盘不占卡)----
if [ "${DRY_RUN:-0}" = "1" ]; then
  exec env CUDA_VISIBLE_DEVICES=$CVD "$TRACE_PYX" scripts/launch_train.py \
    --config "$YAML" --run-name "$NAME" --dry-run
fi

# ---- GPU 清场门(>500MiB 拒发;2026-08-14 残留显存静默死亡事故)----
for g in $(echo "$CVD" | tr "," " "); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  if [ "$used" -gt 500 ]; then
    echo "FATAL: GPU $g 仍占用 ${used}MiB —— 先清场再发车(勿覆盖启动)"; exit 1
  fi
done

RAYDIR=/tmp/ray_${TAG}_${ARM}; mkdir -p "$RAYDIR" "$TRACE_TREE/runs"
setsid env CUDA_VISIBLE_DEVICES=$CVD RAY_TMPDIR=$RAYDIR \
  "$TRACE_PYX" scripts/launch_train.py --config "$YAML" \
    --run-name "$NAME" "+actor_rollout_ref.rollout.engine_kwargs.sglang.port=${PORT}" \
  >> "$TRACE_TREE/runs/${NAME}.launch.log" 2>&1 < /dev/null &
echo "FIRED $NAME host=$TRACE_HOST cfg=${CFG}_${ARM} cvd=$CVD port=$PORT pid=$!"
