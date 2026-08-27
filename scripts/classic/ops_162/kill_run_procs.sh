#!/usr/bin/env bash
# 162:按 /proc cmdline 精确杀某 run 的全部进程(禁 pkill 模式匹配);用法 kill_run_procs.sh <run_name>
NAME=${1:?run_name}
/data4/ws/PGS/env_sglang_train_q35/bin/python - "$NAME" <<'PY'
import os, sys, signal, time
name=sys.argv[1]; me={os.getpid(), os.getppid()}
def hits():
    out=[]
    for p in os.listdir("/proc"):
        if not p.isdigit() or int(p) in me: continue
        try: c=open(f"/proc/{p}/cmdline","rb").read().replace(b"\0",b" ").decode("utf8","replace")
        except Exception: continue
        if f"--run-name {name} " in c+" " or f"experiment_name={name} " in c+" ": out.append(int(p))
    return out
h=hits(); print("TERM ->", len(h))
for p in h:
    try: os.kill(p, signal.SIGTERM)
    except Exception: pass
time.sleep(12)
h=hits(); print("KILL ->", len(h))
for p in h:
    try: os.kill(p, signal.SIGKILL)
    except Exception: pass
time.sleep(4); print("left ->", len(hits()))
PY
# ray/sglang 子进程可能不带 run-name:按 RAY_TMPDIR 归属兜底清理并释放显存
sleep 5
echo "GPU:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "残留 ray::/sglang 进程数:"; ps -eo pid,cmd | grep -cE "[r]ay::|[s]glang.*port=34000|[S]GLangHttpServer"
