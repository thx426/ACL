# PGS 实时显微镜：训练机启动与本地 SSH 隧道手册

> 本文是 `scripts/pgs_microscope_server.py` 的启动、访问和运维单一详细手册。
> 服务实时只读 `<run>/rollouts/`，不控制训练进程，不向公网暴露端口。
>
> 当前 A800 dynamic-researcher 正式 run 的运行身份见
> [`docs/pgs-s2-a800-researcher-run-card.md`](pgs-s2-a800-researcher-run-card.md)。

## 1. 能看什么，以及当前边界

实时显微镜提供以下接口：

| 接口 | 内容 |
|---|---|
| `/` | 前端应用 |
| `/api/run` | train/val step 聚合曲线，包含正在写入的 partial step |
| `/api/step/<step_dir>` | 某 step 的轨迹轻量列表 |
| `/api/traj/<step_dir>/<index>` | 单条轨迹的逐 token 详情 |

界面可展示：

- train/validation 曲线和进行中的 step；
- task、outcome、响应长度、逐轮 thought/action/observation；
- SPS 候选、开门、倾斜、拒绝、预算、`θ_D`；
- token 级 draft/final、`D` 与阈值、两侧 top-K、行为校正账本；
- 新 rollout JSON 出现后的实时更新，无需重启服务。

### 1.1 静态 GT 文档与动态 researcher 文档

`--doc-table` 只用于静态 JSONL：

```json
{"task_id": "...", "doc": "..."}
```

对于静态 GT run，只有在已确认表与该 run 完全对应时才可传 `--doc-table`。

对于当前 dynamic-researcher run：

```text
pgs_s2_sgl_a800_researcher
```

**不要传旧 GT `--doc-table`**。动态文档按 `(global_step, task_id)` 写入各自的
`ResearchArchive`，同一 task 下一 step 的文档也可能不同。显微镜会从每条 rollout 的
`research.research_archive`安全载入当步`doc.md`和`meta.json`，详情页显示：

- 来源状态：dynamic researcher、静态 GT、真正 bare、validation bare、researcher降级、
  archive缺失或读取错误；
- 正文及“已进入教师”/“仅归档审计”标记；
- archive相对路径；
- `global_step`、真实`policy_version`、`experiments_used`、`episodes_consumed`、
  `turns_used`、`finalized`、`error/session_error`等白名单审计字段；
- `<run>/effective_config.yaml`实收`steering.b_max`，当前正式run应显示`B∞=400`。

后端只接受当前run的`research/`相对路径，并拒绝绝对路径、`..`、非research目录及
symlink逃逸。单个坏archive会显示为轨迹级错误，不会让整个服务退出。动态run仍不得为了
页面显示文档而传旧GT表，否则会展示错误的特权信息。

手工定位方法见[第 8 节](#8-独立审计-dynamic-researcher-文档)。

## 2. 当前 A800 正式 run 的参数

截至 2026-08-05，推荐使用以下参数：

| 项 | 值 |
|---|---|
| SSH | `ssh -p 23653 u2021000902@10.10.252.11` |
| bundle | `/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/trace_opd_researcher_pgs` |
| run | `/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/runs/pgs_s2_sgl_a800_researcher` |
| tokenizer | `/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/ckpt/sft_step6` |
| Python | `/fs/fast/u2021000902/wushang/workspace/TRACE-OPD/env_sglang_train/bin/python` |
| 推荐远端端口 | `8794` |
| 推荐本地端口 | `8794`，冲突时只改本地端口为 `18794` |

端口不是永久保留资源。2026-08-05 实测：

- `8792` 已用于旧 base run；
- `8793` 已用于旧 GT run `pgs_s2_sgl_a800`；
- `8794` 当时空闲，推荐给新 researcher run；
- 每次启动前仍必须重新探测，不能只信历史记录。

### 2.1 2026-08-05 当前双机部署真值

| 正式实验 | 远端服务 | 本地入口 | 文档源 | 服务PID / PID文件 |
|---|---|---|---|---|
| A800 `pgs_s2_sgl_a800_researcher` | A800 `127.0.0.1:8794` | `http://127.0.0.1:8794/` | dynamic researcher | `30839` / `$ROOT/logs/pgs_microscope_a800_researcher_8794.pid` |
| 162 `pgs_s2_sgl_1p7b_162_gt_full` | 162 `127.0.0.1:8792` | `http://127.0.0.1:18792/` | 静态 GT | `3797619` / `/data4/ws/PGS/logs/pgs_microscope_1p7b_gt_full_8792.pid` |

本地162隧道PID文件为`~/.trace-opd-viz-162-1p7b-18792.pid`，日志为
`~/Library/Logs/trace-opd-viz-162-1p7b-18792.log`。本地`8792`已用于A800旧base页面，
所以162只改本地左侧端口为`18792`，远端仍是`8792`。两套正式页面已分别打开。

162当前标准bundle为`/data3/wushang/workspace/TRACE-OPD-PGS/trace_opd`，run为
`/data4/ws/PGS/pgs_s2_sgl_1p7b_162_gt_full`，tokenizer为
`/data3/wushang/model/Qwen/Qwen3-1.7B`，启动时须传该run实收的GT表：
`/data3/wushang/workspace/TRACE-OPD/trace_opd-pgc/data/out/gt_docs_rl_train_v2.jsonl`。

大run首次请求`/api/run`会冷聚合全部历史step。162在208个训练step量级实测单请求约39秒；
同key并发请求由single-flight合并，但客户端仍应发送**一个长超时请求**等待完成，不能用5秒
超时循环重试制造无意义等待线程。热缓存后总览约亚秒级，单轨迹约50毫秒。

## 3. 登录远端并做启动前检查

从本地登录 A800：

```bash
ssh -p 23653 u2021000902@10.10.252.11
```

在远端设置变量：

```bash
ROOT=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS
BUNDLE=$ROOT/trace_opd_researcher_pgs
RUN=$ROOT/runs/pgs_s2_sgl_a800_researcher
TOKENIZER=$ROOT/ckpt/sft_step6
PY=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD/env_sglang_train/bin/python
PORT=8794

VIZ_LOG=$ROOT/logs/pgs_microscope_a800_researcher_${PORT}.log
VIZ_PIDFILE=$ROOT/logs/pgs_microscope_a800_researcher_${PORT}.pid
```

检查路径：

```bash
test -d "$RUN" && echo "run dir OK"
test -d "$TOKENIZER" && echo "tokenizer OK"
test -x "$PY" && echo "python OK"
test -f "$BUNDLE/scripts/pgs_microscope_server.py" && echo "server script OK"
test -f "$BUNDLE/scripts/pgs_microscope_app.html" && echo "frontend OK"
```

用 Python 尝试绑定来检查远端端口；这比凭进程名判断可靠：

```bash
PORT=8794 "$PY" - <<'PY'
import os
import socket

port = int(os.environ["PORT"])
sock = socket.socket()
try:
    sock.bind(("127.0.0.1", port))
finally:
    sock.close()
print(f"127.0.0.1:{port} is free")
PY
```

若报 `Address already in use`：

1. 不覆盖、不清理未知服务；
2. 先查看已知显微镜：

   ```bash
   ps -eo pid,pgid,args | grep '[p]gs_microscope_server.py'
   ```

3. 改用空闲端口，例如 `PORT=8795`，并同步修改远端日志、PID文件和本地隧道命令。

## 4. 启动远端显微镜

### 4.1 前台启动：首次排错最直观

首次验证可在远端前台运行：

```bash
"$PY" "$BUNDLE/scripts/pgs_microscope_server.py" \
  --run-dir "$RUN" \
  --tokenizer "$TOKENIZER" \
  --host 127.0.0.1 \
  --port "$PORT"
```

对 dynamic-researcher run，命令中**没有** `--doc-table`。

预期输出：

```text
[server] loading tokenizer ...
[server] doc table: 0 条
[server] budget B∞=400.0 (effective_config.yaml:steering.b_max)
[server] run=.../pgs_s2_sgl_a800_researcher http://127.0.0.1:8794
```

按 `Ctrl-C` 只会停止显微镜，不影响训练。确认前台模式正常后，再使用后台模式。

### 4.2 推荐后台启动：断开 SSH 后仍运行

下面的启动器会：

- 检查脚本、run、tokenizer 和 Python；
- 检查 PID 文件指向的旧进程；
- 检查端口是否空闲；
- 使用 `start_new_session=True` 创建独立 PGID；
- 只绑定 `127.0.0.1`；
- 把 PID 写入专用文件；
- 不接触训练 PID/PGID。

```bash
"$PY" - <<'PY'
from pathlib import Path
import os
import socket
import subprocess

root = Path("/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS")
bundle = root / "trace_opd_researcher_pgs"
run_dir = root / "runs/pgs_s2_sgl_a800_researcher"
tokenizer = root / "ckpt/sft_step6"
python = Path(
    "/fs/fast/u2021000902/wushang/workspace/"
    "TRACE-OPD/env_sglang_train/bin/python"
)
port = 8794

log = root / f"logs/pgs_microscope_a800_researcher_{port}.log"
pidfile = root / f"logs/pgs_microscope_a800_researcher_{port}.pid"
server = bundle / "scripts/pgs_microscope_server.py"
app = bundle / "scripts/pgs_microscope_app.html"

for path in (server, app, run_dir, tokenizer, python):
    if not path.exists():
        raise SystemExit(f"FATAL: missing {path}")

if pidfile.exists():
    try:
        old_pid = int(pidfile.read_text().strip())
    except ValueError:
        old_pid = -1
    cmdline = Path(f"/proc/{old_pid}/cmdline")
    if cmdline.is_file():
        command = cmdline.read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        )
        if "pgs_microscope_server.py" in command:
            raise SystemExit(
                f"FATAL: visualizer already running as PID {old_pid}: {command}"
            )

sock = socket.socket()
try:
    sock.bind(("127.0.0.1", port))
except OSError as exc:
    raise SystemExit(f"FATAL: 127.0.0.1:{port} is not free: {exc}") from exc
finally:
    sock.close()

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
command = [
    str(python),
    str(server),
    "--run-dir",
    str(run_dir),
    "--tokenizer",
    str(tokenizer),
    "--host",
    "127.0.0.1",
    "--port",
    str(port),
]

log.parent.mkdir(parents=True, exist_ok=True)
with log.open("ab", buffering=0) as stream:
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=stream,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
        close_fds=True,
    )

pidfile.write_text(f"{process.pid}\n")
print(f"visualizer PID/PGID: {process.pid}/{process.pid}")
print(f"log: {log}")
print(f"remote URL: http://127.0.0.1:{port}/")
PY
```

记录打印出的 PID/PGID。不要把可视化 PID 与训练 launcher、Ray worker 或 SGLang PID混淆。

## 5. 必须用新 SSH 连接验收

启动命令返回不等于服务存活。退出或另开一个本地终端，重新连接：

```bash
ssh -p 23653 u2021000902@10.10.252.11
```

远端执行：

```bash
ROOT=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS
PORT=8794
VIZ_LOG=$ROOT/logs/pgs_microscope_a800_researcher_${PORT}.log
VIZ_PIDFILE=$ROOT/logs/pgs_microscope_a800_researcher_${PORT}.pid

read -r VIZ_PID < "$VIZ_PIDFILE"
ps -p "$VIZ_PID" -o pid=,pgid=,lstart=,args=
```

cmdline必须同时包含：

```text
pgs_microscope_server.py
--run-dir .../pgs_s2_sgl_a800_researcher
--host 127.0.0.1
--port 8794
```

查看日志：

```bash
tail -n 30 "$VIZ_LOG"
```

远端HTTP验活：

```bash
curl -fsS http://127.0.0.1:8794/api/run | python3 -m json.tool
```

关键字段应包括：

```json
{
  "name": "pgs_s2_sgl_a800_researcher",
  "live": true,
  "hasDocs": false,
  "docs": {
    "staticCount": 0,
    "dynamicArchives": true
  },
  "config": {
    "bmax": 400.0,
    "bmaxSource": "effective_config.yaml:steering.b_max",
    "bmaxWarning": ""
  },
  "steps": [],
  "vals": []
}
```

`steps`和`vals`的实际内容随训练进度变化。兼容字段`hasDocs=false`只表示没有传静态
`--doc-table`，不能再用它判断dynamic researcher是否有文档；看`docs.dynamicArchives`和
单条`/api/traj/...`的`privilege`对象。

首页验活：

```bash
curl -fsS -o /dev/null http://127.0.0.1:8794/
printf 'HTTP check exit code: %s\n' "$?"
```

## 6. 在本地建立 SSH 隧道

### 6.1 推荐：前台隧道

在本地单独开一个终端：

```bash
ssh -N \
  -L 127.0.0.1:8794:127.0.0.1:8794 \
  -p 23653 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  u2021000902@10.10.252.11
```

该终端没有输出并持续占用是正常的。它只维护隧道：

- `Ctrl-C`只关闭本地隧道；
- 不会停止远端显微镜；
- 不会停止训练。

在另一个本地终端打开：

```bash
open http://127.0.0.1:8794/
```

也可直接在浏览器访问：

```text
http://127.0.0.1:8794/
```

### 6.2 本地端口冲突：只改左侧端口

检查本地端口：

```bash
lsof -nP -iTCP:8794 -sTCP:LISTEN
```

若本地8794被占用，远端仍保持8794，只把本地端口改为18794：

```bash
ssh -N \
  -L 127.0.0.1:18794:127.0.0.1:8794 \
  -p 23653 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  u2021000902@10.10.252.11
```

打开：

```bash
open http://127.0.0.1:18794/
```

映射关系：

```text
本地 127.0.0.1:18794
        ↓ SSH tunnel
A800 127.0.0.1:8794
```

### 6.3 当前162正式run隧道

由于本地8792已被A800旧base隧道占用，162远端8792映射到本地18792：

```bash
ssh -N \
  -L 127.0.0.1:18792:127.0.0.1:8792 \
  -p 5102 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  haoyang@10.77.110.162
```

浏览器打开：

```bash
open http://127.0.0.1:18792/
```

### 6.4 可选：本地后台隧道

前台模式最容易判断隧道是否断开。确需后台运行时：

```bash
TUNNEL_LOG="$HOME/Library/Logs/trace-opd-viz-8794.log"
TUNNEL_PIDFILE="$HOME/.trace-opd-viz-8794.pid"
mkdir -p "$HOME/Library/Logs"

nohup ssh -N \
  -L 127.0.0.1:8794:127.0.0.1:8794 \
  -p 23653 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  u2021000902@10.10.252.11 \
  >"$TUNNEL_LOG" 2>&1 < /dev/null &

printf '%s\n' "$!" > "$TUNNEL_PIDFILE"
```

从新本地shell验收：

```bash
read -r TUNNEL_PID < "$HOME/.trace-opd-viz-8794.pid"
ps -p "$TUNNEL_PID" -o pid=,lstart=,args=
curl -fsS http://127.0.0.1:8794/api/run | python3 -m json.tool
```

## 7. 使用 API 和界面

建立隧道后，下列命令在本地执行。

### 7.1 Run总览

```bash
curl -fsS http://127.0.0.1:8794/api/run | python3 -m json.tool
```

### 7.2 某个step的轨迹列表

```bash
curl -fsS \
  http://127.0.0.1:8794/api/step/step_00001 \
  | python3 -m json.tool
```

### 7.3 某条逐token轨迹

`index`是该step目录内按文件名排序后的零基索引：

```bash
curl -fsS \
  http://127.0.0.1:8794/api/traj/step_00001/0 \
  | python3 -m json.tool
```

### 7.4 单条dynamic文档与审计字段

```bash
curl -fsS \
  http://127.0.0.1:8794/api/traj/step_00001/0 \
  | python3 -c '
import json, sys
row = json.load(sys.stdin)
print(json.dumps({"budget": row["m"]["bmax"], "privilege": row["privilege"]}, ensure_ascii=False, indent=2))
'
```

正常dynamic轨迹应满足：

- `privilege.state == "dynamic_researcher"`；
- `privilege.usedByTeacher == true`；
- `privilege.doc`非空；
- `privilege.archive`以`research/`开头；
- `privilege.meta.policy_version`与当步真实rollout权重一致；
- `m.bmax`等于该run实收配置。

### 7.5 页面只有validation、尚无train step

Dynamic researcher的正式时序是：

1. 训前bare validation；
2. 16个researcher session用当步冻结policy做实验；
3. 每个task finalize并归档一份动态文档；
4. 同task的n=8 student共享该文档做SPS rollout；
5. 128条student完成后actor update和SGLang weight update。

因此research阶段页面可能只有`step_00000_val`，而`step_00001`尚未出现。这不是可视化故障。

## 8. 独立审计 dynamic researcher 文档

UI会自动载入动态`doc.md`；需要绕过UI独立核验时，可从一条rollout JSON的
`research.research_archive`定位。以下命令在A800远端运行。

先选一个step和轨迹索引：

```bash
ROOT=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS
RUN=$ROOT/runs/pgs_s2_sgl_a800_researcher
PY=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD/env_sglang_train/bin/python
STEP=step_00001
INDEX=0
```

按与API相同的文件排序取得原始dump，并解析archive路径：

```bash
DUMP=$(find "$RUN/rollouts/$STEP" -maxdepth 1 -type f -name '*.json' \
  | LC_ALL=C sort \
  | sed -n "$((INDEX + 1))p")

ARCHIVE=$("$PY" - "$DUMP" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
row = json.loads(path.read_text())
print((row.get("research") or {}).get("research_archive") or "")
PY
)

printf 'dump: %s\narchive: %s\n' "$DUMP" "$ARCHIVE"
```

若`ARCHIVE`非空：

```bash
less "$RUN/$ARCHIVE/doc.md"
```

同一step、同一task的n=8 student应指向同一个archive；下一step重新遇到该task时会有新archive。

也可查看元数据：

```bash
"$PY" -m json.tool "$RUN/$ARCHIVE/meta.json"
```

重点字段：

- `global_step`：逻辑batch/update编号；
- `policy_version`：research episode实际使用的rollout权重版本；
- `experiments_used`、`episodes_consumed`、`turns_used`；
- `error`、`session_error`、`finalized`、`doc_words`；
- `gt=off`、`gt_injected=false`。

## 9. 停止服务

### 9.1 停止本地前台隧道

在运行`ssh -N -L ...`的终端按：

```text
Ctrl-C
```

远端显微镜和训练均不受影响。

### 9.2 停止本地后台隧道

先核对PID和cmdline：

```bash
read -r TUNNEL_PID < "$HOME/.trace-opd-viz-8794.pid"
ps -p "$TUNNEL_PID" -o pid=,args=
```

确认它是对应的：

```text
ssh -N -L 127.0.0.1:8794:127.0.0.1:8794 ...
```

再停止：

```bash
kill "$TUNNEL_PID"
rm "$HOME/.trace-opd-viz-8794.pid"
```

### 9.3 停止远端显微镜

登录A800并设置：

```bash
ROOT=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS
PORT=8794
VIZ_PIDFILE=$ROOT/logs/pgs_microscope_a800_researcher_${PORT}.pid

read -r VIZ_PID < "$VIZ_PIDFILE"
ps -p "$VIZ_PID" -o pid=,pgid=,args=
```

只有在cmdline精确包含以下三项时才停止：

```text
pgs_microscope_server.py
pgs_s2_sgl_a800_researcher
--port 8794
```

执行：

```bash
kill "$VIZ_PID"
```

确认退出后再删PID文件：

```bash
if kill -0 "$VIZ_PID" 2>/dev/null; then
  echo "visualizer is still stopping; keep PID file and inspect the log"
else
  rm "$VIZ_PIDFILE"
  echo "visualizer stopped"
fi
```

该操作只针对显微镜PID。不要向训练launcher、Ray、SGLang或watchdog发送信号。

## 10. 故障排查

### 10.1 本地页面打不开

按顺序检查，不要先重启：

1. 远端API：

   ```bash
   curl -v http://127.0.0.1:8794/api/run
   ```

2. 远端显微镜PID：

   ```bash
   read -r VIZ_PID < \
     /fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/logs/pgs_microscope_a800_researcher_8794.pid
   ps -p "$VIZ_PID" -o pid=,pgid=,args=
   ```

3. 远端日志：

   ```bash
   tail -n 50 \
     /fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/logs/pgs_microscope_a800_researcher_8794.log
   ```

4. 本地监听：

   ```bash
   lsof -nP -iTCP:8794 -sTCP:LISTEN
   ```

5. 本地隧道API：

   ```bash
   curl -v http://127.0.0.1:8794/api/run
   ```

典型判读：

| 现象 | 含义/动作 |
|---|---|
| 远端API通，本地API不通 | 本地SSH隧道问题；重建隧道，不动远端服务和训练 |
| 远端API不通，PID不存在 | 读显微镜日志后按第4节重新启动；不动训练 |
| 远端API 500 | 查看响应正文和显微镜日志，常见是某个rollout JSON正在写或schema问题 |
| `/api/run`只有val | 可能仍在dynamic researcher阶段，先查run产物，不要盲目重启 |
| `hasDocs=false` | 仅表示没有静态`--doc-table`；dynamic能力看`docs.dynamicArchives`，单条状态看`privilege` |
| `privilege.state=archive_missing/archive_error` | 只检查该rollout引用的archive与显微镜日志；不传旧GT表，不动训练 |
| `config.bmaxWarning`非空 | `effective_config.yaml`缺失或格式异常，UI已回退20；先核对run配置产物，不猜预算 |
| 页面数据不再增长 | 先检查`<run>/rollouts/`文件数是否增长；产物不增长与UI缓存是两类问题 |

### 10.2 端口已占用

不要用模糊清理命令。先列出显微镜：

```bash
ps -eo pid,pgid,args | grep '[p]gs_microscope_server.py'
```

若端口属于另一个正式run，选择新端口。只有确认PID、run-dir和port三者精确匹配时，才按第9节停止。

### 10.3 Tokenizer加载失败

核对：

```bash
test -d /fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/ckpt/sft_step6
/fs/fast/u2021000902/wushang/workspace/TRACE-OPD/env_sglang_train/bin/python \
  -c 'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/ckpt/sft_step6"); print("tokenizer OK")'
```

显微镜只加载tokenizer，不应占用训练GPU。不要为可视化切换或安装新的torch/flash-attn环境。

## 11. 安全红线

- 远端服务必须绑定`127.0.0.1`，不得绑定`0.0.0.0`；
- 本地转发也显式绑定`127.0.0.1`；
- 不向显微镜命令写API key、SSH密码或SwanLab credential；
- 不对dynamic researcher run传旧GT `--doc-table`；
- 不使用`pkill -f 8794`、`pkill -f pgs_microscope`等模糊命令；
- 停止前必须核对PID、PGID、run-dir和port；
- 显微镜异常只处理显微镜或本地隧道，不联动重启/清理训练；
- 训练健康以run产物和正式监控为准，不能以页面是否刷新替代；
- 运行中的SwanLab run禁止执行`swanlab sync`。

## 12. 静态快照导出（可选）

实时服务之外，可以生成可复制的静态显微镜：

```bash
"$PY" "$BUNDLE/scripts/pgs_microscope.py" \
  --run-dir "$RUN" \
  --tokenizer "$TOKENIZER" \
  --last 3
```

默认产物：

```text
<RUN>/microscope/index.html
<RUN>/microscope/data/run.js
<RUN>/microscope/data/<step>.js
```

它是生成时刻的快照，不会随训练自动更新。Dynamic researcher run同样不要传旧GT `--doc-table`；
静态导出会按生成时刻安全载入每条rollout关联的dynamic archive，并写入与live API相同的
`privilege`和`m.bmax`字段。
