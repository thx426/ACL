"""Docker 沙箱后端(162 环境池)—— cloudbox 的同接口本地/远程 docker 实现。

**为什么需要它**(2026-08-11 实测,OPS-LOG [24]):A800 是 Nomad LXC,
`CapEff=0`、无容器工具、namespace 内 mount 被宿主 seccomp 拦;而 TMax 的
14,601 个 setup.sh **100% 需要 root**(apt-get/useradd/写 /app)⇒ 终端环境
在 A800 上 0% 可用。162 是物理机 docker 全绿,故架构 = **A800 训练 +
162 docker 环境池**。

**连接方式**:`DOCKER_HOST=unix:///tmp/docker162.sock`,由 A800 侧一条
`ssh -L <sock>:/var/run/docker.sock` 转发建立(脚本 `~/dockfwd.sh`)。
⚠ **该 ssh 必须 `ControlMaster=no`**:OpenSSH 8.0 下复用会让 unix-socket
转发的远端连接静默失败(实测:socket 在监听但 connect 被拒)。
实测延迟:docker API ping **中位 1.7ms / p95 3.5ms**(对比 ssh+docker CLI
每调用 113ms,快 66×;两机 rtt 0.675ms)。

**与 cloudbox 的关系**:方法面、返回类型(`ExecResult`)、错误类型
(`CloudboxError` ⇒ core 归 ENV_ERROR)全部相同,故 adapter 只需按配置换
后端,交互语义零变化。超时语义同样是**沙箱内 `timeout` 包裹 → exit 124**。
"""
from __future__ import annotations

import io
import logging
import os
import shlex
import tarfile
import threading
import time
import uuid

from .cloudbox import ExecResult, CloudboxError

logger = logging.getLogger(__name__)

_EXEC_GRACE_S = 15.0
#: 容器名前缀(便于按标签批量清理残留)
_NAME_PREFIX = "tbx"
_LABELS = {"app": "trace-dockerbox"}


def _client(docker_host: str = ""):
    """docker-py 客户端。docker_host 空 ⇒ 用环境变量/本机 socket。"""
    try:
        import docker  # noqa: PLC0415 — 惰性:不用 docker 后端的部署无需装
    except ImportError as err:  # pragma: no cover
        raise CloudboxError(
            "docker backend needs the docker-py package (pip install docker)"
        ) from err
    try:
        if docker_host:
            return docker.DockerClient(base_url=docker_host)
        return docker.from_env()
    except Exception as err:  # noqa: BLE001
        raise CloudboxError(
            f"docker connect failed (DOCKER_HOST={docker_host or os.environ.get('DOCKER_HOST', '')!r}) "
            f"— 若走 162:先跑 ~/dockfwd.sh 建 socket 转发: {err}"
        ) from err


class DockerSandbox:
    """One episode = one container. 方法面与 CloudSandbox 完全一致。

    资源闸门在**调用方**(EnvFactory 的 spawn 信号量);本类不做全局限流,
    但每容器带 `--cpus`/`--memory` 硬限,防单任务吃满宿主
    ([[cpu-starvation-kills-vllm]]:162 上还有别的训练在跑)。
    """

    def __init__(
        self,
        *,
        image: str,
        task_id: str = "task",
        cpus: str = "1",
        memory: str = "4g",
        docker_host: str = "",
        create_timeout_s: float = 120.0,
        run_label: str = "",
        network: str = "bridge",
    ) -> None:
        self._cli = _client(docker_host)
        self._name = f"{_NAME_PREFIX}-{task_id[:32]}-{uuid.uuid4().hex[:6]}"
        self._dead = False
        labels = dict(_LABELS)
        if run_label:
            labels["trace-run"] = run_label
        t0 = time.monotonic()
        try:
            self._c = self._cli.containers.run(
                image,
                command=["sleep", "infinity"],
                name=self._name,
                detach=True,
                labels=labels,
                nano_cpus=int(float(cpus) * 1e9),
                mem_limit=memory,
                network_mode=network,
                # 任务 setup.sh 会 useradd/chown/apt —— 容器内 root 是必需的
                # (TMax 任务的既定前提);容器边界即隔离边界。
                user="root",
                working_dir="/",
            )
        except Exception as err:  # noqa: BLE001
            raise CloudboxError(f"container create failed ({self._name}): {err}") from err
        # 就绪判据:能执行一条命令(不是"状态 running"——那不是可用性判据)
        deadline = time.monotonic() + create_timeout_s
        while True:
            try:
                rc, _ = self._c.exec_run("true", demux=False)
                if rc == 0:
                    logger.debug("dockerbox %s ready in %.2fs",
                                 self._name, time.monotonic() - t0)
                    return
            except Exception:  # noqa: BLE001 — 启动窗口内的瞬时失败
                pass
            if time.monotonic() > deadline:
                self.kill()
                raise CloudboxError(
                    f"container {self._name} not usable after {create_timeout_s:.0f}s"
                )
            time.sleep(0.2)

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------ exec

    def _exec(self, command: str, deadline_s: float) -> ExecResult:
        """裸执行(不加沙箱内 timeout);内部件用。"""
        try:
            rc, out = self._c.exec_run(
                ["/bin/bash", "-lc", command], demux=True,
            )
        except Exception as err:  # noqa: BLE001 — 容器死/守护进程不可达
            self._dead = True
            raise CloudboxError(f"exec failed on {self._name}: {err}") from err
        stdout = (out[0] or b"").decode("utf-8", "replace")
        stderr = (out[1] or b"").decode("utf-8", "replace")
        return ExecResult(stdout=stdout, stderr=stderr, exit_code=int(rc or 0))

    def run(self, command: str, timeout_s: float,
            grace_s: float = _EXEC_GRACE_S) -> ExecResult:
        """沙箱内 `timeout` 包裹执行(与 cloudbox 同语义:exit 124 = 超时)。

        `grace_s` 只影响客户端等待余量;docker exec 是同步调用,超时由容器内
        的 `timeout` 保证 —— 不会出现"命令还在跑而客户端已放弃"的悬挂。"""
        wrapped = (
            f"timeout --signal=TERM --kill-after=10 {int(timeout_s)} "
            f"bash -lc {shlex.quote(command)}"
        )
        return self._exec(wrapped, deadline_s=timeout_s + grace_s)

    def run_polled(self, command: str, timeout_s: float,
                   poll_s: float = 5.0,
                   grace_s: float = _EXEC_GRACE_S) -> ExecResult:
        """长命令变体。docker exec 走本地 socket(经 ssh 单连接多路复用),
        **没有公网 websocket 的流中断问题**(那是 ACS 的坑),故这里直接
        复用 run —— 保留方法名是为了与 cloudbox 接口一致,调用方无需分支。"""
        return self.run(command, timeout_s, grace_s=grace_s)

    # ------------------------------------------------------------------ files

    def write_file(self, path: str, content: str | bytes, *,
                   mode: str = "") -> None:
        """put_archive 直传(比 base64-argv 快一个量级,且无 argv 长度上限)。"""
        data = content.encode("utf-8") if isinstance(content, str) else content
        parent, _, base = path.rpartition("/")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=base)
            info.size = len(data)
            info.mode = int(mode, 8) if mode else 0o644
            tar.addfile(info, io.BytesIO(data))
        r = self._exec(f"mkdir -p {shlex.quote(parent or '/')}", deadline_s=30.0)
        if r.exit_code != 0:
            raise CloudboxError(f"write_file mkdir {parent} failed: {r.stderr[:300]}")
        try:
            self._c.put_archive(parent or "/", buf.getvalue())
        except Exception as err:  # noqa: BLE001
            raise CloudboxError(f"write_file {path} failed: {err}") from err

    def upload_dir(self, host_dir: str, dest: str) -> None:
        """宿主目录 → 容器 dest(dest 内含该目录的**内容**,TMax 语义)。"""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            tar.add(host_dir, arcname=".")
        r = self._exec(f"mkdir -p {shlex.quote(dest)}", deadline_s=30.0)
        if r.exit_code != 0:
            raise CloudboxError(f"upload_dir mkdir {dest} failed: {r.stderr[:300]}")
        try:
            self._c.put_archive(dest, buf.getvalue())
        except Exception as err:  # noqa: BLE001
            raise CloudboxError(f"upload_dir {host_dir}→{dest} failed: {err}") from err

    # --------------------------------------------------------------- lifecycle

    def kill(self) -> None:
        """幂等销毁;绝不抛(finally 路径)。"""
        if self._dead and not getattr(self, "_c", None):
            return
        self._dead = True
        try:
            self._c.remove(force=True)
        except Exception:  # noqa: BLE001
            pass


def sweep_orphans(docker_host: str = "", run_label: str = "") -> int:
    """按标签清理残留容器(崩溃/中断后的回收;返回清理数)。"""
    cli = _client(docker_host)
    flt = dict(_LABELS)
    if run_label:
        flt["trace-run"] = run_label
    labels = [f"{k}={v}" for k, v in flt.items()]
    n = 0
    for c in cli.containers.list(all=True, filters={"label": labels}):
        try:
            c.remove(force=True)
            n += 1
        except Exception:  # noqa: BLE001
            pass
    return n
