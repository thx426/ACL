"""ACS cloud sandbox client — shared execution substrate for the cloud-backed
environments (terminal first; SWE / WebArena land on the same surface).

One ``CloudSandbox`` = one Kubernetes Pod on ACS with the Agent Sandbox
compute class (MicroVM isolation). The method surface mirrors the E2B subset
that TMax's ``SandboxBackend`` ABC actually consumes (deep-read
terminal-tmax-seta.md §6.4): create / run / write_file / upload_dir / kill.
Backends with richer semantics (SandboxSet warm pools, the E2B-compatible
gateway) can replace this class without touching any adapter, as long as they
keep this surface.

Implementation notes
--------------------
* Plain Pod + exec (websocket) — the exact channel validated end-to-end in
  the cloud smoke (RUNBOOK-from-zero.md §8/§10: cold start 23s, private image
  pull 1.3s, e2e 15s). CRDs (agents.kruise.io) exist on the cluster but their
  allocation semantics are a pilot item, not a dependency.
* File transfer avoids exec-stdin entirely (stdin-EOF over the k8s exec
  protocol is version-dependent): content goes base64 in argv chunks, decoded
  in the sandbox. Task assets are KB-scale; this costs 1-3 round trips.
* Server-side command timeout is TMax-faithful:
  ``timeout --signal=TERM --kill-after=10 {t} bash -c {cmd}`` with exit 124
  mapped to a "Command timed out" prefix by the caller; the client-side
  websocket deadline is timeout + grace and only guards a dead sandbox.
"""
from __future__ import annotations

import io
import shlex
import tarfile
import threading
import time
import uuid
from dataclasses import dataclass

_B64_CHUNK = 50_000          # argv-safe chunk of base64 payload per exec
_EXEC_GRACE_S = 60.0         # client deadline slack over the in-sandbox timeout

#: Compute-class labels that route the Pod onto ACS Agent Sandbox capacity
#: (RUNBOOK §8; billing and MicroVM isolation both key off these).
_SANDBOX_LABELS = {
    "alibabacloud.com/compute-class": "agent-sandbox",
    "alibabacloud.com/compute-qos": "default",
}


class CloudboxError(RuntimeError):
    """Infrastructure-level sandbox failure (create/exec/transfer)."""


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int


_api_lock = threading.Lock()
_core_v1_cache: dict[str, object] = {}


def _core_v1(kubeconfig: str):
    """One CoreV1Api per kubeconfig path (REST calls are thread-safe; each
    exec opens its own websocket)."""
    with _api_lock:
        api = _core_v1_cache.get(kubeconfig)
        if api is None:
            from kubernetes import client as k8s_client
            from kubernetes import config as k8s_config

            k8s_config.load_kube_config(config_file=kubeconfig)
            api = k8s_client.CoreV1Api()
            _core_v1_cache[kubeconfig] = api
        return api


def _pod_name(task_id: str) -> str:
    base = "".join(c if c.isalnum() else "-" for c in task_id.lower())
    return f"tbx-{base[:48].strip('-')}-{uuid.uuid4().hex[:6]}"


class CloudSandbox:
    """One ACS sandbox Pod; methods are synchronous, one episode per instance."""

    def __init__(
        self,
        *,
        kubeconfig: str,
        namespace: str,
        image: str,
        image_pull_secret: str = "",
        cpu: str = "1",
        memory: str = "2Gi",
        task_id: str = "task",
        create_timeout_s: float = 180.0,
        run_label: str = "",
    ) -> None:
        self._api = _core_v1(kubeconfig)
        self._ns = namespace
        self._name = _pod_name(task_id)
        self._dead = False
        self._create(image, image_pull_secret, cpu, memory,
                     create_timeout_s, run_label)

    @property
    def name(self) -> str:
        return self._name

    # -------------------------------------------------------------- lifecycle

    def _create(self, image: str, pull_secret: str, cpu: str, memory: str,
                timeout_s: float, run_label: str) -> None:
        labels = dict(_SANDBOX_LABELS)
        labels["app"] = "trace-cloudbox"
        if run_label:
            labels["trace-run"] = run_label
        manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": self._name, "labels": labels},
            "spec": {
                "restartPolicy": "Never",
                "containers": [{
                    "name": "main",
                    "image": image,
                    "command": ["sleep", "infinity"],
                    "resources": {
                        "requests": {"cpu": cpu, "memory": memory},
                        "limits": {"cpu": cpu, "memory": memory},
                    },
                }],
                **({"imagePullSecrets": [{"name": pull_secret}]}
                   if pull_secret else {}),
            },
        }
        try:
            self._api.create_namespaced_pod(namespace=self._ns, body=manifest)
        except Exception as err:  # noqa: BLE001
            raise CloudboxError(f"pod create failed ({self._name}): {err}") from err
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                pod = self._api.read_namespaced_pod(self._name, self._ns)
            except Exception as err:  # noqa: BLE001
                self.kill()
                raise CloudboxError(f"pod status read failed: {err}") from err
            phase = getattr(pod.status, "phase", "")
            if phase == "Running":
                return
            if phase in ("Failed", "Succeeded"):
                self.kill()
                raise CloudboxError(
                    f"pod {self._name} reached terminal phase {phase} before Ready"
                )
            # Fail loud on unrecoverable waits (bad image beats a timeout).
            for cs in (pod.status.container_statuses or []):
                waiting = getattr(cs.state, "waiting", None)
                if waiting and waiting.reason in (
                    "ErrImagePull", "ImagePullBackOff", "InvalidImageName",
                    "CreateContainerConfigError",
                ):
                    self.kill()
                    raise CloudboxError(
                        f"pod {self._name} stuck: {waiting.reason} "
                        f"({waiting.message or ''})"
                    )
            if time.monotonic() > deadline:
                self.kill()
                raise CloudboxError(
                    f"pod {self._name} not Running after {timeout_s:.0f}s "
                    f"(last phase {phase!r})"
                )
            time.sleep(1.0)

    def kill(self) -> None:
        """Idempotent pod delete (grace 0); never raises."""
        if self._dead:
            return
        self._dead = True
        try:
            from kubernetes import client as k8s_client

            self._api.delete_namespaced_pod(
                self._name, self._ns,
                body=k8s_client.V1DeleteOptions(grace_period_seconds=0),
            )
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ exec

    def run(self, command: str, timeout_s: float,
            grace_s: float = _EXEC_GRACE_S) -> ExecResult:
        """Run a bash command with the TMax in-sandbox timeout wrapper.

        Exit 124 = the in-sandbox ``timeout`` fired (the caller renders the
        TMax "Command timed out" prefix). Raises CloudboxError only when the
        sandbox itself is unreachable/dead — never for a failing command.
        ``grace_s`` widens the client-side deadline over the in-sandbox
        timeout (scoring passes a large one: outer wall must exceed the inner
        600s — the TMax 120<600 inversion fix, proposal §5).
        """
        t = int(timeout_s)
        wrapped = (
            f"timeout --signal=TERM --kill-after=10 {t} "
            f"bash -c {shlex.quote(command)}"
        )
        return self._exec(wrapped, deadline_s=timeout_s + grace_s)

    def _exec(self, bash_command: str, *, deadline_s: float) -> ExecResult:
        from kubernetes.stream import stream

        # 连接期重试:公网 API server 的 websocket 握手会偶发掉线(实测
        # "Connection to remote host was lost")。握手失败 = 命令未开始执行,
        # 重试安全;流中途失败(下方 no-exit-status 路径)绝不重试——非幂等
        # 命令可能已产生副作用。
        last_err: Exception | None = None
        ws = None
        for attempt in range(3):
            try:
                ws = stream(
                    self._api.connect_get_namespaced_pod_exec,
                    self._name, self._ns,
                    command=["bash", "-c", bash_command],
                    container="main",
                    stderr=True, stdin=False, stdout=True, tty=False,
                    _preload_content=False,
                    _request_timeout=deadline_s + 30.0,
                )
                break
            except Exception as err:  # noqa: BLE001
                last_err = err
                time.sleep(1.0 * (attempt + 1))
        if ws is None:
            raise CloudboxError(
                f"exec connect failed on {self._name} after 3 attempts: {last_err}"
            ) from last_err
        try:
            ws.run_forever(timeout=deadline_s)
            stdout = ws.read_stdout()
            stderr = ws.read_stderr()
            rc: int | None
            try:
                rc = ws.returncode
            except Exception:  # noqa: BLE001
                rc = None
            if rc is None:
                # In-sandbox `timeout` guarantees termination within t+10s;
                # no exit status after the grace window ⇒ the sandbox (or the
                # exec channel) is gone, not the command still running.
                raise CloudboxError(
                    f"exec on {self._name} yielded no exit status within "
                    f"{deadline_s:.0f}s (sandbox dead or channel lost)"
                )
            return ExecResult(stdout=stdout or "", stderr=stderr or "",
                              exit_code=int(rc))
        finally:
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass

    # -------------------------------------------------------------- transfer

    def _put_b64(self, payload: bytes, remote_tmp: str) -> None:
        import base64

        b64 = base64.b64encode(payload).decode("ascii")
        self._exec(f": > {shlex.quote(remote_tmp)}", deadline_s=30.0)
        for i in range(0, len(b64), _B64_CHUNK):
            chunk = b64[i:i + _B64_CHUNK]
            self._exec(
                f"printf '%s' '{chunk}' >> {shlex.quote(remote_tmp)}",
                deadline_s=60.0,
            )

    def write_file(self, path: str, content: str | bytes, *,
                   mode: str = "") -> None:
        data = content.encode("utf-8") if isinstance(content, str) else content
        tmp = f"/tmp/.cbx_{uuid.uuid4().hex[:8]}.b64"
        self._put_b64(data, tmp)
        chmod = f" && chmod {mode} {shlex.quote(path)}" if mode else ""
        r = self._exec(
            f"mkdir -p {shlex.quote(path.rsplit('/', 1)[0] or '/')} && "
            f"base64 -d {shlex.quote(tmp)} > {shlex.quote(path)} && "
            f"rm -f {shlex.quote(tmp)}{chmod}",
            deadline_s=60.0,
        )
        if r.exit_code != 0:
            raise CloudboxError(
                f"write_file {path} failed (rc={r.exit_code}): "
                f"{(r.stderr or r.stdout)[:500]}"
            )

    def upload_dir(self, host_dir: str, dest: str) -> None:
        """put_archive replacement (deep-read §6.4): tar.gz the host dir in
        memory, ship base64, extract at dest (dest ends up containing the
        directory's contents, TMax ``upload_dir(tests → /tests)`` semantics)."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(host_dir, arcname=".")
        tmp = f"/tmp/.cbx_{uuid.uuid4().hex[:8]}.tgz.b64"
        self._put_b64(buf.getvalue(), tmp)
        r = self._exec(
            f"mkdir -p {shlex.quote(dest)} && "
            f"base64 -d {shlex.quote(tmp)} | tar -xz -C {shlex.quote(dest)} && "
            f"rm -f {shlex.quote(tmp)}",
            deadline_s=120.0,
        )
        if r.exit_code != 0:
            raise CloudboxError(
                f"upload_dir {host_dir} -> {dest} failed (rc={r.exit_code}): "
                f"{(r.stderr or r.stdout)[:500]}"
            )
