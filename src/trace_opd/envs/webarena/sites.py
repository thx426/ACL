"""Site pools for the WebArena domain — where episodes get their site
instances from.

Two implementations behind one duck-typed surface (``acquire(sites) →
SiteAllocation``):

* ``StaticSitePool`` — fixed base URLs (local docker containers for smoke,
  or a hand-managed remote deployment). NO state isolation between episodes:
  fine for retrieve/navigate smoke gates; NOT valid for concurrent mutate
  rollouts (documented, fail-safe: nothing enforces it here — the smoke
  driver's job is to pick read-only tasks).

* ``AcsSitePool`` — per-site pools of checkpoint-restored ACS pods
  (OPS-LOG [16] / RUNBOOK §12.5 recipe: SandboxTemplate carrying
  ``ops.alibabacloud.com/checkpoint-id`` + SandboxSet keeping N replicas).
  ``acquire`` claims a Ready pod via annotation CAS (safe across processes),
  ``release`` deletes it — the SandboxSet controller replaces it with a fresh
  restore, which IS the state reset (P-A measured ~16s/instance). Pool
  sizing: replicas must cover concurrency × (1 + restore_time/episode_time).

初版边界(P-B pilot 校准项,标注 TODO):pod label selector 的确切键名、
env-ctrl 健康探测(现以 Running+containersReady 为就绪判据,站点残破由
browser 首个 goto fail-loud 暴露)、跨可用区网络时延。
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class SitePoolError(RuntimeError):
    """Infrastructure-level site-pool failure (no capacity / claim failed)."""


@dataclass
class SiteAllocation:
    """Sites leased to one episode. ``release()`` is idempotent."""

    base_urls: dict[str, str]           # site name → base URL (no trailing /)
    _releaser: object = field(default=None, repr=False)
    _released: bool = field(default=False, repr=False)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if callable(self._releaser):
            try:
                self._releaser()
            except Exception as err:  # noqa: BLE001 — 释放失败不阻断 close 链
                logger.warning("site release failed: %s", err)


class StaticSitePool:
    """Fixed site → URL mapping; no isolation, no capacity limit."""

    def __init__(self, base_urls: dict[str, str]) -> None:
        self._base_urls = {k: v.rstrip("/") for k, v in base_urls.items()}

    def acquire(self, sites: list[str]) -> SiteAllocation:
        missing = [s for s in sites if s not in self._base_urls]
        if missing:
            raise SitePoolError(
                f"static site pool missing sites {missing} "
                f"(configured: {sorted(self._base_urls)})"
            )
        return SiteAllocation({s: self._base_urls[s] for s in sites})


#: 领用标记注解(CAS 依据:带 resourceVersion 的 replace,409 即被抢)。
_CLAIM_ANNOTATION = "trace-opd.io/claimed-by"


class AcsSitePool:
    """Checkpoint-restored pod pools on ACS, one SandboxSet per site."""

    def __init__(
        self,
        *,
        kubeconfig: str,
        namespace: str,
        sandboxsets: dict[str, str],       # site name → SandboxSet name
        site_port: int = 80,               # am1n3e 镜像站点容器端口
        map_port: int = 3000,              # map 单容器 beta 镜像走 3000
        acquire_timeout_s: float = 300.0,
        claimant: str = "",
    ) -> None:
        from ..cloudbox import _core_v1   # 共享 CoreV1Api 缓存(REST 线程安全)

        self._api = _core_v1(kubeconfig)
        self._ns = namespace
        self._sets = dict(sandboxsets)
        self._site_port = site_port
        self._map_port = map_port
        self._timeout = acquire_timeout_s
        self._claimant = claimant or f"pid-{id(self)}"
        self._lock = threading.Lock()

    # TODO(P-B): 确切的 SandboxSet→Pod label 键名以 pilot 实测为准;
    # 当前按 kruise 命名惯例取 sandbox-set 名标签,不匹配时 fail-loud。
    _LABEL_TPL = "agents.kruise.io/sandbox-set-name={name}"

    def acquire(self, sites: list[str]) -> SiteAllocation:
        missing = [s for s in sites if s not in self._sets]
        if missing:
            raise SitePoolError(
                f"no SandboxSet configured for sites {missing} "
                f"(configured: {sorted(self._sets)})"
            )
        claimed: list = []
        try:
            base_urls = {}
            for site in sites:
                pod = self._claim_one(site)
                claimed.append(pod)
                port = self._map_port if site == "map" else self._site_port
                base_urls[site] = f"http://{pod.status.pod_ip}:{port}"
            pods = list(claimed)
            return SiteAllocation(
                base_urls, _releaser=lambda: self._release_pods(pods),
            )
        except Exception:
            self._release_pods(claimed)
            raise

    def _claim_one(self, site: str):
        selector = self._LABEL_TPL.format(name=self._sets[site])
        deadline = time.monotonic() + self._timeout
        while True:
            pods = self._api.list_namespaced_pod(
                self._ns, label_selector=selector,
            ).items
            if not pods and time.monotonic() > deadline:
                raise SitePoolError(
                    f"no pods at all under selector {selector!r} — SandboxSet "
                    f"missing or label key differs (pilot-calibrate _LABEL_TPL)"
                )
            candidates = [
                p for p in pods
                if getattr(p.status, "phase", "") == "Running"
                and not (p.metadata.annotations or {}).get(_CLAIM_ANNOTATION)
                and not p.metadata.deletion_timestamp
                and all(cs.ready for cs in (p.status.container_statuses or []))
            ]
            for pod in candidates:
                if self._try_claim(pod):
                    return pod
            if time.monotonic() > deadline:
                raise SitePoolError(
                    f"acquire timeout for site {site!r}: {len(pods)} pods, "
                    f"0 claimable after {self._timeout:.0f}s (pool exhausted — "
                    f"raise SandboxSet replicas or lower concurrency)"
                )
            time.sleep(2.0)

    def _try_claim(self, pod) -> bool:
        """Annotation CAS: replace with resourceVersion — a concurrent claim
        loses with 409 and moves on to the next candidate."""
        pod.metadata.annotations = dict(pod.metadata.annotations or {})
        pod.metadata.annotations[_CLAIM_ANNOTATION] = self._claimant
        try:
            self._api.replace_namespaced_pod(
                pod.metadata.name, self._ns, pod,
            )
            return True
        except Exception as err:  # noqa: BLE001 — 409 冲突 = 被别人抢了
            logger.debug("claim lost for %s: %s", pod.metadata.name, err)
            return False

    def _release_pods(self, pods: list) -> None:
        """Delete claimed pods; the SandboxSet controller restores fresh
        replicas from the checkpoint (this deletion IS the state reset)."""
        from kubernetes import client as k8s_client

        for pod in pods:
            try:
                self._api.delete_namespaced_pod(
                    pod.metadata.name, self._ns,
                    body=k8s_client.V1DeleteOptions(grace_period_seconds=0),
                )
            except Exception as err:  # noqa: BLE001
                logger.warning("pod delete failed (%s): %s",
                               pod.metadata.name, err)
