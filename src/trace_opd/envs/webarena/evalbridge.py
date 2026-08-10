"""Bridge to the ``webarena-verified`` evaluation package (ServiceNow,
Apache-2.0, version pinned).

Verified evaluation is TERMINAL-ONLY: it consumes the agent's structured
final response (``FinalAgentResponse``: task_type / status / retrieved_data /
error_details) plus the full-episode network HAR — there is no per-step
score. ``TaskEvalResult.score`` is all-or-nothing {0.0, 1.0} (every evaluator
must hit 1.0), which the adapter reports as the episode score.

Concurrency design
------------------
``WebArenaVerified.__init__`` loads and validates the whole 812-task dataset
— too heavy per episode. This module keeps ONE process-level facade and
mutates the environment's active URL per evaluation under a lock
(``EnvironmentConfig.set_active_url``: appends unseen URLs and switches the
active index). URL accumulation across episodes is harmless-by-construction:
``derender_url`` maps ANY known URL back to the same site placeholder, so old
episodes' URLs remaining in the list can only widen recognition, never change
a verdict. Evaluation itself is offline CPU work on (response, HAR) — no live
site access — so serializing it behind the lock costs nothing next to
minutes-long episodes.

The package version is pinned hard: evaluator semantics ARE the reward
function; a silent upgrade would silently change rewards.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: 评测器语义 = reward 函数本体,版本硬钉;升级须经用户裁定 + 对拍。
PINNED_VERSION = "1.2.3"

#: 数据集内出现的全部站点(homepage 不在任何任务的 sites 中,不配)。
SITES = ("shopping", "shopping_admin", "gitlab", "reddit", "map", "wikipedia")

_lock = threading.Lock()
_facade = None


def _import_wav():
    import webarena_verified
    from importlib.metadata import version

    got = version("webarena-verified")
    if got != PINNED_VERSION:
        raise RuntimeError(
            f"webarena-verified version mismatch: installed {got}, pinned "
            f"{PINNED_VERSION} — evaluator semantics are the reward function; "
            f"re-pin deliberately or install the pinned version."
        )
    return webarena_verified


def verified_dataset_path() -> Path:
    """Path of the 812-task dataset JSON inside the installed package."""
    _import_wav()
    from webarena_verified.utils import get_package_assets_path

    return get_package_assets_path() / "dataset" / "webarena-verified.json"


def _get_facade():
    """Process-level singleton WebArenaVerified (dataset loaded once).
    Seeded with placeholder URLs for every site; real per-episode URLs are
    swapped in via set_active_url under the module lock."""
    global _facade
    with _lock:
        if _facade is None:
            _import_wav()
            from webarena_verified.api import WebArenaVerified
            from webarena_verified.types.config import (
                EnvironmentConfig, WebArenaVerifiedConfig,
            )
            from webarena_verified.types.task import WebArenaSite

            from .browser import ACCOUNTS

            environments = {
                WebArenaSite(site): EnvironmentConfig(
                    urls=[f"http://{site}.invalid"],   # 占位,评测时按集换入
                    use_header_login=(site == "shopping_admin"),
                    credentials=ACCOUNTS.get(site),
                )
                for site in SITES
            }
            _facade = WebArenaVerified(
                config=WebArenaVerifiedConfig(environments=environments)
            )
        return _facade


def task_row(task_id: int) -> dict:
    """Raw task definition (sites/start_urls/intent/eval) by id."""
    wa = _get_facade()
    with _lock:
        return wa.get_task(task_id).model_dump(mode="json")


@dataclass
class EvalOutcome:
    score: float          # {0.0, 1.0} (all-or-nothing, verified semantics)
    status: str           # success | partial_match | failure | error
    error_msg: str        # non-empty only for status == error
    detail: dict          # full TaskEvalResult dump (for trajectory records)


def evaluate(
    task_id: int,
    final_response: dict | None,
    har_path: Path | None,
    base_urls: dict[str, str],
) -> EvalOutcome:
    """Run verified terminal evaluation for one episode.

    ``final_response=None``(未提交/预算耗尽)不调包,直接 0——与 TMax
    「不提交=0」及 verified「缺 agent_response 即 fail」语义一致。评测器内部
    异常 → status=error + score 0(基建口径,调用方单列监控)。"""
    if final_response is None:
        return EvalOutcome(
            score=0.0, status="failure",
            error_msg="", detail={"reason": "no final response submitted"},
        )
    trace: object
    if har_path is not None and Path(har_path).is_file():
        trace = Path(har_path)
    else:
        # HAR 缺失:AgentResponse-only 任务仍可判;NetworkEvent 断言会失败
        # (而不是 error)。包拒绝空列表("Trace content list is empty"),
        # 用单个哑 entry 表达"无网络证据"——invalid 域名不可能匹配任何
        # expected_network_events,语义等价于空。
        logger.warning("HAR missing for task %s; evaluating with sentinel "
                       "empty trace", task_id)
        trace = [{
            "request": {"method": "GET",
                        "url": "http://trace-missing.invalid/",
                        "headers": [], "queryString": []},
            "response": {"status": 0, "headers": [], "content": {},
                         "redirectURL": ""},
        }]
    wa = _get_facade()
    with _lock:
        try:
            from webarena_verified.types.task import WebArenaSite

            for site, url in base_urls.items():
                env = wa.config.environments[WebArenaSite(site)]
                env.set_active_url(url.rstrip("/"))
            result = wa.evaluate_task(
                task_id=task_id,
                agent_response=final_response,
                network_trace=trace,
            )
        except Exception as err:  # noqa: BLE001 — 评测器崩溃按 error 口径外显
            logger.error("verified evaluation crashed for task %s: %s",
                         task_id, err)
            return EvalOutcome(score=0.0, status="error",
                               error_msg=str(err)[:500], detail={})
    return EvalOutcome(
        score=float(result.score),
        status=str(result.status),
        error_msg=str(result.error_msg or ""),
        detail=result.model_dump(mode="json"),
    )
