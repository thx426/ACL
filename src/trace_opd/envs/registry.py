"""Environment registry — the ONLY place assembly code resolves a benchmark.

机制层(core/steering/research/train/selfres)永不 import 具体 benchmark 模块;
装配层(agent_loop._shared_state、evalm/run_eval、scripts/sftgen_researcher)
一律经 ``make_env_factory(config)`` 拿工厂,凭 ``environment.name`` 分派。

引入新 benchmark 的注册方式(参照 envs/_template/README.md):
  1. 实现 ``core.adapter.EnvFactory``(adapter_from_item → TaskAdapter);
  2. 在本文件 ``_BUILDERS`` 加一行 builder(或运行时 register_environment);
  3. 配置写 ``environment: {name: <键>}``。

builder 约定:
  * 签名 ``(config, workers) -> EnvFactory``。workers = 本进程所属的并发
    装配份数(训练 = agent worker 数;eval/SFT 单进程 = 1)。配置里的全局
    资源上限(如引擎 boot 并发)按 workers 折算是 builder 自己的责任——
    资源语义归环境,不归训练装配层。
  * import 具体 benchmark 模块必须在 builder 函数体内(延迟);本模块顶部
    保持零重依赖,任何进程(Ray worker/脚本)import 本模块都必须是瞬时的。
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # 仅类型;运行时零依赖
    from ..common.config import Config
    from ..core.adapter import EnvFactory


def _per_worker(cap: int, workers: int) -> int:
    """配置里的资源上限是全局的;N 份装配各拿一份。"""
    return max(1, math.ceil(cap / max(1, workers)))


def _fork_cfg(cfg, **overrides):
    """Per-worker 折算必须作用在副本上——原地改写会让同一 Config 被多次
    装配时反复再折算(审阅 P2-6:16→4→1)。"""
    import dataclasses

    return dataclasses.replace(cfg, **overrides)


def _build_alfworld(config: "Config", workers: int) -> "EnvFactory":
    from .alfworld.adapter import AlfworldEnvFactory

    cfg = config.alfworld
    return AlfworldEnvFactory(_fork_cfg(
        cfg, spawn_concurrency=_per_worker(cfg.spawn_concurrency, workers)))


def _build_spreadsheetbench(config: "Config", workers: int) -> "EnvFactory":
    from .spreadsheetbench.adapter import SpreadsheetbenchEnvFactory

    cfg = config.spreadsheetbench
    return SpreadsheetbenchEnvFactory(_fork_cfg(
        cfg, exec_concurrency=_per_worker(cfg.exec_concurrency, workers)))


def _build_math(config: "Config", workers: int) -> "EnvFactory":
    from .math.adapter import MathEnvFactory

    cfg = config.math
    return MathEnvFactory(_fork_cfg(
        cfg, exec_concurrency=_per_worker(cfg.exec_concurrency, workers)))


def _build_scienceworld(config: "Config", workers: int) -> "EnvFactory":
    from .scienceworld.adapter import SciworldEnvFactory

    cfg = config.scienceworld
    return SciworldEnvFactory(_fork_cfg(
        cfg, spawn_concurrency=_per_worker(cfg.spawn_concurrency, workers)))


def _build_searchqa(config: "Config", workers: int) -> "EnvFactory":
    from .searchqa.adapter import SearchQAEnvFactory

    cfg = config.searchqa
    return SearchQAEnvFactory(_fork_cfg(
        cfg, search_concurrency=_per_worker(cfg.search_concurrency, workers)))


def _build_webshop(config: "Config", workers: int) -> "EnvFactory":
    from .webshop.adapter import WebshopEnvFactory

    cfg = config.webshop
    return WebshopEnvFactory(_fork_cfg(
        cfg, request_concurrency=_per_worker(cfg.request_concurrency, workers)))


def _build_terminal(config: "Config", workers: int) -> "EnvFactory":
    from .terminal.adapter import TerminalEnvFactory

    cfg = config.terminal
    return TerminalEnvFactory(_fork_cfg(
        cfg, spawn_concurrency=_per_worker(cfg.spawn_concurrency, workers)))


def _build_swe(config: "Config", workers: int) -> "EnvFactory":
    from .swe.adapter import SweEnvFactory

    cfg = config.swe
    return SweEnvFactory(_fork_cfg(
        cfg, spawn_concurrency=_per_worker(cfg.spawn_concurrency, workers)))


def _build_webarena(config: "Config", workers: int) -> "EnvFactory":
    """⚠ spawn_concurrency 是 **per-worker** 折算值(全局 cap / workers,
    向上取整、下限 1)——workers > cap 时总并发会超过名义 cap;真正的
    跨进程全局预算需要外部协调(审阅 P0-3 遗留,P-B 按机器资源核算)。"""
    from .webarena.adapter import WebArenaEnvFactory

    cfg = config.webarena
    return WebArenaEnvFactory(_fork_cfg(
        cfg, spawn_concurrency=_per_worker(cfg.spawn_concurrency, workers)))


_BUILDERS: dict[str, Callable[["Config", int], "EnvFactory"]] = {
    "alfworld": _build_alfworld,
    "spreadsheetbench": _build_spreadsheetbench,
    "math": _build_math,
    "searchqa": _build_searchqa,
    "webshop": _build_webshop,
    "scienceworld": _build_scienceworld,
    "terminal": _build_terminal,
    "webarena": _build_webarena,
    "swe": _build_swe,
}


def register_environment(
    name: str, builder: Callable[["Config", int], "EnvFactory"]
) -> None:
    """运行时注册(测试/插件用;内置 benchmark 直接写进 _BUILDERS)。"""
    if name in _BUILDERS:
        raise ValueError(f"environment {name!r} already registered")
    _BUILDERS[name] = builder


def make_env_factory(config: "Config", workers: int = 1) -> "EnvFactory":
    name = config.environment.name
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise ValueError(
            f"unknown environment {name!r} (registered: {sorted(_BUILDERS)}); "
            "add a builder in envs/registry.py — see envs/_template/README.md"
        ) from None
    return builder(config, workers)
