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


def _build_alfworld(config: "Config", workers: int) -> "EnvFactory":
    from .alfworld.adapter import AlfworldEnvFactory

    cfg = config.alfworld
    cfg.spawn_concurrency = _per_worker(cfg.spawn_concurrency, workers)
    return AlfworldEnvFactory(cfg)


def _build_spreadsheetbench(config: "Config", workers: int) -> "EnvFactory":
    from .spreadsheetbench.adapter import SpreadsheetbenchEnvFactory

    cfg = config.spreadsheetbench
    cfg.exec_concurrency = _per_worker(cfg.exec_concurrency, workers)
    return SpreadsheetbenchEnvFactory(cfg)


_BUILDERS: dict[str, Callable[["Config", int], "EnvFactory"]] = {
    "alfworld": _build_alfworld,
    "spreadsheetbench": _build_spreadsheetbench,
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
