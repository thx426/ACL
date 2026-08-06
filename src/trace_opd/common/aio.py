"""Async helpers that must stay importable without verl/torch (unit-tested
locally; agent_loop.py wires them into the verl worker process).
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

__all__ = ["shared_task"]


async def shared_task(
    cache: dict[tuple, asyncio.Task],
    key: tuple,
    factory: Callable[[], Awaitable[Any]],
) -> Any:
    """Memoize an async job per ``key`` within one process/event loop.

    Purpose (rollout_n > 1): the n interleaved copies of one task inside a
    training step must share ONE research session — the first caller starts
    it, the rest await the same result. Correctness rests on two invariants:

    - All copies of a task land in the same AgentLoopWorker process
      (interleave repeat + contiguous chunking; enforced by the
      ``train_batch_size % agent_num_workers == 0`` config check), and all
      samples of a worker run on one event loop — so the get-or-create below
      has no await point and needs no lock.
    - ``key[0]`` is the training step: entries from other steps are dropped
      on entry (the previous step's tasks are past the gather barrier, hence
      complete; popping never cancels a task still referenced by awaiters).

    ``asyncio.shield`` keeps one cancelled copy from killing the shared job
    for its siblings; the cancellation still propagates to the caller.
    """
    for stale in [k for k in cache if k[0] != key[0]]:
        cache.pop(stale)
    task = cache.get(key)
    if task is None:
        task = asyncio.create_task(factory())
        cache[key] = task
    return await asyncio.shield(task)
