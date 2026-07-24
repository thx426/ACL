"""shared_task — the research-session dedupe primitive for rollout_n > 1.

The invariant under test: within one (step, task) key, the factory runs
EXACTLY once no matter how many copies await it concurrently; keys from an
older step are dropped on entry; one cancelled awaiter must not kill the
shared job for its siblings.
"""
import asyncio

import pytest

from trace_opd.common.aio import shared_task


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def test_concurrent_copies_share_one_factory_run():
    async def main():
        cache: dict = {}
        calls = []

        async def factory():
            calls.append(1)
            await asyncio.sleep(0.01)  # force the copies to overlap
            return ("doc", {"research_experiments": 3})

        results = await asyncio.gather(
            *[shared_task(cache, (7, "task/A"), factory) for _ in range(8)]
        )
        assert len(calls) == 1
        assert all(r == ("doc", {"research_experiments": 3}) for r in results)

    _run(main())


def test_distinct_tasks_and_steps_run_separately():
    async def main():
        cache: dict = {}
        seen = []

        def make(tag):
            async def factory():
                seen.append(tag)
                return tag

            return factory

        a = await shared_task(cache, (1, "A"), make("s1-A"))
        b = await shared_task(cache, (1, "B"), make("s1-B"))
        assert (a, b) == ("s1-A", "s1-B") and len(cache) == 2
        # New step: old entries are evicted, same task researches again.
        a2 = await shared_task(cache, (2, "A"), make("s2-A"))
        assert a2 == "s2-A"
        assert set(cache) == {(2, "A")}
        assert seen == ["s1-A", "s1-B", "s2-A"]

    _run(main())


def test_cancelling_one_awaiter_spares_the_shared_job():
    async def main():
        cache: dict = {}
        started = asyncio.Event()

        async def factory():
            started.set()
            await asyncio.sleep(0.05)
            return "survived"

        first = asyncio.create_task(shared_task(cache, (1, "A"), factory))
        await started.wait()
        second = asyncio.create_task(shared_task(cache, (1, "A"), factory))
        await asyncio.sleep(0)  # let `second` attach to the cached task
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert await second == "survived"  # sibling still gets the result

    _run(main())
