"""EngineGenerate sticky-routing key — the prefix-cache affinity fix
(user issue 2026-07-23): the request_id passed to verl's server manager is a
pure routing key (verl sends a fresh uuid to vLLM per call), so all engine
calls about one (step, task) may — and should — share one key.
"""
import asyncio

from trace_opd.train.backends import EngineGenerate


class _FakeServer:
    def __init__(self):
        self.request_ids = []

    async def generate(self, *, request_id, prompt_ids, sampling_params):
        self.request_ids.append(request_id)

        class Out:
            token_ids = [1, 2]
            extra_fields = {}
            stop_reason = "stop"

        return Out()


class _FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "x"


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def test_explicit_routing_key_is_used_for_every_turn():
    server = _FakeServer()

    async def main():
        gen = EngineGenerate(server, _FakeTokenizer(), request_id="task:7:t/A")
        await gen([1], {"max_tokens": 4})
        await gen([1, 2, 3], {"max_tokens": 4})  # later turn, same episode

    _run(main())
    assert server.request_ids == ["task:7:t/A", "task:7:t/A"]


def test_instances_sharing_a_key_pin_to_the_same_route():
    # rollout_n copies + research episodes each build their own EngineGenerate;
    # passing the same (step, task) key makes them one sticky session.
    server = _FakeServer()

    async def main():
        for _ in range(3):
            gen = EngineGenerate(server, _FakeTokenizer(), request_id="task:7:t/A")
            await gen([1], {})

    _run(main())
    assert set(server.request_ids) == {"task:7:t/A"}


def test_default_stays_per_episode_identity():
    server = _FakeServer()

    async def main():
        a, b = EngineGenerate(server, _FakeTokenizer()), EngineGenerate(server, _FakeTokenizer())
        await a([1], {})
        await a([1, 2], {})
        await b([1], {})

    _run(main())
    ids = server.request_ids
    assert ids[0] == ids[1] and ids[2] != ids[0]  # per-instance stable, distinct across


# ---------------------------------------------------------------- 27B DP rank


def test_dp_rank_for_is_deterministic_and_bounded():
    from trace_opd.common.llm import dp_rank_for

    keys = [f"pick_and_place/task-{i}" for i in range(32)]
    ranks = [dp_rank_for(k, 2) for k in keys]
    assert ranks == [dp_rank_for(k, 2) for k in keys]  # stable across calls
    assert set(ranks) <= {0, 1}
    assert len(set(ranks)) == 2  # both engines actually get traffic


def test_chat_sends_dp_rank_header_only_when_routed():
    import asyncio

    from trace_opd.common.llm import OpenAICompatClient, dp_rank_for

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    seen = []

    async def fake_post(url, json=None, headers=None):
        seen.append(headers)
        return _Resp()

    async def main():
        client = OpenAICompatClient(
            role="researcher", endpoint="http://x:1/v1", model="m", dp_size=2
        )
        client._client.post = fake_post
        await client.chat([{"role": "user", "content": "hi"}], route_key="t/A")
        await client.chat([{"role": "user", "content": "hi"}])  # no route_key
        await client.aclose()

    _run(main())
    assert seen[0] == {"X-data-parallel-rank": str(dp_rank_for("t/A", 2))}
    assert seen[1] is None  # unrouted request adds no header
