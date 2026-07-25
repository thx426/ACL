"""PooledClient(多端点池)单元测试:构造分支、会话粘性、least-active 均衡、
故障切换与重粘、全端点失败。风格与 test_backends_routing.py 一致
(monkeypatch `_client.post`,不起真实网络)。"""
import asyncio

import httpx
import pytest

from trace_opd.common.llm import LLMError, OpenAICompatClient, PooledClient, make_client


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


class _Cfg:
    """最小 RoleCfg 替身(make_client 只读这些字段)。"""

    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.model = "m"
        self.api_key = "k"
        self.max_concurrency = 8
        self.temperature = 0.7
        self.max_tokens = 64
        self.timeout_s = 30.0
        self.extra_body = {}
        self.dp_size = 2  # 单端点时应保留;池模式应被强制归零


class _Resp:
    def __init__(self, status_code=200):
        self.status_code = status_code
        self.text = "err"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "boom", request=httpx.Request("POST", "http://x"),
                response=httpx.Response(self.status_code, text="boom"),
            )

    def json(self):
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


def _wire(client: OpenAICompatClient, log: list, status=200, gate: asyncio.Event | None = None):
    async def fake_post(url, json=None, headers=None):
        log.append(url)
        if gate is not None:
            await gate.wait()
        return _Resp(status)

    client._client.post = fake_post


def test_make_client_single_endpoint_unchanged():
    c = make_client("researcher", _Cfg("http://a:1/v1"))
    assert isinstance(c, OpenAICompatClient)
    assert c._dp_size == 2  # 单端点保留 DP 头路由
    _run(c.aclose())


def test_make_client_comma_endpoint_builds_pool():
    c = make_client("researcher", _Cfg("http://a:1/v1, http://b:2/v1"))
    assert isinstance(c, PooledClient)
    assert len(c._clients) == 2
    for inner in c._clients:
        assert inner._dp_size == 0          # 独立副本禁 DP 头
        assert inner._max_retries == PooledClient.INNER_RETRIES
    assert c.model == "m" and "a:1" in c.endpoint and "b:2" in c.endpoint
    _run(c.aclose())


def test_pool_session_sticky_same_key_same_endpoint():
    pool = make_client("researcher", _Cfg("http://a:1/v1,http://b:2/v1"))
    log0, log1 = [], []
    _wire(pool._clients[0], log0)
    _wire(pool._clients[1], log1)

    async def main():
        for _ in range(3):
            await pool.chat([{"role": "user", "content": "hi"}], route_key="task/A")
        await pool.aclose()

    _run(main())
    # 三次调用全部落在同一端点(粘住),另一端点为空
    assert (len(log0), len(log1)) in {(3, 0), (0, 3)}


def test_pool_least_active_spreads_concurrent_sessions():
    pool = make_client("researcher", _Cfg("http://a:1/v1,http://b:2/v1"))
    log0, log1 = [], []
    gate = asyncio.Event()
    _wire(pool._clients[0], log0, gate=gate)
    _wire(pool._clients[1], log1, gate=gate)

    async def main():
        # 两个会话并发在飞:第二个会话选端点时应看见第一个的 active=1 而选另一端
        t1 = asyncio.create_task(
            pool.chat([{"role": "user", "content": "1"}], route_key="task/A")
        )
        await asyncio.sleep(0.01)
        t2 = asyncio.create_task(
            pool.chat([{"role": "user", "content": "2"}], route_key="task/B")
        )
        await asyncio.sleep(0.01)
        gate.set()
        await asyncio.gather(t1, t2)
        await pool.aclose()

    _run(main())
    assert len(log0) == 1 and len(log1) == 1  # 一边一个,均衡


def test_pool_failover_resticks_and_counts():
    pool = make_client("researcher", _Cfg("http://a:1/v1,http://b:2/v1"))
    # 先用一次成功调用确定 task/A 粘在哪个端点,再把该端点打坏
    log0, log1 = [], []
    _wire(pool._clients[0], log0)
    _wire(pool._clients[1], log1)

    async def main():
        await pool.chat([{"role": "user", "content": "warm"}], route_key="task/A")
        stuck = pool._sticky["task/A"]
        other = 1 - stuck
        # 粘住的端点返回 4xx(立即 LLMError,不走退避)⇒ 应切换到另一端点并重粘
        _wire(pool._clients[stuck], [], status=400)
        r = await pool.chat([{"role": "user", "content": "hi"}], route_key="task/A")
        assert r.text == "ok"
        assert pool._sticky["task/A"] == other          # 重粘
        assert pool._stats[other]["failovers_in"] == 1  # 切换计数
        assert pool._stats[stuck]["failures"] >= 1
        await pool.aclose()

    _run(main())


def test_pool_all_endpoints_failed_raises():
    pool = make_client("researcher", _Cfg("http://a:1/v1,http://b:2/v1"))
    _wire(pool._clients[0], [], status=400)
    _wire(pool._clients[1], [], status=400)

    async def main():
        with pytest.raises(LLMError, match="all 2 endpoints failed"):
            await pool.chat([{"role": "user", "content": "hi"}], route_key="task/A")
        await pool.aclose()

    _run(main())
