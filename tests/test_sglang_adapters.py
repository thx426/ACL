"""SGLang 适配层纯函数语义测试(2026-08-05 引擎切换,分支 pgs-sglang)。

被测对象是 vendored verl 的 async_sglang_server 里三个零依赖纯函数——
`_extract_prompt_logprobs_sglang`(verify 侧)、`_extract_response_topk_sglang`
(draft 侧)与 `_normalize_finish_reason`(abort 契约)。模块顶层 import
sglang/ray/torch(本地测试环境无),故用 ast 抽取函数源码执行。

Oracle 口径(codex 审阅 P0 修正,2026-08-05):x[i] 的 logprob 来自前一位置的
logits,extend 从 start 前向只产出 logits[start:] ⇒ 窗口首位 x[start] 物理不可
打分,SGLang 对**每个**请求(start=0 与 start>0 同样)都返回 None 首项。契约 =
无条件丢 None 头 + 追加 dummy,总行数 = L − start;tail=T 的调用方必须请求
start = max(0, L−T)。mock payload 按 0.5.8 真实形态构造(首项恒 None),并做
逐 token 位置身份断言而非只断言行数。最终确认以 PoC 门 3 实测为准。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / (
    "verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py"
)


def _load_pure_functions():
    tree = ast.parse(_SRC.read_text())
    wanted = {
        "_extract_prompt_logprobs_sglang",
        "_extract_response_topk_sglang",
        "_normalize_finish_reason",
    }
    ns: dict = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            mod = ast.Module(body=[node], type_ignores=[])
            exec(compile(mod, str(_SRC), "exec"), ns)  # noqa: S102 — 本仓源码
    assert wanted <= set(ns), f"missing {wanted - set(ns)}"
    return (
        ns["_extract_prompt_logprobs_sglang"],
        ns["_extract_response_topk_sglang"],
        ns["_normalize_finish_reason"],
    )


_extract_prompt, _extract_topk, _norm_finish = _load_pure_functions()

# 一条固定"序列":token id = 100+i,位置 i 的预测 logprob = -0.01*(i+1)。
_TOK = lambda i: 100 + i  # noqa: E731
_LP = lambda i: -0.01 * (i + 1)  # noqa: E731


def _payload_05_8(L: int, start: int, k: int):
    """按 sglang 0.5.8 真实形态构造 input logprobs:覆盖位置 [start, L),
    首项 = (None, token_id@start, None)(窗口头不可打分),其余 =
    (真实 logprob, token_id, None);top 行同窗口,首项占位 None。"""
    tok_rows = [(None, _TOK(start), None)] + [
        (_LP(i), _TOK(i), None) for i in range(start + 1, L)
    ]
    top_rows = [None] + [
        [(_LP(i) - 0.001 * r, 1000 * i + r, None) for r in range(k)]
        for i in range(start + 1, L)
    ]
    return {"input_token_logprobs": tok_rows, "input_top_logprobs": top_rows}


# ------------------------------------------------ verify 侧(prompt logprobs)

def test_prompt_logprobs_full_prefix():
    """start=0:丢 None 头 → L−1 真实行 + dummy = L 行;逐位置身份对账。"""
    L, K = 6, 4
    out: dict = {}
    _extract_prompt(_payload_05_8(L, 0, K), K, L, out, logprob_start_len=0)
    assert len(out["prompt_ids"]) == L
    # 第 r 行预测位置 r+1(消费端契约):top-1 id = 1000*(r+1)+0
    for r in range(L - 1):
        assert out["prompt_ids"][r][0] == 1000 * (r + 1)
        assert out["prompt_logprobs"][r][0] == pytest.approx(_LP(r + 1))
    assert out["prompt_ids"][-1] == [0] * K  # dummy 尾行


def test_prompt_logprobs_tail_window_matches_vllm_contract():
    """tail=T ⇒ start=max(0,L−T):None 头丢弃后恰为预测位置 [L−(T−1), L) 的
    T−1 真实行 + dummy = T 行,与 vLLM extract tail=T 逐行同位(codex P0 修正:
    此前 start=L−(T−1) 差一位,正常请求必炸)。"""
    L, K, n = 10, 3, 3          # draft 段 n=3,hook 请求 T=n+1=4
    T = n + 1
    start = max(0, L - T)       # = 6
    out: dict = {}
    _extract_prompt(_payload_05_8(L, start, K), K, L, out, logprob_start_len=start)
    assert len(out["prompt_ids"]) == T                    # 3 真实 + 1 dummy
    # 真实行 r 预测位置 start+1+r = 7,8,9 —— 正是 draft 段三个 token
    for r in range(T - 1):
        pos = start + 1 + r
        assert out["prompt_ids"][r][0] == 1000 * pos
        assert out["prompt_logprobs"][r][0] == pytest.approx(_LP(pos))
    assert out["prompt_ids"][-1] == [0] * K


def test_prompt_logprobs_short_prompt_clamps_to_full():
    """T ≥ L 时 start clamp 到 0,退化为全前缀行为(行数 = L)。"""
    L, K = 3, 2
    start = max(0, L - 5)       # = 0
    out: dict = {}
    _extract_prompt(_payload_05_8(L, start, K), K, L, out, logprob_start_len=start)
    assert len(out["prompt_ids"]) == L


def test_prompt_logprobs_row_convention_drift_fails_loud():
    """首项必须是 None(窗口头物理不可打分);出现真实值 = 行语义漂移 ⇒ 炸。"""
    L, K, start = 8, 2, 4
    meta = _payload_05_8(L, start, K)
    meta["input_token_logprobs"][0] = (_LP(start), _TOK(start), None)  # 篡改首项
    with pytest.raises(RuntimeError, match="physically unscorable"):
        _extract_prompt(meta, K, L, {}, logprob_start_len=start)
    # 空 payload 同样 fail loud(旧实现会静默出 dummy-only)
    with pytest.raises(RuntimeError, match="empty input_token_logprobs"):
        _extract_prompt({"input_token_logprobs": []}, K, L, {}, logprob_start_len=start)


def test_prompt_logprobs_width_and_rowcount_fail_without_assert():
    """宽度/行数防线必须是 RuntimeError(python -O 下 assert 会消失)。"""
    L, K, start = 8, 3, 5
    meta = _payload_05_8(L, start, K)
    meta["input_top_logprobs"][1] = meta["input_top_logprobs"][1][:2]  # 行宽 2<3
    with pytest.raises(RuntimeError, match="poison the steering candidate set"):
        _extract_prompt(meta, K, L, {}, logprob_start_len=start)
    meta2 = _payload_05_8(L, start, K)
    meta2["input_token_logprobs"] = meta2["input_token_logprobs"][:-1]  # 少一行
    with pytest.raises(RuntimeError, match="!= expected"):
        _extract_prompt(meta2, K, L, {}, logprob_start_len=start)


# ------------------------------------------------ draft 侧(response top-K)

def _row(k, base):
    return [(-0.1 * (r + 1) - base, 1000 * base + r, None) for r in range(k)]


def test_response_topk_shapes_and_rank_order():
    K, n = 4, 3
    meta = {"output_top_logprobs": [_row(K, i) for i in range(n)]}
    out: dict = {}
    _extract_topk(meta, K, n, "req-1", out)
    assert len(out["response_topk_ids"]) == n
    assert all(len(r) == K for r in out["response_topk_ids"])
    for lps in out["response_topk_logprobs"]:
        assert lps == sorted(lps, reverse=True)  # best-first 保序


def test_response_topk_fails_loud_on_holes():
    with pytest.raises(RuntimeError, match="rows"):
        _extract_topk({"output_top_logprobs": [_row(2, 0)]}, 2, 3, "req-2", {})
    with pytest.raises(RuntimeError, match="coverage broken"):
        _extract_topk({"output_top_logprobs": [_row(2, 0)]}, 5, 1, "req-3", {})


# ------------------------------------------------ abort 契约(finish reason)

def test_finish_reason_normalized_to_vllm_domain():
    """SGLang stop|length|abort → vLLM completed|aborted(codex P0:消费端只判
    "aborted",透传 "abort" 会把基建中止静默当普通失败轨迹)。"""
    assert _norm_finish("abort") == "aborted"
    assert _norm_finish("stop") == "completed"
    assert _norm_finish("length") == "completed"
    assert _norm_finish(None) is None
    assert _norm_finish("weird_future_type") == "weird_future_type"  # 透传同 vLLM
