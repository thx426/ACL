"""SGLang 适配层纯函数语义测试(2026-08-05 引擎切换,分支 pgs-sglang)。

被测对象是 vendored verl 的 async_sglang_server 里两个零依赖纯函数——
`_extract_prompt_logprobs_sglang`(verify 侧:vLLM extract 契约 + tail 窗口)
与 `_extract_response_topk_sglang`(draft 侧:response_topk_* 契约)。该模块
顶层 import sglang/ray/torch(本地测试环境无),故用 ast 抽取函数源码执行,
只锁语义不碰引擎。行语义 off-by-one(start=0 跳 None 首行 / start>0 不跳)
是本次迁移最大的静默错位风险,双向 fail-loud 都要有回归。
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
    wanted = {"_extract_prompt_logprobs_sglang", "_extract_response_topk_sglang"}
    ns: dict = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            mod = ast.Module(body=[node], type_ignores=[])
            exec(compile(mod, str(_SRC), "exec"), ns)  # noqa: S102 — 本仓源码
    assert wanted <= set(ns), f"missing {wanted - set(ns)}"
    return ns["_extract_prompt_logprobs_sglang"], ns["_extract_response_topk_sglang"]


_extract_prompt, _extract_topk = _load_pure_functions()


def _row(k, base):
    """一行 ranked top-K:[(logprob, token_id, text), ...] best-first。"""
    return [(-0.1 * (r + 1) - base, 1000 * base + r, None) for r in range(k)]


# ------------------------------------------------ verify 侧(prompt logprobs)

def test_prompt_logprobs_full_prefix_skips_none_head():
    """start=0(全前缀,蒸馏兼容口径):首项 None 跳过,总行=sequence_length。"""
    L, K = 6, 4
    meta = {
        "input_token_logprobs": [(None, 7, None)] + [(-0.5, 10 + i, None) for i in range(L - 1)],
        "input_top_logprobs": [None] + [_row(K, i) for i in range(L - 1)],
    }
    out: dict = {}
    _extract_prompt(meta, K, L, out, logprob_start_len=0)
    assert len(out["prompt_ids"]) == L                      # L-1 真实 + 1 dummy
    assert out["prompt_ids"][0] == [1000 * 0 + r for r in range(K)]
    assert out["prompt_ids"][-1] == [0] * K                 # dummy 尾行
    assert out["prompt_logprobs"][0][0] == pytest.approx(-0.1)


def test_prompt_logprobs_tail_window_keeps_first_row():
    """start>0(tail-only,SPS verify 口径):首项是真实行,不跳;
    总行 = L − start + 1(= hook 的 tail=T 契约:T−1 真实 + dummy)。"""
    L, K, tail = 10, 3, 4                                   # start = L-(T-1) = 7
    start = L - (tail - 1)
    n_real = L - start                                      # = 3
    meta = {
        "input_token_logprobs": [(-0.5, 20 + i, None) for i in range(n_real)],
        "input_top_logprobs": [_row(K, i) for i in range(n_real)],
    }
    out: dict = {}
    _extract_prompt(meta, K, L, out, logprob_start_len=start)
    assert len(out["prompt_ids"]) == tail                   # 3 真实 + 1 dummy = T
    assert out["prompt_ids"][0] == [1000 * 0 + r for r in range(K)]  # 首行保留
    assert out["prompt_ids"][-1] == [0] * K


def test_prompt_logprobs_row_convention_drift_fails_loud():
    """行语义漂移必须炸,不许静默错位(迁移最大风险点的双向回归)。"""
    # start=0 但首项竟有真实 logprob ⇒ 跳首行会丢真实行
    meta = {"input_token_logprobs": [(-0.5, 7, None), (-0.6, 8, None)],
            "input_top_logprobs": [_row(2, 0), _row(2, 1)]}
    with pytest.raises(RuntimeError, match="refusing to skip"):
        _extract_prompt(meta, 2, 3, {}, logprob_start_len=0)
    # start>0 但首项是 None ⇒ 窗口化语义被破坏
    meta2 = {"input_token_logprobs": [(None, 7, None), (-0.6, 8, None)],
             "input_top_logprobs": [None, _row(2, 1)]}
    with pytest.raises(RuntimeError, match="should all have context"):
        _extract_prompt(meta2, 2, 5, {}, logprob_start_len=3)


def test_prompt_logprobs_width_mismatch_fails():
    """top 行宽 != 请求 K ⇒ assert(候选集不许有洞)。"""
    meta = {"input_token_logprobs": [(-0.5, 20, None)],
            "input_top_logprobs": [_row(2, 0)]}             # 宽 2,请求 3
    with pytest.raises(AssertionError):
        _extract_prompt(meta, 3, 3, {}, logprob_start_len=2)


# ------------------------------------------------ draft 侧(response top-K)

def test_response_topk_shapes_and_rank_order():
    K, n = 4, 3
    meta = {"output_top_logprobs": [_row(K, i) for i in range(n)]}
    out: dict = {}
    _extract_topk(meta, K, n, "req-1", out)
    assert len(out["response_topk_ids"]) == n
    assert all(len(r) == K for r in out["response_topk_ids"])
    # rank 序保持(best-first):行内 logprob 单调下降
    for lps in out["response_topk_logprobs"]:
        assert lps == sorted(lps, reverse=True)


def test_response_topk_fails_loud_on_holes():
    # 行数 != 生成 token 数
    with pytest.raises(RuntimeError, match="rows"):
        _extract_topk({"output_top_logprobs": [_row(2, 0)]}, 2, 3, "req-2", {})
    # 行宽不足 K(IF-08 同源:不留哨兵毒化候选集)
    with pytest.raises(RuntimeError, match="coverage broken"):
        _extract_topk({"output_top_logprobs": [_row(2, 0)]}, 5, 1, "req-3", {})
