#!/usr/bin/env python3
"""Qwen3.5-9B × SGLang 0.5.10 PoC 探针(门 2/3 + thinking 探测)。

与生产同口径:PGS verify 走 native /generate 的
return_logprob + logprob_start_len + top_logprobs_num(hook._verify_rows 同款),
故本探针的 PASS/FAIL 直接映射机制可行性,不是泛化健康检查。

用法(A800,先起服):
  python -m sglang.launch_server --model-path $Q35_MODEL --tp 1 \
      --mem-fraction-static 0.7 --port 30000
  python scripts/sgl_probe_q35.py --base http://127.0.0.1:30000 [--stage all]

门 3 判据(docs/qwen35-9b-env-plan.md §3.3):
  硬门(任一 FAIL = 门 3 不过):A1 top 宽度 256;A2 tail 行数/null 模式正确;
    A3 tail 与 full 重叠行 token 序逐行一致(mamba state 下 tail 打分正确性)。
  软门(记录回报,不判死):B1 tail 相对 full 的降时比;B2 radix 复用增益
    (重复/共享前缀增量两种模式)。hybrid 的 mamba cache 是块粒度,收益低于
    162 全注意力实测(41×)是预期内,归零才需要重算 SPS 成本账。
thinking 探测(agent 调研发现的静默风险):SGLang 对未知 chat_template_kwargs
可能静默忽略 ⇒ enable_thinking:false 无效而无人知晓。三组对照判别。
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import requests

SENT = "The quick brown fox jumps over the lazy dog while the observer takes notes. "


def _gen(base: str, payload: dict, timeout: float = 300.0) -> dict:
    r = requests.post(f"{base}/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _score(base: str, text: str, start_len: int, top: int) -> tuple[dict, float]:
    t0 = time.monotonic()
    out = _gen(base, {
        "text": text,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
        "return_logprob": True,
        "logprob_start_len": start_len,
        "top_logprobs_num": top,
    })
    return out, time.monotonic() - t0


def _rows(out: dict) -> list:
    meta = out["meta_info"]
    for key in ("input_top_logprobs", "prompt_top_logprobs"):
        if key in meta and meta[key] is not None:
            return meta[key]
    raise KeyError(f"no prompt-side top logprobs in meta_info keys={sorted(meta)}")


def stage2_roundtrip(base: str) -> bool:
    out = _gen(base, {
        "text": "The capital of France is",
        "sampling_params": {"max_new_tokens": 8, "temperature": 0.0},
    })
    txt = out.get("text", "")
    ok = bool(txt.strip())
    print(f"[gate2] roundtrip text={txt!r} -> {'PASS' if ok else 'FAIL'}")
    return ok


def stage3_logprobs(base: str, top: int, tail: int, target_tokens: int) -> bool:
    prompt = SENT * max(1, target_tokens // 16)
    hard_ok = True

    full_times = []
    full_out = None
    for i in range(3):
        full_out, dt = _score(base, prompt, 0, top)
        full_times.append(dt)
        print(f"[gate3] full-{'ABC'[i]}: {dt:.3f}s")
    frows = _rows(full_out)
    n = len(frows)
    width = len(frows[1]) if n > 1 and frows[1] else 0
    a1 = width == top
    print(f"[gate3] A1 top-width: rows={n} first_null={frows[0] is None} "
          f"width={width} -> {'PASS' if a1 else 'FAIL'}")
    hard_ok &= a1

    tail_times = []
    tail_out = None
    start = n - tail
    for i in range(2):
        tail_out, dt = _score(base, prompt, start, top)
        tail_times.append(dt)
        print(f"[gate3] tail-{'AB'[i]}: {dt:.3f}s")
    trows = _rows(tail_out)
    a2 = (len(trows) == tail and trows[0] is None
          and all(r is not None for r in trows[1:]))
    print(f"[gate3] A2 tail-shape: rows={len(trows)} (want {tail}) "
          f"first_null={trows[0] is None} rest_non_null="
          f"{all(r is not None for r in trows[1:])} -> {'PASS' if a2 else 'FAIL'}")
    hard_ok &= a2

    # A3:重叠行(跳过各自 null 首行)token 序一致 + logprob 数值对拍。
    # full 的第 start+1..n-1 行 ↔ tail 的第 1..tail-1 行。
    mismatch, max_dlp = 0, 0.0
    for j in range(1, tail):
        fr, tr = frows[start + j], trows[j]
        if fr is None or tr is None:
            mismatch += 1
            continue
        fids = [e[1] for e in fr[:8]]
        tids = [e[1] for e in tr[:8]]
        if fids != tids:
            mismatch += 1
        max_dlp = max(max_dlp, max(abs(fe[0] - te[0]) for fe, te in zip(fr[:8], tr[:8])))
    a3 = mismatch == 0
    lvl = "PASS" if a3 else "FAIL"
    warn = "" if max_dlp < 1e-3 else f"  (WARN max|dlogp|={max_dlp:.2e},>1e-3 记录回报)"
    print(f"[gate3] A3 tail-vs-full consistency: mismatched_rows={mismatch}/{tail-1} "
          f"max|dlogp|={max_dlp:.2e} -> {lvl}{warn}")
    hard_ok &= a3

    # B1/B2 软门:计时与 radix 复用。
    full_c, tail_best = min(full_times), min(tail_times)
    ratio = tail_best / full_c if full_c > 0 else float("inf")
    print(f"[gate3] B1 timing: tail/full = {tail_best:.3f}/{full_c:.3f} = {ratio:.3f} "
          f"({'PASS' if ratio < 0.6 else 'SOFT-WARN'};162 全注意力语境实测 0.024)")
    # 共享前缀 + 新尾段(SPS verify 的真实增量形态)
    inc_prompt = prompt + "Meanwhile the second observer records a different note. "
    _, dt_inc = _score(base, inc_prompt, start, top)
    print(f"[gate3] B2 radix-incremental(共享前缀+新尾段): {dt_inc:.3f}s "
          f"(vs tail 重复 {tail_best:.3f}s / full 冷 {full_c:.3f}s;"
          "若 ≈full 则 mamba radix 对 logprobs 无复用,回报重算 SPS 成本账)")
    return hard_ok


def stage_thinking(base: str, model: str) -> None:
    """三组对照判别 enable_thinking 是否真实生效(静默忽略是最危险形态)。"""
    url = f"{base}/v1/chat/completions"
    msgs = [{"role": "user", "content": "9.11 和 9.9 哪个大?一句话回答。"}]
    outs = {}
    for name, extra in (("default", {}),
                        ("think_off", {"chat_template_kwargs": {"enable_thinking": False}}),
                        ("think_on", {"chat_template_kwargs": {"enable_thinking": True}})):
        body = {"model": model, "messages": msgs, "temperature": 0.0, "max_tokens": 512, **extra}
        try:
            r = requests.post(url, json=body, timeout=300)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"].get("content") or ""
            reasoning = r.json()["choices"][0]["message"].get("reasoning_content")
        except Exception as e:  # noqa: BLE001 —— 探针要把失败形态原样打给人看
            outs[name] = (f"<ERROR {e}>", None)
            continue
        outs[name] = (txt, reasoning)
        print(f"[think] {name}: has_think_tag={'<think>' in txt} "
              f"reasoning_content={'yes' if reasoning else 'no'} len={len(txt)} "
              f"head={txt[:80]!r}")
    off, on = outs.get("think_off", ("", None)), outs.get("think_on", ("", None))
    off_th = ("<think>" in off[0]) or bool(off[1])
    on_th = ("<think>" in on[0]) or bool(on[1])
    if off_th:
        print("[think] VERDICT: FAIL — enable_thinking:false 未生效(输出仍含 thinking)。")
    elif not on_th:
        print("[think] VERDICT: INFO — 两档均无 thinking(模型可能无 thinking 模式,"
              "或 kwarg 被静默忽略且默认即无 thinking;对训练语境是安全形态)。")
    else:
        print("[think] VERDICT: PASS — kwarg 真实生效(off 无/on 有)。")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30000")
    ap.add_argument("--model", default="qwen3.5-9b", help="served model name(仅 chat 层用)")
    ap.add_argument("--stage", default="all", choices=["2", "3", "thinking", "all"])
    ap.add_argument("--top", type=int, default=256)
    ap.add_argument("--tail", type=int, default=8)
    ap.add_argument("--prompt-tokens", type=int, default=2000)
    args = ap.parse_args()

    ok = True
    if args.stage in ("2", "all"):
        ok &= stage2_roundtrip(args.base)
    if args.stage in ("3", "all"):
        ok &= stage3_logprobs(args.base, args.top, args.tail, args.prompt_tokens)
    if args.stage in ("thinking", "all"):
        stage_thinking(args.base, args.model)
    print(f"[probe] hard gates -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
