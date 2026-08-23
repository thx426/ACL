#!/usr/bin/env python3
"""MENTOR 单轮档的 prompt 渲染验证(纯 CPU,零 GPU,自包含)。

自包含 = 不 import 部署树的 src —— TRACE-OPD-TERM 正跑着 ALFWorld PGS 臂,
按纪律不碰它。三段 scaffold 在此逐字重抄,脚本自己做逐字对拍(见 CHECKS)。

验证什么(发车前的必要门):
  1. 渲染后的完整 prompt 长什么样(chat template 是否吃掉/改写标签)
  2. token 账:system / 题面 / 总计,对照 MENTOR 的 prompt 1024 上限
  3. 单轮档零工具块(TIR/FC 档会渲染 <tools>,两套协议不能打架)
  4. doc 注入后的长度(PGS 臂的 system 会被 inject_guidance 追加特权文档)
  5. ★ 风险面取证:`<think>` / `\\boxed{}` 是否被 Qwen2.5 tokenizer 切成
     单 token 还是碎片 —— 碎片化会抬高 zero-shot 不服从的概率
     (记忆 qwen25-format-compliance-oneshot:自定义标签 zero-shot 274/274 全灭)

用法: probe_single_prompt.py <model_path>
"""
from __future__ import annotations

import sys

MODEL = sys.argv[1] if len(sys.argv) > 1 else (
    "/fs/fast/u2021000902/wushang/model/Qwen/Qwen2.5-7B-Instruct")

# --- 三段 scaffold(逐字抄自 trace_opd/envs/math/adapter.py)------------------
SINGLE = (
    "You are a helpful AI Assistant that provides well-reasoned and "
    "detailed responses. You FIRST think about the reasoning process as an "
    "internal monologue and then provide the final answer. The reasoning "
    "process MUST BE enclosed within <think> </think> tags. The final "
    "answer MUST BE put in \\boxed{}."
)
# MENTOR verl/utils/dataset.py:174-176 的 system content(逐字,用于对拍)
MENTOR_REF = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed "
    "responses. You FIRST think about the reasoning process as an internal "
    "monologue and then provide the final answer. The reasoning process MUST "
    "BE enclosed within <think> </think> tags. The final answer MUST BE put "
    "in \\boxed{}."
)

PROBLEM = (
    "Let \\[f(x) = \\left\\{\n\\begin{array}{cl} ax+3, &\\text{ if }x>2, \\\\\n"
    "x-5 &\\text{ if } -2 \\le x \\le 2, \\\\\n2x-b &\\text{ if } x <-2.\n"
    "\\end{array}\n\\right.\\]Find $a+b$ if the piecewise function is "
    "continuous (which means that its graph can be drawn without lifting "
    "your pencil from the paper)."
)

# 典型 500 词特权文档(占位,只为量 token 账;真 doc 由 35B researcher 产)
DOC = ("Strategy notes. " + "Check continuity at each breakpoint by matching "
       "one-sided limits; solve the resulting linear system. ") * 18


def main() -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
    print(f"tokenizer: {MODEL}\n  vocab={len(tok)} "
          f"eos={tok.eos_token!r}({tok.eos_token_id})\n")

    # ---- 0. 与 MENTOR 原文逐字对拍 ----
    same = SINGLE == MENTOR_REF
    print(f"[0] system prompt 与 MENTOR 逐字一致: {same}")
    if not same:
        import difflib
        for line in difflib.unified_diff(
                [MENTOR_REF], [SINGLE], "MENTOR", "ours", lineterm=""):
            print("   ", line)
        return 1

    # ---- 1/2. 渲染与 token 账 ----
    msgs = [{"role": "system", "content": SINGLE},
            {"role": "user", "content": PROBLEM}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    # ★ 本版 transformers 的 apply_chat_template(tokenize=True) 返回 Encoding
    #   对象而非 list[int](len() 会得到字段数 2)。走生产同款路径:
    #   tokenize=False 取文本 → encode(add_special_tokens=False)
    #   (= trace_opd 的 core/template_delta.render_ids 与 MENTOR dataset.py:264-265)
    ids = tok.encode(text, add_special_tokens=False)
    print(f"\n[1] 渲染后的完整 prompt(repr,便于看清模板缝合):\n{text!r}")
    print(f"\n[2] token 账:")
    print(f"    system 单独          {len(tok.encode(SINGLE, add_special_tokens=False)):>5}")
    print(f"    题面单独             {len(tok.encode(PROBLEM, add_special_tokens=False)):>5}")
    print(f"    渲染总计(含模板)     {len(ids):>5}   "
          f"[MENTOR data.max_prompt_length=1024;我们 max_prompt_length=4096]")
    msgs_doc = [{"role": "system", "content": SINGLE + "\n\n" + DOC},
                {"role": "user", "content": PROBLEM}]
    text_doc = tok.apply_chat_template(msgs_doc, add_generation_prompt=True,
                                       tokenize=False)
    n_doc = len(tok.encode(text_doc, add_special_tokens=False))
    print(f"    + 500 词 doc 后      {n_doc:>5}   [PGS 臂的 system 注入,余量 "
          f"{4096 - n_doc}]")

    # ---- 3. 零工具块 ----
    has_tools = ("<tools>" in text) or ("<tool_call>" in text)
    print(f"\n[3] 渲染文本含工具块: {has_tools}  (单轮档必须 False —— "
          "chat template 只在传 tools= 时才渲染,我们传 None)")
    # 反证:传 tools 时模板确实会渲染工具块(说明上面的 False 是真的没传,
    # 而不是模板根本不支持)
    probe_tools = [{"type": "function", "function": {
        "name": "run_python", "description": "x",
        "parameters": {"type": "object", "properties": {}}}}]
    t2 = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                 tokenize=False, tools=probe_tools)
    print(f"    反证(同 msgs 但传 tools):含 <tools> = {'<tools>' in t2}")

    # ---- 4. 协议标签的 token 化(zero-shot 服从性的先验风险)----
    print(f"\n[4] 协议标签 token 化(碎片越多,zero-shot 越难稳住格式):")
    for s in ("<think>", "</think>", "<think> </think>", "\\boxed{", "\\boxed{42}",
              "<reasoning>", "</action>"):
        t = tok.encode(s, add_special_tokens=False)
        print(f"    {s!r:<20} -> {len(t):>2} tok  {[tok.decode([x]) for x in t]}")
    print("    (末两行是 ALFWorld 那套自定义标签,作对照:记忆记载其 "
          "zero-shot 服从率 0%,靠 system 内 one-shot 救回 99.9%)")

    # ---- 5. 生成起点 ----
    tail = tok.decode(ids[-8:])
    print(f"\n[5] 生成起点(prompt 末 8 token):{tail!r}")
    print("    ⇒ 模型从这里开始写,MENTOR 期望它首字就是 '<think>'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
