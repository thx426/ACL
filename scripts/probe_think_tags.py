#!/usr/bin/env python3
"""<think> 冲突取证:executor(Qwen2.5)vs 未来的干预模型(Qwen3.6-35B)。

回答四个问题:
  Q1 两边词表是否兼容(token 级分布干预的硬前提)
  Q2 `<think>`/`</think>` 在各自词表里是不是**特殊 token**
  Q3 thinking-family 模板的 gen_prefix 形态(enable_thinking True/False)
  Q4 ★ 历史 assistant 轮的 `<think>` 块是否被**剥离**(协议推理被静默删除)
  Q5 候选替换标签在两个词表上的碎片数
"""
from __future__ import annotations
import sys

EXEC_M = "/fs/fast/u2021000902/wushang/model/Qwen/Qwen2.5-7B-Instruct"
TEACH_M = "/fs/fast/u2021000902/wushang/model/Qwen/Qwen3.6-35B-A3B"

CANDIDATES = ["<think>", "</think>", "<reasoning>", "</reasoning>",
              "<thought>", "</thought>", "<scratch>", "<analysis>",
              "<rationale>", "<plan>", "<work>"]


def load(p):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(p, trust_remote_code=False)


def specials(tok):
    """所有 added/special token 的字面集合。"""
    out = set(tok.all_special_tokens or [])
    try:
        for _i, t in sorted(tok.get_added_vocab().items(), key=lambda kv: kv[1]):
            out.add(_i)
    except Exception:
        pass
    return out


def main() -> int:
    te = load(EXEC_M)
    tt = load(TEACH_M)
    print(f"executor : {EXEC_M}\n  vocab(len)={len(te)}  eos={te.eos_token!r}")
    print(f"teacher  : {TEACH_M}\n  vocab(len)={len(tt)}  eos={tt.eos_token!r}")

    # ---- Q1 词表兼容性 ----
    print("\n[Q1] 词表兼容性(token 级分布干预的硬前提)")
    print(f"    len 相同: {len(te) == len(tt)}   ({len(te)} vs {len(tt)})")
    ve, vt = te.get_vocab(), tt.get_vocab()
    print(f"    get_vocab 大小: {len(ve)} vs {len(vt)}")
    # 抽样对拍:同一 id 是否解到同一字面
    import random
    random.seed(0)
    ids = random.sample(range(min(len(ve), len(vt))), 2000)
    same = sum(1 for i in ids if te.convert_ids_to_tokens(i) == tt.convert_ids_to_tokens(i))
    print(f"    随机 2000 个 id 解码一致: {same}/2000 = {100*same/2000:.1f}%")
    probe = "The final answer is 42, and \\frac{1}{2} + x^2."
    print(f"    同一文本编码相同: {te.encode(probe, add_special_tokens=False) == tt.encode(probe, add_special_tokens=False)}")

    # ---- Q2 特殊 token ----
    print("\n[Q2] <think> 是不是**特殊 token**")
    se, st = specials(te), specials(tt)
    for tag in ("<think>", "</think>"):
        ide = te.convert_tokens_to_ids(tag)
        idt = tt.convert_tokens_to_ids(tag)
        print(f"    {tag!r:<12} executor: special={tag in se} id={ide} "
              f"| teacher: special={tag in st} id={idt}")
    print(f"    teacher 的 added_vocab 里含 think 的项: "
          f"{[k for k in st if 'think' in k.lower()]}")

    # ---- Q3 gen_prefix 形态 ----
    print("\n[Q3] gen_prefix 形态(thinking 模板的两种模式)")
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U1"}]
    for name, tok in (("executor", te), ("teacher ", tt)):
        for flag in (False, True):
            try:
                s = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                            tokenize=False, enable_thinking=flag)
            except Exception as e:
                s = f"<模板不支持 enable_thinking: {e}>"
            tail = s[s.rfind("<|im_start|>assistant"):] if "<|im_start|>assistant" in s else s[-60:]
            print(f"    {name} enable_thinking={flag!s:<5} -> {tail!r}")

    # ---- Q4 ★ 历史轮剥离 ----
    print("\n[Q4] ★ 历史 assistant 轮的 <think> 块是否被剥离")
    for tag, closing in (("<think>", "</think>"), ("<reasoning>", "</reasoning>")):
        conv = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "U1"},
            {"role": "assistant",
             "content": f"{tag}SECRET_REASONING_9931{closing}\n<action>go north</action>"},
            {"role": "user", "content": "OBS1"},
        ]
        for name, tok in (("executor", te), ("teacher ", tt)):
            try:
                s = tok.apply_chat_template(conv, add_generation_prompt=True,
                                            tokenize=False, enable_thinking=False)
                kept = "SECRET_REASONING_9931" in s
            except Exception as e:
                kept = f"ERR {e}"
            flag = "保留 ✓" if kept is True else ("★★ 被剥离!" if kept is False else kept)
            print(f"    {tag:<12} {name}: {flag}")

    # ---- Q5 候选标签碎片数 ----
    print("\n[Q5] 候选标签的 token 碎片数(越少 zero-shot 越稳)")
    print(f"    {'tag':<14} {'executor':<28} {'teacher'}")
    for tag in CANDIDATES:
        a = te.encode(tag, add_special_tokens=False)
        b = tt.encode(tag, add_special_tokens=False)
        sa = f"{len(a)}tok {[te.decode([x]) for x in a]}"
        sb = f"{len(b)}tok {[tt.decode([x]) for x in b]}"
        mark = " ← special!" if (tag in se or tag in st) else ""
        print(f"    {tag:<14} {sa:<28} {sb}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
