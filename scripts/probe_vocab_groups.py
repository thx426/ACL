#!/usr/bin/env python3
"""词表分组:哪些模型两两之间可以做 token 级分布干预(硬前提=词表逐 id 一致)。"""
import hashlib
from transformers import AutoTokenizer

ROOT = "/fs/fast/u2021000902/wushang/model/Qwen/"
MODELS = ["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen3-1.7B",
          "Qwen3-4B-Instruct-2507", "Qwen3.5-4B", "Qwen3.5-9B",
          "Qwen3.6-27B", "Qwen3.6-35B-A3B"]
PROBE = "The final answer is 42, and \\frac{1}{2}+x^2. <think>r</think> <reasoning>r</reasoning>"

rows = []
for m in MODELS:
    try:
        t = AutoTokenizer.from_pretrained(ROOT + m)
    except Exception as e:
        print(f"{m:<26} 载入失败: {e}"); continue
    v = t.get_vocab()
    # 词表指纹:全部 (token,id) 对的哈希
    h = hashlib.sha1(
        "\n".join(f"{k}\t{i}" for k, i in sorted(v.items(), key=lambda kv: kv[1]))
        .encode("utf-8")).hexdigest()[:12]
    think_special = t.convert_tokens_to_ids("<think>")
    # 模板是否 thinking-family(non-thinking 下预填空 think 块)
    try:
        s = t.apply_chat_template([{"role": "user", "content": "x"}],
                                  add_generation_prompt=True, tokenize=False,
                                  enable_thinking=False)
        tf = "<think>" in s
    except Exception:
        tf = "?"
    rows.append((m, len(v), h, think_special, tf,
                 len(t.encode(PROBE, add_special_tokens=False))))

print(f"{'model':<26} {'vocab':>7}  {'词表指纹':<13} {'<think>id':>10} "
      f"{'think模板':>9} {'probe长':>7}")
for m, n, h, ts, tf, pl in rows:
    print(f"{m:<26} {n:>7}  {h:<13} {str(ts):>10} {str(tf):>9} {pl:>7}")

print("\n可互操作分组(词表指纹相同 = 可做 token 级干预):")
groups = {}
for m, n, h, *_ in rows:
    groups.setdefault(h, []).append(f"{m}({n})")
for h, ms in groups.items():
    print(f"  [{h}] {', '.join(ms)}")
