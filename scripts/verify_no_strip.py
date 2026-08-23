#!/usr/bin/env python3
"""验收:新协议的 executor 历史轮在 35B(未来的干预教师)模板下是否幸存。
输入 scaffolds.json 由**真实代码**导出(非手抄)。"""
import json, sys
from transformers import AutoTokenizer

D = json.load(open("/tmp/scaffolds.json", encoding="utf-8"))
MODELS = {
    "executor Qwen2.5-7B": "/fs/fast/u2021000902/wushang/model/Qwen/Qwen2.5-7B-Instruct",
    "teacher  Qwen3.6-35B": "/fs/fast/u2021000902/wushang/model/Qwen/Qwen3.6-35B-A3B",
}
MARK = "I must find the director first."
fail = 0
for mname, path in MODELS.items():
    t = AutoTokenizer.from_pretrained(path)
    print(f"\n=== {mname} ===")
    for tag in ("reasoning", "think"):
        d = D[tag]
        conv = [
            {"role": "system", "content": d["searchqa_scaffold"]},
            {"role": "user", "content": "Who directed Inception and when born?"},
            {"role": "assistant", "content": d["assistant_turn"]},
            {"role": "user", "content": "<information> ... </information>"},
        ]
        s = t.apply_chat_template(conv, add_generation_prompt=True,
                                  tokenize=False, enable_thinking=False)
        kept = MARK in s
        # scaffold(system 轮)永远保留,单独核对一次
        sys_ok = d["open"] in s
        status = "保留 ✓" if kept else "★ 推理被剥离"
        print(f"  tag={tag:<10} 历史轮推理: {status:<16} system 协议说明在场: {sys_ok}")
        if tag == "reasoning" and not kept:
            fail = 1
        if tag == "think" and mname.startswith("teacher") and kept:
            print("    ⚠ 预期 think 在 teacher 上被剥离,实测没有 —— 前提假设需复核")

print("\n=== 两档 scaffold 差异(确认只换了标签)===")
a = D["reasoning"]["math_single_scaffold"]
b = D["think"]["math_single_scaffold"]
print("  math single, 把 reasoning 换回 think 后与 think 档逐字一致:",
      a.replace("<reasoning>", "<think>").replace("</reasoning>", "</think>") == b)
a2 = D["reasoning"]["searchqa_scaffold"]; b2 = D["think"]["searchqa_scaffold"]
print("  searchqa   , 同上:",
      a2.replace("<reasoning>", "<think>").replace("</reasoning>", "</think>") == b2)
print("\n  math single brief(给 researcher)含标签:",
      D["reasoning"]["open"] in D["reasoning"]["math_single_brief"],
      "| 含 <think>:", "<think>" in D["reasoning"]["math_single_brief"])
sys.exit(fail)
