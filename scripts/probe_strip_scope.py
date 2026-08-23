#!/usr/bin/env python3
"""剥离的**边界**在哪:哪些 role / 哪些形态会被 35B 模板吃掉。
决定改造范围 —— 若只 assistant 轮被剥,researcher 的 user 轮转述就是安全的。
"""
from transformers import AutoTokenizer

T = AutoTokenizer.from_pretrained(
    "/fs/fast/u2021000902/wushang/model/Qwen/Qwen3.6-35B-A3B")
MARK = "SECRET_9931"


def has(msgs, **kw):
    s = T.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                              enable_thinking=False, **kw)
    return MARK in s, s


base = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U1"}]

print("[A] 按 role:含 <think>MARK</think> 的内容放在不同 role 里")
cases = {
    "assistant(历史轮)": base + [
        {"role": "assistant", "content": f"<think>{MARK}</think>\nact"},
        {"role": "user", "content": "OBS"}],
    "assistant(末轮)":   base + [
        {"role": "assistant", "content": f"<think>{MARK}</think>\nact"}],
    "user 轮":           base + [
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": f"<think>{MARK}</think> observation"}],
    "system 轮":         [{"role": "system", "content": f"<think>{MARK}</think> SYS"},
                          {"role": "user", "content": "U1"}],
}
for name, msgs in cases.items():
    kept, _ = has(msgs)
    print(f"    {name:<20} {'保留 ✓' if kept else '★ 被剥离'}")

print("\n[B] 按形态:assistant 历史轮里的各种写法")
forms = {
    "<think>X</think> 完整对":  f"<think>{MARK}</think>\nact",
    "只有开标签 <think>X":       f"<think>{MARK}\nact",
    "只有闭标签 X</think>":      f"{MARK}</think>\nact",
    "前面有正文再 <think>":      f"prefix text <think>{MARK}</think> act",
    "<think>X</think> 后无正文": f"<think>{MARK}</think>",
    "<reasoning>X</reasoning>":  f"<reasoning>{MARK}</reasoning>\nact",
    "<thought>X</thought>":      f"<thought>{MARK}</thought>\nact",
}
for name, content in forms.items():
    msgs = base + [{"role": "assistant", "content": content},
                   {"role": "user", "content": "OBS"}]
    kept, _ = has(msgs)
    print(f"    {name:<28} {'保留 ✓' if kept else '★ 被剥离'}")

print("\n[C] 剥离后历史 assistant 轮长什么样(取一例完整渲染)")
msgs = base + [{"role": "assistant", "content": f"<think>{MARK}</think>\nact here"},
               {"role": "user", "content": "OBS"}]
_, s = has(msgs)
i = s.find("<|im_start|>assistant")
print("   ", repr(s[i:i + 200]))
