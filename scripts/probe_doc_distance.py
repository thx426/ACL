"""文档保鲜半衰期探针(2026-07-24,分块打分的 W 取值依据)。

分块刷新打分的真实 teacher 上下文 = 纯学生历史(含错误,无文档拷贝)+ 单份
全量文档(距离被打分轮 Δ 轮)。本探针在错误链 t12 处测:单份全量文档放在
obs_{12-Δ} 之后(Δ=1,2,4,8),teacher 自由生成 t12 能否写出正确动作 cool。
锚点:A(文档仅系统提示,已证 0/6)/ E(每轮注入,已证 6/6)。
读法:F(Δ) 全对的最大 Δ ≈ 保鲜半衰期 ⇒ 取 W ≤ 该值。
"""
import argparse, json, re, sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from trace_opd.core.protocol import build_system_prompt  # noqa: E402


def build_messages(dump, system_text, full_doc=None, doc_after_turn=None, every=False):
    tr = dump["trajectory"]
    msgs = [{"role": "system", "content": system_text},
            {"role": "user", "content": tr["initial_obs"]}]
    for s in tr["steps"]:
        msgs.append({"role": "assistant",
                     "content": f"<reasoning>{s.get('thought','')}</reasoning>\n"
                                f"<action>{s.get('action_raw','')}</action>"})
        obs = s.get("observation", "")
        if full_doc is not None and (every or s["t"] == doc_after_turn):
            obs = obs + "\n\n## Task guidance\n" + full_doc
        msgs.append({"role": "user", "content": obs})
    return msgs


@torch.no_grad()
def gen_turn(model, tok, msgs, turn_t, n, max_new=96, temp=0.7):
    a_seen = 0
    cut = None
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            a_seen += 1
            if a_seen == turn_t:
                cut = i
                break
    prefix = tok.apply_chat_template(msgs[:cut], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    enc = tok(prefix, return_tensors="pt")
    ids = enc.input_ids.to(model.device)
    am = enc.attention_mask.to(model.device)
    verbs = []
    for _ in range(n):   # 逐条生成,压 KV 峰值(探针曾把共享卡挤到 338MB)
        out = model.generate(ids, attention_mask=am, do_sample=True, temperature=temp,
                             top_p=0.95, num_return_sequences=1, max_new_tokens=max_new,
                             pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        m = re.search(r"<action>\s*(\S+)", txt)
        verbs.append(m.group(1).lower() if m else "?")
    return verbs, ids.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--turn", type=int, default=12)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    dump = json.loads(Path(args.dump).read_text())
    doc = Path(args.doc).read_text()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    model.eval()

    t = args.turn
    conds = [("A_sys_only", dict(system=build_system_prompt(doc)))]  # E 全量逐轮非分块形态,撤
    for delta in (1, 2, 4, 8):
        conds.append((f"F_dist{delta}",
                      dict(system=build_system_prompt(doc), full_doc=doc,
                           doc_after_turn=t - delta)))
    print(f"task={dump['task_id']} turn={t} doc_chars={len(doc)}")
    for name, kw in conds:
        msgs = build_messages(dump, kw.pop("system"), **kw)
        verbs, ctx = gen_turn(model, tok, msgs, t, args.n)
        ncool = sum(1 for v in verbs if v == "cool")
        print(f"[{name:11s}] cool {ncool}/{args.n}  verbs={verbs}  ctx={ctx}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
