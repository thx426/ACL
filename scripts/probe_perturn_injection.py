"""逐轮文档注入判决探针(用户方案,2026-07-24)。

问题:framing 探针已证——学生 reasoning 承诺后,动作 token 无药可救。但分叉点
在 reasoning 里。若每轮 observation 后拼接文档(摘要),teacher **自由生成**该轮
时能否产出正确 reasoning+action?能 ⇒ 逐轮注入在 reasoning token 上制造真实
蒸馏信号(teacher 分布偏离学生的错误推理),是新杠杆(S4-R:reasoning 通道版)。

方法:错误链深处(t12/t18)三条件 × 各 n 次自由采样(temp 0.7),数生成的
<action> 首词:A 现行(doc 仅系统提示)/ E 每轮 obs 后拼 doc 摘要 / D 无文档。
GPU 单卡(空闲显存安全门在外层脚本把关)。
"""
import argparse, json, re, sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from trace_opd.core.protocol import build_system_prompt  # noqa: E402


def build_messages(dump, system_text, digest=None):
    tr = dump["trajectory"]
    msgs = [{"role": "system", "content": system_text},
            {"role": "user", "content": tr["initial_obs"]}]
    for s in tr["steps"]:
        msgs.append({"role": "assistant",
                     "content": f"<reasoning>{s.get('thought','')}</reasoning>\n"
                                f"<action>{s.get('action_raw','')}</action>"})
        obs = s.get("observation", "")
        if digest is not None:
            obs = obs + "\n\n[Task guidance reminder]\n" + digest
        msgs.append({"role": "user", "content": obs})
    return msgs


@torch.no_grad()
def gen_turn(model, tok, msgs, turn_t, n, max_new=96, temp=0.7):
    """在第 turn_t 轮 assistant 处自由生成 n 次,返回各次的 action 首词。"""
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
    ids = tok(prefix, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(ids, do_sample=True, temperature=temp, top_p=0.95,
                         num_return_sequences=n, max_new_tokens=max_new,
                         pad_token_id=tok.eos_token_id)
    verbs, texts = [], []
    for row in out:
        txt = tok.decode(row[ids.shape[1]:], skip_special_tokens=True)
        m = re.search(r"<action>\s*(\S+)", txt)
        verbs.append(m.group(1).lower() if m else "?")
        texts.append(txt.replace("\n", " ")[:110])
    return verbs, texts, ids.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--turns", default="12,18")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    dump = json.loads(Path(args.dump).read_text())
    doc = Path(args.doc).read_text()
    digest = doc[:700]  # 文档头部=Verified approach 段(含 cool 关键步)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    model.eval()

    conds = {
        "A_sysdoc":  dict(system=build_system_prompt(doc), digest=None),
        "E_perturn": dict(system=build_system_prompt(doc), digest=digest),
        "D_bare":    dict(system=build_system_prompt(None), digest=None),
    }
    print(f"task={dump['task_id']} digest_chars={len(digest)}")
    for turn in [int(x) for x in args.turns.split(",")]:
        print(f"=== turn {turn}(学生实际: {dump['trajectory']['steps'][turn-1]['action_raw']!r})===")
        for name, c in conds.items():
            msgs = build_messages(dump, c["system"], digest=c["digest"])
            verbs, texts, ctx = gen_turn(model, tok, msgs, turn, args.n)
            ncool = sum(1 for v in verbs if v == "cool")
            print(f"[t{turn} {name:9s}] cool {ncool}/{args.n}  verbs={verbs}  ctx={ctx}")
            for t in texts[:2]:
                print(f"      » {t}")
    print("done")


if __name__ == "__main__":
    main()
