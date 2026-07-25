"""Teacher 框架语义判决探针(CD-006,2026-07-24,用户假设验证)。

假设:OPD 内化不佳与 teacher 系统提示的"演员续写"语义有关——teacher 被喂学生
轨迹当作自己的历史,自条件化(self-conditioning)使其在深处放弃文档、续写学生
的错误模式(深度衰减 50× 的语义成因)。若在系统提示中显式声明"按文档监督/
纠正每一步",深处错误点的 teacher 分布应重新对齐文档。

方法:同一 1.7B(CPU 前向,不碰训练 GPU),对真实失败轨迹的**已知错误决策点**,
在四种 teacher 上下文下测正确动作 token 的概率:
  A current    现行框架(SYSTEM_PROMPT_BARE + ## Task guidance + doc)
  B supervise  A + 监督语义附言(纠正每一步、忽略此前错误、格式不变)
  C reinject   A + 错误点前一轮把 doc 摘要作 user 消息重注入(机械对照)
  D bare       无文档
读法:B>>A ⇒ 语义框架是主因(用户假设成立);C>>A≈B ⇒ 机械注意力稀释是主因;
B≈C>>A ⇒ 双因;全都低 ⇒ 另有他因。

注意:teacher 上下文用 chat template 全量渲染,与训练期 TemplateDelta 增量流
非字节一致——本探针是方向性判决,不是训练复现(结论只用于排序,不报绝对值)。
"""
import argparse, json, sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from trace_opd.core.protocol import SYSTEM_PROMPT_BARE, build_system_prompt  # noqa: E402

SUPERVISE_ADDENDUM = (
    "\n\nIMPORTANT SUPERVISION NOTE: The conversation below replays an episode "
    "attempted by a TRAINEE agent, and its turns may contain mistakes. You are "
    "the guidance-following expert re-executing this episode. At EVERY turn, "
    "decide what should be done in the CURRENT situation strictly according to "
    "the Task guidance above — do NOT imitate or continue the trainee's earlier "
    "mistakes. Reply in exactly the same format: <reasoning>...</reasoning> then "
    "<action>one command</action>."
)


def build_messages(dump, system_text, reinject_doc=None, reinject_before_turn=None):
    tr = dump["trajectory"]
    msgs = [{"role": "system", "content": system_text},
            {"role": "user", "content": tr["initial_obs"]}]
    for s in tr["steps"]:
        if reinject_doc is not None and s["t"] == reinject_before_turn:
            msgs.append({"role": "user",
                         "content": "Reminder — Task guidance (follow it for the "
                                    "remaining turns):\n" + reinject_doc})
            # 该轮观察与提醒合并为同轮 user 消息序列(探针近似,方向性足够)
        msgs.append({"role": "assistant",
                     "content": f"<reasoning>{s.get('thought','')}</reasoning>\n"
                                f"<action>{s.get('action_raw','')}</action>"})
        msgs.append({"role": "user", "content": s.get("observation", "")})
    return msgs


@torch.no_grad()
def action_token_probs(model, tok, msgs, turn_t, cand_words, turn_thought):
    """在第 turn_t 轮 assistant 的 <action> 后首 token 位置,取候选词概率。

    结构化构造前缀(不做标签字符串计数——系统提示/附言里也含 <action> 示例,
    计数会 off-by-one):对话渲染到第 turn_t 轮 assistant 之前 + generation
    prompt,再拼该轮 reasoning 与 '<action>' 作为生成前缀。"""
    # 找到第 turn_t 个 assistant 消息在 msgs 里的下标
    a_seen = 0
    cut = None
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            a_seen += 1
            if a_seen == turn_t:
                cut = i
                break
    if cut is None:
        raise RuntimeError(f"turn {turn_t} assistant not found")
    prefix = tok.apply_chat_template(
        msgs[:cut], tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix += f"<reasoning>{turn_thought}</reasoning>\n<action>"
    ids = tok(prefix, return_tensors="pt").input_ids.to(model.device)
    out = model(ids)
    logits = out.logits[0, -1]
    logp = torch.log_softmax(logits.float(), dim=-1)
    res = {}
    for w in cand_words:
        wid = tok(w, add_special_tokens=False).input_ids
        res[w] = float(logp[wid[0]].exp())
    top = torch.topk(logp, 5)
    res["_top5"] = [(tok.decode([i]), round(float(p.exp()), 4))
                    for p, i in zip(top.values, top.indices)]
    res["_ctx_tokens"] = ids.shape[1]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dump", required=True, help="rollout dump json(失败轨迹)")
    ap.add_argument("--doc", required=True, help="该任务当步 doc.md")
    ap.add_argument("--turns", required=True, help="逗号分隔的探测轮列表,如 6,8,12,18,25")
    ap.add_argument("--correct", required=True, help="正确动作首词,如 'cool'")
    ap.add_argument("--wrong", required=True, help="学生实际首词,如 'put'")
    args = ap.parse_args()

    dump = json.loads(Path(args.dump).read_text())
    doc = Path(args.doc).read_text()
    doc_short = doc[:2000]

    tok = AutoTokenizer.from_pretrained(args.model)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
        dev = "cuda"
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
        dev = "cpu"
        torch.set_num_threads(64)
    model.eval()
    print("device:", dev)

    conds = {
        "A_current":   dict(system=build_system_prompt(doc)),
        "B_supervise": dict(system=build_system_prompt(doc) + SUPERVISE_ADDENDUM),
        "C_reinject":  dict(system=build_system_prompt(doc), reinject=True),
        "D_bare":      dict(system=build_system_prompt(None)),
    }
    cands = [args.correct, args.wrong, " " + args.correct, " " + args.wrong]
    turns = [int(x) for x in args.turns.split(",")]
    print(f"task={dump['task_id']} turns={turns} correct={args.correct!r} wrong={args.wrong!r}")
    for turn in turns:
        thought = dump["trajectory"]["steps"][turn - 1].get("thought", "")
        print(f"--- turn {turn} | student thought: {thought[:90]!r}")
        for name, c in conds.items():
            msgs = build_messages(
                dump, c["system"],
                reinject_doc=doc_short if c.get("reinject") else None,
                reinject_before_turn=turn if c.get("reinject") else None,
            )
            r = action_token_probs(model, tok, msgs, turn, cands, thought)
            pc = max(r[args.correct], r[" " + args.correct])
            pw = max(r[args.wrong], r[" " + args.wrong])
            print(f"[t{turn:02d} {name:11s}] P(cool)={pc:.4f} P(put)={pw:.4f} ctx={r['_ctx_tokens']} top3={r['_top5'][:3]}", flush=True)


if __name__ == "__main__":
    main()
