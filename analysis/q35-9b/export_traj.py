#!/usr/bin/env python3
"""导出完整可读轨迹给人工查看。

每条轨迹输出:元信息 + 逐 step 的完整原始文本(不截断)。
另存一份精简 json(去掉 token_ids 等大字段)便于本地打开。

用法: python3 export_traj.py <run_dir> <step> <out_dir>
"""
import glob, json, os, sys

d, step, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
fs = sorted(glob.glob(os.path.join(d, "rollouts", f"step_{step}", "*.json")))
rows = [(f, json.load(open(f))) for f in fs]

deg = [(f, r) for f, r in rows if str(r.get("outcome", "")).startswith("DEGENERATE")]
suc = [(f, r) for f, r in rows if r.get("outcome") == "SUCCESS"]
fail = [(f, r) for f, r in rows if r.get("outcome") == "FAILED"]
print(f"总 {len(rows)}: DEGENERATE {len(deg)} / SUCCESS {len(suc)} / FAILED {len(fail)}")

BIG = ("response_token_ids", "turn_token_spans")
STEER_BIG = ("steer_delta", "steer_draft_token", "steer_entropy", "steer_eta",
             "steer_final_pi_rank", "steer_final_token", "steer_intensity",
             "steer_log_b", "steer_log_pi", "steer_pi_top16", "steer_positions",
             "steer_probe", "steer_propensity", "steer_pt_top16", "steer_rejected",
             "steer_unlocked_at", "steer_ent_hist", "steer_ell_hist", "steer_d_hist")


def slim(r):
    o = {k: v for k, v in r.items() if k not in BIG}
    st = o.get("steering")
    if isinstance(st, dict):
        o["steering"] = {k: v for k, v in st.items() if k not in STEER_BIG}
    return o


def render(f, r):
    t = r.get("trajectory") or {}
    steps = t.get("steps") or []
    L = []
    L.append("=" * 100)
    L.append(f"文件: {os.path.basename(f)}")
    L.append(f"task_id: {r.get('task_id')}")
    L.append(f"outcome: {r.get('outcome')}   won: {r.get('won')}   truncated: {r.get('truncated')}")
    L.append(f"respLen: {r.get('response_token_len')}   assistant_turns: {r.get('assistant_turns')}"
             f"   env_steps: {r.get('env_steps')}   invalid_format: {r.get('invalid_format')}")
    st = r.get("steering") or {}
    L.append(f"steer_arm: {st.get('steer_arm')}   gate_open_rate: {st.get('steer_gate_open_rate')}"
             f"   n_base_open: {st.get('steer_n_base_open')}")
    rs = r.get("research") or {}
    L.append(f"research: experiments={rs.get('research_experiments')} episodes={rs.get('research_episodes')}"
             f" doc_words={rs.get('research_doc_words')} doc_missing={rs.get('research_doc_missing')}")
    L.append(f"traj.error: {t.get('error')}")
    L.append("-" * 100)
    L.append("initial_obs:")
    L.append(str(t.get("initial_obs")))
    L.append("-" * 100)
    L.append(f"共 {len(steps)} 个 step,以下为完整原文(不截断)")
    for i, s in enumerate(steps, 1):
        L.append("")
        L.append(f"┌─ step {i}  acted={s.get('acted')} " + "─" * 60)
        for key in ("raw", "text", "assistant", "response", "completion"):
            if s.get(key):
                L.append(f"│ [{key}] 模型完整输出:")
                L.append(str(s[key]))
                break
        else:
            if s.get("thought") is not None:
                L.append("│ [thought]:")
                L.append(str(s.get("thought")))
            L.append(f"│ [action]: {s.get('action')!r}")
        for key in ("obs", "observation", "feedback"):
            if s.get(key) is not None:
                L.append(f"│ [{key}] 环境返回:")
                L.append(str(s[key]))
                break
        other = {k: v for k, v in s.items()
                 if k not in ("raw", "text", "assistant", "response", "completion",
                              "thought", "action", "obs", "observation", "feedback", "acted")}
        if other:
            L.append(f"│ [其它字段] {json.dumps(other, ensure_ascii=False)[:600]}")
        L.append("└" + "─" * 70)
    return "\n".join(L)


picks = []
picks += [("DEGENERATE", f, r) for f, r in deg[:3]]
picks += [("SUCCESS", f, r) for f, r in suc[:2]]
picks += [("FAILED", f, r) for f, r in fail[:1]]

for tag, f, r in picks:
    base = f"{tag}__{os.path.basename(f)}"
    with open(os.path.join(out, base.replace(".json", ".txt")), "w") as fh:
        fh.write(render(f, r))
    with open(os.path.join(out, base), "w") as fh:
        json.dump(slim(r), fh, ensure_ascii=False, indent=2)
    print(f"  写出 {base}(.txt 可读 / .json 精简)")

# 一份汇总:所有 DEGENERATE 的首次失败位置
summary = []
for f, r in deg:
    steps = (r.get("trajectory") or {}).get("steps") or []
    fb = next((i for i, s in enumerate(steps, 1) if s.get("acted") is False), None)
    summary.append({"file": os.path.basename(f), "turns": r.get("assistant_turns"),
                    "invalid_format": r.get("invalid_format"),
                    "respLen": r.get("response_token_len"),
                    "first_format_fail_step": fb, "total_steps": len(steps),
                    "env_steps": r.get("env_steps")})
with open(os.path.join(out, "_DEGENERATE_summary.json"), "w") as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=2)
print(f"  写出 _DEGENERATE_summary.json({len(summary)} 条)")
print(f"\n输出目录: {out}")
