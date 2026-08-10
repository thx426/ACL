#!/usr/bin/env python3
"""导出真实轨迹片段,看 DEGENERATE:format 到底长什么样。

用法: python3 dump_traj.py <run_dir> <step> [n_samples]
"""
import glob, json, os, sys

d = sys.argv[1]
step = sys.argv[2] if len(sys.argv) > 2 else "00001"
n = int(sys.argv[3]) if len(sys.argv) > 3 else 2

fs = sorted(glob.glob(os.path.join(d, "rollouts", f"step_{step}", "*.json")))
rows = [json.load(open(f)) for f in fs]

deg = [r for r in rows if str(r.get("outcome", "")).startswith("DEGENERATE")]
suc = [r for r in rows if r.get("outcome") == "SUCCESS"]
fail = [r for r in rows if r.get("outcome") == "FAILED"]
print(f"总 {len(rows)} 行: DEGENERATE {len(deg)} / SUCCESS {len(suc)} / FAILED {len(fail)}\n")


def show(r, tag, max_steps=8, max_chars=700):
    t = r.get("trajectory") or {}
    steps = t.get("steps") or []
    print("=" * 78)
    print(f"【{tag}】task={str(r.get('task_id'))[:60]}")
    print(f"  outcome={r.get('outcome')}  won={r.get('won')}  truncated={r.get('truncated')}")
    print(f"  respLen={r.get('response_token_len')}  turns={r.get('assistant_turns')}  "
          f"env_steps={r.get('env_steps')}  invalid_format={r.get('invalid_format')}")
    print(f"  error={t.get('error')}")
    print(f"  initial_obs: {str(t.get('initial_obs'))[:200]}")
    print(f"  --- 共 {len(steps)} 个 step,展示最后 {min(max_steps,len(steps))} 个 ---")
    for i, s in enumerate(steps[-max_steps:], start=max(1, len(steps) - max_steps + 1)):
        raw = s.get("raw") or s.get("assistant") or s.get("response") or ""
        th = s.get("thought") or s.get("reasoning") or ""
        ac = s.get("action") or ""
        ob = s.get("obs") or s.get("observation") or s.get("feedback") or ""
        acted = s.get("acted")
        print(f"\n  ── step {i}  acted={acted} ──")
        if raw:
            print(f"  [模型原始输出] {str(raw)[:max_chars]}")
        else:
            if th: print(f"  [thought] {str(th)[:300]}")
            print(f"  [action ] {str(ac)[:160]!r}")
        print(f"  [环境返回] {str(ob)[:220]}")


for r in deg[:n]:
    show(r, "格式退化 DEGENERATE:format")
if suc:
    print()
    show(suc[0], "对照:成功轨迹 SUCCESS", max_steps=4)
