"""Won-trajectory evolution under OPD training — token totals and thought
content, per step, won/lost split. Read-only over rollout dumps.

The preregistered T1 read-out (teacher-grounding design §8): the won-stratified
thought_chars_per_turn curve rose x6 (315 -> 1986) on the rkl+RL baseline; T1
predicts flat or strongly damped. Run it on any run's rollouts dir:

    python3 scripts/won_traj_evolution.py --rollouts <run>/rollouts [--json out]
"""
import argparse
import json
import re
import sys
from pathlib import Path


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, int(q * (len(s) - 1) + 0.5))
    return s[i]


def traj_features(d):
    steps = d["trajectory"].get("steps") or []
    thoughts = [str(s.get("thought") or "") for s in steps]
    actions = [str(s.get("action_raw") or "") for s in steps]
    t_chars = sum(len(t) for t in thoughts)
    a_chars = sum(len(a) for a in actions)
    turns = len(steps)
    # adjacent verbatim thought repeats (the established strongest indicator)
    rep = sum(1 for i in range(1, turns) if thoughts[i] and thoughts[i] == thoughts[i - 1])
    rep_rate = rep / max(turns - 1, 1)
    uniq = len(set(t for t in thoughts if t)) / max(sum(1 for t in thoughts if t), 1)
    return {
        "tok": d.get("response_token_len", 0),
        "env_steps": d.get("env_steps", 0),
        "turns": turns,
        "thought_chars": t_chars,
        "thought_chars_per_turn": t_chars / max(turns, 1),
        "action_chars": a_chars,
        "thought_action_ratio": t_chars / max(a_chars, 1),
        "rep_rate": rep_rate,
        "uniq_thought": uniq,
        "truncated": bool(d.get("truncated")),
    }


def agg(feats):
    if not feats:
        return {"n": 0}
    keys = ["tok", "env_steps", "turns", "thought_chars", "thought_chars_per_turn",
            "thought_action_ratio", "rep_rate", "uniq_thought"]
    out = {"n": len(feats)}
    for k in keys:
        vals = [f[k] for f in feats]
        out[k + "_mean"] = round(sum(vals) / len(vals), 2)
    out["tok_p50"] = pct([f["tok"] for f in feats], 0.5)
    out["tok_p90"] = pct([f["tok"] for f in feats], 0.9)
    out["thought_chars_p90"] = pct([f["thought_chars"] for f in feats], 0.9)
    out["trunc_rate"] = round(sum(f["truncated"] for f in feats) / len(feats), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, type=Path,
                    help="a run's rollouts/ dir (step_XXXXX subdirs)")
    ap.add_argument("--json", type=Path, default=None,
                    help="optional: write the full payload here (stdout keeps the table)")
    args = ap.parse_args()
    global ROOT
    ROOT = args.rollouts
    rows = []
    samples = []
    for sd in sorted(ROOT.iterdir()):
        m = re.match(r"^step_(\d{5})(_val)?$", sd.name)
        if not m:
            continue
        step, is_val = int(m.group(1)), bool(m.group(2))
        won_f, lost_f = [], []
        won_docs = []
        for f in sd.glob("*.json"):
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            feat = traj_features(d)
            (won_f if d.get("won") else lost_f).append(feat)
            if d.get("won"):
                won_docs.append(d)
        rows.append({"step": step, "val": is_val,
                     "won": agg(won_f), "lost": agg(lost_f)})
        # qualitative: 2 won thought samples at early/mid/late train steps
        if not is_val and step in (2, 9, 14, 16, 20, 24) and won_docs:
            for d in won_docs[:2]:
                steps = d["trajectory"].get("steps") or []
                samples.append({
                    "step": step, "task": d["task_id"][:60],
                    "tok": d.get("response_token_len"),
                    "turns": len(steps),
                    "thoughts_first3": [str(s.get("thought") or "")[:220] for s in steps[:3]],
                })
    payload = {"rollouts": str(ROOT), "rows": rows, "samples": samples}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    hdr = (f"{'step':>4} {'val':>3} | {'n_w':>4} {'tok_w':>7} {'th/turn':>8} "
           f"{'ratio':>6} {'env':>5} | {'n_l':>4} {'tok_l':>8}")
    print(hdr)
    for r in rows:
        w, l = r["won"], r["lost"]
        g = lambda d, k: d.get(k, "-")
        print(f"{r['step']:>4} {str(r['val'])[0]:>3} | {w['n']:>4} {g(w,'tok_mean'):>7} "
              f"{g(w,'thought_chars_per_turn_mean'):>8} {g(w,'thought_action_ratio_mean'):>6} "
              f"{g(w,'env_steps_mean'):>5} | {l['n']:>4} {g(l,'tok_mean'):>8}")


if __name__ == "__main__":
    sys.exit(main())
