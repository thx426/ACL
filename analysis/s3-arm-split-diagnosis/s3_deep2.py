#!/usr/bin/env python3
"""s3 深诊断第二批(CD-010,只读)。

  A. reasoning 收尾倾向:内容段中途 token 522(`</`)的 top16 出现率/logp,
     以及内容段末 3 位的 522 概率 —— "收尾被压 vs 内容自发变长"的微观判决。
  B. 主模式侵蚀:每位置 π top-1 logp 的分布(抽样 1/4),分赢/输。
  C. A_bare/A_steer 独立复核(GRPO-LOO 重算)+ 退化组。
  D. family × arm 的 won 分解。
  E. rejected 行 final_pi_rank 谱系(1/2/3/4+)。
  F. reasoning 内容段长度(比轮长更干净的"想多久"口径)。

用法: python3 s3_deep2.py <run_dir> <step> [...] > out2.json
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

CLOSE_TOK = 522        # "</"  (</reasoning> 与 </action> 的首 token)
REASON_2ND = 19895     # "reason" 片段:[522,19895]=</reasoning 前缀


def pctl(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def med(xs):
    return pctl(xs, 0.5)


def analyze_step(run_dir, step):
    files = sorted(glob.glob(os.path.join(
        run_dir, "rollouts", f"step_{step:05d}", "*.json")))
    out = {"step": step, "n_files": len(files)}
    if len(files) < 200:
        out["skipped"] = True
        return out

    close_mid = {"in16": 0, "tot": 0, "logp": []}   # 内容段中途(去末3)
    close_tail = {"in16": 0, "tot": 0, "logp": []}  # 内容段末 3 位
    reason_len = []                                  # 内容段 token 长
    top1 = {(True, "normal"): [], (False, "normal"): [], (False, "dead"): [],
            (True, "dead"): []}
    fam = defaultdict(lambda: Counter())             # family -> Counter
    rank_hist = Counter()
    groups = defaultdict(lambda: {"bare": [], "steered": []})

    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        st = d.get("steering", {})
        arm = {0: "bare", 1: "steered"}.get(st.get("steer_arm", "?"), "?")
        won = bool(d.get("won"))
        task = d.get("task_id", "")
        family = task.split("__")[1].split("-")[0] if "__" in task else "?"
        fam[family][f"{arm}_n"] += 1
        fam[family][f"{arm}_won"] += int(won)
        if arm in ("bare", "steered"):
            groups[task][arm].append(int(won))

        if arm != "steered":
            continue
        spans = d.get("turn_token_spans") or []
        steps = (d.get("trajectory") or {}).get("steps") or []
        dead = [False] * len(steps)
        for i in range(1, len(steps)):
            if (steps[i].get("thought") == steps[i - 1].get("thought")
                    and steps[i].get("action_raw") == steps[i - 1].get("action_raw")):
                dead[i] = True
        sp = st.get("steer_positions") or []
        pos2row = {p: i for i, p in enumerate(sp)}
        pi16 = st.get("steer_pi_top16") or []
        ranks = st.get("steer_final_pi_rank") or []
        rejected = st.get("steer_rejected") or []
        ids = d.get("response_token_ids") or []

        for r, isrej in enumerate(rejected):
            if isrej and r < len(ranks) and ranks[r] >= 1:
                rank_hist[min(ranks[r], 4)] += 1

        nturn = min(len(spans), len(steps))
        for i in range(nturn):
            s0, e0 = spans[i][0], spans[i][1]
            kind = "dead" if dead[i] else "normal"
            # reasoning 内容段:轮首 <reasoning>(4 tok)之后到首个 [522,19895]
            c0 = s0 + 4
            c1 = None
            for p in range(c0, e0 - 1):
                if p < len(ids) and ids[p] == CLOSE_TOK and ids[p + 1] == REASON_2ND:
                    c1 = p
                    break
            if c1 is None or c1 <= c0:
                continue
            if kind == "normal":
                reason_len.append(c1 - c0)
            # A: 中途 vs 末 3 位的 close-token 倾向(正常轮)
            if kind == "normal":
                for p in range(c0, c1):
                    slot = close_tail if p >= c1 - 3 else close_mid
                    if (p - c0) % 2 and slot is close_mid:
                        continue          # 中途抽样 1/2
                    row = pos2row.get(p)
                    if row is None or row >= len(pi16) or not pi16[row]:
                        continue
                    slot["tot"] += 1
                    for tid, lp in pi16[row]:
                        if tid == CLOSE_TOK:
                            slot["in16"] += 1
                            slot["logp"].append(lp)
                            break
            # B: top-1 logp 抽样(1/4 位置)
            for p in range(s0, e0, 4):
                row = pos2row.get(p)
                if row is None or row >= len(pi16) or not pi16[row]:
                    continue
                top1[(won, kind)].append(pi16[row][0][1])

    # C: GRPO-LOO 复核(混合组;r=won)
    a_bare, a_bare_fail, a_steer, degen = [], [], [], 0
    for task, g in groups.items():
        rs = [(w, "bare") for w in g["bare"]] + [(w, "steered") for w in g["steered"]]
        if len(rs) < 2:
            continue
        vals = [w for w, _ in rs]
        if len(set(vals)) == 1:
            degen += 1
            continue
        tot = sum(vals)
        n = len(vals)
        for w, arm in rs:
            adv = w - (tot - w) / (n - 1)
            if arm == "bare":
                a_bare.append(adv)
                if not w:
                    a_bare_fail.append(adv)
            else:
                a_steer.append(adv)

    def slot_out(s):
        return {"rate": (s["in16"] / s["tot"]) if s["tot"] else None,
                "tot": s["tot"], "logp_med": med(s["logp"])}

    out["close_mid"] = slot_out(close_mid)
    out["close_tail"] = slot_out(close_tail)
    out["reason_len"] = {"med": med(reason_len), "p90": pctl(reason_len, .9),
                         "n": len(reason_len)}
    out["top1_logp"] = {
        f"{'won' if w else 'lost'}_{k}": {"n": len(v), "med": med(v),
                                          "p25": pctl(v, .25)}
        for (w, k), v in top1.items() if v}
    out["adv"] = {
        "A_bare_mean": round(sum(a_bare) / len(a_bare), 4) if a_bare else None,
        "A_bare_fail_mean": (round(sum(a_bare_fail) / len(a_bare_fail), 4)
                             if a_bare_fail else None),
        "A_steer_mean": round(sum(a_steer) / len(a_steer), 4) if a_steer else None,
        "degenerate_groups": degen, "n_groups": len(groups)}
    out["family"] = {k: dict(v) for k, v in sorted(fam.items())}
    out["rank_hist"] = dict(sorted(rank_hist.items()))
    return out


def main():
    run_dir = sys.argv[1]
    res = []
    for s in [int(x) for x in sys.argv[2:]]:
        print(f"[deep2] step {s} ...", file=sys.stderr, flush=True)
        res.append(analyze_step(run_dir, s))
    json.dump(res, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
