#!/usr/bin/env python3
"""s3 maxopen 深诊断(CD-010 独立分析;只读)。

新测量(相对 CD-006 的 s3_* 脚本):
  M1' 终止倾向:steered 臂逐轮末 token 的 log_pi + mid-turn im_end 在 pi_top16
      的出现率/logp —— 直接检验「失败轨迹负梯度压低终止 token → 每轮变长」假设。
  M3  吸收速率:rejected 行 Δ=logp(draft)−logp(final)(均在 π 下)随 step 的
      演化 + final_is_top1 率 —— 教师"换序"是否被 policy 逐步内化。
  M2  researcher/doc 漂移:doc_words/experiments/final_success/error 随 step。
  形态:正常轮长直方图(平移 vs 尾拉)、轮内 5-gram 重复率(rambling)、
      拒绝位的轮位分布。

用法: python3 s3_deep.py <run_dir> <step> [<step> ...]  > out.json
"""
import glob
import json
import os
import sys
from collections import Counter


def pctl(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def med(xs):
    return pctl(xs, 0.5)


def gram_repeat_rate(text, n=5):
    """轮内 n-gram(词级)重复率:重复出现的 n-gram 占比。"""
    w = text.split()
    if len(w) < n * 2:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    c = Counter(grams)
    rep = sum(v for v in c.values() if v > 1)
    return rep / len(grams)


def hist_bucket(x):
    for b in (25, 50, 75, 100, 150, 200, 300, 500):
        if x <= b:
            return b
    return 999


def analyze_step(run_dir, step):
    files = sorted(glob.glob(os.path.join(
        run_dir, "rollouts", f"step_{step:05d}", "*.json")))
    out = {"step": step, "n_files": len(files)}
    if len(files) < 200:
        out["skipped"] = True
        return out

    # 聚合器
    arm_stat = {}          # arm -> dict
    turn_len = {}          # (arm, kind) -> [len]  kind in dead/normal
    norm_hist = {}         # arm -> Counter bucket
    end_logpi = {"dead": [], "normal": []}      # 轮末 token log_pi(steered)
    end_tok = Counter()                          # 轮末 token id 众数(im_end 检测)
    mid_imend = {"in16": 0, "tot": 0, "logp": []}  # mid-turn im_end 潜在倾向
    pre_imend = {}                                  # 轮末倒数 k 位置 im_end 倾向
    rej = {"dlogp": [], "final_top1": 0, "n": 0, "miss16": 0,
           "pos_rel": [], "turn1": 0, "d": []}
    ram = {"words": [], "rep5": []}              # 正常轮 thought
    research = {}          # task_id -> research dict(去重)
    won_arm = Counter()
    per_traj_deadfrac = {"bare": [], "steered": []}

    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        st = d.get("steering", {})
        arm = st.get("steer_arm", "?")
        arm = {0: "bare", 1: "steered"}.get(arm, str(arm))
        won = bool(d.get("won"))
        won_arm[(arm, won)] += 1
        a = arm_stat.setdefault(arm, {"n": 0, "len": [], "turns": []})
        a["n"] += 1
        a["len"].append(d.get("response_token_len", 0))
        a["turns"].append(d.get("assistant_turns", 0))

        research.setdefault(d.get("task_id"), d.get("research", {}))

        spans = d.get("turn_token_spans") or []
        steps = (d.get("trajectory") or {}).get("steps") or []
        # 死锁轮标记(与 CD-006 s3_deadlock 同定义:与前轮 (thought,action) 逐字同)
        dead = [False] * len(steps)
        for i in range(1, len(steps)):
            if (steps[i].get("thought") == steps[i - 1].get("thought")
                    and steps[i].get("action_raw") == steps[i - 1].get("action_raw")):
                dead[i] = True
        ndead = sum(dead)
        if steps:
            per_traj_deadfrac.setdefault(arm, []).append(ndead / len(steps))

        # steered 臂:position -> record row 索引
        pos2row = {}
        if arm == "steered":
            sp = st.get("steer_positions") or []
            pos2row = {p: i for i, p in enumerate(sp)}
        log_pi = st.get("steer_log_pi") or []
        pi16 = st.get("steer_pi_top16") or []
        rejected = st.get("steer_rejected") or []
        draft_t = st.get("steer_draft_token") or []
        final_t = st.get("steer_final_token") or []
        d_vals = st.get("steer_d") or []
        resp_len = max(1, d.get("response_token_len", 1))
        tok_ids = d.get("response_token_ids") or []

        nturn = min(len(spans), len(steps))
        for i in range(nturn):
            s0, e0 = spans[i][0], spans[i][1]
            tl = e0 - s0
            kind = "dead" if dead[i] else "normal"
            turn_len.setdefault((arm, kind), []).append(tl)
            if kind == "normal":
                norm_hist.setdefault(arm, Counter())[hist_bucket(tl)] += 1
                th = steps[i].get("thought") or ""
                ram["words"].append(len(th.split()))
                ram["rep5"].append(gram_repeat_rate(th))
            if arm != "steered":
                continue
            # M1' c1: 轮末 token log_pi
            pe = e0 - 1
            if tok_ids and pe < len(tok_ids):
                end_tok[tok_ids[pe]] += 1
            r = pos2row.get(pe)
            if r is not None and r < len(log_pi):
                end_logpi[kind].append(log_pi[r])
            # M1' c2: mid-turn im_end 潜在倾向(每轮中点 1 个位置)
            pm = (s0 + e0) // 2
            r = pos2row.get(pm)
            if r is not None and r < len(pi16) and pi16[r]:
                mid_imend["tot"] += 1
                for tid, lp in pi16[r]:
                    if tid == 151645:  # <|im_end|>(以 end_tok 众数复核)
                        mid_imend["in16"] += 1
                        mid_imend["logp"].append(lp)
                        break
            # M1' c3: 轮末倒数 k 位置上 im_end 的预备概率(终止被压的敏感读数)
            for k in (2, 4, 8):
                r = pos2row.get(pe - k)
                if r is None or r >= len(pi16) or not pi16[r]:
                    continue
                slot = pre_imend.setdefault(k, {"in16": 0, "tot": 0, "logp": []})
                slot["tot"] += 1
                for tid, lp in pi16[r]:
                    if tid == 151645:
                        slot["in16"] += 1
                        slot["logp"].append(lp)
                        break

        # M3: rejected 行
        if arm == "steered" and rejected:
            sp = st.get("steer_positions") or []
            for r, isrej in enumerate(rejected):
                if not isrej:
                    continue
                rej["n"] += 1
                if r < len(sp):
                    rej["pos_rel"].append(sp[r] / resp_len)
                    if spans and sp[r] < spans[0][1]:
                        rej["turn1"] += 1
                if r < len(d_vals):
                    rej["d"].append(d_vals[r])
                top = pi16[r] if r < len(pi16) else []
                if not top:
                    continue
                lp = {tid: v for tid, v in top}
                dt, ft = draft_t[r], final_t[r]
                if top and top[0][0] == ft:
                    rej["final_top1"] += 1
                if dt in lp and ft in lp:
                    rej["dlogp"].append(lp[dt] - lp[ft])
                else:
                    rej["miss16"] += 1

    # 汇总输出(小)
    def _arm(a):
        s = arm_stat.get(a, {"n": 0, "len": [], "turns": []})
        return {"n": s["n"], "len_med": med(s["len"]), "len_p90": pctl(s["len"], .9),
                "turns_med": med(s["turns"])}

    out["arms"] = {a: _arm(a) for a in arm_stat}
    out["won"] = {f"{a}_{w}": c for (a, w), c in won_arm.items()}
    out["turn_len"] = {
        f"{a}_{k}": {"n": len(v), "med": med(v), "p90": pctl(v, .9)}
        for (a, k), v in turn_len.items()}
    out["norm_hist"] = {a: dict(sorted(c.items())) for a, c in norm_hist.items()}
    out["dead_frac"] = {a: round(sum(v) / len(v), 4) if v else None
                        for a, v in per_traj_deadfrac.items()}
    out["end_logpi"] = {k: {"n": len(v), "med": med(v), "p25": pctl(v, .25)}
                        for k, v in end_logpi.items()}
    out["end_tok_top"] = end_tok.most_common(3)
    out["mid_imend"] = {"rate": (mid_imend["in16"] / mid_imend["tot"]
                                 if mid_imend["tot"] else None),
                        "tot": mid_imend["tot"],
                        "logp_med": med(mid_imend["logp"])}
    out["pre_imend"] = {
        str(k): {"rate": (v["in16"] / v["tot"]) if v["tot"] else None,
                 "tot": v["tot"], "logp_med": med(v["logp"])}
        for k, v in sorted(pre_imend.items())}
    out["reject"] = {
        "n": rej["n"], "miss16": rej["miss16"],
        "dlogp_med": med(rej["dlogp"]), "dlogp_p25": pctl(rej["dlogp"], .25),
        "dlogp_p75": pctl(rej["dlogp"], .75),
        "final_top1_rate": (rej["final_top1"] / rej["n"]) if rej["n"] else None,
        "pos_rel_med": med(rej["pos_rel"]),
        "turn1_frac": (rej["turn1"] / rej["n"]) if rej["n"] else None,
        "d_med": med(rej["d"]),
    }
    out["rambling"] = {"thought_words_med": med(ram["words"]),
                       "rep5_med": med(ram["rep5"]),
                       "rep5_p90": pctl(ram["rep5"], .9)}
    rs = [r for r in research.values() if r]
    out["research"] = {
        "n_tasks": len(rs),
        "doc_words_med": med([r.get("research_doc_words", 0) for r in rs]),
        "experiments_med": med([r.get("research_experiments", 0) for r in rs]),
        "final_success_mean": (sum(r.get("research_final_success", 0) for r in rs)
                               / len(rs)) if rs else None,
        "error_rate": (sum(1 for r in rs if r.get("research_error"))
                       / len(rs)) if rs else None,
        "doc_missing_rate": (sum(1 for r in rs if r.get("research_doc_missing"))
                             / len(rs)) if rs else None,
        "turns_med": med([r.get("research_turns", 0) for r in rs]),
    }
    return out


def main():
    run_dir = sys.argv[1]
    steps = [int(x) for x in sys.argv[2:]]
    res = []
    for s in steps:
        print(f"[deep] step {s} ...", file=sys.stderr, flush=True)
        res.append(analyze_step(run_dir, s))
    json.dump(res, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
