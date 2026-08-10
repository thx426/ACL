"""python|engine executor 统计对拍门(Engine-Resident SPS P0b,设计文档 §4)。

用法(两个 rollout dump 目录 = 同 ckpt 同任务批、分别以两种 executor 跑出):

    python scripts/engine_sps_parity_check.py \
        --a runs/<python-run>/rollouts/step_XXXXX \
        --b runs/<engine-run>/rollouts/step_XXXXX \
        [--tol-rate 0.03] [--tol-hist 0.10]

只比 steered 执行行(steer_arm==1 且账本在场)。两路径是独立采样,逐 token
不可比——本门比的是**分布层统计**:开门率/干预率/拒绝率/预算/直方图形状,
外加各自独立的密度比界核验(|log_b−log_pi| ≤ 2ηc,违例必须为 0)。
逐位 bit 级等价由单测层锁死(tests/steering/test_engine_executor.py 的
fake 世界对拍);本门管的是真机整链。

退出码 0 = PASS;1 = FAIL(超差项已打印)。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def _load_steered(dirpath: str) -> list[dict]:
    rows = []
    for fp in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
        try:
            with open(fp) as f:
                d = json.load(f)
        except Exception:
            continue
        st = d.get("steering") or {}
        if d.get("sample_role") == "research":
            continue
        if int(st.get("steer_arm", 1)) != 1 or "steer_record_rows" not in st:
            continue
        rows.append(d)
    return rows


def _pct(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    return float(s[min(len(s) - 1, int(q * len(s)))])


def _summarize(rows: list[dict]) -> dict:
    st_all = [r["steering"] for r in rows]
    n_cand = sum(s.get("steer_n_candidates", 0) for s in st_all)
    d_hist = None
    ent_hist = None
    for s in st_all:
        for key, acc in (("steer_d_hist", "d"), ("steer_ent_hist", "e")):
            h = s.get(key)
            if not h:
                continue
            if acc == "d":
                d_hist = [a + b for a, b in zip(d_hist, h)] if d_hist else list(h)
            else:
                ent_hist = [a + b for a, b in zip(ent_hist, h)] if ent_hist else list(h)
    bound_violations = 0
    for s in st_all:
        cap = float(s.get("steer_log_ratio_cap", 0.0)) + 1e-6
        for lp, lb in zip(s.get("steer_log_pi", []), s.get("steer_log_b", [])):
            if abs(float(lb) - float(lp)) > cap:
                bound_violations += 1
    total_rows = sum(s.get("steer_record_rows", 0) for s in st_all)
    return {
        "n_trajs": len(rows),
        "won_rate": (sum(1 for r in rows if r.get("won")) / len(rows)) if rows else 0.0,
        "resp_len_p50": _pct([r.get("response_token_len", 0) for r in rows], 0.50),
        "resp_len_p90": _pct([r.get("response_token_len", 0) for r in rows], 0.90),
        "open_rate": (sum(s.get("steer_n_base_open", 0) for s in st_all) / n_cand) if n_cand else 0.0,
        "intervene_rate": (sum(s.get("steer_count", 0) for s in st_all) / n_cand) if n_cand else 0.0,
        "reject_rate": (sum(s.get("steer_reject_count", 0) for s in st_all) / total_rows) if total_rows else 0.0,
        "gate_bypassed_per_traj": (sum(s.get("steer_n_gate_bypassed", 0) for s in st_all) / len(rows)) if rows else 0.0,
        "budget_spent_mean": (sum(float(s.get("steer_budget_spent", 0.0)) for s in st_all) / len(rows)) if rows else 0.0,
        "d_hist": d_hist,
        "ent_hist": ent_hist,
        "bound_violations": bound_violations,
    }


def _hist_l1(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0 if not a and not b else 1.0
    sa, sb = sum(a) or 1, sum(b) or 1
    return sum(abs(x / sa - y / sb) for x, y in zip(a, b)) / 2.0


def compare(sum_a: dict, sum_b: dict, *, tol_rate: float, tol_hist: float) -> list[str]:
    """返回超差项清单(空 = PASS)。"""
    fails: list[str] = []
    for side, s in (("A", sum_a), ("B", sum_b)):
        if s["n_trajs"] == 0:
            fails.append(f"{side}: no steered rows found")
        if s["bound_violations"]:
            fails.append(f"{side}: {s['bound_violations']} density-ratio bound "
                         "violations (|log_b - log_pi| > 2*eta*c) — kernel broken")
    for key in ("open_rate", "intervene_rate", "reject_rate", "won_rate"):
        d = abs(sum_a[key] - sum_b[key])
        if d > tol_rate:
            fails.append(f"{key}: |{sum_a[key]:.4f} - {sum_b[key]:.4f}| = {d:.4f} > {tol_rate}")
    # 预算相对差(量纲随 run,相对口径)
    ma = max(abs(sum_a["budget_spent_mean"]), abs(sum_b["budget_spent_mean"]), 1e-9)
    d = abs(sum_a["budget_spent_mean"] - sum_b["budget_spent_mean"]) / ma
    if d > max(tol_rate * 3, 0.10):
        fails.append(f"budget_spent_mean: relative diff {d:.4f} > {max(tol_rate * 3, 0.10)}")
    for key in ("d_hist", "ent_hist"):
        l1 = _hist_l1(sum_a[key], sum_b[key])
        if l1 > tol_hist:
            fails.append(f"{key}: normalized L1 {l1:.4f} > {tol_hist}")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="python-executor rollout dir")
    ap.add_argument("--b", required=True, help="engine-executor rollout dir")
    ap.add_argument("--tol-rate", type=float, default=0.03)
    ap.add_argument("--tol-hist", type=float, default=0.10)
    args = ap.parse_args()

    sum_a = _summarize(_load_steered(args.a))
    sum_b = _summarize(_load_steered(args.b))

    print(f"{'metric':<24}{'A(python)':>14}{'B(engine)':>14}")
    for key in ("n_trajs", "won_rate", "resp_len_p50", "resp_len_p90", "open_rate",
                "intervene_rate", "reject_rate", "gate_bypassed_per_traj",
                "budget_spent_mean", "bound_violations"):
        va, vb = sum_a[key], sum_b[key]
        fmt = ".4f" if isinstance(va, float) else "d"
        print(f"{key:<24}{va:>14{fmt}}{vb:>14{fmt}}")
    print(f"{'d_hist L1':<24}{_hist_l1(sum_a['d_hist'], sum_b['d_hist']):>14.4f}")
    print(f"{'ent_hist L1':<24}{_hist_l1(sum_a['ent_hist'], sum_b['ent_hist']):>14.4f}")

    fails = compare(sum_a, sum_b, tol_rate=args.tol_rate, tol_hist=args.tol_hist)
    if fails:
        print("\nPARITY FAIL:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nPARITY PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
