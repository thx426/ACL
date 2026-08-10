"""python|engine executor 统计对拍门(Engine-Resident SPS P0b,设计文档 §4)。

用法(两个 rollout dump 目录 = 同 ckpt 同任务批、分别以两种 executor 跑出):

    python scripts/engine_sps_parity_check.py \
        --a runs/<python-run>/rollouts/step_XXXXX \
        --b runs/<engine-run>/rollouts/step_XXXXX \
        [--tol-rate 0.03] [--tol-hist 0.10]

**PASS 的语义**(codex review P1-2 修订):账本完整性 + 机制统计等价——
即两路径的门控/干预/拒绝行为与密度比界在分布层一致。它**不构成任务效果
等价证明**:won/长度是低功效参考指标(报告差值与 2σ,不进判定),效果面
由 V2 的 bare 同水位硬前提与任务级 outcome 检查承担。

判定项(任一违反即 FAIL,codex review P1-1 修订——fail-open 全部封死):
  * 输入完整性:坏 JSON / 缺 steering 键 / 稀疏数组长度 ≠ steer_record_rows
    / 非有限值 / positions 非严增 / 直方图宽度漂移 —— 一律 FAIL 并点名文件;
  * 逐行密度比界:|log_b − log_pi| ≤ steer_delta(该行实际 2ηc)+1e-6,
    违例数必须为 0(轨迹级全局 cap 只是上界,行级 delta 才是紧界);
  * 任务集合:两侧 steered 行的 task_id 多重集与 global_step 完全一致;
  * 机制率(公共分母 = steer_ent_count,全部执行位含 bypass 段——G-E
    bypass 触发面随 K 合法变化,用 n_candidates 作分母会比较不同总体):
    open_rate / intervene_rate;reject_rate 分母 = steer_record_rows
    (两路径倾斜行集合恒等,总体一致);容差 = max(tol, 两比例合并 2σ);
  * ent_hist 归一化 L1 ≤ tol_hist(熵观察在全部执行位,总体公共)。

报告项(打印不判定):won(差值+2σ)/ resp_len p50、p90 / d_hist L1
(bypass 改变其观察总体,结构性不可判)/ gate_bypassed / draft_discarded
/ budget_spent。

退出码 0 = PASS;1 = FAIL(超差项已打印)。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

_EPS = 1e-6
_ROW_ARRAYS = ("steer_positions", "steer_log_pi", "steer_log_b", "steer_delta")


def load_steered(dirpath: str) -> tuple[list[dict], list[str]]:
    """(steered 行, 输入错误清单)。任何解析失败都进错误清单,绝不静默跳过。"""
    rows: list[dict] = []
    errors: list[str] = []
    files = sorted(glob.glob(os.path.join(dirpath, "*.json")))
    if not files:
        errors.append(f"{dirpath}: no *.json files")
    for fp in files:
        try:
            with open(fp) as f:
                d = json.load(f)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{os.path.basename(fp)}: unreadable JSON ({e})")
            continue
        if d.get("sample_role") == "research":
            continue
        st = d.get("steering")
        if not isinstance(st, dict):
            errors.append(f"{os.path.basename(fp)}: missing steering dict")
            continue
        if int(st.get("steer_arm", 1)) != 1 or "steer_record_rows" not in st:
            continue
        d["_file"] = os.path.basename(fp)
        rows.append(d)
    return rows, errors


def integrity_errors(rows: list[dict]) -> list[str]:
    """账本结构与密度比界的逐行审计(codex review P1-1)。"""
    errs: list[str] = []
    hist_widths: dict[str, int] = {}
    for r in rows:
        st, name = r["steering"], r["_file"]
        n = int(st.get("steer_record_rows", -1))
        for key in _ROW_ARRAYS:
            arr = st.get(key)
            if not isinstance(arr, list) or len(arr) != n:
                errs.append(f"{name}: {key} length {None if arr is None else len(arr)} != steer_record_rows {n}")
                break
        else:
            pos = st["steer_positions"]
            if any(b <= a for a, b in zip(pos, pos[1:])):
                errs.append(f"{name}: steer_positions not strictly increasing")
            for lp, lb, dl in zip(st["steer_log_pi"], st["steer_log_b"], st["steer_delta"]):
                if not (math.isfinite(lp) and math.isfinite(lb) and math.isfinite(dl)):
                    errs.append(f"{name}: non-finite ledger value")
                    break
                if abs(lb - lp) > dl + _EPS:
                    errs.append(
                        f"{name}: density-ratio bound violated |{lb:.4f}-{lp:.4f}| "
                        f"> per-row delta {dl:.4f}"
                    )
                    break
        for hkey in ("steer_d_hist", "steer_ent_hist"):
            h = st.get(hkey)
            if isinstance(h, list):
                w = hist_widths.setdefault(hkey, len(h))
                if len(h) != w:
                    errs.append(f"{name}: {hkey} width {len(h)} != {w}")
    return errs


def _pct(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    return float(s[min(len(s) - 1, int(q * len(s)))])


def summarize(rows: list[dict]) -> dict:
    st_all = [r["steering"] for r in rows]
    n_exec = sum(int(s.get("steer_ent_count", 0)) for s in st_all)   # 公共分母:全部执行位
    total_rows = sum(int(s.get("steer_record_rows", 0)) for s in st_all)

    def _hist(key):
        acc = None
        for s in st_all:
            h = s.get(key)
            if isinstance(h, list) and h:
                acc = [a + b for a, b in zip(acc, h)] if acc else list(h)
        return acc

    return {
        "n_trajs": len(rows),
        "tasks": sorted(str(r.get("task_id")) for r in rows),
        "global_steps": sorted({r.get("global_step") for r in rows}),
        "_denoms": {"won_rate": len(rows), "open_rate": n_exec,
                    "intervene_rate": n_exec, "reject_rate": total_rows},
        "won_rate": (sum(1 for r in rows if r.get("won")) / len(rows)) if rows else 0.0,
        "resp_len_p50": _pct([r.get("response_token_len", 0) for r in rows], 0.50),
        "resp_len_p90": _pct([r.get("response_token_len", 0) for r in rows], 0.90),
        "open_rate": (sum(s.get("steer_n_base_open", 0) for s in st_all) / n_exec) if n_exec else 0.0,
        "intervene_rate": (sum(s.get("steer_count", 0) for s in st_all) / n_exec) if n_exec else 0.0,
        "reject_rate": (sum(s.get("steer_reject_count", 0) for s in st_all) / total_rows) if total_rows else 0.0,
        "gate_bypassed_per_traj": (sum(s.get("steer_n_gate_bypassed", 0) for s in st_all) / len(rows)) if rows else 0.0,
        "draft_discarded_per_traj": (sum(s.get("steer_n_draft_discarded", 0) for s in st_all) / len(rows)) if rows else 0.0,
        "budget_spent_mean": (sum(float(s.get("steer_budget_spent", 0.0)) for s in st_all) / len(rows)) if rows else 0.0,
        "d_hist": _hist("steer_d_hist"),
        "ent_hist": _hist("steer_ent_hist"),
    }


def hist_l1(a, b) -> float:
    if not a and not b:
        return 0.0
    if not a or not b or len(a) != len(b):
        return 1.0
    sa, sb = sum(a) or 1, sum(b) or 1
    return sum(abs(x / sa - y / sb) for x, y in zip(a, b)) / 2.0


def _two_sigma(pa, pb, na, nb) -> float:
    na, nb = max(na, 1), max(nb, 1)
    p = (pa * na + pb * nb) / (na + nb)
    return 2.0 * (max(p * (1 - p), 1e-12) * (1 / na + 1 / nb)) ** 0.5


def compare(sum_a: dict, sum_b: dict, in_errors: list[str], *,
            tol_rate: float, tol_hist: float) -> list[str]:
    """返回超差项清单(空 = PASS)。"""
    fails = list(in_errors)
    for side, s in (("A", sum_a), ("B", sum_b)):
        if s["n_trajs"] == 0:
            fails.append(f"{side}: no steered rows found")
    if sum_a["tasks"] != sum_b["tasks"]:
        only_a = set(sum_a["tasks"]) - set(sum_b["tasks"])
        only_b = set(sum_b["tasks"]) - set(sum_a["tasks"])
        fails.append(f"task multiset mismatch (A-only={sorted(only_a)[:4]}, "
                     f"B-only={sorted(only_b)[:4]}, or count drift)")
    if sum_a["global_steps"] != sum_b["global_steps"]:
        fails.append(f"global_step mismatch: {sum_a['global_steps']} vs {sum_b['global_steps']}")
    for key in ("open_rate", "intervene_rate", "reject_rate"):
        d = abs(sum_a[key] - sum_b[key])
        tol = max(tol_rate, _two_sigma(sum_a[key], sum_b[key],
                                       sum_a["_denoms"][key], sum_b["_denoms"][key]))
        if d > tol:
            fails.append(f"{key}: |{sum_a[key]:.4f} - {sum_b[key]:.4f}| = {d:.4f} > {tol:.4f}")
    l1 = hist_l1(sum_a["ent_hist"], sum_b["ent_hist"])
    if l1 > tol_hist:
        fails.append(f"ent_hist: normalized L1 {l1:.4f} > {tol_hist}")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="python-executor rollout dir")
    ap.add_argument("--b", required=True, help="engine-executor rollout dir")
    ap.add_argument("--tol-rate", type=float, default=0.03)
    ap.add_argument("--tol-hist", type=float, default=0.10)
    args = ap.parse_args()

    rows_a, err_a = load_steered(args.a)
    rows_b, err_b = load_steered(args.b)
    in_errors = [f"A/{e}" for e in err_a + integrity_errors(rows_a)] \
        + [f"B/{e}" for e in err_b + integrity_errors(rows_b)]
    sum_a, sum_b = summarize(rows_a), summarize(rows_b)

    print(f"{'metric':<26}{'A(python)':>14}{'B(engine)':>14}")
    for key in ("n_trajs", "won_rate", "resp_len_p50", "resp_len_p90", "open_rate",
                "intervene_rate", "reject_rate", "gate_bypassed_per_traj",
                "draft_discarded_per_traj", "budget_spent_mean"):
        va, vb = sum_a[key], sum_b[key]
        fmt = ".4f" if isinstance(va, float) else "d"
        print(f"{key:<26}{va:>14{fmt}}{vb:>14{fmt}}")
    won_sigma = _two_sigma(sum_a["won_rate"], sum_b["won_rate"],
                           sum_a["n_trajs"], sum_b["n_trajs"])
    print(f"{'won diff (ref only)':<26}{abs(sum_a['won_rate'] - sum_b['won_rate']):>14.4f}"
          f"{'±' + format(won_sigma, '.4f'):>14}")
    print(f"{'d_hist L1 (ref only)':<26}{hist_l1(sum_a['d_hist'], sum_b['d_hist']):>14.4f}")
    print(f"{'ent_hist L1':<26}{hist_l1(sum_a['ent_hist'], sum_b['ent_hist']):>14.4f}")

    fails = compare(sum_a, sum_b, in_errors, tol_rate=args.tol_rate, tol_hist=args.tol_hist)
    if fails:
        print(f"\nPARITY FAIL ({len(fails)} item(s)):")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nPARITY PASS — mechanism statistics + ledger integrity "
          "(NOT a task-effect equivalence proof; effect checks live in V2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
