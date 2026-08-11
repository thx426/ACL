#!/usr/bin/env python3
"""Rollout 行为漂移体检(benchmark 无关;2026-08-11 诊断指纹固化)。

用法:
    python scripts/rollout_health_check.py DUMP_DIR [DUMP_DIR ...] [--json]

按传入顺序把每个 dump 目录(train step 或 *_val)当作时间序列的一个点,
输出行为分布读数。2026-08-11 事故的教训:policy 退化最早的指纹是**行为
分布**(turns 塌缩/秒交率),不是 won 曲线(小样本 MDE 太大);val 双降
定案靠的正是这里固化的三类读数:

  1. turns 分布与 quick-submit 率(assistant_turns<=2 占比)——特权行为
     克隆的行为面指纹(val00 31 条 → val10 236 条);
  2. mid-EOS 行(turn span 中段出现轮末停止 token)——engine 档 P0-1
     (K 窗尾越过终止符)的零假阳性指纹,bare/val 恒 0;
  3. 分臂对照(won/len/turns by steer_arm)与门读数(theta/open_rate)。

只读、只报告、恒零退出码——与 engine_sps_parity_check.py 分工:parity 是
上线判定门(FAIL 阻断),本脚本是趋势观察(人读/曲线)。

字段依赖:框架统一 dump schema(response_token_ids / turn_token_spans /
won / truncated / invalid_format / assistant_turns / steering.steer_arm),
不依赖任何 benchmark 特定字段;轮末停止 token 从数据自举(全部 turn 末位
token 的众数),不依赖 tokenizer。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter


def load_rows(dirpath: str) -> tuple[list[dict], int, int]:
    """(executor 行, research 行数, 坏文件数)。"""
    rows, n_research, n_bad = [], 0, 0
    for f in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
        try:
            d = json.load(open(f))
        except (OSError, json.JSONDecodeError):
            n_bad += 1
            continue
        if d.get("sample_role") == "research":
            n_research += 1
            continue
        if "response_token_ids" in d:
            rows.append(d)
    return rows, n_research, n_bad


def bootstrap_stop_id(rows: list[dict]) -> int | None:
    """轮末停止 token 自举:所有 turn 末位 token 的众数(与 tokenizer 解耦;
    正常轮以停止符收尾,众数即停止 id——诊断期在 Qwen3.5 上验证为 248046)。"""
    tail = Counter()
    for d in rows:
        ids = d["response_token_ids"]
        for s in d.get("turn_token_spans") or []:
            if s[1] > s[0]:
                tail[int(ids[s[1] - 1])] += 1
    return tail.most_common(1)[0][0] if tail else None


def _p95(vals: list[float]) -> float:
    if not vals:
        return 0.0
    return sorted(vals)[max(0, int(len(vals) * 0.95) - 1)]


def summarize(rows: list[dict], stop_id: int | None) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    won = sum(bool(d.get("won")) for d in rows)
    trunc = sum(bool(d.get("truncated")) for d in rows)
    invalid = sum(1 for d in rows if d.get("invalid_format"))
    lens = [d.get("response_token_len") or len(d["response_token_ids"]) for d in rows]
    turns = [
        d.get("assistant_turns") or len(d.get("turn_token_spans") or [])
        for d in rows
    ]
    quick = sum(1 for t in turns if t <= 2)
    mid_rows = junk = 0
    if stop_id is not None:
        for d in rows:
            ids = d["response_token_ids"]
            hit = False
            for s in d.get("turn_token_spans") or []:
                seg = ids[s[0]:s[1]]
                mids = [i for i, t in enumerate(seg[:-1]) if t == stop_id]
                if mids:
                    hit = True
                    junk += (len(seg) - 1) - mids[0]
            mid_rows += hit
    out = {
        "n": n,
        "won": won / n,
        "won_untruncated": (won / (n - trunc)) if n > trunc else 0.0,
        "trunc": trunc,
        "invalid": invalid,
        "len_mean": sum(lens) / n,
        "len_p95": _p95(lens),
        "turns_mean": sum(turns) / n,
        "quick_submit_rate": quick / n,
        "mid_eos_rows": mid_rows,
        "junk_tokens": junk,
    }
    st_rows = [d for d in rows if (d.get("steering") or {}).get("steer_arm") is not None]
    if st_rows:
        thetas = [
            float(s["steer_entropy_theta"])
            for d in st_rows
            if (s := d["steering"]).get("steer_entropy_theta") is not None
        ]
        opens = [
            float(s["steer_gate_open_rate"])
            for d in st_rows
            if (s := d["steering"]).get("steer_gate_open_rate") is not None
        ]
        # steer_entropy_theta==-1.0 是自标定 warmup 哨兵(agent_loop 落账口径)
        live = [t for t in thetas if t >= 0.0]
        out["theta_mean"] = (sum(live) / len(live)) if live else -1.0
        out["theta_warmup_rows"] = sum(1 for t in thetas if t < 0.0)
        out["open_rate_mean"] = (sum(opens) / len(opens)) if opens else 0.0
    return out


def by_arm(rows: list[dict]) -> dict[str, list[dict]]:
    arms: dict[str, list[dict]] = {}
    for d in rows:
        a = (d.get("steering") or {}).get("steer_arm")
        key = "all" if a is None else f"arm{int(a)}"
        arms.setdefault(key, []).append(d)
    return arms


def fmt_block(label: str, s: dict) -> str:
    if s.get("n", 0) == 0:
        return f"{label:26s} (empty)"
    line = (
        f"{label:26s} n={s['n']:4d} won={s['won']:6.1%}"
        f" (untrunc {s['won_untruncated']:6.1%})"
        f" turns={s['turns_mean']:5.1f} quick<=2={s['quick_submit_rate']:6.1%}"
        f" len={s['len_mean']:7.0f}/p95 {s['len_p95']:6.0f}"
        f" trunc={s['trunc']:3d} inval={s['invalid']:3d}"
        f" midEOS={s['mid_eos_rows']:3d} junk={s['junk_tokens']:5d}"
    )
    if "open_rate_mean" in s:
        line += (
            f" | theta={s['theta_mean']:6.3f}"
            f" (warmup {s.get('theta_warmup_rows', 0)}) open={s['open_rate_mean']:5.3f}"
        )
    return line


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="+", help="dump 目录(按时间序传入)")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    args = ap.parse_args(argv)

    report = []
    for dirpath in args.dirs:
        rows, n_research, n_bad = load_rows(dirpath)
        stop_id = bootstrap_stop_id(rows)
        entry = {
            "dir": dirpath,
            "stop_id": stop_id,
            "research_rows": n_research,
            "unreadable": n_bad,
            "overall": summarize(rows, stop_id),
            "arms": {k: summarize(v, stop_id) for k, v in by_arm(rows).items()},
        }
        report.append(entry)

    if args.json:
        print(json.dumps(report, indent=1))
        return 0

    for e in report:
        name = os.path.basename(os.path.normpath(e["dir"]))
        print(f"== {name}  (stop_id={e['stop_id']}, research={e['research_rows']},"
              f" unreadable={e['unreadable']})")
        print("  " + fmt_block("overall", e["overall"]))
        for k in sorted(e["arms"]):
            if len(e["arms"]) > 1:
                print("  " + fmt_block(k, e["arms"][k]))
    if len(report) >= 2:
        print("\n-- trend (first -> last) --")
        a, b = report[0]["overall"], report[-1]["overall"]
        if a.get("n") and b.get("n"):
            for key, tag in (("won", "won"), ("turns_mean", "turns"),
                             ("quick_submit_rate", "quick<=2"),
                             ("len_mean", "len")):
                print(f"  {tag:9s} {a[key]:8.2f} -> {b[key]:8.2f}"
                      f"  ({b[key] - a[key]:+.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
