#!/usr/bin/env python3
"""阶段D:逐 SFT ckpt 跑 M0 探针并按用户判据选起点 ckpt。

用户裁定(2026-08-07):按 **researcher 过程的工具调用能力** 与 **最终文档有效率**
筛选;**文档产出率相当时取更靠前的 ckpt**(更少 SFT = 更少能力损伤/过拟合,
保留更多 RL 可塑性)。

判据分三层(全部来自真实复采,不看训练 loss):
  ① 协议层 tool_call_ok  —— assistant 轮里合法工具调用占比。学不会协议 ⇒
     飞轮点不着,一票否决(阈值 tool_ok_min)。
  ② 产出层 finalize_rate —— 会话产出非空 doc 的比例。v4 的 r_i 对 no_finalize
     直接判 0,故这是 reward 能否非恒零的前置。
  ③ 形态层 exp/turns 分布 —— 塌缩成一两轮 = 只学会了"立刻 finalize"的捷径,
     doc 无实验支撑;与 35B 参照对比。

"相当"用非劣性判据而非直接比点估计:样本量小(默认 10 任务 × 2 会话 = 20)时
点估计噪声大,比大小会被噪声牵着走。规则:候选 = finalize 的 Wilson **上界** ≥
(最优点估计 − tol),即"数据无法把它与最优区分开";在候选中取 step 最小者。
(注:不能用下界比点估计——n=20 时下界系统性低于点估计约 0.16,超过常用容差,
会让候选集恒空,连最优自己都选不上。)

用法:
  python scripts/probe_sft_ckpts.py plan   --ckpt-root <dir>            列出待探 ckpt
  python scripts/probe_sft_ckpts.py select --probe-root <dir> [--ref <35B report.json>]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 得分区间 (下界, 上界)。n 小时远比正态近似稳。"""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - margin), min(1.0, centre + margin)


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    return _wilson(k, n, z)[0]


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    return _wilson(k, n, z)[1]


def _tool_call_stats(messages: list[dict], protocol: str) -> tuple[int, int]:
    """(合法工具调用轮数, assistant 总轮数)。

    text 协议下合法调用 = 出现 <tool_call>{...}</tool_call> 且 JSON 可解析且带
    name 字段。最后一轮 finalize 也算工具调用。
    """
    ok = tot = 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        tot += 1
        content = m.get("content") or ""
        if protocol == "fncall":
            if m.get("tool_calls"):
                ok += 1
            continue
        for blob in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.S):
            try:
                call = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if isinstance(call, dict) and call.get("name"):
                ok += 1
                break
    return ok, tot


def scan_probe_dir(d: Path, protocol: str = "text") -> dict:
    """一个 ckpt 的探针会话目录 → 三层读数。"""
    sessions = sorted(d.glob("*/s*.json"))
    n = fin = err = 0
    tool_ok = tool_tot = 0
    exps: list[int] = []
    turns: list[int] = []
    doc_words: list[int] = []
    for path in sessions:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        n += 1
        if payload.get("error"):
            err += 1
        doc = payload.get("doc") or ""
        if payload.get("finalized") and doc.strip():
            fin += 1
            doc_words.append(len(doc.split()))
        a, b = _tool_call_stats(payload.get("messages") or [], protocol)
        tool_ok += a
        tool_tot += b
        exps.append(int(payload.get("experiments_used") or 0))
        turns.append(int(payload.get("turns_used") or 0))

    def med(xs):
        return sorted(xs)[len(xs) // 2] if xs else 0

    return {
        "n": n,
        "finalize": fin,
        "finalize_rate": round(fin / n, 4) if n else 0.0,
        "finalize_lower": round(wilson_lower(fin, n), 4),
        "finalize_upper": round(wilson_upper(fin, n), 4),
        "error": err,
        "tool_call_ok": tool_ok,
        "tool_call_turns": tool_tot,
        "tool_ok_rate": round(tool_ok / tool_tot, 4) if tool_tot else 0.0,
        "tool_ok_lower": round(wilson_lower(tool_ok, tool_tot), 4),
        "exp_median": med(exps),
        "exp_mean": round(sum(exps) / n, 2) if n else 0.0,
        "turns_median": med(turns),
        "doc_words_median": med(doc_words),
    }


def _step_of(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else -1


def cmd_plan(args) -> int:
    root = Path(args.ckpt_root)
    ckpts = sorted(
        (p for p in root.glob("global_step_*") if p.is_dir()),
        key=lambda p: _step_of(p.name),
    )
    rows = []
    for c in ckpts:
        hf = c / "huggingface"
        hf = hf if hf.is_dir() else (c / "hf")
        rows.append({"step": _step_of(c.name), "dir": str(c),
                     "hf": str(hf) if hf.is_dir() else "",
                     "hf_ready": hf.is_dir()})
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    ready = sum(1 for r in rows if r["hf_ready"])
    print(f"# {len(rows)} ckpt, {ready} 个已有 HF 目录", file=sys.stderr)
    return 0


def cmd_select(args) -> int:
    root = Path(args.probe_root)
    dirs = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.startswith("step_")),
        key=lambda p: _step_of(p.name),
    )
    if not dirs:
        print("FATAL: 探针目录为空", file=sys.stderr)
        return 1
    stats = []
    for d in dirs:
        s = scan_probe_dir(d, args.protocol)
        s["step"] = _step_of(d.name)
        s["dir"] = str(d)
        stats.append(s)

    ref = {}
    if args.ref and Path(args.ref).is_file():
        ref = json.loads(Path(args.ref).read_text(encoding="utf-8"))

    print(f"{'step':>6} {'n':>4} {'finalize':>9} {'下界':>7} {'tool_ok':>8} {'下界':>7} "
          f"{'上界':>7} {'exp中位':>8} {'turns':>6} {'doc词':>7} {'err':>4}")
    for s in stats:
        print(f"{s['step']:>6} {s['n']:>4} {s['finalize_rate']:>9.3f} {s['finalize_lower']:>7.3f} "
              f"{s['tool_ok_rate']:>8.3f} {s['tool_ok_lower']:>7.3f} "
              f"{s['finalize_upper']:>7.3f} "
              f"{s['exp_median']:>8} {s['turns_median']:>6} {s['doc_words_median']:>7} {s['error']:>4}")
    if ref:
        print(f"{'35B参照':>6} {ref.get('total', 0):>4} "
              f"{ref.get('kept', 0) / max(ref.get('total', 1), 1):>9.3f} {'—':>7} {'—':>8} {'—':>7} "
              f"{ref.get('exp_median', '—'):>8} {ref.get('turns_median', '—'):>6} "
              f"{ref.get('doc_words_median', '—'):>7}")

    # ① 协议层硬门
    eligible = [s for s in stats if s["tool_ok_lower"] >= args.tool_ok_min]
    if not eligible:
        best = max(stats, key=lambda s: s["tool_ok_lower"])
        print(f"\nNO-GO: 无 ckpt 的 tool_ok 下界 ≥ {args.tool_ok_min}"
              f"(最好 step {best['step']} = {best['tool_ok_lower']:.3f}) —— 协议没学会,"
              f"飞轮点不着。按 GO-V4 指导应停并反馈,勿带病进 v4。")
        return 2
    # ② 产出层硬门
    eligible = [s for s in eligible if s["finalize_lower"] >= args.finalize_min]
    if not eligible:
        best = max(stats, key=lambda s: s["finalize_lower"])
        print(f"\nNO-GO: 无 ckpt 的 finalize 下界 ≥ {args.finalize_min}"
              f"(最好 step {best['step']} = {best['finalize_lower']:.3f})")
        return 2
    # ③ "相当则取靠前":非劣性判据 —— 候选的 Wilson **上界** ≥ 最优点估计 − tol,
    #    即"数据无法把它和最优区分开"。不能拿下界比点估计:n=20 时下界系统性
    #    低于点估计约 0.16(> 常用容差),那样候选集恒空,最优自己都选不上。
    top = max(s["finalize_rate"] for s in eligible)
    close = [s for s in eligible if s["finalize_upper"] >= top - args.tol]
    pick = min(close, key=lambda s: s["step"])
    print(f"\nGO: 选 step {pick['step']}")
    print(f"  finalize {pick['finalize_rate']:.3f}"
          f"(95% CI {pick['finalize_lower']:.3f}-{pick['finalize_upper']:.3f})"
          f"  tool_ok {pick['tool_ok_rate']:.3f}(下界 {pick['tool_ok_lower']:.3f})")
    print(f"  最优点估计 {top:.3f};非劣容差 {args.tol};"
          f"候选 {[s['step'] for s in close]} → 取最靠前(更少 SFT,保留 RL 可塑性)")
    print(f"  ckpt dir: {pick['dir']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("plan")
    pp.add_argument("--ckpt-root", required=True)
    ps = sub.add_parser("select")
    ps.add_argument("--probe-root", required=True)
    ps.add_argument("--ref", default="")
    ps.add_argument("--protocol", default="text")
    ps.add_argument("--tool-ok-min", type=float, default=0.80)
    ps.add_argument("--finalize-min", type=float, default=0.50)
    ps.add_argument("--tol", type=float, default=0.10)
    args = ap.parse_args()
    return cmd_plan(args) if args.cmd == "plan" else cmd_select(args)


if __name__ == "__main__":
    sys.exit(main())
