#!/usr/bin/env python
"""SpreadsheetBench 任务的 reward 可靠性审计。

判据(唯一):把 golden 原样当作 agent 交出的**完美答案**(含公式,即"允许写
公式"的真实训练情形),喂进与生产完全相同的评分链路 ——

    find_test_cases → recalculated_for_grading(有公式才重算) → evaluate

全部 case 都 ok 才算这个任务能给出可靠 reward。达不到的任务,agent 逐格复现
verified 答案 reward 仍是 0:留在 train 里组内 rollout 全 0 ⇒ GRPO advantage
全零、无梯度、纯烧算力;留在 val 里压低天花板。

为什么会不可靠(2026-08-09 实测,逐个打开 golden 读公式得到):
  · golden 用了 Excel 365 新函数(_xlfn. 前缀:XLOOKUP/UNIQUE/LET/TEXTJOIN/
    AGGREGATE/TEXTSPLIT/FILTERXML/…),LibreOffice 不认识 ⇒ 重算成 #NAME?
  · 数组公式 / 结构化引用(Table1[#This Row])算不出 ⇒ #VALUE!/#N/A/None
  · 少数纯数值算错(如 54144: 37.46 → 17.46)
  · 我们自己的 answer_position 解析缺陷(sheet 名含逗号被当多区域切分)
  · 数据集文件名 typo(42930 目录下是 1_43930_*.xlsx)
LibreOffice 6.4.7.2 与 7.6.7.2 实测逐任务同结果、失败集合完全一致 ⇒ 与版本
无关,是 LibreOffice 与 Excel 的语义鸿沟,换版本无解。

用法(必须在**训练同环境**跑 —— openpyxl 版本影响渲染与比对):
    python scripts/ssb_audit_reward_reliability.py --data $SSB_DATA            # 全量重审
    python scripts/ssb_audit_reward_reliability.py --data $SSB_DATA --verify   # 抽检现有清单
    python scripts/ssb_audit_reward_reliability.py --data $SSB_DATA --write    # 重写清单
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from trace_opd.envs.spreadsheetbench.evaluator import (  # noqa: E402
    evaluate, find_test_cases)
from trace_opd.envs.spreadsheetbench.recalc import (  # noqa: E402
    find_soffice, recalculated_for_grading)

LIST_PATH = (_HERE.parent / "src" / "trace_opd" / "envs" / "spreadsheetbench"
             / "unreliable_tasks.json")


def probe_formulas(path: str) -> tuple[list[str], bool]:
    """从 xlsx 的 sheet XML 直接读:(Excel 新函数名, 是否含数组公式)。"""
    try:
        with zipfile.ZipFile(path) as z:
            xml = "".join(
                z.read(n).decode("utf-8", "ignore") for n in z.namelist()
                if n.startswith("xl/worksheets/") and n.endswith(".xml"))
    except (zipfile.BadZipFile, OSError):
        return [], False
    fns = sorted({f for f in re.findall(r"_xlfn\.([A-Z_0-9.]+)", xml) if f != "_"})
    return fns, 't="array"' in xml


def classify(reason: str, fns: list[str], has_array: bool) -> str:
    if "find_test_cases" in reason:
        return "dataset_typo"
    if "not a valid coordinate" in reason or "invalid literal for int" in reason:
        return "our_parser"
    if fns:
        return "libreoffice_missing_function"
    if has_array:
        return "libreoffice_array_formula"
    return "libreoffice_semantics"


def audit_one(item: dict, data_root: str, soffice: str | None,
              timeout_s: float) -> tuple[bool, str, list[str], bool]:
    tid = str(item["id"])
    task_dir = os.path.join(data_root, str(item["spreadsheet_path"]).lstrip("./"))
    cases = find_test_cases(task_dir)
    if not cases:
        return False, "find_test_cases 返回空(文件命名与 id 不符?)", [], False
    fns, arr = probe_formulas(cases[0][2])
    ap = str(item.get("answer_position", ""))
    sheet = str(item.get("answer_sheet", "") or "")
    if ap and sheet and "!" not in ap:
        ap = f"{sheet}!{ap}"           # 与 build_dataset.make_row 同法合成
    for _no, _ip, gold in cases:
        tmp = None
        try:
            graded = recalculated_for_grading(
                Path(gold).read_bytes(), soffice=soffice,
                timeout_s=timeout_s, task_id=tid)
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as fh:
                fh.write(graded)
                tmp = fh.name
            verdict = evaluate(tmp, gold, ap)
            if not verdict["ok"]:
                return False, str(verdict["reason"])[:120], fns, arr
        except Exception as exc:  # noqa: BLE001 — 审计要吃下一切并如实记录
            return False, f"{type(exc).__name__}: {exc}"[:120], fns, arr
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
    return True, "", fns, arr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="SSB 数据根($SSB_DATA)")
    ap.add_argument("--verify", action="store_true",
                    help="抽检模式:只复核现有清单里的任务仍然不可靠(启动门禁用)")
    ap.add_argument("--sample", type=int, default=0, help="只审前 N 个(冒烟)")
    ap.add_argument("--write", action="store_true", help="把结果写回清单文件")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    ds_path = Path(args.data) / "dataset.json"
    if not ds_path.is_file():
        print(f"FATAL: 找不到 {ds_path}", file=sys.stderr)
        return 2
    items = json.loads(ds_path.read_text(encoding="utf-8"))
    soffice = find_soffice()
    ver = ""
    if soffice:
        try:
            ver = subprocess.run([soffice, "--version"], capture_output=True,
                                 text=True, timeout=60).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            ver = "<--version 失败>"

    known: set[str] = set()
    if LIST_PATH.is_file():
        known = {str(t["task_id"]) for t in
                 json.loads(LIST_PATH.read_text(encoding="utf-8"))["unreliable_tasks"]}

    if args.verify:
        # 只复核已知不可靠的任务:它们若变可靠了,说明环境/代码变了,清单须重审。
        items = [it for it in items if str(it["id"]) in known]
        if not args.quiet:
            print(f"[verify] 复核清单中的 {len(items)} 个任务")
    elif args.sample:
        items = items[:args.sample]

    if not args.quiet:
        print(f"[audit] soffice={soffice} | {ver}")
        print(f"[audit] 任务 {len(items)} 个,数据根 {args.data}")

    ok_ids: list[str] = []
    bad: list[dict] = []
    for i, item in enumerate(items):
        good, reason, fns, arr = audit_one(item, args.data, soffice, args.timeout)
        tid = str(item["id"])
        if good:
            ok_ids.append(tid)
        else:
            bad.append({"task_id": tid, "bucket": classify(reason, fns, arr),
                        "excel_functions": fns, "failure": reason})
        if not args.quiet and (i + 1) % 50 == 0:
            print(f"  …{i+1}/{len(items)} 可靠 {len(ok_ids)} 不可靠 {len(bad)}",
                  flush=True)

    if args.verify:
        # 门禁语义:清单里的任务必须**仍然**不可靠。有任何一个变可靠 ⇒ 清单过期。
        regained = ok_ids
        if regained:
            print(f"STALE {len(regained)} 个已变可靠,清单需重审: {sorted(regained)[:10]}")
            return 1
        print(f"OK 清单 {len(bad)}/{len(items)} 仍全部不可靠")
        return 0

    n = len(items)
    print(f"\n{'='*66}")
    print(f"reward 可靠性审计:可靠 {len(ok_ids)}/{n} = {len(ok_ids)/max(n,1):.1%}"
          f" | 不可靠 {len(bad)} = {len(bad)/max(n,1):.1%}")
    print(f"{'='*66}")
    for b in sorted(bad, key=lambda x: (x["bucket"], x["task_id"])):
        fn = ",".join(b["excel_functions"][:3])
        print(f"  {b['task_id']:>9}  {b['bucket']:<30} {fn:<22} {b['failure'][:46]}")
    print(f"\nbucket 分布: {dict(collections.Counter(b['bucket'] for b in bad))}")

    if known and not args.sample:
        now = {b["task_id"] for b in bad}
        if now != known:
            print(f"\n⚠ 与现有清单不一致:新增 {sorted(now - known)} | 消失 {sorted(known - now)}")
        else:
            print("\n与现有清单完全一致 ✓")

    if args.write:
        doc = json.loads(LIST_PATH.read_text(encoding="utf-8")) if LIST_PATH.is_file() else {}
        doc["统计"] = {"total": n, "reliable": len(ok_ids), "unreliable": len(bad)}
        doc["判定环境"] = {**doc.get("判定环境", {}), "libreoffice": ver,
                       "openpyxl": __import__("openpyxl").__version__}
        doc["unreliable_tasks"] = sorted(bad, key=lambda x: (x["bucket"], x["task_id"]))
        doc["reliable_task_ids"] = sorted(ok_ids)
        LIST_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                             encoding="utf-8")
        print(f"\n已写回 {LIST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
