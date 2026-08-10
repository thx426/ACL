#!/usr/bin/env python
"""SSB 数据集的 reward 信号正确性验证(六道门,全程走生产路径)。

"可靠"不等于"完美答案能得满分" —— 那只是信号的一半。一个对任何输入都给
满分的任务同样是坏信号:agent 什么都不做也拿分,组内 rollout 全 1 ⇒ advantage
同样全零、无梯度,而且会把 val 读数抬成假象。故正反两向都必须验。

全部评分经 `SpreadsheetbenchTaskAdapter._score_output` —— 训练时算 reward 走的
就是这个函数,不另写等价实现(等价实现会漏掉 adapter 内的 case 发现、并发门、
异常吞掉等真实行为)。

  门① 正向     pred = golden 原样(含公式)          期望 score == 1.0
               不满分 ⇒ 完美答案也拿不到分,reward 恒零
  门② 反向-未操作 pred = init 原样(agent 什么都没做)  期望 score < 1.0
               满分 ⇒ 初始状态即答案,任务是空的、白给
  门③ 反向-扰动  pred = golden 但答案区被改坏        期望 score < 1.0
               满分 ⇒ 比对没真比(区域解析错/区域为空)
  门④ 确定性    同一 pred 连评两次                   期望两次 score 相同
               不同 ⇒ 重算不确定,reward 是噪声
  门⑤ 答案区非空 answer_position 区域至少一个非空单元格
               全空 ⇒ "空 vs 空"恒真,门③ 也测不出来
  门⑥ 单调     score(完美) > score(未操作)          综合判据

用法:
  python scripts/ssb_verify_reward_signal.py --data $SSB_DATA \
      --dataset /path/to/ssb_reliable373_seed42          # 验证过滤后的数据集
  ... --sample 30      只验前 30 个(冒烟)
  ... --det-sample 40  门④ 的抽样数(默认 40;全量太贵)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openpyxl
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from trace_opd.common.config import SpreadsheetbenchCfg  # noqa: E402
from trace_opd.envs.spreadsheetbench.adapter import (  # noqa: E402
    SpreadsheetbenchTaskAdapter)
from trace_opd.envs.spreadsheetbench.evaluator import find_test_cases  # noqa: E402


def answer_cells(wb, ap: str):
    """answer_position → 单元格列表。

    解析必须与生产 `evaluator.compare_workbooks` **逐行同源**:按逗号切多区域、
    每段有 `!` 才拆 sheet、两侧都 strip 引号。自己另写一套(如按 rpartition("!")
    取最后一段)会在多区域 `D1,D5,D9`、`'A2:M70`(引号不配对)上解析失败,于是
    验证脚本跳过任务、却给出"全 PASS"的假象 —— 2026-08-09 实测 8 个任务因此
    被跳过,而它们在生产路径下是能正常评分的。
    """
    out = []
    for scr in (ap or "").split(","):
        scr = scr.strip()
        if not scr:
            continue
        if "!" in scr:
            sheet_name, cell_range = scr.split("!", 1)
            sheet_name = sheet_name.strip().strip("'\"")
        else:
            sheet_name, cell_range = wb.sheetnames[0], scr
        cell_range = cell_range.strip().strip("'\"")
        if sheet_name not in wb.sheetnames:
            continue
        ws = wb[sheet_name]
        try:
            region = ws[cell_range]
        except (ValueError, KeyError, TypeError):
            continue          # 与生产同:该段解析不了就跳过该段,不炸整个任务
        if not isinstance(region, tuple):
            out.append(region)
            continue
        for row in region:
            out.extend(row if isinstance(row, tuple) else [row])
    return out


def perturb(golden_path: str, ap: str) -> bytes:
    """把答案区每个单元格改成不可能的值 —— 模拟"答错了"。

    改后 workbook 里答案区不再有公式,故生产路径不会对它重算(有公式才重算),
    直接拿静态错值与 golden 比 ⇒ 正确的评分链路必须判 0。
    """
    wb = openpyxl.load_workbook(golden_path, data_only=False)
    for c in answer_cells(wb, ap):
        try:
            c.value = "__PERTURBED__"
        except (AttributeError, ValueError):
            pass
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def nonempty_answer_cells(golden_path: str, ap: str) -> int:
    wb = openpyxl.load_workbook(golden_path, data_only=True)
    return sum(1 for c in answer_cells(wb, ap) if c.value not in (None, ""))


def score_bytes(adapter, data: bytes) -> float:
    """走生产 _score_output。"""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as fh:
        fh.write(data)
        p = fh.name
    try:
        return float(adapter._score_output(p))  # noqa: SLF001 — 刻意验生产路径
    finally:
        try:
            os.unlink(p)
        except OSError:
            pass


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--data", required=True, help="xlsx 根($SSB_DATA)")
    ap_.add_argument("--dataset", required=True, help="待验 parquet 目录")
    ap_.add_argument("--splits", nargs="+", default=["train", "valid"])
    ap_.add_argument("--sample", type=int, default=0)
    ap_.add_argument("--det-sample", type=int, default=40)
    ap_.add_argument("--workers", type=int, default=16,
                     help="并行度。每次评分要冷启动一个 soffice(独立 profile,"
                          "线程安全),串行时 80 核只跑满 1 个 —— 373 任务串行要 "
                          "50+ 分钟,16 路并行 3-5 分钟。机器上有训练在跑时别拉满,"
                          "CPU 饥饿会饿死推理引擎。")
    ap_.add_argument("--out", default="/tmp/ssb_signal_verify.json")
    ap_.add_argument("--verify-list", action="store_true",
                     help="门禁模式:只复核 unreliable_tasks.json 里的任务**仍然**"
                          "不可靠(每个至少命中 ①/②/③ 之一)。清单由六门定,复核也"
                          "必须用六门 —— 只验正向的脚本会把反向剔除的任务(不做也"
                          "满分)误判成'已变可靠'。")
    args = ap_.parse_args()

    cfg = SpreadsheetbenchCfg(data_root=args.data)
    gate = threading.Semaphore(max(1, args.workers))  # recalc 的 soffice
    # 各用独立 UserInstallation profile(recalc.py),故线程安全

    # item 直接取 parquet 的 extra_info —— adapter 的自足契约就是它,
    # 拿 dataset.json 的原始 item 冒充会缺 task_id/spreadsheet_dir 而全量跳过
    # (2026-08-09 实测:12/12 跳过而六道门仍显示 PASS,典型假阳性)。
    todo = []
    for split in args.splits:
        df = pd.read_parquet(Path(args.dataset) / f"{split}.parquet")
        for x in df["extra_info"]:
            v = dict(x) if isinstance(x, dict) else json.loads(x)
            todo.append((split, v))
    if args.verify_list:
        from trace_opd.envs.spreadsheetbench.build_dataset import (  # noqa: PLC0415
            load_unreliable)
        want, _ = load_unreliable()
        todo = [(sp, v) for sp, v in todo if str(v.get("task_id")) in want]
        if len(todo) != len(want):
            missing = want - {str(v.get("task_id")) for _sp, v in todo}
            print(f"STALE 清单里有 {len(missing)} 个任务不在该数据集中: "
                  f"{sorted(missing)[:6]}(--verify-list 需喂全量未剔除底本)")
            return 1
    if args.sample:
        todo = todo[:args.sample]
    print(f"[verify] 数据集 {args.dataset} · 任务 {len(todo)} 个")

    fail = {k: [] for k in ("g1_perfect", "g2_noop", "g3_perturb",
                            "g4_determinism", "g5_empty_answer", "g6_monotonic")}
    skipped: list = []
    lock = threading.Lock()
    tested = 0
    done = 0

    def run_one(idx: int, it: dict):
        """单任务六门。返回 (tid, 违例列表, 是否计入 tested)。"""
        tid = str(it.get("task_id", "?"))
        hits: list[tuple[str, tuple]] = []
        ad = SpreadsheetbenchTaskAdapter(it, cfg, gate)
        cases = find_test_cases(ad._task_dir())  # noqa: SLF001 — 与生产同源
        if not cases:
            raise RuntimeError("无 case")
        init_p, gold_p = cases[0][1], cases[0][2]
        if nonempty_answer_cells(gold_p, it["answer_position"]) == 0:
            hits.append(("g5_empty_answer", (tid,)))
        s_perfect = score_bytes(ad, Path(gold_p).read_bytes())
        if s_perfect != 1.0:
            hits.append(("g1_perfect", (tid, s_perfect)))
        s_noop = score_bytes(ad, Path(init_p).read_bytes())
        if s_noop >= 1.0:
            hits.append(("g2_noop", (tid, s_noop)))
        s_pert = score_bytes(ad, perturb(gold_p, it["answer_position"]))
        if s_pert >= 1.0:
            hits.append(("g3_perturb", (tid, s_pert)))
        if not (s_perfect > s_noop):
            hits.append(("g6_monotonic", (tid, s_perfect, s_noop)))
        if idx < args.det_sample:
            s2 = score_bytes(ad, Path(gold_p).read_bytes())
            if s2 != s_perfect:
                hits.append(("g4_determinism", (tid, s_perfect, s2)))
        return tid, hits

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, i, it): (i, it)
                for i, (_split, it) in enumerate(todo)}
        for fu in as_completed(futs):
            i, it = futs[fu]
            tid = str(it.get("task_id", "?"))
            with lock:
                done += 1
                try:
                    _tid, hits = fu.result()
                    for key, payload in hits:
                        fail[key].append(payload if len(payload) > 1 else payload[0])
                    tested += 1
                except Exception as exc:  # noqa: BLE001
                    skipped.append((tid, f"{type(exc).__name__}: {str(exc)[:70]}"))
                if done % 50 == 0:
                    print(f"  …{done}/{len(todo)} "
                          f"g1={len(fail['g1_perfect'])} g2={len(fail['g2_noop'])} "
                          f"g3={len(fail['g3_perturb'])} g6={len(fail['g6_monotonic'])}",
                          flush=True)

    n = len(todo)
    if args.verify_list:
        # 每个清单任务必须**仍然**至少命中 ①/②/③ 之一。全过 = 它已变可靠
        # (换了 LibreOffice、修了解析 bug…),清单过期须重审,否则白白少训。
        hit = set()
        for k in ("g1_perfect", "g2_noop", "g3_perturb"):
            for x in fail[k]:
                hit.add(str(x[0]) if isinstance(x, (list, tuple)) else str(x))
        regained = sorted({str(v.get("task_id")) for _sp, v in todo} - hit)
        if skipped:
            print(f"STALE 复核中 {len(skipped)} 个任务评分异常: {skipped[:4]}")
            return 1
        if regained:
            print(f"STALE {len(regained)} 个已变可靠,清单需重审: {regained[:10]}")
            return 1
        print(f"OK 清单 {len(hit)}/{n} 仍全部不可靠(六门口径)")
        return 0
    print(f"\n{'='*70}\nreward 信号正确性验证 · 计划 {n} 个 / **实测 {tested} 个**\n{'='*70}")
    if tested == 0:
        print("  [FAIL] 有效测试数为 0 —— 下面的门一律不成立(假阳性防护)")
    names = {
        "g1_perfect":      "① 正向:完美答案 score==1.0",
        "g2_noop":         "② 反向:未操作 init 不得满分",
        "g3_perturb":      "③ 反向:答案区改坏不得满分",
        "g4_determinism":  f"④ 确定性:重复评分一致(抽样 {min(args.det_sample, n)})",
        "g5_empty_answer": "⑤ answer 区非空",
        "g6_monotonic":    "⑥ 单调:完美 > 未操作",
    }
    allok = True
    for k, label in names.items():
        bad = fail[k]
        mark = "PASS" if (not bad and tested > 0) else "FAIL"
        if bad or tested == 0:
            allok = False
        den = min(args.det_sample, tested) if k == "g4_determinism" else tested
        print(f"  [{mark}] {label}  违例 {len(bad)}/{den}")
        if bad:
            print(f"         {bad[:8]}")
    print(f"\n  跳过/异常 {len(skipped)}: {skipped[:6]}")
    print(f"\n=> {'全部通过 —— 该数据集的 reward 信号正反两向均正确' if allok and not skipped else '存在违例,须处置'}")
    Path(args.out).write_text(json.dumps(
        {"planned": n, "tested": tested,
         "fail": {k: v for k, v in fail.items()}, "skipped": skipped},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"明细 -> {args.out}")
    return 0 if (allok and not skipped) else 1


if __name__ == "__main__":
    raise SystemExit(main())
