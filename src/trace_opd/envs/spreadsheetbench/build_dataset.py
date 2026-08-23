"""SpreadsheetBench dataset builder — verified 400 → verl parquet.

用户裁定的划分:union(全部 items)→ 按 id 排序 → random.Random(42).shuffle
→ 前 200 train / 后 200 test(test 即训练期 val 集)。与旧
skills_evolve seed42 manifest 同源同法,仅切分比例改为 200/200。

行契约与 envs/_template/build_dataset.py 相同(extra_info 自足;
agent_name 恒 trace_opd)。初始观察在此捕获(preview_workbook 渲染 +
initial_obs_of 组装),与 live episode 同源函数 ⇒ 逐字节一致是确定性
红线,故本脚本必须在与训练运行相同的环境(同 openpyxl 版本)执行。

运行(数据所在服务器上):
  python -m trace_opd.envs.spreadsheetbench.build_dataset \
    --items <dir-or-json ...> --data-root $SSB_DATA --out <dir>
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from ...core import protocol
from .adapter import SYSTEM_PROMPT, initial_obs_of
from .evaluator import find_test_cases, preview_workbook


def load_items(paths: list[str]) -> list[dict]:
    """吃 items.json 文件或含 items.json 的目录树;按 id 去重后并集。"""
    seen: dict[str, dict] = {}
    files: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            files.extend(sorted(pp.rglob("items.json")))
        else:
            files.append(pp)
    if not files:
        raise SystemExit(f"FATAL: no items.json found under {paths}")
    for f in files:
        for it in json.loads(f.read_text(encoding="utf-8")):
            seen[str(it["id"])] = it
    return list(seen.values())


UNRELIABLE_PATH = Path(__file__).with_name("unreliable_tasks.json")


def load_unreliable() -> tuple[set[str], dict]:
    """reward 恒零的任务集(判定过程与根因见同目录 unreliable_tasks.json)。

    这些任务的 golden 交给生产评分链路自比都拿不到 1.0 —— agent 逐格复现
    verified 答案 reward 仍是 0。留在 train 里,组内 rollout 全 0 ⇒ GRPO
    advantage 全零、该组无梯度、纯烧算力;留在 val 里则压低天花板。
    用户裁定 2026-08-09:全部剔除,对所有 SpreadsheetBench 实验生效。
    """
    doc = json.loads(UNRELIABLE_PATH.read_text(encoding="utf-8"))
    return {str(t["task_id"]) for t in doc["unreliable_tasks"]}, doc


def split_seed42(items: list[dict], n_train: int = 200, n_test: int = 200):
    """union → sort(id) → shuffle(seed42) → 切分(旧 manifest 同法)。

    注意:不可靠任务的剔除**发生在本函数之后**,不在 pool 里预先剔除 ——
    预先剔除会让 shuffle 的落位整体平移、划分全变;放在切分后剔除,得到的
    train/valid 是原 200/200 的严格子集,与已按全 400 起跑的对照臂仍保持
    子集关系,val 曲线可比。
    """
    pool = sorted(items, key=lambda it: str(it["id"]))
    random.Random(42).shuffle(pool)
    if len(pool) < n_train + n_test:
        raise SystemExit(
            f"FATAL: need {n_train + n_test} tasks, got {len(pool)}"
        )
    return pool[:n_train], pool[n_train:n_train + n_test]


def make_row(
    item: dict, split: str, data_root: str, source_name: str = "spreadsheetbench"
) -> dict:
    answer_position = str(item.get("answer_position", ""))
    answer_sheet = str(item.get("answer_sheet", "") or "")
    if answer_position and answer_sheet and "!" not in answer_position:
        # answer_sheet 可能是**多个 sheet**(逗号分隔),此时必须给每个 sheet 各配
        # 一份 range。旧实现直接 f"{answer_sheet}!{answer_position}" 会拼出
        # `A,B,C!A3:E11` —— evaluator 按逗号切段后,前几段没有 `!`,整段被当成
        # cell_range 送进 openpyxl ⇒ ValueError ⇒ _score_output 的 except 吞掉
        # ⇒ 该任务 reward 恒 0。2026-08-09 实测命中 60-7(四个 sheet)。
        sheets = [s.strip().strip("'\"") for s in answer_sheet.split(",") if s.strip()]
        if len(sheets) > 1:
            answer_position = ",".join(f"'{s}'!{answer_position}" for s in sheets)
        else:
            answer_position = f"{answer_sheet}!{answer_position}"
    task_dir = Path(data_root) / str(item["spreadsheet_path"])
    cases = find_test_cases(str(task_dir))
    if not cases:
        raise SystemExit(f"FATAL: no test cases for task {item['id']} at {task_dir}")
    extra_info = {
        "task_id": str(item["id"]),
        "instruction": str(item["instruction"]),
        "instruction_type": str(item.get("instruction_type", "")),
        "answer_position": answer_position,
        "spreadsheet_dir": str(item["spreadsheet_path"]),
        "split": split,
        "n_cases": len(cases),
        "initial_obs": initial_obs_of(
            {
                "instruction": str(item["instruction"]),
                "instruction_type": str(item.get("instruction_type", "")),
                "answer_position": answer_position,
            },
            preview_workbook(cases[0][1]),
        ),
    }
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": protocol.truncate_obs(extra_info["initial_obs"])},
        ],
        "data_source": source_name if split == "train" else f"{source_name}/{split}",
        "agent_name": "trace_opd",
        "reward_model": {"style": "env", "ground_truth": ""},
        "extra_info": extra_info,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", required=True,
                    help="items.json 文件或含它们的目录(旧 split 树可整目录传入)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-unreliable", action="store_true",
                    help="保留 reward 恒零的任务(仅用于复现旧数据集或重新审计;"
                         "训练侧一律不加 —— 用户裁定 2026-08-09。经典线终评"
                         "口径为 400 全集报数(用户裁定 2026-08-17):终评"
                         "parquet 用 --n-train 0 --n-test 400 --keep-unreliable,"
                         "分母对齐官方 leaderboard 与 SRL 论文)")
    ap.add_argument("--n-train", type=int, default=200,
                    help="划分 train 数(默认 200 = 旧行为;0 = 不产 train)")
    ap.add_argument("--n-test", type=int, default=200,
                    help="划分 valid 数(默认 200;400 = 全集单档)")
    args = ap.parse_args()

    items = load_items(args.items)
    print(f"[ssb] union: {len(items)} tasks")
    train, test = split_seed42(items, args.n_train, args.n_test)
    excluded_doc = None
    if args.keep_unreliable:
        print("[ssb] ⚠ --keep-unreliable:保留 reward 恒零任务,不得用于正式实验")
    else:
        bad, excluded_doc = load_unreliable()
        n0 = (len(train), len(test))
        train = [it for it in train if str(it["id"]) not in bad]
        test = [it for it in test if str(it["id"]) not in bad]
        print(f"[ssb] 剔除 reward 恒零任务 {len(bad)} 个 "
              f"(train {n0[0]}→{len(train)}, valid {n0[1]}→{len(test)})")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, part in (("train", train), ("valid", test)):
        if not part:
            print(f"[ssb] {split}: 0 tasks — skipped (empty split)")
            continue
        rows = [make_row(it, split, args.data_root) for it in part]
        fams: dict[str, int] = {}
        n_multi = 0
        for r in rows:
            fams[r["extra_info"]["instruction_type"]] = (
                fams.get(r["extra_info"]["instruction_type"], 0) + 1
            )
            if r["extra_info"]["n_cases"] > 1:
                n_multi += 1
        path = out / f"{split}.parquet"
        pd.DataFrame(rows).to_parquet(path)
        print(f"[ssb] {split}: {len(rows)} tasks -> {path}")
        print(f"[ssb]   instruction_type: {fams}; multi-case tasks: {n_multi}")
    manifest = {
        "source_items": args.items,
        "method": ("union -> sort(id) -> random.Random(42).shuffle -> "
                   f"({args.n_train}, {args.n_test})"
                   + ("" if args.keep_unreliable
                      else " -> 剔除 reward 恒零任务(切分后剔除,故为原划分子集)")),
        "counts": {"train": len(train), "valid": len(test)},
        "unreliable_excluded": (
            None if excluded_doc is None else {
                "n": len(excluded_doc["unreliable_tasks"]),
                "source": "envs/spreadsheetbench/unreliable_tasks.json",
                "决定": excluded_doc["决定"],
                "判定环境": excluded_doc["判定环境"],
                "task_ids": sorted(str(t["task_id"])
                                   for t in excluded_doc["unreliable_tasks"]),
            }),
    }
    (out / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
