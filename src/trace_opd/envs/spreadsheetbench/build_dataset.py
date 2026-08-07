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


def split_seed42(items: list[dict], n_train: int = 200, n_test: int = 200):
    """union → sort(id) → shuffle(seed42) → 切分(旧 manifest 同法)。"""
    pool = sorted(items, key=lambda it: str(it["id"]))
    random.Random(42).shuffle(pool)
    if len(pool) < n_train + n_test:
        raise SystemExit(
            f"FATAL: need {n_train + n_test} tasks, got {len(pool)}"
        )
    return pool[:n_train], pool[n_train:n_train + n_test]


def make_row(item: dict, split: str, data_root: str) -> dict:
    answer_position = str(item.get("answer_position", ""))
    answer_sheet = str(item.get("answer_sheet", "") or "")
    if answer_position and answer_sheet and "!" not in answer_position:
        answer_position = f"{answer_sheet}!{answer_position}"   # 旧实现同法合成
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
        "data_source": "spreadsheetbench" if split == "train" else f"spreadsheetbench/{split}",
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
    args = ap.parse_args()

    items = load_items(args.items)
    print(f"[ssb] union: {len(items)} tasks")
    train, test = split_seed42(items)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, part in (("train", train), ("valid", test)):
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
        "method": "union -> sort(id) -> random.Random(42).shuffle -> (200, 200)",
        "counts": {"train": len(train), "valid": len(test)},
    }
    (out / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
