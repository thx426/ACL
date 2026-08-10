"""Build train/valid parquet for the WebArena domain from WebArena-Verified.

Data ruling (2026-08-10, webarena-env-proposal §9): training AND testing use
the WebArena-Verified corpus exclusively (812 human-verified tasks, typed
deterministic evaluators; ServiceNow, Apache-2.0, package pinned 1.2.3).

Split construction
------------------
* **valid = the 165 WebArena-Lite test tasks**, identified by their ORIGINAL
  task ids. Provenance: WebServ's ``rl_web_agent/dataset/test_webarena_lite``
  configs carry ``old_task_id`` (the Lite→original mapping); the sorted list
  is frozen below as ``LITE_165`` with a sha256 guard. Community-comparable
  task set, verified scoring semantics.
* **train = the remaining 647 tasks** (812 ∖ 165) — strictly disjoint.

为什么不用 WebServ 自带划分:实测其 ``train_webarena``(647)是模板重实例化
**新造**任务(全部无 ``old_task_id``,内容对拍仅 74/165 命中),与原 812 不
同名空间,且无 Verified 评测配置——不可判分;其重编号 task_id 还与 lite 重
叠(假交集 123)。故划分只认原 id 空间,train 求补集(数量恰巧同为 647)。

Row schema: task_id(int) / sites(list) / start_urls(list, placeholder form) /
intent / intent_template_id / expected_task_type / user_prompt(=intent 逐字,
包装归 scaffold/首轮拼接) / n_eval / has_network_eval。

Usage:
  python -m trace_opd.envs.webarena.build_dataset --out <dir> [--source <json>]
(source 缺省 = 已安装 webarena-verified 包内的 assets 数据集)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

#: WebArena-Lite 165 test tasks, ORIGINAL task-id space (provenance above).
LITE_165 = [
    4, 7, 15, 20, 23, 27, 33, 37, 43, 44, 48, 56, 58, 65, 69, 71, 75, 77,
    82, 88, 93, 95, 96, 97, 98, 103, 109, 115, 117, 118, 123, 125, 127, 131,
    135, 139, 144, 149, 155, 156, 157, 162, 167, 169, 173, 182, 190, 196,
    202, 205, 211, 215, 220, 221, 225, 227, 235, 236, 240, 247, 250, 254,
    258, 259, 268, 270, 276, 283, 285, 287, 288, 296, 300, 311, 313, 318,
    321, 324, 333, 335, 348, 349, 354, 357, 361, 367, 368, 369, 374, 376,
    381, 382, 383, 384, 386, 387, 392, 401, 404, 415, 419, 423, 426, 431,
    440, 448, 454, 458, 464, 466, 470, 476, 485, 488, 491, 497, 505, 506,
    509, 514, 516, 521, 524, 528, 534, 538, 548, 566, 567, 574, 577, 582,
    599, 601, 605, 612, 619, 626, 631, 641, 645, 652, 657, 668, 673, 678,
    682, 686, 693, 704, 710, 714, 720, 729, 733, 741, 745, 748, 760, 762,
    768, 791, 798, 809, 811,
]
_LITE_SHA = "f86fc0fd28dbd25b4c3e15ab77d35c4405821bc12ff902fe6805da65f133e524"

_EXPECTED_TOTAL = 812
_EXPECTED_TRAIN = 647
_EXPECTED_VALID = 165


def _load_source(source: str | None) -> tuple[list[dict], str]:
    if source:
        path = Path(source)
    else:
        from .evalbridge import verified_dataset_path

        path = verified_dataset_path()
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _row(task: dict) -> dict:
    agent_evals = [e for e in task["eval"]
                   if e.get("evaluator") == "AgentResponseEvaluator"]
    if len(agent_evals) != 1:
        raise AssertionError(
            f"task {task['task_id']}: expected exactly 1 AgentResponseEvaluator"
            f", got {len(agent_evals)} (dataset drift?)"
        )
    expected_type = str(agent_evals[0]["expected"]["task_type"]).upper()
    if expected_type not in ("RETRIEVE", "MUTATE", "NAVIGATE"):
        raise AssertionError(
            f"task {task['task_id']}: unknown expected task_type "
            f"{expected_type!r}"
        )
    intent = str(task["intent"])
    if not intent.strip():
        raise AssertionError(f"task {task['task_id']}: empty intent")
    return {
        "task_id": int(task["task_id"]),
        "sites": list(task["sites"]),
        "start_urls": list(task["start_urls"]),
        "intent": intent,
        "intent_template_id": int(task["intent_template_id"]),
        "expected_task_type": expected_type,
        "user_prompt": intent,          # 逐字;包装归 scaffold/首轮拼接
        "n_eval": len(task["eval"]),
        "has_network_eval": any(
            e.get("evaluator") == "NetworkEventEvaluator" for e in task["eval"]
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="", help="verified json(缺省=包内资产)")
    ap.add_argument("--out", required=True, help="输出目录(dataset/)")
    args = ap.parse_args()

    import pandas as pd

    lite_sha = hashlib.sha256(
        ",".join(map(str, LITE_165)).encode()).hexdigest()
    assert lite_sha == _LITE_SHA, "LITE_165 constant corrupted"
    lite = set(LITE_165)
    assert len(lite) == _EXPECTED_VALID

    tasks, source_sha = _load_source(args.source or None)
    assert len(tasks) == _EXPECTED_TOTAL, (
        f"verified dataset has {len(tasks)} tasks, expected {_EXPECTED_TOTAL}"
    )
    ids = {int(t["task_id"]) for t in tasks}
    assert len(ids) == _EXPECTED_TOTAL, "duplicate task_id in source"
    missing_lite = lite - ids
    assert not missing_lite, f"lite ids missing from verified: {missing_lite}"

    rows = [_row(t) for t in tasks]
    train = [r for r in rows if r["task_id"] not in lite]
    valid = [r for r in rows if r["task_id"] in lite]
    assert len(train) == _EXPECTED_TRAIN and len(valid) == _EXPECTED_VALID
    assert not ({r["task_id"] for r in train} & {r["task_id"] for r in valid})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df_train = pd.DataFrame(sorted(train, key=lambda r: r["task_id"]))
    df_valid = pd.DataFrame(sorted(valid, key=lambda r: r["task_id"]))
    df_train.to_parquet(out / "train.parquet", index=False)
    df_valid.to_parquet(out / "valid.parquet", index=False)

    def dist(rs: list[dict]) -> dict:
        return {
            "sites": dict(Counter(s for r in rs for s in r["sites"])),
            "expected_task_type": dict(
                Counter(r["expected_task_type"] for r in rs)),
            "multi_site": sum(1 for r in rs if len(r["sites"]) > 1),
            "with_network_eval": sum(1 for r in rs if r["has_network_eval"]),
        }

    manifest = {
        "source_sha256": source_sha,
        "webarena_verified_pinned": "1.2.3",
        "lite165_sha256": _LITE_SHA,
        "split_rule": "valid = WebArena-Lite 165 (original task ids via "
                      "old_task_id provenance); train = the other 647; "
                      "strictly disjoint",
        "counts": {"train": len(train), "valid": len(valid)},
        "train": dist(train),
        "valid": dist(valid),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[build] train={len(train)} valid={len(valid)} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
