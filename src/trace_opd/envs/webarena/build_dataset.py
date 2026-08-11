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

Row schema(仓库通用外层,审阅 P0-1)::

    prompt        [system=scaffold, user=intent]   ← 静态部分;live 首轮 user
                  是 intent+页面首观测(episode 级动态,见 adapter
                  validate_initial_obs),extra_info.prompt_is_partial 标记
    data_source   "webarena" / "webarena/valid"
    agent_name    "trace_opd"
    reward_model  {"style": "env", "ground_truth": ""}
    extra_info    工厂 item 全字段(task_id/sites/start_urls/user_prompt/…)

完整性门全部为显式异常(``python -O`` 不可关,审阅 P2-1);默认源钉死
verified 1.2.3 数据 JSON 的 sha256。

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
#: webarena-verified 1.2.3 包内 assets/dataset/webarena-verified.json 的
#: sha256(默认源硬门;--source 显式覆盖时只跳过此门,结构门照跑)。
_VERIFIED_SHA = "d65275660814663375028e9017e1f929e3c38321041b125795e2713b52243d30"

_EXPECTED_TOTAL = 812
_EXPECTED_TRAIN = 647
_EXPECTED_VALID = 165


class DatasetIntegrityError(RuntimeError):
    """任何完整性门失败(显式异常,-O 不可关)。"""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise DatasetIntegrityError(msg)


def _load_source(source: str | None) -> tuple[list[dict], str, bool]:
    if source:
        path, is_default = Path(source), False
    else:
        from .evalbridge import verified_dataset_path

        path, is_default = verified_dataset_path(), True
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest(), is_default


def _extra_info(task: dict, split: str) -> dict:
    agent_evals = [e for e in task["eval"]
                   if e.get("evaluator") == "AgentResponseEvaluator"]
    _require(len(agent_evals) == 1,
             f"task {task['task_id']}: expected exactly 1 "
             f"AgentResponseEvaluator, got {len(agent_evals)}")
    expected_type = str(agent_evals[0]["expected"]["task_type"]).upper()
    _require(expected_type in ("RETRIEVE", "MUTATE", "NAVIGATE"),
             f"task {task['task_id']}: unknown expected task_type "
             f"{expected_type!r}")
    intent = str(task["intent"])
    _require(bool(intent.strip()), f"task {task['task_id']}: empty intent")
    sites = [str(s) for s in task["sites"]]
    _require(bool(sites), f"task {task['task_id']}: no sites")
    return {
        "task_id": int(task["task_id"]),
        "task_type": sites[0],          # family 口径 = 主站点(adapter 同源)
        "split": split,
        "sites": sites,
        "start_urls": [str(u) for u in task["start_urls"]],
        "intent": intent,
        "user_prompt": intent,          # 逐字;包装归 scaffold/首轮拼接
        "intent_template_id": int(task["intent_template_id"]),
        "expected_task_type": expected_type,
        "n_eval": len(task["eval"]),
        "has_network_eval": any(
            e.get("evaluator") == "NetworkEventEvaluator" for e in task["eval"]
        ),
        "prompt_is_partial": True,      # live 首轮 user 会追加页面首观测
    }


def _row(task: dict, split: str) -> dict:
    from .adapter import SYSTEM_PROMPT

    info = _extra_info(task, split)
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": info["user_prompt"]},
        ],
        "data_source": "webarena" if split == "train" else "webarena/valid",
        "agent_name": "trace_opd",
        "reward_model": {"style": "env", "ground_truth": ""},
        "extra_info": info,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="", help="verified json(缺省=包内资产)")
    ap.add_argument("--out", required=True, help="输出目录(dataset/)")
    args = ap.parse_args()

    import pandas as pd

    lite_sha = hashlib.sha256(
        ",".join(map(str, LITE_165)).encode()).hexdigest()
    _require(lite_sha == _LITE_SHA, "LITE_165 constant corrupted")
    lite = set(LITE_165)
    _require(len(lite) == _EXPECTED_VALID, "LITE_165 has duplicates")

    tasks, source_sha, is_default = _load_source(args.source or None)
    if is_default:
        _require(source_sha == _VERIFIED_SHA,
                 f"default verified dataset sha mismatch: {source_sha} != "
                 f"{_VERIFIED_SHA} (package content drifted — re-pin "
                 "deliberately)")
    _require(len(tasks) == _EXPECTED_TOTAL,
             f"verified dataset has {len(tasks)} tasks, "
             f"expected {_EXPECTED_TOTAL}")
    ids = {int(t["task_id"]) for t in tasks}
    _require(ids == set(range(_EXPECTED_TOTAL)),
             "task ids are not exactly 0..811")

    train = [_row(t, "train") for t in tasks if int(t["task_id"]) not in lite]
    valid = [_row(t, "valid") for t in tasks if int(t["task_id"]) in lite]
    _require(len(train) == _EXPECTED_TRAIN and len(valid) == _EXPECTED_VALID,
             f"split sizes wrong: {len(train)}/{len(valid)}")
    tr_ids = {r["extra_info"]["task_id"] for r in train}
    va_ids = {r["extra_info"]["task_id"] for r in valid}
    _require(not (tr_ids & va_ids), "train/valid overlap")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    key = lambda r: r["extra_info"]["task_id"]  # noqa: E731
    df_train = pd.DataFrame(sorted(train, key=key))
    df_valid = pd.DataFrame(sorted(valid, key=key))
    # 先写临时名,双双复读通过后原子换名(避免半成品目录,审阅 P2-1)。
    for name, df in (("train", df_train), ("valid", df_valid)):
        tmp = out / f".{name}.parquet.tmp"
        df.to_parquet(tmp, index=False)
        back = pd.read_parquet(tmp)
        _require(len(back) == len(df) and "extra_info" in back.columns
                 and "prompt" in back.columns,
                 f"{name}: parquet re-read failed")
    (out / ".train.parquet.tmp").replace(out / "train.parquet")
    (out / ".valid.parquet.tmp").replace(out / "valid.parquet")

    def dist(rs: list[dict]) -> dict:
        infos = [r["extra_info"] for r in rs]
        return {
            "sites": dict(Counter(s for i in infos for s in i["sites"])),
            "expected_task_type": dict(
                Counter(i["expected_task_type"] for i in infos)),
            "multi_site": sum(1 for i in infos if len(i["sites"]) > 1),
            "with_network_eval": sum(1 for i in infos if i["has_network_eval"]),
        }

    manifest = {
        "source_sha256": source_sha,
        "source_is_default_pinned": is_default,
        "webarena_verified_pinned": "1.2.3",
        "lite165_sha256": _LITE_SHA,
        "row_contract": "prompt/data_source/agent_name/reward_model/extra_info",
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
