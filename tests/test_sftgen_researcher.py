"""sftgen_researcher(v4 SFT 冷启动管线)——分层抽样与转换契约。

红线:固定 seed 逐字节可复现;family 桶间数量差 ≤1(均匀);convert 只
保留 finalized & doc 非空 & 无 error 的会话,messages 原样落 parquet
(verl MultiTurnSFTDataset 的 assistant-only loss mask 依赖 role 字段)。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sftgen_researcher.py"
_spec = importlib.util.spec_from_file_location("sftgen_researcher", _SCRIPT)
sftgen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sftgen)


class _FakeFactory:
    """duck-typed benchmark 工厂:adapter_from_item → task_type/task_id。"""

    def adapter_from_item(self, item):
        class _A:
            task_id = item["task_id"]
            task_type = item["task_id"].split("/")[1].split("-", 1)[0]
        return _A()


def _items():
    fams = {"pick": 30, "heat": 20, "cool": 10, "look": 4}
    items = []
    for fam, n in fams.items():
        for i in range(n):
            items.append({"task_id": f"train/{fam}-Obj-{i}/trial_{i}"})
    return items


def test_stratified_sampling_is_uniform_and_seed_deterministic():
    items = _items()
    picked = sftgen.stratified_sample_tasks(items, 16, seed=7, factory=_FakeFactory())
    assert len(picked) == 16
    counts: dict[str, int] = {}
    for it in picked:
        fam = it["task_id"].split("/")[1].split("-", 1)[0]
        counts[fam] = counts.get(fam, 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1   # 桶间差 ≤1
    again = sftgen.stratified_sample_tasks(items, 16, seed=7, factory=_FakeFactory())
    assert [x["task_id"] for x in again] == [x["task_id"] for x in picked]
    other = sftgen.stratified_sample_tasks(items, 16, seed=8, factory=_FakeFactory())
    assert [x["task_id"] for x in other] != [x["task_id"] for x in picked]


def test_stratified_sampling_exhausts_small_buckets_then_fills():
    items = _items()
    picked = sftgen.stratified_sample_tasks(items, 60, seed=1, factory=_FakeFactory())
    counts: dict[str, int] = {}
    for it in picked:
        fam = it["task_id"].split("/")[1].split("-", 1)[0]
        counts[fam] = counts.get(fam, 0) + 1
    assert counts["look"] == 4                 # 小桶取尽
    assert len(picked) == 60
    capped = sftgen.stratified_sample_tasks(items, 999, seed=1, factory=_FakeFactory())
    assert len(capped) == len(items)           # 总量不足:取尽为止


def _session(tmp_path, task, idx, *, finalized=True, doc="DOC", error=None):
    d = tmp_path / task.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": task, "family": task.split("/")[1].split("-", 1)[0],
        "session_idx": idx,
        "messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "KICK"},
            {"role": "assistant", "content": "think + tool call"},
            {"role": "user", "content": "tool result"},
            {"role": "assistant", "content": "finalize call"},
        ],
        "doc": doc, "finalized": finalized,
        "no_finalize": not finalized, "error": error,
        "turns_used": 2, "experiments_used": 1, "episodes_consumed": 4,
    }
    (d / f"s{idx}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_convert_filters_and_writes_multiturn_parquet(tmp_path):
    _session(tmp_path, "train/pick-A-1/trial_1", 0)
    _session(tmp_path, "train/pick-A-1/trial_1", 1, finalized=False)   # 筛掉
    _session(tmp_path, "train/heat-B-2/trial_2", 0, doc="  ")          # 空 doc
    _session(tmp_path, "train/heat-B-2/trial_2", 1, error="timeout")   # error
    _session(tmp_path, "train/cool-C-3/trial_3", 0)
    out = tmp_path / "out" / "sft.parquet"

    class _Args:
        sessions_dir = str(tmp_path)
        out_parquet = str(out)

    assert sftgen.cmd_convert(_Args()) == 0
    df = pd.read_parquet(out)
    assert len(df) == 2
    msgs = list(df.iloc[0]["messages"])
    assert msgs[0]["role"] == "system"          # role 结构原样(loss mask 依赖)
    assert sum(1 for m in msgs if m["role"] == "assistant") == 2
    report = json.loads(out.with_suffix(".report.json").read_text())
    assert report["total"] == 5 and report["kept"] == 2
    assert report["no_finalize"] == 1 and report["empty_doc"] == 1
    assert report["error"] == 1
    assert report["turns_p50"] == 2


def test_convert_fails_loud_when_nothing_survives(tmp_path):
    _session(tmp_path, "train/pick-A-1/trial_1", 0, finalized=False)

    class _Args:
        sessions_dir = str(tmp_path)
        out_parquet = str(tmp_path / "sft.parquet")

    with pytest.raises(SystemExit, match="no sessions survived"):
        sftgen.cmd_convert(_Args())
