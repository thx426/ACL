"""Build ScienceWorld train/eval parquet datasets from the vendored ETO split.

数据口径(用户裁定 2026-08-08):ETO 划分——24 任务,train 1,483 /
dev(seen)194 / test(unseen)211,变体级 seen/unseen(vendored
data/eto_indices.json,出处 ETO@a2fc5da)。每行 boot 一次引擎(JVM,
本地实测 boot+load ≈1.2s)捕获确定性 initial_obs(taskDesc+首观察),
train 行同时预提取引擎原生 gold action sequence 作 gt_text(GT 防火墙:
只进 researcher/teacher;gold 不跟随 easy 简化,含冗余 open-door 步,
harmless)。

Output parquet rows (verl RLHFDataset-compatible,对齐 alfworld build):
  prompt       : [{system: FC scaffold(per-task,electrical 任务多两行)},
                  {user: initial obs}]
  data_source  : "scienceworld"(train)| "scienceworld/valid_seen|valid_unseen"
  extra_info   : {task_id, task_name, variation_idx, split, initial_obs,
                  gt_text, gt_source}

Usage(worker venv 需装 scienceworld+py4j 且 PATH 有 java>=1.8):
  python -m trace_opd.envs.scienceworld.build_dataset \
      --worker-python $SCIWORLD_WORKER_PYTHON --out-dir data/sciworld \
      [--config configs/experiments/<run>.yaml] [--limit 0]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ...common.config import SciworldCfg, load_config
from ...core import protocol
from .adapter import GT_SENTINEL, _Worker, _is_electrical, _scaffold
from .adapter import COMMAND_REFERENCE, ELECTRICAL_EXTRA, WorkerFailure

_INDICES = Path(__file__).with_name("data") / "eto_indices.json"

# 划分基数(vendored ETO;缩水必须 fail 而非静默减池——alfworld P1-4 同款纪律)
EXPECTED = {"train": 1483, "dev": 194, "test": 211}
_SPLIT_SOURCE = {"train": "scienceworld",
                 "dev": "scienceworld/valid_seen",
                 "test": "scienceworld/valid_unseen"}


def render_gold(plan: list[str] | None) -> str:
    if not plan:
        return ""
    return "\n".join(f"{i + 1}. {cmd}" for i, cmd in enumerate(plan))


def boot_capture(args, cfg: SciworldCfg, task_name: str, variation_idx: int,
                 want_gt: bool) -> tuple[str, str]:
    worker = _Worker(
        python_exe=args.worker_python,
        task_name=task_name,
        variation_idx=variation_idx,
        simplifications=cfg.simplifications,
        env_step_limit=cfg.env_step_limit,
        start_timeout_s=cfg.start_timeout_s,
        step_timeout_s=cfg.step_timeout_s,
        want_gt=want_gt,
    )
    try:
        gold = ""
        if want_gt:
            event = worker.request(GT_SENTINEL)
            gold = render_gold(event.get("plan"))
        return worker.initial_obs, gold
    finally:
        worker.close()


def build_rows(args, cfg: SciworldCfg, split: str,
               pairs: list[list]) -> pd.DataFrame:
    with_gt = split == "train"
    print(f"[boot] {split}: capturing initial obs"
          f"{' + gold path' if with_gt else ''} for {len(pairs)} variations...")

    def one(pair: list) -> dict:
        task_name, variation_idx = str(pair[0]), int(pair[1])
        try:
            initial_obs, gold = boot_capture(
                args, cfg, task_name, variation_idx, want_gt=with_gt
            )
        except WorkerFailure as err:
            raise SystemExit(
                f"FATAL: engine boot failed for {task_name}/{variation_idx}: {err}"
            ) from err
        if not initial_obs.strip():
            raise SystemExit(f"FATAL: empty initial obs for {task_name}/{variation_idx}")
        return {"task_name": task_name, "variation_idx": variation_idx,
                "initial_obs": initial_obs, "gt_text": gold}

    with ThreadPoolExecutor(max_workers=cfg.boot_concurrency) as pool:
        captured = list(pool.map(one, pairs))

    rows = []
    gt_missing = 0
    for cap in captured:
        task_name = cap["task_name"]
        tid = f"{task_name}/{cap['variation_idx']}"
        if with_gt and not cap["gt_text"]:
            gt_missing += 1
        ref = COMMAND_REFERENCE + ("\n" + ELECTRICAL_EXTRA
                                   if _is_electrical(task_name) else "")
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": _scaffold(ref)},
                    {"role": "user", "content": protocol.truncate_obs(cap["initial_obs"])},
                ],
                "data_source": _SPLIT_SOURCE[split],
                "agent_name": "trace_opd",
                "reward_model": {"style": "env", "ground_truth": ""},
                "extra_info": {
                    "task_id": tid,
                    "task_name": task_name,
                    "variation_idx": cap["variation_idx"],
                    "split": split,
                    "initial_obs": cap["initial_obs"],
                    "gt_text": cap["gt_text"],
                    "gt_source": ("engine" if cap["gt_text"] else "none")
                                 if with_gt else "",
                },
            }
        )
    if with_gt:
        print(f"[gt] {split}: {len(rows) - gt_missing}/{len(rows)} variations "
              f"have engine gold paths ({gt_missing} missing)")
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-python", required=True,
                        help="venv with scienceworld+py4j; java>=1.8 on PATH")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="",
                        help="run yaml — simplifications/timeouts 单一真源")
    parser.add_argument("--limit", type=int, default=0,
                        help="per-split cap for smoke builds (0 = full)")
    args = parser.parse_args()

    if args.config:
        cfg_all = load_config(args.config)
        cfg = cfg_all.scienceworld
        protocol.configure_episode_caps(
            cfg_all.episode.max_env_steps,
            cfg_all.episode.max_assistant_turns,
            cfg_all.episode.max_tokens_per_turn,
        )
    else:
        cfg = SciworldCfg()
    cfg.worker_python = args.worker_python

    indices = json.loads(_INDICES.read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        pairs = indices[split]
        if len(pairs) != EXPECTED[split]:
            raise SystemExit(
                f"FATAL: {split} has {len(pairs)} pairs, expected the vendored "
                f"ETO {EXPECTED[split]}"
            )
        if args.limit:
            pairs = pairs[: args.limit]
        df = build_rows(args, cfg, split, pairs)
        name = {"train": "train", "dev": "valid_seen", "test": "valid_unseen"}[split]
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"[write] {path}: {len(df)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
