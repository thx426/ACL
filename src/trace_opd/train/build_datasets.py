"""Build train/eval parquet datasets from the OFFICIAL alfworld layout.

First-principles data source: scan ALFWORLD_DATA/json_2.1.1/{train,valid_seen,
valid_unseen} for game.tw-pddl files (official split; expected counts
valid_seen=140, valid_unseen=134). For every task, boot the engine once to
capture the deterministic initial observation (doubles as a boot validation).

GT pre-extraction (train split only): the expert plan is resolved OFFLINE at
build time — ALFRED traj_data.json high_pddl first, engine walkthrough
(policy_commands) as fallback — and rides the row as extra_info.gt_text.
Whether the researcher sees it is a RUNTIME switch (research.gt); the
training loop never touches the engine for GT.

Output parquet rows (verl RLHFDataset-compatible):
  prompt       : [{system: bare contract}, {user: initial obs}]
  data_source  : "alfworld" (train) | "alfworld/<split>" (val: per-split curves)
  agent_name   : "trace_opd"
  reward_model : {"style": "env", "ground_truth": ""}   (reward comes from the env)
  extra_info   : {task_id, gamefile (relative), split, initial_obs, gt_text}

Usage (server, training venv):
  python -m trace_opd.train.build_datasets \
      --data-root $ALFWORLD_DATA --worker-python $ALFWORLD_WORKER_PYTHON \
      --out-dir trace_opd/data/out [--n-train 32 --seed 42]
"""
from __future__ import annotations

import argparse
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ..common.config import AlfworldCfg, load_config
from ..core import protocol
from ..envs.alfworld import gt as gt_mod
from ..envs.alfworld.adapter import GT_SENTINEL, WorkerFailure, _Worker

# Official split cardinalities — a shrunken data directory must fail the
# build, not silently produce a reduced task pool (review P1-4).
EXPECTED_COUNTS = {"train": 3553, "valid_seen": 140, "valid_unseen": 134}


def scan_split(data_root: Path, split: str) -> list[Path]:
    games = sorted((data_root / "json_2.1.1" / split).glob("**/*.tw-pddl"))
    print(f"[scan] {split}: {len(games)} games")
    expected = EXPECTED_COUNTS.get(split)
    if expected is not None and len(games) != expected:
        raise SystemExit(
            f"FATAL: {split} has {len(games)} games, expected the official {expected}"
        )
    return games


def task_id_of(data_root: Path, gamefile: Path) -> str:
    # json_2.1.1/train/pick_..._simple-Vase-.../trial_T.../game.tw-pddl
    rel = gamefile.relative_to(data_root)
    return "/".join(rel.parts[1:-1])  # drop leading json_2.1.1 and trailing filename


def boot_capture(args, env_cfg: AlfworldCfg, gamefile: Path, want_gt: bool) -> tuple[str, str | None]:
    """One engine boot: initial obs + (optionally) the walkthrough GT."""
    worker = _Worker(
        python_exe=args.worker_python,
        gamefile=str(gamefile),
        max_steps=protocol.GT_EXTRACT_MAX_STEPS,  # expert GT must not be cut
                                                  # by the training budget
        data_root=args.data_root,
        start_timeout_s=env_cfg.start_timeout_s,
        step_timeout_s=env_cfg.step_timeout_s,
        want_gt=want_gt,
    )
    try:
        walkthrough = None
        if want_gt:
            event = worker.request(GT_SENTINEL)
            walkthrough = gt_mod.render_walkthrough(event.get("plan"))
        return worker.initial_obs, walkthrough
    finally:
        worker.close()


def build_rows(
    args, env_cfg: AlfworldCfg, data_root: Path, split: str, games: list[Path]
) -> pd.DataFrame:
    with_gt = split == "train"
    print(f"[boot] {split}: capturing initial obs{' + GT' if with_gt else ''} "
          f"for {len(games)} games...")

    def one(gamefile: Path) -> dict:
        annotation_gt = (
            gt_mod.expert_plan_from_traj_data(args.data_root, str(gamefile.relative_to(data_root)))
            if with_gt
            else None
        )
        try:
            initial_obs, walkthrough = boot_capture(
                args, env_cfg, gamefile, want_gt=with_gt and annotation_gt is None
            )
        except WorkerFailure as err:
            raise SystemExit(f"FATAL: engine boot failed for {gamefile}: {err}") from err
        if not initial_obs.strip():
            raise SystemExit(f"FATAL: empty initial obs for {gamefile}")
        gt_text = annotation_gt or walkthrough
        gt_source = "annotation" if annotation_gt else ("walkthrough" if walkthrough else "none")
        return {
            "gamefile": gamefile,
            "initial_obs": initial_obs,
            "gt_text": gt_text,
            "gt_source": gt_source if with_gt else "",
        }

    with ThreadPoolExecutor(max_workers=env_cfg.boot_concurrency) as pool:
        captured = list(pool.map(one, games))

    rows = []
    gt_missing = 0
    for cap in captured:
        gamefile = cap["gamefile"]
        tid = task_id_of(data_root, gamefile)
        if with_gt and not cap["gt_text"]:
            gt_missing += 1
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": protocol.SYSTEM_PROMPT_BARE},
                    {"role": "user", "content": protocol.truncate_obs(cap["initial_obs"])},
                ],
                # verl groups val metrics by data_source: per-split values give
                # separate swanlab curves (val-core/alfworld/valid_seen/... and
                # .../valid_unseen/...) — the seen/unseen comparison IS the
                # core curve (feedback 2026-07-23). Train keeps the plain name
                # so no training-path behavior can change. Safe because reward
                # comes from the agent loop (rm_scores), never from
                # data_source-routed reward functions.
                "data_source": "alfworld" if split == "train" else f"alfworld/{split}",
                "agent_name": "trace_opd",
                "reward_model": {"style": "env", "ground_truth": ""},
                "extra_info": {
                    "task_id": tid,
                    "gamefile": str(gamefile.relative_to(data_root)),
                    "split": split,
                    "initial_obs": cap["initial_obs"],
                    "gt_text": cap["gt_text"] or "",
                    "gt_source": cap["gt_source"],
                },
            }
        )
    if with_gt:
        print(f"[gt] {split}: {len(rows) - gt_missing}/{len(rows)} tasks have GT "
              f"({gt_missing} missing from both sources)")
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--worker-python", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="",
                        help="trace.yaml — single source for engine timeouts/boot concurrency")
    parser.add_argument("--n-train", type=int, default=0,
                        help="sample this many train tasks (0 = full train split)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        env_cfg = cfg.alfworld
        protocol.configure_episode_caps(
            cfg.episode.max_env_steps, cfg.episode.max_assistant_turns
        )
    else:
        env_cfg = AlfworldCfg()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_games = scan_split(data_root, "train")
    if args.n_train:
        rng = random.Random(args.seed)
        train_games = sorted(rng.sample(train_games, args.n_train))
        print(f"[sample] train: {args.n_train} tasks, seed={args.seed}")
    train_name = f"train_{args.n_train}" if args.n_train else f"train_{len(train_games)}"

    jobs = [
        (train_name, "train", train_games),
        ("valid_seen", "valid_seen", scan_split(data_root, "valid_seen")),
        ("valid_unseen", "valid_unseen", scan_split(data_root, "valid_unseen")),
    ]
    for name, split, games in jobs:
        df = build_rows(args, env_cfg, data_root, split, games)
        out = out_dir / f"{name}.parquet"
        df.to_parquet(out, index=False)
        print(f"[write] {out} rows={len(df)}")
    print("[build_datasets] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
