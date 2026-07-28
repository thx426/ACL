"""Researcher-SFT offline sampling — entry point (src/trace_opd/sftgen/).

Samples a fixed, stratified set of TRAIN tasks; a strong researcher (the
roles.researcher endpoint, e.g. 35B-A3B) studies each one while the INITIAL
policy (a vLLM server you point --executor-endpoint at) plays executor and
narrator; every document's utility is measured on that same policy. Archives
are resumable (per-task DONE); export to verl SFT parquet is a separate step
(python -m trace_opd.sftgen.export).

Typical use (server side):

  # 1. serve the initial policy (executor + narrator)
  #    vllm serve Qwen3-4B-Instruct-2507 --port 8311 ... (127.0.0.1 only)
  # 2. make sure roles.researcher points at the 35B endpoint (same env vars
  #    the training launcher uses)
  PYTHONPATH=src python3 scripts/sample_researcher_sft.py \
      --config configs/trace.yaml \
      --out-dir /abs/path/runs/sftgen_v1 \
      --executor-endpoint http://127.0.0.1:8311/v1 \
      --executor-model Qwen3-4B-Instruct-2507 \
      --tasks 32                      # small calibration batch first

gt / tool protocol / session budgets all come from the SHARED research
section of the config — the one self-evolve training reads — so the SFT-time
and train-time settings agree by construction (user ruling 2026-07-28).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trace_opd.common.config import load_config  # noqa: E402
from trace_opd.sftgen.pipeline import run_pipeline  # noqa: E402
from trace_opd.sftgen.strata import STRATUM_FNS, stratified_sample  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True, help="archive root (absolute)")
    parser.add_argument("--set", action="append", default=[], dest="overrides",
                        help="config override path=value (repeatable)")
    parser.add_argument("--tasks", type=int, default=None, help="override sft_gen.tasks")
    parser.add_argument("--executor-endpoint", default=None)
    parser.add_argument("--executor-model", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the sampled task list and exit (no serving needed)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.tasks is not None:
        if int(args.tasks) < 1:
            raise SystemExit("FATAL: --tasks must be >= 1")  # codex review P1-8
        cfg.sft_gen.tasks = int(args.tasks)
    if args.executor_endpoint:
        cfg.sft_gen.executor_endpoint = args.executor_endpoint
    if args.executor_model:
        cfg.sft_gen.executor_model = args.executor_model

    if not cfg.train.train_file:
        raise SystemExit(
            "FATAL: train.train_file is empty — the pipeline samples from the TRAIN "
            "parquet (export the same env vars the training launcher uses, or --set "
            "train.train_file=/abs/path/train.parquet)"
        )
    if args.dry_run:
        from trace_opd.evalm.run_eval import load_tasks

        items = load_tasks([cfg.train.train_file])
        picked = stratified_sample(
            items, cfg.sft_gen.tasks, cfg.sft_gen.seed, STRATUM_FNS[cfg.sft_gen.benchmark]
        )
        strata: dict[str, int] = {}
        for item in picked:
            key = STRATUM_FNS[cfg.sft_gen.benchmark](item)
            strata[key] = strata.get(key, 0) + 1
        print(f"population={len(items)} sampled={len(picked)} strata={strata}")
        for item in picked:
            print(item["task_id"])
        return 0

    if not cfg.sft_gen.executor_endpoint or not cfg.sft_gen.executor_model:
        raise SystemExit(
            "FATAL: --executor-endpoint and --executor-model are required (the vLLM "
            "server holding the INITIAL policy — executor episodes and narration)"
        )
    if not cfg.role("researcher").endpoint:
        raise SystemExit(
            "FATAL: roles.researcher.endpoint is empty — export the same env vars the "
            "training launcher uses (the 35B sampler endpoint)"
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.train.model_path)
    # codex review P1-7 (light): the tokenizer renders prompts the SERVED model
    # must interpret — a mismatched pair corrupts every episode silently.
    served = Path(cfg.sft_gen.executor_model).name
    local = Path(cfg.train.model_path).name
    if served and local and served not in (local, cfg.train.model_path):
        print(
            f"[sftgen] WARNING executor model {served!r} != tokenizer source "
            f"{local!r} — token ids will be interpreted by the SERVED model; "
            "make sure the vocabularies match"
        )
    summary = asyncio.run(run_pipeline(cfg, Path(args.out_dir), tokenizer))
    return 0 if summary["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
