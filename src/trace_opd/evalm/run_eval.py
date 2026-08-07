"""Standalone evaluator over a vLLM OpenAI server (external backend).

Runs the SAME episode runner as research experiments and the training
rollout (core/episode.py), with the external GenerateFn: the chat template
is applied locally and TOKEN IDS go to /v1/completions — the server never
re-templates anything. Greedy by default; protocol identity with training is
by construction.

Arms:
  bare       system prompt without guidance          (pre/post-training test-set runs)
  doc        + per-task document under "## Task guidance"
             (--doc-dir <dir> with <task_id_safe>.md files, or --doc-file
             one document for every task) — diagnostic arm for train tasks.

Reporting discipline (review P1-9): ENV_ERROR episodes are infrastructure
faults, never task attempts — success rates are computed over attempts, env
errors are reported separately, and any unrecovered error marks the run
incomplete (exit code 2). A non-empty output file is refused unless --force
(review P1-10), and every run writes a provenance manifest (review P3-2).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from ..common.config import load_config
from ..core import protocol
from ..core.episode import GenResult, run_episode
from ..core.template_delta import TemplateDelta
from ..core.types import OUTCOME_ENV_ERROR
from ..envs.registry import make_env_factory


def load_tasks(parquet_paths: list[str]) -> list[dict]:
    tasks = []
    for path in parquet_paths:
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            tasks.append(dict(row["extra_info"]))
    return tasks


def external_generate(http: httpx.AsyncClient, api_base: str, model: str, tokenizer):
    async def gen(prompt_ids: list[int], sampling_params: dict[str, Any]) -> GenResult:
        resp = await http.post(
            f"{api_base}/completions",
            json={"model": model, "prompt": prompt_ids, **sampling_params},
            timeout=300.0,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        text = choice["text"]
        return GenResult(
            token_ids=tokenizer.encode(text, add_special_tokens=False),
            text=text,
            extra={"completion_tokens": data.get("usage", {}).get("completion_tokens", 0)},
        )

    return gen


def _doc_for(task_id: str, args) -> str:
    if args.doc_file:
        return Path(args.doc_file).read_text(encoding="utf-8")
    if args.doc_dir:
        safe = task_id.replace("/", "__").replace(" ", "_")
        path = Path(args.doc_dir) / f"{safe}.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"no document for {task_id} under {args.doc_dir}")
    raise ValueError("--arm doc requires --doc-dir or --doc-file")


def summarize(results: list[dict], arm: str) -> tuple[dict[str, Any], bool]:
    """Per-split summary. ENV_ERROR rows are excluded from the success-rate
    denominator and reported separately; their presence marks the run
    incomplete."""
    splits: dict[str, dict[str, Any]] = {}
    incomplete = False
    for split in sorted({r["split"] for r in results}):
        rows = [r for r in results if r["split"] == split]
        errs = [r for r in rows if r["outcome"] == OUTCOME_ENV_ERROR]
        attempts = [r for r in rows if r["outcome"] != OUTCOME_ENV_ERROR]
        wins = sum(1 for r in attempts if r["won"])
        if errs:
            incomplete = True
        splits[split] = {
            "total": len(rows),
            "attempts": len(attempts),
            "won": wins,
            "success_rate": round(wins / len(attempts), 4) if attempts else 0.0,
            "env_errors": len(errs),
            "mean_env_steps": (
                round(sum(r["env_steps"] for r in attempts) / len(attempts), 1)
                if attempts
                else 0.0
            ),
        }
    return {"arm": arm, "incomplete": incomplete, "splits": splits}, incomplete


async def main_async(args) -> int:
    from transformers import AutoTokenizer

    config = load_config(args.config)
    from ..core import protocol

    protocol.configure_episode_caps(
        config.episode.max_env_steps, config.episode.max_assistant_turns
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    delta = TemplateDelta(tokenizer)
    factory = make_env_factory(config)
    tasks = load_tasks(args.parquet)
    if not tasks:
        raise SystemExit("FATAL: no tasks loaded from the given parquet files")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    if out_path.exists():
        if not args.force:
            raise SystemExit(
                f"FATAL: {out_path} already exists — a rerun would interleave "
                "episodes from different runs. Use --force to overwrite."
            )
        out_path.unlink()

    # Provenance manifest: the artifacts must prove their own identity.
    manifest = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arm": args.arm,
        "model_path": args.model_path,
        "served_model": args.served_model,
        "api_base": args.api_base,
        "parquet": [str(Path(p).resolve()) for p in args.parquet],
        "n_tasks": len(tasks),
        "temperature": config.eval.temperature,
        "response_budget": config.eval.response_budget,
        "doc_dir": args.doc_dir,
        "doc_file": args.doc_file,
        # 生效 scaffold 的身份指纹(经 adapter,任务无关;ALFWorld 下与
        # 历史 protocol.SYSTEM_PROMPT_BARE 哈希逐字节一致)
        "system_prompt_sha1": hashlib.sha1(
            factory.adapter_from_item(tasks[0]).executor_scaffold().encode("utf-8")
        ).hexdigest()[:12],
        "config": config.raw,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    sem = asyncio.Semaphore(args.parallel)
    sampling = {"temperature": config.eval.temperature}

    async with httpx.AsyncClient() as http:
        gen = external_generate(http, args.api_base, args.served_model, tokenizer)

        async def one(task: dict) -> dict:
            async with sem:
                started = time.time()
                guidance = _doc_for(task["task_id"], args) if args.arm == "doc" else ""
                adapter = factory.adapter_from_item(task)
                res = await run_episode(
                    adapter=adapter,
                    generate=gen,
                    tokenizer=tokenizer,
                    delta=delta,
                    guidance=guidance,
                    episode_id="eval",
                    response_budget=config.eval.response_budget,
                    sampling_params=sampling,
                )
                traj = res.trajectory
                record = {
                    "task_id": task["task_id"],
                    "split": task["split"],
                    "arm": args.arm,
                    "outcome": traj.outcome,
                    "won": traj.won,
                    # env_steps = commands that reached the engine (h3-comparable);
                    # turns/trajectory keep the full picture.
                    "env_steps": sum(1 for s in traj.steps if s.acted),
                    "assistant_turns": res.assistant_turns,
                    "invalid_format": res.invalid_format,
                    "truncated": res.truncated,
                    "error": traj.error,
                    "wall_s": round(time.time() - started, 1),
                    "trajectory": traj.to_dict(),
                }
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                return record

        results = list(await asyncio.gather(*[one(t) for t in tasks]))

    summary, incomplete = summarize(results, args.arm)
    for split, s in summary["splits"].items():
        line = (
            f"[{args.arm}] {split}: {s['won']}/{s['attempts']} = "
            f"{s['success_rate']:.2%} (mean_env_steps={s['mean_env_steps']})"
        )
        if s["env_errors"]:
            line += f"  !! env_errors={s['env_errors']} (excluded from denominator)"
        print(line)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if incomplete:
        print("[eval] INCOMPLETE: environment errors occurred — numbers are not final.")
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True, help="HF dir (tokenizer/template source)")
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--parquet", nargs="+", required=True)
    parser.add_argument("--arm", choices=["bare", "doc"], default="bare")
    parser.add_argument("--doc-dir", default="")
    parser.add_argument("--doc-file", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--parallel", type=int, default=32)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing episodes.jsonl")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
