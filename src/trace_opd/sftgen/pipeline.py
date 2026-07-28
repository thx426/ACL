"""Researcher-SFT offline sampling pipeline (user requirement 2026-07-28).

A dedicated, benchmark-extensible pipeline that produces the cold-start data
for the self-research channel — deliberately NOT the training run's research/
archives: here every ingredient is pinned and inspectable.

  tasks     a fixed-size stratified sample from the TRAIN parquet (same file
            training reads — task identity, GT extraction and initial_obs are
            identical by construction)
  researcher the strong sampling model (e.g. 35B-A3B) over the standard
            OpenAI surface (roles.researcher; fncall by default so any
            OpenAI-compatible sampler plugs in — export converts to the text
            protocol the policy trains under; research.tool_protocol=text
            samples natively instead)
  executor  the INITIAL policy (Qwen3-4B-Instruct-2507) served by vLLM — the
            researcher's experiment episodes see the same feedback
            distribution the self-evolve run will start from
  narrator  the same initial-policy server, thinking off
  gt        research.gt — ONE config key shared with the self-evolve runtime
            (user ruling: the SFT-time and train-time settings must agree;
            per-benchmark choice). gt only gates the reference solution in the
            researcher prompt; run_experiment success/failure feedback is the
            environment's and always present.
  utility   every finalized document is measured on the initial policy
            (eval_k doc-conditioned + bare_k bare rollouts, mechanism
            temperature) — the gain is ANNOTATED, never used to drop data at
            sampling time; it is also the 35B-side baseline the M0 probe
            compares the cold-started policy against.

Per-task directories carry a DONE marker: reruns skip finished tasks, so the
pipeline is resumable and idempotent. Single-task failures print TASK_ERROR
and never kill the batch (h1 discipline).
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from ..common.config import Config, RoleCfg
from ..common.llm import make_client
from ..common.tape import TAPE_CONTEXT, TapeWriter
from ..core.episode import run_episode
from ..core.template_delta import TemplateDelta
from ..envs.alfworld.adapter import AlfworldEnvFactory
from ..evalm.run_eval import external_generate, load_tasks
from ..research.prompts import TOOL_SCHEMAS, build_researcher_system
from ..research.session import ResearchSession
from ..research.toolproto import make_tool_protocol
from ..research.tools import ToolRouter
from .strata import STRATUM_FNS, stratified_sample


def task_safe(task_id: str) -> str:
    return task_id.replace("/", "__").replace(" ", "_")


class SftGenDeps:
    """Process-wide shared clients/singletons for one pipeline run."""

    def __init__(self, cfg: Config, out_dir: Path, tokenizer) -> None:
        self.cfg = cfg
        self.out_dir = out_dir
        self.tokenizer = tokenizer
        self.delta = TemplateDelta(tokenizer)
        self.factory = AlfworldEnvFactory(cfg.alfworld)
        sg = cfg.sft_gen
        tape_dir = out_dir / "tape"
        self.researcher_client = make_client(
            "researcher",
            cfg.role("researcher"),
            tape=TapeWriter(tape_dir / "researcher.jsonl"),
        )
        # Executor + narrator share ONE initial-policy server. The executor
        # goes through the token-level completions surface (chat template
        # applied locally — protocol identity with training by construction,
        # same GenerateFn evalm uses); the narrator goes through chat with
        # thinking off (playbook 2.2: narration must not burn its budget on an
        # unclosed think block).
        self.http = httpx.AsyncClient()
        self.executor_generate = external_generate(
            self.http, sg.executor_endpoint.rstrip("/"), sg.executor_model, tokenizer
        )
        narrator_role = RoleCfg(
            endpoint=sg.executor_endpoint,
            model=sg.executor_model,
            temperature=cfg.narration.temperature,
            max_tokens=max(1000, cfg.narration.max_words * 3),
            max_concurrency=sg.task_parallel * 2,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.narrator_client = make_client(
            "narrator", narrator_role, tape=TapeWriter(tape_dir / "narrator.jsonl")
        )

    async def aclose(self) -> None:
        await self.http.aclose()


def _narrate_fn(deps: SftGenDeps):
    async def narrate(messages, temperature, max_tokens):
        result = await deps.narrator_client.chat(
            messages, temperature=temperature, max_tokens=max_tokens
        )
        return result.text

    return narrate


def _researcher_chat(deps: SftGenDeps, task_id: str):
    rcfg = deps.cfg.research

    async def chat(messages, tools):
        return await deps.researcher_client.chat(
            messages,
            tools=tools,
            max_tokens=rcfg.researcher_max_tokens,
            route_key=task_id,
        )

    return chat


async def _measure_doc(
    deps: SftGenDeps, item: dict[str, Any], doc: str | None
) -> dict[str, Any]:
    """Doc utility on the initial policy — the PG-7 signal at the mechanism's
    own temperature (matches the runtime's p_i, NOT evalm's greedy eval arm),
    so these numbers are directly comparable with training-time rewards."""
    sg = deps.cfg.sft_gen
    sampling = {"temperature": deps.cfg.research.episode_temperature}

    async def one(guidance: str, tag: str, index: int):
        adapter = deps.factory.adapter_from_item(item)
        result = await run_episode(
            adapter=adapter,
            generate=deps.executor_generate,
            tokenizer=deps.tokenizer,
            delta=deps.delta,
            guidance=guidance,
            episode_id=f"{tag}{index}",
            response_budget=deps.cfg.train.response_length,
            sampling_params=sampling,
        )
        return {
            "won": bool(result.trajectory.won),
            "outcome": result.trajectory.outcome,
            "env_steps": sum(1 for s in result.trajectory.steps if s.acted),
        }

    doc_evals: list[dict[str, Any]] = []
    if doc and sg.eval_k > 0:
        doc_evals = list(
            await asyncio.gather(*[one(doc, "d", i) for i in range(sg.eval_k)])
        )
    bare_evals: list[dict[str, Any]] = []
    if sg.bare_k > 0:
        bare_evals = list(
            await asyncio.gather(*[one("", "b", i) for i in range(sg.bare_k)])
        )
    p_doc = (
        sum(e["won"] for e in doc_evals) / len(doc_evals) if doc_evals else 0.0
    )
    p_bare = (
        sum(e["won"] for e in bare_evals) / len(bare_evals) if bare_evals else 0.0
    )
    return {
        "p_doc": p_doc,
        "p_bare": p_bare,
        "gain": p_doc - p_bare,
        "doc_evals": doc_evals,
        "bare_evals": bare_evals,
    }


async def run_one_session(
    deps: SftGenDeps, item: dict[str, Any], session_index: int
) -> dict[str, Any]:
    """One researcher session + utility measurement + archive. Returns the
    summary row; writes DONE last so a crash mid-task is retried on rerun."""
    cfg, sg, rcfg = deps.cfg, deps.cfg.sft_gen, deps.cfg.research
    task_id = str(item["task_id"])
    task_dir = deps.out_dir / "tasks" / task_safe(task_id) / f"s{session_index}"
    done = task_dir / "DONE"
    if done.exists():
        return json.loads((task_dir / "meta.json").read_text(encoding="utf-8"))

    task_dir.mkdir(parents=True, exist_ok=True)
    adapter = deps.factory.adapter_from_item(item)
    router = ToolRouter(
        adapter=adapter,
        research=rcfg,
        narration=cfg.narration,
        generate_factory=lambda: deps.executor_generate,
        tokenizer=deps.tokenizer,
        delta=deps.delta,
        episode_response_budget=cfg.train.response_length,
        narrate_fn=_narrate_fn(deps),
    )
    session = ResearchSession(
        researcher_chat=_researcher_chat(deps, task_id),
        toolproto=make_tool_protocol(rcfg.tool_protocol),
        router=router,
        system_prompt=build_researcher_system(
            task_brief=adapter.task_brief(),
            executor_model=sg.executor_model or "the initial policy",
            n_exp=rcfg.n_exp,
            doc_max_words=rcfg.doc_max_words,
            gt_text=adapter.gt_text() if rcfg.gt == "on" else None,
        ),
        max_turns=rcfg.max_turns,
        wrapup_buffer=rcfg.wrapup_buffer,
    )
    token = TAPE_CONTEXT.set(f"sftgen:{task_id}:s{session_index}")
    try:
        result = await asyncio.wait_for(session.run(), timeout=rcfg.session_timeout_s)
    except asyncio.TimeoutError:
        result = session.snapshot(error=f"session exceeded {rcfg.session_timeout_s:.0f}s")
    finally:
        TAPE_CONTEXT.reset(token)

    utility = await _measure_doc(deps, item, result.doc)

    (task_dir / "session.json").write_text(
        json.dumps(
            {"messages": result.messages, "tool_protocol": rcfg.tool_protocol},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    if result.doc:
        (task_dir / "doc.md").write_text(result.doc, encoding="utf-8")
    (task_dir / "evals.json").write_text(
        json.dumps(utility, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    meta = {
        "task_id": task_id,
        "session_index": session_index,
        "finalized": bool(result.doc),
        "no_finalize": result.no_finalize,
        "error": result.error,
        "turns_used": result.turns_used,
        "experiments_used": result.experiments_used,
        "episodes_consumed": result.episodes_consumed,
        "doc_words": len(result.doc.split()) if result.doc else 0,
        "p_doc": utility["p_doc"],
        "p_bare": utility["p_bare"],
        "gain": utility["gain"],
        "researcher_model": cfg.role("researcher").model,
        "executor_model": sg.executor_model,
        "tool_protocol": rcfg.tool_protocol,
        "gt": rcfg.gt,
        "ts": time.time(),
    }
    (task_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    done.write_text("ok", encoding="utf-8")
    print(
        f"[sftgen] {task_id} s{session_index}: doc={'yes' if result.doc else 'NO'} "
        f"p_doc={utility['p_doc']:.2f} p_bare={utility['p_bare']:.2f} "
        f"gain={utility['gain']:+.2f} turns={result.turns_used}"
        + (f" ERROR={result.error}" if result.error else ""),
        flush=True,
    )
    return meta


async def run_pipeline(cfg: Config, out_dir: Path, tokenizer) -> dict[str, Any]:
    from ..core import protocol

    protocol.configure_episode_caps(
        cfg.episode.max_env_steps, cfg.episode.max_assistant_turns
    )
    sg = cfg.sft_gen
    stratum_fn = STRATUM_FNS[sg.benchmark]
    all_items = load_tasks([cfg.train.train_file])
    picked = stratified_sample(all_items, sg.tasks, sg.seed, stratum_fn)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark": sg.benchmark,
        "train_file": cfg.train.train_file,
        "population": len(all_items),
        "sampled_tasks": len(picked),
        "sessions_per_task": sg.sessions_per_task,
        "seed": sg.seed,
        "sft_gen": asdict(sg),
        "research": asdict(cfg.research),
        "narration": asdict(cfg.narration),
        "researcher_model": cfg.role("researcher").model,
        "researcher_endpoint": cfg.role("researcher").endpoint,
        "strata": sorted({stratum_fn(t) for t in picked}),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    deps = SftGenDeps(cfg, out_dir, tokenizer)
    sem = asyncio.Semaphore(sg.task_parallel)
    rows: list[dict[str, Any]] = []

    async def guarded(item, index):
        async with sem:
            try:
                rows.append(await run_one_session(deps, item, index))
            except Exception as err:  # noqa: BLE001 — isolate, never kill the batch
                print(f"TASK_ERROR {item.get('task_id')}: {type(err).__name__}: {err}", flush=True)
                rows.append(
                    {"task_id": str(item.get("task_id")), "error": f"{type(err).__name__}: {err}"}
                )

    try:
        await asyncio.gather(
            *[
                guarded(item, index)
                for item in picked
                for index in range(sg.sessions_per_task)
            ]
        )
    finally:
        await deps.aclose()

    finalized = [r for r in rows if r.get("finalized")]
    gains = [r["gain"] for r in finalized if "gain" in r]
    summary = {
        "sessions": len(rows),
        "errors": sum(1 for r in rows if r.get("error")),
        "finalized": len(finalized),
        "finalize_rate": round(len(finalized) / len(rows), 4) if rows else 0.0,
        "mean_gain": round(sum(gains) / len(gains), 4) if gains else 0.0,
        "positive_gain": sum(1 for g in gains if g > 0),
        "mean_p_doc": round(
            sum(r["p_doc"] for r in finalized) / len(finalized), 4
        )
        if finalized
        else 0.0,
        "mean_p_bare": round(
            sum(r.get("p_bare", 0.0) for r in rows if "p_bare" in r)
            / max(1, sum(1 for r in rows if "p_bare" in r)),
            4,
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"[sftgen] summary: {json.dumps(summary, ensure_ascii=False)}", flush=True)
    return summary
