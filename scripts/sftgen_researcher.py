#!/usr/bin/env python3
"""v4 researcher SFT 冷启动数据管线(用户裁定 2026-08-06;
docs/pgs-self-researcher-design.md §SFT)。

两个子命令(可断点续跑):

  sample   固定 seed 按 benchmark 类别分层均匀抽 N 任务(默认 100),每任务
           M 个独立 research 会话(默认 8),researcher = 外部强模型端点
           (Qwen3.6-35B-A3B),executor/narrator = 目标 policy 的推理 server。
           **机制参数与 v4 训练器同一配置真源**(load_config 同一份 yaml:
           text 协议、researcher system prompt、n_exp、max_turns、k、
           researcher_temperature、episode caps 全部同源,不复制数字)。
  convert  会话目录 → verl multiturn SFT parquet(messages 列;verl
           MultiTurnSFTDataset 对 assistant 轮自动 loss=1、其余=0)+
           质量筛选(finalized & doc 非空 & 无 error)+ 统计报告。

分层声明(用户要求显式划分):
  * benchmark 特定:任务分层抽样的 family 推导(ALFWorld task_type,经
    AlfworldEnvFactory 取)、executor scaffold / task brief(经 adapter);
  * 通用机制:research 会话循环(session/toolproto/tools/narration)、
    续跑/落盘/转换、verl SFT 训练本身。换 benchmark 只换「任务抽样 +
    adapter 工厂」两处(标注 [BENCHMARK-SPECIFIC])。

信息状态:researcher 无 GT(gt: off 同源生效),会话数据 = researcher
可见信息,SFT 无特权泄漏(信息状态一致性红线合规)。

部署事实走 CLI/env(端点、模型路径),不进 yaml(配置纪律)。
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402
import pandas as pd  # noqa: E402

from trace_opd.common.config import load_config  # noqa: E402
from trace_opd.common.llm import make_client  # noqa: E402
from trace_opd.core.template_delta import TemplateDelta, render_ids  # noqa: E402
from trace_opd.evalm.run_eval import external_generate, load_tasks  # noqa: E402
from trace_opd.research.prompts import build_researcher_system  # noqa: E402
from trace_opd.research.session import ResearchSession  # noqa: E402
from trace_opd.research.toolproto import make_tool_protocol  # noqa: E402
from trace_opd.research.tools import ToolRouter  # noqa: E402


def _safe(task_id: str) -> str:
    return task_id.replace("/", "__").replace(" ", "_")


# ---------------------------------------------------------------------------
# [BENCHMARK-SPECIFIC] 任务分层抽样(ALFWorld:family = adapter.task_type)
# ---------------------------------------------------------------------------

def stratified_sample_tasks(
    items: list[dict], n_tasks: int, seed: int, factory
) -> list[dict]:
    """按 family 分层均匀抽样(固定 seed 可复现):family 桶内确定性排序后
    shuffle,round-robin 轮转取满 n_tasks——桶间数量差 ≤1。family 推导经
    benchmark 工厂的 adapter(唯一 benchmark 耦合点)。"""
    buckets: dict[str, list[dict]] = {}
    for item in items:
        fam = factory.adapter_from_item(item).task_type
        buckets.setdefault(fam, []).append(item)
    rng = random.Random(seed)
    order = sorted(buckets)
    for fam in order:
        buckets[fam].sort(key=lambda it: str(it.get("task_id", "")))
        rng.shuffle(buckets[fam])
    picked: list[dict] = []
    idx = {fam: 0 for fam in order}
    while len(picked) < n_tasks:
        progressed = False
        for fam in order:
            if len(picked) >= n_tasks:
                break
            if idx[fam] < len(buckets[fam]):
                picked.append(buckets[fam][idx[fam]])
                idx[fam] += 1
                progressed = True
        if not progressed:
            break  # 任务总量不足 n_tasks:取尽为止(报告如实)
    return picked


# ---------------------------------------------------------------------------
# 通用:离线会话采样(与 v4 训练器同一机制组件、同一配置真源)
# ---------------------------------------------------------------------------

def _offline_narrate_fn(http: httpx.AsyncClient, api_base: str, model: str,
                        tokenizer):
    """narration.backend=engine 的离线等价:policy(executor server)叙述。"""
    async def narrate(messages, temperature: float, max_tokens: int) -> str:
        prompt_ids = render_ids(tokenizer, messages, True)
        resp = await http.post(
            f"{api_base}/completions",
            json={"model": model, "prompt": prompt_ids,
                  "temperature": temperature, "max_tokens": max_tokens},
            timeout=600.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"]

    return narrate


async def _one_session(
    *, cfg, item, session_idx: int, researcher_client, tokenizer, delta,
    http, args, factory, out_dir: Path,
) -> dict:
    adapter = factory.adapter_from_item(item)
    out_path = out_dir / _safe(adapter.task_id) / f"s{session_idx}.json"
    if out_path.exists():
        return {"task_id": adapter.task_id, "session": session_idx, "skipped": True}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async def researcher_chat(messages, tools):
        return await researcher_client.chat(
            messages, tools=tools,
            max_tokens=cfg.research.researcher_max_tokens,
            route_key=f"{adapter.task_id}:{session_idx}",
        )

    router = ToolRouter(
        adapter=adapter,
        research=cfg.research,
        narration=cfg.narration,
        generate_factory=lambda: external_generate(
            http, args.executor_api_base, args.executor_model, tokenizer
        ),
        tokenizer=tokenizer,
        delta=delta,
        episode_response_budget=cfg.train.response_length,
        narrate_fn=_offline_narrate_fn(
            http, args.executor_api_base, args.executor_model, tokenizer
        ),
    )
    session = ResearchSession(
        researcher_chat=researcher_chat,
        toolproto=make_tool_protocol(cfg.research.tool_protocol),
        router=router,
        system_prompt=build_researcher_system(
            task_brief=adapter.task_brief(),
            executor_model=Path(args.model_path).name,
            n_exp=cfg.research.n_exp,
            doc_max_words=cfg.research.doc_max_words,
            gt_text=None,
        ),
        max_turns=cfg.research.max_turns,
        wrapup_buffer=cfg.research.wrapup_buffer,
    )
    try:
        result = await asyncio.wait_for(
            session.run(), timeout=cfg.research.session_timeout_s
        )
    except asyncio.TimeoutError:
        result = session.snapshot(error="session timeout")
    payload = {
        "task_id": adapter.task_id,
        "family": adapter.task_type,          # [BENCHMARK-SPECIFIC] 观察字段
        "session_idx": session_idx,
        "seed": args.seed,
        "messages": result.messages,
        "doc": result.doc,
        "finalized": result.finalized,
        "no_finalize": result.no_finalize,
        "error": result.error,
        "turns_used": result.turns_used,
        "experiments_used": result.experiments_used,
        "episodes_consumed": result.episodes_consumed,
        "meta": {
            "researcher_model": args.researcher_model,
            "researcher_temperature": cfg.research_rl.researcher_temperature,
            "tool_protocol": cfg.research.tool_protocol,
            "n_exp": cfg.research.n_exp,
            "max_turns": cfg.research.max_turns,
            "config": str(args.config),
            "ts": time.time(),
        },
    }
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.rename(out_path)
    return {"task_id": adapter.task_id, "session": session_idx,
            "finalized": result.finalized, "error": result.error}


async def cmd_sample(args) -> int:
    from transformers import AutoTokenizer

    from trace_opd.core import protocol
    from trace_opd.envs.alfworld.adapter import AlfworldEnvFactory  # [BENCHMARK-SPECIFIC]

    cfg = load_config(args.config)
    protocol.configure_episode_caps(
        cfg.episode.max_env_steps, cfg.episode.max_assistant_turns
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    delta = TemplateDelta(tokenizer)
    factory = AlfworldEnvFactory(cfg.alfworld)   # [BENCHMARK-SPECIFIC]

    items = load_tasks([args.train_file])
    tasks = stratified_sample_tasks(items, args.n_tasks, args.seed, factory)
    fam_counts: dict[str, int] = {}
    for it in tasks:
        fam = factory.adapter_from_item(it).task_type
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
    print(f"[sftgen] sampled {len(tasks)} tasks (seed={args.seed}): {fam_counts}")

    # researcher = 外部强模型端点(35B);温度与 v4 训练器同源
    # (research_rl.researcher_temperature,用户裁定 1.0);thinking 关闭
    # (历史裁定:researcher 统一关 thinking)。
    # 客户端经 make_client 从 RoleCfg 构造 —— 与训练器 agent_loop 同一路径。
    # 两条必须走工厂的理由(均为实测事故,2026-08-07 162 阶段A 试跑):
    #   ① 鉴权:配置已声明 roles.researcher.api_key_env,RoleCfg 构造时解析为
    #      role.api_key;硬编码空串 ⇒ 全量 401,表征为 "finalize 0/8" 而非鉴权错;
    #   ② 多端点:逗号分隔的 endpoint 必须走 PooledClient(会话粘滞+故障切换);
    #      直接塞进单端点客户端会拼出 ".../v1,http://.../v1/chat/completions" ⇒ 全量 404。
    # 部署事实(端点/模型名)由 CLI 覆盖配置,配置纪律不变。
    researcher_role = replace(
        cfg.role("researcher"),
        endpoint=args.researcher_api_base,
        model=args.researcher_model,
        max_concurrency=args.researcher_concurrency,
        temperature=cfg.research_rl.researcher_temperature,
        max_tokens=cfg.research.researcher_max_tokens,
        timeout_s=600.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    if not researcher_role.api_key_env:
        raise SystemExit("FATAL: roles.researcher.api_key_env is empty")
    if not researcher_role.api_key:
        raise SystemExit(
            f"FATAL: researcher API key env var {researcher_role.api_key_env!r} "
            "is unset or empty — 采样会全量 401"
        )
    researcher_client = make_client("researcher-sftgen", researcher_role)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient() as http:
        async def guarded(item, k):
            async with sem:
                return await _one_session(
                    cfg=cfg, item=item, session_idx=k,
                    researcher_client=researcher_client, tokenizer=tokenizer,
                    delta=delta, http=http, args=args, factory=factory,
                    out_dir=out_dir,
                )

        results = await asyncio.gather(*[
            guarded(item, k)
            for item in tasks for k in range(args.sessions_per_task)
        ])
    done = [r for r in results if not r.get("skipped")]
    fin = sum(1 for r in done if r.get("finalized"))
    print(f"[sftgen] sessions: {len(results)} total, {len(results) - len(done)} "
          f"skipped(resume), {len(done)} new, finalize {fin}/{len(done) or 1}")
    return 0


# ---------------------------------------------------------------------------
# 通用:convert → verl multiturn SFT parquet + 质量筛选 + 报告
# ---------------------------------------------------------------------------

def cmd_convert(args) -> int:
    sessions = sorted(Path(args.sessions_dir).glob("*/s*.json"))
    rows, report = [], {
        "total": 0, "kept": 0, "no_finalize": 0, "empty_doc": 0, "error": 0,
        "by_family": {}, "turns": [], "assistant_chars": [],
    }
    for path in sessions:
        payload = json.loads(path.read_text(encoding="utf-8"))
        report["total"] += 1
        if payload.get("error"):
            report["error"] += 1
            continue
        if not payload.get("finalized"):
            report["no_finalize"] += 1
            continue
        if not (payload.get("doc") or "").strip():
            report["empty_doc"] += 1
            continue
        messages = payload["messages"]
        if not messages or messages[0].get("role") != "system":
            report["error"] += 1
            continue
        rows.append({
            "messages": messages,
            "task_id": payload["task_id"],
            "family": payload.get("family", ""),
            "session_idx": payload.get("session_idx", -1),
        })
        report["kept"] += 1
        fam = payload.get("family", "?")
        report["by_family"][fam] = report["by_family"].get(fam, 0) + 1
        report["turns"].append(payload.get("turns_used", 0))
        report["assistant_chars"].append(sum(
            len(m.get("content") or "") for m in messages
            if m.get("role") == "assistant"
        ))
    if not rows:
        raise SystemExit("FATAL: no sessions survived filtering — nothing to train on")
    out = Path(args.out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    if report["turns"]:
        ts = sorted(report["turns"])
        cs = sorted(report["assistant_chars"])
        report["turns_p50"] = ts[len(ts) // 2]
        report["turns_max"] = ts[-1]
        report["assistant_chars_p50"] = cs[len(cs) // 2]
        report["assistant_chars_max"] = cs[-1]
    for k in ("turns", "assistant_chars"):
        report.pop(k)
    report_path = out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[sftgen] wrote {len(rows)} rows -> {out}")
    print(f"[sftgen] report: {json.dumps(report, ensure_ascii=False)}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("sample", help="35B researcher 会话采样(可断点续跑)")
    ps.add_argument("--config", required=True, help="v4 配置 yaml(机制参数真源)")
    ps.add_argument("--train-file", required=True, help="训练集 parquet")
    ps.add_argument("--model-path", required=True, help="目标 policy HF 目录(tokenizer/命名)")
    ps.add_argument("--executor-api-base", required=True, help="policy 推理 server /v1")
    ps.add_argument("--executor-model", required=True, help="server 侧 served 模型名")
    ps.add_argument("--researcher-api-base", required=True, help="35B 端点 /v1(可逗号分隔)")
    ps.add_argument("--researcher-model", required=True)
    ps.add_argument("--out-dir", required=True, help="会话落盘目录(续跑判据)")
    ps.add_argument("--n-tasks", type=int, default=100)
    ps.add_argument("--sessions-per-task", type=int, default=8)
    ps.add_argument("--seed", type=int, default=42)
    ps.add_argument("--concurrency", type=int, default=16, help="并发会话数")
    ps.add_argument("--researcher-concurrency", type=int, default=64)

    pc = sub.add_parser("convert", help="会话目录 → verl multiturn SFT parquet")
    pc.add_argument("--sessions-dir", required=True)
    pc.add_argument("--out-parquet", required=True)

    args = p.parse_args()
    if args.cmd == "sample":
        return asyncio.run(cmd_sample(args))
    return cmd_convert(args)


if __name__ == "__main__":
    raise SystemExit(main())
