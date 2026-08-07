"""TraceAgentLoop(PGS)—— 每训练样本的 rollout 机制(规范 docs/pgs-final-spec.md)。

每个样本(= 一个任务):
  1. STUDENT ROLLOUT:裸 on-policy episode(steering.enabled 时经 SPS 保尾核
     倾斜采样——P2 接入 episode 钩子;损失中无教师,教师只出现在采样端)。
  2. 记账:逐 token steering ledger 随 extra_fields 进训练端(W_beh 数据源)。
  3. 观察:每样本 dump 完整轨迹到 <run_dir>/rollouts/step_<N>[_val]/。

特权文档来源由 steering.doc_source 选择:static 读取 doc_table；researcher
则每个(global_step,task_id)冻结当前 policy 跑一次在线研究，n=8 共享同一文档。
逐任务失败降级为空文档；系统配置错误在启动前 fail-loud。

环境变量(launcher 注入):
  TRACE_CONFIG          configs/*.yaml 路径(必需)
  TRACE_RUN_DIR         run 根目录(必需;server 侧绝对路径)
  TRACE_AGENT_WORKERS   agent-loop worker 进程数(全局并发上限按此均分)
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)

from ..common.aio import shared_task
from ..common.config import Config, load_config
from ..common.hashing import response_token_hash
from ..common.llm import make_client
from ..common.tape import TapeWriter
from ..core.episode import EpisodeResult, run_episode
from ..core.template_delta import TemplateDelta, render_ids
from ..core.types import OUTCOME_ENV_ERROR, inject_guidance
from ..docstore.archive import ResearchArchive
from ..core.adapter import EnvFactory
from ..envs.registry import make_env_factory
from ..research.prompts import build_researcher_system
from ..research.session import ResearchSession, SessionResult
from ..research.toolproto import make_tool_protocol
from ..research.tools import ToolRouter
from ..research.versioning import (
    assert_rollout_policy_version,
    research_policy_version,
)
from ..steering.arms import ROLE_BARE, ROLE_RESEARCH, ROLE_STEERED, is_steered_arm, role_of, steered_arm_count
from .backends import EngineGenerate, EngineResearcherChat, engine_narrate_fn
from .selfres import CoordRegistry, TaskCoord, run_research_row

#: v4 executor 行的 research 记账零占位(键集必须与 selfres.run_research_row
#: 的 fields 一致——IF-09 批内 schema 恒齐纪律)。
_V4_RESEARCH_ZERO: dict[str, Any] = {
    "research_no_finalize": 0, "research_doc_words": 0, "research_turns": 0,
    "research_experiments": 0, "research_episodes": 0, "research_error": 0,
    "research_doc_overflow": 0, "research_valid_continuations": 0,
    "research_degraded": 0, "research_replay_drift": 0, "research_overlong": 0,
    "research_r_raw": 0.0, "research_warmup_active": 0,
}

logger = logging.getLogger(__file__)

_STATE_LOCK = threading.Lock()
_STATE: dict[str, Any] = {}


def _as_int(value: Any) -> int | None:
    """V1/TQ 可能给 NonTensorData 包装——防御性解包,不信任传输类型。"""
    value = getattr(value, "data", value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _per_worker(cap: int, workers: int) -> int:
    """配置里的并发上限是全局的;N 个 worker 进程各拿一份。"""
    return max(1, math.ceil(cap / max(1, workers)))


def _agent_loop_worker_index() -> int | None:
    """从 verl 的命名 actor 取稳定 worker index；非 Ray 调用保留池默认策略。"""
    try:
        import ray

        if not ray.is_initialized():
            return None
        name = ray.get_runtime_context().get_actor_name() or ""
    except Exception:  # noqa: BLE001——身份不可用时只降级调度，不影响机制
        return None
    prefix = "agent_loop_worker_"
    if not name.startswith(prefix):
        return None
    try:
        return int(name[len(prefix):].split("_", 1)[0])
    except ValueError:
        return None


def _load_doc_table(path: str) -> dict[str, str]:
    """静态特权文档表:jsonl,每行 {"task_id": ..., "doc": ...}。
    空 doc 的行保留为空串(降级可见,不静默剔除)。"""
    table: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            table[str(row["task_id"])] = str(row.get("doc") or "")
    if not table:
        raise ValueError(f"doc table {path} is empty")
    return table


def _shared_state(tokenizer) -> dict[str, Any]:
    """进程级单例:config、模板、env factory、doc 表。系统性配置错误在启动时死。"""
    with _STATE_LOCK:
        if "config" in _STATE:
            return _STATE
        config_path = os.environ["TRACE_CONFIG"]
        run_dir = Path(os.environ["TRACE_RUN_DIR"])
        config: Config = load_config(config_path)
        from ..core import protocol

        protocol.configure_episode_caps(
            config.episode.max_env_steps, config.episode.max_assistant_turns
        )
        workers = int(os.environ.get("TRACE_AGENT_WORKERS", config.train.agent_num_workers))
        doc_table: dict[str, str] = {}
        gate_tracker = None
        researcher_client = None
        if config.steering.enabled:
            if config.steering.doc_source == "static":
                doc_table = _load_doc_table(config.steering.doc_table)
                logger.info("steering doc table loaded: %d tasks", len(doc_table))
            elif config.research_rl.enabled:
                # v4 自体研究员:researcher chat 走本机引擎,无 external client
                # (roles.researcher 已被校验豁免,不得在此求值)。
                logger.info("v4 self-researcher: engine-backed researcher, no external client")
            else:
                role = config.role("researcher")
                # 配置并发是全局上限，按 agent worker 分摊；绝不按 n=8 副本放大。
                role.max_concurrency = _per_worker(role.max_concurrency, workers)
                pid = os.getpid()
                tape = (
                    TapeWriter(run_dir / "tape" / f"researcher.{pid}.jsonl")
                    if config.run.tape else None
                )
                worker_index = _agent_loop_worker_index()
                researcher_client = make_client(
                    "researcher",
                    role,
                    tape=tape,
                    pool_start_index=worker_index,
                )
                logger.info(
                    "dynamic researcher ready: model=%s concurrency/worker=%d "
                    "worker_index=%s pool_start=%s",
                    role.model,
                    role.max_concurrency,
                    worker_index,
                    None if worker_index is None else worker_index % max(
                        1, len([e for e in role.endpoint.split(",") if e.strip()])
                    ),
                )
            if config.steering.gate_adaptive:
                from ..steering.gate import GateThresholdTracker
                from ..steering.ledger import D_HIST_BINS

                gate_tracker = GateThresholdTracker(
                    theta_init=config.steering.theta_d,
                    target_rate=config.steering.gate_target_rate,
                    ema_beta=config.steering.gate_ema_beta,
                    d_hist_max=config.steering.c * config.steering.c,
                    bins=D_HIST_BINS,
                )
                logger.info(
                    "gate adaptive tracking on: q=%.2f beta=%.2f theta_init=%.4f",
                    config.steering.gate_target_rate,
                    config.steering.gate_ema_beta,
                    config.steering.theta_d,
                )
        _STATE.update(
            {
                "config": config,
                "run_dir": run_dir,
                "delta": TemplateDelta(tokenizer),
                "factory": make_env_factory(config, workers=workers),
                "doc_table": doc_table,
                "gate_tracker": gate_tracker,
                "archive": ResearchArchive(run_dir),
                "researcher_client": researcher_client,
                # 同一 worker/event loop 内按(step,task)共享一次 producer。
                "research_tasks": {},
                # v4 自体研究员:worker 级 (step,task)→TaskCoord 注册表
                # (新 step 换桶回收;selfres.CoordRegistry)
                "v4_registry": CoordRegistry(),
                "executor_model_name": Path(config.train.model_path).name
                or "the policy under training",
            }
        )
        return _STATE


class TraceAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        state = _shared_state(self.tokenizer)
        self.cfg: Config = state["config"]
        self.run_dir: Path = state["run_dir"]
        self.delta: TemplateDelta = state["delta"]
        self.factory: EnvFactory = state["factory"]
        self.doc_table: dict[str, str] = state["doc_table"]
        self.gate_tracker = state["gate_tracker"]
        self.archive: ResearchArchive = state["archive"]
        self.researcher_client = state["researcher_client"]
        self.research_tasks: dict = state["research_tasks"]
        self._v4_registry: CoordRegistry = state["v4_registry"]
        self.executor_model_name: str = state["executor_model_name"]

    # -------------------------------------------------------------- research

    def _narrate_fn(self, route: str):
        """叙述始终复用当前 rollout engine，兼容 SGLang EngineGenerate 面。"""
        return engine_narrate_fn(self.server_manager, self.tokenizer, request_id=route)

    def _researcher_chat(self, task_id: str):
        client = self.researcher_client
        rcfg = self.cfg.research

        async def chat(messages, tools):
            return await client.chat(
                messages,
                tools=tools,
                max_tokens=rcfg.researcher_max_tokens,
                route_key=task_id,
            )

        return chat

    async def _research(
        self, adapter, global_step: int | None, route: str
    ) -> tuple[str, dict[str, Any]]:
        """在线 producer：任务级失败返回空文档；会话/实验/错误完整归档。"""
        rcfg = self.cfg.research
        router: ToolRouter | None = None
        session: ResearchSession | None = None
        result: SessionResult | None = None
        error: str | None = None
        try:
            router = ToolRouter(
                adapter=adapter,
                research=rcfg,
                narration=self.cfg.narration,
                # 每个实验 episode 独立路由；均使用当前 step 冻结的 rollout policy。
                generate_factory=lambda: EngineGenerate(
                    self.server_manager,
                    self.tokenizer,
                    request_id=f"{route}:research:{uuid4().hex[:6]}",
                ),
                tokenizer=self.tokenizer,
                delta=self.delta,
                episode_response_budget=self.rollout_config.response_length,
                narrate_fn=self._narrate_fn(route),
            )
            session = ResearchSession(
                researcher_chat=self._researcher_chat(adapter.task_id),
                toolproto=make_tool_protocol(rcfg.tool_protocol),
                router=router,
                system_prompt=build_researcher_system(
                    task_brief=adapter.task_brief(),
                    executor_model=self.executor_model_name,
                    n_exp=rcfg.n_exp,
                    doc_max_words=rcfg.doc_max_words,
                    gt_text=None,
                ),
                max_turns=rcfg.max_turns,
                wrapup_buffer=rcfg.wrapup_buffer,
            )
            result = await asyncio.wait_for(session.run(), timeout=rcfg.session_timeout_s)
            error = result.error
        except asyncio.TimeoutError:
            error = f"research session exceeded {rcfg.session_timeout_s:.0f}s"
            if session is not None:
                result = session.snapshot(error=error)
        except Exception as err:  # noqa: BLE001——仅任务级 producer 失败可降级
            error = f"{type(err).__name__}: {err}"
        if error:
            logger.warning("research degraded for %s: %s", adapter.task_id, error)

        records = list(router.state.episodes.values()) if router is not None else []
        actual_policy_version: int | None = None
        try:
            actual_policy_version = research_policy_version(records)
            if (
                result is not None
                and (result.experiments_used > 0 or bool(result.doc))
                and actual_policy_version is None
            ):
                raise RuntimeError(
                    "research output has no rollout policy version evidence"
                )
        except Exception as err:  # noqa: BLE001——版本不可信则文档不可入SPS
            version_error = f"policy version {type(err).__name__}: {err}"
            error = f"{error}; {version_error}" if error else version_error
            logger.warning("research version invalid for %s: %s", adapter.task_id, err)

        archive_path = ""
        try:
            path = await asyncio.to_thread(
                self.archive.save,
                task_id=adapter.task_id,
                result=result,
                records=records,
                policy_version=actual_policy_version,
                global_step=global_step,
                extra_meta={
                    "researcher_model": self.researcher_client.model,
                    "tool_protocol": rcfg.tool_protocol,
                    "narration_backend": self.cfg.narration.backend,
                    "gt": rcfg.gt,
                    "gt_injected": False,
                    "error": error,
                },
            )
            archive_path = str(path.relative_to(self.run_dir))
        except Exception as err:  # noqa: BLE001——无 lineage 的文档绝不进入 SPS
            logger.exception("research archive write failed for %s", adapter.task_id)
            archive_error = f"archive {type(err).__name__}: {err}"
            error = f"{error}; {archive_error}" if error else archive_error

        doc = (
            (result.doc or "")
            if result is not None and archive_path and error is None
            else ""
        )
        attempts = result.last_exp_attempts if result is not None else 0
        fields: dict[str, Any] = {
            "research_error": int(error is not None),
            "research_experiments": result.experiments_used if result else 0,
            "research_episodes": result.episodes_consumed if result else 0,
            "research_turns": result.turns_used if result else 0,
            "research_final_success": (
                result.last_exp_successes / attempts if result is not None and attempts else 0.0
            ),
            "research_doc_words": len(doc.split()) if doc else 0,
            "research_doc_missing": int(not doc),
            "research_policy_version": (
                actual_policy_version if actual_policy_version is not None else -1
            ),
            "research_archive": archive_path,
        }
        logger.info(
            "research %s step=%s exp=%s eps=%s doc_words=%s%s archive=%s",
            adapter.task_id,
            global_step,
            fields["research_experiments"],
            fields["research_episodes"],
            fields["research_doc_words"],
            " ERROR" if error else "",
            archive_path,
        )
        return doc, fields

    # --------------------------------------------------------------- prompts

    def _teacher_prompt_ids(self, adapter, doc: str, student_prompt_ids, initial_obs: str):
        """特权前缀(SPS verify 用):scaffold + "## Task guidance" + doc,文档在
        system prompt 静态最前部 ⇒ 教师侧 KV 跨轮可复用(规范 §2.5)。
        超出引擎上下文时返回 None(降级为纯 bare,记账可见)。"""
        from ..core import protocol

        teacher_prompt_ids = render_ids(
            self.tokenizer,
            [
                {
                    "role": "system",
                    "content": inject_guidance(adapter.executor_scaffold(), doc),
                },
                {"role": "user", "content": protocol.truncate_obs(initial_obs)},
            ],
            True,
            tools=adapter.tools(),
        )
        max_len = self.cfg.train.rollout_max_model_len
        if len(teacher_prompt_ids) + self.rollout_config.response_length + 8 > max_len:
            logger.warning(
                "doc for %s overflows teacher context (%d prompt tokens) — bare fallback",
                adapter.task_id,
                len(teacher_prompt_ids),
            )
            return None
        # 特权/裸前缀必须终止于同一生成上下文(逐 token 对齐的前提)。
        # 机制红线用显式异常,不用 assert(python -O 会剥掉 assert;审阅 IF-15)。
        n = len(self.delta.gen_prefix) + 2
        if teacher_prompt_ids[-n:] != student_prompt_ids[-n:]:
            raise RuntimeError(
                "teacher/bare prompts diverge near the generation prefix"
            )
        return teacher_prompt_ids

    # ------------------------------------------------------- v4 research 行

    async def _research_row_v4(
        self, adapter, global_step, route, kwargs, coord, metrics, extra_info
    ) -> AgentLoopOutput:
        """v4 research 行:自体会话 → doc → 前缀续写评估 → 登记 → 训练行
        (docs/pgs-self-researcher-design.md §0 尾段;编排在 train/selfres.py)。"""
        rrl = self.cfg.research_rl
        rcfg = self.cfg.research
        sid = _as_int(kwargs.get("session_id"))
        tag = f"{route}:res{sid}"
        response_length = self.rollout_config.response_length

        async def run_session():
            chat = EngineResearcherChat(
                generate=EngineGenerate(
                    self.server_manager, self.tokenizer, request_id=tag
                ),
                tokenizer=self.tokenizer,
                delta=self.delta,
                temperature=rrl.researcher_temperature,
                max_tokens_per_turn=rcfg.researcher_max_tokens,
                max_total_tokens=max(
                    4096, self.cfg.train.rollout_max_model_len - 2048
                ),
            )
            router = ToolRouter(
                adapter=adapter,
                research=rcfg,
                narration=self.cfg.narration,
                generate_factory=lambda: EngineGenerate(
                    self.server_manager, self.tokenizer,
                    request_id=f"{tag}:exp:{uuid4().hex[:6]}",
                ),
                tokenizer=self.tokenizer,
                delta=self.delta,
                episode_response_budget=response_length,
                narrate_fn=self._narrate_fn(tag),
            )
            session = ResearchSession(
                researcher_chat=chat,
                toolproto=make_tool_protocol(rcfg.tool_protocol),
                router=router,
                system_prompt=build_researcher_system(
                    task_brief=adapter.task_brief(),
                    executor_model=self.executor_model_name,
                    n_exp=rcfg.n_exp,
                    doc_max_words=rcfg.doc_max_words,
                    gt_text=None,
                ),
                max_turns=rcfg.max_turns,
                wrapup_buffer=rcfg.wrapup_buffer,
            )
            try:
                sr = await asyncio.wait_for(
                    session.run(), timeout=rcfg.session_timeout_s
                )
            except asyncio.TimeoutError:
                sr = session.snapshot(
                    error=f"research session exceeded {rcfg.session_timeout_s:.0f}s"
                )
            return sr, chat

        # 教师前缀构造(对齐校验)复用 PGS 路径;student 前缀与 _make_generate
        # 逐字节同参。
        from ..core import protocol

        student_prompt_ids = render_ids(
            self.tokenizer,
            [
                {"role": "system",
                 "content": inject_guidance(adapter.executor_scaffold(), "")},
                {"role": "user",
                 "content": protocol.truncate_obs(adapter.initial_obs)},
            ],
            True,
            tools=adapter.tools(),
        )
        t0 = time.time()
        rowres = await run_research_row(
            session_id=sid,
            global_step=int(global_step),
            coord=coord,
            run_session=run_session,
            teacher_ids_for_doc=lambda d: self._teacher_prompt_ids(
                adapter, d, student_prompt_ids, adapter.initial_obs
            ),
            make_generate=lambda tag2: EngineGenerate(
                self.server_manager, self.tokenizer, request_id=f"{tag}:cont"
            ),
            adapter=adapter,
            tokenizer=self.tokenizer,
            delta=self.delta,
            response_budget=response_length,
            continuation_samples=rrl.continuation_samples,
            continuation_temperature=rrl.continuation_temperature,
            research_rl_weight=rrl.research_rl_weight,
            research_warmup_steps=rrl.research_warmup_steps,
            research_max_sample_tokens=rrl.research_max_sample_tokens,
        )
        metrics.generate_sequences += time.time() - t0

        from ..steering.ledger import SteeringLedger

        steer_zero = SteeringLedger(
            d_hist_max=self.cfg.steering.c * self.cfg.steering.c
        ).finalize()
        fields: dict[str, Any] = {
            "task_id": adapter.task_id,
            "won": False,
            "turn_spans": [],
            "outcome": "research",
            "env_steps": 0,
            "invalid_format": 0,
            "truncated": bool(rowres.fields.get("research_overlong")),
            **steer_zero,
            "steer_w_beh_mode": self.cfg.steering.w_beh_mode,
            "steer_log_ratio_cap": 2.0 * self.cfg.steering.eta_max * self.cfg.steering.c,
            "steer_theta_d": 0.0,
            "doc_missing": 0,
            "doc_overflow": 0,
            "steer_arm": 0,
            "sample_role": "research",
            "role_research": 1,
            **rowres.fields,
        }
        output = AgentLoopOutput(
            prompt_ids=rowres.prompt_ids,
            response_ids=rowres.response_ids,
            response_mask=rowres.response_mask,
            reward_score=rowres.reward_score,
            num_turns=2 * int(rowres.fields.get("research_turns", 0)) + 1,
            metrics=metrics,
            extra_fields=dict(fields),
        )
        output.extra_fields["reward_extra_info"] = {
            k: v for k, v in fields.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        sr = rowres.session_result
        try:
            await asyncio.to_thread(
                self._dump_rollout,
                step=global_step,
                validate=False,
                payload={
                    "task_id": adapter.task_id,
                    "split": extra_info.get("split", ""),
                    "validate": False,
                    "global_step": global_step,
                    "outcome": "research",
                    "sample_role": "research",
                    "session_id": sid,
                    "research": dict(rowres.fields),
                    "doc": (sr.doc or "") if sr is not None else "",
                    "messages": list(sr.messages) if sr is not None else [],
                    "continuations": [
                        {"won": o.won, "error": o.error, "env_steps": o.env_steps,
                         "gen_tokens": o.gen_tokens, "truncated": o.truncated}
                        for o in rowres.outcomes
                    ],
                    "response_token_len": len(rowres.response_ids),
                    "ts": time.time(),
                },
            )
        except Exception:  # noqa: BLE001 — dump 永不搞挂样本
            logger.exception("research row dump failed for %s", adapter.task_id)
        return output

    # ------------------------------------------------------------------ run

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        metrics = AgentLoopMetrics()
        validate = bool(kwargs.get("validate", False))
        global_step = _as_int(kwargs.get("global_step"))
        extra_info = dict(kwargs["extra_info"])
        adapter = self.factory.adapter_from_item(extra_info)
        response_length = self.rollout_config.response_length
        if (
            self.cfg.steering.enabled
            and self.cfg.steering.doc_source == "researcher"
            and not validate
            and global_step is None
        ):
            raise RuntimeError(
                "dynamic researcher requires a non-null global_step for (step,task) dedupe"
            )
        # 路由:per-episode 唯一后缀(长尾任务不锁死单 replica),轮内共享
        # 一个 key ⇒ 增长前缀的 KV 存活。
        route = f"task:{global_step}:{adapter.task_id}"

        # ---- v4 自体研究员分流(docs/pgs-self-researcher-design.md §0)----
        # research 行完全独立路径;executor 行注入 v4 上下文后重入本方法
        # (现有 v3.1/v3.2 组装路径零复制)。validate 照常 bare greedy。
        if (
            self.cfg.research_rl.enabled
            and not validate
            and "_v4_role" not in kwargs
        ):
            role = role_of(
                _as_int(kwargs.get("session_id")),
                self.cfg.train.rollout_n,
                self.cfg.steering.steered_fraction,
                self.cfg.research_rl.research_n,
            )
            rrl = self.cfg.research_rl
            bare_expected = self.cfg.train.rollout_n - steered_arm_count(
                self.cfg.train.rollout_n, self.cfg.steering.steered_fraction
            )
            coord = await self._v4_registry.get(
                int(global_step), adapter.task_id,
                lambda: TaskCoord(
                    bare_expected=bare_expected,
                    research_expected=rrl.research_n,
                    quantiles=rrl.prefix_quantiles,
                    tokenizer=self.tokenizer,
                ),
            )
            if role == ROLE_RESEARCH:
                return await self._research_row_v4(
                    adapter, global_step, route, kwargs, coord, metrics, extra_info
                )
            kwargs["_v4_role"], kwargs["_v4_coord"] = role, coord
            return await self.run(sampling_params, **kwargs)

        steer_ctx = None
        research_fields: dict[str, Any] = {}
        v4_coord = kwargs.get("_v4_coord")
        if v4_coord is not None:
            # v4 executor 行:bare 立即起跑(零依赖),steered 等 argmax doc;
            # research_fields 用零占位保批内 schema 恒齐(键集与 research 行
            # 的 fields 一致)。
            steered_arm = kwargs["_v4_role"] == ROLE_STEERED
            if steered_arm:
                doc, _r_star = await v4_coord.argmax_future
            else:
                doc = ""
            research_fields = dict(_V4_RESEARCH_ZERO)
            research_fields["role_research"] = 0
            research_fields["sample_role"] = kwargs["_v4_role"]
            steer_ctx = {
                "doc": doc if steered_arm else "",
                "doc_missing": int(not doc) if steered_arm else 0,
                "steered_arm": int(steered_arm),
            }
        elif self.cfg.steering.enabled and not validate:
            # 分臂采样(v3.1,用户裁定 2026-08-06):session_id 由 verl TQ 层
            # 逐副本传入,round(n×fraction) 个副本走 steered 臂,其余为纯 π
            # 的 bare 臂——同组进 GRPO,恢复 bare 逐步读数与原生负梯度,
            # 混合组 baseline 把"教师能成/policy 失败"的差距转成对比信号。
            steered_arm = is_steered_arm(
                _as_int(kwargs.get("session_id")),
                self.cfg.train.rollout_n,
                self.cfg.steering.steered_fraction,
            )
            if self.gate_tracker is not None:
                # 率控自适应:step 边界用上一 step 的 D 直方图 EMA 更新 θ_D
                # (步内冻结;滞后一步)。
                self.gate_tracker.maybe_step_update(global_step)
            if self.cfg.steering.doc_source == "researcher":
                # student 启动前 await 文档;同 step 的 n=8 全部等待同一共享任务。
                # bare 臂也 await:producer 按 (step,task) 只跑一次,bare 副本
                # 等它只为携带同一份 research_fields(IF-09 批内 schema 恒齐),
                # 文档本身不注入。
                t_research = time.time()
                doc, research_fields = await shared_task(
                    self.research_tasks,
                    (global_step, adapter.task_id),
                    lambda: self._research(adapter, global_step, route),
                )
                metrics.tool_calls += time.time() - t_research
            else:
                doc = self.doc_table.get(adapter.task_id, "")
            steer_ctx = {
                # bare 臂强制空 doc ⇒ _make_generate 走纯 EngineGenerate,
                # 零 verify、零倾斜、空账(W≡1)。doc_missing 保持任务级观测
                # 语义(两臂同值),臂身份走独立字段。
                "doc": doc if steered_arm else "",
                "doc_missing": int(not doc),
                "steered_arm": int(steered_arm),
            }

        t0 = time.time()
        result = await self._student_rollout(
            adapter, sampling_params, response_length, route, steer_ctx
        )
        if v4_coord is not None and kwargs["_v4_role"] == ROLE_BARE:
            # v4:bare 行终态登记进前缀池(第 bare_expected 条 resolve)。
            traj = result.trajectory
            v4_coord.register_bare({
                "won": traj.won,
                "response_ids": list(result.tokens.response_ids),
                "response_mask": list(result.tokens.response_mask),
                "turn_spans": [list(x) for x in result.tokens.turn_spans],
                "steps": [
                    {"action_norm": s.action_norm, "observation": s.observation,
                     "acted": s.acted}
                    for s in traj.steps
                ],
            })
        if (
            v4_coord is None
            and self.cfg.steering.enabled
            and self.cfg.steering.doc_source == "researcher"
            and not validate
            and not research_fields.get("research_error")
            and not research_fields.get("research_doc_missing")
        ):
            version = _as_int(research_fields.get("research_policy_version"))
            if version is None or version < 0:
                raise RuntimeError(
                    "dynamic researcher produced a document without policy version evidence"
                )
            assert_rollout_policy_version(
                result.gen_extra, version, label="student rollout"
            )
        metrics.generate_sequences += time.time() - t0

        traj = result.trajectory
        env_steps = sum(1 for s in traj.steps if s.acted)
        tokens = result.tokens

        steer_fields: dict[str, Any] = {}
        if steer_ctx is not None:
            from ..steering.ledger import SteeringLedger

            steer_fields["doc_missing"] = steer_ctx["doc_missing"]
            steer_fields["doc_overflow"] = steer_ctx.get("doc_overflow", 0)
            # 分臂身份(v3.1)进账与曲线:steer_arm ∈ {0,1};聚合端
            # (metric_utils)据此产出 won_bare/won_steered 条件率与 δ gap。
            steer_fields["steer_arm"] = steer_ctx.get("steered_arm", 1)
            state = steer_ctx.get("state")
            # 审阅 IF-09:steering run 的每条训练轨迹都必须携带同一套 steer
            # 字段(缺 doc/超上下文的轨迹用空账 = 全 1 权重)。TQ 的
            # list_of_dict_to_tensordict 以首行 keys 为准,混合批里 schema
            # 不齐会静默丢掉整批 rollout_is_weights——已倾斜轨迹未经校正训练。
            ledger = (
                state.ledger if state is not None
                else SteeringLedger(d_hist_max=self.cfg.steering.c * self.cfg.steering.c)
            )
            steer_fields.update(ledger.finalize())
            steer_fields["steer_w_beh_mode"] = self.cfg.steering.w_beh_mode
            steer_fields["steer_log_ratio_cap"] = (
                2.0 * self.cfg.steering.eta_max * self.cfg.steering.c
            )
            if state is not None and self.gate_tracker is not None:
                # 本轨迹的 D 观测并入 worker 级追踪器(下一 step 边界生效)
                self.gate_tracker.observe(state.ledger.d_hist, state.ledger.d_count)
            # 本轨迹实际使用的 θ_D(自适应轨迹审计;swanlab 曲线自动收)。
            # always_on 档 θ=−inf 无语义,写 0.0 哨兵防 inf 污染数值曲线。
            theta_used = (
                state.gate.theta_d if state is not None
                else (self.gate_tracker.theta if self.gate_tracker is not None
                      else self.cfg.steering.theta_d)
            )
            steer_fields["steer_theta_d"] = (
                theta_used if math.isfinite(theta_used) else 0.0
            )
            if state is not None:
                alarm = state.gate.rate_alarm()
                if alarm:
                    logger.warning("steering gate alarm for %s: %s", adapter.task_id, alarm)
                    steer_fields["steer_gate_alarm"] = 1

        output = AgentLoopOutput(
            prompt_ids=tokens.prompt_ids,
            response_ids=tokens.response_ids,
            response_mask=tokens.response_mask,
            reward_score=1.0 if traj.won else 0.0,
            num_turns=2 * result.assistant_turns + 1,
            metrics=metrics,
            extra_fields={
                "task_id": adapter.task_id,
                "won": traj.won,
                "turn_spans": [list(x) for x in result.tokens.turn_spans],
                "outcome": traj.outcome,
                "env_steps": env_steps,
                "invalid_format": result.invalid_format,
                "truncated": result.truncated,
                **result.gen_extra,
                **steer_fields,
                **research_fields,
            },
        )
        # 曲线通道:数值标量 → swanlab(train: rollout_extra/<key>/mean)。
        output.extra_fields["reward_extra_info"] = {
            "won": int(traj.won),
            "env_steps": env_steps,
            "invalid_format": int(result.invalid_format),
            "truncated": int(result.truncated),
            **{
                k: v
                for k, v in {**steer_fields, **research_fields}.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            },
        }

        try:
            await asyncio.to_thread(
                self._dump_rollout,
                step=global_step,
                validate=validate,
                payload={
                    "task_id": adapter.task_id,
                    "split": extra_info.get("split", ""),
                    "validate": validate,
                    "global_step": global_step,
                    "outcome": traj.outcome,
                    "won": traj.won,
                    "env_steps": env_steps,
                    "assistant_turns": result.assistant_turns,
                    "invalid_format": result.invalid_format,
                    "truncated": result.truncated,
                    "steering": {
                        k: v for k, v in steer_fields.items() if k != "ledger_fields"
                    },
                    "research": dict(research_fields),
                    "policy_version": {
                        "min_global_steps": result.gen_extra.get("min_global_steps"),
                        "max_global_steps": result.gen_extra.get("max_global_steps"),
                    },
                    "trajectory": traj.to_dict(),
                    "response_token_len": len(result.tokens.response_ids),
                    "response_token_hash": response_token_hash(result.tokens.response_ids),
                    # viz token 级对齐通道(2026-08-04 实验侧规格;~10KB/轨迹)
                    "response_token_ids": list(result.tokens.response_ids),
                    "turn_token_spans": result.tokens.turn_spans,
                    "ts": time.time(),
                },
            )
        except Exception:  # noqa: BLE001 — dump 永不搞挂样本
            logger.exception("rollout dump failed for %s", adapter.task_id)
        return output

    # ---------------------------------------------------------------- rollout

    def _make_generate(self, adapter, epi_route: str, steer_ctx: dict[str, Any] | None):
        """学生 episode 的生成通道:纯 EngineGenerate,或(有 doc 时)SPS 包装。

        每次调用新建 TrajectorySteerState(轨迹级生命周期;retry 也拿新状态),
        并把 state 挂回 steer_ctx 供 run() 取记账。
        """
        engine = EngineGenerate(self.server_manager, self.tokenizer, request_id=epi_route)
        if not steer_ctx or not steer_ctx.get("doc"):
            return engine
        from ..core import protocol
        from ..steering import SteeringGate, TrajectoryBudget, TurnRejectCap
        from ..steering.hook import SpsGenerate, TrajectorySteerState

        # 与 episode 内部的构造逐字节同参(episode 会断言 env 初始观察与
        # dataset 一致,不一致早退 ENV_ERROR,不会触发生成)。
        student_prompt_ids = render_ids(
            self.tokenizer,
            [
                {"role": "system", "content": inject_guidance(adapter.executor_scaffold(), "")},
                {"role": "user", "content": protocol.truncate_obs(adapter.initial_obs)},
            ],
            True,
            tools=adapter.tools(),
        )
        teacher_prompt_ids = self._teacher_prompt_ids(
            adapter, steer_ctx["doc"], student_prompt_ids, adapter.initial_obs
        )
        if teacher_prompt_ids is None:
            steer_ctx["doc_overflow"] = 1
            return engine  # doc 超上下文:降级纯 bare,记账可见
        st_cfg = self.cfg.steering
        from ..steering import SteeringLedger

        if st_cfg.gate_mode == "always_on":
            # maxopen 档(v3.2):无门判定——θ_D=−inf 使 decide 恒 base_open,
            # 探针已被校验强制 ε=0,率告警带置 (0,1) 关闭语义。gate 代码零改动,
            # θ 标定/率控自适应子系统对该档整体退役(tracker 经 gate_adaptive=
            # false 已不构造)。
            theta_now = float("-inf")
            rate_bounds = (0.0, 1.0)
        else:
            theta_now = (
                self.gate_tracker.theta if self.gate_tracker is not None else st_cfg.theta_d
            )
            rate_bounds = tuple(st_cfg.gate_rate_bounds)
        state = TrajectorySteerState(
            gate=SteeringGate(
                theta_d=theta_now,
                eta_max=st_cfg.eta_max,
                epsilon=st_cfg.epsilon,
                rate_bounds=rate_bounds,
            ),
            budget=TrajectoryBudget(
                b_max=st_cfg.b_max, c=st_cfg.c,
                pacing=st_cfg.budget_pacing,
                u0=st_cfg.budget_u0,
                profile_power=st_cfg.budget_profile_power,
                # T_max = 环境硬配置的会话轮上限(零任务先验,通用)
                t_max=self.cfg.episode.max_assistant_turns,
            ),
            reject_cap=TurnRejectCap(r_max=st_cfg.r_max),
            # IF-12:D 直方图上界随 run 的 c 派生(=c²),坐标系随账落盘
            ledger=SteeringLedger(d_hist_max=st_cfg.c * st_cfg.c),
        )
        steer_ctx["state"] = state
        return SpsGenerate(
            draft=engine,
            verify=engine,           # 同引擎同路由:两个前缀的 KV 同 replica
            teacher_prefix_ids=teacher_prompt_ids,
            student_prompt_ids=student_prompt_ids,
            state=state,
            tokenizer=self.tokenizer,
            topk=st_cfg.topk,
            c=st_cfg.c,
            obs_topk=st_cfg.obs_topk,
            # IF-06:引擎的自然停止口径(qwen 系 eos = <|im_end|>);重采出
            # 停止符时钩子按同一口径终止本轮。
            stop_token_ids=frozenset(
                t for t in (self.tokenizer.eos_token_id,) if t is not None
            ),
        )

    async def _student_rollout(
        self,
        adapter,
        sampling_params: dict[str, Any],
        response_length: int,
        route: str,
        steer_ctx: dict[str, Any] | None = None,
    ) -> EpisodeResult:
        """学生 episode(steer_ctx 有 doc 时经 SPS 保尾核倾斜采样)。
        per-episode 唯一路由后缀;基建错误重试一次(新 steering 状态)。"""
        epi_route = f"{route}:s{uuid4().hex[:6]}"
        result = await run_episode(
            adapter=adapter,
            generate=self._make_generate(adapter, epi_route, steer_ctx),
            tokenizer=self.tokenizer,
            delta=self.delta,
            guidance="",
            episode_id="student",
            response_budget=response_length,
            sampling_params=sampling_params,
        )
        # 审阅 IF-05:基础设施故障(ENV_ERROR)一律整 episode 作废——包括已有
        # 部分响应的情形。旧行为只在响应为空时重试,残缺轨迹会以 reward 0 进
        # GRPO,把基建故障当环境失败学习。重试拿全新路由与全新 steering 状态
        # (_make_generate 每次重建并覆盖 steer_ctx["state"],残账不外泄)。
        if result.trajectory.outcome == OUTCOME_ENV_ERROR:
            logger.warning(
                "student episode infra error for %s (%s; %d response tokens discarded); "
                "retrying once with fresh state",
                adapter.task_id,
                result.trajectory.error,
                len(result.tokens.response_ids),
            )
            retry_route = f"{route}:r{uuid4().hex[:6]}"
            result = await run_episode(
                adapter=adapter,
                generate=self._make_generate(adapter, retry_route, steer_ctx),
                tokenizer=self.tokenizer,
                delta=self.delta,
                guidance="",
                episode_id="student-retry",
                response_budget=response_length,
                sampling_params=sampling_params,
            )
        if result.trajectory.outcome == OUTCOME_ENV_ERROR:
            # 二次仍是基建故障:显式丢弃(verl 的 prompt failure path 接手),
            # 绝不给残缺轨迹赋 0 reward 入训。
            raise RuntimeError(
                f"student episode failed on infrastructure twice for {adapter.task_id}: "
                f"{result.trajectory.error}"
            )
        return result

    # ------------------------------------------------------------------ dump

    def _dump_rollout(self, *, step: int | None, validate: bool, payload: dict[str, Any]) -> None:
        tag = f"step_{step:05d}" if step is not None else "step_unknown"
        if validate:
            tag += "_val"
        path = (
            self.run_dir / "rollouts" / tag
            / f"{payload['task_id'].replace('/', '__')}_{uuid4().hex[:8]}.json"
        )
        from ..common.jsonl import write_json

        write_json(path, payload)
