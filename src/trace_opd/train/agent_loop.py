"""TraceAgentLoop(PGS)—— 每训练样本的 rollout 机制(规范 docs/pgs-final-spec.md)。

每个样本(= 一个任务):
  1. STUDENT ROLLOUT:裸 on-policy episode(steering.enabled 时经 SPS 保尾核
     倾斜采样——P2 接入 episode 钩子;损失中无教师,教师只出现在采样端)。
  2. 记账:逐 token steering ledger 随 extra_fields 进训练端(W_beh 数据源)。
  3. 观察:每样本 dump 完整轨迹到 <run_dir>/rollouts/step_<N>[_val]/。

特权文档来源 = 静态 doc 表(steering.doc_table:task_id→doc 的 jsonl;
离线制备,无在线 researcher)。缺 doc 的任务自然退化为纯 bare(记账可见)。

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

from ..common.config import Config, load_config
from ..common.hashing import response_token_hash
from ..core.episode import EpisodeResult, run_episode
from ..core.template_delta import TemplateDelta, render_ids
from ..core.types import OUTCOME_ENV_ERROR, inject_guidance
from ..envs.alfworld.adapter import AlfworldEnvFactory
from .backends import EngineGenerate

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
        config.alfworld.spawn_concurrency = _per_worker(
            config.alfworld.spawn_concurrency, workers
        )
        doc_table: dict[str, str] = {}
        if config.steering.enabled:
            doc_table = _load_doc_table(config.steering.doc_table)
            logger.info("steering doc table loaded: %d tasks", len(doc_table))
        _STATE.update(
            {
                "config": config,
                "run_dir": run_dir,
                "delta": TemplateDelta(tokenizer),
                "factory": AlfworldEnvFactory(config.alfworld),
                "doc_table": doc_table,
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
        self.factory: AlfworldEnvFactory = state["factory"]
        self.doc_table: dict[str, str] = state["doc_table"]

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
        n = len(self.delta.gen_prefix) + 2
        assert teacher_prompt_ids[-n:] == student_prompt_ids[-n:], (
            "teacher/bare prompts diverge near the generation prefix"
        )
        return teacher_prompt_ids

    # ------------------------------------------------------------------ run

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        metrics = AgentLoopMetrics()
        validate = bool(kwargs.get("validate", False))
        global_step = _as_int(kwargs.get("global_step"))
        extra_info = dict(kwargs["extra_info"])
        adapter = self.factory.adapter_from_item(extra_info)
        response_length = self.rollout_config.response_length
        # 路由:per-episode 唯一后缀(长尾任务不锁死单 replica),轮内共享
        # 一个 key ⇒ 增长前缀的 KV 存活。
        route = f"task:{global_step}:{adapter.task_id}"

        steer_ctx = None
        if self.cfg.steering.enabled and not validate:
            doc = self.doc_table.get(adapter.task_id, "")
            steer_ctx = {"doc": doc, "doc_missing": int(not doc)}

        t0 = time.time()
        result = await self._student_rollout(
            adapter, sampling_params, response_length, route, steer_ctx
        )
        metrics.generate_sequences += time.time() - t0

        traj = result.trajectory
        env_steps = sum(1 for s in traj.steps if s.acted)
        tokens = result.tokens

        steer_fields: dict[str, Any] = {}
        if steer_ctx is not None:
            steer_fields["doc_missing"] = steer_ctx["doc_missing"]
            steer_fields["doc_overflow"] = steer_ctx.get("doc_overflow", 0)
            state = steer_ctx.get("state")
            if state is not None:
                steer_fields.update(state.ledger.finalize())
                # 训练侧展开参数(verl agent_loop_tq 消费):W_beh 口径与
                # 逐 token |log(π/b)| 的核界(2ηc)——越界即记账损坏,fail loud。
                steer_fields["steer_w_beh_mode"] = self.cfg.steering.w_beh_mode
                steer_fields["steer_log_ratio_cap"] = (
                    2.0 * self.cfg.steering.eta_max * self.cfg.steering.c
                )
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
                for k, v in steer_fields.items()
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
                    "trajectory": traj.to_dict(),
                    "response_token_len": len(result.tokens.response_ids),
                    "response_token_hash": response_token_hash(result.tokens.response_ids),
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
        )
        teacher_prompt_ids = self._teacher_prompt_ids(
            adapter, steer_ctx["doc"], student_prompt_ids, adapter.initial_obs
        )
        if teacher_prompt_ids is None:
            steer_ctx["doc_overflow"] = 1
            return engine  # doc 超上下文:降级纯 bare,记账可见
        st_cfg = self.cfg.steering
        state = TrajectorySteerState(
            gate=SteeringGate(
                theta_d=st_cfg.theta_d,
                eta_max=st_cfg.eta_max,
                epsilon=st_cfg.epsilon,
                rate_bounds=tuple(st_cfg.gate_rate_bounds),
            ),
            budget=TrajectoryBudget(b_max=st_cfg.b_max, c=st_cfg.c),
            reject_cap=TurnRejectCap(r_max=st_cfg.r_max),
        )
        steer_ctx["state"] = state
        return SpsGenerate(
            draft=engine,
            verify=engine,           # 同引擎同路由:两个前缀的 KV 同 replica
            teacher_prefix_ids=teacher_prompt_ids,
            student_prompt_len=len(student_prompt_ids),
            state=state,
            tokenizer=self.tokenizer,
            topk=st_cfg.topk,
            c=st_cfg.c,
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
        if result.trajectory.outcome == OUTCOME_ENV_ERROR and not result.tokens.response_ids:
            logger.warning(
                "student episode infra error for %s (%s); retrying once",
                adapter.task_id,
                result.trajectory.error,
            )
            result = await run_episode(
                adapter=adapter,
                generate=self._make_generate(adapter, epi_route, steer_ctx),
                tokenizer=self.tokenizer,
                delta=self.delta,
                guidance="",
                episode_id="student-retry",
                response_budget=response_length,
                sampling_params=sampling_params,
            )
        if result.trajectory.outcome == OUTCOME_ENV_ERROR and not result.tokens.response_ids:
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
