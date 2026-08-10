"""WebArena TaskAdapter (WebArena-Verified corpus, pooled site instances).

Design sources: experiments/bench_env/docs/2026-08-10-webarena-env-proposal.md
(user rulings §9: Verified 数据全替换 / 历史观测占位压缩 + 每轮一行打包 /
Checkpoint-Restore 站点池) + WebServ interaction protocol (deep-read;
``step_browser`` single-tool FC with semantic-ID actions) + the
``webarena-verified`` package contracts (FinalAgentResponse / terminal-only
evaluation, see evalbridge.py).

Protocol surface
----------------
* Single ``step_browser`` tool, one call per turn. Action set is WebServ's
  (click/type/hover/select/clear/key_press/goto_url/back/forward/refresh/
  new_tab/switch_tab/close_tab) with ONE deviation: ``terminate`` carries the
  STRUCTURED verified final response (task_type/status/retrieved_data/
  error_details) instead of a free-text ``answer`` — the evaluator consumes
  exactly these fields, so the model emits them directly (no post-hoc
  translation layer that could distort training signal).
* Rewards are terminal-only and all-or-nothing {0,1} (verified semantics):
  the episode is evaluated ONLY when the model terminates; budget exhaustion
  without terminate scores 0 (mirrors TMax "no submit → 0", proposal §6).
* Observation = WebServ ``format_llm_observation`` text (URL/Title/pruned
  HTML with semantic ids/element inventories/tabs), truncated head+tail at
  ``max_obs_chars``.

历史观测占位压缩(用户裁定 #1)
------------------------------
训练/推理都按「每轮重建上下文」:历史轮的观测替换为 ``OBS_PLACEHOLDER``,
模型自身输出(思考+工具调用)全保留,当前轮观测全量。训练打包 = 每轮一行
样本、共享终局 reward。``build_compressed_messages`` 是该协议的参考实现,
verl 侧(CD-010)按同一规则实现;smoke 的 episode 门直接用它驱动,保证
参考实现被真实执行过。
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from pathlib import Path

from ...common.config import WebArenaCfg
from ...core import protocol
from .browser import BrowserError, BrowserSession, render_start_urls
from .sites import SitePoolError

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- tools

_STATUS_VALUES = [
    "SUCCESS", "ACTION_NOT_ALLOWED_ERROR", "PERMISSION_DENIED_ERROR",
    "NOT_FOUND_ERROR", "DATA_VALIDATION_ERROR", "UNKNOWN_ERROR",
]
_TASK_TYPE_VALUES = ["RETRIEVE", "MUTATE", "NAVIGATE"]

_ACTIONS = [
    "click", "type", "hover", "select", "clear", "key_press", "goto_url",
    "back", "forward", "refresh", "new_tab", "switch_tab", "close_tab",
    "terminate",
]

TOOLS_FC: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "step_browser",
            "description": (
                "Execute an action in the web browser environment and get "
                "the resulting observation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Type of action to perform",
                        "enum": _ACTIONS,
                    },
                    "target": {
                        "type": "string",
                        "description": "Semantic ID of the element to interact with",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to type (for type action)",
                    },
                    "enter": {
                        "type": "boolean",
                        "description": "Whether to press Enter after typing",
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to select (for select action)",
                    },
                    "key": {
                        "type": "string",
                        "description": "Key to press (for key_press action)",
                    },
                    "url": {
                        "type": "string",
                        "description": "URL to navigate to",
                    },
                    "tab_id": {
                        "type": "integer",
                        "description": "Tab ID for tab operations",
                    },
                    "task_type": {
                        "type": "string",
                        "description": (
                            "For terminate: overall type of work performed — "
                            "RETRIEVE (finding/reading data was the objective), "
                            "MUTATE (creating/updating/deleting data or state), "
                            "NAVIGATE (showing a specific page/location)."
                        ),
                        "enum": _TASK_TYPE_VALUES,
                    },
                    "status": {
                        "type": "string",
                        "description": (
                            "For terminate: task outcome. SUCCESS = objective "
                            "fully achieved; NOT_FOUND_ERROR = target entity "
                            "could not be located; ACTION_NOT_ALLOWED_ERROR = "
                            "the platform does not support/allow the request; "
                            "PERMISSION_DENIED_ERROR = current user lacks "
                            "permission; DATA_VALIDATION_ERROR = required "
                            "input missing/invalid; UNKNOWN_ERROR = other."
                        ),
                        "enum": _STATUS_VALUES,
                    },
                    "retrieved_data": {
                        "type": "array",
                        "description": (
                            "For terminate on RETRIEVE tasks: the retrieved "
                            "items (empty array if nothing found; null/omit "
                            "for MUTATE and NAVIGATE). All items must share "
                            "one type; use proper types (numbers for "
                            "amounts/counts, true/false for booleans — not "
                            "strings). For lists of objects, follow the "
                            "format specified in the task instruction."
                        ),
                        "items": {},
                    },
                    "error_details": {
                        "type": "string",
                        "description": (
                            "For terminate with a non-SUCCESS status: concise "
                            "failure reason. Omit when status is SUCCESS."
                        ),
                    },
                },
                "required": ["action"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a web agent operating a browser to complete a task on one or "
    "more websites. On every turn you must make exactly one call to the "
    "`step_browser` tool.\n\n"
    "Observations show the current page as stripped HTML in which "
    "interactive elements carry semantic IDs, followed by inventories of "
    "clickable/input/select elements and the list of open tabs. Interact "
    "with elements by passing their semantic ID as `target`.\n\n"
    "Actions: click, type (text into a field, optionally enter=true), hover, "
    "select (choose dropdown value), clear, key_press, goto_url, back, "
    "forward, refresh, new_tab, switch_tab, close_tab, terminate.\n\n"
    "When the task is complete (or definitely impossible), call terminate "
    "with your final report: `task_type` (RETRIEVE if the objective was "
    "finding/reading information, MUTATE if it was creating/updating/"
    "deleting data or state, NAVIGATE if it was showing a specific page), "
    "`status` (SUCCESS, or the most fitting error status), and — for "
    "RETRIEVE tasks — `retrieved_data` as an array of the found items "
    "(empty array if nothing matches; use numbers for numeric values, "
    "booleans for yes/no, and follow any format the task specifies). "
    "Give `error_details` only for non-SUCCESS statuses."
)

#: 历史轮观测的占位模板(WebAgent-R1 式;{i} 从 1 计数)。
OBS_PLACEHOLDER = (
    "[The browser observation for this step has been omitted to save "
    "context. It is no longer available.]"
)

_OBS_TOO_LONG_HINT = (
    "(observation truncated: page content exceeded the observation budget; "
    "middle elided, head and tail kept)"
)


def truncate_observation(text: str, max_chars: int) -> str:
    """Head+tail truncation(与终端域同构;比例 60/40,HTML 头部承载导航,
    尾部承载表格/分页)。max_chars<=0 关闭截断。"""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    tail = max_chars - head
    elided = len(text) - head - tail
    return (
        f"{text[:head]}\n"
        f"---- {elided} chars elided ----\n{_OBS_TOO_LONG_HINT}\n"
        f"{text[-tail:]}"
    )


def build_compressed_messages(
    scaffold: str,
    task_text: str,
    first_obs: str,
    turns: list[tuple[str, str]],
) -> list[dict]:
    """历史观测占位压缩的参考实现(裁定 #1;verl 侧按同一规则落地)。

    turns: 已完成轮的 ``(assistant_text, feedback_text)`` 序列。产出规则:
    * 首轮 user = 任务指令 + 首观测;一旦存在后续轮,首观测也换占位——
      任务指令永不占位。
    * 历史轮 feedback 一律换占位;仅最新一条 feedback 全量。
    * assistant 文本(思考+工具调用)全保留。
    返回的 tool 轮以 role="tool" 表示;具体 chat-template 渲染(含
    tool_call_id 回填)由驱动器/训练侧处理。
    """
    has_history = len(turns) > 0
    first_user = task_text + "\n\n" + (OBS_PLACEHOLDER if has_history else first_obs)
    msgs: list[dict] = [
        {"role": "system", "content": scaffold},
        {"role": "user", "content": first_user},
    ]
    for i, (assistant_text, feedback) in enumerate(turns):
        is_last = i == len(turns) - 1
        msgs.append({"role": "assistant", "content": assistant_text})
        msgs.append({
            "role": "tool",
            "content": feedback if is_last else OBS_PLACEHOLDER,
        })
    return msgs


# ------------------------------------------------------------------ handle

class WebArenaEnvHandle:
    """One live WebArena episode: pooled site instances + one browser session.

    ``step(action_dict)`` → (obs, done, score)。score 仅在 terminate 时经
    verified 终局评测产生;预算耗尽 = 0。浏览器/站点池基建故障映射为
    done + score 0 + 文本说明(infra_failure 单列监控,评测协议双报口径)。
    """

    def __init__(
        self,
        *,
        session: BrowserSession,
        alloc,
        task_id: int,
        task_text: str,
        episode_dir: Path,
        cfg: WebArenaCfg,
        max_env_steps: int,
    ) -> None:
        self._session = session
        self._alloc = alloc
        self._task_id = task_id
        self._episode_dir = episode_dir
        self._cfg = cfg
        self._max_env_steps = max_env_steps
        self._steps = 0
        self._closed = False
        self.task_text = task_text
        self.first_obs = ""              # set by adapter right after start
        self.infra_failure = ""          # 非空 = 因基建故障终局(可观测)
        self.eval_detail: dict = {}      # TaskEvalResult dump(轨迹记账)

    @property
    def initial_obs(self) -> str:
        return f"{self.task_text}\n\n{self.first_obs}"

    def observe(self) -> str:
        obs = self._session.observation()
        from .browser import format_llm_observation

        return truncate_observation(
            format_llm_observation(obs), self._cfg.max_obs_chars)

    def step(self, action: dict, *, replay: bool = False) -> tuple[str, bool, float]:
        self._steps += 1
        name = str(action.get("action", ""))
        try:
            if name == "terminate":
                return self._terminate(action, replay=replay)
            err = self._dispatch(name, action)
            if err is not None:
                obs = f"Action failed: {err}\n\n{self.observe()}"
            else:
                obs = self.observe()
        except (BrowserError, SitePoolError) as infra:
            return self._infra_failure(infra)
        if self._steps >= self._max_env_steps:
            return (
                f"{obs}\n\nStep budget exhausted; the episode ends now. "
                "No final response was submitted.",
                True, 0.0,
            )
        return obs, False, 0.0

    def _dispatch(self, name: str, a: dict) -> str | None:
        """Run one browser action; returns an error string for MODEL-level
        failures (bad target, malformed params — feedback, not exceptions);
        infra failures propagate as BrowserError."""
        s = self._session
        try:
            if name == "click":
                s.act("click", str(a["target"]))
            elif name == "type":
                s.act("type_text", str(a["target"]), str(a.get("text", "")),
                      bool(a.get("enter", False)))
            elif name == "hover":
                s.act("hover", str(a["target"]))
            elif name == "select":
                s.act("select", str(a["target"]), str(a.get("value", "")))
            elif name == "clear":
                s.act("clear", str(a["target"]))
            elif name == "key_press":
                s.act("key_press", str(a.get("key", "")),
                      str(a["target"]) if a.get("target") else None)
            elif name == "goto_url":
                s.act("goto_url", str(a.get("url", "")))
            elif name == "back":
                s.act("back")
            elif name == "forward":
                s.act("forward")
            elif name == "refresh":
                s.act("refresh")
            elif name == "new_tab":
                s.act("new_tab", str(a["url"]) if a.get("url") else None)
            elif name == "switch_tab":
                s.act("switch_tab", int(a["tab_id"]))
            elif name == "close_tab":
                s.act("close_tab", int(a["tab_id"]))
            else:
                return (f"Unknown action '{name}'. Valid actions: "
                        f"{', '.join(_ACTIONS)}.")
            return None
        except BrowserError:
            raise
        except KeyError as err:
            return f"Missing required parameter for '{name}': {err}"
        except Exception as err:  # noqa: BLE001 — playwright 动作失败是环境反馈
            return f"{type(err).__name__}: {str(err).splitlines()[0][:300]}"

    def _terminate(self, a: dict, *, replay: bool) -> tuple[str, bool, float]:
        problems = []
        task_type = str(a.get("task_type", "") or "").upper()
        status = str(a.get("status", "") or "").upper()
        if task_type not in _TASK_TYPE_VALUES:
            problems.append(
                f"'task_type' must be one of {_TASK_TYPE_VALUES}")
        if status not in _STATUS_VALUES:
            problems.append(f"'status' must be one of {_STATUS_VALUES}")
        retrieved = a.get("retrieved_data")
        if retrieved is not None and not isinstance(retrieved, list):
            problems.append("'retrieved_data' must be an array (or omitted)")
        if problems:
            # 终局报文残缺:反馈格式错误,episode 继续(模型可修正重提)。
            msg = ("Format error in terminate call: " + "; ".join(problems)
                   + ". The episode continues — fix the terminate call.")
            if self._steps >= self._max_env_steps:
                return (msg + "\nStep budget exhausted; episode ends with no "
                        "valid final response.", True, 0.0)
            return msg, False, 0.0
        final_response = {
            "task_type": task_type,
            "status": status,
            "retrieved_data": retrieved,
            "error_details": (str(a["error_details"])
                              if a.get("error_details") else None),
        }
        (self._episode_dir / "agent_response.json").write_text(
            json.dumps(final_response, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            self._session.finalize()      # flush HAR
        except BrowserError as infra:
            return self._infra_failure(infra)
        if replay:
            return "Final response recorded (replay; not evaluated).", True, 0.0
        from . import evalbridge

        outcome = evalbridge.evaluate(
            self._task_id, final_response,
            self._episode_dir / "network.har", self._alloc.base_urls,
        )
        self.eval_detail = outcome.detail
        if outcome.status == "error":
            self.infra_failure = f"evaluator error: {outcome.error_msg}"
        (self._episode_dir / "eval_result.json").write_text(
            json.dumps({"score": outcome.score, "status": outcome.status,
                        "error_msg": outcome.error_msg,
                        "detail": outcome.detail},
                       ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return (
            f"Final response submitted. Evaluation: {outcome.status}.",
            True, outcome.score,
        )

    def _infra_failure(self, err: Exception) -> tuple[str, bool, float]:
        self.infra_failure = str(err)[:500]
        logger.warning("[task %s] infra failure: %s", self._task_id, err)
        return (
            "Browser/site infrastructure failure; the episode cannot "
            "continue. Ending episode with reward 0.",
            True, 0.0,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._session.close()
        finally:
            self._alloc.release()


# ----------------------------------------------------------------- adapter

class WebArenaTaskAdapter:
    """TaskAdapter bound to one WebArena-Verified dataset row."""

    def __init__(
        self,
        item: dict,
        cfg: WebArenaCfg,
        *,
        pool,
        spawn_gate: threading.Semaphore,
    ) -> None:
        self._item = item
        self._cfg = cfg
        self._pool = pool
        self._gate = spawn_gate

    # ------------------------------------------------------------- identity

    @property
    def task_id(self) -> str:
        return str(self._item["task_id"])

    @property
    def task_type(self) -> str:
        """Family = 主站点(分层采样口径;多站点任务归首站点)。"""
        return str(self._item["sites"][0])

    @property
    def initial_obs(self) -> str:
        """任务指令块(静态;页面首观测是 episode 级的,见
        EnvHandle.initial_obs = 指令块 + 首观测)。"""
        return str(self._item["user_prompt"])

    # ------------------------------------------------------------- protocol

    def task_brief(self) -> str:
        sites = ", ".join(self._item["sites"])
        return (
            "Environment: a web browser on self-hosted website(s) "
            f"[{sites}] (WebArena). The executor operates the browser via "
            "the step_browser(action, ...) tool, one action per turn: "
            "clicking/typing/selecting on elements addressed by semantic "
            "IDs shown in the page observation, plus URL navigation and tab "
            "management. It finishes by calling terminate with a structured "
            "final report (task_type / status / retrieved_data).\n\n"
            "Task instruction:\n"
            f"{self.initial_obs}\n\n"
            "Evaluation: all-or-nothing — the final report and the "
            "episode's network activity are checked against typed expected "
            "values; partial matches score 0. No terminate call means "
            f"score 0. Episode step budget: {protocol.MAX_ENV_STEPS} tool "
            "calls; long page observations are truncated head+tail."
        )

    def executor_scaffold(self) -> str:
        return SYSTEM_PROMPT

    def tools(self) -> list[dict] | None:
        return TOOLS_FC

    def open_episode(self) -> WebArenaEnvHandle:
        sites = list(self._item["sites"])
        with self._gate:
            alloc = self._pool.acquire(sites)
            episode_dir = (Path(self._cfg.work_dir)
                           / f"{self.task_id}-{uuid.uuid4().hex[:8]}")
            episode_dir.mkdir(parents=True, exist_ok=True)
            session = BrowserSession(
                base_urls=alloc.base_urls,
                har_path=episode_dir / "network.har",
                headless=self._cfg.headless,
            )
            try:
                start_urls = render_start_urls(
                    list(self._item["start_urls"]), alloc.base_urls)
                session.start(sites, start_urls)
                env = WebArenaEnvHandle(
                    session=session,
                    alloc=alloc,
                    task_id=int(self._item["task_id"]),
                    task_text=self.initial_obs,
                    episode_dir=episode_dir,
                    cfg=self._cfg,
                    max_env_steps=protocol.MAX_ENV_STEPS,
                )
                env.first_obs = env.observe()
                return env
            except Exception:
                session.close()
                alloc.release()
                raise

    def gt_text(self) -> str | None:
        return None      # Verified 无逐步 oracle;特权文档走 steering 通道

    def domain_notes(self) -> str:
        return (
            "Key events in this environment: which pages/sections the agent "
            "navigated to (URLs, menu paths); searches and filters applied "
            "and whether results matched intent; forms filled and whether "
            "submission was confirmed by the page; data it read off the page "
            "(names, counts, amounts); tab usage; whether the final "
            "terminate report matched what was actually observed/changed."
        )

    def normalize_action(self, raw: str) -> str:
        return raw

    def observation_indicates_invalid(self, observation: str) -> bool:
        return observation.startswith(("Action failed:", "Unknown action"))

    def format_reminder(self) -> str:
        return (
            "Format error: Your last response did not include a valid "
            "`step_browser` tool call.\n\n"
            "Please always provide EXACTLY ONE call to the `step_browser` "
            "tool. To finish the task, call it with action='terminate' and "
            "your structured final report (task_type, status, and "
            "retrieved_data for RETRIEVE tasks)."
        )

    def react(self, env: WebArenaEnvHandle, assistant_text: str):
        """FC react(与终端域同框架):每回复恰一个 step_browser 调用;
        no-call → 格式提醒(裸文本,不耗步);unknown tool / 缺 action →
        错误反馈且耗步;正常 → env.step。"""
        from ...core.adapter import ActionOutcome

        calls = protocol.parse_tool_calls(assistant_text, tools=self.tools())
        thought = protocol.strip_tool_calls(assistant_text)
        if not calls:
            return ActionOutcome(
                thought=assistant_text.strip(),
                action_raw="", action_norm="",
                feedback=self.format_reminder(),
                done=False, score=0.0,
                acted=False, valid=False, consumed_env_step=False,
            )
        extra_note = ""
        if len(calls) > 1:
            extra_note = (
                f"\n(note: {len(calls) - 1} extra tool call(s) in the same "
                "reply were ignored — make exactly one call per turn.)"
            )
        call = calls[0]
        if call["name"] != "step_browser":
            return ActionOutcome(
                thought=thought,
                action_raw=call["name"], action_norm="",
                feedback=protocol.tool_response_wrap(
                    f"Unknown tool '{call['name']}'. The only available tool "
                    "is `step_browser`." + extra_note
                ),
                done=False, score=0.0,
                acted=True, valid=False, consumed_env_step=True,
            )
        args = dict(call["arguments"] or {})
        raw = json.dumps(args, ensure_ascii=False, sort_keys=True)
        if "action" not in args:
            return ActionOutcome(
                thought=thought,
                action_raw=raw, action_norm="",
                feedback=protocol.tool_response_wrap(
                    "Missing required 'action' parameter. "
                    f"Valid actions: {', '.join(_ACTIONS)}." + extra_note
                ),
                done=False, score=0.0,
                acted=True, valid=False, consumed_env_step=True,
            )
        obs, done, score = env.step(args)
        return ActionOutcome(
            thought=thought,
            action_raw=raw, action_norm=raw,
            feedback=protocol.tool_response_wrap(obs + extra_note),
            done=done, score=score,
            acted=True,
            valid=not self.observation_indicates_invalid(obs),
            consumed_env_step=True,
        )

    def replay_action(self, env: WebArenaEnvHandle, action_norm: str) -> tuple[str, bool]:
        args = json.loads(action_norm)
        obs, done, _score = env.step(args, replay=True)
        return obs, done


class WebArenaEnvFactory:
    """Mints per-task adapters sharing one site pool + spawn gate."""

    def __init__(self, cfg: WebArenaCfg) -> None:
        if cfg.pool == "static":
            from .sites import StaticSitePool

            if not cfg.static_base_urls:
                raise ValueError(
                    "webarena.pool='static' requires webarena.static_base_urls"
                )
            self._pool = StaticSitePool(cfg.static_base_urls)
        elif cfg.pool == "acs":
            from .sites import AcsSitePool

            if not (cfg.kubeconfig and cfg.sandboxsets):
                raise ValueError(
                    "webarena.pool='acs' requires webarena.kubeconfig and "
                    "webarena.sandboxsets (site → SandboxSet name)"
                )
            self._pool = AcsSitePool(
                kubeconfig=cfg.kubeconfig,
                namespace=cfg.namespace,
                sandboxsets=cfg.sandboxsets,
                acquire_timeout_s=cfg.acquire_timeout_s,
                claimant=cfg.run_label,
            )
        else:
            raise ValueError(f"unknown webarena.pool {cfg.pool!r}")
        if not cfg.work_dir:
            raise ValueError("webarena.work_dir required (HAR/eval artifacts)")
        self._cfg = cfg
        self._gate = threading.Semaphore(cfg.spawn_concurrency)

    def adapter_from_item(self, item: dict) -> WebArenaTaskAdapter:
        return WebArenaTaskAdapter(
            item, self._cfg, pool=self._pool, spawn_gate=self._gate,
        )
