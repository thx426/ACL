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
    verified 终局评测产生;预算耗尽 = 0。

    基建故障(浏览器死/站点池丢/HAR 冲不出/评测器崩)**抛异常**,由 core
    的 episode 兜底统一归 OUTCOME_ENV_ERROR(训练侧重试/丢弃)——绝不
    返回 done+0,否则基建压力会被当成模型失败入训(审阅 P0-5)。
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
        gate_release=None,
    ) -> None:
        self._session = session
        self._alloc = alloc
        self._task_id = task_id
        self._episode_dir = episode_dir
        self._cfg = cfg
        self._max_env_steps = max_env_steps
        self._gate_release = gate_release   # 并发门 permit,close 时归还
        self._steps = 0
        self._closed = False
        self.task_text = task_text
        self.first_obs = ""              # set by adapter right after start
        self.infra_failure = ""          # 记账镜像(异常已抛给 core)
        self.eval_detail: dict = {}      # TaskEvalResult dump(轨迹记账)

    @property
    def initial_obs(self) -> str:
        return f"{self.task_text}\n\n{self.first_obs}"

    @property
    def base_urls(self) -> dict[str, str]:
        """本 episode 的站点 base URL 映射(含入口路径后缀;轨迹记账/
        调试用——评测端在 terminate 支路内部自取)。"""
        return dict(self._alloc.base_urls)

    def observe(self) -> str:
        obs = self._session.observation()
        from .browser import format_llm_observation

        return truncate_observation(
            format_llm_observation(obs), self._cfg.max_obs_chars)

    def step(self, action: dict, *, replay: bool = False) -> tuple[str, bool, float]:
        """基建异常(BrowserError/SitePoolError)直接向上抛——core 统一归
        OUTCOME_ENV_ERROR;这里只捕获并文本化模型级失败。"""
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
            self.infra_failure = str(infra)[:500]
            raise
        if self._steps >= self._max_env_steps:
            return (
                f"{obs}\n\nStep budget exhausted; the episode ends now. "
                "No final response was submitted.",
                True, 0.0,
            )
        return obs, False, 0.0

    #: 必需参数表(审阅 P1-7:缺参必须格式反馈,绝不静默补空执行——
    #: `type` 缺 text 会真实清空字段)。
    _REQUIRED_PARAMS = {
        "click": ("target",), "type": ("target", "text"),
        "hover": ("target",), "select": ("target", "value"),
        "clear": ("target",), "key_press": ("key",),
        "goto_url": ("url",), "switch_tab": ("tab_id",),
        "close_tab": ("tab_id",),
    }
    #: Playwright 异常里的"浏览器已死"指纹 → 基建口径,不是模型反馈
    #: (审阅 P0-5:Chromium OOM 后的 TargetClosedError 不能当动作失败)。
    _DEAD_BROWSER_MARKERS = (
        "TargetClosedError", "has been closed", "browser has disconnected",
        "Connection closed", "Browser closed",
    )

    def _dispatch(self, name: str, a: dict) -> str | None:
        """Run one browser action; returns an error string for MODEL-level
        failures (bad target, malformed params — feedback, not exceptions);
        infra failures propagate as BrowserError."""
        missing = [p for p in self._REQUIRED_PARAMS.get(name, ())
                   if a.get(p) in (None, "")]
        if missing:
            return (f"Missing required parameter(s) {missing} for "
                    f"'{name}'. Nothing was executed.")
        s = self._session
        try:
            if name == "click":
                s.act("click", str(a["target"]))
            elif name == "type":
                s.act("type_text", str(a["target"]), str(a["text"]),
                      bool(a.get("enter", False)))
            elif name == "hover":
                s.act("hover", str(a["target"]))
            elif name == "select":
                s.act("select", str(a["target"]), str(a["value"]))
            elif name == "clear":
                s.act("clear", str(a["target"]))
            elif name == "key_press":
                s.act("key_press", str(a["key"]),
                      str(a["target"]) if a.get("target") else None)
            elif name == "goto_url":
                s.act("goto_url", str(a["url"]))
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
        except Exception as err:  # noqa: BLE001
            msg = f"{type(err).__name__}: {str(err).splitlines()[0][:300]}"
            if any(m in msg for m in self._DEAD_BROWSER_MARKERS):
                raise BrowserError(f"browser dead during '{name}': {msg}") from err
            return msg              # 模型级失败(元素不存在等)= 环境反馈

    def _validate_final(self, a: dict) -> tuple[dict | None, str]:
        """Build + validate the FinalAgentResponse via the pinned package's
        own model(审阅 P1-7:公开 schema 拒绝的报文——嵌套 list、深层
        结构——不得写盘终局;一切校验失败都是可修正的格式反馈)。"""
        candidate = {
            "task_type": a.get("task_type"),
            "status": a.get("status"),
            "retrieved_data": a.get("retrieved_data"),
            "error_details": (str(a["error_details"])
                              if a.get("error_details") else None),
        }
        try:
            json.dumps(candidate)          # 深度/循环防线(先于 pydantic)
            from webarena_verified.types.agent_response import (
                FinalAgentResponse,
            )

            model = FinalAgentResponse.model_validate(candidate)
            return model.model_dump(mode="json"), ""
        except ImportError:
            raise                          # 环境残破是基建问题,不给模型
        except Exception as err:  # noqa: BLE001 — 模型报文非法=格式反馈
            return None, str(err).splitlines()[0][:400]

    def _terminate(self, a: dict, *, replay: bool) -> tuple[str, bool, float]:
        final_response, problem = self._validate_final(a)
        if final_response is None:
            # 终局报文残缺:反馈格式错误,episode 继续(模型可修正重提)。
            msg = (f"Format error in terminate call: {problem}. "
                   "The episode continues — fix the terminate call.")
            if self._steps >= self._max_env_steps:
                return (msg + "\nStep budget exhausted; episode ends with no "
                        "valid final response.", True, 0.0)
            return msg, False, 0.0
        (self._episode_dir / "agent_response.json").write_text(
            json.dumps(final_response, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._session.finalize()           # flush HAR;失败=BrowserError 上抛
        if replay:
            return "Final response recorded (replay; not evaluated).", True, 0.0
        from . import evalbridge

        outcome = evalbridge.evaluate(
            self._task_id, final_response,
            self._episode_dir / "network.har", self._alloc.base_urls,
        )
        self.eval_detail = outcome.detail
        (self._episode_dir / "eval_result.json").write_text(
            json.dumps({"score": outcome.score, "status": outcome.status,
                        "error_msg": outcome.error_msg,
                        "detail": outcome.detail},
                       ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        if outcome.status == "error":
            # 评测器崩溃/HAR 缺失是基建口径:抛给 core → ENV_ERROR,
            # 不得作为 0 分模型失败入训。
            self.infra_failure = f"evaluator error: {outcome.error_msg}"
            raise SitePoolError(
                f"verified evaluator error for task {self._task_id}: "
                f"{outcome.error_msg}"
            )
        return (
            f"Final response submitted. Evaluation: {outcome.status}.",
            True, outcome.score,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._session.close()
        finally:
            try:
                self._alloc.release()
            finally:
                if self._gate_release is not None:
                    self._gate_release()
                if not self._cfg.keep_har:
                    # HAR 量大(实测 4-8MB/episode);默认评测后即弃,
                    # 保留 agent_response/eval_result 小件(审阅 P1-9)。
                    try:
                        (self._episode_dir / "network.har").unlink(
                            missing_ok=True)
                    except OSError:
                        pass


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
        """并发门 permit 覆盖**整个 episode 生命周期**(审阅 P0-3:浏览器
        活到 close,门只罩启动波次会让存活 Chromium 无上界);permit 移交
        EnvHandle,close/失败清理归还。alloc 领用后立刻进入保护块(审阅
        P1-1:mkdir/构造失败不得泄漏站点租约)。"""
        sites = list(self._item["sites"])
        self._gate.acquire()
        released = [False]

        def _release_gate() -> None:
            if not released[0]:
                released[0] = True
                self._gate.release()

        alloc = None
        session = None
        try:
            alloc = self._pool.acquire(sites)
            episode_dir = (Path(self._cfg.work_dir)
                           / f"{self.task_id}-{uuid.uuid4().hex}")
            episode_dir.mkdir(parents=True, exist_ok=False)
            session = BrowserSession(
                base_urls=alloc.base_urls,
                har_path=episode_dir / "network.har",
                headless=self._cfg.headless,
            )
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
                gate_release=_release_gate,
            )
            env.first_obs = env.observe()
            return env
        except Exception:
            if session is not None:
                session.close()
            if alloc is not None:
                alloc.release()
            _release_gate()
            raise

    def validate_initial_obs(self, live: str) -> bool:
        """core 开局校验 hook(审阅 P0-2):任务指令部分必须逐字复现,
        页面观测部分是 episode 级动态内容,只要求非空。"""
        task = self.initial_obs.strip()
        live = live.strip()
        return live.startswith(task) and len(live) > len(task)

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
        return observation.startswith(
            ("Action failed:", "Unknown action",
             "Format error in terminate call"))

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
        import math

        # 底层自保校验(load_config._validate 之外的直构入口同样受protect,
        # 审阅 P2-5:spawn=0 会永久卡 semaphore、NaN timeout 永不超时)。
        if cfg.spawn_concurrency < 1:
            raise ValueError("webarena.spawn_concurrency must be >= 1")
        if (not math.isfinite(cfg.acquire_timeout_s)
                or cfg.acquire_timeout_s <= 0):
            raise ValueError("webarena.acquire_timeout_s must be finite > 0")
        if not cfg.work_dir:
            raise ValueError("webarena.work_dir required (HAR/eval artifacts)")
        work = Path(cfg.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        probe = work / f".writable-{uuid.uuid4().hex[:6]}"
        try:
            probe.write_text("", encoding="utf-8")
            probe.unlink()
        except OSError as err:
            raise ValueError(
                f"webarena.work_dir not writable: {cfg.work_dir!r} ({err})"
            ) from err
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
        self._cfg = cfg
        self._gate = threading.Semaphore(cfg.spawn_concurrency)

    def adapter_from_item(self, item: dict) -> WebArenaTaskAdapter:
        # static 池无状态隔离:MUTATE 任务默认拒绝(审阅 P1-11——安全不能
        # 依赖某个调用脚本记得筛 task_type;串行调试经显式开关放行)。
        if (self._cfg.pool == "static"
                and not self._cfg.allow_static_mutate
                and str(item.get("expected_task_type", "")).upper() == "MUTATE"):
            raise ValueError(
                f"task {item.get('task_id')} is MUTATE but pool='static' has "
                "no state isolation; use pool='acs' or set "
                "webarena.allow_static_mutate=true for serial debugging"
            )
        return WebArenaTaskAdapter(
            item, self._cfg, pool=self._pool, spawn_gate=self._gate,
        )
