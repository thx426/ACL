"""ALFWorld TaskAdapter (host side).

One adapter instance per task, minted from a dataset row (the same
``extra_info`` dict that rides the training parquet — docs/design.md §5):
    {task_id, gamefile (relative to data_root), split, initial_obs, gt_text}

The executor scaffold lives in this module (SYSTEM_PROMPT_FC / _BARE, picked
by ``alfworld.protocol``) — research experiments, the training rollout, and
the evaluator share one contract by construction.

GT firewall: gt_text is pre-extracted at dataset build time and rides the
row; it is routed ONLY to the researcher prompt when the GT switch is on —
never to the executor, never through fetch_raw.

Worker-subprocess management (select-based reads, spawn gate, process-group
kill) is the pilot-audited pattern.
"""
from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import threading
import time
from pathlib import Path

from ...common.config import AlfworldCfg
from ...core import protocol

_WORKER_SCRIPT = str(Path(__file__).with_name("worker.py"))
_CLOSE_SENTINEL = "__CLOSE__"
GT_SENTINEL = "__GT__"


class WorkerFailure(RuntimeError):
    pass


class _Worker:
    """Host side of one engine subprocess (line protocol, own process group).

    want_gt is used only at dataset build time (walkthrough fallback for GT
    pre-extraction); the runtime episode path never requests GT (firewall)."""

    def __init__(
        self,
        *,
        python_exe: str,
        gamefile: str,
        max_steps: int,
        data_root: str,
        start_timeout_s: float,
        step_timeout_s: float,
        want_gt: bool = False,
    ) -> None:
        env = dict(os.environ)
        env["ALFWORLD_DATA"] = data_root
        self._step_timeout_s = step_timeout_s
        self._lock = threading.Lock()
        argv = [python_exe, _WORKER_SCRIPT, gamefile, str(max_steps)]
        if want_gt:
            argv.append("--gt")
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            start_new_session=True,
            env=env,
        )
        ready = self._read(start_timeout_s)
        if ready.get("event") != "ready":
            self.close()
            raise WorkerFailure(f"alfworld worker failed to start: {str(ready)[:2000]}")
        self.initial_obs = str(ready.get("obs", ""))

    def _read(self, timeout_s: float) -> dict:
        assert self._proc.stdout is not None
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close()
                raise WorkerFailure(f"alfworld worker timed out after {timeout_s:.0f}s")
            ready, _, _ = select.select([self._proc.stdout], [], [], min(remaining, 5.0))
            if not ready:
                if self._proc.poll() is not None:
                    raise WorkerFailure(f"alfworld worker exited (rc={self._proc.returncode})")
                continue
            line = self._proc.stdout.readline()
            if not line:
                raise WorkerFailure(f"alfworld worker closed its pipe (rc={self._proc.poll()})")
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue  # stray non-protocol line; the channel dup makes this rare

    def request(self, line: str) -> dict:
        with self._lock:
            assert self._proc.stdin is not None
            self._proc.stdin.write(line.replace("\n", " ") + "\n")
            self._proc.stdin.flush()
            event = self._read(self._step_timeout_s)
        if event.get("event") == "fatal":
            raise WorkerFailure(str(event.get("error", "fatal"))[:2000])
        return event

    def close(self) -> None:
        proc = self._proc
        if proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write(_CLOSE_SENTINEL + "\n")
                    proc.stdin.flush()
            except Exception:  # noqa: BLE001
                pass
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                # Group kill: the engine shells out; explicit PGID, never pkill.
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                # Reap after kill — otherwise timeout storms accumulate zombies.
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    pass
        # Idempotent pipe cleanup (FD hygiene under repeated close()).
        for pipe in (proc.stdin, proc.stdout):
            try:
                if pipe is not None:
                    pipe.close()
            except Exception:  # noqa: BLE001
                pass


class AlfworldEnvHandle:
    def __init__(self, worker: _Worker) -> None:
        self._worker = worker

    @property
    def initial_obs(self) -> str:
        return self._worker.initial_obs

    def step(self, command: str) -> tuple[str, bool, bool]:
        event = self._worker.request(command)
        return (
            str(event.get("obs", "")),
            bool(event.get("done", False)),
            bool(event.get("won", False)),
        )

    def close(self) -> None:
        self._worker.close()


def normalize_command(raw: str) -> str:
    """Conservative, semantics-free normalization (prior-project discipline:
    ungrammatical actions must reach the engine and fail visibly)."""
    command = (raw or "").strip()
    command = command.splitlines()[0].strip() if command else ""
    return command.strip("\"'").strip().lower()


# --- ALFWorld executor prompt material(2026-08-06 自 core/protocol.py 迁入,
#     内容逐字节不变;通用性整改:任务提示词归 benchmark 实现自持)---------
# COMMAND_REFERENCE is quoted from the engine's own ``help`` output
# (AlfredTWEnv text mode), captured verbatim on the target server (CD-003,
# experiments/h3_pilot/notes/engine_help.txt) — never paraphrase.
COMMAND_REFERENCE = """\
Available commands:
  look:                             look around your current location
  inventory:                        check your current inventory
  go to (receptacle):               move to a receptacle
  open (receptacle):                open a receptacle
  close (receptacle):               close a receptacle
  take (object) from (receptacle):  take an object from a receptacle
  move (object) to (receptacle):  place an object in or on a receptacle
  examine (something):              examine a receptacle or an object
  use (object):                     use an object
  heat (object) with (receptacle):  heat an object using a receptacle
  clean (object) with (receptacle): clean an object using a receptacle
  cool (object) with (receptacle):  cool an object using a receptacle
  slice (object) with (object):     slice an object using a sharp object"""

SYSTEM_PROMPT_BARE = (
    "You are an agent in the ALFRED text-based household environment. "
    "Complete the task stated after 'Your task is to:' by issuing exactly one "
    "environment command per turn.\n\n"
    + COMMAND_REFERENCE
    + "\n\n"
    "Use exact entity names with their numbers as observed (e.g. 'bread 1', "
    "'countertop 2'). An invalid or currently impossible command returns "
    "'Nothing happens.'.\n\n"
    "On every turn respond with exactly this format and nothing else:\n"
    "<reasoning>brief reasoning about the current situation</reasoning>\n"
    "<action>one command</action>\n\n"
    "The episode ends automatically when the task goal is reached; there is "
    "no submit command and no partial credit."
)

#: ALFWorld 的无效动作环境响应(判定与提醒习语;core 不再持有该字面量)。
INVALID_ACTION_OBS = "Nothing happens."


# --- Function-Call 协议面(2026-08-08 用户裁定:全环境统一 OpenAI 格式 FC,
#     ALFWorld 用单工具 execute(command)——动作空间与 text 协议逐字相同,
#     GT walkthrough / researcher 资产零改动)。schema 经 adapter.tools() 交
#     chat template 原生渲染;解析在文本层(parse_tool_calls),生成的原始
#     token 原样留在训练流。描述纪律同 SSB:写全执行模型/步预算语义/失败
#     行为/终局语义——agent 依赖的行为契约,不是摆设。 -------------------
TOOLS_FC: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "execute",
            "description": (
                "Execute exactly one ALFWorld text command and return the "
                "environment's observation. Each call consumes one step of "
                "the episode budget. An invalid or currently impossible "
                "command returns 'Nothing happens.' and still consumes the "
                "step. The episode ends automatically the moment the task "
                "goal is satisfied — there is no submit command and no "
                "partial credit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "One command following the 'Available commands' "
                            "grammar in the system prompt, lowercase, using "
                            "exact entity names with their numbers as "
                            "observed (e.g. 'take bread 1 from countertop 2')."
                        ),
                    },
                },
                "required": ["command"],
            },
        },
    },
]

#: FC 语境系统提示:role / 命令语法 / 实体命名规则 / 终局语义与 BARE 版
#: 逐点对齐,仅输出协议段不同(工具调用形态由 chat template 按 TOOLS_FC
#: 原生渲染,此处绝不手写调用格式)。
SYSTEM_PROMPT_FC = (
    "You are an agent in the ALFRED text-based household environment. "
    "Complete the task stated after 'Your task is to:' by issuing "
    "environment commands through the execute tool, one command per turn.\n\n"
    + COMMAND_REFERENCE
    + "\n\n"
    "Use exact entity names with their numbers as observed (e.g. 'bread 1', "
    "'countertop 2'). An invalid or currently impossible command returns "
    "'Nothing happens.'.\n\n"
    "Each turn, reason briefly about the current situation in plain text, "
    "then make exactly one execute call. The episode ends automatically "
    "when the task goal is reached; there is no submit command and no "
    "partial credit."
)


class AlfworldTaskAdapter:
    """TaskAdapter implementation bound to one dataset row."""

    def __init__(
        self,
        item: dict,
        cfg: AlfworldCfg,
        *,
        spawn_gate: threading.Semaphore,
    ) -> None:
        self._item = item
        self._cfg = cfg
        self._gate = spawn_gate

    # ------------------------------------------------------------- identity

    @property
    def task_id(self) -> str:
        return str(self._item["task_id"])

    @property
    def initial_obs(self) -> str:
        return str(self._item.get("initial_obs", ""))

    @property
    def task_type(self) -> str:
        # task_id = "<split>/<type>-<Obj>-.../trial_..." (build_datasets keeps
        # the split segment) — skip the split, then take the '-'-prefix of the
        # task directory. Falls back gracefully for ids without a split prefix.
        parts = self.task_id.split("/")
        head = parts[1] if len(parts) >= 2 and "-" not in parts[0] else parts[0]
        return head.split("-", 1)[0]

    # ------------------------------------------------------------- protocol

    def task_brief(self) -> str:
        via = (
            "via the execute(command) tool, one command per turn"
            if self._cfg.protocol == "fncall"
            else "via text commands"
        )
        return (
            "Environment: ALFWorld — the ALFRED household simulator as a "
            "text game. The executor moves through rooms and manipulates "
            f"objects/receptacles {via}; state changes only "
            "through valid commands.\n"
            f"Task type: {self.task_type}\n\n"
            "Exactly what the executor sees at episode start (scene + goal):\n"
            f"{self.initial_obs}\n\n"
            "Action grammar available to the executor:\n"
            f"{COMMAND_REFERENCE}\n\n"
            "Evaluation: binary — the engine itself detects goal satisfaction "
            f"(no partial credit). Episode step budget: {protocol.MAX_ENV_STEPS} "
            "steps; invalid commands return 'Nothing happens.' and consume a "
            "step."
        )

    def executor_scaffold(self) -> str:
        if self._cfg.protocol == "fncall":
            return SYSTEM_PROMPT_FC
        return SYSTEM_PROMPT_BARE

    def open_episode(self) -> AlfworldEnvHandle:
        gamefile = str(Path(self._cfg.data_root) / str(self._item["gamefile"]))
        with self._gate:
            worker = _Worker(
                python_exe=self._cfg.worker_python,
                gamefile=gamefile,
                max_steps=protocol.MAX_ENV_STEPS + 10,  # engine cap above our budget
                data_root=self._cfg.data_root,
                start_timeout_s=self._cfg.start_timeout_s,
                step_timeout_s=self._cfg.step_timeout_s,
            )
        return AlfworldEnvHandle(worker)

    def gt_text(self) -> str | None:
        text = self._item.get("gt_text")
        return str(text) if text else None

    def domain_notes(self) -> str:
        return (
            "Key events in this environment: finding (or failing to find) the "
            "target object; take/move/open/close outcomes; appliance use "
            "(heat/cool/clean); any 'Nothing happens.' responses and what the "
            "agent did next."
        )

    def normalize_action(self, raw: str) -> str:
        return normalize_command(raw)

    def observation_indicates_invalid(self, observation: str) -> bool:
        """无效/不可执行命令的引擎响应恒为 'Nothing happens.'(环境语义
        判定归 benchmark;core 不解析环境文本)。"""
        return observation.strip() == INVALID_ACTION_OBS

    def format_reminder(self) -> str:
        """格式失败提醒:以 ALFWorld 习语开头,读起来像一次正常环境响应
        (裸文本——assistant 未发起合法动作,反馈不是 tool 响应)。"""
        if self._cfg.protocol == "fncall":
            return "Nothing happens. " + protocol.fc_format_reminder("execute")
        return "Nothing happens. " + protocol.DEFAULT_FORMAT_REMINDER

    def tools(self) -> list[dict] | None:
        """fncall 协议:单工具 execute;text 协议:无 tools(<reasoning>/
        <action> 文本动作)。"""
        return TOOLS_FC if self._cfg.protocol == "fncall" else None

    def react(self, env, assistant_text: str):
        """协议分派:fncall 走 FC react(parse_tool_calls + tool 响应包裹);
        text 委托标准文本动作协议件。"""
        if self._cfg.protocol == "fncall":
            return self._react_fncall(env, assistant_text)
        return protocol.text_action_react(self, env, assistant_text)

    def _react_fncall(self, env, assistant_text: str):
        """FC react(与 SSB/Math 同框架):每回复恰一个 execute 调用;
        unknown tool / 空 command 以 tool 响应报错且不消耗 env step;真执行
        的观察经 tool_response_wrap 包裹回注(与推理引擎对 role=tool 消息的
        渲染同形态,fc_stream_equivalence_probe 守护)。"""
        from ...core.adapter import ActionOutcome

        calls = protocol.parse_tool_calls(assistant_text, tools=self.tools())
        thought = protocol.strip_tool_calls(assistant_text)
        if not calls:
            return ActionOutcome(
                thought=assistant_text.strip(),
                action_raw="",
                action_norm="",
                feedback=self.format_reminder(),
                done=False,
                score=0.0,
                acted=False,
                valid=False,
                consumed_env_step=False,
            )
        extra_note = ""
        if len(calls) > 1:
            extra_note = (
                f"\n(note: {len(calls) - 1} extra tool call(s) in the same reply "
                "were ignored — make exactly one call per turn.)"
            )
        call = calls[0]
        if call["name"] != "execute":
            return ActionOutcome(
                thought=thought,
                action_raw=call["name"],
                action_norm="",
                feedback=protocol.tool_response_wrap(
                    f"unknown tool: {call['name']!r}. Available: execute."
                    + extra_note
                ),
                done=False,
                score=0.0,
                acted=True,
                valid=False,
                # 与 math/SSB 语义一致:发起了调用即消耗环境步——否则
                # acted=True 重置 DEGENERATE breaker 且步数不涨,unknown/空
                # 调用可零成本烧满 40 轮(2026-08-08 审阅)。
                consumed_env_step=True,
            )
        raw = str(call["arguments"].get("command", "") or "")
        norm = normalize_command(raw)
        if not norm:
            return ActionOutcome(
                thought=thought,
                action_raw=raw,
                action_norm="",
                feedback=protocol.tool_response_wrap(
                    "execute: empty command." + extra_note
                ),
                done=False,
                score=0.0,
                acted=True,
                valid=False,
                consumed_env_step=True,
            )
        obs, done, won = env.step(norm)
        return ActionOutcome(
            thought=thought,
            action_raw=raw,
            action_norm=norm,
            feedback=protocol.tool_response_wrap(obs + extra_note),
            done=done,
            score=float(won),
            acted=True,
            valid=not self.observation_indicates_invalid(obs),
            consumed_env_step=True,
        )

    def replay_action(self, env, action_norm: str) -> tuple[str, bool]:
        obs, done, _won = env.step(action_norm)
        return obs, done


class AlfworldEnvFactory:
    """Mints per-task adapters sharing one spawn gate (per process)."""

    def __init__(self, cfg: AlfworldCfg) -> None:
        if not cfg.data_root or not cfg.worker_python:
            raise ValueError(
                "alfworld config incomplete: data_root and worker_python are required "
                "(set ALFWORLD_DATA / ALFWORLD_WORKER_PYTHON)"
            )
        self._cfg = cfg
        self._gate = threading.Semaphore(cfg.spawn_concurrency)

    def adapter_from_item(self, item: dict) -> AlfworldTaskAdapter:
        return AlfworldTaskAdapter(item, self._cfg, spawn_gate=self._gate)
