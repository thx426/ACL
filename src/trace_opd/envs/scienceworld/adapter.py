"""ScienceWorld TaskAdapter(host side)。

引入裁定(用户 2026-08-08,docs/plans/2026-08-08-scienceworld-proposal.md
增补节):simplifications="easy" 全开(社区两谱系一致)/ 全局步预算
(RLVMR 口径,episode 段配置,非 ETO per-task 表)/ **终局二元 reward**
(won = engine completed 且 score>=100;RLVMR 的宽松版 score>0 不采,
对照时披露)/ ETO 数据口径(24 任务,train 1483 / seen 194 / unseen 211,
变体级划分)/ **不注入 admissible-action 表**(动作知识是 researcher 文档
的信息面——与 ALFWorld 的 No-admissible-list 纪律同源;RLVMR 注入,对照
时披露该差异)。

协议:统一 FC 单工具 execute(command)(与 ALFWorld 同构;M0/M1 基座)。
worker 子进程(JVM)每 episode 冷启——本地实测 boot 0.6s + load 0.6s,
池化收益不成立。GT = 引擎原生 gold action sequence(build 期预提取;注意
gold 不跟随 easy 简化,含冗余 open-door 步,harmless——researcher 语料
本就要提炼)。

One adapter instance per task, minted from a dataset row (extra_info):
    {task_id, task_name, variation_idx, split, initial_obs, gt_text}
task_id = "<eto_task_name>/<variation_idx>"(ETO 长名保社区对照;引擎侧
load 时包内 infer_task 自动转短名)。
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

from ...common.config import SciworldCfg
from ...core import protocol

_WORKER_SCRIPT = str(Path(__file__).with_name("worker.py"))
_CLOSE_SENTINEL = "__CLOSE__"
GT_SENTINEL = "__GT__"

#: 引擎无效动作响应(判定与提醒习语;本地真机捕获 2026-08-08)。
INVALID_ACTION_OBS = "No known action matches that input."


class WorkerFailure(RuntimeError):
    pass


class _Worker:
    """Host side of one JVM subprocess(line 协议;lineage: alfworld._Worker,
    argv 换 ScienceWorld 五元组)。want_gt 仅 build_dataset 用(GT 防火墙:
    运行时 episode 永不请求)。"""

    def __init__(
        self,
        *,
        python_exe: str,
        task_name: str,
        variation_idx: int,
        simplifications: str,
        env_step_limit: int,
        start_timeout_s: float,
        step_timeout_s: float,
        want_gt: bool = False,
    ) -> None:
        self._step_timeout_s = step_timeout_s
        self._lock = threading.Lock()
        argv = [
            python_exe, _WORKER_SCRIPT, task_name, str(variation_idx),
            simplifications, str(env_step_limit),
        ]
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
            env=dict(os.environ),
        )
        ready = self._read(start_timeout_s)
        if ready.get("event") != "ready":
            self.close()
            raise WorkerFailure(f"scienceworld worker failed to start: {str(ready)[:2000]}")
        self.initial_obs = str(ready.get("obs", ""))

    def _read(self, timeout_s: float) -> dict:
        assert self._proc.stdout is not None
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close()
                raise WorkerFailure(f"scienceworld worker timed out after {timeout_s:.0f}s")
            ready, _, _ = select.select([self._proc.stdout], [], [], min(remaining, 5.0))
            if not ready:
                if self._proc.poll() is not None:
                    raise WorkerFailure(
                        f"scienceworld worker exited (rc={self._proc.returncode})"
                    )
                continue
            line = self._proc.stdout.readline()
            if not line:
                raise WorkerFailure(f"scienceworld worker closed its pipe (rc={self._proc.poll()})")
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue  # stray non-protocol line (JVM/log noise); rare after fd dup

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
                # Group kill(JVM 子孙一并);explicit PGID, never pkill.
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    pass
        for pipe in (proc.stdin, proc.stdout):
            try:
                if pipe is not None:
                    pipe.close()
            except Exception:  # noqa: BLE001
                pass


class SciworldEnvHandle:
    def __init__(self, worker: _Worker) -> None:
        self._worker = worker
        self.last_score = 0  # 引擎原始分(0-100,可负;观察面,不进 reward)

    @property
    def initial_obs(self) -> str:
        return self._worker.initial_obs

    def step(self, command: str) -> tuple[str, bool, bool]:
        event = self._worker.request(command)
        self.last_score = int(event.get("score", 0))
        return (
            str(event.get("obs", "")),
            bool(event.get("done", False)),
            bool(event.get("won", False)),
        )

    def close(self) -> None:
        self._worker.close()


def normalize_command(raw: str) -> str:
    """Conservative, semantics-free(同 ALFWorld 纪律:不修复,坏命令必须
    到达引擎可见地失败)。引擎语法全小写。"""
    command = (raw or "").strip()
    command = command.splitlines()[0].strip() if command else ""
    return command.strip("\"'").strip().lower()


# --- 动作模板表:引擎 get_possible_actions() 逐字捕获(easy 语境,本地
#     真机 2026-08-08)——never paraphrase(ALFWorld COMMAND_REFERENCE 纪律)。
#     electrical 任务(power-component*/test-conductivity*)引擎自动豁免
#     noElectricalAction,表恰多 connect/disconnect 两行(实测)。--------
COMMAND_REFERENCE = """\
Available commands:
  activate OBJ
  close OBJ
  deactivate OBJ
  dunk OBJ in OBJ
  eat OBJ
  flush OBJ
  focus on OBJ
  go OBJ
  inventory
  look around
  look at OBJ
  look in OBJ
  mix OBJ
  move OBJ to OBJ
  open OBJ
  pick up OBJ
  pour OBJ in OBJ
  put down OBJ
  read OBJ
  teleport OBJ
  use OBJ on OBJ
  wait
  wait1"""

ELECTRICAL_EXTRA = """\
  connect OBJ to OBJ
  disconnect OBJ"""

#: electrical 任务前缀(ETO 长名;动作表附加 connect/disconnect 的判据)
_ELECTRICAL_PREFIXES = ("task-2-power-component", "task-2a-test-conductivity")


def _is_electrical(task_name: str) -> bool:
    return task_name.startswith(_ELECTRICAL_PREFIXES)


TOOLS_FC: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "execute",
            "description": (
                "Execute exactly one ScienceWorld command and return the "
                "environment's observation. Each call consumes one step of "
                "the episode budget. An unrecognized command returns "
                "'No known action matches that input.' and still consumes "
                "the step. 'focus on OBJ' signals task intent and is scored: "
                "focusing on a wrong object can permanently fail the task "
                "and end the episode immediately. The episode ends "
                "automatically the moment the task is fully accomplished — "
                "there is no submit command and no partial credit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "One command following the 'Available commands' "
                            "grammar in the system prompt, lowercase, with "
                            "OBJ replaced by object names as observed "
                            "(e.g. 'teleport to kitchen', "
                            "'pour metal pot in sink')."
                        ),
                    },
                },
                "required": ["command"],
            },
        },
    },
]


def _scaffold(command_reference: str) -> str:
    return (
        "You are an agent in the ScienceWorld text environment — a simulated "
        "house with rooms and scientific instruments where you carry out "
        "elementary-science experiment tasks. Complete the task stated after "
        "'Task:' by issuing environment commands through the execute tool, "
        "one command per turn.\n\n"
        + command_reference
        + "\n\n"
        "Replace OBJ with object names exactly as observed. You can teleport "
        "to any room in one step (e.g. 'teleport to kitchen'); containers "
        "are already open. 'focus on OBJ' declares the object of your "
        "experiment and is scored — focusing on a wrong object can "
        "permanently fail the task, so focus only on objects the task asks "
        "for.\n\n"
        "Each turn, reason briefly about the current situation in plain "
        "text, then make exactly one execute call. The episode ends "
        "automatically when the task is fully accomplished; there is no "
        "submit command and no partial credit."
    )


class SciworldTaskAdapter:
    """TaskAdapter implementation bound to one dataset row."""

    def __init__(
        self,
        item: dict,
        cfg: SciworldCfg,
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
        # task_id = "<eto_task_name>/<variation_idx>";family = 任务类型
        return self.task_id.split("/", 1)[0]

    @property
    def _task_name(self) -> str:
        return str(self._item["task_name"])

    def _command_reference(self) -> str:
        if _is_electrical(self._task_name):
            return COMMAND_REFERENCE + "\n" + ELECTRICAL_EXTRA
        return COMMAND_REFERENCE

    # ------------------------------------------------------------- protocol

    def task_brief(self) -> str:
        return (
            "Environment: ScienceWorld — a text simulation of a house with "
            "scientific instruments (thermometer, stove, electrical "
            "components, plants...). The executor performs an "
            "elementary-science experiment via text commands; state changes "
            "only through valid commands.\n"
            f"Task type: {self.task_type}\n\n"
            "Exactly what the executor sees at episode start (task + scene):\n"
            f"{self.initial_obs}\n\n"
            "Action grammar available to the executor:\n"
            f"{self._command_reference()}\n\n"
            "Key mechanics: teleport moves to any room in one step; "
            "containers start open; 'focus on OBJ' declares the experiment "
            "object and is SCORED — focusing on a wrong object can "
            "permanently fail the task (episode ends immediately).\n"
            "Evaluation: binary — the engine detects full task completion "
            f"(no partial credit). Episode step budget: {protocol.MAX_ENV_STEPS} "
            "steps; unrecognized commands return 'No known action matches "
            "that input.' and consume a step."
        )

    def executor_scaffold(self) -> str:
        return _scaffold(self._command_reference())

    def open_episode(self) -> SciworldEnvHandle:
        with self._gate:
            worker = _Worker(
                python_exe=self._cfg.worker_python,
                task_name=self._task_name,
                variation_idx=int(self._item["variation_idx"]),
                simplifications=self._cfg.simplifications,
                env_step_limit=self._cfg.env_step_limit,
                start_timeout_s=self._cfg.start_timeout_s,
                step_timeout_s=self._cfg.step_timeout_s,
            )
        return SciworldEnvHandle(worker)

    def gt_text(self) -> str | None:
        text = self._item.get("gt_text")
        return str(text) if text else None

    def domain_notes(self) -> str:
        return (
            "Key events in this environment: focus actions and their targets "
            "(wrong focus permanently fails the task); reading the task via "
            "measurements (thermometer readings, state changes); device "
            "activation outcomes; any 'No known action matches that input.' "
            "responses and what the agent did next."
        )

    def normalize_action(self, raw: str) -> str:
        return normalize_command(raw)

    def observation_indicates_invalid(self, observation: str) -> bool:
        return observation.strip() == INVALID_ACTION_OBS

    def format_reminder(self) -> str:
        """裸文本提醒(assistant 未发起合法调用,反馈不是 tool 响应);
        以引擎习语开头,读起来像一次正常环境响应。"""
        return (
            INVALID_ACTION_OBS + " " + protocol.fc_format_reminder("execute")
        )

    def tools(self) -> list[dict] | None:
        return TOOLS_FC

    def react(self, env, assistant_text: str):
        """FC react(与 alfworld/math/SSB 同框架):每回复恰一个 execute
        调用;unknown tool/空命令以 tool 响应报错且消耗环境步(审阅语义:
        发起调用即耗步,防 breaker 重置 + 零步循环);观察经
        tool_response_wrap 包裹回注。"""
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
            # 终局二元(用户裁定):score = float(won);引擎部分分只作
            # 观察面(handle.last_score),不进 reward。
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


class SciworldEnvFactory:
    """Mints per-task adapters sharing one spawn gate (per process)."""

    def __init__(self, cfg: SciworldCfg) -> None:
        if not cfg.worker_python:
            raise ValueError(
                "scienceworld config incomplete: worker_python is required "
                "(a venv with `scienceworld` + `py4j` installed and `java` "
                "(>=1.8) on PATH; set SCIWORLD_WORKER_PYTHON)"
            )
        self._cfg = cfg
        self._gate = threading.Semaphore(cfg.spawn_concurrency)

    def adapter_from_item(self, item: dict) -> SciworldTaskAdapter:
        return SciworldTaskAdapter(item, self._cfg, spawn_gate=self._gate)
