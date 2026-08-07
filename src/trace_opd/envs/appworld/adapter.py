"""AppWorld TaskAdapter — stateful-REPL coding agent over simulated apps.

react 第三形态(方案 §1):动作 = **fenced Python code block**(官方/社区
标准;代码就是动作本身,FC 的 JSON 包装对多行代码是纯转义噪声),
``tools()=None``。终局内生:代码调 ``apis.supervisor.complete_task()`` →
worker 报 completed → done + evaluate(won=TGC success;score=requirement
通过比例,部分分抽象第三实例)。

EnvHandle = worker 子进程(每 episode 一个,AppWorld 官方单 world/进程
硬约束;RAM ~300-500MB/world ⇒ engine_slots 门在 factory,acquire 覆盖
episode 全生命期)。

红线:评估信息(success/report)零反馈进 executor 流;难度 metadata 只在
数据侧;gt_text=None(gold op 存在但宿主不调,GT 防火墙继承)。
"""
from __future__ import annotations

import json
import os
import re
import select
import subprocess
import threading
import time
import uuid
from pathlib import Path

from ...common.config import AppworldCfg
from ...core.adapter import ActionOutcome

_WORKER_SCRIPT = str(Path(__file__).resolve().parent / "worker.py")

#: 最后一个 fenced 代码块(```python 或裸 ```);块外文本 = 推理记账
_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

FORMAT_REMINDER = (
    "(Your reply contained no Python code block. Write your reasoning as "
    "text, then end the message with exactly one fenced block:\n"
    "```python\n<code>\n```\n"
    "The environment executes the block and returns its output.)"
)

# 旧引入谈判定稿(2026-07-04)的 OFFICIAL-ONBOARDING 原则继承:只教 REPL
# 协议、三个 api_docs 元 API、完成契约;API 文档由 agent 运行时探索
# (探索是被训练的行为);官方 worked example 内嵌;GT/难度零泄漏。
SYSTEM_PROMPT = """\
You are an autonomous assistant completing your supervisor's day-to-day \
digital tasks in the AppWorld environment by operating a stateful Python REPL.

ENVIRONMENT PROTOCOL:
- Each turn, write your reasoning as normal text, then end your message \
with ONE fenced Python code block; the environment executes the block and \
returns the printed output, or the error traceback if it fails. Variables \
persist across turns (like a notebook) — reuse logins, tokens, and \
intermediate results instead of recomputing them.
- You interact with apps (e.g. amazon, spotify, venmo, splitwise, gmail, \
phone, simple_note, todoist, file_system, and the supervisor app) through \
their APIs, called as: apis.<app_name>.<api_name>(...)
- Discover what exists with the three documentation meta-APIs:
    print(apis.api_docs.show_app_descriptions())
    print(apis.api_docs.show_api_descriptions(app_name='supervisor'))
    print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))
- Your supervisor's account credentials are available through the \
supervisor app. Outputs can be very long — query narrowly (use the APIs' \
filter and pagination parameters) to keep the output focused.
- The task is complete ONLY when you call apis.supervisor.complete_task(). \
If the task asks a question, submit the result with \
apis.supervisor.complete_task(answer=...). The episode never ends on its own.
- Success is judged by the FINAL DATABASE STATE (plus the answer, when one \
is asked): achieve exactly the requested goal and do not make collateral \
changes to unrelated data.

WORKED EXAMPLE — task: "What is the password for my Spotify account?"
  Turn 1:
    ```python
    # Find which supervisor APIs exist.
    print(apis.api_docs.show_api_descriptions(app_name='supervisor'))
    ```
    -> output includes: show_account_passwords : Show your supervisor's account passwords.
  Turn 2:
    ```python
    passwords = apis.supervisor.show_account_passwords()
    print(passwords)
    ```
    -> output: [{"account_name": "amazon", "password": "..."}, {"account_name": "spotify", "password": "dummy_spotify_pass"}, ...]
  Turn 3:
    ```python
    spotify_password = [p for p in passwords if p["account_name"] == "spotify"][0]["password"]
    apis.supervisor.complete_task(answer=spotify_password)
    ```"""


def initial_obs_of(item: dict) -> str:
    """初始观察(build 捕获与 live ready 同源构造,episode 开局比对)。"""
    sup = item.get("supervisor") or {}
    who = " ".join(x for x in (sup.get("first_name"), sup.get("last_name")) if x)
    contact = ", ".join(x for x in (sup.get("email"), sup.get("phone_number")) if x)
    head = f"Your supervisor: {who}" + (f" ({contact})" if contact else "")
    return f"{head}\n\nTask: {item['instruction']}"


class WorkerFailure(RuntimeError):
    """worker 协议故障(fatal 事件/超时/进程死亡)——基建故障语义,
    episode 层转 ENV_ERROR。"""


class AppworldEpisode:
    """EnvHandle:一个 live worker 子进程。slot 生命周期由本类全权管理
    (__init__ 首行 acquire,任何路径经幂等 close() 恰好 release 一次
    ——审阅修复 2026-08-07:此前 factory 与本类各 release 一处,ready
    失败路径双 release 使 RAM 门逐次膨胀失效)。"""

    def __init__(self, task_id: str, cfg: AppworldCfg, slots: threading.Semaphore) -> None:
        self._cfg = cfg
        self._slots = slots
        self.closed = False
        self._proc: subprocess.Popen | None = None
        self._buf = ""
        slots.acquire()
        try:
            env = {
                **os.environ,
                "APPWORLD_ROOT": cfg.appworld_root,
                "PYTHONUNBUFFERED": "1",
            }
            self._proc = subprocess.Popen(
                [
                    cfg.worker_python,
                    cfg.worker_script or _WORKER_SCRIPT,
                    task_id,
                    f"trace_{uuid.uuid4().hex[:12]}",
                    "minimal",                  # gt 防火墙:宿主永不调 gold
                    str(cfg.max_interactions),
                    str(cfg.obs_max_chars),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                env=env,
            )
            ready = self._read_event(cfg.start_timeout_s)
            if ready.get("event") != "ready":
                raise WorkerFailure(f"worker did not become ready: {ready}")
            self.ready = ready
            # live 构造初始观察(与 build_dataset 同源函数)——episode
            # 开局与数据集捕获值比对是真实的 drift 检测(审阅修复:此前
            # 回传捕获值,比对恒真空转)。
            self._initial_obs = initial_obs_of({
                "instruction": str(ready.get("instruction", "")),
                "supervisor": ready.get("supervisor") or {},
            })
        except BaseException:
            self.close()
            raise

    # ---- 协议往返 --------------------------------------------------------

    def _read_line(self, timeout_s: float) -> str:
        """fd 级自缓冲行读取:select 看 fd、os.read 进自缓冲——与
        BufferedReader 的预读缓冲彻底解耦(审阅改进 2026-08-07:此前
        select+readline 的正确性依赖「严格一问一答」协议契约;现在 worker
        连发多行也不会造成 select 假阻塞)。绝不再碰 proc.stdout.readline。"""
        fd = self._proc.stdout.fileno()  # type: ignore[union-attr]
        deadline = time.monotonic() + timeout_s
        while "\n" not in self._buf:
            remain = deadline - time.monotonic()
            if remain <= 0:
                raise WorkerFailure(f"worker event timeout after {timeout_s:.0f}s")
            ready, _, _ = select.select([fd], [], [], remain)
            if not ready:
                raise WorkerFailure(f"worker event timeout after {timeout_s:.0f}s")
            chunk = os.read(fd, 65536)
            if not chunk:
                raise WorkerFailure(
                    f"worker exited rc={self._proc.poll()} mid-protocol"
                )
            self._buf += chunk.decode("utf-8", errors="replace")
        line, self._buf = self._buf.split("\n", 1)
        return line

    def _read_event(self, timeout_s: float) -> dict:
        line = self._read_line(timeout_s)
        try:
            event = json.loads(line)
        except json.JSONDecodeError as err:
            raise WorkerFailure(f"malformed worker event: {line[:200]!r}") from err
        if event.get("event") == "fatal":
            raise WorkerFailure(f"worker fatal: {str(event.get('error'))[:800]}")
        return event

    def _request(self, payload: dict, timeout_s: float) -> dict:
        if self._proc.poll() is not None or self._proc.stdin is None:
            raise WorkerFailure("worker process is gone")
        self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        return self._read_event(timeout_s)

    # ---- 环境操作(react/replay 消费;core 不直接调) --------------------

    @property
    def initial_obs(self) -> str:
        return self._initial_obs

    def execute(self, code: str) -> tuple[str, bool]:
        ev = self._request({"op": "execute", "code": code}, self._cfg.exec_timeout_s)
        if ev.get("event") != "execute_result":
            raise WorkerFailure(f"unexpected event for execute: {ev}")
        return str(ev.get("output", "")), bool(ev.get("completed"))

    def evaluate(self) -> tuple[bool, dict]:
        # 终局一次性评估:大任务的 unit-test 走查可能远慢于单步 execute,
        # 用独立宽超时(审阅改进;超时仍 fail-loud 转 ENV_ERROR 不误判)。
        ev = self._request({"op": "evaluate"}, self._cfg.eval_timeout_s)
        if ev.get("event") != "evaluation":
            raise WorkerFailure(f"unexpected event for evaluate: {ev}")
        return bool(ev.get("success")), dict(ev.get("report") or {})

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if (self._proc is not None and self._proc.poll() is None
                    and self._proc.stdin is not None):
                self._proc.stdin.write("__CLOSE__\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        finally:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.kill()
            self._slots.release()


def _find_requirement_list(obj, names: tuple[str, ...]):
    """递归找 requirement 列表(worker 的 report 可能是顶层键,也可能藏在
    TestTracker.to_dict() 的 detail 嵌套里——审阅改进:两种形态都吃)。"""
    if isinstance(obj, dict):
        for n in names:
            v = obj.get(n)
            if isinstance(v, list):
                return v
        for v in obj.values():
            found = _find_requirement_list(v, names)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_requirement_list(v, names)
            if found is not None:
                return found
    return None


def _partial_score(success: bool, report: dict) -> float:
    """score = requirement 通过比例(拿得到明细时);兜底二元(TGC)。"""
    passes = _find_requirement_list(report, ("passes", "passed_tests"))
    failures = _find_requirement_list(report, ("failures", "failed_tests"))
    if passes is not None and failures is not None:
        total = len(passes) + len(failures)
        if total > 0:
            return len(passes) / total
    return 1.0 if success else 0.0


class AppworldTaskAdapter:
    """One adapter per task. item = extra_info(task_id/instruction/
    supervisor/difficulty/initial_obs,build_dataset 捕获)。"""

    def __init__(self, item: dict, cfg: AppworldCfg, slots: threading.Semaphore) -> None:
        self._item = dict(item)
        self._cfg = cfg
        self._slots = slots

    @property
    def task_id(self) -> str:
        return str(self._item["task_id"])

    @property
    def task_type(self) -> str:
        """难度桶(旧实现同款;per-difficulty 曲线/SFT 分层)。"""
        d = self._item.get("difficulty")
        return f"difficulty_{d}" if d not in (None, "") else "unknown"

    @property
    def initial_obs(self) -> str:
        return str(self._item["initial_obs"])

    def task_brief(self) -> str:
        return (
            "Environment: AppWorld — the executor operates a STATEFUL "
            "Python REPL (variables persist across turns) to complete a "
            "supervisor's digital task across simulated apps (email, "
            "payments, shopping, music, notes, ...). Each turn it writes "
            "one fenced Python code block; the environment returns printed "
            "output or the error traceback. It calls app APIs as "
            "apis.<app>.<api>(...) and can discover documentation at "
            "runtime through the api_docs meta-APIs.\n"
            "Completion contract: the episode ends ONLY when the executor "
            "calls apis.supervisor.complete_task(answer=...) (answer only "
            "when the task asks a question). Grading checks the FINAL "
            "DATABASE STATE (plus the answer): the requested change must "
            "be made exactly, and unrelated data must not be modified "
            "(collateral changes fail requirements). Score is the fraction "
            "of programmatic requirements passed (1.0 = full success); an "
            "episode that never declares completion scores 0. The executor "
            "never sees any grading feedback.\n"
            f"Task instruction: {self._item['instruction']}\n"
            f"Budget: {self._cfg.max_interactions} code executions per "
            "episode."
        )

    def executor_scaffold(self) -> str:
        return SYSTEM_PROMPT

    def tools(self) -> list[dict] | None:
        """fenced-code 协议(代码即动作),非 FC:无 tools。"""
        return None

    def gt_text(self) -> str | None:
        """gold solution 存在于数据包但宿主不取(GT 防火墙;现行全线
        gt=off)。要启用时走 build 捕获另议。"""
        return None

    def domain_notes(self) -> str:
        return ""

    def open_episode(self) -> AppworldEpisode:
        """RAM-bound slot 覆盖 episode 全生命期(每 world ~300-500MB);
        slot 记账由 AppworldEpisode 单一负责。"""
        return AppworldEpisode(self.task_id, self._cfg, self._slots)

    # ---- 交互面(fenced-code react) -------------------------------------

    def react(self, env: AppworldEpisode, assistant_text: str) -> ActionOutcome:
        blocks = _FENCE_RE.findall(assistant_text or "")
        thought = _FENCE_RE.sub("", assistant_text or "").strip()
        if not blocks or not blocks[-1].strip():
            return ActionOutcome(
                thought=assistant_text.strip(),
                action_raw="",
                action_norm="",
                feedback=FORMAT_REMINDER,
                done=False,
                score=0.0,
                acted=False,
                valid=False,
                consumed_env_step=False,
            )
        code = blocks[-1].strip()
        output, completed = env.execute(code)
        done, score = False, 0.0
        if completed:
            success, report = env.evaluate()
            done = True
            score = _partial_score(success, report)
        # 红线:评估信息不进反馈——completed 轮的 feedback 仍是 execute
        # 的原始输出(complete_task 的打印),不附加任何评分内容。
        return ActionOutcome(
            thought=thought or assistant_text.strip(),
            action_raw=code,
            action_norm=code,
            feedback=output,
            done=done,
            score=score,
            acted=True,
            valid="Traceback (most recent call last)" not in output,
            consumed_env_step=True,
        )

    def replay_action(self, env: AppworldEpisode, action_norm: str) -> tuple[str, bool]:
        output, completed = env.execute(action_norm)
        return output, completed

    # ---- 文本动作协议件(不适用;平凡实现) ------------------------------

    def normalize_action(self, raw: str) -> str:
        return raw

    def observation_indicates_invalid(self, observation: str) -> bool:
        return False

    def format_reminder(self) -> str:
        return FORMAT_REMINDER


class AppworldEnvFactory:
    """EnvFactory:进程级一个;持 RAM-bound engine slots(per-worker
    折算由 registry builder 做)。"""

    def __init__(self, cfg: AppworldCfg) -> None:
        if not cfg.appworld_root or not cfg.worker_python:
            raise ValueError(
                "appworld config incomplete: appworld_root and worker_python "
                "are required (set APPWORLD_ROOT / APPWORLD_WORKER_PYTHON; "
                "the worker interpreter must be a py>=3.11 venv with the "
                "appworld package installed)"
            )
        self._cfg = cfg
        self._slots = threading.Semaphore(cfg.engine_slots)

    def adapter_from_item(self, item: dict) -> AppworldTaskAdapter:
        return AppworldTaskAdapter(item, self._cfg, self._slots)
