"""SpreadsheetBench TaskAdapter — 双协议 executor(fncall | text)。

fncall(用户裁定 2026-08-07,Qwen3 系验证档):executor 走 Qwen 原生
function-calling,两个工具 ``run_python(code)`` / ``submit()``,解析用
core 通用件 ``protocol.parse_tool_calls``。

text(用户裁定 2026-08-17,经典线 Qwen2.5 档):```python 围栏 +
```output 回注 + ```submit 终局,one-shot worked example 内嵌 system。
形态与 envs/math/adapter.py 的文本 TIR 档同构(ToRL/SimpleTIR/
Qwen2.5-Math-TIR 家族);裁定依据 = Math 域实测 FC→text +21.6pp(同为
写代码任务)、SQA 域 Qwen2.5 FC 违规率 11.8%、大段代码进 JSON 转义是
Qwen2.5 已证痛点。解析翻译成与 FC 同构的 call dict,`_dispatch` 复用。

信息状态红线(全部继承旧实现纪律):
  * 评估结果永不进 executor 可见流(submit 反馈恒 "Submitted.");
  * golden workbook 只被本文件的评估路径读取——不拷入 episode 沙箱、不进
    extra_info、gt_text=None(golden 是测试答案本体,绝不给 researcher);
  * 所有 feedback/reason 只含 agent 自己的输出,不含期望值。

确定性(v4 前缀续写的硬前提):episode 沙箱用固定相对路径
(input.xlsx/output.xlsx),命令流零绝对路径;子进程环境净化
(PYTHONHASHSEED=0);代码本身含时间/随机时由机制侧逐步校验作废。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading

from ...common.config import SpreadsheetbenchCfg
from ...core import protocol
from ...core.adapter import ActionOutcome
from ..pytool import clip_middle, run_python_script
from .evaluator import evaluate, find_test_cases, preview_workbook  # noqa: F401 — preview 供 live 重算
from .recalc import find_soffice, recalc_remote_url, recalculated_for_grading

#: 无合法 tool call 时的提醒(裸文本——assistant 未发起调用,反馈不是
#: tool 响应)
def _format_reminder() -> str:
    # 调用时求值:措辞随运行时安装的 FC dialect(模块 import 早于
    # configure_fc_dialect,常量会把 hermes 措辞固化进 qwen35_xml 语境)
    return protocol.fc_format_reminder("run_python", "submit")

# 工具 schema(OpenAI 通用 FC 格式,单一真源;用户裁定 2026-08-07):
# 经 adapter.tools() 交 chat template 原生渲染进 prompt(分布内形态,
# 绝不手写渲染块);解析仍在文本层(parse_tool_calls)——生成的原始
# token 原样留在训练流,引擎的结构化 tool_calls 表示从不参与还原。
# 描述纪律:每个工具写全「执行模型(fresh interpreter,无跨调用状态)/
# 前置条件 / 返回语义 / 失败与截断行为」——agent 依赖的行为契约。
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run a Python script in the sandbox working directory and "
                "return its combined stdout and stderr. Each call starts a "
                "fresh interpreter: variables and imports do NOT persist "
                "between calls — every script must be self-contained and "
                "re-read any file it needs. openpyxl is available for "
                "workbook access. A non-zero exit is reported with its exit "
                "code; scripts exceeding the time limit are killed; long "
                "output is truncated in the middle."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Complete Python source to execute as a "
                            "standalone script (not a REPL fragment)."
                        ),
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "Declare the task finished and end the episode. Call it "
                "only after output.xlsx exists in the working directory; "
                "if output.xlsx is missing, the episode continues and you "
                "are told so. Grading happens after submission; no grading "
                "feedback is ever returned."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# 评分前公式重算(官方 open_spreadsheet.py 语义;recalc.py)⇒ 写公式与写
# 静态值都是合法解法。scaffold 只作中性说明(第二代同款口径,用户裁定
# 2026-07-16「公式解法必须可判分」;48 探针台账的"禁公式"文案是第一代
# 环境缺陷下的 workaround,勿回退)。工具块不在此手写——由 chat template
# 按 TOOLS 原生渲染。
SYSTEM_PROMPT = (
    "You are a spreadsheet-manipulation specialist working in a sandboxed "
    "directory. Solve the task by editing a workbook with Python, then "
    "submit.\n\n"
    "Files (fixed names, relative to the working directory):\n"
    "- input.xlsx — the task workbook, already present; read from here.\n"
    "- output.xlsx — your deliverable; create it. Grading reads ONLY this "
    "file.\n\n"
    "How your work is graded:\n"
    "- Only the cells inside the answer position stated in the task are "
    "checked; keep every other cell exactly as it is in input.xlsx.\n"
    "- The workbook is opened and formulas are recalculated before "
    "grading, so writing Excel formulas and writing pre-computed values "
    "are both acceptable.\n"
    "- openpyxl itself never evaluates formulas: reading a formula cell "
    "back in a later script returns the formula string, not its value — "
    "verify formula results by recomputing them in Python, not by reading "
    "them back."
)


# --- 文本协议(2026-08-17 用户裁定:经典线 Qwen2.5 直接 text 化)----------
# 形态 = envs/math/adapter.py 文本 TIR 档同构(ToRL arXiv 2503.23383 /
# SimpleTIR arXiv 2509.02479 / Qwen2.5-Math-TIR 家族):```python 围栏代码
# → 环境回注 ```output\nOBS\n``` → 终局 ```submit```(SSB 与 Math 的差异:
# Math 的终答 \boxed{} 自带内容,SSB 的终局是无参提交声明,取 fence 家族
# 形态与代码块对称、误触发面最小)。one-shot worked example 内嵌 system
# (SearchQA/Math 同款纪律;Qwen2.5 zero-shot 不服从自定义标签,system
# 单示例 0%→99.9% 实证)。多轮承载:代码块作为 assistant 轮结束,执行
# 结果以 user 轮回注(PG-3 评测训练同构)。
SYSTEM_PROMPT_TEXT = (
    "You are a spreadsheet-manipulation specialist working in a sandboxed "
    "directory. Solve the task by editing a workbook with Python, then "
    "submit.\n\n"
    "Files (fixed names, relative to the working directory):\n"
    "- input.xlsx — the task workbook, already present; read from here.\n"
    "- output.xlsx — your deliverable; create it. Grading reads ONLY this "
    "file.\n\n"
    "Each reply, reason first, then do exactly one of the following:\n"
    "(1) Write ONE fenced Python code block — a complete standalone "
    "script (including imports; openpyxl is available) that uses print() "
    "to show what you need to see. Variables do NOT persist between "
    "blocks: every script must re-read any file it needs. End your reply "
    "right after the closing ``` and never guess the output — the real "
    "execution result is returned to you wrapped as:\n"
    "```output\n"
    "...\n"
    "```\n"
    "(2) When output.xlsx is written and verified, end the episode with a "
    "reply that contains no Python block, only the marker:\n"
    "```submit\n"
    "```\n"
    "Grading happens after submission; no grading feedback is ever "
    "returned.\n\n"
    "Worked example — task: put the sum of A2:A4 of Sheet1 into B2.\n"
    "Your reply:\n"
    "I need to see the data first.\n"
    "```python\n"
    "import openpyxl\n"
    "wb = openpyxl.load_workbook('input.xlsx')\n"
    "ws = wb['Sheet1']\n"
    "print([ws.cell(row=r, column=1).value for r in range(2, 5)])\n"
    "```\n"
    "You then receive:\n"
    "```output\n"
    "[10, 20, 30]\n"
    "```\n"
    "Your next reply:\n"
    "The sum is 60. I write it to B2 and save output.xlsx.\n"
    "```python\n"
    "import openpyxl\n"
    "wb = openpyxl.load_workbook('input.xlsx')\n"
    "wb['Sheet1']['B2'] = 60\n"
    "wb.save('output.xlsx')\n"
    "print('saved')\n"
    "```\n"
    "After the output confirms it, your next reply is just:\n"
    "```submit\n"
    "```\n\n"
    "How your work is graded:\n"
    "- Only the cells inside the answer position stated in the task are "
    "checked; keep every other cell exactly as it is in input.xlsx.\n"
    "- The workbook is opened and formulas are recalculated before "
    "grading, so writing Excel formulas and writing pre-computed values "
    "are both acceptable.\n"
    "- openpyxl itself never evaluates formulas: reading a formula cell "
    "back in a later script returns the formula string, not its value — "
    "verify formula results by recomputing them in Python, not by reading "
    "them back."
)

#: 围栏代码块(与 envs/math/adapter.py 的 _TEXT_CODE_RE 同款:要求
#: ```python/```py 标注;裸 ``` 不收——可能是伪代码或输出转贴)。
_TEXT_CODE_RE = re.compile(r"```(?:python|py)\b[^\n]*\n(.*?)```",
                           re.DOTALL | re.IGNORECASE)
#: 终局标记(fence 形态;内容留空,宽松吞吸空白/杂字直到闭栏)。
_TEXT_SUBMIT_RE = re.compile(r"```submit\b[^`]*```", re.IGNORECASE)


def parse_text_action(assistant_text: str) -> dict | None:
    """文本回复 → 与 FC 同构的 call dict(下游 _dispatch 零改动)。

    优先级:**代码块 > submit**(Math 档同理):模型常在同一回复里写
    "运行这段并提交"——那是对未执行结果的乐观断言;先执行代码、忽略
    submit,真实输出回注后模型下一轮再提交,防止代码失败仍白白终局。
    多代码块取**最后一个**(ToRL 同款:执行 latest)。
    """
    t = assistant_text or ""
    blocks = _TEXT_CODE_RE.findall(t)
    if blocks:
        return {"name": "run_python", "arguments": {"code": blocks[-1].strip()}}
    if _TEXT_SUBMIT_RE.search(t):
        return {"name": "submit", "arguments": {}}
    return None


def text_extra_note(assistant_text: str) -> str:
    """同轮多动作的忽略说明(FC 档 extra_note 的文本档对应物):
    多代码块提醒 + 代码/submit 同现提醒(后者尤其重要——不说明的话
    模型会以为已经提交,下一轮凭空等待终局)。"""
    t = assistant_text or ""
    notes = []
    n = len(_TEXT_CODE_RE.findall(t))
    if n > 1:
        notes.append(
            f"{n - 1} earlier code block(s) in the same reply were "
            "ignored — one block per reply, the last one runs"
        )
    if n >= 1 and _TEXT_SUBMIT_RE.search(t):
        notes.append(
            "the ```submit``` marker was ignored because a code block was "
            "present — the code ran instead; to finish, send ```submit``` "
            "alone in a reply without any code block"
        )
    return "".join(f"\n(note: {s})" for s in notes)


def text_thought(assistant_text: str, call: dict | None) -> str:
    """thought 记账:代码轮取最后一个围栏之前的正文;终局/无动作轮取全文。"""
    t = (assistant_text or "").strip()
    if call and call.get("name") == "run_python":
        m = None
        for m in _TEXT_CODE_RE.finditer(assistant_text or ""):
            pass
        if m is not None:
            return (assistant_text[: m.start()]).strip()
    return t


def text_format_reminder() -> str:
    """void turn(无代码亦无终局标记)的格式提醒。"""
    return (
        "\n(Your reply contained neither a ```python code block nor a "
        "```submit``` marker. To work on the task, write one complete "
        "standalone script inside a fenced block that starts with "
        "```python and ends with ```. To finish after output.xlsx is "
        "written, reply with only ```submit``` on its own fenced line. "
        "Let me try again.)\n"
    )


def text_obs_wrap(body: str) -> str:
    """执行结果包裹:ToRL/Qwen2.5-Math-TIR 同款 ```output 围栏。"""
    return f"```output\n{body}\n```"


def initial_obs_of(item: dict, preview: str) -> str:
    """初始观察(build_dataset 捕获与 live episode 同源,逐字节一致)。
    文件路径约定已在 scaffold 声明,此处只给任务变量与工作簿预览。"""
    return (
        f"Task: {item['instruction']}\n\n"
        f"Answer type: {item['instruction_type']}\n"
        f"Answer position: {item['answer_position']}\n\n"
        "# Preview of input.xlsx (first rows of each sheet)\n"
        f"{preview}"
    )


class SpreadsheetEpisode:
    """EnvHandle:一个隔离沙箱目录,内含 case1 input 的拷贝。

    交互执行经 adapter.react(run_python 子进程);本类只管目录生命周期。
    close() 幂等。"""

    def __init__(self, case1_input: str, item: dict) -> None:
        self.workdir = tempfile.mkdtemp(prefix="ssb_ep_")
        shutil.copy(case1_input, os.path.join(self.workdir, "input.xlsx"))
        # live 重算初始观察(build_dataset 同源函数)——episode 开局与
        # 数据集捕获值的比对是真实 drift 检测(xlsx 文件或 openpyxl 版本
        # 漂移会被抓;审阅修复 2026-08-07:此前回传捕获值,比对恒真)。
        self._initial_obs = initial_obs_of(item, preview_workbook(case1_input))
        self.closed = False

    @property
    def initial_obs(self) -> str:
        return self._initial_obs

    @property
    def output_path(self) -> str:
        return os.path.join(self.workdir, "output.xlsx")

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            shutil.rmtree(self.workdir, ignore_errors=True)


class SpreadsheetbenchTaskAdapter:
    """One adapter per task. item = parquet extra_info(自足契约)。"""

    def __init__(
        self, item: dict, cfg: SpreadsheetbenchCfg, exec_gate: threading.Semaphore
    ) -> None:
        self._item = dict(item)
        self._cfg = cfg
        self._gate = exec_gate
        self._cases: list[tuple[str, str, str]] | None = None

    # ---- 身份面 ----------------------------------------------------------

    @property
    def task_id(self) -> str:
        return str(self._item["task_id"])

    @property
    def task_type(self) -> str:
        return str(self._item["instruction_type"])

    @property
    def initial_obs(self) -> str:
        return str(self._item["initial_obs"])

    # ---- 数据资产 --------------------------------------------------------

    def _task_dir(self) -> str:
        if not self._cfg.data_root:
            raise RuntimeError(
                "spreadsheetbench.data_root is required (set SSB_DATA)"
            )
        return os.path.join(self._cfg.data_root, str(self._item["spreadsheet_dir"]))

    def _test_cases(self) -> list[tuple[str, str, str]]:
        if self._cases is None:
            self._cases = find_test_cases(self._task_dir())
            if not self._cases:
                raise RuntimeError(
                    f"no test cases found for task {self.task_id} "
                    f"under {self._task_dir()}"
                )
        return self._cases

    # ---- researcher 面 ---------------------------------------------------

    def _text_mode(self) -> bool:
        return getattr(self._cfg, "protocol", "fncall") == "text"

    def task_brief(self) -> str:
        n_cases = len(self._test_cases())
        if self._text_mode():
            actions = (
                "Executor actions (one per reply):\n"
                "  (1) one fenced ```python code block — a complete "
                "standalone script run in the sandbox (openpyxl available; "
                "fresh interpreter each time, no state carries over); its "
                "printed output is returned wrapped in an ```output block\n"
                "  (2) a fenced ```submit``` marker in a reply without any "
                "code block — ends the episode; the produced output.xlsx "
                "is then graded\n"
            )
        else:
            actions = (
                "Executor tools (one call per turn):\n"
                "  run_python(code) — run a standalone Python script in the "
                "sandbox (openpyxl available; fresh interpreter per call, no "
                "state carries over between calls)\n"
                "  submit() — end the episode; the produced output.xlsx is "
                "then graded\n"
            )
        return (
            "Environment: SpreadsheetBench — one Excel-workbook manipulation "
            "task. The executor works in a sandbox directory holding "
            "input.xlsx (a copy of the task workbook) and must produce "
            "output.xlsx.\n"
            + actions +
            f"Grading: cell values in the answer region(s) "
            f"[{self._item['answer_position']}] are compared against the "
            f"expected result on {n_cases} test case(s); score = fraction "
            "of cases passed (1.0 = success). The sandbox holds test case "
            "1; the single submitted output.xlsx is graded against every "
            "case. The executor never sees any grading feedback "
            "(\"Submitted.\" is all it gets back).\n"
            "Evaluation mechanic: the produced workbook is opened and "
            "formulas are recalculated before grading, so Excel formulas "
            "and pre-computed static values are both valid solution "
            "styles. Caveat worth teaching: openpyxl never evaluates "
            "formulas — reading a formula cell back inside the sandbox "
            "returns the formula string, not its value.\n"
            f"Task instruction ({self._item['instruction_type']}): "
            f"{self._item['instruction']}\n"
            f"Budget: the executor has {protocol.MAX_ENV_STEPS} "
            + ("actions (code executions or submit)"
               if self._text_mode() else "tool calls")
            + " per episode."
        )

    def executor_scaffold(self) -> str:
        if self._text_mode():
            return SYSTEM_PROMPT_TEXT
        return SYSTEM_PROMPT

    def tools(self) -> list[dict] | None:
        # 文本协议不渲染 tools(否则 chat template 同时教 FC 语法,与
        # 围栏协议打架;Math/SearchQA 同款处理)。
        if self._text_mode():
            return None
        return TOOLS

    def gt_text(self) -> str | None:
        """golden workbook 是测试答案本体(非过程参考),绝不进 researcher。"""
        return None

    def domain_notes(self) -> str:
        return ""

    # ---- 环境面 ----------------------------------------------------------

    def open_episode(self) -> SpreadsheetEpisode:
        cases = self._test_cases()
        return SpreadsheetEpisode(cases[0][1], self._item)

    # ---- 交互面(fncall | text react)------------------------------------

    def react(self, env: SpreadsheetEpisode, assistant_text: str) -> ActionOutcome:
        if self._text_mode():
            return self._react_text(env, assistant_text)
        return self._react_fncall(env, assistant_text)

    def _react_text(self, env: SpreadsheetEpisode, assistant_text: str) -> ActionOutcome:
        """文本协议(Math 文本 TIR 档同构)。执行与判分完全复用
        _dispatch —— 只是把围栏代码/```submit 翻译成同构 call dict。"""
        call = parse_text_action(assistant_text)
        if call is None:
            return ActionOutcome(
                thought=assistant_text.strip(),
                action_raw="",
                action_norm="",
                feedback=text_format_reminder(),
                done=False,
                score=0.0,
                acted=False,
                valid=False,
                consumed_env_step=False,
            )
        feedback, done, score, valid = self._dispatch(
            env, call, text_extra_note(assistant_text), text_mode=True
        )
        return ActionOutcome(
            thought=text_thought(assistant_text, call) or assistant_text.strip(),
            action_raw=json.dumps(call, ensure_ascii=False),
            action_norm=_norm_call(call),
            feedback=feedback,
            done=done,
            score=score,
            acted=True,
            valid=valid,
            consumed_env_step=True,
        )

    def _react_fncall(self, env: SpreadsheetEpisode, assistant_text: str) -> ActionOutcome:
        calls = protocol.parse_tool_calls(assistant_text, tools=self.tools())
        thought = protocol.strip_tool_calls(assistant_text)
        if not calls:
            return ActionOutcome(
                thought=assistant_text.strip(),
                action_raw="",
                action_norm="",
                feedback=_format_reminder(),
                done=False,
                score=0.0,
                acted=False,
                valid=False,
                consumed_env_step=False,
            )
        call = calls[0]
        extra_note = (
            f"\n(note: {len(calls) - 1} extra tool call(s) in the same reply "
            "were ignored — one call per turn)"
            if len(calls) > 1 else ""
        )
        outcome = self._dispatch(env, call, extra_note)
        return ActionOutcome(
            thought=thought or assistant_text.strip(),
            action_raw=json.dumps(call, ensure_ascii=False),
            action_norm=_norm_call(call),
            feedback=outcome[0],
            done=outcome[1],
            score=outcome[2],
            acted=True,
            valid=outcome[3],
            consumed_env_step=True,
        )

    def replay_action(self, env: SpreadsheetEpisode, action_norm: str) -> tuple[str, bool]:
        call = json.loads(action_norm)
        feedback, done, _score, _valid = self._dispatch(
            env, call, "", text_mode=self._text_mode()
        )
        return feedback, done

    def _dispatch(
        self, env: SpreadsheetEpisode, call: dict, extra_note: str,
        text_mode: bool = False,
    ) -> tuple[str, bool, float, bool]:
        """(feedback, done, score, valid) — react 与 replay 共用的执行核。

        观察包裹随协议切换:FC 用 <tool_response>,文本档用 ```output 围栏
        (ToRL 同款)。文本档的 extra_note 放围栏**外**——放进去会伪装成
        程序输出(Math 档同款处理)。"""
        if text_mode:
            def wrap_obs(body: str) -> str:
                return text_obs_wrap(body) + extra_note

            def wrap_msg(body: str) -> str:
                return body + extra_note
        else:
            def wrap_obs(body: str) -> str:
                return protocol.tool_response_wrap(body + extra_note)

            wrap_msg = wrap_obs
        name = call.get("name")
        if name == "run_python":
            code = str(call.get("arguments", {}).get("code", "") or "")
            if not code.strip():
                return wrap_msg("run_python: empty code."), False, 0.0, False
            out_text, rc = self._run_python(env.workdir, code)
            body = out_text if rc == 0 else f"[exit {rc}]\n{out_text}"
            return wrap_obs(self._clip(body)), False, 0.0, rc == 0
        if name == "submit":
            if not os.path.exists(env.output_path):
                return (
                    wrap_msg(
                        "submit: output.xlsx not found in the working "
                        "directory — write it first, then submit."
                    ),
                    False, 0.0, False,
                )
            score = self._score_output(env.output_path)
            # 信息状态红线:评估结果不反馈(轨迹终态另行记账)。
            return wrap_msg("Submitted."), True, score, True
        return (
            wrap_msg(f"unknown tool: {name!r}. Available: run_python, submit."),
            False, 0.0, False,
        )

    def _run_python(self, workdir: str, code: str) -> tuple[str, int]:
        return run_python_script(
            code, cwd=workdir, timeout_s=self._cfg.python_timeout_s, gate=self._gate
        )

    def _score_output(self, output_path: str) -> float:
        """单一 output 对全部 test case 评;soft = 通过比例。

        评分前按官方语义重算公式(recalc.py;重算一次、全 case 复用——
        重算与 case 无关)。重算 soffice 子进程共用 run_python 的并发门
        (同为重 CPU 工作)。golden 不重算(官方发布件自带缓存值)。"""
        with open(output_path, "rb") as f:
            raw = f.read()
        with self._gate:
            graded = recalculated_for_grading(
                raw,
                soffice=find_soffice(),
                timeout_s=self._cfg.recalc_timeout_s,
                task_id=self.task_id,
            )
        cases = self._test_cases()
        n_pass = 0
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp.write(graded)
            graded_path = tmp.name
        try:
            for _no, _ip, answer_path in cases:
                try:
                    if evaluate(
                        graded_path, answer_path, self._item["answer_position"]
                    )["ok"]:
                        n_pass += 1
                except Exception:  # noqa: BLE001 — 单 case 评估崩溃计失败
                    pass
        finally:
            os.unlink(graded_path)
        return n_pass / len(cases)

    def _clip(self, text: str) -> str:
        return clip_middle(text, self._cfg.obs_max_chars)

    # ---- 文本动作协议件(SSB 不经 text_action_react;平凡实现) ----------

    def normalize_action(self, raw: str) -> str:
        return raw.strip()

    def observation_indicates_invalid(self, observation: str) -> bool:
        # valid 由 react 直接判定(returncode);该协议件不使用。
        return False

    def format_reminder(self) -> str:
        if self._text_mode():
            return text_format_reminder()
        return _format_reminder()




def _norm_call(call: dict) -> str:
    return json.dumps(call, ensure_ascii=False, sort_keys=True)


class SpreadsheetbenchEnvFactory:
    """EnvFactory:进程级一个实例;持 python 子进程并发门(CPU 饥饿教训:
    环境子进程抢光 CPU 会饿死推理引擎,门限走配置并按 worker 折算)。"""

    def __init__(self, cfg: SpreadsheetbenchCfg) -> None:
        if not cfg.data_root:
            raise ValueError(
                "spreadsheetbench config incomplete: data_root is required "
                "(set SSB_DATA)"
            )
        if find_soffice() is None and recalc_remote_url() is None:
            # 装配期 fail loud(比第二代运行时降级更严):没有重算后端,
            # 公式解法会被静默判 0 —— 那是 reward 语义漂移,不是可降级
            # 的小故障。设 SSB_RECALC_URL(Windows Excel 服务,与 golden
            # 同引擎,优先)或装 LibreOffice / 设 SSB_SOFFICE 后再起。
            raise ValueError(
                "spreadsheetbench: no formula-recalc backend — set "
                "SSB_RECALC_URL (Windows Excel service, preferred) or "
                "install libreoffice/soffice into PATH or set SSB_SOFFICE "
                "(grading without recalc would silently fail every formula "
                "answer)"
            )
        self._cfg = cfg
        self._gate = threading.Semaphore(cfg.exec_concurrency)

    def adapter_from_item(self, item: dict) -> SpreadsheetbenchTaskAdapter:
        return SpreadsheetbenchTaskAdapter(item, self._cfg, self._gate)
