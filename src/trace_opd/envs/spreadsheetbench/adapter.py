"""SpreadsheetBench TaskAdapter — native function-calling executor.

用户裁定(2026-08-07):SSB 不用 `<action>` 文本协议(那是 ALFWorld 特定
形态),executor 走 Qwen 原生 function-calling:两个工具
``run_python(code)`` / ``submit()``。react 自写(协议抽象的预期用法),
解析用 core 通用件 ``protocol.parse_tool_calls``。

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
import shutil
import subprocess
import sys
import tempfile
import threading

from ...common.config import SpreadsheetbenchCfg
from ...core import protocol
from ...core.adapter import ActionOutcome
from .evaluator import evaluate, find_test_cases, preview_workbook

#: 无合法 tool call 时的提醒(本环境习语;裸文本——assistant 未发起调用,
#: 反馈不是 tool 响应)
FORMAT_REMINDER = (
    "(No valid tool call found. Reply with "
    '<tool_call>{"name": <function-name>, "arguments": <args-json-object>}'
    "</tool_call> using one of: run_python, submit.)"
)

_TOOLS_BLOCK = """# Tools

You may call one function per turn to interact with the environment:

<tools>
{"type": "function", "function": {"name": "run_python", "description": "Execute a Python script in the working directory (openpyxl is available). Returns stdout and stderr.", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source code to execute"}}, "required": ["code"]}}}
{"type": "function", "function": {"name": "submit", "description": "Finish the task. Call this only after output.xlsx has been written; the episode ends.", "parameters": {"type": "object", "properties": {}, "required": []}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

# 公式陷阱三要素文案(因果机理 + MUST 规则 + 字面指令覆盖授权)来自旧项目
# 48 探针实证收敛(skills_evolve explorer_trajectories_review/
# 06_ss_global_residual_48probe_ledger.md,probe_046 generic 段):泛泛的
# "写静态值"指令不足以让 agent 放弃写公式(53647 连败 23 探针,签名恒
# pred=None),显式解释评估器行为后才翻转(49036 同款 8/8)。勿削弱措辞。
SYSTEM_PROMPT = (
    "You are a spreadsheet expert who manipulates Excel workbooks with "
    "Python. Your goal is functional correctness: output.xlsx must contain "
    "the correct final values in the specified cells.\n\n"
    "Work in the current working directory:\n"
    "- Read the input workbook from input.xlsx\n"
    "- Write the completed workbook to output.xlsx\n"
    "- Only modify cells within the answer position given in the task; keep "
    "everything else unchanged.\n\n"
    "CRITICAL CONSTRAINT: The evaluation harness does NOT re-evaluate Excel "
    "formulas. It reads the static cell values as they exist in the file. "
    "Writing a formula string (e.g. '=IF(...)') into a cell will result in "
    "that cell being read back as empty/None by the evaluator, causing "
    "failure.\n\n"
    "RULE: For any cell that requires a calculation or conditional logic, "
    "you MUST pre-compute the result in Python and write the STATIC VALUE "
    "directly to the cell. Do NOT write Excel formulas for calculated "
    "results — even when the task instruction mentions formulas, what is "
    "graded is the final cell values.\n\n"
    + _TOOLS_BLOCK
)


def initial_obs_of(item: dict, preview: str) -> str:
    """初始观察(build_dataset 捕获与 live episode 同源,逐字节一致)。"""
    return (
        f"Task: {item['instruction']}\n\n"
        f"instruction_type: {item['instruction_type']}\n"
        f"answer_position: {item['answer_position']}\n"
        "The input workbook is at input.xlsx; save your result to "
        "output.xlsx.\n\n"
        "# Workbook preview (first rows of each sheet)\n"
        f"{preview}"
    )


class SpreadsheetEpisode:
    """EnvHandle:一个隔离沙箱目录,内含 case1 input 的拷贝。

    交互执行经 adapter.react(run_python 子进程);本类只管目录生命周期。
    close() 幂等。"""

    def __init__(self, case1_input: str, initial_obs: str) -> None:
        self.workdir = tempfile.mkdtemp(prefix="ssb_ep_")
        shutil.copy(case1_input, os.path.join(self.workdir, "input.xlsx"))
        self._initial_obs = initial_obs
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

    def task_brief(self) -> str:
        n_cases = len(self._test_cases())
        return (
            "Environment: a spreadsheet manipulation task (SpreadsheetBench). "
            "The executor works in a sandbox directory containing input.xlsx "
            "(a copy of the task workbook) and must produce output.xlsx.\n"
            "Executor tools, one call per turn:\n"
            "  run_python(code) — run a Python script (openpyxl available) in "
            "the sandbox\n"
            "  submit() — finish; the produced output.xlsx is scored and the "
            "episode ends\n"
            f"Scoring: cell values in the answer region(s) "
            f"[{self._item['answer_position']}] are compared against the "
            f"expected result on {n_cases} hidden test case(s); the score is "
            "the fraction passed (1.0 = success).\n"
            "Evaluation mechanic (critical for any guidance you write): the "
            "checker does NOT re-evaluate Excel formulas — it reads static "
            "cell values, so a formula written to a cell is read back as "
            "empty/None and fails. Correct solutions pre-compute results in "
            "Python and write plain values, even when the task instruction "
            "mentions formulas.\n"
            f"Task instruction ({self._item['instruction_type']}): "
            f"{self._item['instruction']}\n"
            f"Budget: the executor has {protocol.MAX_ENV_STEPS} tool calls "
            "per episode."
        )

    def executor_scaffold(self) -> str:
        return SYSTEM_PROMPT

    def gt_text(self) -> str | None:
        """golden workbook 是测试答案本体(非过程参考),绝不进 researcher。"""
        return None

    def domain_notes(self) -> str:
        return ""

    # ---- 环境面 ----------------------------------------------------------

    def open_episode(self) -> SpreadsheetEpisode:
        cases = self._test_cases()
        return SpreadsheetEpisode(cases[0][1], self.initial_obs)

    # ---- 交互面(function-calling react) --------------------------------

    def react(self, env: SpreadsheetEpisode, assistant_text: str) -> ActionOutcome:
        calls = protocol.parse_tool_calls(assistant_text)
        thought = _strip_tool_calls(assistant_text)
        if not calls:
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
        feedback, done, _score, _valid = self._dispatch(env, call, "")
        return feedback, done

    def _dispatch(
        self, env: SpreadsheetEpisode, call: dict, extra_note: str
    ) -> tuple[str, bool, float, bool]:
        """(feedback, done, score, valid) — react 与 replay 共用的执行核。"""
        name = call.get("name")
        if name == "run_python":
            code = str(call.get("arguments", {}).get("code", "") or "")
            if not code.strip():
                return _tool_resp("run_python: empty code." + extra_note), False, 0.0, False
            out_text, rc = self._run_python(env.workdir, code)
            body = out_text if rc == 0 else f"[exit {rc}]\n{out_text}"
            return _tool_resp(self._clip(body) + extra_note), False, 0.0, rc == 0
        if name == "submit":
            if not os.path.exists(env.output_path):
                return (
                    _tool_resp(
                        "submit: output.xlsx not found in the working "
                        "directory — write it first, then submit." + extra_note
                    ),
                    False, 0.0, False,
                )
            score = self._score_output(env.output_path)
            # 信息状态红线:评估结果不反馈(轨迹终态另行记账)。
            return _tool_resp("Submitted." + extra_note), True, score, True
        return (
            _tool_resp(
                f"unknown tool: {name!r}. Available: run_python, submit."
                + extra_note
            ),
            False, 0.0, False,
        )

    def _run_python(self, workdir: str, code: str) -> tuple[str, int]:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": workdir,
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",   # 重放确定性
        }
        with self._gate:
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=workdir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self._cfg.python_timeout_s,
                )
            except subprocess.TimeoutExpired:
                return f"[timeout after {self._cfg.python_timeout_s}s]", 124
        out = (proc.stdout or "") + (proc.stderr or "")
        return out if out.strip() else "(no output)", proc.returncode

    def _score_output(self, output_path: str) -> float:
        """单一 output 对全部 test case 评(旧实现语义);soft = 通过比例。"""
        cases = self._test_cases()
        n_pass = 0
        for _no, _ip, answer_path in cases:
            try:
                if evaluate(output_path, answer_path, self._item["answer_position"])["ok"]:
                    n_pass += 1
            except Exception:  # noqa: BLE001 — 单 case 评估崩溃计失败
                pass
        return n_pass / len(cases)

    def _clip(self, text: str) -> str:
        limit = self._cfg.obs_max_chars
        if len(text) <= limit:
            return text
        head = limit // 2
        tail = limit - head
        return text[:head] + "\n[... output truncated ...]\n" + text[-tail:]

    # ---- 文本动作协议件(SSB 不经 text_action_react;平凡实现) ----------

    def normalize_action(self, raw: str) -> str:
        return raw.strip()

    def observation_indicates_invalid(self, observation: str) -> bool:
        # valid 由 react 直接判定(returncode);该协议件不使用。
        return False

    def format_reminder(self) -> str:
        return FORMAT_REMINDER


def _tool_resp(content: str) -> str:
    """反馈按 Qwen3 chat template 对 tool role 消息的渲染形态包裹,使
    文本流与原生 function-calling 分布对齐(服务器侧真 tokenizer 对拍
    冒烟见 GO 清单;不等价则升级 TemplateDelta 变体)。"""
    return f"<tool_response>\n{content}\n</tool_response>"


def _strip_tool_calls(text: str) -> str:
    return protocol._TOOL_CALL_RE.sub("", text or "").strip()


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
        self._cfg = cfg
        self._gate = threading.Semaphore(cfg.exec_concurrency)

    def adapter_from_item(self, item: dict) -> SpreadsheetbenchTaskAdapter:
        return SpreadsheetbenchTaskAdapter(item, self._cfg, self._gate)
