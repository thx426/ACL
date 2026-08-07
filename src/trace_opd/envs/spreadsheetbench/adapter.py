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
import tempfile
import threading

from ...common.config import SpreadsheetbenchCfg
from ...core import protocol
from ...core.adapter import ActionOutcome
from ..pytool import clip_middle, run_python_script
from .evaluator import evaluate, find_test_cases, preview_workbook  # noqa: F401 — preview 供 live 重算
from .recalc import find_soffice, recalculated_for_grading

#: 无合法 tool call 时的提醒(裸文本——assistant 未发起调用,反馈不是
#: tool 响应)
FORMAT_REMINDER = protocol.fc_format_reminder("run_python", "submit")

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

    def task_brief(self) -> str:
        n_cases = len(self._test_cases())
        return (
            "Environment: SpreadsheetBench — one Excel-workbook manipulation "
            "task. The executor works in a sandbox directory holding "
            "input.xlsx (a copy of the task workbook) and must produce "
            "output.xlsx.\n"
            "Executor tools (one call per turn):\n"
            "  run_python(code) — run a standalone Python script in the "
            "sandbox (openpyxl available; fresh interpreter per call, no "
            "state carries over between calls)\n"
            "  submit() — end the episode; the produced output.xlsx is then "
            "graded\n"
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
            f"Budget: the executor has {protocol.MAX_ENV_STEPS} tool calls "
            "per episode."
        )

    def executor_scaffold(self) -> str:
        return SYSTEM_PROMPT

    def tools(self) -> list[dict] | None:
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

    # ---- 交互面(function-calling react) --------------------------------

    def react(self, env: SpreadsheetEpisode, assistant_text: str) -> ActionOutcome:
        calls = protocol.parse_tool_calls(assistant_text)
        thought = protocol.strip_tool_calls(assistant_text)
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
        return FORMAT_REMINDER


def _tool_resp(content: str) -> str:
    """反馈按 Qwen3 chat template 对 tool role 消息的渲染形态包裹,使
    文本流与原生 function-calling 分布对齐(服务器侧真 tokenizer 对拍
    冒烟见 GO 清单;不等价则升级 TemplateDelta 变体)。"""
    return f"<tool_response>\n{content}\n</tool_response>"


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
        if find_soffice() is None:
            # 装配期 fail loud(比第二代运行时降级更严):没有重算后端,
            # 公式解法会被静默判 0 —— 那是 reward 语义漂移,不是可降级
            # 的小故障。装 LibreOffice 或设 SSB_SOFFICE 后再起。
            raise ValueError(
                "spreadsheetbench: no LibreOffice backend found for formula "
                "recalculation — install libreoffice/soffice into PATH or "
                "set SSB_SOFFICE (grading without recalc would silently "
                "fail every formula answer)"
            )
        self._cfg = cfg
        self._gate = threading.Semaphore(cfg.exec_concurrency)

    def adapter_from_item(self, item: dict) -> SpreadsheetbenchTaskAdapter:
        return SpreadsheetbenchTaskAdapter(item, self._cfg, self._gate)
