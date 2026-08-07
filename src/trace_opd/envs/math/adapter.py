"""Math TaskAdapter — tool-integrated reasoning over competition problems.

用户四裁定(2026-08-07):DAPO-17k 全量训练 / 六源 valid greedy@1 / 预算
12 工具步 + 单轮 8192 / researcher 沿用 n_exp=16。协议与 SSB 同构:原生
function-calling(OpenAI 格式 tools() + 模板渲染 + 文本层解析),两工具:

  run_python(code)   — 沙箱计算(数值/sympy/枚举验证;envs/pytool 共享件)
  submit(answer)     — 终局:answer 参数即最终答案,math-verify 等价判分

信息状态红线:ground_truth 随 extra_info 携带(评分必需),但只被本文件
的评分路径读取——零进提示词/反馈/researcher(gt_text=None:答案即测试
答案本体);submit 反馈恒 "Submitted."。

环境无状态:episode 沙箱只是 run_python 的 cwd(临时目录,空);重放 =
重跑工具调用(PYTHONHASHSEED=0,确定性 ⇒ v4 前缀续写零障碍)。
"""
from __future__ import annotations

import json
import shutil
import tempfile
import threading

from ...common.config import MathCfg
from ...core import protocol
from ...core.adapter import ActionOutcome
from ..pytool import clip_middle, run_python_script
from .verifier import answers_equivalent, ensure_math_verify

#: 无合法 tool call 时的提醒(裸文本;协议说明由模板渲染的工具块承担)
FORMAT_REMINDER = (
    "(Your reply contained no valid tool call. End your message with "
    "exactly one block of the form "
    '<tool_call>{"name": <function-name>, "arguments": <args-json-object>}'
    "</tool_call>. Available functions: run_python, submit.)"
)

# 工具 schema(OpenAI 格式单一真源,模板原生渲染;描述纪律:执行模型/
# 前置条件/返回语义/失败与截断行为写全)。
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run a Python script in a scratch directory and return its "
                "combined stdout and stderr. Use it to compute, simplify, "
                "enumerate, or verify intermediate results (sympy is "
                "available if installed in the runtime). Each call starts a "
                "fresh interpreter: variables do NOT persist between calls "
                "— every script must be self-contained and print what you "
                "need to see. A non-zero exit is reported with its exit "
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
                "Submit the final answer and end the episode. The answer "
                "is graded for mathematical equivalence, so give the exact "
                "value in its simplest form (an integer, a fraction, or a "
                "LaTeX expression) — not a decimal approximation unless "
                "the problem asks for one. No grading feedback is ever "
                "returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": (
                            "The final answer only — no reasoning, no "
                            "'Answer:' prefix, no surrounding sentence."
                        ),
                    },
                },
                "required": ["answer"],
            },
        },
    },
]

# 长推理主体写在 tool call 之前的文本部分(FC 形态天然支持);不需计算的
# 题可直接首轮 submit。
SYSTEM_PROMPT = (
    "You are a competition mathematician. Solve the given problem with "
    "rigorous step-by-step reasoning, then submit the final answer.\n\n"
    "Working style:\n"
    "- Write your reasoning as normal text before any tool call.\n"
    "- Use run_python to compute, enumerate, or verify anything "
    "error-prone — arithmetic slips are the most common way correct "
    "reasoning produces a wrong answer.\n"
    "- When the answer is established, call submit with the exact value "
    "in simplest form. Submitting ends the episode."
)


def initial_obs_of(item: dict) -> str:
    """初始观察 = 纯题面(确定性锚点;求解纪律在 scaffold)。"""
    return str(item["problem"])


class MathEpisode:
    """EnvHandle:一个空的 scratch 目录(run_python 的 cwd)。环境本身
    无状态——数学任务的全部状态在推理文本里;无外部资产 ⇒ 无 drift 源,
    initial_obs 传入即 live(SSB/AppWorld 则须 live 重构造)。close() 幂等。"""

    def __init__(self, initial_obs: str) -> None:
        self.workdir = tempfile.mkdtemp(prefix="math_ep_")
        self._initial_obs = initial_obs
        self.closed = False

    @property
    def initial_obs(self) -> str:
        return self._initial_obs

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            shutil.rmtree(self.workdir, ignore_errors=True)


class MathTaskAdapter:
    """One adapter per problem. item = parquet extra_info(自足:problem/
    ground_truth/task_id/task_type/split)。"""

    def __init__(self, item: dict, cfg: MathCfg, exec_gate: threading.Semaphore) -> None:
        self._item = dict(item)
        self._cfg = cfg
        self._gate = exec_gate

    # ---- 身份面 ----------------------------------------------------------

    @property
    def task_id(self) -> str:
        return str(self._item["task_id"])

    @property
    def task_type(self) -> str:
        """训练集无自然分类 ⇒ 常量 "math"(协议允许);valid 侧 =
        benchmark 名(aime24/... ,build_dataset 写入)。"""
        return str(self._item.get("task_type") or "math")

    @property
    def initial_obs(self) -> str:
        return str(self._item["initial_obs"])

    # ---- researcher 面 ---------------------------------------------------

    def task_brief(self) -> str:
        return (
            "Environment: one competition math problem. The executor "
            "reasons in text and may call tools (one call per turn):\n"
            "  run_python(code) — run a standalone Python script in a "
            "scratch directory (fresh interpreter per call, no state "
            "carries over); for computing, enumerating, or verifying "
            "intermediate results\n"
            "  submit(answer) — submit the final answer and end the "
            "episode\n"
            "Grading: the submitted answer is checked for mathematical "
            "equivalence against the reference (exact value; equivalent "
            "forms such as 1/2 vs 0.5 vs \\frac{1}{2} all count). Score "
            "is binary. The executor never sees any grading feedback "
            "(\"Submitted.\" is all it gets back).\n"
            f"Problem: {self._item['problem']}\n"
            f"Budget: the executor has {protocol.MAX_ENV_STEPS} tool calls "
            "per episode; reasoning between calls is free-form text."
        )

    def executor_scaffold(self) -> str:
        return SYSTEM_PROMPT

    def tools(self) -> list[dict] | None:
        return TOOLS

    def gt_text(self) -> str | None:
        """ground_truth 即测试答案本体——绝不进 researcher。"""
        return None

    def domain_notes(self) -> str:
        return ""

    # ---- 环境面 ----------------------------------------------------------

    def open_episode(self) -> MathEpisode:
        return MathEpisode(self.initial_obs)

    # ---- 交互面 ----------------------------------------------------------

    def react(self, env: MathEpisode, assistant_text: str) -> ActionOutcome:
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
        feedback, done, score, valid = self._dispatch(env, call, extra_note)
        return ActionOutcome(
            thought=thought or assistant_text.strip(),
            action_raw=json.dumps(call, ensure_ascii=False),
            action_norm=json.dumps(call, ensure_ascii=False, sort_keys=True),
            feedback=feedback,
            done=done,
            score=score,
            acted=True,
            valid=valid,
            consumed_env_step=True,
        )

    def replay_action(self, env: MathEpisode, action_norm: str) -> tuple[str, bool]:
        call = json.loads(action_norm)
        feedback, done, _score, _valid = self._dispatch(env, call, "")
        return feedback, done

    def _dispatch(
        self, env: MathEpisode, call: dict, extra_note: str
    ) -> tuple[str, bool, float, bool]:
        name = call.get("name")
        if name == "run_python":
            code = str(call.get("arguments", {}).get("code", "") or "")
            if not code.strip():
                return _tool_resp("run_python: empty code." + extra_note), False, 0.0, False
            out, rc = run_python_script(
                code, cwd=env.workdir,
                timeout_s=self._cfg.python_timeout_s, gate=self._gate,
            )
            body = out if rc == 0 else f"[exit {rc}]\n{out}"
            return (
                _tool_resp(clip_middle(body, self._cfg.obs_max_chars) + extra_note),
                False, 0.0, rc == 0,
            )
        if name == "submit":
            answer = str(call.get("arguments", {}).get("answer", "") or "")
            if not answer.strip():
                return (
                    _tool_resp(
                        "submit: empty answer — pass the final answer as "
                        "the 'answer' argument." + extra_note
                    ),
                    False, 0.0, False,
                )
            score = 1.0 if answers_equivalent(
                str(self._item["ground_truth"]), answer
            ) else 0.0
            # 信息状态红线:评分结果不反馈。
            return _tool_resp("Submitted." + extra_note), True, score, True
        return (
            _tool_resp(
                f"unknown tool: {name!r}. Available: run_python, submit."
                + extra_note
            ),
            False, 0.0, False,
        )

    # ---- 文本动作协议件(不经 text_action_react;平凡实现) --------------

    def normalize_action(self, raw: str) -> str:
        return raw.strip()

    def observation_indicates_invalid(self, observation: str) -> bool:
        return False

    def format_reminder(self) -> str:
        return FORMAT_REMINDER


def _tool_resp(content: str) -> str:
    """与 Qwen3 模板 tool role 渲染形态对齐的包裹(SSB 同款;服务器真
    tokenizer 对拍见 GO 清单反馈侧一项)。"""
    return f"<tool_response>\n{content}\n</tool_response>"


class MathEnvFactory:
    """EnvFactory:进程级一个;持 run_python 并发门。无文件资产
    (data_root 不存在——任务全自足于 extra_info)。"""

    def __init__(self, cfg: MathCfg) -> None:
        ensure_math_verify()   # 依赖缺失 = 表达式答案系统性误判,装配期死
        self._cfg = cfg
        self._gate = threading.Semaphore(cfg.exec_concurrency)

    def adapter_from_item(self, item: dict) -> MathTaskAdapter:
        return MathTaskAdapter(item, self._cfg, self._gate)
