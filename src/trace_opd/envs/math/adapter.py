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

三档协议(`math.protocol`,进 objective_fingerprint):
  fncall  CD-010 原档 —— run_python + submit 两工具,多轮 TIR
  text    2026-08-13 裁定 —— ```python 围栏 + ```output 回注 + \\boxed 终答,
          多轮 TIR(对照 ToRL/SimpleTIR/Qwen2.5-Math-TIR)
  single  2026-08-16 裁定 —— **单轮**,对齐 MENTOR(2510.04140/ICLR2026):
          无工具、无沙箱、无 observation,一次 completion 即终局;
          推理标签(protocol.REASONING_TAG,默认 <reasoning>)+ \\boxed{}；
          score=(1-w)·acc + w·format。
          caps 必须 1/1(config._validate 钉死),reward 走
          train.reward_from_score=true 的连续通道。
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import threading

from ...common.config import MathCfg
from ...core import protocol
from ...core.adapter import ActionOutcome
from ..pytool import clip_middle, run_python_script
from .verifier import answers_equivalent, ensure_math_verify

#: 无合法 tool call 时的提醒(裸文本;协议说明由模板渲染的工具块承担)
def _format_reminder() -> str:
    # 调用时求值:措辞随运行时安装的 FC dialect(模块 import 早于
    # configure_fc_dialect,常量会把 hermes 措辞固化进 qwen35_xml 语境)
    return protocol.fc_format_reminder("run_python", "submit")

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


# --- 文本 TIR 协议(2026-08-13 用户裁定:经典线 Qwen2.5 回退文本格式)------
# 依据(对照 TIR 多轮工具调用数学推理相关工作,均代码/论文级核实):
#   ToRL(arXiv 2503.23383):```python 围栏代码 → 环境回注 ```output\nOBS\n``` →
#     终答 \boxed{};错误信息回喂(利于自纠);obs token 全 mask。
#   SimpleTIR(arXiv 2509.02479,基座同为 Qwen2.5-7B):完整独立脚本 + print
#     中间量;代码块闭合即停止生成、真实工具反馈再续(防幻觉输出);
#     "void turn"(无代码亦无终答)是多轮 TIR 崩塌主因。
#   Qwen2.5-Math 官方 TIR 模板:同 ```python/```output 围栏 + \boxed{} 约定。
# 多轮承载:两家原版是单序列暂停-续写(base 模型),我们是 Instruct 模型的
# 多轮 chat(PG-3 评测训练同构)——代码块作为 assistant 轮结束,执行结果以
# user 轮回注(SearchQA 文本协议同构)。one-shot 示例内嵌 system(SearchQA
# 实证:Qwen2.5 zero-shot 不服从自定义标签,system 单示例 0%→99.9%)。
SYSTEM_PROMPT_TEXT = (
    "You are a competition mathematician who integrates step-by-step "
    "reasoning with Python programs to solve problems.\n\n"
    "Each reply, reason first, then do exactly one of the following:\n"
    "(1) Write ONE Python code block to compute, enumerate, or verify "
    "anything error-prone. The block must be a complete standalone script "
    "(including imports) that uses print() to show what you need to see — "
    "variables do NOT persist between blocks. Format:\n"
    "```python\n"
    "print(6 * 7)\n"
    "```\n"
    "End your reply right after the closing ``` and never guess the "
    "output — the real execution result is then returned to you wrapped "
    "as:\n"
    "```output\n"
    "42\n"
    "```\n"
    "(2) When the answer is established, state it with \\boxed{} in a "
    "reply that contains NO code block, e.g.: The final answer is "
    "\\boxed{42}. Give the exact value in simplest form (integer, "
    "fraction, or LaTeX expression), not a decimal approximation unless "
    "the problem asks for one. Boxing the answer ends the episode.\n\n"
    "Worked example — problem: What is 7^4 - 3*41?\n"
    "Your reply:\n"
    "7^4 = 2401 and 3*41 = 123, so the difference should be 2278. Let me "
    "verify.\n"
    "```python\n"
    "print(7**4 - 3*41)\n"
    "```\n"
    "You then receive:\n"
    "```output\n"
    "2278\n"
    "```\n"
    "Your next reply:\n"
    "The computation confirms it. The final answer is \\boxed{2278}.\n\n"
    "Arithmetic slips are the most common way correct reasoning produces "
    "a wrong answer — verify anything nontrivial with code before boxing "
    "the answer."
)

#: 围栏代码块(要求 ```python/```py 标注;裸 ``` 不收——可能是伪代码或
#: LaTeX,协议措辞与提醒都写明标注)。
_TEXT_CODE_RE = re.compile(r"```(?:python|py)\b[^\n]*\n(.*?)```",
                           re.DOTALL | re.IGNORECASE)


# --- 单轮协议(2026-08-16 用户裁定:对齐 MENTOR 的 setting)-----------------
# 依据 MENTOR(arXiv 2510.04140,ICLR 2026;代码 github.com/Jiangzs1028/MENTOR
# commit b1cc4c6,已在 related_work/code/MENTOR 逐行核实):
#   * 环境:无环境。EasyR1 标准单轮 RLHF 数据流 —— 一个 prompt 进、一段
#     completion 出、规则判分。无工具、无沙箱、无 observation 回注。
#   * 格式协议:system prompt **硬编码在 verl/utils/dataset.py:174-176 的
#     _build_messages()**(不在 jinja 里 —— train.sh 传的 no.jinja 内容是
#     `{{ content | trim }}` 什么都不加,config 默认指向的 math.jinja 在仓库
#     里根本不存在)。下面这段是逐字移植,论文 Appx C p18 亦逐字复述。
#   * reward:`overall = 0.9·accuracy + 0.1·format`
#     (examples/reward_function/math.py:compute_score,format_weight=0.1)。
# 与 MENTOR 的两处刻意差异(记账,不对齐):
#   1. 基座 Qwen2.5-7B-**Instruct**(我们主线,2026-08-16 用户裁定)vs 他们的
#      Base —— 我们的 researcher/doc/narration 链路都在 Instruct 语境验证过。
#   2. 判分用 math-verify 子进程(我们既有的、防过 sympy 满核自旋的那套)vs
#      他们的 mathruler。★ 论文 Appx C 写的是 "Math-Verify",与其代码不符
#      —— 他们代码里是 mathruler,是两套等价性判定。
# ★ 推理标签(2026-08-16 用户裁定):MENTOR 原文用 `<think>`,我们统一改用
# protocol.REASONING_TAG(默认 <reasoning>)。动因见 core/protocol.py 的
# REASONING_TAG 注释——thinking-family 模板会剥离历史 assistant 轮的
# <think> 块、其 non-thinking 模式是预填空 think 块、且该字面在 Qwen3.5/3.6
# 词表里是特殊 token。除标签外句式逐字保持 MENTOR 原文。
# ★ 必须是**调用时求值**的函数(模块常量会被 `from … import` 绑定旧值)。
def system_prompt_single() -> str:
    ro, rc = protocol.rtag_open(), protocol.rtag_close()
    return (
        "You are a helpful AI Assistant that provides well-reasoned and "
        "detailed responses. You FIRST think about the reasoning process as an "
        "internal monologue and then provide the final answer. The reasoning "
        f"process MUST BE enclosed within {ro} {rc} tags. The final "
        "answer MUST BE put in \\boxed{}."
    )


#: MENTOR 判分前的预处理,逐字移植:删掉 < > / 三个字符周围的空白。
#: 其注释说是为 qwen2.5vl-32b,实际效果是让 `< reasoning >`、`</ reasoning >`
#: 这类带空格的标签也能通过格式判定 —— 等于给协议加了一层容错。
_MENTOR_WS_RE = re.compile(r"\s*(<|>|/)\s*")


def mentor_format_re() -> "re.Pattern[str]":
    """MENTOR 的 format 正则(结构逐字同构,标签随 REASONING_TAG)。
    fullmatch + DOTALL ⇒ 回复必须以开标签起头、其后任意处出现 \\boxed{...}。
    注意它**不校验花括号平衡**(答案抽取我们仍走 extract_boxed 的平衡扫描,
    比 MENTOR 严 —— 见下)。"""
    tag = protocol.REASONING_TAG
    return re.compile(rf"<{tag}>.*</{tag}>.*\\boxed\{{.*\}}.*", re.DOTALL)


def mentor_format_ok(response: str) -> bool:
    """MENTOR 的 format_reward 判定(逐字同构,返回 bool 而非 0/1)。"""
    return bool(mentor_format_re().fullmatch(_MENTOR_WS_RE.sub(r"\1", response or "")))


def single_think(response: str) -> str:
    """thought 记账:推理块里的内容;没有闭合标签时取全文。"""
    t = response or ""
    ro, rc = protocol.rtag_open(), protocol.rtag_close()
    i = t.find(ro)
    j = t.find(rc, i + len(ro)) if i >= 0 else -1
    if i >= 0 and j > i:
        return t[i + len(ro): j].strip()
    return t.strip()


def extract_boxed(text: str) -> str | None:
    """最后一个 \\boxed{...} 的内容。花括号平衡扫描——正则不识别嵌套
    (\\frac{1}{2} 会被 `[^}]*` 截半)。不闭合按无终答。"""
    key = "\\boxed{"
    i = text.rfind(key)
    if i < 0:
        return None
    depth, j = 1, i + len(key)
    while j < len(text) and depth:
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        j += 1
    if depth:
        return None
    return text[i + len(key): j - 1].strip()


def parse_text_action(assistant_text: str) -> dict | None:
    """文本回复 → 与 FC 同构的 call dict(下游 _dispatch 零改动)。

    优先级:**代码块 > \\boxed**(与 SearchQA 的 answer 优先相反,是有意
    的):TIR 模型常在代码块后顺手写"答案应为 \\boxed{X}"——那是对未执行
    输出的猜测(SimpleTIR 明确指出幻觉输出是病灶,靠代码块后强停防)。
    多轮 chat 下无法截停生成,改为语义层防:同现时执行代码、忽略 boxed,
    真实输出回注后模型下一轮再交答案。终答通道 = 无代码块的 \\boxed 轮
    (system 措辞同款)。多代码块取**最后一个**(ToRL 同款:执行 latest)。
    """
    t = assistant_text or ""
    blocks = _TEXT_CODE_RE.findall(t)
    if blocks:
        return {"name": "run_python", "arguments": {"code": blocks[-1].strip()}}
    boxed = extract_boxed(t)
    if boxed is not None:
        return {"name": "submit", "arguments": {"answer": boxed}}
    return None


def text_extra_note(assistant_text: str) -> str:
    """同轮多代码块的忽略说明(FC 档 extra_note 的文本档对应物)。"""
    n = len(_TEXT_CODE_RE.findall(assistant_text or ""))
    if n > 1:
        return (f"\n(note: {n - 1} earlier code block(s) in the same reply "
                "were ignored — one block per reply, the last one runs)")
    return ""


def text_thought(assistant_text: str, call: dict | None) -> str:
    """thought 记账:代码轮取最后一个围栏之前的正文;终答/无动作轮取全文。"""
    t = (assistant_text or "").strip()
    if call and call.get("name") == "run_python":
        m = None
        for m in _TEXT_CODE_RE.finditer(assistant_text or ""):
            pass
        if m is not None:
            return (assistant_text[: m.start()]).strip()
    return t


def text_format_reminder() -> str:
    """void turn(SimpleTIR 语汇:无代码亦无终答)的格式提醒。"""
    return (
        "\n(Your reply contained neither a ```python code block nor a "
        "\\boxed{} final answer. To compute, write one complete standalone "
        "script inside a fenced block that starts with ```python and ends "
        "with ```. To finish, state the final answer as \\boxed{...} in a "
        "reply without any code block. Let me try again.)\n"
    )


def text_obs_wrap(body: str) -> str:
    """执行结果包裹:ToRL/Qwen2.5-Math-TIR 同款 ```output 围栏。"""
    return f"```output\n{body}\n```"


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
        if self._single_mode():
            # 单轮环境的 researcher brief:没有"动作"可言,只有一次作答。
            # 预算写实际值(MAX_ENV_STEPS 恒 1),不提工具/沙箱 —— 否则
            # researcher 会写出教执行器调用工具的 doc,而执行器根本没有工具。
            w = float(getattr(self._cfg, "format_weight", 0.0) or 0.0)
            fmt_line = (
                f"Scoring: {1.0 - w:.2f} for a correct answer plus {w:.2f} for "
                "adhering to the format above.\n"
                if w > 0.0 else
                "Scoring: binary — 1 for a correct answer, 0 otherwise.\n"
            )
            return (
                "Environment: one competition math problem, answered in a "
                "SINGLE reply. There are no tools, no code execution, and no "
                "follow-up turns — the executor writes its reasoning inside "
                f"{protocol.rtag_open()} {protocol.rtag_close()} tags and puts "
                "the final answer in \\boxed{}, and the episode ends there.\n"
                + fmt_line +
                "The answer is checked for mathematical equivalence against "
                "the reference (exact value; equivalent forms such as 1/2 vs "
                "0.5 vs \\frac{1}{2} all count). The executor never sees any "
                "grading feedback.\n"
                f"Problem: {self._item['problem']}"
            )
        if self._text_mode():
            actions = (
                "  (1) write one fenced ```python code block — a complete "
                "standalone script run in a scratch directory (fresh "
                "interpreter each time, no state carries over); its "
                "printed output is returned wrapped in an ```output "
                "block\n"
                "  (2) state the final answer as \\boxed{...} in a reply "
                "without a code block — this ends the episode\n"
            )
        else:
            actions = (
                "  run_python(code) — run a standalone Python script in a "
                "scratch directory (fresh interpreter per call, no state "
                "carries over); for computing, enumerating, or verifying "
                "intermediate results\n"
                "  submit(answer) — submit the final answer and end the "
                "episode\n"
            )
        return (
            "Environment: one competition math problem. The executor "
            "reasons in text and each reply takes exactly one action:\n"
            + actions +
            "Grading: the final answer is checked for mathematical "
            "equivalence against the reference (exact value; equivalent "
            "forms such as 1/2 vs 0.5 vs \\frac{1}{2} all count). Score "
            "is binary. The executor never sees any grading feedback.\n"
            f"Problem: {self._item['problem']}\n"
            f"Budget: the executor has {protocol.MAX_ENV_STEPS} code "
            "executions per episode; reasoning is free-form text."
        )

    def _text_mode(self) -> bool:
        return getattr(self._cfg, "protocol", "fncall") == "text"

    def _single_mode(self) -> bool:
        return getattr(self._cfg, "protocol", "fncall") == "single"

    def executor_scaffold(self) -> str:
        if self._single_mode():
            return system_prompt_single()
        if self._text_mode():
            return SYSTEM_PROMPT_TEXT
        return SYSTEM_PROMPT

    def tools(self) -> list[dict] | None:
        # 文本/单轮协议不渲染 tools(否则 chat template 同时教 FC 语法,与
        # 标签协议打架;SearchQA 同款处理)。single 档更是根本没有工具。
        if self._text_mode() or self._single_mode():
            return None
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
        if self._single_mode():
            return self._react_single(env, assistant_text)
        if self._text_mode():
            return self._react_text(env, assistant_text)
        return self._react_fncall(env, assistant_text)

    def _react_single(self, env: MathEpisode, assistant_text: str) -> ActionOutcome:
        """MENTOR 单轮档:一次回复即终局,**无条件 done**。

        没有工具、没有沙箱、没有格式重试 —— 单轮语义下"没写对格式"就是
        这次作答失败,不存在下一轮。故恒 acted=True/consumed_env_step=True,
        走不到 core 的 invalid_format 反螺旋路径(那是多轮才有的概念);
        格式是否合规改由 `valid` 承载(observability 字段)。

        score = (1-w)·accuracy + w·format,与 MENTOR compute_score 同构。
        w=0 时退化为二值答案正确率。★ score 的四个取值 {0, w, 1-w, 1} 与
        (acc, fmt) 一一对应,可无损反推 —— dump 里只带 score 即可还原分项。
        """
        text = assistant_text or ""
        fmt_ok = mentor_format_ok(text)
        answer = extract_boxed(text)          # 花括号平衡扫描,比 MENTOR 的
                                              # `\\boxed\{.*\}` 正则严
        acc = 0.0
        if answer:
            # 判分与 TIR 档完全同一条路径(math-verify 子进程 + 硬超时),
            # 保证跨协议的正确性口径可比。
            acc = 1.0 if answers_equivalent(
                str(self._item["ground_truth"]), answer,
                timeout_s=float(getattr(self._cfg, "verify_timeout_s", 20.0)),
                gate=self._gate,
            ) else 0.0
        w = float(getattr(self._cfg, "format_weight", 0.0) or 0.0)
        score = (1.0 - w) * acc + w * (1.0 if fmt_ok else 0.0)
        return ActionOutcome(
            thought=single_think(text),
            action_raw=answer or "",
            action_norm=json.dumps(
                {"name": "submit", "arguments": {"answer": answer or ""}},
                ensure_ascii=False, sort_keys=True,
            ),
            feedback="",          # 单轮:没有下一轮,反馈无处可去
            done=True,
            score=score,
            acted=True,
            valid=fmt_ok,         # 观察面:MENTOR 口径的格式合规
            consumed_env_step=True,
        )

    def _react_text(self, env: MathEpisode, assistant_text: str) -> ActionOutcome:
        """文本 TIR 协议(ToRL/SimpleTIR 同构)。执行与判分完全复用
        _dispatch —— 只是把围栏代码/\\boxed 翻译成同构 call dict。"""
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
            action_norm=json.dumps(call, ensure_ascii=False, sort_keys=True),
            feedback=feedback,
            done=done,
            score=score,
            acted=True,
            valid=valid,
            consumed_env_step=True,
        )

    def _react_fncall(self, env: MathEpisode, assistant_text: str) -> ActionOutcome:
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
        # 单轮档没有可重放的环境交互(无工具、无状态,一步即终局)。
        if self._single_mode():
            return "", True
        call = json.loads(action_norm)
        feedback, done, _score, _valid = self._dispatch(
            env, call, "", text_mode=self._text_mode()
        )
        return feedback, done

    def _dispatch(
        self, env: MathEpisode, call: dict, extra_note: str,
        text_mode: bool = False,
    ) -> tuple[str, bool, float, bool]:
        # 观察包裹随协议切换:FC 用 <tool_response>,文本档用 ```output 围栏
        # (ToRL 同款)。文本档的 extra_note 放围栏**外**——放进去会伪装成
        # 程序输出。
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
            out, rc = run_python_script(
                code, cwd=env.workdir,
                timeout_s=self._cfg.python_timeout_s, gate=self._gate,
            )
            body = out if rc == 0 else f"[exit {rc}]\n{out}"
            return (
                wrap_obs(clip_middle(body, self._cfg.obs_max_chars)),
                False, 0.0, rc == 0,
            )
        if name == "submit":
            answer = str(call.get("arguments", {}).get("answer", "") or "")
            if not answer.strip():
                return (
                    wrap_msg(
                        "submit: empty answer — state the final answer as "
                        "\\boxed{...}." if text_mode else
                        "submit: empty answer — pass the final answer as "
                        "the 'answer' argument."
                    ),
                    False, 0.0, False,
                )
            # 判分走子进程 + 硬超时(2026-08-13 冒烟实锤:病态表达式让
            # sympy 在 to_thread 里满核自旋、线程杀不死,拖死整臂收尾)。
            # 复用 run_python 的并发门限制子进程数。
            score = 1.0 if answers_equivalent(
                str(self._item["ground_truth"]), answer,
                timeout_s=float(getattr(self._cfg, "verify_timeout_s", 20.0)),
                gate=self._gate,
            ) else 0.0
            # 信息状态红线:评分结果不反馈。
            return wrap_msg("Submitted."), True, score, True
        return (
            wrap_msg(f"unknown tool: {name!r}. Available: run_python, submit."),
            False, 0.0, False,
        )

    # ---- 文本动作协议件(不经 text_action_react;平凡实现) --------------

    def normalize_action(self, raw: str) -> str:
        return raw.strip()

    def observation_indicates_invalid(self, observation: str) -> bool:
        return False

    def format_reminder(self) -> str:
        # single 档不会走到格式提醒(react 恒 acted=True);保留一个正确的
        # 措辞以防将来有调用方拿它做展示。
        if self._single_mode():
            return (
                f"\n(Enclose the reasoning in {protocol.rtag_open()} "
                f"{protocol.rtag_close()} tags and put the final answer in "
                "\\boxed{}.)\n"
            )
        if self._text_mode():
            return text_format_reminder()
        return _format_reminder()




class MathEnvFactory:
    """EnvFactory:进程级一个;持 run_python 并发门。无文件资产
    (data_root 不存在——任务全自足于 extra_info)。"""

    def __init__(self, cfg: MathCfg) -> None:
        ensure_math_verify()   # 依赖缺失 = 表达式答案系统性误判,装配期死
        self._cfg = cfg
        self._gate = threading.Semaphore(cfg.exec_concurrency)

    def adapter_from_item(self, item: dict) -> MathTaskAdapter:
        return MathTaskAdapter(item, self._cfg, self._gate)
