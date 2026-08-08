"""Interaction contract (single source of truth).

Everything that defines the executor<->environment conversation lives here so
research experiments, the training rollout (verl agent loop), the teacher
prompt construction, and the standalone evaluator can never drift apart:

  * guidance injection via ``types.inject_guidance`` (the SAME ``## Task
    guidance`` block serves researcher guidance, the privileged teacher
    prefix, and +doc eval arms); the bare system prompt itself is benchmark
    material and lives with each adapter (``executor_scaffold``);
  * the assistant output contract: ``<reasoning>...</reasoning>`` followed by
    ``<action>...</action>`` in plain content.  Custom tags are load-bearing —
    the Qwen chat template strips literal ``<think>...</think>`` blocks from
    prior-turn content, which would silently delete the reasoning from every
    past turn;
  * action extraction + conservative, semantics-free command normalization;
  * episode limits.

No admissible-command list is ever shown: dynamic action knowledge is exactly
the surface the experience document fills, and the privileged/bare contrast
must share one contract.

Inherited verbatim from the CD-003 contract (PG-3) with ONE deliberate
change: the privileged section header is the project-wide GUIDANCE_HEADER
("## Task guidance") instead of the h3 static-skill header, so the teacher
prefix is byte-identical with research-time guidance injection (PG-7).
"""

from __future__ import annotations

import ast
import json
import re

from .types import inject_guidance

# Episode limits ------------------------------------------------------------
# The step caps are yaml-configurable (episode: section; user ruling
# 2026-07-23: 30/40, down from the community-standard 50/60 — CD-006's h3
# per-episode data shows cap=30 keeps 94.2% of trained-policy successes while
# failure episodes stop burning the full budget). Entry points call
# configure_episode_caps() right after load_config(); consumers read the
# module attributes at run time, so a 30/50 comparison run needs only a yaml
# change.

MAX_ENV_STEPS = 30          # env-step budget per episode (one act per turn)
MAX_ASSISTANT_TURNS = 40    # env budget + slack for format-failure turns
GT_EXTRACT_MAX_STEPS = 50   # walkthrough-GT extraction cap (build_datasets):
                            # the expert reference must NOT be truncated by the
                            # training budget — decoupled on purpose
GEN_MAX_TOKENS_PER_TURN = 2048  # per-turn generation ceiling (reasoning+action);
                                # a ceiling, not a target — the model emits only
                                # what it needs, but long reasoning is not cut
OBS_TRUNCATE_CHARS = 4000   # defensive cap, engine observations are short
MAX_CONSECUTIVE_FORMAT_FAILURES = 3  # break the reminder->longer-context spiral


def configure_episode_caps(max_env_steps: int, max_assistant_turns: int) -> None:
    """Install the yaml-configured caps (single source: configs/trace.yaml
    episode: section). Idempotent; must run before any adapter/runner use."""
    global MAX_ENV_STEPS, MAX_ASSISTANT_TURNS
    if max_env_steps < 1 or max_assistant_turns <= max_env_steps:
        raise ValueError(
            f"episode caps invalid: max_env_steps={max_env_steps}, "
            f"max_assistant_turns={max_assistant_turns} (need >=1 and turns > steps)"
        )
    MAX_ENV_STEPS = max_env_steps
    MAX_ASSISTANT_TURNS = max_assistant_turns

# Benchmark prompt material -------------------------------------------------
# (v4 通用性整改,用户裁定 2026-08-06)ALFWorld 的 COMMAND_REFERENCE 与
# SYSTEM_PROMPT_BARE 已迁至 envs/alfworld/adapter.py —— core 只持有与任务
# 无关的输出协议(<reasoning>/<action> 契约、解析、通用格式提醒)。
# executor 提示词一律经 adapter.executor_scaffold() 获取。

# Assistant output parsing --------------------------------------------------

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE)

#: 通用格式提醒(输出协议自身的内容,无环境语感)。benchmark 可在
#: adapter.format_reminder() 里加环境习语前缀(如 ALFWorld 的
#: "Nothing happens. " 开头),使提醒读起来像一次正常环境响应。
DEFAULT_FORMAT_REMINDER = (
    "(Your reply had no <action>...</action> block — respond "
    "with <reasoning>...</reasoning> then <action>one command</action>.)"
)


def parse_action(assistant_text: str) -> str | None:
    """Extract the command from the LAST <action> block, or None if absent.

    Normalization is conservative and semantics-free: strip whitespace and
    surrounding quotes, keep only the first line, lowercase (the engine grammar
    is lowercase). No repair — ungrammatical commands must reach the engine
    and fail visibly.
    """
    matches = _ACTION_RE.findall(assistant_text or "")
    if not matches:
        return None
    command = matches[-1].strip()
    if not command:
        return None
    command = command.splitlines()[0].strip()
    command = command.strip("\"'`").strip().lower()
    return command or None


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# FC dialect(模板家族的 function-calling wire 形态)---------------------
# 训练通道是 token-in-token-out,拿不到服务端 tool-call parser,解析必须在
# 文本层完成且与服务端逐语义等价(用户裁定 2026-08-08:SSB 模式)。dialect
# 由 template_delta.fc_dialect(tokenizer) 行为探测,入口经
# configure_fc_dialect() 安装;缺省 hermes 保持 Qwen3 语境的历史行为。
#   hermes      — Qwen3 系:<tool_call>{"name":..., "arguments":{...}}</tool_call>
#   qwen35_xml  — Qwen3.5 系:<tool_call><function=名><parameter=键>值…
#   none        — 模板不渲染 tool_calls(FC 环境下由 preflight 拦截)
FC_DIALECTS = ("hermes", "qwen35_xml", "none")
_FC_DIALECT = "hermes"

#: dialect → 起 OpenAI 兼容服务时唯一合法的 ``--tool-call-parser``。
#: 我方文本层解析器与该服务端 parser 逐语义对拍(golden fixture);换 parser
#: 名(例如未来 sglang 引入 qwen3_xml,或换 vLLM 后端——vLLM 的 qwen3_xml
#: 是 Expat 路径、与 qwen3_coder 有已知语义差异,如 Python 风格 ``True``
#: 的接受性)= 训练/推理两条通道解析分歧,必须先重新生成 golden 重新对拍,
#: 再改这里(调研定案 2026-08-08:sglang 0.5.8/0.5.10 均只有 qwen3_coder,
#: 官方文档对 Qwen3.5 的推荐亦是它)。
SERVER_TOOL_CALL_PARSER = {"qwen35_xml": "qwen3_coder", "hermes": "hermes"}


def configure_fc_dialect(dialect: str) -> None:
    """Install the probed FC dialect (single source: the live chat template
    via template_delta.fc_dialect). Idempotent; call at process entry next to
    configure_episode_caps."""
    global _FC_DIALECT
    if dialect not in FC_DIALECTS:
        raise ValueError(f"unknown FC dialect: {dialect!r} (known: {FC_DIALECTS})")
    _FC_DIALECT = dialect


def fc_dialect_configured() -> str:
    return _FC_DIALECT


def parse_tool_calls(
    assistant_text: str,
    tools: list[dict] | None = None,
    dialect: str | None = None,
) -> list[dict]:
    """Extract tool calls in the configured template-family dialect.

    模板家族级协议件,与 parse_action 平级——用哪一个由各 benchmark 的
    react 决定。返回按出现顺序的调用列表,每项 ``{"name": str,
    "arguments": dict}``;坏输出跳过(交由 react 以 acted=False 处理,
    不在这里 raise)。

    ``tools``(OpenAI schema,来自 adapter.tools())仅 qwen35_xml 用:XML
    参数值是无类型文本,按 schema 还原 int/float/bool/null/object——语义与
    sglang 0.5.10 Qwen3CoderDetector 逐条对拍(tests/fixtures/
    fc_qwen35_xml_golden.json 为真 detector 生成的 golden)。hermes 的 JSON
    自带类型,忽略该参数。"""
    d = dialect or _FC_DIALECT
    if d == "qwen35_xml":
        return _parse_qwen35_xml(assistant_text or "", tools)[0]
    if d == "none":
        raise RuntimeError(
            "parse_tool_calls called with FC dialect 'none' — the chat template "
            "does not render tool calls; an FC benchmark cannot run on it"
        )
    calls: list[dict] = []
    for raw in _TOOL_CALL_RE.findall(assistant_text or ""):
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            continue
        args = obj.get("arguments")
        calls.append({
            "name": obj["name"],
            "arguments": args if isinstance(args, dict) else {},
        })
    return calls


def fc_format_reminder(*tool_names: str, dialect: str | None = None) -> str:
    """FC 协议 benchmark 的标准格式提醒(parse_tool_calls 的配套件;
    单一维护点,各 benchmark 传自己的工具名列表)。措辞随 dialect——提醒
    必须教模型它的模板所教的那种形态,不能跨家族串。"""
    d = dialect or _FC_DIALECT
    if d == "qwen35_xml":
        # 示例用第一个真实工具名:占位符 "function-name" 会被模型照抄成
        # unknown-tool 循环(2026-08-08 审阅)。
        example = tool_names[0] if tool_names else "tool-name"
        return (
            "(Your reply contained no valid tool call. End your message with "
            "exactly one block of the form <tool_call>\n"
            f"<function={example}>\n<parameter=param-name>\nvalue\n"
            "</parameter>\n</function>\n</tool_call>. "
            f"Available functions: {', '.join(tool_names)}.)"
        )
    return (
        "(Your reply contained no valid tool call. End your message with "
        "exactly one block of the form "
        '<tool_call>{"name": <function-name>, "arguments": <args-json-object>}'
        f"</tool_call>. Available functions: {', '.join(tool_names)}.)"
    )


def strip_tool_calls(assistant_text: str, dialect: str | None = None) -> str:
    """react 的 thought 记账文本(parse_tool_calls 的配套公共件——各
    benchmark 勿引私有正则)。

    hermes:删除全部 tool_call 块,保留其余文本(历史语义不变)。
    qwen35_xml:首个调用块之前的文本——与服务端 parser 的 normal_text
    语义一致(该模板家族要求 reasoning 在调用之前,调用之后的文本不属于
    协议内容,与 sglang 同样丢弃)。"""
    d = dialect or _FC_DIALECT
    if d == "qwen35_xml":
        return _parse_qwen35_xml(assistant_text or "", None)[1].strip()
    return _TOOL_CALL_RE.sub("", assistant_text or "").strip()


_TOOL_RESP_RE = re.compile(r"\A\s*<tool_response>\n(.*)\n</tool_response>\s*\Z", re.DOTALL)


def strip_tool_response(text: str) -> str:
    """tool_response_wrap 的逆(不是包裹形态时原样返回)。协议无关比较用
    ——如 continuation 重放:录制侧 observation 是回注进流的包裹文本,
    重放侧 env 返回裸观察,两侧归一化后才可比(text 协议下双侧 no-op)。"""
    m = _TOOL_RESP_RE.match(text or "")
    return m.group(1) if m else (text or "")


def tool_response_wrap(content: str, dialect: str | None = None) -> str:
    """工具执行结果的回注包裹(react 的 feedback 单一公共件)。

    必须与推理引擎对 ``role="tool"`` 消息的 chat-template 渲染**逐字一致**
    ——训练通道以 user 轮文本注入该反馈,两条通道的上下文才 byte 级同分布。
    Qwen3 与 Qwen3.5 实测同形态(``<tool_response>\\n内容\\n</tool_response>``,
    A800 2026-08-08);形态一致性由 template_delta.fc_stream_equivalence_probe
    对真模板差分守护,漂移在 preflight 即 FAIL,勿手改此处再"顺手适配"。
    dialect 参数为未来家族差异预留,当前两家族共用一个形态。"""
    del dialect  # 当前无家族差异;探针守护该假设
    return f"<tool_response>\n{content}\n</tool_response>"


# qwen35_xml 解析(语义单一真源 = sglang 0.5.10 Qwen3CoderDetector.
# detect_and_parse;逐条对齐,勿"顺手改进"——任何偏差都是训练/推理两条
# 通道的解析分歧)-------------------------------------------------------

_Q35_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_Q35_FUNC_RE = re.compile(r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)
_Q35_PARAM_RE = re.compile(
    r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)", re.DOTALL
)
_Q35_START = "<tool_call>"
_Q35_FUNC_PREFIX = "<function="


def _q35_props(func_name: str, tools: list[dict] | None) -> dict:
    if not tools:
        return {}
    for t in tools:
        fn = t.get("function", {}) if isinstance(t, dict) else {}
        if t.get("type") == "function" and fn.get("name") == func_name:
            params = fn.get("parameters")
            if isinstance(params, dict) and "properties" in params:
                return params["properties"]
            return params if isinstance(params, dict) else {}
    return {}


def _q35_convert(value: str, name: str, props: dict) -> object:
    if value.lower() == "null":
        return None
    if name not in props:
        return value
    spec = props[name]
    ptype = str(spec["type"]).strip().lower() if isinstance(spec, dict) and "type" in spec else "string"
    if ptype in ("string", "str", "text", "varchar", "char", "enum"):
        return value
    if ptype.startswith(("int", "uint", "long", "short", "unsigned")):
        try:
            return int(value)
        except Exception:  # noqa: BLE001 — sglang 同款宽度
            return value
    if ptype.startswith(("num", "float")):
        try:
            keep_int = not ("." in value or "e" in value.lower())
            f = float(value)
            return int(f) if keep_int and f.is_integer() else f
        except Exception:  # noqa: BLE001
            return value
    if ptype in ("boolean", "bool", "binary"):
        return value.lower() == "true"
    if ptype in ("object", "array", "arr") or ptype.startswith(("dict", "list")):
        try:
            return json.loads(value)
        except Exception:  # noqa: BLE001 — 深嵌套 RecursionError 等也须降级
            pass
    try:
        return ast.literal_eval(value)
    except Exception:  # noqa: BLE001
        return value


def _parse_qwen35_xml(
    text: str, tools: list[dict] | None
) -> tuple[list[dict], str]:
    try:
        calls, normal = _parse_qwen35_xml_inner(text, tools)
        json.dumps([c["arguments"] for c in calls])  # 下游可序列化防线
        return calls, normal
    except Exception:  # noqa: BLE001 — sglang detect_and_parse 同款整体降级
        return [], text


def _parse_qwen35_xml_inner(
    text: str, tools: list[dict] | None
) -> tuple[list[dict], str]:
    """(calls, normal_text) — sglang Qwen3CoderDetector.detect_and_parse 等价。

    容忍三级未闭合(</parameter>/</function>/</tool_call> 缺失均可解析);
    裸 ``<function=`` 无 ``<tool_call>`` 开标签时不解析(与服务端一致);
    参数值仅剥恰好一个前导与一个尾随换行(不是 strip);normal_text =
    首个调用块之前的文本(块后文本丢弃)。"""
    if _Q35_START not in text:
        return [], text
    blocks = _Q35_BLOCK_RE.findall(text)
    if not blocks and _Q35_FUNC_PREFIX in text:
        blocks = [text]
    calls: list[dict] = []
    for content in blocks:
        for m in _Q35_FUNC_RE.findall(content):
            body = m[0] or m[1]
            if ">" not in body:
                continue
            cut = body.index(">")
            func_name = body[:cut]
            props = _q35_props(func_name, tools)
            args: dict = {}
            for p in _Q35_PARAM_RE.findall(body[cut + 1:]):
                if ">" not in p:
                    continue
                pcut = p.index(">")
                pname, pval = p[:pcut], p[pcut + 1:]
                if pval.startswith("\n"):
                    pval = pval[1:]
                if pval.endswith("\n"):
                    pval = pval[:-1]
                args[pname] = _q35_convert(pval, pname, props)
            calls.append({"name": func_name, "arguments": args})
    start = text.find(_Q35_START)
    if start == -1:
        start = text.find(_Q35_FUNC_PREFIX)
    normal_text = text[:start] if start > 0 else ""
    return calls, normal_text


def parse_reasoning(assistant_text: str) -> str | None:
    """Content of the LAST <reasoning> block, or None if absent."""
    matches = _REASONING_RE.findall(assistant_text or "")
    if not matches:
        return None
    return matches[-1].strip() or None


def text_action_react(adapter, env, assistant_text: str):
    """Reusable ``TaskAdapter.react`` for string-command benchmarks: the
    ``<reasoning>...</reasoning><action>one command</action>`` contract,
    executed against an env exposing ``step(command) -> (obs, done, result)``.

    这只是文本单动作协议的标准实现件——react 抽象本身不假定它;工具调用/
    代码执行型 benchmark 自写 react,不经此函数。step 的第三个返回值按
    float 保真进 ActionOutcome.score:bool(传统二元 won,True→1.0)与
    [0,1] 标量(部分分)都直接成立。Blocking (env.step) — callers put
    the whole react call in a thread."""
    from .adapter import ActionOutcome

    command = parse_action(assistant_text)
    if command is None:
        return ActionOutcome(
            thought=assistant_text.strip(),
            action_raw="",
            action_norm="",
            feedback=adapter.format_reminder(),
            done=False,
            score=0.0,
            acted=False,
            valid=False,
            consumed_env_step=False,
        )
    thought = parse_reasoning(assistant_text) or assistant_text.strip()
    norm = adapter.normalize_action(command)
    obs, done, result = env.step(norm)
    return ActionOutcome(
        thought=thought,
        action_raw=command,
        action_norm=norm,
        feedback=obs,
        done=done,
        score=float(result),
        acted=True,
        valid=not adapter.observation_indicates_invalid(obs),
        consumed_env_step=True,
    )


def truncate_obs(obs: str, limit: int = OBS_TRUNCATE_CHARS) -> str:
    if len(obs) <= limit:
        return obs
    head = limit // 2
    tail = limit - head
    return obs[:head] + "\n[... observation truncated ...]\n" + obs[-tail:]
