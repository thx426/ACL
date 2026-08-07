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


def parse_tool_calls(assistant_text: str) -> list[dict]:
    """Extract hermes-style tool calls: ``<tool_call>{json}</tool_call>``.

    模板家族级协议件(Qwen3 系 chat template 的原生 function-calling 输出
    形态),与 parse_action 平级——用哪一个由各 benchmark 的 react 决定。
    返回按出现顺序的合法调用列表,每项形如
    ``{"name": str, "arguments": dict}``;JSON 不合法或缺 name 的块跳过
    (坏输出交由 react 以 acted=False 处理,不在这里 raise)。"""
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


def fc_format_reminder(*tool_names: str) -> str:
    """FC 协议 benchmark 的标准格式提醒(parse_tool_calls 的配套件;
    单一维护点,各 benchmark 传自己的工具名列表)。"""
    return (
        "(Your reply contained no valid tool call. End your message with "
        "exactly one block of the form "
        '<tool_call>{"name": <function-name>, "arguments": <args-json-object>}'
        f"</tool_call>. Available functions: {', '.join(tool_names)}.)"
    )


def strip_tool_calls(assistant_text: str) -> str:
    """去除全部 tool_call 块后的剩余文本(react 的 thought 记账用;
    parse_tool_calls 的配套公共件——各 benchmark 勿引私有正则)。"""
    return _TOOL_CALL_RE.sub("", assistant_text or "").strip()


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
