"""The researcher session: a free-form multi-turn agent loop.

Structure is deliberately minimal (pilot-validated rulings): no fixed
"rounds", no programmatic finalize gate — the researcher decides when it is
done; the program only maintains counters, injects the near-limit wrap-up
reminder, and records every message verbatim (SessionResult.messages is the
archive/SFT source of truth).

Backend-agnostic: the researcher LLM is a ResearcherChatFn and tool
transport is a ToolProtocol, so the strong-model form (external + fncall)
and the policy form (engine + text) share this loop.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..common.llm import ChatResult
from ..common.tape import TAPE_CONTEXT
from .prompts import KICKOFF_MESSAGE, WRAPUP_REMINDER, tool_schemas_for
from .toolproto import ToolProtocol
from .tools import ToolRouter

# (messages, tools) -> ChatResult. External: common/llm client. Engine
# (policy-as-researcher): train/backends.py renders + generates + parses.
ResearcherChatFn = Callable[
    [list[dict[str, Any]], list[dict[str, Any]] | None], Awaitable[ChatResult]
]


@dataclass
class SessionResult:
    finalized: bool
    no_finalize: bool
    turns_used: int
    experiments_used: int
    episodes_consumed: int
    doc: str | None
    self_assessment: str
    flags: list[str]
    last_exp_successes: int
    last_exp_attempts: int
    # 会话累计成败(2026-08-10 零正例裁定):零正例文档的机器判据,
    # 下游择优/降权用它,不依赖模型在文档里的自我陈述。
    exp_successes_total: int = 0
    exp_attempts_total: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    guidance_log: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None  # set when the session ended on an exception
    # Session-total researcher usage (SFT filtering signal; 0 when the
    # backend reports none, e.g. the engine form).
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ResearchSession:
    def __init__(
        self,
        *,
        researcher_chat: ResearcherChatFn,
        toolproto: ToolProtocol,
        router: ToolRouter,
        system_prompt: str,
        max_turns: int,
        wrapup_buffer: int = 2,
    ) -> None:
        self._chat = researcher_chat
        self._proto = toolproto
        self._router = router
        self._max_turns = max_turns
        self._wrapup_at = max(1, max_turns - 1 - wrapup_buffer)
        # 工具描述按 router 的 ResearchCfg 生成:k 的 default/max 必须与配置一致,
        # 否则模型照着写死的文案显式填 k,cfg.k_default 永远不生效(见
        # prompts.tool_schemas_for 的注释与 2026-08-07 的实测)。
        prepared_system, self._tools = toolproto.prepare(
            system_prompt, tool_schemas_for(router.cfg)
        )
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": prepared_system}
        ]
        self._turns = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Live message record — readable by the caller even after a timeout
        cancellation (partial archives must never lose the conversation)."""
        return self._messages

    def snapshot(self, error: str | None = None) -> SessionResult:
        """SessionResult from the CURRENT router/session state (used for the
        normal return and for partial results after errors/timeouts)."""
        state = self._router.state
        if not state.finalized:
            state.no_finalize = True
        return SessionResult(
            finalized=state.finalized,
            no_finalize=state.no_finalize,
            turns_used=self._turns,
            experiments_used=state.experiments_used,
            episodes_consumed=state.episodes_consumed,
            doc=state.doc,
            self_assessment=state.self_assessment,
            flags=list(state.flags),
            last_exp_successes=state.last_exp_successes,
            last_exp_attempts=state.last_exp_attempts,
            exp_successes_total=state.exp_successes_total,
            exp_attempts_total=state.exp_attempts_total,
            messages=list(self._messages),
            guidance_log=list(state.guidance_log),
            error=error,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
        )

    async def run(self) -> SessionResult:
        """Never raises (except CancelledError): mid-session failures return a
        partial SessionResult carrying everything accumulated so far."""
        state = self._router.state
        n_exp = self._router.cfg.n_exp
        self._messages.append(
            {"role": "user", "content": KICKOFF_MESSAGE.format(n_exp=n_exp)}
        )

        reminded = False
        try:
            while self._turns < self._max_turns and not state.finalized:
                self._turns += 1
                if not reminded and self._turns >= self._wrapup_at:
                    reminded = True
                    self._messages.append({"role": "user", "content": WRAPUP_REMINDER})

                token = TAPE_CONTEXT.set(
                    f"research:{self._router.adapter.task_id}:turn{self._turns:02d}"
                )
                try:
                    result = await self._chat(self._messages, self._tools)
                finally:
                    TAPE_CONTEXT.reset(token)
                self._prompt_tokens += result.prompt_tokens
                self._completion_tokens += result.completion_tokens

                if result.text:
                    self._messages.append(self._proto.canonicalize_assistant(result))

                if result.finish_reason in ("length", "aborted"):
                    # 预算耗尽/引擎中止:本轮内容不可信(截断可能产出半个
                    # 调用——qwen35_xml 解析容忍未闭合标签,半篇 doc 会被
                    # 当合法 finalize),不解析、不续轮,优雅收敛为 snapshot
                    # (2026-08-08 审阅:engine 后端 length 后继续轮转还会
                    # 触发 assistant 对拍 raise,把正常终态变 error)。
                    break

                tool_calls = self._proto.extract(result)
                if not tool_calls:
                    hint = self._proto.malformed_hint(result)
                    if hint is None:
                        # 纯思考轮也必须回一个 user 轮:engine 后端的
                        # append-only 流假设每个 assistant 轮后跟 user 结构段,
                        # 不回则下一轮生成紧贴 <|im_end|> 无轮头,训练行破
                        # 轮边界(2026-08-08 审阅)。
                        hint = ("(no tool call detected — continue; make a "
                                "tool call when you need an instrument.)")
                    self._messages.append({"role": "user", "content": hint})
                    continue
                # One call per message: execute the FIRST, tell the model the
                # rest were ignored (it must see feedback before spending more).
                first, extras = tool_calls[0], tool_calls[1:]
                tool_result = await self._router.dispatch(first)
                if extras:
                    tool_result += (
                        f"\n\n(NOTE: {len(extras)} additional tool call(s) in the "
                        "same message were IGNORED — issue one call per message "
                        "and read its result before the next.)"
                    )
                self._messages.append(self._proto.tool_result_message(first, tool_result))
        except asyncio.CancelledError:
            raise  # timeout owner takes the partial state via snapshot()
        except Exception as err:  # noqa: BLE001 — degrade, never lose the record
            return self.snapshot(error=f"{type(err).__name__}: {err}")
        return self.snapshot()
