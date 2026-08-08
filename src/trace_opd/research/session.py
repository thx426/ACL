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

                self._messages.append(self._proto.canonicalize_assistant(result))

                tool_calls = self._proto.extract(result)
                if not tool_calls:
                    hint = self._proto.malformed_hint(result)
                    if hint:
                        self._messages.append({"role": "user", "content": hint})
                    # else: plain thinking-aloud; the turn counter bounds it.
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
