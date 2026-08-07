"""TaskAdapter protocol — the ONLY surface the task-agnostic core sees.

Every benchmark implements this protocol independently (docs/design.md §5).
The core mechanism (episode runner, researcher loop, narration pipeline,
training agent loop) never imports a concrete environment module; concrete
adapters are constructed from dataset rows by the environment's factory and
passed in.

An adapter instance is bound to ONE task (one researcher session studies one
task; the per-task document feeds that task's OPD teacher prefix downstream).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class EnvHandle(Protocol):
    """One live episode. Implementations must be safe to close() twice.

    Deliberately minimal: the core NEVER drives the environment directly —
    all interaction goes through ``TaskAdapter.react`` / ``replay_action``,
    so the action transport (string commands, JSON tool calls, code
    execution, ...) is entirely the benchmark's business. String-command
    environments typically expose an internal ``step()`` that their
    adapter's react consumes; that is an implementation detail, not part
    of this contract."""

    @property
    def initial_obs(self) -> str: ...

    def close(self) -> None: ...


@dataclass
class ActionOutcome:
    """Result of one executor turn, produced by ``TaskAdapter.react``.

    This is the ONLY surface through which action semantics, environment
    feedback, and final evaluation reach the task-agnostic core — the core
    never assumes a specific action grammar (text tags / tool calls / code),
    feedback shape, or binary outcomes.

    观测字段(thought/action_raw/action_norm/valid)只进轨迹记账与渲染,
    永不进 loss。"""

    thought: str            # reasoning extracted for the trajectory record
    action_raw: str         # action as emitted by the model ("" if none parsed)
    action_norm: str        # normalized/executable form ("" if none parsed)
    feedback: str           # text fed back to the executor as the next user turn
                            # (core applies token-budget truncation, nothing else)
    done: bool              # episode reached a terminal state
    score: float            # terminal evaluation in [0, 1]; 0.0 while not done.
                            # Binary win/loss benchmarks emit {0.0, 1.0}; partial
                            # credit is first-class (e.g. fraction of unit tests
                            # passed). The current GRPO recipe derives its binary
                            # outcome as score >= 1.0; wiring partial credit into
                            # the reward is a training-side config decision, not
                            # a protocol change.
    acted: bool             # False = the reply never reached the environment
                            # (unparseable action); the core counts consecutive
                            # failures for the anti-spiral break, nothing more
    valid: bool             # environment-level no-op indicator (observability)
    consumed_env_step: bool  # whether this turn spends one unit of the
                             # env-step budget (protocol.MAX_ENV_STEPS)


@runtime_checkable
class EnvFactory(Protocol):
    """Mints per-task adapters from dataset rows. One instance per process;
    owns whatever shared resources the benchmark needs (spawn gates,
    connection pools). Assembly code obtains one via
    ``envs.registry.make_env_factory(config)`` — never by importing a
    concrete benchmark module."""

    def adapter_from_item(self, item: dict) -> "TaskAdapter": ...


@runtime_checkable
class TaskAdapter(Protocol):
    @property
    def task_id(self) -> str: ...

    @property
    def task_type(self) -> str:
        """Task category/family for stratified sampling and per-family
        stats (v4 SFT pipeline, analysis). Benchmarks without a natural
        taxonomy return a constant (e.g. "default")."""
        ...

    @property
    def initial_obs(self) -> str:
        """Deterministic initial observation (captured at dataset build time;
        deterministic engines must reproduce it on open_episode)."""
        ...

    def task_brief(self) -> str:
        """Task panel for the RESEARCHER: goal, environment mechanics,
        action grammar, evaluation criterion. Never includes GT."""
        ...

    def executor_scaffold(self) -> str:
        """Executor system prompt base (role + action grammar + response
        format). The guidance block is appended via types.inject_guidance;
        the scaffold itself is locked — researchers cannot alter it."""
        ...

    def tools(self) -> list[dict] | None:
        """Native function-calling schemas for the EXECUTOR, in the OpenAI
        tools format, or None for benchmarks whose action protocol is not
        tool-calling (text-action benchmarks return None).

        When present, prompt construction passes them to
        ``apply_chat_template(tools=...)`` so the chat template renders the
        tool block natively (byte-exact in-distribution formatting — never
        hand-write the rendered block into the scaffold). Parsing stays
        text-level (``protocol.parse_tool_calls`` inside react): the raw
        generated tokens are what the training stream keeps, so the engine's
        structured tool_calls representation is never round-tripped."""
        ...

    def open_episode(self) -> EnvHandle:
        """Fresh episode of THIS task. Deterministic engines must yield an
        identical initial state every time. May block (subprocess boot);
        callers wrap in asyncio.to_thread."""
        ...

    def gt_text(self) -> str | None:
        """Privileged reference (expert plan / gold answer) or None.
        Routed ONLY to the researcher prompt when the GT switch is on;
        never to the executor, never through fetch_raw."""
        ...

    def domain_notes(self) -> str:
        """Optional environment-specific hints for the narrator about what
        counts as a key event. Empty string when the generic contract is
        enough."""
        ...

    def react(self, env: EnvHandle, assistant_text: str) -> ActionOutcome:
        """THE interaction step: interpret one raw executor reply, drive the
        environment however this benchmark works, and report the outcome.

        The core imposes NO protocol here — action grammar (text tags, JSON
        tool calls, multi-action turns, code patches), execution transport,
        and evaluation all live behind this method. String-command
        benchmarks delegate to ``protocol.text_action_react`` (the reusable
        ``<reasoning>/<action>`` implementation); anything else implements
        its own parsing/execution and fills ActionOutcome directly.

        May block (environment I/O) — callers run it in a thread. Must not
        raise on bad model output (that's an ActionOutcome with
        acted=False); raise only on infrastructure faults."""
        ...

    def replay_action(self, env: EnvHandle, action_norm: str) -> tuple[str, bool]:
        """Re-execute one recorded action (a Step.action_norm) on a fresh
        episode; return (feedback_text, episode_ended).

        Determinism contract behind v4 prefix continuation: replaying a
        trajectory's recorded actions must reproduce its recorded feedback
        byte-for-byte — the caller verifies and voids the sample on drift.
        May block; callers run it in a thread."""
        ...

    # --- text-action protocol pieces -------------------------------------
    # The three members below are standard parts consumed by
    # ``protocol.text_action_react``. Benchmarks with a different action
    # protocol (whose react never delegates there) implement them as
    # trivial values (return False / protocol.DEFAULT_FORMAT_REMINDER).

    def normalize_action(self, raw: str) -> str:
        """Conservative, semantics-free normalization (no repair)."""
        ...

    def observation_indicates_invalid(self, observation: str) -> bool:
        """Whether this observation means the action was a no-op/invalid —
        an ENVIRONMENT-TEXT judgement, so it belongs to the benchmark, not
        to the task-agnostic core (通用性硬约束:core 不解析环境语义)。
        Benchmarks without such a signal return False (Step.valid stays
        True; it is an observability field, never a loss input)."""
        ...

    def format_reminder(self) -> str:
        """Observation text fed back when the assistant reply had no
        parseable action block. The PROTOCOL part (the <action> contract)
        is generic; benchmarks may prefix it with their own environment
        idiom so the reminder reads like a normal environment response
        (protocol.DEFAULT_FORMAT_REMINDER is the neutral base)."""
        ...
