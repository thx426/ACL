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

from typing import Protocol, runtime_checkable


@runtime_checkable
class EnvHandle(Protocol):
    """One live episode. Implementations must be safe to close() twice."""

    @property
    def initial_obs(self) -> str: ...

    def step(self, command: str) -> tuple[str, bool, bool]:
        """Execute one normalized command; return (observation, done, won)."""
        ...

    def close(self) -> None: ...


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
