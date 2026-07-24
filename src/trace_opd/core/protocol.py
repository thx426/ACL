"""Interaction contract (single source of truth).

Everything that defines the executor<->environment conversation lives here so
research experiments, the training rollout (verl agent loop), the teacher
prompt construction, and the standalone evaluator can never drift apart:

  * the bare system prompt, and guidance injection via
    ``types.inject_guidance`` (the SAME ``## Task guidance`` block serves
    researcher guidance, the privileged teacher prefix, and +doc eval arms);
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

COMMAND_REFERENCE is quoted from the engine's own ``help`` output (AlfredTWEnv
text mode), captured verbatim on the target server (CD-003,
experiments/h3_pilot/notes/engine_help.txt) — never paraphrase.

Inherited verbatim from the CD-003 contract (PG-3) with ONE deliberate
change: the privileged section header is the project-wide GUIDANCE_HEADER
("## Task guidance") instead of the h3 static-skill header, so the teacher
prefix is byte-identical with research-time guidance injection (PG-7).
"""

from __future__ import annotations

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

# Command reference ---------------------------------------------------------
COMMAND_REFERENCE = """\
Available commands:
  look:                             look around your current location
  inventory:                        check your current inventory
  go to (receptacle):               move to a receptacle
  open (receptacle):                open a receptacle
  close (receptacle):               close a receptacle
  take (object) from (receptacle):  take an object from a receptacle
  move (object) to (receptacle):  place an object in or on a receptacle
  examine (something):              examine a receptacle or an object
  use (object):                     use an object
  heat (object) with (receptacle):  heat an object using a receptacle
  clean (object) with (receptacle): clean an object using a receptacle
  cool (object) with (receptacle):  cool an object using a receptacle
  slice (object) with (object):     slice an object using a sharp object"""

SYSTEM_PROMPT_BARE = (
    "You are an agent in the ALFRED text-based household environment. "
    "Complete the task stated after 'Your task is to:' by issuing exactly one "
    "environment command per turn.\n\n"
    + COMMAND_REFERENCE
    + "\n\n"
    "Use exact entity names with their numbers as observed (e.g. 'bread 1', "
    "'countertop 2'). An invalid or currently impossible command returns "
    "'Nothing happens.'.\n\n"
    "On every turn respond with exactly this format and nothing else:\n"
    "<reasoning>brief reasoning about the current situation</reasoning>\n"
    "<action>one command</action>\n\n"
    "The episode ends automatically when the task goal is reached; there is "
    "no submit command and no partial credit."
)


def build_system_prompt(guidance: str | None = None) -> str:
    """Bare system prompt, or the privileged variant with guidance appended
    under the project-wide GUIDANCE_HEADER (single injection format)."""
    return inject_guidance(SYSTEM_PROMPT_BARE, guidance or "")


# Assistant output parsing --------------------------------------------------

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE)

INVALID_ACTION_OBS = "Nothing happens."
FORMAT_REMINDER_OBS = (
    "Nothing happens. (Your reply had no <action>...</action> block — respond "
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


def parse_reasoning(assistant_text: str) -> str | None:
    """Content of the LAST <reasoning> block, or None if absent."""
    matches = _REASONING_RE.findall(assistant_text or "")
    if not matches:
        return None
    return matches[-1].strip() or None


def truncate_obs(obs: str, limit: int = OBS_TRUNCATE_CHARS) -> str:
    if len(obs) <= limit:
        return obs
    head = limit // 2
    tail = limit - head
    return obs[:head] + "\n[... observation truncated ...]\n" + obs[-tail:]
