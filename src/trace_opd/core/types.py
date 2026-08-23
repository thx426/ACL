"""Core data types shared by the whole mechanism.

Design contract:
  - Trajectory is the structured ground truth of one executor episode,
    produced by program code only; every downstream artifact (rollout dump,
    eval record) derives from it mechanically.
  - The guidance slot is a protocol-level constant (PG-7): the SAME header
    ("## Task guidance") carries the privileged doc in the SPS teacher
    prefix and in +doc evaluation arms — byte-identical everywhere.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

# ----------------------------------------------------------------- outcomes

OUTCOME_SUCCESS = "SUCCESS"
OUTCOME_FAILED = "FAILED"
OUTCOME_ENV_ERROR = "ENV_ERROR"
DEGENERATE_PREFIX = "DEGENERATE:"  # e.g. DEGENERATE:format


def is_degenerate(outcome: str) -> bool:
    return outcome.startswith(DEGENERATE_PREFIX)


def counts_as_attempt(outcome: str) -> bool:
    """ENV_ERROR episodes are infrastructure faults: they never count as
    task attempts (neither success nor failure) and are refunded."""
    return outcome != OUTCOME_ENV_ERROR


# ----------------------------------------------------------- guidance slot

GUIDANCE_HEADER = "## Task guidance"

#: 特权块的框(2026-08-11 裁定)。**为什么需要它**:guidance 与任务陈述都是
#: 大段自然语言,拼在一起后没有任何标记区分"参考资料"与"要做的事"。任务陈述
#: 越长(终端域实测 3.4-6.5k 字符),这个歧义越贵:执行体要么把 guidance 当成
#: 新任务去照做,要么把它当噪声略过。同一个歧义在 researcher 侧已实证有害
#: (角色段被 9 倍长的任务陈述压过,模型滑回 executor 身份)。
#:
#: 框里同时消解**特权幻觉**:guidance 由研究者从若干次尝试里蒸馏而来,可能
#: 过时或本就观察有偏;执行体每个 episode 从零开始,无法核对这些断言。明说
#: "这是建议不是事实、依赖前先自己确认",让它保持自行查证的习惯 —— 与文档
#: 侧的信息状态规则(prompts.DOC_RULES_TEMPLATE)是同一约束的两端。
#:
#: 协议级常量:teacher 前缀与 +doc 评测臂共用这一份,改动会改变两者的字节
#: (guidance_hash 只对 guidance 正文取值,不含框,故指纹口径不受影响)。
GUIDANCE_FRAME_OPEN = (
    "(Reference notes distilled by a researcher from earlier attempts at this "
    "same task. They are advisory background, NOT the task and NOT commands to "
    "replay: they may be incomplete, stale, or wrong, so confirm anything you "
    "rely on with your own actions before trusting it.)"
)
GUIDANCE_FRAME_CLOSE = (
    "(End of reference notes. Your task is the one stated in the user message; "
    "carry that out — the notes only inform how you go about it.)"
)


def inject_guidance(system_base: str, guidance: str) -> str:
    """Append the guidance block to an executor system prompt.

    Single source of truth for the injection format (protocol-level
    constant; keep byte-stable — the OPD teacher prefix and the +doc
    evaluation arm use the same)."""
    guidance = (guidance or "").strip()
    if not guidance:
        return system_base
    return (
        f"{system_base}\n\n{GUIDANCE_HEADER}\n"
        f"{GUIDANCE_FRAME_OPEN}\n\n{guidance}\n\n{GUIDANCE_FRAME_CLOSE}"
    )


def guidance_hash(guidance: str) -> str:
    guidance = (guidance or "").strip()
    if not guidance:
        return "bare"
    return hashlib.sha1(guidance.encode("utf-8")).hexdigest()[:12]


# ----------------------------------------------------------------- episode


@dataclass
class Step:
    """One executor turn.

    acted=False marks turns that never reached the environment (the reply
    carried no parseable action under the benchmark's protocol; the
    recorded observation is then the adapter's format reminder).
    valid=False marks environment no-ops
    (adapter.observation_indicates_invalid; observability only, never a
    loss input)."""

    t: int
    thought: str
    action_raw: str
    action_norm: str
    observation: str
    valid: bool
    acted: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "thought": self.thought,
            "action_raw": self.action_raw,
            "action_norm": self.action_norm,
            "observation": self.observation,
            "valid": self.valid,
            "acted": self.acted,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Step":
        return Step(
            t=int(d["t"]),
            thought=str(d.get("thought", "")),
            action_raw=str(d.get("action_raw", "")),
            action_norm=str(d.get("action_norm", "")),
            observation=str(d.get("observation", "")),
            valid=bool(d.get("valid", False)),
            acted=bool(d.get("acted", True)),
        )


@dataclass
class Trajectory:
    episode_id: str
    task_id: str
    guidance_hash: str
    sample_idx: int
    initial_obs: str
    steps: list[Step] = field(default_factory=list)
    outcome: str = OUTCOME_FAILED
    won: bool = False
    #: 终态评估分数(ActionOutcome.score 透传;won = score >= 1.0 的派生
    #: 视图,二元 benchmark 下两者信息等价;部分分 benchmark 的保真通道)
    score: float = 0.0
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def acted_actions(self) -> list[str]:
        """Normalized actions that actually reached the engine, in order
        (includes invalid ones)."""
        return [s.action_norm for s in self.steps if s.acted]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "guidance_hash": self.guidance_hash,
            "sample_idx": self.sample_idx,
            "initial_obs": self.initial_obs,
            "steps": [s.to_dict() for s in self.steps],
            "outcome": self.outcome,
            "won": self.won,
            "score": self.score,
            "error": self.error,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Trajectory":
        return Trajectory(
            episode_id=str(d["episode_id"]),
            task_id=str(d["task_id"]),
            guidance_hash=str(d.get("guidance_hash", "bare")),
            sample_idx=int(d.get("sample_idx", 0)),
            initial_obs=str(d.get("initial_obs", "")),
            steps=[Step.from_dict(s) for s in d.get("steps", [])],
            outcome=str(d.get("outcome", OUTCOME_FAILED)),
            won=bool(d.get("won", False)),
            score=float(d.get("score", 1.0 if d.get("won") else 0.0)),
            error=d.get("error"),
            meta=dict(d.get("meta", {})),
        )

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @staticmethod
    def loads(s: str) -> "Trajectory":
        return Trajectory.from_dict(json.loads(s))


# ---------------------------------------------------------------- fact card


@dataclass
class FactCard:
    """Program-held truth about one episode, placed NEXT TO the narration
    (never merged into it). Only enumerable, program-verifiable facts."""

    outcome: str
    steps: int
    invalid_count: int
    not_acted_count: int
    longest_repeat_action: str
    longest_repeat_count: int
    final_obs_excerpt: str

    def render(self) -> str:
        parts = [
            f"outcome={self.outcome}",
            f"steps={self.steps}",
            f"invalid_actions={self.invalid_count}",
        ]
        if self.not_acted_count:
            parts.append(f"unparseable_replies={self.not_acted_count}")
        if self.longest_repeat_count >= 3:
            parts.append(
                f'longest_repeat="{self.longest_repeat_action}"x{self.longest_repeat_count}'
            )
        parts.append(f'final_obs="{self.final_obs_excerpt}"')
        return "FACT CARD (program-verified): " + " | ".join(parts)


@dataclass
class EvalResult:
    outcome: str
    won: bool
    steps: int
    detail: str = ""

    def render(self) -> str:
        return f"EVALUATION: {self.outcome} (steps={self.steps}){(' — ' + self.detail) if self.detail else ''}"
