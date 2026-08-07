"""Test doubles: a chat-template-faithful fake tokenizer, a scripted
environment/adapter, scripted GenerateFn and researcher-chat backends.

The FakeTokenizer implements enough of the HF tokenizer surface for
TemplateDelta probing and episode-stream construction: special markers map
to reserved ids, ordinary text tokenizes per character."""
from __future__ import annotations

from typing import Any

from trace_opd.core import protocol
from trace_opd.core.episode import GenResult
from trace_opd.core.types import Trajectory

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


class FakeTokenizer:
    _SPECIALS = {IM_END: 1000001, IM_START: 1000002}
    _BY_ID = {v: k for k, v in _SPECIALS.items()}
    eos_token_id = 1000001

    def convert_tokens_to_ids(self, token: str):
        return self._SPECIALS.get(token)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            for special, sid in self._SPECIALS.items():
                if text.startswith(special, i):
                    ids.append(sid)
                    i += len(special)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        parts: list[str] = []
        for tid in ids:
            tid = int(tid)
            if tid in self._BY_ID:
                if not skip_special_tokens:
                    parts.append(self._BY_ID[tid])
            else:
                parts.append(chr(tid))
        return "".join(parts)

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs: Any,
    ):
        s = ""
        for m in messages:
            s += f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n"
        if add_generation_prompt:
            s += f"{IM_START}assistant\n"
        return self.encode(s) if tokenize else s


class ScriptedEnv:
    """EnvHandle double: a fixed initial obs and a scripted command->response
    table; ``win_on`` command ends the episode with won=True."""

    def __init__(self, initial_obs: str, responses: dict[str, str], win_on: str) -> None:
        self.initial_obs = initial_obs
        self._responses = responses
        self._win_on = win_on
        self.commands: list[str] = []
        self.closed = False

    def step(self, command: str) -> tuple[str, bool, bool]:
        self.commands.append(command)
        if command == self._win_on:
            return "You won!", True, True
        return self._responses.get(command, "Nothing happens."), False, False

    def close(self) -> None:
        self.closed = True


class ScriptedAdapter:
    """TaskAdapter double bound to a ScriptedEnv factory."""

    def __init__(
        self,
        task_id: str = "fake/task_1",
        initial_obs: str = "You are in a room. Your task is to: win.",
        responses: dict[str, str] | None = None,
        win_on: str = "do the thing",
        gt: str | None = "1. do the thing",
        fail_open: bool = False,
        env_initial_obs: str | None = None,  # differing = simulated env drift
    ) -> None:
        self._task_id = task_id
        self._initial_obs = initial_obs
        self._env_initial_obs = env_initial_obs
        self._responses = responses or {"look": "An empty room."}
        self._win_on = win_on
        self._gt = gt
        self._fail_open = fail_open
        self.envs: list[ScriptedEnv] = []

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def task_type(self) -> str:
        return "fake"

    @property
    def initial_obs(self) -> str:
        return self._initial_obs

    def task_brief(self) -> str:
        return f"A fake task. Start: {self._initial_obs}"

    def executor_scaffold(self) -> str:
        return (
            "You are a fake agent. Respond with <reasoning>..</reasoning>\n"
            "<action>one command</action>"
        )

    def open_episode(self) -> ScriptedEnv:
        if self._fail_open:
            raise RuntimeError("engine exploded")
        env = ScriptedEnv(
            self._env_initial_obs or self._initial_obs, self._responses, self._win_on
        )
        self.envs.append(env)
        return env

    def gt_text(self) -> str | None:
        return self._gt

    def domain_notes(self) -> str:
        return ""

    def normalize_action(self, raw: str) -> str:
        return raw.strip().lower()

    def observation_indicates_invalid(self, observation: str) -> bool:
        # ScriptedEnv 的 miss 响应就是这个字面量(与 ALFWorld 同风格)
        return observation.strip() == "Nothing happens."

    def format_reminder(self) -> str:
        return "Nothing happens. " + protocol.DEFAULT_FORMAT_REMINDER


class ScriptedGenerate:
    """GenerateFn double: yields the scripted texts in order (re-tokenized by
    the fake tokenizer, external-backend style)."""

    def __init__(self, tokenizer: FakeTokenizer, texts: list[str]) -> None:
        self._tokenizer = tokenizer
        self._texts = list(texts)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, prompt_ids: list[int], sampling_params: dict[str, Any]) -> GenResult:
        self.calls.append({"prompt_len": len(prompt_ids), "params": dict(sampling_params)})
        if not self._texts:
            return GenResult(token_ids=[], text="")
        text = self._texts.pop(0)
        return GenResult(
            token_ids=self._tokenizer.encode(text), text=text, extra={"min_global_steps": 7}
        )


def reply(reasoning: str, action: str) -> str:
    return f"<reasoning>{reasoning}</reasoning>\n<action>{action}</action>"


def make_trajectory(task_id: str = "fake/task_1", won: bool = False) -> Trajectory:
    from trace_opd.core.types import OUTCOME_FAILED, OUTCOME_SUCCESS, Step

    traj = Trajectory(
        episode_id="ep001",
        task_id=task_id,
        guidance_hash="bare",
        sample_idx=0,
        initial_obs="You are in a room.",
        outcome=OUTCOME_SUCCESS if won else OUTCOME_FAILED,
        won=won,
    )
    traj.steps.append(
        Step(t=1, thought="try", action_raw="look", action_norm="look",
             observation="An empty room.", valid=True)
    )
    return traj
