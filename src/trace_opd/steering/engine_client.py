"""Engine-Resident SPS 的训练进程侧薄客户端(P1,设计文档 §3.2)。

`EngineSteeredGenerate` 与 `SpsGenerate` 同 GenerateFn 契约、同构造语义,
但每个 assistant turn 只发**一次** RPC(`server_manager.steered_generate`,
sticky 路由与普通 generate 同一 request_id ⇒ 两前缀 KV 同 replica);
draft→verify→judge 循环在 server actor 进程内跑(server_loop.run_steered_turn)。

客户端持有的 `TrajectorySteerState` 仍是轨迹状态与总账的**真源**——每 turn
把可变状态(budget/cap/gate 计数/rng)wire 过去、把回写与本 turn 账本增量
merge 回来。因此 agent_loop 下游(steer_ctx["state"]、ledger.finalize、
记账消费)对两种 executor **零感知零改动**。

机制构造参数每 turn 从 state/构造参数忠实快照重传(无状态 server,
replica 无关;几十 KB/turn,对比 python 路径每周期 MiB 级明细可忽略)。
"""
from __future__ import annotations

from typing import Any

from ..core.episode import GenResult
from .hook import TrajectorySteerState
from .wire import ledger_from_wire, merge_ledger, steer_state_apply_wire, steer_state_to_wire

__all__ = ["EngineSteeredGenerate"]


class EngineSteeredGenerate:
    def __init__(
        self,
        *,
        server_manager,
        tokenizer,
        request_id: str,
        teacher_prefix_ids: list[int],
        student_prompt_ids: list[int],
        state: TrajectorySteerState,
        topk: int,
        c: float,
        obs_topk: int,
        stop_token_ids: set[int] | frozenset[int],
        draft_horizon: int,
    ) -> None:
        self._server = server_manager
        self._tokenizer = tokenizer
        self._request_id = request_id
        self._teacher_prefix = [int(t) for t in teacher_prefix_ids]
        self._student_prompt = [int(t) for t in student_prompt_ids]
        self._state = state
        self._topk = int(topk)
        self._c = float(c)
        self._obs_topk = int(obs_topk)
        self._stop_ids = sorted(int(t) for t in stop_token_ids)
        self._draft_horizon = int(draft_horizon)

    def _mech_snapshot(self) -> dict[str, Any]:
        """构造参数的忠实快照(gate 阈值取轨迹冻结的当前值——与 python
        executor 的「构造时快照、轨迹内冻结」同语义)。"""
        g, b, cap = self._state.gate, self._state.budget, self._state.reject_cap
        return {
            "topk": self._topk,
            "c": self._c,
            "obs_topk": self._obs_topk,
            "draft_horizon": self._draft_horizon,
            "stop_token_ids": self._stop_ids,
            "d_hist_max": self._state.ledger.d_hist_max,
            "r_max": cap.r_max,
            "gate": {
                "mode": g.mode, "eta_max": g.eta_max, "epsilon": g.epsilon,
                "theta_d": g.theta_d, "entropy_theta": g.entropy_theta,
                "w_min": g.w_min, "rate_bounds": list(g.rate_bounds),
            },
            "budget": {
                "b_max": b.b_max, "c": b.c, "pacing": b.pacing, "u0": b.u0,
                "profile_power": b.profile_power, "t_max": b.t_max,
            },
        }

    async def __call__(self, prompt_ids: list[int], sampling_params: dict[str, Any]) -> GenResult:
        payload = {
            "prompt_ids": [int(t) for t in prompt_ids],
            "sampling_params": dict(sampling_params),
            "teacher_prefix_ids": self._teacher_prefix,
            "student_prompt_ids": self._student_prompt,
            "mech": self._mech_snapshot(),
            "state": steer_state_to_wire(self._state),
        }
        resp = await self._server.steered_generate(
            request_id=self._request_id, payload=payload
        )
        # 真源回写:先状态(budget/cap/gate/rng),再账本增量汇入总账
        steer_state_apply_wire(self._state, resp["state"])
        merge_ledger(self._state.ledger, ledger_from_wire(resp["ledger"]))
        ids = [int(t) for t in resp["token_ids"]]
        return GenResult(
            token_ids=ids,
            text=self._tokenizer.decode(ids, skip_special_tokens=True),
            extra=dict(resp["extra"]),
        )
