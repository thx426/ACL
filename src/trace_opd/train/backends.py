"""Engine-backend adapter: the verl rollout engine (current policy weights,
auto-synced every step) exposed as core.episode.GenerateFn — the student
rollout's唯一生成通道(PGS:SPS 钩子在 episode 层复用同一通道做
draft/verify,见 docs/pgs-impl-design.md §2)。
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from ..core.episode import GenResult
from ..core.template_delta import render_ids, turn_stop_token_ids


class EngineGenerate:
    """The request_id here is a ROUTING KEY only: verl's LLMServerClient uses
    it for sticky replica selection and sends a fresh uuid to vLLM per call
    (llm_server.py), so any number of concurrent requests may share one key.

    Pass a per-(step, task) key so EVERYTHING about one task — all rollout_n
    student episodes (each multi-turn), every research experiment episode —
    lands on the same TP2 replica and hits its prefix cache (user issue
    2026-07-23: no task affinity ⇒ shared prefixes re-prefilled per replica).
    Default (None) keeps the old per-episode identity."""

    def __init__(self, server_manager, tokenizer, request_id: str | None = None) -> None:
        self._server = server_manager
        self._tokenizer = tokenizer
        self._request_id = request_id or uuid4().hex
        # 轮末停止符必须随请求显式下发(见 turn_stop_token_ids docstring):
        # verl 起引擎用 skip_tokenizer_init,引擎侧停止集只剩模型资产推导
        # (Qwen3.5 无 generation_config ⇒ 只认 <|endoftext|>,轮末 <|im_end|>
        # 失守 ⇒ 越界续写伪多轮)。这里是所有引擎生成的唯一通道,单点兜底;
        # 调用方显式传入的 stop_token_ids 优先。
        self._stop_token_ids = turn_stop_token_ids(tokenizer)

    async def __call__(self, prompt_ids: list[int], sampling_params: dict[str, Any]) -> GenResult:
        params = dict(sampling_params)
        if self._stop_token_ids:
            params.setdefault("stop_token_ids", self._stop_token_ids)
        out = await self._server.generate(
            request_id=self._request_id,
            prompt_ids=prompt_ids,
            sampling_params=params,
        )
        ids = list(out.token_ids)
        extra: dict[str, Any] = {}
        out_extra = out.extra_fields or {}
        for key in ("min_global_steps", "max_global_steps"):
            if key in out_extra:
                extra[key] = out_extra[key]
        # PGS/SPS 数据通道(请求带 logprobs=K / prompt_logprobs=K 时才在场):
        #   log_probs             采样 token 自身的 logprob(draft 的 log π(y))
        #   response_topk_*       生成位置的 π top-K 明细(rank 序;S 的 π 侧)
        #   prompt_ids/logprobs   prompt_logprobs 打分行(verify 的 p_T 侧;
        #                         行语义:第 j 行预测位置 j+1,末行 dummy)
        if getattr(out, "log_probs", None) is not None:
            extra["log_probs"] = list(out.log_probs)
        for key in ("response_topk_ids", "response_topk_logprobs", "prompt_ids", "prompt_logprobs"):
            if key in out_extra:
                extra[key] = out_extra[key]
        # verl aborts (e.g. weight sync preemption) return empty ids with
        # stop_reason="aborted" — the episode runner classifies that as an
        # infrastructure fault, not a task outcome.
        stop_reason = getattr(out, "stop_reason", None)
        if stop_reason is not None:
            extra["stop_reason"] = stop_reason
        return GenResult(
            token_ids=ids,
            text=self._tokenizer.decode(ids, skip_special_tokens=True),
            extra=extra,
        )


class EngineResearcherChat:
    """policy-as-researcher 的引擎聊天适配(v4,docs/pgs-self-researcher-design.md §1.1)。

    对 ResearchSession 暴露与 external client 相同的 (messages, tools) →
    ChatResult 契约(session.py 零改动),内部维护 **append-only token 流**
    ——训练 token 必须 = 引擎实际采样 token(messages 重渲染有 BPE
    roundtrip 不一致风险,历史设计红线):

      * 首调:全量渲染当时 messages(system+kickoff[+reminder])+ 生成前缀
        = 训练行 prompt_ids;
      * 生成 token ids 原样 append(mask=1);
      * 此后每调:diff 新增消息——assistant 轮必须逐字等于上次生成文本
        (fail loud 防会话改写);新增 user 轮**合并**为一个 user 结构段
        (glue + text + post_obs,mask=0;chat template 层面连续 user 的
        安全合并,messages 归档保持原分列结构,语义一致);
      * 会话结束由 training_row() 取 (prompt_ids, response_ids, mask)。

    token 总量防线:流长逼近 max_total_tokens 时返回 finish_reason="length"
    的空 tool 轮——会话靠轮数上限自然收敛;训练行超长降级由调用方按
    research_max_sample_tokens 判(§1.6)。
    """

    def __init__(
        self,
        *,
        generate,                     # (prompt_ids, sampling_params) -> GenResult
        tokenizer,
        delta,                        # core.template_delta.TemplateDelta
        temperature: float,
        max_tokens_per_turn: int,
        max_total_tokens: int,
    ) -> None:
        self._generate = generate
        self._tokenizer = tokenizer
        self._delta = delta
        self._temperature = float(temperature)
        self._max_tokens_per_turn = int(max_tokens_per_turn)
        self._max_total_tokens = int(max_total_tokens)
        self.prompt_ids: list[int] = []
        self._stream: list[int] = []          # response 流(prompt 之后)
        self._mask: list[int] = []
        self._seen_messages = 0
        self._last_gen_text: str | None = None
        self.aborted = False                  # 引擎 abort(基建故障,调用方处置)
        # research 行的训练 token 同样由引擎生成,必须与 executor 行一样携带
        # policy 版本区间:trainer 的 _compute_metrics 对**每一行**都做
        # np.array([tag["min_global_steps"] ...], dtype=int),缺键取到 None
        # 就 TypeError 整步崩(2026-08-07 162 v4 冒烟实测)。
        self.min_global_steps: int | None = None
        self.max_global_steps: int | None = None

    def training_row(self) -> tuple[list[int], list[int], list[int]]:
        return list(self.prompt_ids), list(self._stream), list(self._mask)

    def _track_policy_version(self, extra) -> None:
        """累积本会话全部生成的 policy 版本区间(min of mins / max of maxes)。

        与 executor 行同源:两者的训练 token 都出自同一 rollout 引擎,版本证据
        必须同样落到 extra_fields,否则 research 行在 trainer 侧是 None。
        """
        for key, agg in (("min_global_steps", min), ("max_global_steps", max)):
            raw = (extra or {}).get(key)
            raw = getattr(raw, "data", raw)
            if raw is None:
                continue
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            current = getattr(self, key)
            setattr(self, key, value if current is None else agg(current, value))

    def policy_version_fields(self) -> dict[str, int]:
        """给行组装用:缺版本证据时返回空 dict,由调用方 fail loud。"""
        if self.min_global_steps is None or self.max_global_steps is None:
            return {}
        return {"min_global_steps": self.min_global_steps,
                "max_global_steps": self.max_global_steps}

    def _append_user_block(self, texts: list[str]) -> None:
        """把新增 user 内容合并为一个 user 结构段(mask=0)。glue 的
        stop-token 去重语义由 after_generation 按流尾真值处理(引擎生成
        可能自带停止符);post_obs 末尾自带下一轮生成前缀。"""
        if not texts:
            return
        merged = "\n\n".join(t for t in texts if t)
        tail = self._stream[-1:]              # 上一段生成的最后一个 token
        glue = self._delta.after_generation(tail)
        obs_ids = self._tokenizer.encode(merged, add_special_tokens=False)
        structure = glue + obs_ids + self._delta.post_obs
        self._stream += structure
        self._mask += [0] * len(structure)

    async def __call__(self, messages: list[dict[str, Any]], tools) -> "ChatResult":
        from ..common.llm import ChatResult

        first_call = self._seen_messages == 0
        new = messages[self._seen_messages:]
        if first_call:
            # tools 交 chat template 原生渲染(TextProtocol 升级为客户端解析
            # FC 后 session 传入 schema;None 时与历史渲染逐字节等价)。
            self.prompt_ids = render_ids(self._tokenizer, list(messages), True, tools=tools)
        else:
            user_texts: list[str] = []
            for m in new:
                if m["role"] == "assistant":
                    # 会话对 assistant 轮的唯一合法来源是本对象的上一次生成
                    if (m.get("content") or "") != (self._last_gen_text or ""):
                        raise RuntimeError(
                            "EngineResearcherChat: assistant message diverged from "
                            "the engine's own generation — token stream would desync"
                        )
                elif m["role"] == "user":
                    user_texts.append(str(m.get("content") or ""))
                else:
                    raise RuntimeError(
                        f"EngineResearcherChat: unexpected role {m['role']!r} mid-session"
                    )
            self._append_user_block(user_texts)
        self._seen_messages = len(messages)

        budget = self._max_total_tokens - len(self.prompt_ids) - len(self._stream)
        if budget < 32:
            return ChatResult(content="", finish_reason="length")
        gen = await self._generate(
            self.prompt_ids + self._stream,
            {
                "temperature": self._temperature,
                "max_tokens": min(self._max_tokens_per_turn, budget),
            },
        )
        if gen.extra.get("stop_reason") == "aborted":
            self.aborted = True
            return ChatResult(content="", finish_reason="aborted")
        self._track_policy_version(gen.extra)
        gen_ids = list(gen.token_ids)
        self._stream += gen_ids
        self._mask += [1] * len(gen_ids)
        self._last_gen_text = gen.text
        return ChatResult(
            content=gen.text,
            prompt_tokens=len(self.prompt_ids) if first_call else 0,
            completion_tokens=len(gen_ids),
            finish_reason="stop",
        )


def engine_narrate_fn(server_manager, tokenizer, request_id: str | None = None):
    """在当前 rollout engine 上做单轮轨迹叙述；不假设 vLLM 私有接口。"""

    async def narrate(messages: list[dict[str, Any]], temperature: float, max_tokens: int) -> str:
        prompt_ids = render_ids(tokenizer, messages, True)
        out = await server_manager.generate(
            request_id=request_id or uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params={"temperature": temperature, "max_tokens": max_tokens},
        )
        return tokenizer.decode(list(out.token_ids), skip_special_tokens=True)

    return narrate
