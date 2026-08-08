"""Turn-boundary token constants derived from the live chat template.

Standalone (no verl imports) so it can be probed and unit-tested outside the
training venv.

Measured Qwen template behavior (enable_thinking=False, probed on the target
server, CD-003):
  * the generation prefix may CONTAIN an empty think block
    (``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`` on thinking-family
    templates; plain ``<|im_start|>assistant\\n`` on non-thinking ones);
  * HISTORY assistant turns are re-rendered WITHOUT that block (the template
    strips think content from prior turns).
Therefore a full re-render of the conversation never matches the sampled
token stream, and the rollout must be built append-only from delta segments:

  stream = first_turn_prompt(+gen_prefix)
           | G (generated, mask 1)
           | glue (assistant close + user open, mask 0)
           | obs tokens (mask 0)
           | post_obs (user close + gen_prefix, mask 0)
           | G ...

The teacher prompt is rendered the same way, so teacher and student share the
generation context for every response token.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__file__)

_CHAT_KWARGS = {"enable_thinking": False}


def render_ids(
    tokenizer, messages, add_generation_prompt: bool, tools: list[dict] | None = None
) -> list[int]:
    """apply_chat_template → flat list[int], across transformers versions
    (5.x returns a BatchEncoding where 4.x returned a plain list).

    tools(OpenAI 格式,来自 adapter.tools())非 None 时交给模板原生渲染
    ——工具块必须是分布内形态,绝不在 scaffold 里手写;None 时不传该
    kwarg,与历史调用逐字节等价。"""
    extra = {"tools": tools} if tools is not None else {}
    out = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        **extra,
        **_CHAT_KWARGS,
    )
    if not isinstance(out, list):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return [int(x) for x in out]


def turn_stop_token_ids(tokenizer) -> list[int]:
    """轮末停止符全集:tokenizer eos + 模板轮闭合符 ``<|im_end|>``(去重排序)。

    停止条件是对话协议的一部分,必须随每个生成请求显式下发,绝不依赖
    模型资产的隐式默认——Qwen3.5-9B 官方 repo 无 generation_config.json,
    config.json 只声明 <|endoftext|>(248044);而 verl 语境 skip_tokenizer_init
    下引擎拿不到 tokenizer 的 eos,停止集里没有轮末符 <|im_end|>(248046),
    每轮生成越过轮末续写伪多轮对话(2026-08-08 9B step1 实测 5265/5268 轮
    中段含 im_end 未停、86.5% response token 是越界垃圾)。Qwen3 系
    generation_config 自带 [im_end, endoftext] 掩盖了这个假设,9B 是首个
    暴露它的模型。
    """
    ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    return sorted(t for t in ids if isinstance(t, int) and t >= 0)


_FC_PROBE_TOOL = [{
    "type": "function",
    "function": {
        "name": "fcprobe771",
        "description": "probe",
        "parameters": {
            "type": "object",
            "properties": {"arg": {"type": "string"}},
            "required": ["arg"],
        },
    },
}]


def fc_dialect(tokenizer) -> str:
    """行为探测模板家族的 function-calling wire 形态(单一真源=模板对
    assistant tool_calls 历史轮的实际渲染,不看模板源文本、不查模型名)。

    返回 protocol.FC_DIALECTS 之一:
      qwen35_xml — 渲染出 ``<function=fcprobe771>``(Qwen3.5 系,A800 实测
                   2026-08-08:外壳 <tool_call> + XML 参数体);
      hermes     — 渲染出 ``"name": "fcprobe771"`` 的 JSON 块(Qwen3 系);
      none       — 模板不渲染 tool_calls(或不支持 tools kwarg)。FC
                   benchmark 跑在 none 上必须由 preflight 拦截。
    """
    msgs = [
        {"role": "user", "content": "UPROBE551"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"type": "function",
                         "function": {"name": "fcprobe771",
                                      "arguments": {"arg": "VPROBE882"}}}]},
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            msgs, tools=_FC_PROBE_TOOL, tokenize=False,
            add_generation_prompt=False, **_CHAT_KWARGS,
        )
    except Exception:
        logger.warning("fc_dialect probe: template rejected tools/tool_calls", exc_info=True)
        return "none"
    if "<function=fcprobe771>" in rendered:
        return "qwen35_xml"
    if '"name": "fcprobe771"' in rendered or '"name":"fcprobe771"' in rendered:
        return "hermes"
    return "none"


def fc_roundtrip_probe(tokenizer, dialect: str) -> None:
    """硬门:模板渲染出的 tool_call 必须能被我方文本层解析器逐值还原。

    这是「训练通道自解析 == 模板所教形态」的两端等价断言(EOS 事故的
    姊妹防线:那次是停止符,这次是 FC wire 形态)。失败即 raise,由
    preflight 转 FAIL;dialect="none" 时调用方不应进来。"""
    from . import protocol  # 函数内 import,避免 core 模块级依赖环

    msgs = [
        {"role": "user", "content": "UPROBE551"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"type": "function",
                         "function": {"name": "fcprobe771",
                                      "arguments": {"arg": "VPROBE882"}}}]},
    ]

    def render(m):
        return tokenizer.apply_chat_template(
            m, tools=_FC_PROBE_TOOL, tokenize=False,
            add_generation_prompt=False, **_CHAT_KWARGS,
        )

    # 差分取 assistant 段:生产语境 parser 只吃 assistant 生成文本,而全量
    # 渲染产物含 system Tools 块里的格式示例调用(Qwen3.5 实测
    # <function=example_function_name>),整段喂 parser 会把示例当首个调用。
    base, full = render(msgs[:-1]), render(msgs)
    if not full.startswith(base):
        raise RuntimeError(
            "FC roundtrip probe: rendering the assistant turn does not extend "
            "the prior render as a prefix — cannot isolate the assistant segment"
        )
    assistant_seg = full[len(base):]
    schema = [_FC_PROBE_TOOL[0]]
    calls = protocol.parse_tool_calls(assistant_seg, tools=schema, dialect=dialect)
    if not (calls and calls[0]["name"] == "fcprobe771"
            and calls[0]["arguments"].get("arg") == "VPROBE882"):
        raise RuntimeError(
            f"FC roundtrip failed for dialect={dialect!r}: template renders a "
            f"tool call our text-side parser cannot recover (parsed={calls!r}). "
            f"Assistant segment: {assistant_seg[-400:]!r}"
        )


def fc_stream_equivalence_probe(tokenizer) -> dict:
    """渲染方向的正确性硬门(用户要求 2026-08-08):训练通道把 OpenAI 格式
    的 FC 交互转成原始 token 流(append-only:首轮 prompt + 生成段 + glue +
    tool_response_wrap(obs) + post_obs …),必须与推理引擎对同一 OpenAI
    messages 数组的 chat-template 整体渲染**逐 token 一致**——两条通道的
    上下文同分布是机制正确性的前提。

    两层断言,FAIL 即 raise(preflight 转 FAIL):

    A. tool 轮渲染一致性(全模板硬断言):模板渲染 ``role="tool"`` 消息的
       完整轮 == 我方 [user-open] + encode(tool_response_wrap(obs)) +
       [user-close] 拼接——守护 tool_response_wrap 字面量与模板不漂移。
    B. 完整两轮 FC 会话流等价:append-only 拼接 == 整体渲染(带下一轮
       generation prompt)。历史 assistant 轮渲染与 gen_prefix 不同头的
       模板(Qwen3 thinking 系剥 think 块,TemplateDelta docstring 记录的
       已知性质)结构上不可全等,该层降级 SKIPPED 并在返回值注明——运行
       正确性不受影响(训练/评测都走 append-only,从不重渲染)。

    返回 {"tool_turn": "PASS", "full_stream": "PASS"|"SKIPPED:think-strip"}。
    """
    from . import protocol  # 函数内 import,避免 core 模块级依赖环

    t = tokenizer
    delta = TemplateDelta(t)
    tools = _FC_PROBE_TOOL

    def call(n):
        return {"type": "function",
                "function": {"name": "fcprobe771",
                             "arguments": {"arg": f"VPROBE88{n}"}}}

    obs1 = "OPROBE313 first tool result."
    obs2 = "OPROBE314 second tool result."
    msgs = [
        {"role": "system", "content": "SPROBE224"},
        {"role": "user", "content": "UPROBE551"},
        {"role": "assistant", "content": "Reasoning one.", "tool_calls": [call(1)]},
        {"role": "tool", "content": obs1},
        {"role": "assistant", "content": "Reasoning two.", "tool_calls": [call(2)]},
        {"role": "tool", "content": obs2},
    ]

    def rids(m, gen):
        return render_ids(t, m, gen, tools=tools)

    # prefixes[j] = render(msgs[:j+2]) — j=0 是 sys+user,此后每级 +1 轮。
    prefixes = [rids(msgs[:k], False) for k in range(2, len(msgs) + 1)]
    for a, b in zip(prefixes, prefixes[1:]):
        if b[: len(a)] != a:
            # Qwen3 thinking 系指纹:末轮 assistant 带空 think、变历史轮后
            # 被剥 ⇒ 逐级渲染非前缀扩展,差分法不可用(实测 Qwen3-1.7B,
            # 2026-08-08)。该家族当前只跑 text 协议;FC 环境若要上此类
            # 模板,preflight 会报 SKIPPED,需人工专项验证后放行。
            logger.info(
                "FC stream probe: incremental renders are not prefix-extending "
                "(think-stripping template family) — equivalence checks skipped."
            )
            return {"tool_turn": "SKIPPED:think-strip",
                    "full_stream": "SKIPPED:think-strip"}

    # ---- A. tool 轮渲染一致性 ----------------------------------------
    # 模板对 tool 轮的增量 vs 我方注入形态。glue_full=[stop, …user-open…];
    # post_obs=[…user-close…]+gen_prefix(TemplateDelta 差分定义)。
    tool_seg = prefixes[2][len(prefixes[1]):]
    user_close = delta.post_obs[: len(delta.post_obs) - len(delta.gen_prefix)]
    ours_tool = (
        delta.glue_full[2:]
        + t.encode(protocol.tool_response_wrap(obs1), add_special_tokens=False)
        + user_close
    )
    if ours_tool != tool_seg:
        raise RuntimeError(
            "FC stream probe FAILED (tool turn): our injected tool feedback "
            "does not match the template's rendering of a role=tool message.\n"
            f"  template: {t.decode(tool_seg)!r}\n"
            f"  ours:     {t.decode(ours_tool)!r}\n"
            "tool_response_wrap (core/protocol.py) has drifted from the chat "
            "template — fix the wrap, never the template."
        )

    # ---- B. 完整流等价 ------------------------------------------------
    a_seg = prefixes[1][len(prefixes[0]):]
    if a_seg[: len(delta.gen_prefix)] != delta.gen_prefix:
        logger.info(
            "FC stream probe: history-assistant rendering differs from the "
            "generation prefix (think-stripping template family) — full-stream "
            "equality is structurally inapplicable; tool-turn check passed."
        )
        return {"tool_turn": "PASS", "full_stream": "SKIPPED:think-strip"}

    stream = list(rids(msgs[:2], True))
    # assistant 轮增量:msgs[2] → prefixes[1]−prefixes[0];msgs[4] →
    # prefixes[3]−prefixes[2](prefixes 下标与 msgs 下标差 2)。
    for a_idx, obs in ((1, obs1), (3, obs2)):
        seg = prefixes[a_idx][len(prefixes[a_idx - 1]):]
        if seg[: len(delta.gen_prefix)] != delta.gen_prefix:
            raise RuntimeError("FC stream probe: assistant turn head drifted mid-probe")
        body = seg[len(delta.gen_prefix):]
        stops = [i for i, tok_id in enumerate(body) if tok_id in delta.stop_ids]
        if not stops:
            raise RuntimeError("FC stream probe: assistant turn carries no stop token")
        gen = body[: stops[-1] + 1]  # 引擎语义:停止符保留在生成段尾
        stream += gen
        stream += delta.after_generation(gen)
        stream += t.encode(protocol.tool_response_wrap(obs), add_special_tokens=False)
        stream += delta.post_obs
    truth = rids(msgs, True)
    if stream != truth:
        cut = next(
            (i for i, (x, y) in enumerate(zip(stream, truth)) if x != y),
            min(len(stream), len(truth)),
        )
        raise RuntimeError(
            "FC stream probe FAILED (full stream): append-only assembly diverges "
            f"from the engine-side whole render at token {cut}.\n"
            f"  ours:  …{t.decode(stream[max(0, cut - 20): cut + 20])!r}\n"
            f"  truth: …{t.decode(truth[max(0, cut - 20): cut + 20])!r}"
        )
    return {"tool_turn": "PASS", "full_stream": "PASS"}


def _find_subseq(haystack: list[int], needle: list[int]) -> int:
    n, m = len(haystack), len(needle)
    for i in range(n - m + 1):
        if haystack[i : i + m] == needle:
            return i
    return -1


class TemplateDelta:
    """Probe the chat template with sentinel strings and extract, as token
    lists, the append-only segments between turns. Hard-asserts on any
    template drift instead of corrupting the stream."""

    def __init__(self, tokenizer):
        S, U, B, O = "SPROBE224", "UPROBE551", "BPROBE917", "OPROBE313"
        t = tokenizer

        def render(messages, add_gen):
            return render_ids(t, messages, add_gen)

        msgs0 = [
            {"role": "system", "content": S},
            {"role": "user", "content": U},
        ]
        base = render(msgs0, False)
        with_gen = render(msgs0, True)
        assert with_gen[: len(base)] == base, "gen prompt does not extend base render"
        self.gen_prefix: list[int] = with_gen[len(base) :]

        # One full exchange after the first turn: history-assistant + user + gen prefix.
        msgs_u = msgs0 + [
            {"role": "assistant", "content": B},
            {"role": "user", "content": O},
        ]
        full_u = render(msgs_u, True)
        assert full_u[: len(base)] == base, "extended render does not start with base"
        tail = full_u[len(base) :]
        b_ids = t.encode(B, add_special_tokens=False)
        o_ids = t.encode(O, add_special_tokens=False)
        i = _find_subseq(tail, b_ids)
        j = _find_subseq(tail, o_ids)
        assert 0 <= i and i + len(b_ids) <= j, "cannot locate probes in rendered turns"

        # glue: everything between assistant content and the user obs
        # (assistant close + user open); post_obs: everything after the obs
        # (user close + next generation prefix).
        self.glue_full: list[int] = tail[i + len(b_ids) : j]
        self.post_obs: list[int] = tail[j + len(o_ids) :]

        self.stop_ids = set(turn_stop_token_ids(t))
        assert self.glue_full and self.glue_full[0] in self.stop_ids, (
            "glue does not start with a stop token"
        )
        assert self.post_obs[-len(self.gen_prefix) :] == self.gen_prefix, (
            "post_obs does not end with the generation prefix"
        )

        #: 模板家族的 FC wire 形态("none" 合法——纯文本协议 benchmark 不用;
        #: FC benchmark 跑在 none 上由 preflight/入口拦截,不在此硬断)。
        self.fc_dialect: str = fc_dialect(t)

        logger.info(
            "TemplateDelta ready: gen_prefix=%r glue=%r post_obs=%r fc_dialect=%s",
            t.decode(self.gen_prefix),
            t.decode(self.glue_full),
            t.decode(self.post_obs),
            self.fc_dialect,
        )

    def after_generation(self, gen_ids: list[int]) -> list[int]:
        """Structure tokens to append between a generation and the next obs.

        Handles both backends: engine generations may end with their own stop
        token (don't duplicate it); re-tokenized external generations never do
        (the completions API strips the stop token), so they get the full glue.
        """
        if gen_ids and gen_ids[-1] in self.stop_ids:
            return self.glue_full[1:]
        return self.glue_full
