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


def render_ids(tokenizer, messages, add_generation_prompt: bool) -> list[int]:
    """apply_chat_template → flat list[int], across transformers versions
    (5.x returns a BatchEncoding where 4.x returned a plain list)."""
    out = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        **_CHAT_KWARGS,
    )
    if not isinstance(out, list):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return [int(x) for x in out]


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

        eos = t.eos_token_id
        im_end = t.convert_tokens_to_ids("<|im_end|>")
        self.stop_ids = {x for x in (eos, im_end) if x is not None}
        assert self.glue_full and self.glue_full[0] in self.stop_ids, (
            "glue does not start with a stop token"
        )
        assert self.post_obs[-len(self.gen_prefix) :] == self.gen_prefix, (
            "post_obs does not end with the generation prefix"
        )

        logger.info(
            "TemplateDelta ready: gen_prefix=%r glue=%r post_obs=%r",
            t.decode(self.gen_prefix),
            t.decode(self.glue_full),
            t.decode(self.post_obs),
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
