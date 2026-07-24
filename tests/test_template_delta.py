from fakes import IM_END, FakeTokenizer

from trace_opd.core.template_delta import TemplateDelta, render_ids


def test_probe_extracts_consistent_segments():
    t = FakeTokenizer()
    delta = TemplateDelta(t)
    # gen prefix is the assistant opener
    assert t.decode(delta.gen_prefix, skip_special_tokens=False) == "<|im_start|>assistant\n"
    # glue closes the assistant turn and opens the user turn
    assert delta.glue_full[0] in delta.stop_ids
    assert t.decode(delta.glue_full, skip_special_tokens=False) == (
        "<|im_end|>\n<|im_start|>user\n"
    )
    # post_obs closes the user turn and ends with the generation prefix
    assert delta.post_obs[-len(delta.gen_prefix):] == delta.gen_prefix


def test_append_only_stream_matches_full_render():
    """prompt + gen + glue + obs + post_obs must equal a full re-render of the
    two-turn conversation (the fake template has no think-stripping, so the
    equality is exact — the delta construction itself is what we verify)."""
    t = FakeTokenizer()
    delta = TemplateDelta(t)
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "OBS0"},
    ]
    prompt = render_ids(t, msgs, True)
    gen = t.encode("hello")  # external-style: no trailing stop token
    obs = t.encode("OBS1")
    stream = prompt + gen + delta.after_generation(gen) + obs + delta.post_obs
    full = render_ids(
        t,
        msgs + [{"role": "assistant", "content": "hello"}, {"role": "user", "content": "OBS1"}],
        True,
    )
    assert stream == full


def test_after_generation_stop_token_branches():
    t = FakeTokenizer()
    delta = TemplateDelta(t)
    no_stop = t.encode("text")
    with_stop = t.encode("text" + IM_END)
    assert delta.after_generation(no_stop) == delta.glue_full
    assert delta.after_generation(with_stop) == delta.glue_full[1:]
