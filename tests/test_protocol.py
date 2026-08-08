from trace_opd.core import protocol
from trace_opd.core.types import (
    GUIDANCE_HEADER,
    guidance_hash,
    inject_guidance,
)


def test_parse_action_last_block_wins():
    text = "<action>go north</action> nah <action>Take Key from Box</action>"
    assert protocol.parse_action(text) == "take key from box"


def test_parse_action_none_and_empty():
    assert protocol.parse_action("no tags here") is None
    assert protocol.parse_action("<action>   </action>") is None
    assert protocol.parse_action("") is None


def test_parse_action_strips_quotes_and_lines():
    assert protocol.parse_action('<action>"look"\nextra</action>') == "look"


def test_parse_reasoning():
    text = "<reasoning>I should look around.</reasoning>\n<action>look</action>"
    assert protocol.parse_reasoning(text) == "I should look around."
    assert protocol.parse_reasoning("plain") is None


def test_alfworld_scaffold_guidance_injection():
    # scaffold 归 benchmark 自持(通用性整改);注入组合契约不变
    from trace_opd.envs.alfworld.adapter import SYSTEM_PROMPT_BARE

    assert GUIDANCE_HEADER not in SYSTEM_PROMPT_BARE
    privileged = inject_guidance(SYSTEM_PROMPT_BARE, "Always go to the shelf first.")
    assert privileged.startswith(SYSTEM_PROMPT_BARE)
    assert f"\n\n{GUIDANCE_HEADER}\nAlways go to the shelf first." in privileged


def test_parse_tool_calls_hermes_format():
    text = (
        "let me check\n"
        '<tool_call>\n{"name": "run_python", "arguments": {"code": "print(1)"}}\n</tool_call>'
    )
    calls = protocol.parse_tool_calls(text)
    assert calls == [{"name": "run_python", "arguments": {"code": "print(1)"}}]
    # 多调用保序;坏 JSON / 缺 name / arguments 非 dict 的块跳过或矫正
    messy = (
        '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'
        "<tool_call>not json</tool_call>"
        '<tool_call>{"arguments": {}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": "oops"}</tool_call>'
    )
    calls = protocol.parse_tool_calls(messy)
    assert [c["name"] for c in calls] == ["a", "b"]
    assert calls[1]["arguments"] == {}
    assert protocol.parse_tool_calls("no calls here") == []
    assert protocol.parse_tool_calls("") == []


def test_inject_guidance_empty_is_noop():
    assert inject_guidance("BASE", "") == "BASE"
    assert inject_guidance("BASE", "  \n ") == "BASE"


def test_guidance_hash_stability():
    assert guidance_hash("") == "bare"
    assert guidance_hash(" x ") == guidance_hash("x")
    assert guidance_hash("a") != guidance_hash("b")


def test_truncate_obs_marks_omission():
    obs = "a" * 5000
    out = protocol.truncate_obs(obs, 100)
    assert len(out) < 5000
    assert "truncated" in out
    assert protocol.truncate_obs("short") == "short"


def test_configure_episode_caps_installs_and_validates():
    orig = (protocol.MAX_ENV_STEPS, protocol.MAX_ASSISTANT_TURNS,
            protocol.GEN_MAX_TOKENS_PER_TURN)
    try:
        protocol.configure_episode_caps(50, 60, 4096)  # the 30/50-comparison path
        assert (protocol.MAX_ENV_STEPS, protocol.MAX_ASSISTANT_TURNS) == (50, 60)
        # 行为级断言(CD-006 P0 教训:只断言 cfg 读到值是语义层空测——
        # 必须断言注入后消费方读的模块常量真的变了)
        assert protocol.GEN_MAX_TOKENS_PER_TURN == 4096
        import pytest as _pytest

        with _pytest.raises(ValueError, match="episode caps"):
            protocol.configure_episode_caps(30, 30, 2048)  # turns must exceed steps
        with _pytest.raises(ValueError, match="episode caps"):
            protocol.configure_episode_caps(0, 10, 2048)
        with _pytest.raises(ValueError, match="max_tokens_per_turn"):
            protocol.configure_episode_caps(30, 40, 0)
    finally:
        protocol.configure_episode_caps(*orig)


def test_gt_extract_cap_is_decoupled_from_training_budget():
    # The expert walkthrough must never be truncated by the (smaller)
    # training budget — GT extraction keeps its own 50-step cap.
    assert protocol.GT_EXTRACT_MAX_STEPS == 50
    assert protocol.MAX_ENV_STEPS == 30  # user ruling default
