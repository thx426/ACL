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


def test_build_system_prompt_uses_guidance_header():
    bare = protocol.build_system_prompt(None)
    assert GUIDANCE_HEADER not in bare
    privileged = protocol.build_system_prompt("Always go to the shelf first.")
    assert privileged.startswith(bare)
    assert f"\n\n{GUIDANCE_HEADER}\nAlways go to the shelf first." in privileged


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
    orig = (protocol.MAX_ENV_STEPS, protocol.MAX_ASSISTANT_TURNS)
    try:
        protocol.configure_episode_caps(50, 60)  # the 30/50-comparison path
        assert (protocol.MAX_ENV_STEPS, protocol.MAX_ASSISTANT_TURNS) == (50, 60)
        import pytest as _pytest

        with _pytest.raises(ValueError, match="episode caps"):
            protocol.configure_episode_caps(30, 30)  # turns must exceed steps
        with _pytest.raises(ValueError, match="episode caps"):
            protocol.configure_episode_caps(0, 10)
    finally:
        protocol.configure_episode_caps(*orig)


def test_gt_extract_cap_is_decoupled_from_training_budget():
    # The expert walkthrough must never be truncated by the (smaller)
    # training budget — GT extraction keeps its own 50-step cap.
    assert protocol.GT_EXTRACT_MAX_STEPS == 50
    assert protocol.MAX_ENV_STEPS == 30  # user ruling default
