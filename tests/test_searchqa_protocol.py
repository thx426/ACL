"""SearchQA 文本协议的标签语义锁(2026-08-16 新建)。

该 benchmark 此前**零测试覆盖** —— 把 `<think>` 换成 protocol.REASONING_TAG
时全量测试仍全绿,等于协议改动无人守门。本文件补上最小但真实的锁:

  * scaffold / one-shot / thought 抽取三处必须用**同一个**运行时标签;
  * 标签切回 "think" 后 scaffold 必须与 Search-R1 / SEED 原生措辞逐字一致
    (一个断言同时锁住"除标签外没动别的"与"回退开关真能复现原文");
  * 动作解析与标签无关(<search>/<answer> 是任务自己的动作语法,不参与替换)。

不碰网络:只测纯函数,不构造 SearchQATaskAdapter(它要 httpx client 与检索
批量聚合器)。
"""
from __future__ import annotations

import pytest

from trace_opd.core import protocol as P
from trace_opd.envs.searchqa.adapter import (
    parse_text_action,
    system_prompt_text,
    text_thought,
)


@pytest.fixture(autouse=True)
def _restore_tag():
    """任何用例改了全局标签都必须还原 —— 它是模块级状态。"""
    old = P.REASONING_TAG
    yield
    P.configure_reasoning_tag(old)


def test_scaffold_uses_runtime_tag_and_drops_think():
    s = system_prompt_text()
    ro, rc = P.rtag_open(), P.rtag_close()
    assert ro in s and rc in s
    assert "<think>" not in s and "</think>" not in s
    # one-shot 示例必须与协议说明用同一个标签(否则 one-shot 反而在教错格式)
    assert s.count(ro) >= 2, "协议说明与 one-shot 示例都应出现开标签"
    # 任务自己的动作语法不参与替换
    for tag in ("<search>", "</search>", "<answer>", "</answer>",
                "<information>", "</information>"):
        assert tag in s


def test_scaffold_reverts_to_search_r1_wording_verbatim():
    """切回 think = Search-R1 / SEED 原生措辞(逐字)。"""
    P.configure_reasoning_tag("think")
    assert system_prompt_text() == (
        "You answer factual questions with the help of a Wikipedia search "
        "engine, reasoning step by step.\n\n"
        "Every turn you must first reason inside <think> and </think>. "
        "After reasoning, choose exactly one of the following actions "
        "(never both in one reply):\n"
        "(1) If you lack some knowledge, call the search engine as "
        "<search> your query </search>. It returns the top passages between "
        "<information> and </information>.\n"
        "(2) If you already know enough, give the final answer inside "
        "<answer> and </answer>, with no explanation — just the minimal "
        "span (entity / name / date / number). For example, "
        "<answer> Beijing </answer>.\n\n"
        "Worked example of one turn:\n"
        "<think>I need the birth year of the director, but I do not know who "
        "directed the film yet, so I will look that up first.</think>\n"
        "<search> director of the film Inception </search>\n\n"
        "Notes: multi-hop questions need several searches — resolve one entity "
        "first, then search the next fact using what you found. Passages are "
        "evidence, not truth. Answering ends the episode; there is no partial "
        "credit."
    )


def test_thought_extraction_follows_the_tag():
    ro, rc = P.rtag_open(), P.rtag_close()
    assert text_thought(f"{ro}find the director{rc}\n<search> q </search>") == (
        "find the director")
    # 旧标签在新协议下不再被当作推理块 —— 退回"动作标签之前的正文"
    assert text_thought("<think>stale</think>\n<search> q </search>") == (
        "<think>stale</think>")
    # 无推理块:取动作之前的正文
    assert text_thought("just reasoning\n<answer> Beijing </answer>") == "just reasoning"

    P.configure_reasoning_tag("think")
    assert text_thought("<think>stale</think>\n<search> q </search>") == "stale"


def test_action_parsing_is_tag_independent():
    """动作语法不随推理标签变(它是任务的,不是协议面的)。"""
    for tag in ("reasoning", "think", "thought"):
        P.configure_reasoning_tag(tag)
        ro, rc = P.rtag_open(), P.rtag_close()
        assert parse_text_action(f"{ro}r{rc}<search> who directed it </search>") == {
            "name": "search", "arguments": {"query": "who directed it"}}
        assert parse_text_action(f"{ro}r{rc}<answer> Beijing </answer>") == {
            "name": "submit", "arguments": {"answer": "Beijing"}}
        # answer 优先于 search(终局动作;Search-R1 同序)
        assert parse_text_action(
            "<search> q </search><answer> A </answer>")["name"] == "submit"
        assert parse_text_action(f"{ro}only reasoning{rc}") is None


def test_fingerprint_tracks_the_tag_per_env(monkeypatch, tmp_path):
    """指纹按 env 的**历史**标签条件追加:
      searchqa 历史 = think ⇒ 换到 reasoning 必须改指纹(新协议 = 新曲线);
      alfworld 历史 = reasoning ⇒ 保持默认时指纹不因新键而变(既有 run 可续跑)。
    """
    from dataclasses import replace
    from pathlib import Path

    from trace_opd.common.config import load_config, objective_fingerprint

    for k, v in {"TRACE_MODEL_PATH": str(tmp_path / "m"),
                 "TRACE_RESEARCHER_API_KEY": "dummy",
                 "TRACE_35B_KEY": "dummy"}.items():
        monkeypatch.setenv(k, v)
    root = Path(__file__).resolve().parents[1]
    sqa = load_config(root / "configs/classic/searchqa_7b_pgs.yaml")
    alf = load_config(root / "configs/classic/alfworld_7b_pgs.yaml")

    def with_tag(cfg, tag):
        return objective_fingerprint(
            replace(cfg, episode=replace(cfg.episode, reasoning_tag=tag)))

    assert sqa.environment.name == "searchqa"
    assert with_tag(sqa, "think") != with_tag(sqa, "reasoning")
    assert alf.environment.name == "alfworld"
    assert with_tag(alf, "reasoning") != with_tag(alf, "think")
    # ★ 既有 run 保护:alfworld 在默认标签下,指纹与"没有这个键"的历史材料
    #   一致 —— 条件追加保证了这一点(reasoning == alfworld 的历史值)。
    assert alf.episode.reasoning_tag == "reasoning"
