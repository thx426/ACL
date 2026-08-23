"""Math 环境:verifier 等价语义锁、FC react、端到端与信息红线。

verifier 走真 math-verify(本地已装,训练环境依赖在 GO 清单);episode
端到端跑真 subprocess(run_python)——TIR 形态在机制上成立的机器证明。
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from fakes import FakeTokenizer, ScriptedGenerate
from trace_opd.common.config import MathCfg, load_config, objective_fingerprint
from trace_opd.core.episode import run_episode
from trace_opd.core.template_delta import TemplateDelta
from trace_opd.envs import registry
from trace_opd.envs.math.adapter import (
    _format_reminder,
    SYSTEM_PROMPT,
    TOOLS,
    MathEnvFactory,
    MathTaskAdapter,
    initial_obs_of,
)
from trace_opd.envs.math.build_dataset import _extract_answer, strip_dapo_template
from trace_opd.envs.math.verifier import answers_equivalent

TOK = FakeTokenizer()
DELTA = TemplateDelta(TOK)

ITEM = {
    "task_id": "dapo/test1",
    "task_type": "math",
    "split": "train",
    "problem": "Compute 4 * 13.",
    "ground_truth": "98765",   # 特意与题面无关的哨兵值(信息红线断言用)
}
ITEM["initial_obs"] = initial_obs_of(ITEM)


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def _adapter(item=None) -> MathTaskAdapter:
    return MathTaskAdapter(item or ITEM, MathCfg(), threading.Semaphore(4))


def _call(name: str, **arguments) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"reasoning here\n<tool_call>\n{payload}\n</tool_call>"


def _episode(item, texts):
    return _run(run_episode(
        adapter=_adapter(item),
        generate=ScriptedGenerate(TOK, texts),
        tokenizer=TOK,
        delta=DELTA,
        guidance="",
        episode_id="ep-math",
        response_budget=200000,
        sampling_params={"temperature": 0.5},
    ))


# ------------------------------------------------------------ verifier 语义锁

def test_answers_equivalent_semantics():
    assert answers_equivalent("204", "204")
    assert answers_equivalent("204", "\\boxed{204}")
    assert answers_equivalent("\\frac{1}{2}", "0.5")
    assert answers_equivalent("1/2", "\\frac{1}{2}")
    # latex 元组(实测坑:裸 parse 丢元素,封装用 $ 包裹组合)
    assert answers_equivalent("\\left( 3, \\frac{\\pi}{2} \\right)", "(3, \\pi/2)")
    assert not answers_equivalent("204", "205")
    assert not answers_equivalent("1/2", "1/3")
    assert not answers_equivalent("", "204")
    assert not answers_equivalent("204", "")


# ------------------------------------------------------------ 数据管线单元

def test_strip_dapo_template_anchors():
    content = (
        "Solve the following math problem step by step. The last line of "
        "your response should be of the form Answer: $Answer (without "
        "quotes) where $Answer is the answer to the problem.\n\n"
        "What is 2+2?\n\n"
        'Remember to put your answer on its own line after "Answer:".'
    )
    assert strip_dapo_template(content) == "What is 2+2?"
    with pytest.raises(SystemExit, match="anchors did not match"):
        strip_dapo_template("Some drifted prompt\nWhat is 2+2?")


def test_extract_answer_per_source():
    assert _extract_answer("aime24", {"solution": "\\boxed{204}"}) == "204"
    assert _extract_answer("aime26", {"answer": 277}) == "277"
    assert _extract_answer("amc23", {"answer": "27"}) == "27"
    assert _extract_answer("olympiad", {"final_answer": ["2"]}) == "2"
    assert _extract_answer("olympiad", {"final_answer": ["2", "3"]}) is None  # 多答案跳过
    assert _extract_answer("aime24", {"solution": "no box here"}) is None


# ------------------------------------------------------------ episode 端到端

def test_episode_tool_then_submit_wins():
    result = _episode(
        {**ITEM, "ground_truth": "52"},
        [_call("run_python", code="print(4 * 13)"), _call("submit", answer="52")],
    )
    traj = result.trajectory
    assert traj.won and traj.score == 1.0
    assert "52" in traj.steps[0].observation          # 工具真执行
    assert traj.steps[0].observation.startswith("<tool_response>")
    assert traj.steps[1].observation.startswith("<tool_response>\nSubmitted.")


def test_episode_direct_submit_and_wrong_answer():
    win = _episode({**ITEM, "ground_truth": "52"}, [_call("submit", answer="52")])
    assert win.trajectory.won and win.trajectory.step_count == 1
    lose = _episode({**ITEM, "ground_truth": "52"}, [_call("submit", answer="53")])
    assert lose.trajectory.won is False and lose.trajectory.score == 0.0
    # 信息红线:错误提交的反馈与胜利提交逐字节同形(零评分信息)
    assert lose.trajectory.steps[-1].observation == win.trajectory.steps[-1].observation


def test_episode_format_failure_and_empty_submit():
    result = _episode({**ITEM, "ground_truth": "52"}, [
        "just thinking, no call",
        _call("submit", answer=""),
        _call("submit", answer="52"),
    ])
    traj = result.trajectory
    assert traj.steps[0].acted is False
    assert traj.steps[0].observation == _format_reminder()
    assert traj.steps[1].acted and not traj.steps[1].valid
    assert "empty answer" in traj.steps[1].observation
    assert traj.won


def test_ground_truth_never_leaks():
    """哨兵 gt=98765 不得出现在 scaffold/brief/tools/任何反馈里。"""
    adapter = _adapter()
    assert "98765" not in adapter.executor_scaffold()
    assert "98765" not in adapter.task_brief()
    assert "98765" not in json.dumps(TOOLS)
    assert adapter.gt_text() is None
    result = _episode(ITEM, [_call("submit", answer="1")])
    assert all("98765" not in s.observation for s in result.trajectory.steps)


def test_replay_action_deterministic():
    adapter = _adapter()
    env = adapter.open_episode()
    try:
        norm = json.dumps(
            {"name": "run_python", "arguments": {"code": "print(sorted({'b','a'}))"}},
            sort_keys=True,
        )
        fb1, e1 = adapter.replay_action(env, norm)
        fb2, e2 = adapter.replay_action(env, norm)
        assert fb1 == fb2 and not e1 and not e2       # PYTHONHASHSEED=0 确定性
    finally:
        env.close()
    assert not Path(env.workdir).exists()


# ------------------------------------------------------------- 装配与配置

def test_registry_dispatch_math():
    cfg = SimpleNamespace(
        environment=SimpleNamespace(name="math"),
        math=MathCfg(),
    )
    assert isinstance(registry.make_env_factory(cfg), MathEnvFactory)
    # scaffold 零手写工具块(模板渲染)
    assert "<tools>" not in SYSTEM_PROMPT and "<tool_call>" not in SYSTEM_PROMPT
    assert [t["function"]["name"] for t in TOOLS] == ["run_python", "submit"]
    assert TOOLS[1]["function"]["parameters"]["required"] == ["answer"]


@pytest.fixture
def deploy_env(monkeypatch, tmp_path):
    for k, v in {
        "TRACE_MODEL_PATH": str(tmp_path / "model"),
        "MATH_TRAIN_FILE": str(tmp_path / "t.parquet"),
        "MATH_VAL_FILE": str(tmp_path / "v.parquet"),
        "TRACE_RESEARCHER_API_KEY": "dummy-for-validation",
        # configs/classic/* 的 researcher 走 35B 外部池,api_key_env 是这个
        "TRACE_35B_KEY": "dummy-for-validation",
    }.items():
        monkeypatch.setenv(k, v)


def test_math_yaml_pair_loads_and_fingerprints(deploy_env):
    root = Path(__file__).resolve().parents[1]
    selfres = load_config(root / "configs/experiments/a800-math-s1-selfres.yaml")
    extres = load_config(root / "configs/experiments/a800-math-s1-extres.yaml")
    assert selfres.environment.name == "math"
    assert selfres.episode.max_env_steps == 12
    assert selfres.episode.max_tokens_per_turn == 8192
    assert selfres.research_rl.enabled and not extres.research_rl.enabled
    assert objective_fingerprint(selfres) != objective_fingerprint(extres)


# ------------------------------------------------- MENTOR 单轮档(protocol=single)
# 对齐对象:arXiv 2510.04140 / ICLR 2026,代码 commit b1cc4c6。
# 这一组是**语义锁**:格式判定与 reward 结构必须与 MENTOR 的
# examples/reward_function/math.py 逐项同构,漂了就该红。

SINGLE_CFG = MathCfg(protocol="single", format_weight=0.1)


def _single_adapter(item=None, cfg=None) -> MathTaskAdapter:
    return MathTaskAdapter(item or ITEM, cfg or SINGLE_CFG, threading.Semaphore(4))


def _reply(think: str, tail: str) -> str:
    """按**运行时**推理标签构造回复(2026-08-16 起默认 <reasoning>)。"""
    from trace_opd.core import protocol as P
    return f"{P.rtag_open()}{think}{P.rtag_close()}{tail}"


def test_mentor_format_ok_matches_reference_regex():
    """MENTOR format_reward 的语义锁(其 re.fullmatch + DOTALL + 空白预处理)。"""
    from trace_opd.envs.math.adapter import mentor_format_ok

    assert mentor_format_ok(_reply("reasoning", " The answer is \\boxed{42}."))
    assert mentor_format_ok(_reply("multi\nline\nreasoning", "\\boxed{\\frac{1}{2}}"))
    # 空白预处理:`< reasoning >` 这类带空格的标签也过(MENTOR re.sub 副作用)
    from trace_opd.core import protocol as P
    tag = P.REASONING_TAG
    assert mentor_format_ok(f"< {tag} >r< /{tag} >x\\boxed{{1}}")
    # fullmatch ⇒ 必须以 <think> 起头
    assert not mentor_format_ok("Sure! " + _reply("r", "\\boxed{1}"))
    assert not mentor_format_ok("\\boxed{42}")                    # 无推理块
    # 旧的 <think> 在新标签下不再被认作合法格式(协议确实换了)
    assert not mentor_format_ok("<think>r</think>\\boxed{1}")
    assert not mentor_format_ok(_reply("r", "the answer is 42"))   # 无 boxed
    assert not mentor_format_ok(_reply("r", ""))
    assert not mentor_format_ok("")


def test_single_react_score_is_mentor_weighted():
    """score = 0.9·acc + 0.1·format,四个取值与 (acc, fmt) 一一对应。"""
    adapter = _single_adapter({**ITEM, "ground_truth": "52"})
    env = adapter.open_episode()
    try:
        both = adapter.react(env, _reply("compute", " so \\boxed{52}"))
        assert both.score == pytest.approx(1.0) and both.valid
        # 答对但格式错(无 think 块)⇒ 0.9 —— 二值通道会误判成 0,正是
        # 需要 train.reward_from_score 的原因
        acc_only = adapter.react(env, "The answer is \\boxed{52}")
        assert acc_only.score == pytest.approx(0.9) and not acc_only.valid
        # 格式对但答错 ⇒ 0.1
        fmt_only = adapter.react(env, _reply("compute", " so \\boxed{53}"))
        assert fmt_only.score == pytest.approx(0.1) and fmt_only.valid
        both_bad = adapter.react(env, "no answer at all")
        assert both_bad.score == pytest.approx(0.0) and not both_bad.valid
        # 单轮不变量:恒终局、恒消耗环境步、永不走多轮的格式重试路径
        for out in (both, acc_only, fmt_only, both_bad):
            assert out.done and out.acted and out.consumed_env_step
            assert out.feedback == ""
    finally:
        env.close()


def test_single_format_weight_zero_is_binary():
    adapter = _single_adapter({**ITEM, "ground_truth": "52"},
                              MathCfg(protocol="single", format_weight=0.0))
    env = adapter.open_episode()
    try:
        assert adapter.react(env, "\\boxed{52}").score == pytest.approx(1.0)
        assert adapter.react(env, _reply("r", "\\boxed{53}")).score == pytest.approx(0.0)
    finally:
        env.close()


def test_single_answer_extraction_beats_mentor_regex():
    """嵌套花括号:我们的平衡扫描能取全,MENTOR 的 `\\boxed\\{.*\\}` 靠贪婪
    也能取全,但两者对未闭合的处理不同 —— 未闭合按无终答(不冒判)。"""
    adapter = _single_adapter({**ITEM, "ground_truth": "\\frac{1}{2}"})
    env = adapter.open_episode()
    try:
        ok = adapter.react(env, _reply("r", " \\boxed{\\frac{1}{2}}"))
        assert ok.score == pytest.approx(1.0)
        broken = adapter.react(env, _reply("r", " \\boxed{\\frac{1}{2}"))
        assert broken.action_raw == ""            # 未闭合 ⇒ 无终答
        assert broken.score == pytest.approx(0.1)  # 只剩格式分…
    finally:
        env.close()


def test_single_surface_has_no_tools():
    """单轮档的任务表面:MENTOR system prompt 逐字、零工具、brief 不提工具。"""
    from trace_opd.core import protocol as P
    from trace_opd.envs.math.adapter import system_prompt_single

    adapter = _single_adapter()
    assert adapter.executor_scaffold() == system_prompt_single()
    assert adapter.tools() is None
    # 默认标签下 scaffold 用 <reasoning>,不含 <think>
    assert f"<{P.REASONING_TAG}>" in system_prompt_single()
    assert "<think>" not in system_prompt_single()
    # ★ 把标签切回 think 后必须与 MENTOR verl/utils/dataset.py:174-176 的
    #   system content **逐字一致** —— 一个断言同时锁住两件事:除标签外
    #   句式没动;回退开关真的能复现原文。
    try:
        P.configure_reasoning_tag("think")
        assert system_prompt_single() == (
            "You are a helpful AI Assistant that provides well-reasoned and "
            "detailed responses. You FIRST think about the reasoning process as "
            "an internal monologue and then provide the final answer. The "
            "reasoning process MUST BE enclosed within <think> </think> tags. "
            "The final answer MUST BE put in \\boxed{}."
        )
    finally:
        P.configure_reasoning_tag("reasoning")
    brief = adapter.task_brief()
    for banned in ("run_python", "submit(", "```python", "code block"):
        assert banned not in brief, f"单轮 brief 不该提 {banned!r}(执行器没有工具)"
    assert "SINGLE reply" in brief
    assert "98765" not in brief and "98765" not in adapter.executor_scaffold()


def test_single_episode_end_to_end_is_one_turn():
    """端到端:caps 1/1 下恰好一轮、非截断、outcome 正常。"""
    from trace_opd.core import protocol as P

    old = (P.MAX_ENV_STEPS, P.MAX_ASSISTANT_TURNS, P.GEN_MAX_TOKENS_PER_TURN)
    P.configure_episode_caps(1, 1, 8192)
    try:
        result = _run(run_episode(
            adapter=_single_adapter({**ITEM, "ground_truth": "52"}),
            generate=ScriptedGenerate(TOK, [_reply("compute 4*13", " \\boxed{52}")]),
            tokenizer=TOK, delta=DELTA, guidance="", episode_id="ep-single",
            response_budget=8704, sampling_params={"temperature": 0.5},
        ))
        traj = result.trajectory
        assert result.assistant_turns == 1 and traj.step_count == 1
        assert traj.won and traj.score == pytest.approx(1.0)
        assert not result.truncated and result.invalid_format == 0
        # 答错但格式对:won=False 而 score=0.1 —— 二值 won 会丢掉格式分
        lose = _run(run_episode(
            adapter=_single_adapter({**ITEM, "ground_truth": "52"}),
            generate=ScriptedGenerate(TOK, [_reply("oops", " \\boxed{53}")]),
            tokenizer=TOK, delta=DELTA, guidance="", episode_id="ep-single2",
            response_budget=8704, sampling_params={"temperature": 0.5},
        ))
        assert lose.trajectory.won is False
        assert lose.trajectory.score == pytest.approx(0.1)
    finally:
        P.configure_episode_caps(*old)


def test_single_config_guards(deploy_env):
    """两道 fail-loud:caps 必须 1/1;format_weight>0 必须配 reward_from_score。"""
    from dataclasses import replace

    from trace_opd.common.config import _validate

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/classic/math_single_7b_pgs.yaml")
    assert cfg.math.protocol == "single" and cfg.math.format_weight == 0.1
    assert cfg.train.reward_from_score is True
    assert (cfg.episode.max_env_steps, cfg.episode.max_assistant_turns) == (1, 1)
    assert cfg.episode.max_tokens_per_turn == 8192

    bad_caps = replace(cfg, episode=replace(cfg.episode, max_env_steps=6,
                                            max_assistant_turns=8))
    with pytest.raises(ValueError, match="必须都为 1"):
        _validate(bad_caps)
    bad_reward = replace(cfg, train=replace(cfg.train, reward_from_score=False))
    with pytest.raises(ValueError, match="reward_from_score"):
        _validate(bad_reward)


def test_single_fingerprint_separates_protocols(deploy_env):
    """换协议 = 换优化目标:三档指纹两两不同,且 format_weight 也进指纹。"""
    from dataclasses import replace

    root = Path(__file__).resolve().parents[1]
    single = load_config(root / "configs/classic/math_single_7b_pgs.yaml")
    tir = load_config(root / "configs/classic/math_7b_pgs.yaml")
    assert objective_fingerprint(single) != objective_fingerprint(tir)

    w0 = replace(single, math=replace(single.math, format_weight=0.0))
    assert objective_fingerprint(w0) != objective_fingerprint(single)
    # reward 取数口径同样进指纹
    binary = replace(w0, train=replace(w0.train, reward_from_score=False))
    assert objective_fingerprint(binary) != objective_fingerprint(w0)
    # 条件追加的兜底:TIR 档(reward_from_score=False)的指纹不因新键而改变
    assert tir.train.reward_from_score is False
