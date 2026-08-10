"""§7.5 预算解耦(用户裁定 2026-08-08):必配三键与保险丝校验链。

上限(rollout_max_model_len,保险丝)与控制器(session_token_budget,流
预算)是两个独立的量;budget 允许 < max_turns×per-turn(单轮上限是峰值
弹性非典型量);超 budget 截断轨迹(length 优雅终局)。
"""
import pytest

from trace_opd.common.config import load_config

_BASE = """
run: {name: t, runs_root: runs}
alfworld: {data_root: /d, worker_python: /p}
train: {model_path: /m, train_file: /t.parquet, val_files: [/v.parquet],
        agent_task_timeout_s: 0%s}
eval: {}
research: {enabled: true, backend: engine, tool_protocol: text, gt: "off"%s}
narration: {backend: engine%s}
roles:
  researcher:
    endpoint: "http://x/v1"
    model: m
    api_key_env: TEST_BUDGET_KEY
    mode: external_pool
    max_concurrency: 8
    extra_body: {chat_template_kwargs: {enable_thinking: false}}
steering: {enabled: true, doc_source: researcher, theta_d: 0.8}
research_rl: {enabled: true, research_n: 8}
"""


def _load(tmp_path, monkeypatch, train="", research="", narration="", extra=""):
    monkeypatch.setenv("TEST_BUDGET_KEY", "k")
    p = tmp_path / "b.yaml"
    # extra:追加顶层段;同名顶层键按 yaml 语义整体覆盖 _BASE 里的段
    # (steering 档位测试等借此换段,test_engine_executor.py)。
    p.write_text(_BASE % (train, research, narration) + extra)
    return load_config(str(p))


GOOD = dict(research=", session_token_budget: 24576, max_prompt_tokens: 1024",
            narration=", max_prompt_tokens: 6144")


def test_mandatory_keys_fail_loud(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="session_token_budget 必配"):
        _load(tmp_path, monkeypatch, narration=GOOD["narration"])
    with pytest.raises(ValueError, match="max_prompt_tokens 必配"):
        _load(tmp_path, monkeypatch,
              research=", session_token_budget: 24576",
              narration=GOOD["narration"])
    with pytest.raises(ValueError, match="narration.max_prompt_tokens 必配"):
        _load(tmp_path, monkeypatch, research=GOOD["research"])


def test_good_config_loads_and_budget_may_undershoot_turn_product(tmp_path, monkeypatch):
    cfg = _load(tmp_path, monkeypatch, **GOOD)
    # budget(24576)远小于 max_turns×researcher_max_tokens(30×8192=245760)
    # ——用户裁定:单轮上限是峰值弹性,budget < 乘积是常态设计,必须合法。
    assert cfg.research.session_token_budget < (
        cfg.research.max_turns * cfg.research.researcher_max_tokens
    )


def test_fuse_inequalities(tmp_path, monkeypatch):
    # budget 超 response_length(padding 维度只补不截)
    with pytest.raises(ValueError, match="response_length"):
        _load(tmp_path, monkeypatch,
              research=", session_token_budget: 40000, max_prompt_tokens: 512",
              narration=GOOD["narration"])
    # prompt 超 train.max_prompt_length(prompt padding 维度)
    with pytest.raises(ValueError, match="max_prompt_length"):
        _load(tmp_path, monkeypatch,
              research=", session_token_budget: 24576, max_prompt_tokens: 2048",
              narration=GOOD["narration"])
    # budget+prompt 超模型上下文保险丝
    with pytest.raises(ValueError, match="rollout_max_model_len"):
        _load(tmp_path, monkeypatch,
              train=", response_length: 60000, max_prompt_length: 8192,"
                    " rollout_max_model_len: 40960",
              research=", session_token_budget: 40000, max_prompt_tokens: 4096",
              narration=GOOD["narration"])


def test_executor_budget_decoupled_from_padding_dimension(tmp_path, monkeypatch):
    """§7.6(CD-006 复现的反向验证):executor 16K + researcher 48K 必须可
    表达——response_length 作维度配 49152,executor 流预算独立 16384。"""
    cfg = _load(
        tmp_path, monkeypatch,
        train=", response_length: 49152, rollout_max_model_len: 65536",
        research=", session_token_budget: 49152, max_prompt_tokens: 1024",
        narration=GOOD["narration"],
    )
    # episode 键缺省 0 = 跟随维度
    assert cfg.episode.session_token_budget == 0

    p = tmp_path / "b2.yaml"
    p.write_text((_BASE % (", response_length: 49152, rollout_max_model_len: 65536",
                           ", session_token_budget: 49152, max_prompt_tokens: 1024",
                           GOOD["narration"]))
                 + "episode: {session_token_budget: 16384}\n")
    cfg2 = load_config(str(p))
    assert cfg2.episode.session_token_budget == 16384

    # executor 预算超维度 → FAIL(与 research 侧同款不变式)
    p3 = tmp_path / "b3.yaml"
    p3.write_text((_BASE % ("", GOOD["research"], GOOD["narration"]))
                  + "episode: {session_token_budget: 40000}\n")
    with pytest.raises(ValueError, match="episode.session_token_budget"):
        load_config(str(p3))
