"""v4 research_rl 配置契约:依赖校验、参数域、指纹条件追加。"""
from __future__ import annotations

import pytest

from trace_opd.common.config import load_config, objective_fingerprint

YAML = """
run: {name: t, runs_root: runs}
alfworld: {data_root: /d, worker_python: /p}
train: {model_path: /m, train_file: /t.parquet, val_files: [/v1.parquet], agent_task_timeout_s: 0}
eval: {}
research: {enabled: true, backend: engine, tool_protocol: text, gt: "off"}
narration: {backend: engine}
roles:
  researcher:
    endpoint: "http://x/v1"
    model: m
    api_key_env: TEST_RRL_KEY
    mode: external_pool
    max_concurrency: 8
    extra_body: {chat_template_kwargs: {enable_thinking: false}}
steering: {enabled: true, doc_source: researcher, theta_d: 0.8%s}
%s
"""


def _load(tmp_path, monkeypatch, steer_extra="", rrl_extra=""):
    monkeypatch.setenv("TEST_RRL_KEY", "k")
    path = tmp_path / f"c{abs(hash((steer_extra, rrl_extra)))}.yaml"
    path.write_text(YAML % (steer_extra, rrl_extra))
    return load_config(path)


_RRL = "research_rl: {enabled: true%s}"


def test_research_rl_defaults_load_and_require_researcher_stack(tmp_path, monkeypatch):
    cfg = _load(tmp_path, monkeypatch, rrl_extra=_RRL % "")
    rrl = cfg.research_rl
    assert rrl.research_n == 8 and rrl.continuation_samples == 4
    assert rrl.prefix_quantiles == [0.05, 0.15, 0.30, 0.50, 0.75]
    assert rrl.researcher_temperature == 1.0 and rrl.continuation_temperature == 1.0
    # 用户裁定:默认对齐生产配置。2026-08-07 由 16/4/8 降到 8/2/4 以砍 episode 开销
    # (162 v4 实测 128 条 research 会话烧掉 3054 个 ALFWorld episode,
    #  而 timing_s/gen 占整步墙钟 84.5%)。k_max 必须一起降 ——
    #  tools.py 用 min(k_req, k_max),留着 8 则模型填 8 即绕过 k_default。
    assert cfg.research.n_exp == 8
    assert cfg.research.k_default == 2
    assert cfg.research.k_max == 4
    # 依赖:必须 steering+doc_source=researcher+research.enabled
    with pytest.raises(ValueError, match="research_rl.enabled"):
        _load(tmp_path, monkeypatch,
              steer_extra=", doc_source: static, doc_table: /tmp/d.jsonl",
              rrl_extra=_RRL % "")


def test_research_rl_parameter_domains(tmp_path, monkeypatch):
    for bad, match in [
        (", research_n: 1", "research_n"),
        (", prefix_quantiles: [0.5, 0.3]", "prefix_quantiles"),
        (", prefix_quantiles: [0.0, 0.5]", "prefix_quantiles"),
        (", continuation_samples: 0", "continuation_samples"),
        (", researcher_temperature: 0.0", "researcher_temperature"),
        (", research_rl_weight: -0.1", "research_rl_weight"),
        (", research_warmup_steps: -1", "research_warmup"),
    ]:
        with pytest.raises(ValueError, match=match):
            _load(tmp_path, monkeypatch, rrl_extra=_RRL % bad)


def test_research_rl_fingerprint_conditional(tmp_path, monkeypatch):
    # engine+text 仅在 research_rl.enabled 时合法(校验分叉),条件追加的
    # 对照因此各用各的合法 YAML:external 基线上验证「显式 disabled 与
    # 不写 research_rl 段指纹逐字节相同」;enabled 档验证参数进指纹。
    ext = YAML.replace("backend: engine, tool_protocol: text",
                       "backend: external, tool_protocol: fncall")

    def _load_ext(rrl_extra=""):
        monkeypatch.setenv("TEST_RRL_KEY", "k")
        path = tmp_path / f"e{abs(hash(rrl_extra))}.yaml"
        path.write_text(ext % ("", rrl_extra))
        return load_config(path)

    base = objective_fingerprint(_load_ext())
    disabled = objective_fingerprint(_load_ext("research_rl: {enabled: false}"))
    assert disabled == base                  # 关断 = 历史指纹逐字节不变
    on = objective_fingerprint(_load(tmp_path, monkeypatch, rrl_extra=_RRL % ""))
    assert on != base                        # 开启 = 双目标,换指纹
    tweaked = objective_fingerprint(
        _load(tmp_path, monkeypatch, rrl_extra=_RRL % ", continuation_samples: 2")
    )
    assert tweaked not in (base, on)         # reward 定义参数进指纹
