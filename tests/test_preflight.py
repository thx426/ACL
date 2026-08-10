"""统一预检入口(envs/preflight)+ 审阅改进项的单元锁。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from trace_opd.core import protocol
from trace_opd.envs.preflight import main as preflight_main


@pytest.fixture(autouse=True)
def _restore_episode_caps():
    """preflight 真跑会 configure_episode_caps(模块全局)——测试后恢复,
    否则污染后续测试(本文件首次引入时的实锤事故)。"""
    saved = (protocol.MAX_ENV_STEPS, protocol.MAX_ASSISTANT_TURNS)
    yield
    protocol.MAX_ENV_STEPS, protocol.MAX_ASSISTANT_TURNS = saved


@pytest.fixture
def math_env(monkeypatch, tmp_path):
    for k, v in {
        "TRACE_MODEL_PATH": str(tmp_path / "model"),
        "MATH_TRAIN_FILE": str(tmp_path / "t.parquet"),
        "MATH_VAL_FILE": str(tmp_path / "v.parquet"),
        "TRACE_RESEARCHER_API_KEY": "dummy-for-validation",
    }.items():
        monkeypatch.setenv(k, v)


def test_preflight_full_probe_on_math(math_env, monkeypatch, tmp_path):
    """三级预检在 math 上端到端真跑(config→factory→live episode probe,
    含 initial_obs 比对)。"""
    from trace_opd.envs.math.adapter import initial_obs_of

    item = {
        "task_id": "dapo/pf1", "task_type": "math", "split": "valid",
        "problem": "Compute 1+1.", "ground_truth": "2",
    }
    item["initial_obs"] = initial_obs_of(item)
    pq = tmp_path / "probe.parquet"
    pd.DataFrame([{"extra_info": item}]).to_parquet(pq)
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(sys, "argv", [
        "preflight",
        "--config", str(root / "configs/experiments/a800-math-s1-selfres.yaml"),
        "--parquet", str(pq),
    ])
    assert preflight_main() == 0


def test_preflight_fails_loud_on_bad_factory(math_env, monkeypatch, tmp_path):
    """factory 级失败(SSB 无 data_root)非零退出且不抛未捕获异常。"""
    yaml = tmp_path / "bad.yaml"
    src = (Path(__file__).resolve().parents[1] /
           "experiments/ssb-9b-selfres-a800/config.yaml").read_text()
    yaml.write_text(src)
    monkeypatch.setenv("SSB_TRAIN_FILE", str(tmp_path / "t.parquet"))
    monkeypatch.setenv("SSB_VAL_FILE", str(tmp_path / "v.parquet"))
    monkeypatch.setenv("SSB_DATA", "")          # data_root 缺失 ⇒ factory 死
    monkeypatch.setattr(sys, "argv", ["preflight", "--config", str(yaml)])
    assert preflight_main() != 0
