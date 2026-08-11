"""SWE 域交互环境:工具协议 / git 封禁 / 动作映射 / 判分语义 / 数据契约。

对照锚点(全部 experiments/bench_env/codebases/ 实读):
  - 工具 schema = R2E openhands 三件套;
  - 判分 = R2E _calculate_reward_r2e(二值,parse_log_pytest 已与官方
    __main__ 样本逐字节对拍,见 judge 移植注释);
  - git 封禁 = 比 R2E 首词/SWE-Master 子串更严的命令位置精确拦。
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from trace_opd.envs.swe import tools
from trace_opd.envs.swe.judge import parse_log_pytest, score_from_logs
from trace_opd.envs.swe.tools import bash_is_blocked, editor_to_command, path_is_blocked


# ----------------------------------------------------- 工具 schema 契约

def test_tools_schema():
    names = [t["function"]["name"] for t in tools.TOOLS]
    assert names == ["str_replace_editor", "execute_bash", "submit"]
    # execute_bash required 补了 command(上游为空)
    eb = next(t for t in tools.TOOLS if t["function"]["name"] == "execute_bash")
    assert eb["function"]["parameters"]["required"] == ["command"]
    sre = next(t for t in tools.TOOLS if t["function"]["name"] == "str_replace_editor")
    assert sre["function"]["parameters"]["required"] == ["command", "path"]
    assert sre["function"]["parameters"]["properties"]["command"]["enum"] == \
        ["view", "create", "str_replace", "insert"]


# ----------------------------------------------------- git 封禁(纵深第一层)

@pytest.mark.parametrize("cmd", [
    "git log",
    "cd /testbed && git log",
    "git --no-pager show HEAD",
    "cat .git/ORIG_HEAD",
    "g=git; $g log",                 # 变量别名
    "GIT=/usr/bin/git; $GIT show",
    "echo $(git rev-parse HEAD)",    # 子命令替换
    "x=`git log`",                   # 反引号
    "true; git diff",                # 分号后命令位置
    "ls | git apply",                # 管道后命令位置
    "ipython",                       # 其他黑名单
])
def test_bash_blocked(cmd):
    assert bash_is_blocked(cmd) is not None


@pytest.mark.parametrize("cmd", [
    'python -c "import os"',
    "grep -rn foo .",
    "ls -la",
    "grep git README.md",            # git 作参数,放行
    "cat gitignore.txt",             # 含 git 子串但非 .git/,放行
    "python -m pytest r2e_tests",
    'find . -name "*.py"',
    "A=1 python x.py",               # 前导赋值 + 正常命令
])
def test_bash_allowed(cmd):
    assert bash_is_blocked(cmd) is None


def test_path_blocked():
    assert path_is_blocked("/testbed/.git/config") is not None
    assert path_is_blocked("/testbed/pkg/mod.py") is None


# ----------------------------------------------------- 动作 → 命令映射

def test_editor_to_command_view_file():
    cmd = editor_to_command({"command": "view", "path": "/testbed/a.py"})
    assert "cat -n" in cmd and "/testbed/a.py" in cmd


def test_editor_to_command_view_range():
    cmd = editor_to_command({"command": "view", "path": "/testbed/a.py",
                             "view_range": [10, 20]})
    assert "sed -n" in cmd and "10,20p" in cmd


def test_editor_to_command_create_heredoc():
    cmd = editor_to_command({"command": "create", "path": "/testbed/new.py",
                             "file_text": "print(1)\n"})
    assert "__TRACE_EOF__" in cmd and "print(1)" in cmd and "mkdir -p" in cmd


def test_editor_to_command_str_replace_is_python():
    cmd = editor_to_command({"command": "str_replace", "path": "/testbed/a.py",
                             "old_str": "x = 1", "new_str": "x = 2"})
    # base64 传参,不含裸引号注入;是 python3 -c
    assert cmd.startswith("python3 -c ")
    assert "x = 1" not in cmd  # 已 base64,原文不裸露在命令行


# ----------------------------------------------------- 判分语义(R2E 二值)

def _log(passed, failed=()):
    lines = ["=" * 10 + " short test summary info " + "=" * 10]
    for t in passed:
        lines.append(f"PASSED r2e_tests/test_1.py::{t}")
    for t in failed:
        lines.append(f"FAILED r2e_tests/test_1.py::{t} - AssertionError")
    return "\n".join(lines)


def test_score_full_match():
    expected = json.dumps({"test_a": "PASSED", "test_b": "PASSED"})
    log = _log(["test_a", "test_b"])
    assert score_from_logs(log, expected) == 1.0


def test_score_any_mismatch_zero():
    expected = json.dumps({"test_a": "PASSED", "test_b": "PASSED"})
    # test_b 实际 FAILED ⇒ 0.0(二值,无部分分)
    log = _log(["test_a"], ["test_b"])
    assert score_from_logs(log, expected) == 0.0


def test_score_count_mismatch_zero():
    expected = json.dumps({"test_a": "PASSED", "test_b": "PASSED"})
    log = _log(["test_a"])  # 少一个 ⇒ len 不等 ⇒ 0.0
    assert score_from_logs(log, expected) == 0.0


def test_score_empty_log_zero():
    expected = json.dumps({"test_a": "PASSED"})
    assert score_from_logs("no summary here", expected) == 0.0


def test_parse_log_strips_reason():
    log = _log(["ok"], ["bad"])
    parsed = parse_log_pytest(log)
    # FAILED 的 ` - reason` 尾巴被去掉
    assert any(v == "FAILED" for v in parsed.values())
    assert all(" - " not in k for k in parsed)


# ----------------------------------------------------- 数据契约(pilot parquet)

def test_pilot_dataset_contract(tmp_path):
    """若 pilot 数据集已构建则校验行契约与特权分离红线。"""
    from pathlib import Path

    ds = Path("experiments/bench_env/data/swe/dataset/train.parquet")
    if not ds.is_file():
        pytest.skip("pilot dataset not built")
    df = pd.read_parquet(ds)
    for _, r in df.iterrows():
        assert r["data_source"] == "swe_r2e"
        assert r["agent_name"] == "trace_opd"
        assert [m["role"] for m in r["prompt"]] == ["system", "user"]
        ei = r["extra_info"]
        assert ei["source"] == "r2e"
        # 红线:agent 可见面无 GT
        blob = json.dumps({k: str(v) for k, v in dict(ei).items()})
        assert "parsed_commit_content" not in blob
        assert "expected_output_json" not in blob
