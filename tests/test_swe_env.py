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


# ----------------------------------------------------- episode:setup / 预算 / 判分
# FakeSandbox 覆盖容器交互全路径(不需要真容器):记录命令、按前缀给回放。

class FakeSandbox:
    def __init__(self, script_out="__SETUP_OK__", cmd_out="ok", exit_code=0,
                 grade_log=""):
        self.cmds: list[str] = []
        self.polled: list[str] = []
        self._script_out = script_out
        self._cmd_out = cmd_out
        self._exit = exit_code
        self._grade_log = grade_log
        self.killed = False

    def _res(self, out, code=0):
        from trace_opd.envs.cloudbox import ExecResult
        return ExecResult(stdout=out, stderr="", exit_code=code)

    def run(self, command, timeout_s, grace_s=None):
        self.cmds.append(command)
        return self._res(self._cmd_out, self._exit)

    def run_polled(self, command, timeout_s, poll_s=5.0, grace_s=None):
        self.polled.append(command)
        if "__SETUP_OK__" in command:      # setup 脚本自带哨兵 echo
            return self._res(self._script_out)
        return self._res(self._grade_log)  # 判分

    def kill(self):
        self.killed = True


def _episode(**cfgkw):
    from trace_opd.common.config import SweCfg
    from trace_opd.envs.swe.adapter import SweEpisode
    sbx = FakeSandbox(**{k: v for k, v in cfgkw.items() if k in
                         ("script_out", "cmd_out", "exit_code", "grade_log")})
    cfg = SweCfg(data_root="x", **{k: v for k, v in cfgkw.items() if k not in
                                   ("script_out", "cmd_out", "exit_code",
                                    "grade_log", "max_env_steps")})
    item = {"initial_obs": "issue text", "instance_id": "repo__abc",
            "repo_name": "coveragepy", "docker_image": "ns/coveragepy_final:abc"}
    ep = SweEpisode(sbx, item, cfg, cfgkw.get("max_env_steps", 3))
    return ep, sbx


def test_setup_raw_image_moves_tests_and_hides_grader():
    """raw 镜像必需的三件:r2e_tests 从 / 归位、run_tests.sh 藏走、软链回 testbed。
    (镜像事实:Dockerfile COPY r2e_tests /r2e_tests;不做则判分找不到测试)"""
    _ep, sbx = _episode()
    script = sbx.polled[0]
    assert "mv -f /r2e_tests /root/r2e_tests" in script
    assert "mv -f /testbed/run_tests.sh /root/run_tests.sh" in script
    assert "ln -sfn /root/r2e_tests /testbed/r2e_tests" in script
    assert "-name '*.pyc' -delete" in script          # 陈旧 pyc 会掩盖修复


def test_setup_tests_invisible_skips_symlink():
    _ep, sbx = _episode(tests_visible=False)
    script = sbx.polled[0]
    assert "ln -sfn /root/r2e_tests /testbed/r2e_tests" not in script
    assert "mv -f /r2e_tests /root/r2e_tests" in script  # 判分侧仍需要


def test_setup_fail_loud_when_tests_missing():
    """测试目录缺失 = 镜像不对 ⇒ 抛(否则每条轨迹静默判 0 = reward 事故)。"""
    from trace_opd.envs.cloudbox import CloudboxError
    with pytest.raises(CloudboxError, match="setup failed"):
        _episode(script_out="__SETUP_NO_TESTS__")


def test_prebaked_setup_is_short():
    _ep, sbx = _episode(prebaked_setup=True)
    script = sbx.polled[0]
    assert "mv -f /r2e_tests" not in script      # 已烘进镜像
    assert "__SETUP_OK__" in script              # 但仍校验


def test_budget_line_and_forced_grading():
    """预算三件套:逐步文案 → 到点无条件判分 → score×α。"""
    from trace_opd.common.config import SweCfg
    from trace_opd.envs.swe.adapter import SweTaskAdapter

    grade_log = "\n".join([
        "=== short test summary info ===",
        "PASSED r2e_tests/test_1.py::test_a",
    ])
    ep, sbx = _episode(max_env_steps=2, grade_log=grade_log)
    cfg = SweCfg(data_root="x")
    item = {"instance_id": "repo__abc", "repo_name": "coveragepy",
            "initial_obs": "issue", "docker_image": "ns/coveragepy_final:abc"}
    ad = SweTaskAdapter(item, cfg, factory=None)
    ad._priv = {"expected_output_json": json.dumps({"test_a": "PASSED"})}

    call = '<tool_call>\n{"name": "execute_bash", "arguments": {"command": "ls"}}\n</tool_call>'
    out1 = ad.react(ep, call)
    assert "Steps Remaining: 1" in out1.feedback and not out1.done
    out2 = ad.react(ep, call)                      # 第 2 步 = 预算耗尽
    assert out2.done is True
    assert out2.score == pytest.approx(0.5)        # 1.0 × α(默认 0.5)
    assert "maximum number of steps" in out2.feedback


def test_grading_command_ensures_symlink_and_prefers_image_script():
    ep, sbx = _episode(grade_log="")
    ep.run_and_score(json.dumps({"t": "PASSED"}), "coveragepy")
    grade_cmd = sbx.polled[-1]
    assert "ln -sfn /root/r2e_tests r2e_tests" in grade_cmd   # tests_visible=False 时也能判
    assert "bash /root/run_tests.sh" in grade_cmd             # 优先镜像自带
    assert "pytest -rA r2e_tests" in grade_cmd                # 缺失时回落


def test_timeout_exit124_is_rendered():
    ep, _sbx = _episode(exit_code=124, cmd_out="")
    body, code = ep.exec_in_testbed("sleep 999")
    assert code == 124 and "timed out" in body


# ----------------------------------------------------- 评测侧(SWE-bench Verified)

def test_eval_dataset_contract():
    """评测 parquet:同一 prompt 真源、source=swebench、零 GT 泄漏。"""
    from pathlib import Path
    ds = Path("experiments/bench_env/data/swe/eval/test.parquet")
    if not ds.is_file():
        pytest.skip("eval dataset not built")
    from trace_opd.envs.swe.prompts import SYSTEM_PROMPT
    df = pd.read_parquet(ds)
    assert len(df) == 500                      # Verified 官方规模
    ids = set()
    for _, r in df.iterrows():
        assert r["data_source"] == "swe_verified"
        assert r["prompt"][0]["content"] == SYSTEM_PROMPT   # 与训练同一真源
        ei = r["extra_info"]
        assert ei["source"] == "swebench"
        assert ei["docker_image"].startswith("slimshetty/swebench-verified:")
        ids.add(ei["instance_id"])
        blob = json.dumps([dict(m) for m in r["prompt"]], ensure_ascii=False)
        assert "FAIL_TO_PASS" not in blob and "PASS_TO_PASS" not in blob
    assert len(ids) == 500                     # instance_id 唯一(harness 按它对齐)


def test_eval_privileged_holds_gt_and_tests():
    from pathlib import Path
    pdir = Path("experiments/bench_env/data/swe/eval/privileged")
    if not pdir.is_dir():
        pytest.skip("eval dataset not built")
    sample = json.loads(sorted(pdir.glob("*.json"))[0].read_text())
    for k in ("patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS",
              "environment_setup_commit"):
        assert k in sample and sample[k]


def test_swebench_episode_setup_and_conda_prefix():
    """swebench 源:用另一套 setup(无 r2e_tests)+ 命令前缀激活 conda。"""
    from trace_opd.common.config import SweCfg
    from trace_opd.envs.swe.adapter import SweEpisode
    sbx = FakeSandbox(script_out="__SETUP_OK__")
    item = {"initial_obs": "issue", "instance_id": "astropy__astropy-12907",
            "repo_name": "astropy", "repo": "astropy/astropy",
            "source": "swebench",
            "docker_image": "slimshetty/swebench-verified:sweb.eval.x86_64.x"}
    ep = SweEpisode(sbx, item, SweCfg(data_root="x"), 5)
    setup = sbx.polled[0]
    assert "safe.directory /testbed" in setup      # 否则收 patch 静默失败
    assert "envs/testbed" in setup                 # 镜像身份哨兵
    assert "r2e_tests" not in setup                # 与 R2E 分支互斥
    ep.exec_in_testbed("ls")
    assert "conda activate testbed" in sbx.cmds[-1]


def test_swebench_deferred_grading_collects_patch_scores_zero():
    """deferred:episode 只收 patch、分数 0(权威分离线出)。"""
    from trace_opd.common.config import SweCfg
    from trace_opd.envs.swe.adapter import SweEpisode, SweTaskAdapter
    sbx = FakeSandbox(cmd_out="diff --git a/x.py b/x.py\n+fix")
    item = {"initial_obs": "issue", "instance_id": "astropy__astropy-12907",
            "repo_name": "astropy", "repo": "astropy/astropy",
            "source": "swebench", "docker_image": "ns/img:tag"}
    cfg = SweCfg(data_root="x", grading_mode="deferred")
    ep = SweEpisode(sbx, item, cfg, 5)
    ad = SweTaskAdapter(item, cfg, factory=None)
    out = ad.react(ep, '<tool_call>\n{"name": "submit", "arguments": {}}\n</tool_call>')
    assert out.done and out.score == 0.0
    assert "diff --git" in ad.last_patch and ad.last_patch == ep.final_patch
    assert "Submitted." in out.feedback          # 零判分信息回流


def test_write_predictions_official_contract(tmp_path):
    from trace_opd.envs.swe.judge_swebench import write_predictions
    out = write_predictions(
        [{"instance_id": "astropy__astropy-12907", "patch": "diff --git a/x"},
         {"instance_id": "django__django-11039", "patch": ""}],   # 空 patch 也必须写
        tmp_path / "predictions.jsonl", model_name="trace-opd-9b")
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(lines) == 2
    assert set(lines[0]) == {"instance_id", "model_name_or_path", "model_patch"}
    assert lines[1]["model_patch"] == ""


def test_grading_mode_validation():
    from trace_opd.common.config import SweCfg
    with pytest.raises(ValueError, match="grading_mode"):
        SweCfg(data_root="x", grading_mode="bogus")._validate()
