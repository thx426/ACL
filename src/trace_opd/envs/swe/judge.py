"""SWE 域判分 —— R2E-Gym execution-based reward 的移植。

逐段移植自 R2E-Gym(experiments/bench_env/codebases/r2e-gym/):
  - 日志解析:`repo_analysis/execution_log_parser.py:4-35` parse_log_pytest
    (切 `short test summary info` 段,10 repo 同款;PASSED/FAILED/ERROR);
  - reward 比对:`agenthub/runtime/docker.py:1031-1066` _calculate_reward_r2e
    (逐 key 严格比对:len 不等或任一状态不符 → 0.0,全等 → 1.0,无部分分);
  - decolor:同文件 execution_log_parser.py:68-70(去 ANSI 色码)。

判分基准 `expected_output_json` 从 privileged 文件读(不进 agent 面);测试
运行命令是容器内 pytest(R2E `repo_analysis_args.py:91-114`,orange3 需 xvfb
但 pilot 不含 orange3)。

信息状态红线:judge 只被 adapter 的评分路径调用,返回 float;测试日志本身
含期望值不可反馈给 agent(submit 恒 "Submitted.")。
"""
from __future__ import annotations

import re

_ANSI = re.compile(r"\[\d+m")

#: 各 repo 的 r2e_tests 运行命令(R2E repo_analysis_args.py:91-114)。9 repo
#: 同款,orange3 需虚拟显示(pilot 不含,列全供扩量)。
_PYTEST_BASE = (
    "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' "
    ".venv/bin/python -W ignore -m pytest -rA r2e_tests"
)
_RUN_TESTS_CMD = {
    "orange3": "QT_QPA_PLATFORM=minimal " + _PYTEST_BASE.replace(
        ".venv/bin/python", "xvfb-run --auto-servernum .venv/bin/python"
    ),
}


def run_tests_command(repo_name: str) -> str:
    """容器内 /testbed 下运行 r2e_tests 的命令(pilot 全走默认 pytest)。"""
    return _RUN_TESTS_CMD.get(repo_name, _PYTEST_BASE)


def _decolor(d: dict) -> dict:
    return {_ANSI.sub("", k): v for k, v in d.items()}


def parse_log_pytest(log: str | None) -> dict[str, str]:
    """{test: status};切 short test summary info 段(R2E 逐字移植)。"""
    if not log or "short test summary info" not in log:
        return {}
    tail = log.split("short test summary info")[1].strip()
    out: dict[str, str] = {}
    for line in tail.split("\n"):
        if "PASSED" in line:
            out[".".join(line.split("::")[1:])] = "PASSED"
        elif "FAILED" in line:
            out[".".join(line.split("::")[1:]).split(" - ")[0]] = "FAILED"
        elif "ERROR" in line:
            try:
                name = ".".join(line.split("::")[1:])
            except IndexError:
                name = line
            out[name.split(" - ")[0]] = "ERROR"
    return out


def score_from_logs(test_log: str, expected_output_json: str) -> float:
    """R2E _calculate_reward_r2e 语义:逐 key 严格比对,二值。

    key 规范化 = `k.split(" - ")[0]`(去 pytest 的 ` - <reason>` 尾巴),两侧
    都做;数量不等或任一状态不符 → 0.0,全等 → 1.0。"""
    import json

    parse = _decolor(parse_log_pytest(test_log))
    expected = _decolor(json.loads(expected_output_json))
    parse = {k.split(" - ")[0]: parse[k] for k in sorted(parse)}
    expected = {k.split(" - ")[0]: expected[k] for k in sorted(expected)}
    if len(parse) != len(expected):
        return 0.0
    for k in parse:
        if not k:
            continue
        if k not in expected or parse[k] != expected[k]:
            return 0.0
    return 1.0
