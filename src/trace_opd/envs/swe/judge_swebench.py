"""SWE-bench Verified 判分 —— 官方语义(不自己实现,调官方函数)。

与训练侧(R2E `judge.py` 的 expected_output 逐 key 比对)是**两套判分**:
本文件是 SWE-bench 官方口径 —— `FAIL_TO_PASS` 全过 **AND** `PASS_TO_PASS`
全过 ⇒ resolved(`swebench/harness/grading.py:195-212 get_resolution_status`,
本地 codebases/swe-bench 实读)。报表分数只认这一套。

两条路径(SweCfg.grading_mode):
  * ``deferred``(默认,**权威**):episode 只收 patch,不判分;导出
    predictions.jsonl → 在有 docker 的机器上跑官方
    ``python -m swebench.harness.run_evaluation``。零 swebench 依赖入训练环境,
    且判分链就是官方 harness 本身,不可能漂移。
  * ``inline``(可选,给 in-loop val 曲线):在同一沙箱内跑官方 eval_script
    并用官方 parser/report 判定。**需训练环境装 `swebench` 包**(R2E /
    SWE-Master 同样依赖它),缺失时 fail loud。

为何不自己写 parser:官方 `PARSER_REGISTRY` 是 per-repo 表(pytest/tox/
django/sympy 等各异),移植=引入静默错判风险;R2E 的 swebench_utils 也是
直接 import 官方包。
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

_IMPORT_HINT = (
    "SWE-bench inline grading needs the official `swebench` package "
    "(pip install swebench) — it owns the per-repo log parsers and the "
    "F2P/P2P resolution semantics. Use swe.grading_mode='deferred' to grade "
    "offline with the official harness instead."
)


def _swebench_mods():
    """惰性导入(deferred 模式下训练环境无需装 swebench)。"""
    try:
        from swebench.harness.grading import (  # noqa: PLC0415
            get_eval_tests_report,
            get_resolution_status,
        )
        from swebench.harness.log_parsers import (  # noqa: PLC0415
            MAP_REPO_TO_PARSER,
            get_eval_type,
        )
        from swebench.harness.test_spec.test_spec import (  # noqa: PLC0415
            make_test_spec,
        )
        from swebench.harness.constants import (  # noqa: PLC0415
            FAIL_TO_PASS,
            KEY_INSTANCE_ID,
            PASS_TO_PASS,
            ResolvedStatus,
        )
    except ImportError as err:  # pragma: no cover - 环境相关
        raise RuntimeError(f"{_IMPORT_HINT} (import error: {err})") from err
    return {
        "make_test_spec": make_test_spec,
        "get_eval_tests_report": get_eval_tests_report,
        "get_resolution_status": get_resolution_status,
        "MAP_REPO_TO_PARSER": MAP_REPO_TO_PARSER,
        "get_eval_type": get_eval_type,
        "FAIL_TO_PASS": FAIL_TO_PASS,
        "PASS_TO_PASS": PASS_TO_PASS,
        "KEY_INSTANCE_ID": KEY_INSTANCE_ID,
        "ResolvedStatus": ResolvedStatus,
    }


def swebench_available() -> bool:
    try:
        _swebench_mods()
    except RuntimeError:
        return False
    return True


def _instance_from(privileged: dict, item: dict) -> dict:
    """拼出官方 make_test_spec 需要的 instance dict(字段全在 privileged/
    extra_info 里;F2P/P2P 数据集里是 JSON 字符串,官方要 list)。"""
    def _lst(v):
        return json.loads(v) if isinstance(v, str) else list(v or [])

    return {
        "instance_id": str(item["instance_id"]),
        "repo": str(item["repo"]),
        "base_commit": str(item["base_commit"]),
        "version": str(item["version"]),
        "environment_setup_commit": str(privileged.get(
            "environment_setup_commit", item["base_commit"])),
        "problem_statement": "",          # eval 不需要
        "patch": privileged.get("patch", ""),
        "test_patch": privileged.get("test_patch", ""),
        "FAIL_TO_PASS": _lst(privileged.get("FAIL_TO_PASS")),
        "PASS_TO_PASS": _lst(privileged.get("PASS_TO_PASS")),
    }


def eval_script_for(privileged: dict, item: dict) -> str:
    """官方 eval_script(与离线 harness 用的是同一份生成逻辑,故 inline 与
    官方口径同源)。数据集自带的 `run_tests` 是旧版脚本(无 START/END 标记,
    与当前 parser 不配),仅作显式回退。"""
    mods = _swebench_mods()
    spec = mods["make_test_spec"](_instance_from(privileged, item))
    return spec.eval_script


def resolved_from_log(privileged: dict, item: dict, log: str) -> float:
    """官方判定:F2P 全过 AND P2P 全过 ⇒ 1.0,否则 0.0(二值)。"""
    mods = _swebench_mods()
    spec = mods["make_test_spec"](_instance_from(privileged, item))
    parser = mods["MAP_REPO_TO_PARSER"][spec.repo]
    # 官方 get_logs_eval 读文件且做标记检查;这里复用其语义但直接吃字符串:
    # 有标记就切段(与官方一致),没有就整段(旧脚本回退)。
    from swebench.harness.constants import (  # noqa: PLC0415
        APPLY_PATCH_FAIL, END_TEST_OUTPUT, RESET_FAILED,
        START_TEST_OUTPUT, TESTS_ERROR, TESTS_TIMEOUT,
    )
    for bad in (APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT):
        if bad in log:
            return 0.0
    content = log
    if START_TEST_OUTPUT in log and END_TEST_OUTPUT in log:
        content = log.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
    status_map = parser(content, spec)
    report = mods["get_eval_tests_report"](
        status_map,
        {
            mods["KEY_INSTANCE_ID"]: spec.instance_id,
            mods["FAIL_TO_PASS"]: spec.FAIL_TO_PASS,
            mods["PASS_TO_PASS"]: spec.PASS_TO_PASS,
        },
        eval_type=mods["get_eval_type"](spec),
    )
    return float(
        mods["get_resolution_status"](report) == mods["ResolvedStatus"].FULL.value
    )


def write_predictions(
    records: list[dict], out_path: str | Path, model_name: str
) -> Path:
    """轨迹记录 → 官方 harness 的 predictions 文件。

    官方契约(`RG/.../trajectory.py:553-558` 与 harness README 一致):每行
    ``{"instance_id", "model_name_or_path", "model_patch"}``。空 patch 也要
    写(否则该实例被当"未提交"而不是"提交了空解",统计口径不同)。"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "instance_id": str(r["instance_id"]),
                "model_name_or_path": model_name,
                "model_patch": str(r.get("patch", "") or ""),
            }, ensure_ascii=False) + "\n")
    return out
