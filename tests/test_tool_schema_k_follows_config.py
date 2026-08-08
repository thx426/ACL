"""工具 schema 里 k 的默认值必须跟随 ResearchCfg。

2026-08-07 实测事故:`TOOL_SCHEMAS` 里 k 的描述硬编码 "Number of parallel
episodes (default 4)."。模型把这句话当权威,于是**每次都显式填 k=4** ——
而 `tools.py` 是 `k_req = args.get("k", cfg.k_default)`,一旦显式填了就
永远取不到 `cfg.k_default`。

后果:把 `k_default` 从 4 改成 2 之后,两个 run(s3 与 v33)的
**episode/实验 仍恒为 4.00**,该配置项完全失效,只有 `k_max` 的 clamp 起作用;
而"降 k 省一半 episode"的预期收益一次也没兑现。
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace_opd.research.prompts import TOOL_SCHEMAS, tool_schemas_for  # noqa: E402


@dataclass
class _Cfg:
    k_default: int
    k_max: int


def _k_desc(schemas):
    for t in schemas:
        fn = t.get("function", {})
        if fn.get("name") != "run_experiment":
            continue
        return fn["parameters"]["properties"]["k"]["description"]
    raise AssertionError("run_experiment schema 不存在")


def test_k_description_carries_configured_default_and_max():
    d = _k_desc(tool_schemas_for(_Cfg(k_default=2, k_max=4)))
    assert "default 2" in d, f"描述未带上配置的 k_default:{d}"
    assert "maximum 4" in d, f"描述未带上配置的 k_max:{d}"
    # 反例:配置是 2 时,描述里不能再出现会诱导模型填 4 的 "default 4"
    assert "default 4" not in d


def test_k_description_tracks_a_different_config():
    d = _k_desc(tool_schemas_for(_Cfg(k_default=4, k_max=8)))
    assert "default 4" in d and "maximum 8" in d


def test_module_constant_has_no_hardcoded_default():
    """兜底文案本身不得写死某个数字,否则 text 协议下会与配置冲突。"""
    d = _k_desc(TOOL_SCHEMAS)
    assert "default" not in d.lower(), f"TOOL_SCHEMAS 仍硬编码默认值:{d}"


def test_returns_a_copy_not_the_module_constant():
    """必须深拷贝 —— 就地改会污染同进程后续所有会话的 schema。"""
    before = json.dumps(TOOL_SCHEMAS, sort_keys=True)
    tool_schemas_for(_Cfg(k_default=7, k_max=9))
    assert json.dumps(TOOL_SCHEMAS, sort_keys=True) == before


def test_other_tools_survive_untouched():
    out = tool_schemas_for(_Cfg(k_default=2, k_max=4))
    names = {t["function"]["name"] for t in out}
    assert {"run_experiment", "fetch_raw", "finalize"} <= names
