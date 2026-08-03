"""GT-doc 消融实验(docs/gt-doc-ablation.md;ALFWorld 专属验证)——
渲染函数、research 短路、指标占位键集、config 护栏与 fingerprint。
独立测试文件:验证件不混入通用机制的测试面。"""
import pytest

torch = pytest.importorskip("torch")

from trace_opd.envs.alfworld.gt import _WALKTHROUGH_HEADER, render_gt_as_doc  # noqa: E402


# ---------------------------------------------------------------------------
# 1. 渲染:只认 walkthrough 可执行形态
# ---------------------------------------------------------------------------
def test_render_walkthrough_form():
    gt = (f"{_WALKTHROUGH_HEADER}\n"
          "1. go to shelf 2\n2. take alarmclock 1 from shelf 2\n"
          "3. put alarmclock 1 in/on desk 1")
    doc = render_gt_as_doc(gt)
    assert doc.startswith("A verified step-by-step reference solution")
    assert "1. go to shelf 2" in doc and "3. put alarmclock 1" in doc
    # 原引擎术语头不进教师文档(注入姿态最小化)。
    assert "Engine walkthrough" not in doc


def test_render_rejects_non_executable_forms():
    # high_pddl 高层计划(无实例编号、不可执行)⇒ None ⇒ 走 nodoc 降级。
    assert render_gt_as_doc(
        "High-level expert plan (ALFRED annotation):\n1. GotoLocation(shelf)") is None
    assert render_gt_as_doc(None) is None
    assert render_gt_as_doc("") is None
    assert render_gt_as_doc(f"{_WALKTHROUGH_HEADER}\n   ") is None
    assert render_gt_as_doc("1. go to shelf 2") is None  # 无头部,来源不明


# ---------------------------------------------------------------------------
# 2. research 短路:组合语义(render → fields)+ agent_loop 源 pin
#    (agent_loop 顶层 import 背 verl 训练栈,单测环境不可 import ⇒ 短路块
#    以源 pin 锁定,其两个组件各自有真实功能测试)
# ---------------------------------------------------------------------------
def test_static_source_semantics_render_plus_fields():
    from trace_opd.research.session import static_doc_fields

    gt = f"{_WALKTHROUGH_HEADER}\n1. go to drawer 1\n2. open drawer 1"
    doc = render_gt_as_doc(gt)
    fields = static_doc_fields(doc)
    assert doc.startswith("A verified step-by-step reference solution")
    assert fields["doc_missing"] == 0 and fields["doc_words"] > 0
    assert fields["research_experiments"] == 0
    # 无 walkthrough 形态 ⇒ nodoc 降级语义(doc=None, doc_missing=1)。
    doc2 = render_gt_as_doc("High-level expert plan (ALFRED annotation):\n1. X")
    fields2 = static_doc_fields(doc2)
    assert doc2 is None and fields2["doc_missing"] == 1


def test_agent_loop_short_circuit_source_pin():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src" / "trace_opd" / "train"
           / "agent_loop.py").read_text(encoding="utf-8")
    body = src.split("async def _research(")[1].split("async def ")[0]
    # 短路块在会话构造之前,且三件套齐:开关判断、渲染、占位 fields。
    gate = body.index('static_doc_source == "gt"')
    assert gate < body.index("ToolRouter(")
    assert "render_gt_as_doc(adapter.gt_text())" in body
    assert "static_doc_fields(doc)" in body


def test_static_fields_key_set_matches_real_session_path():
    # 曲线键批一致纪律(codex tg-review P0-3):占位键集 = 真实会话 fields 键集,
    # 且覆盖 _RESEARCH 曲线族全部键。
    from trace_opd.research.session import static_doc_fields
    from trace_opd.train.curve import _RESEARCH

    fields = static_doc_fields("a b c")
    assert set(_RESEARCH) <= set(fields)
    assert set(fields) == {
        "research_error", "research_experiments", "research_episodes",
        "research_turns", "research_final_success", "doc_words",
        "doc_missing", "research_archive",
    }
    assert fields["doc_words"] == 3 and fields["research_final_success"] == 0.0


# ---------------------------------------------------------------------------
# 3. config 护栏与 fingerprint
# ---------------------------------------------------------------------------
def _load(tmp_path, name, overrides):
    from test_mixed_objective import _TEMPLATE_YAML, _write

    from trace_opd.common.config import load_config

    return load_config(_write(tmp_path / name, "c.yaml", _TEMPLATE_YAML),
                       overrides=overrides)


def test_config_static_source_rules(tmp_path):
    cfg = _load(tmp_path, "g0", ["research.static_doc_source=gt"])
    assert cfg.research.static_doc_source == "gt"
    with pytest.raises(ValueError, match="null or 'gt'"):
        _load(tmp_path, "g1", ["research.static_doc_source=file"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        _load(tmp_path, "g2", ["research.static_doc_source=gt",
                               'research.gt="on"'])
    with pytest.raises(ValueError, match="research_rl"):
        _load(tmp_path, "g3", ["research.static_doc_source=gt",
                               "research_rl.weight=0.1"])


def test_fingerprint_static_source_is_conditional(tmp_path):
    from trace_opd.common.config import objective_fingerprint

    fp_default = objective_fingerprint(_load(tmp_path, "f0", None))
    fp_gt = objective_fingerprint(
        _load(tmp_path, "f1", ["research.static_doc_source=gt"]))
    assert fp_gt != fp_default  # 教师信息源 = objective 语义,拒跨臂 resume
    fp_default2 = objective_fingerprint(_load(tmp_path, "f2", None))
    assert fp_default2 == fp_default  # 默认 null 不进 hash,存量指纹不变
