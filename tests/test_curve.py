"""build_curve_metrics — the swanlab per-sample scalar channel.

Locks the homogeneity contract (verl indexes the whole batch with the first
sample's key set) and the doc_missing merge semantics (teacher-context
overflow flips it after research reported 0).
"""
from trace_opd.train.curve import build_curve_metrics

_BASE = {"env_steps": 12, "invalid_format": 0, "truncated": False}
_RESEARCH = {
    "research_final_success": 0.75,
    "doc_missing": 0,
    "research_error": 0,
    "doc_words": 544,
    "research_experiments": 4,
    "research_episodes": 16,
}


def test_train_shape_is_base_plus_research_and_all_float():
    m = build_curve_metrics(
        validate=False, research_enabled=True, fields={**_BASE, **_RESEARCH}
    )
    assert set(m) == {
        "env_steps", "invalid_format", "truncated",
        "research_final_success", "doc_missing", "research_error",
        "doc_words", "research_experiments", "research_episodes",
    }
    assert all(isinstance(v, float) for v in m.values())
    assert m["research_final_success"] == 0.75 and m["env_steps"] == 12.0


def test_val_and_research_off_shapes_are_base_only_and_identical():
    val = build_curve_metrics(validate=True, research_enabled=True, fields=dict(_BASE))
    off = build_curve_metrics(validate=False, research_enabled=False, fields=dict(_BASE))
    assert set(val) == set(off) == {"env_steps", "invalid_format", "truncated"}


def test_doc_missing_reflects_teacher_overflow_degradation():
    # research said doc_missing=0, then the doc overflowed the teacher context
    # and run() flipped extra_fields["doc_missing"]=1 — the curve must see 1.
    fields = {**_BASE, **_RESEARCH, "doc_missing": 1}
    m = build_curve_metrics(validate=False, research_enabled=True, fields=fields)
    assert m["doc_missing"] == 1.0


def test_degraded_sample_stays_homogeneous():
    # research failure: fields carry zeros/None-ish values but every key must
    # still be present so the batch stays index-safe for verl.
    fields = {**_BASE, "research_error": 1, "doc_missing": 1, "doc_words": 0,
              "research_final_success": 0.0, "research_experiments": 0,
              "research_episodes": 0}
    m = build_curve_metrics(validate=False, research_enabled=True, fields=fields)
    assert m["research_error"] == 1.0 and m["doc_words"] == 0.0
    assert len(m) == 9
