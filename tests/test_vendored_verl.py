"""Vendored-verl export integrity — every trace-opd patch must be present in
the exported tree ON ITS ACTUAL EXECUTION PATH.

Lesson from the f8f26d5 incident (CD-006, 2026-07-24): the rollout_extra
curves patch sat only in the V0 trainer (ray_trainer.py) while training
actually runs the V1 trainer (TaskRunnerV1 -> v1/trainer_base.py), so the
curves silently never appeared. These assertions grep the vendored files the
launcher puts on PYTHONPATH, so a stale export or a patch landing on a dead
path fails loudly here.
"""
from pathlib import Path

_VERL = Path(__file__).resolve().parents[1] / "verl" / "verl"


def _read(rel):
    return (_VERL / rel).read_text(encoding="utf-8")


def test_rollout_extra_curves_on_v1_trainer_path():
    # e8f697c: import + call site must both be in the V1 trainer, which is
    # what training executes (V0 keeps its copy for parity, not sufficient).
    src = _read("trainer/ppo/v1/trainer_base.py")
    assert src.count("compute_rollout_extra_metrics") >= 2
    assert "def compute_rollout_extra_metrics" in _read("trainer/ppo/metric_utils.py")  # f8f26d5


def test_micro_batch_budget_is_packing_target_not_assert():
    # 6d77dc7: the old hard assert would kill the whole run on one long-tail
    # trajectory; the export must carry the clamp+warning form.
    assert "below longest sequence" in _read("utils/seqlen_balancing.py")


def test_teacher_scoring_keeps_sticky_routing_key():
    # 79d4252: self-teacher scoring shares the per-(step,task) route key.
    assert "teacher_routing_key" in _read("experimental/agent_loop/agent_loop.py")
