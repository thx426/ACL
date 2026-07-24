"""Generic trajectory stats, plugin classification, stratified aggregates."""
from trace_opd.viz import plugins
from trace_opd.viz.queries import _stratified
from trace_opd.viz.schema import traj_stats

from viz_fixtures import (
    ALF_TASK,
    EXHAUSTIVE_STEPS,
    GRID_LOSE,
    GRID_TASK,
    MOVED_NO_TAKE_STEPS,
    NEVER_MOVED_STEPS,
    TOOK_NO_FINISH_STEPS,
    WIN_STEPS,
    _traj,
)


def test_traj_stats_counts_protocol_flags_only():
    s = traj_stats(_traj(ALF_TASK, NEVER_MOVED_STEPS, False))
    assert (s.n_steps, s.n_noact, s.n_invalid) == (4, 1, 3)
    assert s.max_repeat == 2  # examine drawer 1 twice in a row
    s = traj_stats(_traj(GRID_TASK, GRID_LOSE, False))
    assert (s.n_noact, s.n_invalid, s.max_repeat) == (1, 2, 3)


def test_alfworld_failure_stages():
    cls = lambda steps: plugins.classify(ALF_TASK, _traj(ALF_TASK, steps, False), False)
    assert cls(NEVER_MOVED_STEPS) == "never_moved"
    assert cls(MOVED_NO_TAKE_STEPS) == "moved_no_take"
    assert cls(TOOK_NO_FINISH_STEPS) == "took_no_finish"
    assert plugins.classify(ALF_TASK, _traj(ALF_TASK, WIN_STEPS, True), True) == "success"
    plugin = plugins.plugin_for(ALF_TASK)
    assert plugin.traj_tags(_traj(ALF_TASK, EXHAUSTIVE_STEPS, False)) == ["exhaustive_search"]
    assert plugins.task_family(ALF_TASK) == "pick_and_place_simple"


def test_unplugged_environment_degrades_not_breaks():
    assert plugins.plugin_for(GRID_TASK) is None
    assert plugins.classify(GRID_TASK, _traj(GRID_TASK, GRID_LOSE, False), False) == "unclassified"
    assert plugins.task_family(GRID_TASK) == "maze"
    assert plugins.classifier_version(GRID_TASK) == "none@0"


def test_stratified_always_splits_success_and_failure():
    rows = [
        {"won": 1, "env_steps": 8, "fail_label": "success", "family": "f"},
        {"won": 0, "env_steps": 30, "fail_label": "never_moved", "family": "f"},
        {"won": 0, "env_steps": 30, "fail_label": "never_moved", "family": "f"},
    ]
    s = _stratified(rows)
    assert s["success_rate"] == round(1 / 3, 4)
    assert s["env_steps_success_median"] == 8
    assert s["env_steps_failed_median"] == 30  # the CD-006 median=30 trap, layered
    assert s["fail_labels"][0] == {"value": "never_moved", "count": 2}
