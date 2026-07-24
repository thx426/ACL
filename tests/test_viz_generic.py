"""Genericity acceptance (viz-design.md §9-⑧): a made-up second environment
with NO observation plugin gets the full kernel — index, curves, stratified
listing, annotated trajectory detail, research join, task history.
"""
from trace_opd.viz.index import Index
from trace_opd.viz.queries import Queries

from viz_fixtures import GRID_TASK, make_gridworld_run


def _queries(tmp_path):
    make_gridworld_run(tmp_path / "runs")
    idx = Index(tmp_path / "runs", tmp_path / "data" / "idx.sqlite3")
    idx.scan()
    return Queries(idx)


def test_kernel_fully_functional_without_plugin(tmp_path):
    q = _queries(tmp_path)

    curves = q.curves("grid_run", ["score", "invalid_rate", "max_repeat"], "train")
    assert curves["series"]["score"]["points"] == [{"step": 1, "value": 0.5, "n": 2}]
    assert curves["series"]["max_repeat"]["points"][0]["value"] == 2.0  # (1+3)/2

    step = q.step_rollouts("grid_run", 1, False, {})
    assert step["stats"]["success_rate"] == 0.5
    assert step["stats"]["fail_labels"] == [{"value": "unclassified", "count": 1}]
    assert step["stats"]["families"] == [{"value": "maze", "count": 2}]
    assert "classifier=none@0" in step["provenance"]["citation"]

    lost = next(r for r in step["rollouts"] if not r["won"])
    detail = q.rollout_detail("grid_run", lost["file"])
    tags = [s["tags"] for s in detail["steps"]]
    assert tags[1] == ["invalid", "repeat"]  # generic events still annotated
    assert tags[3] == ["no_action"]
    assert detail["traj_tags"] == []

    research = q.research_detail("grid_run", detail["research_archive"])
    assert research is not None and len(research["consumers"]) == 2

    history = q.task_history("grid_run", GRID_TASK)
    assert history["family"] == "maze"
    assert history["history"][0]["env_steps_failed_median"] == 3
