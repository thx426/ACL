"""Index: scan counts, incrementality, corrupt-file tolerance, audit."""
from trace_opd.viz.index import Index, discover_runs
from trace_opd.viz.queries import Queries

from viz_fixtures import ALF_TASK, WIN_STEPS, _traj, make_alfworld_run, write_dump


def _index(tmp_path):
    make_alfworld_run(tmp_path / "runs")
    return Index(tmp_path / "runs", tmp_path / "data" / "idx.sqlite3")


def test_scan_counts_and_incremental(tmp_path):
    idx = _index(tmp_path)
    report = idx.scan()
    assert report["alf_run"]["added"] == 10 + 4  # 10 rollouts + 4 research sessions
    assert idx.scan()["alf_run"]["added"] == 0  # second scan: nothing new
    write_dump(tmp_path / "runs" / "alf_run", step=3, validate=False,
               task_id=ALF_TASK, traj=_traj(ALF_TASK, WIN_STEPS, True), uid="late0000")
    assert idx.scan()["alf_run"]["added"] == 1  # picked up incrementally


def test_corrupt_file_skipped_not_fatal(tmp_path):
    idx = _index(tmp_path)
    bad = tmp_path / "runs" / "alf_run" / "rollouts" / "step_00001" / "broken_x.json"
    bad.write_text("{not json", encoding="utf-8")
    report = idx.scan()
    assert report["alf_run"]["skipped"] == 1
    assert report["alf_run"]["added"] == 14


def test_discover_and_audit(tmp_path):
    idx = _index(tmp_path)
    idx.scan()
    assert discover_runs(tmp_path / "runs") == ["alf_run"]
    overview = Queries(idx).overview("alf_run")
    assert overview["expected_per_step"] == 4
    train_steps = [s for s in overview["steps"] if not s["validate"]]
    assert [s["n"] for s in train_steps] == [4, 4]
    assert not any(s["short"] for s in train_steps)
    # a partially-dumped step gets flagged
    write_dump(tmp_path / "runs" / "alf_run", step=3, validate=False,
               task_id=ALF_TASK, traj=_traj(ALF_TASK, WIN_STEPS, True), uid="part0000")
    idx.scan()
    step3 = [s for s in Queries(idx).overview("alf_run")["steps"]
             if s["step"] == 3 and not s["validate"]][0]
    assert step3["short"] is True


def test_research_rows(tmp_path):
    idx = _index(tmp_path)
    idx.scan()
    with idx.connect() as db:
        rows = db.execute(
            "SELECT * FROM research WHERE run='alf_run' ORDER BY dir"
        ).fetchall()
    assert len(rows) == 4
    assert all(r["doc_present"] == 1 and r["final_success"] == 1.0 for r in rows)
    assert {r["policy_version"] for r in rows} == {1, 2}
