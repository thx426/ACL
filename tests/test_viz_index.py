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


def test_expert_rollouts_are_kept_out_of_every_aggregate(tmp_path):
    """Mixed objective: expert rollouts run WITH the document and win far more
    often. They must never reach a curve, a score, or the completeness audit —
    otherwise the observatory reports progress the policy never made."""
    idx = _index(tmp_path)
    run_dir = tmp_path / "runs" / "alf_run"
    q = Queries(idx)
    idx.scan()
    before_curve = q.curves("alf_run", ["score"], "train")["series"]["score"]["points"]
    before_overview = q.overview("alf_run")
    before_runs = q.runs()[0]

    for i in range(4):  # four winning expert rows on an existing train step
        write_dump(run_dir, step=1, validate=False, task_id=ALF_TASK,
                   traj=_traj(ALF_TASK, WIN_STEPS, True), uid=f"expert00{i}",
                   sample_role="sft")
    assert idx.scan()["alf_run"]["added"] == 4

    assert q.curves("alf_run", ["score"], "train")["series"]["score"]["points"] == before_curve
    assert q.overview("alf_run")["steps"] == before_overview["steps"]
    assert q.runs()[0]["train_score"] == before_runs["train_score"]
    history = q.task_history("alf_run", ALF_TASK)["history"]
    assert not any(r["file"].count("expert") for g in history for r in g["rollouts"])

    # ...but they stay browsable in the step drill-down, labelled and excluded
    # from the success split shown above the listing.
    step = q.step_rollouts("alf_run", 1, False, {})
    assert step["stats"]["n_expert"] == 4
    assert step["stats"]["n"] == before_overview["steps"][0]["n"]
    assert sum(1 for r in step["rollouts"] if r["sample_role"] == "sft") == 4
