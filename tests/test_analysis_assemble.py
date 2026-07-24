"""Real-data assembly layer: probe-set construction (partial-assistant
decision points), ledger/action-flow/fork builders, token-type annotation."""
import json

import numpy as np
import pytest

from fakes import FakeTokenizer

from trace_opd.analysis import assemble
from trace_opd.analysis.annotate import annotate
from trace_opd.analysis.probe import build_probe_set, doc_for
from trace_opd.analysis.schema import TokenRecord, TurnProfile, write_jsonl

TOK = FakeTokenizer()


def _write_dump(run_dir, step, name, *, won, priv, steps_n=2):
    step_dir = run_dir / "rollouts" / f"step_{step:05d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / f"{name}.json").write_text(json.dumps({
        "task_id": "maze/room-1", "split": "train", "validate": False,
        "global_step": step, "won": won, "outcome": "SUCCESS" if won else "FAILED",
        "env_steps": steps_n, "research": {"research_final_success": priv},
        "trajectory": {
            "task_id": "maze/room-1", "initial_obs": "A room.",
            "steps": [
                {"t": i + 1, "thought": f"th{i}", "action_raw": f"do {i}",
                 "action_norm": f"do {i}", "observation": f"obs{i}",
                 "valid": True, "acted": True}
                for i in range(steps_n)
            ],
        },
    }), encoding="utf-8")


def _write_doc(run_dir, task_safe, step, text):
    d = run_dir / "research" / task_safe / f"20260724_00000{step}_v{step}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "doc.md").write_text(text, encoding="utf-8")
    (d / "meta.json").write_text(json.dumps({"task_id": "maze/room-1"}), encoding="utf-8")


def test_build_probe_set_ends_in_open_action_tag(tmp_path):
    run = tmp_path / "run"
    _write_dump(run, 1, "a", won=False, priv=1.0)
    _write_doc(run, "maze__room-1", 1, "early doc")
    _write_doc(run, "maze__room-1", 9, "late doc")
    items = build_probe_set(run, n_items=4, turns_at=(1, 2))
    assert len(items) == 2  # one per (task, depth)
    for item in items:
        last = item.dialogue[-1]
        assert last["role"] == "assistant" and last["content"].endswith("<action>")
        assert item.doc_fix == "early doc"  # frozen = earliest at step<=1
        assert item.scaffold  # protocol scaffold, no provider needed
    assert doc_for(run, "maze/room-1", step=9) == "late doc"
    assert doc_for(run, "maze/room-1", step=9, earliest=True) == "early doc"


def test_ledger_points_merges_three_sources(tmp_path):
    run = tmp_path / "run"
    for step, wins in ((1, (0, 0)), (2, (1, 0))):
        for i, won in enumerate(wins):
            _write_dump(run, step, f"d{i}", won=bool(won), priv=0.9)
    write_jsonl(run / "analysis" / "turn_profiles.jsonl", [
        TurnProfile(step=1, traj_id="x", turn=1, g_mean=0.8, g_gated=0.9,
                    overlap=0.8, h_teacher=1.0, h_student=-1, n_tokens=10),
        TurnProfile(step=1, traj_id="x", turn=2, g_mean=0.4, g_gated=0.5,
                    overlap=0.8, h_teacher=1.0, h_student=-1, n_tokens=30),
    ])
    sweeps = run / "sweeps.jsonl"
    sweeps.write_text(json.dumps({"step": 2, "d_init": 0.1, "d_gain": 0.5,
                                  "d_frozen": 0.45}) + "\n", encoding="utf-8")
    pts = assemble.ledger_points(run, sweeps_path=sweeps)
    assert [p.step for p in pts] == [1, 2]
    assert pts[0].score == 0.0 and pts[1].score == 0.5
    assert pts[0].priv_success == pytest.approx(0.9)
    assert pts[0].g == pytest.approx((0.8 * 10 + 0.4 * 30) / 40)  # token-weighted
    assert np.isnan(pts[0].g_frozen) and pts[1].g_frozen == pytest.approx(0.45)


def _save_dist(path, dists):
    np.savez_compressed(path, **{k: np.log(np.array(v, dtype=np.float32)).astype(np.float16)
                                 for k, v in dists.items()})


def test_action_flow_and_fork_from_sweep_caches(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    group_map = {0: "navigate", 1: "inspect"}
    # step 20: bare mass sits on token1 (inspect), teacher pushes token0.
    _save_dist(state / "bare_step00020.npz",
               {"p0": [0.1, 0.7, 0.2], "p1": [0.2, 0.6, 0.2]})
    _save_dist(state / "teacher_step00020.npz",
               {"p0": [0.6, 0.2, 0.2], "p1": [0.7, 0.1, 0.2]})
    flow = assemble.action_flow_from_sweeps(state, group_map)
    assert flow.steps == [20]
    assert flow.student_probs["inspect"][0] == pytest.approx(0.65, abs=1e-3)
    assert flow.doc_uplift["navigate"][0] == pytest.approx(0.5, abs=1e-3)
    assert flow.student_probs["other"][0] == pytest.approx(0.2, abs=1e-3)

    fork = assemble.fork_stats_from_sweeps(state, group_map, fork_below=0.65,
                                           lock_above=0.65)
    assert fork.steps == [20]
    assert fork.fork_rate[0] == pytest.approx(0.5)  # p1 top1=0.6 < 0.65; p0 0.7 not
    assert fork.lock_rate[0] == pytest.approx(0.5)  # p0 0.7 > 0.65
    assert fork.fork_uplift["navigate"][0] == pytest.approx(0.5, abs=1e-3)


def test_annotate_types_and_text():
    text = "<reasoning>ok</reasoning>\n<action>go x</action>"
    ids = TOK.encode(text)
    records = [
        TokenRecord(step=1, traj_id="tr", t=i, turn=1, token_id=tid, kl=0.1)
        for i, tid in enumerate(ids)
    ]
    annotate(records, TOK)
    labels = {r.t: r.token_type for r in records}
    body = text.index("ok")
    assert labels[body] == labels[body + 1] == "reasoning-connective"
    g = text.index("go x")
    assert labels[g] == "action-verb"
    assert labels[g + 1] == labels[g + 3] == "action-arg"
    assert labels[0] == "format"                      # '<' of <reasoning>
    assert labels[text.index("\n")] == "format"       # glue between blocks
    assert records[g].token_text == "g"

    steps, shares = assemble.type_shares(records)
    assert steps == [1]
    total = sum(shares[t][0] for t in shares)
    assert total == pytest.approx(1.0)
    assert shares["action-arg"][0] > 0


def test_action_group_map_covers_space_variants():
    mapping = assemble.action_group_map(TOK, {"navigate": ["go"]})
    assert mapping[ord("g")] == "navigate"
    assert mapping[ord(" ")] == "navigate"  # leading-space variant's first id
