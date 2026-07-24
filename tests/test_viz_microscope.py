"""viz microscope layer: terrain/track queries over collect outputs + routes."""
import json
import threading
import urllib.request

import pytest

from trace_opd.analysis.schema import TokenRecord, TurnProfile, write_jsonl
from trace_opd.viz.index import Index
from trace_opd.viz.queries import Queries
from trace_opd.viz.server import make_server

from viz_fixtures import make_alfworld_run


@pytest.fixture()
def setup(tmp_path):
    make_alfworld_run(tmp_path / "runs")
    analysis = tmp_path / "runs" / "alf_run" / "analysis"
    write_jsonl(analysis / "turn_profiles.jsonl", [
        TurnProfile(step=1, traj_id="rollouts/step_00001/a.json", turn=1,
                    g_mean=0.9, g_gated=1.0, overlap=0.8, h_teacher=1.0,
                    h_student=-1, n_tokens=10),
        TurnProfile(step=1, traj_id="rollouts/step_00001/a.json", turn=2,
                    g_mean=0.3, g_gated=0.4, overlap=0.8, h_teacher=1.0,
                    h_student=-1, n_tokens=10),
        TurnProfile(step=2, traj_id="rollouts/step_00002/b.json", turn=1,
                    g_mean=0.5, g_gated=0.6, overlap=0.9, h_teacher=0.9,
                    h_student=-1, n_tokens=10),
    ])
    write_jsonl(analysis / "token_records.jsonl", [
        TokenRecord(step=1, traj_id="rollouts/step_00001/a.json", t=t, turn=1,
                    token_id=100 + t, kl=0.1 * t, token_type="action-verb" if t == 2 else "",
                    token_text="go" if t == 2 else "")
        for t in range(4)
    ])
    idx = Index(tmp_path / "runs", tmp_path / "data" / "idx.sqlite3")
    idx.scan()
    return Queries(idx)


def test_microscope_overview_matrix_and_trajs(setup):
    d = setup.microscope_overview("alf_run")
    assert d["available"] and d["steps"] == [1, 2] and d["turns"] == [1, 2]
    assert d["matrix"][0] == [0.9, 0.3]
    assert d["matrix"][1] == [0.5, None]  # step-2 has no turn-2 sample
    assert d["trajs"][0]["g_mean"] == pytest.approx(0.6)  # token-weighted mean


def test_microscope_track_filters_and_sorts(setup):
    d = setup.microscope_track("alf_run", 1, "rollouts/step_00001/a.json")
    assert [t["t"] for t in d["tokens"]] == [0, 1, 2, 3]
    assert d["tokens"][2]["token_text"] == "go"
    assert setup.microscope_track("alf_run", 9, "nope")["tokens"] == []


def test_microscope_routes(setup):
    server = make_server("127.0.0.1", 0, setup)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/api/run/alf_run/microscope") as r:
            assert json.loads(r.read())["available"]
        with urllib.request.urlopen(
            f"{base}/api/run/alf_run/microscope_track?step=1"
            "&traj=rollouts%2Fstep_00001%2Fa.json"
        ) as r:
            assert len(json.loads(r.read())["tokens"]) == 4
        with urllib.request.urlopen(f"{base}/api/run/grid/microscope") as r:
            pass
    except urllib.error.HTTPError as err:
        assert err.code in (404, 500)  # unknown run only
    finally:
        server.shutdown()
