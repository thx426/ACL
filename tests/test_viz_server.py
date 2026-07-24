"""End-to-end over a real HTTP server (stdlib client, random port)."""
import json
import threading
import urllib.error
import urllib.request
from urllib.parse import quote

import pytest

from trace_opd.viz.index import Index
from trace_opd.viz.queries import Queries
from trace_opd.viz.server import make_server

from viz_fixtures import make_alfworld_run


@pytest.fixture()
def base_url(tmp_path):
    make_alfworld_run(tmp_path / "runs")
    idx = Index(tmp_path / "runs", tmp_path / "data" / "idx.sqlite3")
    idx.scan()
    server = make_server("127.0.0.1", 0, Queries(idx))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _get(url):
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def test_full_drilldown_chain(base_url):
    runs = _get(f"{base_url}/api/runs")["runs"]
    assert runs[0]["run"] == "alf_run" and runs[0]["samples"] == 10

    curves = _get(f"{base_url}/api/run/alf_run/curves?keys=score,env_steps&mode=train")
    score = curves["series"]["score"]["points"]
    assert [p["step"] for p in score] == [1, 2]
    assert score[1]["value"] == 0.25 and score[1]["n"] == 4

    val = _get(f"{base_url}/api/run/alf_run/curves?keys=score&mode=val")
    assert set(val["series"]["score"]["splits"]) == {"valid_seen", "valid_unseen"}

    step = _get(f"{base_url}/api/run/alf_run/step/2/rollouts")
    assert step["stats"]["n"] == 4 and step["stats"]["n_success"] == 1
    assert {d["value"] for d in step["stats"]["fail_labels"]} == {
        "moved_no_take", "took_no_finish"
    }
    assert "classifier=alfworld@1" in step["provenance"]["citation"]

    file = next(r["file"] for r in step["rollouts"] if r["won"])
    detail = _get(f"{base_url}/api/run/alf_run/rollout?file={file}")
    assert detail["steps"][0]["tags"] == ["move"]
    assert detail["research_archive"]

    research = _get(
        f"{base_url}/api/run/alf_run/research?dir={detail['research_archive']}"
    )
    assert research["doc"].startswith("# Guidance")
    assert any(c["file"] == file for c in research["consumers"])

    history = _get(
        f"{base_url}/api/run/alf_run/task?task_id={quote(detail['task_id'], safe='')}"
    )
    assert [h["step"] for h in history["history"] if not h["validate"]] == [1, 2]
    assert len(history["research_sessions"]) == 2


def test_step_filters(base_url):
    lost = _get(f"{base_url}/api/run/alf_run/step/2/rollouts?won=0")
    assert lost["stats"]["n"] == 3 and lost["stats"]["n_success"] == 0
    fam = _get(
        f"{base_url}/api/run/alf_run/step/2/rollouts?family=pick_heat_then_place_in_recep"
    )
    assert fam["stats"]["n"] == 2


def test_path_traversal_refused(base_url):
    for bad in ("../../etc/passwd", "/etc/passwd"):
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(f"{base_url}/api/run/alf_run/rollout?file={quote(bad, safe='')}")
        assert err.value.code == 400


def test_unknown_curve_key_is_400(base_url):
    with pytest.raises(urllib.error.HTTPError) as err:
        _get(f"{base_url}/api/run/alf_run/curves?keys=nope")
    assert err.value.code == 400
