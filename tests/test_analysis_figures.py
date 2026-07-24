"""Figure suite: every paper figure renders to non-trivial PDF+PNG from the
demo fixtures, and the JSONL data contract round-trips."""
import matplotlib

matplotlib.use("Agg")

from trace_opd.analysis import demo
from trace_opd.analysis.schema import ProbePoint, TurnProfile, read_jsonl, write_jsonl


def test_all_figures_render(tmp_path):
    made = demo.render_all(tmp_path)
    names = {p.name for p in made}
    for fig in ("a_ledger", "b_trajectory", "c_terrain", "d_microscope",
                "e_action_flow", "f_fingerprint", "g_fork"):
        assert f"fig_{fig}.pdf" in names and f"fig_{fig}.png" in names
    for p in made:
        assert p.stat().st_size > 2000, f"{p.name} looks empty"


def test_terrain_matrix_masks_missing_cells():
    from trace_opd.analysis.figures import _terrain_matrix

    profiles = demo.turn_profiles()
    profiles = [p for p in profiles if not (p.step == 50 and p.turn == 3)]
    steps, turns, mat = _terrain_matrix(profiles, gated=False)
    assert mat.mask[list(steps).index(50), list(turns).index(3)]
    assert not mat.mask.all()


def test_jsonl_roundtrip(tmp_path):
    rows = demo.turn_profiles()[:5]
    path = write_jsonl(tmp_path / "turns.jsonl", rows)
    back = read_jsonl(path, TurnProfile)
    assert back == rows

    probes = demo.probe_arms()["dynamic docs (ours)"][:3]
    back_p = read_jsonl(write_jsonl(tmp_path / "p.jsonl", probes), ProbePoint)
    assert [p.step for p in back_p] == [p.step for p in probes]
    assert back_p[0].d_frozen == probes[0].d_frozen
