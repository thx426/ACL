"""P0b 统计对拍脚本的判定逻辑单测(合成 dump,无 GPU)。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "engine_sps_parity_check",
    Path(__file__).resolve().parents[2] / "scripts" / "engine_sps_parity_check.py",
)
parity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_spec and parity)


def _traj(i, *, won, cand, opens, count, rejects, rows, spent, log_pairs,
          cap=2.0, d_hist=(3, 1), arm=1, role=None):
    d = {
        "won": won, "response_token_len": 100 + i,
        "steering": {
            "steer_arm": arm, "steer_record_rows": rows,
            "steer_n_candidates": cand, "steer_n_base_open": opens,
            "steer_count": count, "steer_reject_count": rejects,
            "steer_n_gate_bypassed": 0, "steer_budget_spent": spent,
            "steer_log_ratio_cap": cap,
            "steer_log_pi": [p for p, _ in log_pairs],
            "steer_log_b": [b for _, b in log_pairs],
            "steer_d_hist": list(d_hist), "steer_ent_hist": [1, 2],
        },
    }
    if role:
        d["sample_role"] = role
    return d


def _write(dirpath: Path, trajs):
    dirpath.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(trajs):
        (dirpath / f"t{i:03d}.json").write_text(json.dumps(t))


def _batch(n, *, won_rate=0.5, reject=2, spent=3.0, pairs=((-1.0, -0.5),)):
    return [
        _traj(i, won=(i / n) < won_rate, cand=100, opens=40, count=10,
              rejects=reject, rows=12, spent=spent, log_pairs=list(pairs))
        for i in range(n)
    ]


def test_parity_pass_on_statistically_identical_batches(tmp_path):
    _write(tmp_path / "a", _batch(20))
    _write(tmp_path / "b", _batch(20))
    sa = parity._summarize(parity._load_steered(str(tmp_path / "a")))
    sb = parity._summarize(parity._load_steered(str(tmp_path / "b")))
    assert parity.compare(sa, sb, tol_rate=0.03, tol_hist=0.10) == []


def test_parity_fails_on_rate_divergence_and_bound_violation(tmp_path):
    _write(tmp_path / "a", _batch(20, reject=2))
    _write(tmp_path / "b", _batch(20, reject=8))          # 拒绝率翻倍以上
    sa = parity._summarize(parity._load_steered(str(tmp_path / "a")))
    sb = parity._summarize(parity._load_steered(str(tmp_path / "b")))
    fails = parity.compare(sa, sb, tol_rate=0.03, tol_hist=0.10)
    assert any("reject_rate" in f for f in fails)

    # 密度比界违例:|log_b − log_pi| > cap ⇒ 单侧即 FAIL
    _write(tmp_path / "c", _batch(5, pairs=((-1.0, 2.0),)))  # 差 3.0 > cap 2.0
    sc = parity._summarize(parity._load_steered(str(tmp_path / "c")))
    fails2 = parity.compare(sa, sc, tol_rate=1.0, tol_hist=1.0)  # 率全放行
    assert any("bound" in f for f in fails2)


def test_loader_excludes_research_and_bare_rows(tmp_path):
    rows = _batch(4) + [
        _traj(90, won=True, cand=1, opens=1, count=1, rejects=0, rows=1,
              spent=0.1, log_pairs=[(-1.0, -1.0)], role="research"),
        _traj(91, won=True, cand=1, opens=1, count=1, rejects=0, rows=1,
              spent=0.1, log_pairs=[(-1.0, -1.0)], arm=0),
    ]
    _write(tmp_path / "a", rows)
    assert len(parity._load_steered(str(tmp_path / "a"))) == 4
