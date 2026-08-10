"""P0b 统计对拍脚本的判定逻辑单测(合成 dump,无 GPU)。

覆盖 codex review P1-1/P1-2 修订后的契约:输入完整性 fail-loud(坏 JSON/
数组长度/非有限值/行级密度界)、任务集合一致性、公共分母(ent_count)、
won 降为参考指标(不进判定)。
"""
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


def _traj(i, *, won, rejects=2, rows=3, spent=3.0, delta=2.0, off=0.5,
          ent_count=100, opens=40, count=10, arm=1, role=None, task=None,
          d_hist=(3, 1)):
    log_pi = [-1.0] * rows
    log_b = [-1.0 + off] * rows          # off > delta 即构造行级界违例
    d = {
        "won": won, "response_token_len": 100 + i,
        "task_id": task or f"t{i % 8}", "global_step": 5,
        "steering": {
            "steer_arm": arm, "steer_record_rows": rows,
            "steer_positions": list(range(3, 3 + rows)),
            "steer_log_pi": log_pi, "steer_log_b": log_b,
            "steer_delta": [delta] * rows,
            "steer_n_candidates": 60, "steer_n_base_open": opens,
            "steer_count": count, "steer_reject_count": rejects,
            "steer_n_gate_bypassed": 0, "steer_n_draft_discarded": 5,
            "steer_budget_spent": spent, "steer_ent_count": ent_count,
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


def _batch(n, **kw):
    won_rate = kw.pop("won_rate", 0.5)
    return [_traj(i, won=(i / n) < won_rate, task=f"t{i}", **kw) for i in range(n)]


def _judge(tmp_path, a, b, **tol):
    _write(tmp_path / "a", a)
    _write(tmp_path / "b", b)
    ra, ea = parity.load_steered(str(tmp_path / "a"))
    rb, eb = parity.load_steered(str(tmp_path / "b"))
    errs = [f"A/{e}" for e in ea + parity.integrity_errors(ra)] \
        + [f"B/{e}" for e in eb + parity.integrity_errors(rb)]
    return parity.compare(parity.summarize(ra), parity.summarize(rb), errs,
                          tol_rate=tol.get("tol_rate", 0.03),
                          tol_hist=tol.get("tol_hist", 0.10))


def test_pass_on_statistically_identical_batches(tmp_path):
    assert _judge(tmp_path, _batch(20), _batch(20)) == []


def test_won_divergence_alone_does_not_fail(tmp_path):
    # won 是低功效参考指标(codex P1-2):单独的 won 差异不判 FAIL
    fails = _judge(tmp_path, _batch(20, won_rate=0.2), _batch(20, won_rate=0.8))
    assert fails == []


def test_mechanism_rate_divergence_fails(tmp_path):
    fails = _judge(tmp_path, _batch(20, rejects=0), _batch(20, rejects=3))
    assert any("reject_rate" in f for f in fails)


def test_row_level_density_bound_violation_fails(tmp_path):
    # 行级界:|log_b−log_pi|=1.5 > 该行 delta=1.0(轨迹级 cap 更宽也不豁免)
    bad = _batch(5, delta=1.0, off=1.5)
    fails = _judge(tmp_path, _batch(5), bad, tol_rate=1.0, tol_hist=1.0)
    assert any("density-ratio bound" in f for f in fails)


def test_corrupt_json_and_array_length_fail_loud(tmp_path):
    _write(tmp_path / "a", _batch(4))
    (tmp_path / "a" / "zz_bad.json").write_text("{not json")
    ra, ea = parity.load_steered(str(tmp_path / "a"))
    assert len(ra) == 4 and any("unreadable JSON" in e for e in ea)

    broken = _batch(4)
    broken[0]["steering"]["steer_log_b"] = [-1.0]     # 长度 != record_rows
    errs = parity.integrity_errors([dict(t, _file=f"f{i}") for i, t in enumerate(broken)])
    assert any("length" in e for e in errs)


def test_task_multiset_mismatch_fails(tmp_path):
    b = _batch(20)
    b[0]["task_id"] = "alien-task"
    fails = _judge(tmp_path, _batch(20), b)
    assert any("task multiset" in f for f in fails)


def test_loader_excludes_research_and_bare_rows(tmp_path):
    rows = _batch(4) + [
        _traj(90, won=True, role="research"),
        _traj(91, won=True, arm=0),
    ]
    _write(tmp_path / "a", rows)
    got, errs = parity.load_steered(str(tmp_path / "a"))
    assert len(got) == 4 and errs == []
