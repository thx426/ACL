"""OPD-KL thought/action span decomposition (scripts/opd_kl_span_decomp.py):
classification boundary, SP segment reassembly against the observation.py
npz contract (REAL field set and step_00004 dir names — codex impl-review
P1-2/P1-3/P1-4), row identity, mask/coverage filtering."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location(
        "opd_kl_span_decomp", _ROOT / "scripts" / "opd_kl_span_decomp.py")
    m = importlib.util.module_from_spec(spec)
    saved = sys.modules.get("opd_kl_span_decomp")
    sys.modules["opd_kl_span_decomp"] = m
    spec.loader.exec_module(m)
    yield m
    if saved is None:
        sys.modules.pop("opd_kl_span_decomp", None)
    else:
        sys.modules["opd_kl_span_decomp"] = saved


# One synthetic vocabulary: each id decodes to a fixed string, so the span
# text and per-token char offsets are exact and human-checkable.
_VOCAB = {
    1: "<reasoning>", 2: "look around", 3: "</reasoning>",
    4: "<action>", 5: "go north", 6: "</action>", 7: "obs:",
}


def _decode(tid):
    return _VOCAB[int(tid)]


def test_step_dir_regex_matches_writer_format(mod):
    # codex impl-review P1-2: the writer emits f"step_{step:05d}" (leading
    # zeros are consumed by the pattern; the captured group is the bare int).
    assert int(mod._STEP_DIR.match("step_00004").group(1)) == 4
    assert mod._STEP_DIR.match("step4")            # legacy/synthetic tolerated
    assert mod._STEP_DIR.match("step_backup") is None
    assert mod._STEP_DIR.match("stepX") is None


def test_classify_span_boundary_inclusive_of_close_tag(mod):
    span = [1, 2, 3, 4, 5, 6]
    cats = mod.classify_span_tokens(span, _decode)
    # <reasoning>, narration, </reasoning> -> thought; <action>.. -> action.
    assert cats == ["thought"] * 3 + ["action"] * 3


def test_classify_span_malformed_defaults(mod):
    # No close tag but an action tag: boundary at the tag start.
    assert mod.classify_span_tokens([2, 4, 5], _decode) == \
        ["thought", "action", "action"]
    # Neither tag (fully malformed narration turn): all thought.
    assert mod.classify_span_tokens([2, 2], _decode) == ["thought", "thought"]


def _npz(dirpath, rank, *, row_hash, resp_ids, token_loss, seg_start,
         turn_spans, won, role, row_uid="", response_mask=None, task="t0"):
    """Mirror of observation.py's per-row array layout (single-row pack)."""
    resp_ids = np.asarray(resp_ids, dtype=np.int32)
    token_loss = np.asarray(token_loss, dtype=np.float16)
    if response_mask is None:
        response_mask = np.ones(len(resp_ids), dtype=np.int8)
    spans_flat = [v for ab in turn_spans for v in ab]
    arrays = {
        "token_loss": token_loss,
        "response_ids": resp_ids,
        "response_mask": np.asarray(response_mask, dtype=np.int8),
        "realized_student_lp": np.zeros(len(resp_ids), dtype=np.float16),
        "row_hash": np.array([row_hash], dtype=np.int64),
        "row_uid": np.array([row_uid], dtype="U"),
        "row_won": np.array([won], dtype=np.int8),
        "row_reward": np.array([float(won)], dtype=np.float32),
        "row_task": np.array([task], dtype="U"),
        "row_role": np.array([role], dtype="U"),
        "seg_start": np.array([seg_start], dtype=np.int32),
        "seg_len": np.array([len(token_loss)], dtype=np.int32),
        "resp_len": np.array([len(resp_ids)], dtype=np.int32),
        "seg_offsets": np.array([0, len(token_loss)], dtype=np.int64),
        "resp_offsets": np.array([0, len(resp_ids)], dtype=np.int64),
        "turn_spans": np.array(spans_flat, dtype=np.int32),
        "turn_spans_offsets": np.array([0, len(spans_flat)], dtype=np.int64),
        "sp_info": np.array([rank, 2], dtype=np.int32),
        "global_step": np.array([4], dtype=np.int64),
    }
    np.savez_compressed(dirpath / f"rank{rank}.npz", **arrays)


def test_sp_reassembly_and_lost_bare_aggregation(mod, tmp_path):
    step = tmp_path / "step_00004"          # real writer dir name
    step.mkdir()
    # Lost bare row split across two ranks: span [0,6) is one assistant turn
    # (3 thought tokens with KL 2.0 each, 3 action tokens with KL 0.5 each);
    # token 6 sits outside any span and must be ignored.
    ids = [1, 2, 3, 4, 5, 6, 7]
    _npz(step, 0, row_hash=11, resp_ids=ids, token_loss=[2.0, 2.0, 2.0, 0.5],
         seg_start=0, turn_spans=[(0, 6)], won=0, role="bare", row_uid="fp:0")
    _npz(step, 1, row_hash=11, resp_ids=ids, token_loss=[0.5, 0.5, 9.0],
         seg_start=4, turn_spans=[(0, 6)], won=0, role="bare", row_uid="fp:0")
    # A won row and a tg row: both excluded from the lost-bare readout.
    _npz(step, 2, row_hash=22, resp_ids=ids, token_loss=[9.0] * 7,
         seg_start=0, turn_spans=[(0, 6)], won=1, role="bare", row_uid="fp:1")
    _npz(step, 3, row_hash=33, resp_ids=ids, token_loss=[9.0] * 7,
         seg_start=0, turn_spans=[(0, 6)], won=0, role="tg", row_uid="fp:2")

    rows = mod.load_step(step)
    assert set(rows) == {"fp:0", "fp:1", "fp:2"}
    np.testing.assert_allclose(
        rows["fp:0"]["token_loss"], [2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 9.0])
    assert rows["fp:0"]["covered"].all()

    summ = mod.step_summary(rows, _decode)
    assert summ["lost_rows"] == 1 and summ["partial_rows_excluded"] == 0
    assert summ["thought_tokens"] == 3 and summ["action_tokens"] == 3
    assert summ["thought_kl_per_token"] == pytest.approx(2.0)
    assert summ["action_kl_per_token"] == pytest.approx(0.5)
    assert summ["thought_kl_share"] == pytest.approx(6.0 / 7.5)
    assert summ["thought_token_share"] == pytest.approx(0.5)


def test_identical_responses_stay_distinct_rows_via_uid(mod, tmp_path):
    # codex impl-review P1-3: same resp_hash (identical short responses) must
    # NOT merge — row_uid separates them.
    step = tmp_path / "step_00006"
    step.mkdir()
    ids = [1, 2, 3, 4, 5, 6]
    _npz(step, 0, row_hash=77, resp_ids=ids, token_loss=[1.0] * 6,
         seg_start=0, turn_spans=[(0, 6)], won=0, role="bare", row_uid="fp:0")
    _npz(step, 1, row_hash=77, resp_ids=ids, token_loss=[3.0] * 6,
         seg_start=0, turn_spans=[(0, 6)], won=0, role="bare", row_uid="fp:1")
    rows = mod.load_step(step)
    assert len(rows) == 2
    summ = mod.step_summary(rows, _decode)
    assert summ["lost_rows"] == 2
    assert summ["thought_kl_per_token"] == pytest.approx(2.0)  # mean of 1 & 3


def test_legacy_pack_key_collision_fails_loud(mod, tmp_path):
    # Legacy packs (no row_uid): fallback key is (task, hash); a genuine
    # collision with conflicting metadata must fail, never overwrite.
    step = tmp_path / "step_00007"
    step.mkdir()
    ids = [1, 2, 3, 4, 5, 6]
    kw = dict(resp_ids=ids, token_loss=[1.0] * 6, seg_start=0,
              turn_spans=[(0, 6)], row_uid="")
    _npz(step, 0, row_hash=88, won=0, role="bare", **kw)
    _npz(step, 1, row_hash=88, won=1, role="bare", **kw)   # same key, won flips
    with pytest.raises(SystemExit, match="collides"):
        mod.load_step(step)


def test_mask_and_coverage_filtering(mod, tmp_path):
    # codex impl-review P1-4: masked-out tokens never enter the stats, and a
    # row with an uncovered masked span position is excluded as partial.
    step = tmp_path / "step_00008"
    step.mkdir()
    ids = [1, 2, 3, 4, 5, 6]
    # Row A: action tokens masked out -> only the 3 thought tokens count.
    _npz(step, 0, row_hash=1, resp_ids=ids, token_loss=[2.0] * 6, seg_start=0,
         turn_spans=[(0, 6)], won=0, role="bare", row_uid="fp:0",
         response_mask=[1, 1, 1, 0, 0, 0])
    # Row B: rank file for tokens [3,6) missing -> masked span position
    # uncovered -> partial, excluded and counted.
    _npz(step, 1, row_hash=2, resp_ids=ids, token_loss=[9.0, 9.0, 9.0],
         seg_start=0, turn_spans=[(0, 6)], won=0, role="bare", row_uid="fp:1")
    rows = mod.load_step(step)
    summ = mod.step_summary(rows, _decode)
    assert summ["lost_rows"] == 1 and summ["partial_rows_excluded"] == 1
    assert summ["thought_tokens"] == 3 and summ["action_tokens"] == 0
    assert summ["thought_kl_per_token"] == pytest.approx(2.0)
