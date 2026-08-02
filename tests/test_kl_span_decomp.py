"""OPD-KL thought/action span decomposition (scripts/opd_kl_span_decomp.py):
classification boundary, SP segment reassembly against the observation.py
npz contract, and the lost-bare aggregation filter."""
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
         turn_spans, won, role):
    resp_ids = np.asarray(resp_ids, dtype=np.int32)
    token_loss = np.asarray(token_loss, dtype=np.float16)
    spans_flat = [v for ab in turn_spans for v in ab]
    arrays = {
        "token_loss": token_loss,
        "response_ids": resp_ids,
        "row_hash": np.array([row_hash], dtype=np.int64),
        "row_won": np.array([won], dtype=np.int8),
        "row_role": np.array([role], dtype="U"),
        "seg_start": np.array([seg_start], dtype=np.int32),
        "seg_len": np.array([len(token_loss)], dtype=np.int32),
        "resp_len": np.array([len(resp_ids)], dtype=np.int32),
        "seg_offsets": np.array([0, len(token_loss)], dtype=np.int64),
        "resp_offsets": np.array([0, len(resp_ids)], dtype=np.int64),
        "turn_spans": np.array(spans_flat, dtype=np.int32),
        "turn_spans_offsets": np.array([0, len(spans_flat)], dtype=np.int64),
    }
    np.savez_compressed(dirpath / f"rank{rank}.npz", **arrays)


def test_sp_reassembly_and_lost_bare_aggregation(mod, tmp_path):
    step = tmp_path / "step4"
    step.mkdir()
    # Lost bare row split across two ranks: span [0,6) is one assistant turn
    # (3 thought tokens with KL 2.0 each, 3 action tokens with KL 0.5 each);
    # token 6 sits outside any span and must be ignored.
    ids = [1, 2, 3, 4, 5, 6, 7]
    _npz(step, 0, row_hash=11, resp_ids=ids, token_loss=[2.0, 2.0, 2.0, 0.5],
         seg_start=0, turn_spans=[(0, 6)], won=0, role="bare")
    _npz(step, 1, row_hash=11, resp_ids=ids, token_loss=[0.5, 0.5, 9.0],
         seg_start=4, turn_spans=[(0, 6)], won=0, role="bare")
    # A won row and a tg row: both excluded from the lost-bare readout.
    _npz(step, 2, row_hash=22, resp_ids=ids, token_loss=[9.0] * 7,
         seg_start=0, turn_spans=[(0, 6)], won=1, role="bare")
    _npz(step, 3, row_hash=33, resp_ids=ids, token_loss=[9.0] * 7,
         seg_start=0, turn_spans=[(0, 6)], won=0, role="tg")

    rows = mod.load_step(step)
    assert set(rows) == {11, 22, 33}
    np.testing.assert_allclose(
        rows[11]["token_loss"], [2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 9.0])

    summ = mod.step_summary(rows, _decode)
    assert summ["lost_rows"] == 1
    assert summ["thought_tokens"] == 3 and summ["action_tokens"] == 3
    assert summ["thought_kl_per_token"] == pytest.approx(2.0)
    assert summ["action_kl_per_token"] == pytest.approx(0.5)
    assert summ["thought_kl_share"] == pytest.approx(6.0 / 7.5)
    assert summ["thought_token_share"] == pytest.approx(0.5)
