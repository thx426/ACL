"""Distillation microscope: the end-to-end truth chain.

The load-bearing test builds a synthetic batch, runs it through the VENDORED
collection path (observation.py flush), reads the pack back (reader), derives
KL from the quartet (derive), and pins it against BOTH the vendored training
function's output and the pack's stored truth anchor — data-layer correctness
as a checked property, not a belief."""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from trace_opd.microscope import derive, segments  # noqa: E402
from trace_opd.microscope.reader import StepPack, list_steps  # noqa: E402

_VERL = Path(__file__).resolve().parents[1] / "verl" / "verl"


def _exec_vendored(rel, stubs, package=None):
    saved = {k: sys.modules.get(k) for k in stubs}
    for name, attrs in stubs.items():
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
    ns = {}
    if package:
        # relative imports inside the vendored file resolve against this
        ns["__name__"] = package + ".exec"
        ns["__package__"] = package
    try:
        src = (_VERL / rel).read_text(encoding="utf-8")
        exec(compile(src, str(rel), "exec"), ns)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return ns


@pytest.fixture()
def fsdp_ns(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACE_OBS_DIR", str(tmp_path / "obs"))
    obs_ns = _exec_vendored("trainer/distillation/observation.py", {})
    # register the package chain FIRST: the fsdp module's lazy relative import
    # (`from ..observation import ...`) resolves through sys.modules
    obs_mod = types.ModuleType("verl.trainer.distillation.observation")
    for k, v in obs_ns.items():
        setattr(obs_mod, k, v)
    chain = ["verl", "verl.trainer", "verl.trainer.distillation",
             "verl.trainer.distillation.fsdp"]
    saved_chain = {name: sys.modules.get(name) for name in chain}
    for name in chain:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["verl.trainer.distillation.observation"] = obs_mod
    stubs = {
        "verl.utils": {},
        "verl.utils.ulysses": {
            "get_ulysses_sequence_parallel_world_size": lambda: 1,
            "slice_input_tensor": lambda t, dim: t,
        },
        "verl.workers": {},
        "verl.workers.config": {"DistillationConfig": object, "DistillationLossConfig": object},
    }
    ns = _exec_vendored(
        "trainer/distillation/fsdp/losses.py", stubs,
        package="verl.trainer.distillation.fsdp",
    )
    yield ns, obs_ns, tmp_path / "obs"
    sys.modules.pop("verl.trainer.distillation.observation", None)
    for name, mod in saved_chain.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


class _Cfg:
    class distillation_loss:
        use_chunked_topk = False
        chunked_topk_chunk_size = 4096
        log_prob_min_clamp = -10.0


def _mk_batch(n_tokens=9, vocab=48, k=8, seed=3):
    g = torch.Generator().manual_seed(seed)
    student_logits = torch.randn(1, n_tokens, vocab, generator=g) * 1.5
    teacher_logits = torch.randn(1, n_tokens, vocab, generator=g) * 1.5
    t_lp, t_ids = torch.topk(F.log_softmax(teacher_logits, dim=-1), k=k, dim=-1)
    nlp = torch.nested.nested_tensor([t_lp[0]], layout=torch.jagged)
    nids = torch.nested.nested_tensor([t_ids[0]], layout=torch.jagged)
    return student_logits, nlp, nids


def test_end_to_end_pack_roundtrip_pins_derive_against_the_vendored_loss(fsdp_ns):
    ns, obs_ns, obs_dir = fsdp_ns
    student_logits, nlp, nids = _mk_batch()
    out = ns["compute_forward_kl_topk"](student_logits, nlp, nids, _Cfg(), "thd")
    obs = out["obs"]
    assert obs is not None, "TRACE_OBS_DIR is set — the payload must be collected"

    R = student_logits.shape[1]
    resp_ids = np.arange(R, dtype=np.int32) % 48
    realized = np.random.RandomState(0).randn(R).astype(np.float32)
    row = {
        "resp_hash": 424242, "task_id": "toy/task", "sample_role": "bare",
        "won": 1, "reward": 1.0, "turn_spans": [[0, 4], [6, R]],
        "response_ids": resp_ids, "response_mask": np.ones(R, dtype=np.int8),
        "realized_student_lp": realized, "seg_start": 0, "seg_len": R,
        "loss_mode": "forward_kl_topk", "log_prob_min_clamp": -10.0,
    }
    for key in ("teacher_ids", "teacher_lps", "student_ids", "student_lps",
                "student_support_lps"):
        row[key] = obs[key][0].float().numpy()
    row["token_loss"] = out["distillation_losses"][0].detach().numpy()

    obs_ns["_flush_step"](12, [row])
    assert list_steps(obs_dir) == [12]
    pack = StepPack(obs_dir, 12)
    got = pack.row(424242)
    assert got["task_id"] == "toy/task"
    assert got["covered"].all()
    assert got["turn_spans"] == [[0, 4], [6, R]]

    # THE chain: derive(quartet) == vendored loss == stored truth anchor
    rederived = derive.forward_kl(got)
    vendored = out["distillation_losses"][0].detach().numpy()
    assert np.allclose(rederived, vendored, atol=5e-2)  # fp16 pack round-trip
    audit = derive.audit_row(got, loss_mode=pack.loss_mode)
    assert audit["ok"] and audit["checked"] == R

    # spot-check a couple of derived tracks against direct computation
    tr = derive.tracks(got)
    want_t_mass = np.exp(got["teacher_lps"]).sum(-1)
    assert np.allclose(tr["teacher_mass"], want_t_mass, atol=1e-3)
    assert (tr["overlap_count"] >= 0).all() and (tr["overlap_count"] <= 8).all()


def test_reverse_kl_derivation_matches_the_vendored_reverse_function(fsdp_ns):
    ns, obs_ns, _ = fsdp_ns
    student_logits, nlp, nids = _mk_batch(seed=11)
    out = ns["compute_reverse_kl_topk"](student_logits, nlp, nids, _Cfg(), "thd")
    obs = out["obs"]
    row = {k: obs[k][0].float().numpy() for k in (
        "teacher_ids", "teacher_lps", "student_ids", "student_lps", "student_support_lps")}
    rederived = derive.reverse_kl(row)
    vendored = out["distillation_losses"][0].detach().numpy()
    assert np.allclose(rederived, vendored, atol=1e-4)  # no fp16 round-trip here


def test_sp_segments_reassemble_across_ranks(fsdp_ns, tmp_path):
    _, obs_ns, obs_dir = fsdp_ns
    R, K = 10, 4
    rng = np.random.RandomState(7)
    full = {
        "teacher_ids": rng.randint(0, 99, (R, K)).astype(np.int32),
        "teacher_lps": -rng.rand(R, K).astype(np.float16),
        "student_ids": rng.randint(0, 99, (R, K)).astype(np.int32),
        "student_lps": -rng.rand(R, K).astype(np.float16),
        "student_support_lps": -rng.rand(R, K).astype(np.float16),
    }
    loss = -rng.rand(R).astype(np.float16)
    base = {
        "resp_hash": 7, "task_id": "t", "sample_role": "bare", "won": 0, "reward": 0.0,
        "turn_spans": [], "response_ids": np.arange(R, dtype=np.int32),
        "response_mask": np.ones(R, dtype=np.int8),
        "realized_student_lp": np.zeros(R, dtype=np.float16),
    }
    import trace_opd.microscope.reader as reader_mod

    # two "ranks" write disjoint halves of the same row
    for rank, (a, b) in enumerate(((0, 6), (6, R))):
        row = dict(base, seg_start=a, seg_len=b - a, sp_rank=rank, sp_size=2)
        for key, arr in full.items():
            row[key] = arr[a:b]
        row["token_loss"] = loss[a:b]

        class _Dist:
            @staticmethod
            def is_initialized():
                return True

            @staticmethod
            def get_rank():
                return rank

        obs_ns["torch"].distributed = _Dist  # route each flush to its own file
        obs_ns["_flush_step"](3, [row])
    pack = StepPack(obs_dir, 3)
    got = pack.row(7)
    assert got["covered"].all()
    for key, arr in full.items():
        assert np.array_equal(got[key].astype(arr.dtype), arr)


def test_repeat_positions_flags_recurring_ngrams():
    ids = np.array([1, 2, 3, 4, 9, 1, 2, 3, 4, 8, 1, 2, 3, 4], dtype=np.int64)
    hits = derive.repeat_positions({"response_ids": ids}, n=4, min_repeats=2)
    assert hits[5] and hits[10] and not hits[0]

def test_reverse_tail_floor_clamps_both_sides():
    """codex review P1-15: a 0.99999-mass teacher against a 0.5-mass student
    put the old derive ~0.756 off the training value (teacher tail log was
    unclamped). Reproduce the construction and pin the parity."""
    K = 4
    t_lps = np.log(np.full((1, K), 0.99999 / K, dtype=np.float64))
    s_lps = np.log(np.full((1, K), 0.5 / K, dtype=np.float64))
    row = {"teacher_lps": t_lps, "student_support_lps": s_lps}
    got = derive.reverse_kl(row, clamp=-10.0)[0]
    # hand computation with BOTH tails floored at -10 (the vendored rule)
    member = (0.5 * (np.log(0.5 / K) - np.log(0.99999 / K)) * K * (0.5 / K) * K)
    member = (np.exp(s_lps) * (s_lps - t_lps)).sum()
    s_tail, t_tail = 0.5, max(1 - 0.99999, 1e-6)
    want = member + s_tail * (max(np.log(s_tail), -10.0) - max(np.log(t_tail), -10.0))
    assert abs(got - want) < 1e-6


def test_audit_reports_failure_on_empty_data():
    """codex review P1-16: zero checked tokens must never read as OK."""
    R = 4
    row = {
        "response_mask": np.zeros(R, dtype=bool),
        "covered": np.zeros(R, dtype=bool),
        "token_loss": np.full(R, np.nan),
    }
    assert derive.audit_row(row)["ok"] is False


def test_segment_spans_maps_tags_across_piece_boundaries():
    """The tag text survives arbitrary BPE cuts — the scan is character-level.
    Layout: <reasoning>AB</reasoning><action>go</action> over 7 pieces with
    both tags split mid-marker."""
    pieces = ["<rea", "soning>", "AB", "</reasoning>", "<act", "ion>go", "</action>"]
    seg = segments.segment_spans(pieces)
    assert seg["reasoning"] == [[2, 3]]
    # piece 5 ("ion>go") straddles the open marker and the content — the
    # overlap rule admits it into the segment (±1-token contract)
    assert seg["action"] == [[5, 6]]
    assert seg["closings"]["reasoning"] == [3]
    assert seg["closings"]["action"] == [6]
    assert [0, 2] in seg["structure"] and [3, 4] in seg["structure"]


def test_segment_spans_unclosed_tag_runs_to_end_without_closing():
    pieces = ["<reasoning>", "still", " thinking", " forever"]
    seg = segments.segment_spans(pieces)
    assert seg["reasoning"] == [[1, 4]]
    assert seg["closings"]["reasoning"] == []


def test_aggregate_segments_budget_share_and_clamp_rate():
    R = 6
    row = {
        "response_ids": np.arange(R),
        "response_mask": np.ones(R, dtype=bool),
        "covered": np.ones(R, dtype=bool),
        # positions 0-1: both support entries at the clamp; others well above
        "student_support_lps": np.array(
            [[-10.0, -10.0], [-10.0, -10.0], [-1.0, -1.0],
             [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]]
        ),
    }
    tr = {
        "kl_forward": np.array([1.0, 3.0, 2.0, 2.0, np.nan, 4.0]),
        "student_entropy": np.full(R, 0.5),
        "teacher_entropy": np.full(R, 0.25),
        "teacher_mass": np.full(R, 0.9),
        "token_loss": np.array([1.0, 3.0, 2.0, 2.0, np.nan, 4.0]),
    }
    seg = {"reasoning": [[0, 2]], "action": [[2, 6]]}
    agg = segments.aggregate_segments(row, tr, seg, clamp=-10.0)
    # total finite masked KL = 1+3+2+2+4 = 12
    assert agg["reasoning"]["tokens"] == 2
    assert abs(agg["reasoning"]["kl_mean"] - 2.0) < 1e-9
    assert abs(agg["reasoning"]["kl_share"] - 4.0 / 12.0) < 1e-9
    assert abs(agg["reasoning"]["clamp_hit_rate"] - 1.0) < 1e-9
    assert agg["action"]["tokens"] == 4  # NaN position still counts as a token
    assert abs(agg["action"]["kl_mean"] - 8.0 / 3.0) < 1e-9
    assert abs(agg["action"]["kl_share"] - 8.0 / 12.0) < 1e-9
    assert abs(agg["action"]["clamp_hit_rate"] - 0.0) < 1e-9


def test_closing_token_stats_reads_both_sides_and_flags_topk_miss():
    R = 5
    row = {
        "response_ids": np.arange(R),
        "response_mask": np.ones(R, dtype=bool),
        "covered": np.ones(R, dtype=bool),
        "realized_student_lp": np.array([-9.0, -0.5, -9.0, -0.7, -9.0]),
    }
    tr = {"realized_teacher_lp": np.array([np.nan, -1.5, np.nan, np.nan, np.nan])}
    seg = {"closings": {"reasoning": [1, 3]}}
    stats = segments.closing_token_stats(row, tr, seg)
    assert stats[0] == {"pos": 1, "teacher_lp": -1.5, "student_lp": -0.5}
    # teacher pushed the closing token out of its top-K → None, the loud signal
    assert stats[1]["teacher_lp"] is None and stats[1]["student_lp"] == -0.7


def test_reader_refuses_a_row_key_collision(fsdp_ns):
    """codex review P1-14: same (hash, length) with different tokens must
    raise, never silently merge."""
    _, obs_ns, obs_dir = fsdp_ns
    R, K = 4, 2
    base = {
        "resp_hash": 99, "task_id": "t", "sample_role": "bare", "won": 0,
        "reward": 0.0, "turn_spans": [], "seg_start": 0, "seg_len": R,
        "response_mask": np.ones(R, dtype=np.int8),
        "realized_student_lp": np.zeros(R, dtype=np.float16),
        "token_loss": np.zeros(R, dtype=np.float16),
    }
    for key in ("teacher_ids", "student_ids"):
        base[key] = np.zeros((R, K), dtype=np.int32)
    for key in ("teacher_lps", "student_lps", "student_support_lps"):
        base[key] = np.zeros((R, K), dtype=np.float16)

    class _D:
        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_rank():
            return 0

    obs_ns["torch"].distributed = _D
    obs_ns["_flush_step"](5, [dict(base, response_ids=np.array([1, 2, 3, 4], dtype=np.int32))])

    class _D1(_D):
        @staticmethod
        def get_rank():
            return 1

    obs_ns["torch"].distributed = _D1
    obs_ns["_flush_step"](5, [dict(base, response_ids=np.array([9, 8, 7, 6], dtype=np.int32))])
    with pytest.raises(RuntimeError, match="collision"):
        StepPack(obs_dir, 5)



def test_audit_verdict_is_fraction_based_not_max_based():
    """CD-006 req 3-V: a handful of fp16-tail-cancellation tokens in a long
    row is a storage artefact — the row passes on exceed FRACTION, with the
    count reported; a broadly wrong row still fails."""
    R, K = 1000, 4
    rng = np.random.RandomState(0)
    lps = np.log(rng.dirichlet(np.ones(8), size=R)[:, :K]).astype(np.float64)
    row = {
        "response_mask": np.ones(R, dtype=bool),
        "covered": np.ones(R, dtype=bool),
        "teacher_lps": lps,
        "student_support_lps": lps - 0.1,
    }
    truth = derive.forward_kl(row)
    # 3/1000 tokens off by a lot (cancellation-shaped): row must PASS
    bad3 = truth.copy(); bad3[[10, 500, 900]] += 1.0
    res = derive.audit_row({**row, "token_loss": bad3})
    assert res["ok"] and res["n_exceed"] == 3
    assert res["exceed_frac"] == pytest.approx(0.003)
    assert res["max_err"] > 0.5  # max alone would have failed it
    # 20/1000 off: broadly wrong, must FAIL
    bad20 = truth.copy(); bad20[:20] += 1.0
    res = derive.audit_row({**row, "token_loss": bad20})
    assert not res["ok"] and res["n_exceed"] == 20


def test_sample_summary_light_path_matches_tracks(fsdp_ns):
    """CD-006 req 3-Z: the listing summary no longer runs the full tracks()
    sweep — pin the light path against the heavy one on a real pack row."""
    ns, obs_ns, obs_dir = fsdp_ns
    student_logits, nlp, nids = _mk_batch(seed=5)
    out = ns["compute_forward_kl_topk"](student_logits, nlp, nids, _Cfg(), "thd")
    obs = out["obs"]
    R = student_logits.shape[1]
    row = {
        "resp_hash": 7, "step": 1, "task_id": "t", "sample_role": "bare",
        "won": 0, "reward": 0.0, "turn_spans": [[0, R]],
        "response_ids": np.arange(R, dtype=np.int64) % 48,
        "response_mask": np.ones(R, dtype=bool),
        "covered": np.ones(R, dtype=bool),
        "realized_student_lp": -np.random.RandomState(1).rand(R).astype(np.float32),
    }
    for key in ("teacher_ids", "teacher_lps", "student_ids", "student_lps",
                "student_support_lps"):
        row[key] = obs[key][0].float().numpy()
    row["token_loss"] = out["distillation_losses"][0].detach().numpy()

    summary = derive.sample_summary(row)
    tr = derive.tracks(row)
    m = row["response_mask"] & row["covered"]
    assert summary["kl_mean"] == pytest.approx(float(tr["kl_forward"][m].mean()), abs=5e-2)
    assert summary["overlap_mean"] == pytest.approx(float(tr["overlap_count"][m].mean()))
    assert summary["teacher_entropy_mean"] == pytest.approx(
        float(tr["teacher_entropy"][m].mean()), abs=1e-4)
    gap = tr["gap"][m]
    assert summary["gap_mean"] == pytest.approx(
        float(gap[np.isfinite(gap)].mean()), abs=1e-4)
