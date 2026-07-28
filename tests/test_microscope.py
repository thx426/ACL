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

from trace_opd.microscope import derive  # noqa: E402
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
