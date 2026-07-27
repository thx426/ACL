"""Numeric truth of the OPD KL-mode family, executed against the VENDORED verl
tree (the code training actually imports), without installing verl.

Technique: the loss modules only need a handful of verl symbols, so the tests
register stub modules and ``exec`` the real source files — the math under test
is the shipped math, not a transcription. The gated mode is additionally pinned
to SEED's own test vectors (refs/SEED tests/trainer/ppo/test_opd_loss.py), so
"we imported their objective" is a checked claim, not a belief.
"""
import ast
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

_VERL = Path(__file__).resolve().parents[1] / "verl" / "verl"


@pytest.fixture()
def fsdp_losses():
    """exec the vendored fsdp losses module with its three imports stubbed."""
    saved = {k: sys.modules.get(k) for k in
             ("verl", "verl.utils", "verl.utils.ulysses", "verl.workers", "verl.workers.config")}

    def _register(name, **attrs):
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod

    _register("verl")
    _register("verl.utils")
    _register("verl.utils.ulysses",
              get_ulysses_sequence_parallel_world_size=lambda: 1,
              slice_input_tensor=lambda t, dim: t)
    _register("verl.workers")
    _register("verl.workers.config", DistillationConfig=object, DistillationLossConfig=object)
    ns = {}
    src = (_VERL / "trainer" / "distillation" / "fsdp" / "losses.py").read_text(encoding="utf-8")
    exec(compile(src, "vendored_fsdp_losses.py", "exec"), ns)
    yield ns
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


class _LossCfg:
    use_chunked_topk = False
    chunked_topk_chunk_size = 4096
    log_prob_min_clamp = -10.0


class _Cfg:
    distillation_loss = _LossCfg()


def _topk_inputs(n_tokens=7, vocab=32, k=4, seed=0, sharpen=1.0):
    g = torch.Generator().manual_seed(seed)
    student_logits = torch.randn(1, n_tokens, vocab, generator=g) * sharpen
    teacher_logits = torch.randn(1, n_tokens, vocab, generator=g) * sharpen
    t_lp, t_ids = torch.topk(F.log_softmax(teacher_logits, dim=-1), k=k, dim=-1)
    return (
        student_logits,
        torch.nested.nested_tensor([t_lp[0]], layout=torch.jagged),
        torch.nested.nested_tensor([t_ids[0]], layout=torch.jagged),
        teacher_logits,
    )


def test_reverse_kl_topk_is_zero_when_student_equals_teacher(fsdp_losses):
    _, nlp, nids, t_logits = _topk_inputs(seed=1)
    out = fsdp_losses["compute_reverse_kl_topk"](t_logits, nlp, nids, _Cfg(), "thd")
    assert out["distillation_losses"].abs().max().item() < 1e-5


def test_reverse_kl_topk_matches_hand_computed_probability_arithmetic(fsdp_losses):
    sl = torch.tensor([[[0.5, -0.2, 1.0, 0.0, -1.0, 0.3, 2.0, -0.5]]])
    tl = torch.tensor([[[1.5, 0.1, -0.3, 0.2, 0.0, -2.0, 0.4, 0.9]]])
    t_lp_k, t_id_k = torch.topk(F.log_softmax(tl, dim=-1), k=3, dim=-1)
    out = fsdp_losses["compute_reverse_kl_topk"](
        sl,
        torch.nested.nested_tensor([t_lp_k[0]], layout=torch.jagged),
        torch.nested.nested_tensor([t_id_k[0]], layout=torch.jagged),
        _Cfg(), "thd")
    sk = F.log_softmax(sl, dim=-1)[0, 0, t_id_k[0, 0]]
    tk = t_lp_k[0, 0]
    m_s, m_t = sk.exp().sum(), tk.exp().sum()
    expected = (sk.exp() * (sk - tk)).sum() + (1 - m_s) * (torch.log1p(-m_s) - torch.log1p(-m_t))
    assert torch.allclose(out["distillation_losses"][0, 0], expected, atol=1e-5)
    assert torch.allclose(out["student_mass"][0, 0], m_s, atol=1e-6)


def test_reverse_kl_topk_is_a_true_kl_hence_non_negative(fsdp_losses):
    for seed in range(20):
        s, nlp, nids, _ = _topk_inputs(seed=seed, sharpen=2.0)
        out = fsdp_losses["compute_reverse_kl_topk"](s, nlp, nids, _Cfg(), "thd")
        assert out["distillation_losses"].min().item() > -1e-6


def test_reverse_kl_topk_tail_bucket_carries_the_out_of_support_gradient(fsdp_losses):
    """The whole reason the tail bucket exists: student mass parked outside the
    teacher's top-k must be pushed down, which a support-restricted sum cannot
    express. Teacher concentrates on tokens 0..2, student on token 6."""
    tl = torch.zeros(1, 1, 8); tl[0, 0, :3] = 5.0
    sl = torch.zeros(1, 1, 8); sl[0, 0, 6] = 6.0
    sl.requires_grad_(True)
    t_lp_k, t_id_k = torch.topk(F.log_softmax(tl, dim=-1), k=3, dim=-1)
    out = fsdp_losses["compute_reverse_kl_topk"](
        sl,
        torch.nested.nested_tensor([t_lp_k[0]], layout=torch.jagged),
        torch.nested.nested_tensor([t_id_k[0]], layout=torch.jagged),
        _Cfg(), "thd")
    out["distillation_losses"].sum().backward()
    assert sl.grad[0, 0, 6].item() > 0          # out-of-support: pushed down
    assert sl.grad[0, 0, :3].sum().item() < 0   # teacher's support: pulled up


def test_reverse_kl_topk_chunked_path_matches_the_default_path(fsdp_losses):
    class _ChunkCfg:
        class distillation_loss:
            use_chunked_topk = True
            chunked_topk_chunk_size = 3
            log_prob_min_clamp = -10.0

    s, nlp, nids, _ = _topk_inputs(seed=7)
    dense = fsdp_losses["compute_reverse_kl_topk"](s.clone(), nlp, nids, _Cfg(), "thd")
    chunked = fsdp_losses["compute_reverse_kl_topk"](s.clone(), nlp, nids, _ChunkCfg(), "thd")
    assert torch.allclose(dense["distillation_losses"], chunked["distillation_losses"], atol=1e-5)


def test_reverse_direction_differs_from_forward_on_the_same_data(fsdp_losses):
    s, nlp, nids, _ = _topk_inputs(seed=7)
    rev = fsdp_losses["compute_reverse_kl_topk"](s.clone(), nlp, nids, _Cfg(), "thd")
    fwd = fsdp_losses["compute_forward_kl_topk"](s.clone(), nlp, nids, _Cfg(), "thd")
    assert not torch.allclose(rev["distillation_losses"], fwd["distillation_losses"], atol=1e-3)


# ------------------------------------------------------------- gated (SDAR/SEED)


def _extract(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns[name]


def test_gated_logp_shift_reproduces_seed_golden_vectors():
    """The vectors, expected loss, and expected GRADIENT from SEED's own
    tests/trainer/ppo/test_opd_loss.py — bit-for-bit. If this fails, we are no
    longer training the objective the paper trained."""
    gated = _extract(_VERL / "trainer" / "distillation" / "losses.py", "_gated_logp_shift")

    log_prob = torch.tensor([[-2.0, -1.0], [-0.5, -0.25]], requires_grad=True)
    teacher = torch.tensor([[-1.0, -2.0], [0.0, 0.0]])
    losses, gate, gap = gated(log_prob, teacher, gate_beta=1.0)

    expected_gate = torch.sigmoid(teacher[0] - log_prob.detach()[0])
    expected_loss = expected_gate * (teacher[0] - log_prob.detach()[0])
    assert torch.allclose(losses[0], expected_loss, atol=1e-6)
    assert torch.allclose(gate[0], expected_gate, atol=1e-6)
    assert not gate.requires_grad  # their gate is detached; so is ours

    losses[0].mean().backward()
    assert torch.allclose(log_prob.grad[0], -expected_gate / 2, atol=1e-6)
    assert torch.allclose(log_prob.grad[1], torch.zeros(2))


def test_gated_beta5_endorses_and_rejects_as_the_papers_describe():
    gated = _extract(_VERL / "trainer" / "distillation" / "losses.py", "_gated_logp_shift")
    # token 0: teacher far above student -> gate ~ 1; token 1: rejected -> ~ 0.
    _, gate, _ = gated(torch.tensor([[-2.0, -0.1]]), torch.tensor([[-0.1, -2.0]]), 5.0)
    assert gate[0, 0].item() > 0.99
    assert gate[0, 1].item() < 0.01


def test_realized_token_scoring_equals_log_softmax_gather():
    realized = _extract(_VERL / "trainer" / "distillation" / "teacher_topk_fsdp.py",
                        "_score_rows_realized")
    g = torch.Generator().manual_seed(3)
    rows = torch.randn(9, 40, generator=g)
    resp = torch.randint(0, 40, (9,), generator=g)
    ids, lps = realized(rows, resp, chunk=4)
    want = F.log_softmax(rows.float(), dim=-1).gather(1, resp.unsqueeze(-1))
    assert torch.allclose(lps, want, atol=1e-5)
    assert (ids.squeeze(-1) == resp.to(torch.int32)).all()
    assert tuple(lps.shape) == (9, 1) and tuple(ids.shape) == (9, 1)
