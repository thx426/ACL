"""Teacher grounding (tg) arm — the math, the wiring, and the contracts.

Design: docs/teacher-grounding-design.md. Tested against the VENDORED verl
sources (the code training actually imports), same discipline as
test_loss_aggregation.py: the vendored file is the artifact, not a transcript.

What is pinned here:
  1. the tg soft-CE math — forward_kl_topk's per-token formula applied to a
     RENORMALISED bare-policy label is KL(p̄_norm ‖ q_restricted) ≥ 0, with
     equality exactly when q matches the label on its support;
  2. the fsdp scorer's tg contract — bare-prefix scoring, renormalisation
     (proper distribution out), tg_skip dummy rows, fail-loud on a lost
     bare prefix / topk=0 / unknown mode;
  3. the loss call-site wiring — tg term reads tg_mask fail-loud, divides by
     the engine-stamped batch-global tg row count under seq-mean-*, keeps the
     sft-style zero-safe local denominator under token-mean, ships nothing
     when the coefficient is 0;
  4. the engine stamp, TQ rerouting and agent-loop won-split source pins;
  5. trace_opd config validation, conditional fingerprint, launcher pass-through
     and the tg curve key family.
"""
import ast
import contextlib
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_ROOT = Path(__file__).resolve().parents[1]
_VERL = _ROOT / "verl" / "verl"


# ---------------------------------------------------------------------------
# 1. the math: renormalised label + forward_kl_topk formula == tg soft-CE
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def kl_divergence():
    src = (_VERL / "trainer" / "distillation" / "fsdp" / "losses.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "kl_divergence")
    mod = ast.Module(body=[fn], type_ignores=[])
    ns = {"torch": torch}
    exec(compile(mod, "vendored_kl_divergence.py", "exec"), ns)
    return ns["kl_divergence"]


def test_renormalised_label_soft_ce_matches_hand_math_and_is_nonneg(kl_divergence):
    # Un-normalised bare top-3 logprobs (mass 0.9) -> renormalise -> label.
    raw = torch.log(torch.tensor([[0.5, 0.3, 0.1]]))
    label = raw - torch.logsumexp(raw, dim=-1, keepdim=True)
    p = label.exp()
    assert p.sum().item() == pytest.approx(1.0, abs=1e-6)

    # q restricted to the label support, total mass 0.8 (rest of q elsewhere).
    q_lp = torch.log(torch.tensor([[0.4, 0.3, 0.1]]))
    out = kl_divergence(log_q=q_lp, log_p=label).item()
    hand = float((p * (label - q_lp)).sum())
    assert out == pytest.approx(hand, rel=1e-6)
    # Lower bound: -log(restricted mass) > 0, so clamp_min(0) is lossless.
    assert out > 0

    # Equality case: q equals the renormalised label on its support -> 0.
    assert kl_divergence(log_q=label, log_p=label).item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. the fsdp scorer: tg rows are scored under the bare prefix and come back
#    renormalised; skip/fail contracts hold
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def scorer_ns():
    src = (_VERL / "trainer" / "distillation" / "teacher_topk_fsdp.py").read_text(encoding="utf-8")
    ns = {}
    exec(compile(src, "vendored_teacher_topk_fsdp.py", "exec"), ns)
    return ns


class _TinyLM(torch.nn.Module):
    """Deterministic toy LM: logits depend on position AND on the token at the
    position (so different prefixes give different distributions)."""

    VOCAB = 7

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, use_cache=False):
        T = input_ids.shape[1]
        pos = torch.arange(T, dtype=torch.float32).unsqueeze(-1)
        vocab = torch.arange(self.VOCAB, dtype=torch.float32).unsqueeze(0)
        logits = torch.sin(pos * 0.7 + vocab * 1.3) * 2.0
        logits = logits + input_ids[0].unsqueeze(-1).float() * 0.11
        return types.SimpleNamespace(logits=logits.unsqueeze(0))


class _StubEngine:
    def __init__(self):
        self.module = _TinyLM()

    def eval_mode(self, zero_grad_on_exit=False):
        return contextlib.nullcontext()


def _score_batch(scorer_ns, extras, P=4, R=3):
    n = len(extras)
    data = {
        "extra_fields": extras,
        "prompts": torch.arange(n * P).reshape(n, P) % 5,
        "responses": (torch.arange(n * R).reshape(n, R) % 5) + 1,
        "attention_mask": torch.ones(n, P + R, dtype=torch.int64),
    }
    scorer_ns["compute_teacher_topk_fsdp"](data, _StubEngine(), chunk=2)
    return data


def test_scorer_tg_rows_come_back_renormalised_privileged_stay_raw(scorer_ns):
    P, R, K = 4, 3, 4
    extras = [
        {"teacher_mode": "privileged", "teacher_topk": K,
         "teacher_prefix_ids": list(range(6))},
        {"teacher_mode": "tg", "teacher_topk": K,
         "teacher_prefix_ids": list(range(P))},   # bare prompt as tg label prefix
        {"teacher_mode": "tg_skip", "teacher_topk": K},
    ]
    data = _score_batch(scorer_ns, extras, P=P, R=R)
    lps = [t for t in data["teacher_logprobs"].unbind()]
    tg_lps = [t for t in data["tg_label_logprobs"].unbind()]

    # Row layout is (P-1) zeros | R valid | 1 dummy for scored rows.
    priv_valid = lps[0][P - 1 : P - 1 + R]
    tg_valid = tg_lps[1][P - 1 : P - 1 + R]
    # privileged: raw top-K logprobs on the TEACHER channel — truncated mass
    # < 1 with K=4 < V=7 — and a zero stub on the tg channel.
    assert (torch.logsumexp(priv_valid, dim=-1) < -1e-4).all()
    assert tg_lps[0].abs().sum().item() == 0.0
    assert data["tg_label_ids"].unbind()[0].abs().sum().item() == 0.0
    # tg: renormalised proper distribution on the TG LABEL channel — and a
    # zero stub on the teacher channel (decoupling: the OPD formulas must
    # never see a bare-policy distribution wearing a teacher field name).
    assert torch.logsumexp(tg_valid, dim=-1).abs().max().item() < 1e-5
    assert lps[1].abs().sum().item() == 0.0
    assert data["teacher_ids"].unbind()[1].abs().sum().item() == 0.0
    # tg_skip: all-zero dummy rows on both channels, nothing scored.
    assert data["teacher_logprobs"].unbind()[2].abs().sum().item() == 0.0
    assert data["teacher_ids"].unbind()[2].abs().sum().item() == 0.0
    assert tg_lps[2].abs().sum().item() == 0.0


def test_scorer_no_tg_rows_ships_no_tg_fields(scorer_ns):
    # tg-free batches carry zero extra fields — the tg channel exists only
    # when a tg row exists (arm off == byte-identical transfer).
    extras = [{"teacher_mode": "privileged", "teacher_topk": 4,
               "teacher_prefix_ids": list(range(6))}]
    data = _score_batch(scorer_ns, extras)
    assert "tg_label_ids" not in data and "tg_label_logprobs" not in data


def test_scorer_tg_contracts_fail_loud(scorer_ns):
    ok = {"teacher_mode": "tg", "teacher_topk": 4, "teacher_prefix_ids": [0, 1, 2, 3]}
    with pytest.raises(RuntimeError, match="bare prompt"):
        _score_batch(scorer_ns, [dict(ok, teacher_prefix_ids=None)])
    with pytest.raises(RuntimeError, match="topk > 0"):
        _score_batch(scorer_ns, [dict(ok, teacher_topk=0)])
    with pytest.raises(RuntimeError, match="provenance"):
        _score_batch(scorer_ns, [dict(ok, teacher_mode="nonsense")])


# ---------------------------------------------------------------------------
# 3. the call-site wiring (distillation_ppo_loss), vendored losses.py with the
#    same stub harness test_loss_aggregation.py uses
# ---------------------------------------------------------------------------
@pytest.fixture()
def losses_ns():
    stub_names = ["tensordict", "verl", "verl.base_config", "verl.trainer",
                  "verl.trainer.ppo", "verl.trainer.ppo.core_algos",
                  "verl.utils", "verl.utils.metric", "verl.workers",
                  "verl.workers.config", "verl.workers.utils",
                  "verl.workers.utils.losses", "verl.workers.utils.padding"]
    saved = {k: sys.modules.get(k) for k in stub_names}
    calls = []

    def spy_agg_loss(**kwargs):
        calls.append(kwargs)
        return torch.tensor(0.0)

    class _Metric:
        def __init__(self, *a, **k):
            pass

    def _register(name, **attrs):
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod

    _register("tensordict", TensorDict=dict)
    _register("verl")
    _register("verl.base_config", BaseConfig=object)
    _register("verl.trainer")
    _register("verl.trainer.ppo")
    _register("verl.trainer.ppo.core_algos", agg_loss=spy_agg_loss,
              get_policy_loss_fn=lambda *_: None, kl_penalty=None)
    _register("verl.utils")
    _register("verl.utils.metric",
              AggregationType=types.SimpleNamespace(SUM="sum", MEAN="mean",
                                                    MIN="min", MAX="max"),
              Metric=_Metric)
    _register("verl.workers")
    _register("verl.workers.config", ActorConfig=object,
              DistillationConfig=object, DistillationLossConfig=object)
    _register("verl.workers.utils")
    _register("verl.workers.utils.losses", ppo_loss=lambda *a, **k: None)
    _register("verl.workers.utils.padding",
              no_padding_2_padding=lambda t, d: t)

    ns = {}
    src = (_VERL / "trainer" / "distillation" / "losses.py").read_text(encoding="utf-8")
    exec(compile(src, "vendored_distillation_losses.py", "exec"), ns)
    yield ns, calls
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def _call_ppo(ns, tg_coef, agg_mode, data, losses=None, tg_losses=None):
    losses = losses if losses is not None else torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    ns["get_distillation_loss_fn"] = lambda mode: (
        lambda config, distillation_config, model_output, data: (losses, {}))
    config = types.SimpleNamespace(loss_agg_mode=agg_mode, global_batch_info={})
    loss_cfg = types.SimpleNamespace(
        loss_mode="fake", loss_max_clamp=None, use_policy_gradient=False,
        use_task_rewards=False, rl_loss_coef=0.0, sft_loss_coef=0.0,
        research_loss_coef=0.0, tg_loss_coef=tg_coef,
        distillation_loss_coef=1.0, opd_loss_coef=1.0,
    )
    distillation_config = types.SimpleNamespace(distillation_loss=loss_cfg)
    # Decoupled channels: the OPD matrix and the tg matrix are separate
    # model_output entries — exactly what the logits processor now ships.
    model_output = {"distillation_losses": losses}
    if tg_losses is not None:
        model_output["tg_losses"] = tg_losses
    return ns["distillation_ppo_loss"](config, distillation_config, model_output, data)


def test_tg_call_site_fails_loud_without_tg_mask(losses_ns):
    ns, _ = losses_ns
    data = {"response_mask": torch.tensor([[True, True], [False, False]]),
            "batch_num_seqs": 1, "dp_size": 1}
    with pytest.raises(RuntimeError, match="tg_mask"):
        _call_ppo(ns, 1.0, "seq-mean-token-mean", data)


def test_tg_call_site_seq_mean_uses_frequency_weighted_total(losses_ns):
    # User ruling 2026-08-02 (codex P1-1): BOTH terms divide by the TOTAL
    # signal-row count N_lost + N_tg, so each term's effective weight is its
    # batch share (true annealing), not an unconditional conditional mean.
    ns, calls = losses_ns
    tg_mask = torch.tensor([[False, False], [True, True]])
    rmask = torch.tensor([[True, True], [False, False]])
    data = {"response_mask": rmask, "tg_mask": tg_mask,
            "batch_num_seqs": 5, "dp_size": 2, "batch_num_tg_seqs": 3}
    opd_mat = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    tg_mat = torch.tensor([[9.0, 9.0], [0.5, 1.5]])
    _, metrics = _call_ppo(ns, 1.0, "seq-mean-token-mean", data,
                           losses=opd_mat, tg_losses=tg_mat)
    tg_calls = [c for c in calls if c.get("loss_mask") is not None
                and torch.equal(c["loss_mask"], tg_mask)]
    opd_calls = [c for c in calls if c.get("loss_mask") is not None
                 and torch.equal(c["loss_mask"], rmask)]
    assert len(tg_calls) == 1 and len(opd_calls) == 1
    assert tg_calls[0]["global_batch_size"] == 8.0    # 5 lost + 3 tg
    assert opd_calls[0]["global_batch_size"] == 8.0   # same total for OPD
    assert tg_calls[0]["dp_size"] == 2.0
    # Decoupling: each term aggregates its OWN matrix.
    assert torch.equal(tg_calls[0]["loss_mat"], tg_mat)
    assert torch.equal(opd_calls[0]["loss_mat"], opd_mat)
    assert "tg/loss" in metrics and "tg/rows" in metrics and "tg/tokens" in metrics


def test_opd_denominator_unchanged_without_tg_stamp(losses_ns):
    # tg off (no engine stamp): the historical lost-only denominator is
    # byte-identical — frequency weighting must not leak into pure-OPD runs.
    ns, calls = losses_ns
    rmask = torch.tensor([[True, True], [True, False]])
    data = {"response_mask": rmask, "batch_num_seqs": 5, "dp_size": 2}
    _call_ppo(ns, 0.0, "seq-mean-token-mean", data)
    opd_calls = [c for c in calls if c.get("loss_mask") is not None]
    assert opd_calls[0]["global_batch_size"] == 5.0


def test_tg_call_site_seq_mean_fails_loud_without_stamp(losses_ns):
    ns, _ = losses_ns
    data = {"response_mask": torch.tensor([[True, True], [False, False]]),
            "tg_mask": torch.tensor([[False, False], [True, True]]),
            "batch_num_seqs": 5, "dp_size": 2}
    with pytest.raises(RuntimeError, match="batch_num_tg_seqs"):
        _call_ppo(ns, 1.0, "seq-mean-token-mean", data,
                  tg_losses=torch.tensor([[0.0, 0.0], [0.5, 1.5]]))


def test_tg_call_site_token_mean_keeps_local_zero_safe_denominator(losses_ns):
    ns, calls = losses_ns
    tg_mask = torch.tensor([[False, False], [True, True]])
    data = {"response_mask": torch.tensor([[True, True], [False, False]]),
            "tg_mask": tg_mask}
    _call_ppo(ns, 1.0, "token-mean", data,
              tg_losses=torch.tensor([[0.0, 0.0], [0.5, 1.5]]))
    tg_calls = [c for c in calls if c.get("loss_mask") is not None
                and torch.equal(c["loss_mask"], tg_mask)]
    assert len(tg_calls) == 1
    assert int(tg_calls[0]["batch_num_tokens"]) == 2  # the two tg tokens


def test_tg_call_site_coef_zero_ships_nothing(losses_ns):
    ns, calls = losses_ns
    data = {"response_mask": torch.tensor([[True, True], [True, False]])}
    _, metrics = _call_ppo(ns, 0.0, "token-mean", data)  # no tg_mask, no error
    assert "tg/loss" not in metrics
    assert all("tg" not in str(c.get("loss_mask", "")) for c in calls)


def test_tg_reads_dedicated_matrix_and_clamps_min_zero(losses_ns):
    # Decoupling: the tg term reads model_output["tg_losses"], NEVER the OPD
    # matrix; a negative per-token value (fp jitter) is clamped to 0 before
    # aggregation.
    ns, calls = losses_ns
    tg_mask = torch.tensor([[True, True]])
    data = {"response_mask": torch.tensor([[False, False]]),
            "tg_mask": tg_mask, "batch_num_seqs": 1, "dp_size": 1,
            "batch_num_tg_seqs": 1}
    opd_mat = torch.tensor([[9.0, 9.0]])           # poisoned OPD matrix
    tg_mat = torch.tensor([[-0.5, 2.0]])
    _call_ppo(ns, 1.0, "seq-mean-token-mean", data,
              losses=opd_mat, tg_losses=tg_mat)
    tg_calls = [c for c in calls if torch.equal(c.get("loss_mask"), tg_mask)]
    assert torch.equal(tg_calls[0]["loss_mat"], torch.tensor([[0.0, 2.0]]))


def test_tg_fails_loud_without_dedicated_matrix(losses_ns):
    # tg rows in the batch but the logits processor shipped no tg_losses —
    # a broken invariant, never a silent fallback to the OPD matrix.
    ns, _ = losses_ns
    data = {"response_mask": torch.tensor([[False, False]]),
            "tg_mask": torch.tensor([[True, True]]),
            "batch_num_seqs": 1, "dp_size": 1, "batch_num_tg_seqs": 1}
    with pytest.raises(RuntimeError, match="tg_losses"):
        _call_ppo(ns, 1.0, "seq-mean-token-mean", data,
                  losses=torch.tensor([[0.0, 0.0]]))


# ---------------------------------------------------------------------------
# 4. source pins: engine stamp, TQ rerouting, agent-loop won split
# ---------------------------------------------------------------------------
def test_engine_stamps_batch_num_tg_seqs():
    src = (_VERL / "workers" / "engine" / "fsdp" / "transformer_impl.py").read_text(encoding="utf-8")
    assert 'if "tg_mask" in data.keys():' in src
    block = src.split('if "tg_mask" in data.keys():', 1)[1]
    assert "all_reduce" in block.split("assign_non_tensor")[0]
    assert "batch_num_tg_seqs=batch_num_tg_seqs.item()" in block


def test_tq_put_reroutes_tg_rows_out_of_response_mask():
    src = (_VERL / "trainer" / "ppo" / "v1" / "agent_loop_tq.py").read_text(encoding="utf-8")
    i_sft = src.index('if sample_role == "sft":')
    i_tg = src.index('if sample_role == "tg":')
    assert i_sft < i_tg, "tg branch must run after the sft mask capture"
    tg_block = src[i_tg : i_tg + 400]
    # capture-then-clear order: tg_mask takes the row's mask BEFORE clearing.
    assert tg_block.index('field["tg_mask"] = field["response_mask"]') < tg_block.index(
        'field["response_mask"] = torch.zeros_like(field["response_mask"])'
    )


def test_agent_loop_won_split_source_pins():
    src = (_ROOT / "src" / "trace_opd" / "train" / "agent_loop.py").read_text(encoding="utf-8")
    # The tg conversion must displace the OPD teacher fields, both of them.
    assert src.count("not tg_converted") == 2
    # Full tg row: doc-prefixed training prompt, bare prompt as the label prefix.
    assert "output.prompt_ids = list(tg_prompt_ids)" in src
    block = src.split("output.prompt_ids = list(tg_prompt_ids)", 1)[1]
    assert 'output.extra_fields["teacher_prompt_ids"] = list(' in block.split("else:")[0]
    # Placeholder rows clear the mask so tg_mask is zero too (TQ takes the
    # cleared mask), and every drop reason lands on a curve.
    assert 'output.response_mask = [0] * len(output.response_ids)' in src
    for reason in ("overflow", "nodoc", "lostonly"):
        assert reason in src


def test_experimental_agent_loop_plans_tg_and_blocks_vllm_backend():
    src = (_VERL / "experimental" / "agent_loop" / "agent_loop.py").read_text(encoding="utf-8")
    assert '"tg" if (_tp is not None and has_signal) else "tg_skip"' in src
    assert "require TRACE_TEACHER_SCORING_BACKEND=fsdp" in src


# ---------------------------------------------------------------------------
# 5. trace_opd config: validation, conditional fingerprint, launcher
# ---------------------------------------------------------------------------
def _load(tmp_path, name, overrides):
    from test_mixed_objective import _TEMPLATE_YAML, _write

    from trace_opd.common.config import load_config

    return load_config(_write(tmp_path / name, "c.yaml", _TEMPLATE_YAML),
                       overrides=overrides)


def test_config_tg_requires_lost_only(tmp_path):
    with pytest.raises(ValueError, match="opd_lost_only"):
        _load(tmp_path, "a", ["train.mix_tg_weight=1.0"])


def test_config_tg_mutually_exclusive_with_rl_and_sft(tmp_path):
    base = ["train.mix_tg_weight=1.0", "train.opd_lost_only=true"]
    with pytest.raises(ValueError, match="mutually exclusive"):
        _load(tmp_path, "b", base + ["train.mix_rl_weight=0.5"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        _load(tmp_path, "c", base + ["train.mix_sft_weight=0.5"])


def test_config_tg_kl_mode_whitelist_and_fsdp_backend(tmp_path):
    # Decoupling ruling (2026-08-02): the OPD divergence is free within the
    # top-k family — reverse_kl_topk is a legal T1 base (rkl_seqmean reuse) —
    # while k3/gated (teacher_topk=0 single-column layout) stays structurally
    # incompatible with the tg top-k label.
    base = ["train.mix_tg_weight=1.0", "train.opd_lost_only=true"]
    cfg = _load(tmp_path, "d0", base + ["train.opd_kl_mode=reverse_kl_topk"])
    assert cfg.train.opd_kl_mode == "reverse_kl_topk"
    with pytest.raises(ValueError, match="single-column"):
        _load(tmp_path, "d1", base + ["train.opd_kl_mode=reverse_kl_k3"])
    with pytest.raises(ValueError, match="single-column"):
        _load(tmp_path, "d2", base + ["train.opd_kl_mode=reverse_kl_gated"])
    with pytest.raises(ValueError, match="fsdp"):
        _load(tmp_path, "e", base + ["train.teacher_scoring_backend=vllm"])


def test_config_health_gate_rules(tmp_path):
    # The gate is meaningless without the tg arm, must be positive, and a
    # legal combo loads with the value intact.
    with pytest.raises(ValueError, match="health_gate"):
        _load(tmp_path, "hg0", ["train.mix_tg_health_gate=900"])
    with pytest.raises(ValueError, match="positive"):
        _load(tmp_path, "hg1", ["train.mix_tg_weight=1.0",
                                "train.opd_lost_only=true",
                                "train.mix_tg_health_gate=-1"])
    cfg = _load(tmp_path, "hg2", ["train.mix_tg_weight=1.0",
                                  "train.opd_lost_only=true",
                                  "train.mix_tg_health_gate=900"])
    assert cfg.train.mix_tg_health_gate == 900.0


def test_fingerprint_health_gate_is_conditional(tmp_path):
    from trace_opd.common.config import objective_fingerprint

    base = ["train.mix_tg_weight=1.0", "train.opd_lost_only=true"]
    fp_tg = objective_fingerprint(_load(tmp_path, "fg0", base))
    fp_gate = objective_fingerprint(
        _load(tmp_path, "fg1", base + ["train.mix_tg_health_gate=900"]))
    assert fp_gate != fp_tg                # the gate changes the label set
    fp_gate2 = objective_fingerprint(
        _load(tmp_path, "fg2", base + ["train.mix_tg_health_gate=600"]))
    assert fp_gate2 != fp_gate             # and so does its threshold


def test_config_tg_valid_combo_loads(tmp_path):
    cfg = _load(tmp_path, "f", ["train.mix_tg_weight=1.0", "train.opd_lost_only=true"])
    assert cfg.train.mix_tg_weight == 1.0 and cfg.train.opd_lost_only is True


def test_fingerprint_tg_and_lost_only_are_conditional(tmp_path):
    from trace_opd.common.config import objective_fingerprint

    fp_default = objective_fingerprint(_load(tmp_path, "g", None))
    fp_explicit0 = objective_fingerprint(
        _load(tmp_path, "h", ["train.mix_tg_weight=0.0"]))
    fp_lost = objective_fingerprint(
        _load(tmp_path, "i", ["train.opd_lost_only=true"]))
    fp_tg = objective_fingerprint(
        _load(tmp_path, "j", ["train.mix_tg_weight=1.0", "train.opd_lost_only=true"]))
    assert fp_default == fp_explicit0      # defaults never enter the hash
    assert fp_lost != fp_default           # T1' changes the OPD data distribution
    assert fp_tg != fp_default and fp_tg != fp_lost


def test_launcher_carries_tg_coef_to_verl(tmp_path):
    import importlib.util

    launcher = _ROOT / "scripts" / "launch_train.py"
    spec = importlib.util.spec_from_file_location("launch_train_tg", launcher)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = _load(tmp_path, "k", ["train.mix_tg_weight=1.0", "train.opd_lost_only=true"])
    cmd = mod.build_command(cfg, tmp_path / "run")
    assert "+distillation.distillation_loss.tg_loss_coef=1.0" in cmd


# ---------------------------------------------------------------------------
# 6. the tg curve key family
# ---------------------------------------------------------------------------
def test_curve_mixed_tg_keys():
    from trace_opd.train.curve import build_curve_metrics

    fields = {"env_steps": 3, "invalid_format": 0, "truncated": 0,
              "tg_role": 1, "tg_dropped_overflow": 0,
              "tg_label_thought_chars": 946, "tg_label_thought_turns": 3,
              "tg_label_thought_cpt": 946 / 3}
    on = build_curve_metrics(validate=False, research_enabled=False,
                             mixed_tg=True, fields=fields)
    assert on["tg_role"] == 1.0 and on["tg_dropped_nodoc"] == 0.0
    assert "tg_dropped_lostonly" in on and "tg_dropped_unhealthy" in on
    # Label-quality triple (pre-registered sharp readout, online form):
    # cpt key is the per-row chars/turns — mean(cpt)/mean(tg_role) recovers
    # the trajectory-mean of label rows, the SAME estimand as the offline
    # won_traj_evolution baseline (codex impl-review P1-5); chars/turns stay
    # as the turn-weighted totals view. Placeholder rows default all to 0.
    assert on["tg_label_thought_chars"] == 946.0
    assert on["tg_label_thought_turns"] == 3.0
    assert on["tg_label_thought_cpt"] == pytest.approx(946 / 3)
    on_placeholder = build_curve_metrics(
        validate=False, research_enabled=False, mixed_tg=True,
        fields={"env_steps": 1, "invalid_format": 0, "truncated": 0})
    assert on_placeholder["tg_label_thought_chars"] == 0.0
    assert on_placeholder["tg_label_thought_cpt"] == 0.0
    off = build_curve_metrics(validate=False, research_enabled=False,
                              mixed_tg=False, fields=fields)
    assert "tg_role" not in off


# ---------------------------------------------------------------------------
# 7. tg rows leave the on-policy machinery (0d653b0 self-review fixes)
# ---------------------------------------------------------------------------
def test_tg_rows_in_behaviour_metrics_but_isolated_from_grpo_and_padding():
    trainer = (_VERL / "trainer" / "ppo" / "v1" / "trainer_base.py").read_text(encoding="utf-8")
    # codex tg-review P0-1: a tg row is a bare rollout the policy actually
    # WON — it BELONGS in the behaviour metrics (excluding it makes
    # critic/score/mean a lost-only ~0 curve). The on-policy exclusion list
    # must therefore NOT contain tg...
    i = trainer.index("on_policy_mask = non_padding_mask")
    excl = trainer[i : i + 1200]
    assert 'in ("sft", "research")' in excl and '"tg"' not in excl.split("is_expert")[1][:400]
    # ...while the GRPO group isolation DOES (a reward-1.0 group-mate would
    # hand every lost bare row a negative advantage baseline)...
    assert 'if role in ("sft", "research", "tg") else uid' in trainer
    # ...and the gate that enables the role machinery must see the tg arm
    # (codex P1-3: without this the isolation is unreachable in T1).
    gate = trainer.split("def _mixed_objective_enabled")[1][:1600]
    assert "tg_loss_coef" in gate
    # codex P1-2: the min-signal floor counts tg signal too.
    sig = trainer.split("min_signal > 0:")[1][:1600]
    assert "tg_mask" in sig
    padding = (_VERL / "trainer" / "ppo" / "padding_utils.py").read_text(encoding="utf-8")
    assert 'if "tg_mask" in template_sample:' in padding
    assert 'template_sample["tg_mask"] = torch.zeros_like(response_mask)' in padding


# ---------------------------------------------------------------------------
# 8. codex tg-review P0-2, made STRUCTURAL by the decoupling ruling
#    (2026-08-02): the OPD -10 floor makes a renormalised-label "KL" genuinely
#    negative — which is why tg no longer rides the OPD formula at all. The
#    hand-computed trigger case now pins the CONTRAST: the pure OPD formula
#    still floors this input negative (its own semantics, untouched), while
#    the standalone tg soft-CE returns the true non-negative value with no
#    clamp anywhere near it.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fsdp_losses_ns():
    stub_names = ["verl", "verl.utils", "verl.utils.ulysses", "verl.workers",
                  "verl.workers.config"]
    saved = {k: sys.modules.get(k) for k in stub_names}

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
    _register("verl.workers.config", DistillationConfig=object,
              DistillationLossConfig=object)
    ns = {}
    src = (_VERL / "trainer" / "distillation" / "fsdp" / "losses.py").read_text(encoding="utf-8")
    exec(compile(src, "vendored_fsdp_losses.py", "exec"), ns)
    yield ns
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def _p02_trigger_inputs():
    # Renormalised label [0.99995, 2.5e-5, 2.5e-5]: tail log-prob -10.6 sits
    # BELOW the -10 floor. q from logits [30,0,0,0,0]: top ~1, tail ~9e-14
    # (log ~ -30). Unclamped soft-CE ~ +9.2e-4 > 0; with both sides floored
    # at -10 the tail term vanishes and the top term ~ -5e-5 < 0 survives.
    label = torch.log(torch.tensor([[[0.99995, 2.5e-5, 2.5e-5]]]))
    ids = torch.tensor([[[0, 1, 2]]], dtype=torch.int64)
    label_nested = torch.nested.nested_tensor([label[0]], layout=torch.jagged)
    ids_nested = torch.nested.nested_tensor([ids[0]], layout=torch.jagged)
    logits = torch.zeros(1, 1, 5)
    logits[0, 0, 0] = 30.0
    loss_cfg = types.SimpleNamespace(log_prob_min_clamp=-10.0,
                                     use_chunked_topk=False,
                                     chunked_topk_chunk_size=4096)
    cfg = types.SimpleNamespace(distillation_loss=loss_cfg)
    return label_nested, ids_nested, logits, cfg


def test_forward_kl_topk_restored_to_pure_opd_form(fsdp_losses_ns):
    import inspect

    # The tg hole-punching is gone: no tg parameter, byte-comparable OPD math.
    sig = inspect.signature(fsdp_losses_ns["compute_forward_kl_topk"])
    assert "tg_token_mask" not in sig.parameters
    # The trigger case still floors negative under the OPD formula — its own
    # (OPD) semantics, untouched; exactly why tg rows must not ride it.
    label_nested, ids_nested, logits, cfg = _p02_trigger_inputs()
    out = fsdp_losses_ns["compute_forward_kl_topk"](
        student_logits=logits, teacher_topk_log_probs=label_nested,
        teacher_topk_ids=ids_nested, config=cfg, data_format="thd")
    assert out["distillation_losses"][0, 0].item() < -1e-5


def test_tg_soft_ce_structurally_immune_to_p02(fsdp_losses_ns):
    # Same trigger input through the standalone formula: the true
    # non-negative soft-CE, no floor, no max-clamp, gradient alive exactly
    # where q parks mass outside the label support.
    label_nested, ids_nested, logits, cfg = _p02_trigger_inputs()
    logits = logits.clone().requires_grad_(True)
    out = fsdp_losses_ns["compute_tg_soft_ce"](
        student_logits=logits, tg_label_log_probs=label_nested,
        tg_label_ids=ids_nested, tg_token_mask=torch.tensor([[True]]),
        config=cfg)
    val = out[0, 0].item()
    assert val > 0 and val == pytest.approx(9.2e-4, rel=0.2)
    out.sum().backward()
    assert logits.grad.abs().max().item() > 0
    # Source pin: the standalone formula never touches the OPD clamps.
    src = (_VERL / "trainer" / "distillation" / "fsdp" / "losses.py").read_text(encoding="utf-8")
    tg_src = src.split("def compute_tg_soft_ce")[1]
    assert "log_prob_min_clamp" not in tg_src.split("Args:")[1]
    assert "loss_max_clamp" not in tg_src.split("Args:")[1]


def test_tg_soft_ce_hand_math_gradient_and_masking(fsdp_losses_ns):
    # Full-formula pin on the standalone soft-CE: hand-computed parity on tg
    # positions, exact zeros elsewhere, analytic gradient q - label through
    # the chunked gather (validates the index_put autograd path).
    torch.manual_seed(0)
    T, V, K = 6, 50, 4
    logits = torch.randn(1, T, V, requires_grad=True)
    mask = torch.tensor([[True, True, True, False, False, False]])
    ref_lp = torch.log_softmax(torch.randn(T, V), dim=-1)
    top_lp, top_ids = torch.topk(ref_lp, k=K, dim=-1)
    lab_lp = top_lp - torch.logsumexp(top_lp, dim=-1, keepdim=True)
    lab_lp_full = torch.zeros(T, K)
    lab_ids_full = torch.zeros(T, K, dtype=torch.int32)
    lab_lp_full[:3] = lab_lp[:3]
    lab_ids_full[:3] = top_ids[:3].to(torch.int32)
    nt_lp = torch.nested.nested_tensor([lab_lp_full], layout=torch.jagged)
    nt_id = torch.nested.nested_tensor([lab_ids_full], layout=torch.jagged)
    loss_cfg = types.SimpleNamespace(chunked_topk_chunk_size=3)
    cfg = types.SimpleNamespace(distillation_loss=loss_cfg)

    out = fsdp_losses_ns["compute_tg_soft_ce"](logits, nt_lp, nt_id, mask, cfg)
    q_lp_full = torch.log_softmax(logits[0].detach(), dim=-1)
    for t in range(3):
        p = lab_lp[t].exp()
        hand = (p * (lab_lp[t] - q_lp_full[t, top_ids[t]])).sum()
        assert out[0, t].item() == pytest.approx(hand.item(), abs=1e-5)
    assert (out[0, 3:] == 0).all()

    out.sum().backward()
    g = logits.grad[0]
    for t in range(3):
        q = torch.softmax(logits[0, t].detach(), dim=-1)
        expected = q.clone()
        expected[top_ids[t]] -= lab_lp[t].exp()
        assert torch.allclose(g[t], expected, atol=1e-5)
    assert (g[3:] == 0).all()

    # Empty-mask fast path: zero matrix, right shape.
    out2 = fsdp_losses_ns["compute_tg_soft_ce"](
        torch.randn(1, T, V), nt_lp, nt_id,
        torch.zeros(1, T, dtype=torch.bool), cfg)
    assert (out2 == 0).all() and out2.shape == (1, T)


def test_tg_branch_drops_loss_max_clamp_and_reports_preclamp_min(losses_ns):
    ns, calls = losses_ns
    tg_mask = torch.tensor([[True, True]])
    data = {"response_mask": torch.tensor([[False, False]]),
            "tg_mask": tg_mask, "batch_num_seqs": 1, "dp_size": 1,
            "batch_num_tg_seqs": 1}
    # A legitimate large tg loss (q far off-support) must reach the spy
    # un-truncated even with loss_max_clamp configured.
    losses = torch.tensor([[27.0, 2.0]])
    ns["get_distillation_loss_fn"] = lambda mode: (
        lambda config, distillation_config, model_output, data: (losses, {}))
    config = types.SimpleNamespace(loss_agg_mode="seq-mean-token-mean",
                                   global_batch_info={})
    loss_cfg = types.SimpleNamespace(
        loss_mode="fake", loss_max_clamp=10.0, use_policy_gradient=False,
        use_task_rewards=False, rl_loss_coef=0.0, sft_loss_coef=0.0,
        research_loss_coef=0.0, tg_loss_coef=1.0,
        distillation_loss_coef=1.0, opd_loss_coef=1.0)
    distillation_config = types.SimpleNamespace(distillation_loss=loss_cfg)
    _, metrics = ns["distillation_ppo_loss"](
        config, distillation_config,
        {"distillation_losses": torch.zeros_like(losses), "tg_losses": losses},
        data)
    tg_calls = [c for c in calls if torch.equal(c.get("loss_mask"), tg_mask)]
    assert tg_calls[-1]["loss_mat"].max().item() == pytest.approx(27.0)
    assert "tg/loss_min_preclamp" in metrics and "tg/loss_max" in metrics


def test_research_replica_curve_call_carries_mixed_tg():
    # codex tg-review P0-3: the curve key set must be batch-uniform.
    src = (_ROOT / "src" / "trace_opd" / "train" / "agent_loop.py").read_text(encoding="utf-8")
    calls = src.split("build_curve_metrics(")[1:]
    assert len(calls) == 2 and all("mixed_tg" in c[:700] for c in calls)


def test_config_rejects_tg_with_token_mean(tmp_path):
    # codex tg-review P1-5: the token-mean fallback is a different estimand.
    with pytest.raises(ValueError, match="seq-mean-token-mean"):
        _load(tmp_path, "tm", ["train.mix_tg_weight=1.0", "train.opd_lost_only=true",
                               "train.loss_agg_mode=token-mean"])



def test_dispatch_decoupling_contracts(losses_ns):
    """compute_topk_loss dispatch: (a) tg arm on + no tg rows in the
    micro-batch -> a zero tg matrix ships (uniform read path downstream);
    (b) tg rows present but the label channel missing -> fail loud;
    (c) tg rows + labels -> the standalone formula's output ships."""
    ns, _ = losses_ns
    T, V = 4, 7
    logits = torch.zeros(1, T, V)

    def fake_opd(student_logits, teacher_topk_log_probs, teacher_topk_ids,
                 config, data_format):
        shape = student_logits.shape[:2]
        return {"distillation_losses": torch.zeros(shape),
                "student_mass": torch.zeros(shape),
                "teacher_mass": torch.zeros(shape), "obs": None}

    seen = {}

    def fake_tg(student_logits, tg_label_log_probs, tg_label_ids,
                tg_token_mask, config, obs_sink=None):
        seen["mask"] = tg_token_mask.clone()
        return torch.full(student_logits.shape[:2], 7.0)

    fsdp_mod = types.ModuleType("verl.trainer.distillation.fsdp.losses")
    fsdp_mod.compute_forward_kl_topk = fake_opd
    fsdp_mod.compute_reverse_kl_topk = fake_opd
    fsdp_mod.compute_tg_soft_ce = fake_tg
    pkg_dist = types.ModuleType("verl.trainer.distillation")
    pkg_fsdp = types.ModuleType("verl.trainer.distillation.fsdp")
    saved = {k: sys.modules.get(k) for k in
             ("verl.trainer.distillation", "verl.trainer.distillation.fsdp",
              "verl.trainer.distillation.fsdp.losses")}
    sys.modules["verl.trainer.distillation"] = pkg_dist
    sys.modules["verl.trainer.distillation.fsdp"] = pkg_fsdp
    sys.modules["verl.trainer.distillation.fsdp.losses"] = fsdp_mod
    try:
        config = types.SimpleNamespace(strategy="fsdp")
        loss_cfg = types.SimpleNamespace(loss_mode="reverse_kl_topk",
                                         log_prob_min_clamp=-10.0,
                                         tg_loss_coef=1.0)
        dcfg = types.SimpleNamespace(distillation_loss=loss_cfg)
        prompts = torch.nested.nested_tensor(
            [torch.zeros(1, dtype=torch.long)] * 2, layout=torch.jagged)
        responses = torch.nested.nested_tensor(
            [torch.zeros(1, dtype=torch.long)] * 2, layout=torch.jagged)

        # (a) no tg rows -> zero matrix, formula not called
        data = {"extra_fields": [{"sample_role": "bare"}, {"sample_role": "bare"}],
                "prompts": prompts, "responses": responses,
                "teacher_logprobs": None, "teacher_ids": None}
        out = ns["compute_topk_loss"](config, dcfg, data, logits, "thd")
        assert torch.equal(out["tg_losses"], torch.zeros(1, T)) and not seen

        # (a') codex impl-review P0-1: a PLACEHOLDER-only shard (dropped won
        # rows: sample_role "tg" but teacher_mode "tg_skip", no label shipped)
        # must take the zero-matrix path, NOT demand the label channel — the
        # predicate is teacher_mode, matching the scorer's has_tg.
        data_ph = dict(data)
        data_ph["extra_fields"] = [
            {"sample_role": "bare"},
            {"sample_role": "tg", "teacher_mode": "tg_skip",
             "tg_dropped_nodoc": 1},
        ]
        out = ns["compute_topk_loss"](config, dcfg, data_ph, logits, "thd")
        assert torch.equal(out["tg_losses"], torch.zeros(1, T)) and not seen

        # (b) REAL tg rows, label channel missing -> fail loud
        data_tg = dict(data)
        data_tg["extra_fields"] = [
            {"sample_role": "bare"},
            {"sample_role": "tg", "teacher_mode": "tg"},
        ]
        with pytest.raises(RuntimeError, match="tg_label_ids"):
            ns["compute_topk_loss"](config, dcfg, data_tg, logits, "thd")

        # (c) tg rows + labels -> standalone output ships, mask marks ONLY the
        # real tg row's span
        data_tg["tg_label_ids"] = "stub"
        data_tg["tg_label_logprobs"] = "stub"
        out = ns["compute_topk_loss"](config, dcfg, data_tg, logits, "thd")
        assert (out["tg_losses"] == 7.0).all()
        assert torch.equal(seen["mask"],
                           torch.tensor([[False, False, True, True]]))

        # (c') a placeholder row NEXT TO a real tg row stays out of the mask
        seen.clear()
        data_mix = dict(data_tg)
        data_mix["extra_fields"] = [
            {"sample_role": "tg", "teacher_mode": "tg_skip"},
            {"sample_role": "tg", "teacher_mode": "tg"},
        ]
        ns["compute_topk_loss"](config, dcfg, data_mix, logits, "thd")
        assert torch.equal(seen["mask"],
                           torch.tensor([[False, False, True, True]]))
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_verl_config_tg_whitelist_source_pin():
    # Decoupling ruling: the OPD loss_mode is free within the top-k family;
    # k3/gated (teacher_topk=0 single-column) is a structural layout clash.
    src = (_VERL / "workers" / "config" / "distillation.py").read_text(encoding="utf-8")
    assert "loss_mode not in (" in src
    seg = src.split("tg_loss_coef) > 0.0 and self.loss_mode not in (")[1][:400]
    assert "forward_kl_topk" in seg and "reverse_kl_topk" in seg


def test_tg_soft_ce_bf16_near_converged_stays_nonneg(fsdp_losses_ns):
    """codex impl-review P0-2: with BF16 logits the q_lp gather must stay
    fp32 end to end. A BF16 round-trip of the log-probs carries ~1e-3
    absolute error — it flips the near-converged KL (~1e-5) negative, and
    clamp_min(0) then kills loss AND gradient exactly in the fine-tuning
    regime. Pin: the shipped path matches a float64 reference and stays
    non-negative; the legacy round-trip provably goes negative on the same
    input."""
    torch.manual_seed(3)
    V = 8
    # Label: renormalised two-bucket distribution ~ (0.397, 0.603).
    lab_p = torch.tensor([0.397, 0.603], dtype=torch.float64)
    lab_lp64 = torch.log(lab_p)
    # q: the same buckets at (0.399, 0.601) — near-converged teacher.
    q_target = torch.full((V,), 1e-8, dtype=torch.float64)
    q_target[3], q_target[5] = 0.399, 0.601
    logits64 = torch.log(q_target)
    logits_bf16 = logits64.to(torch.bfloat16)

    lab_lp_full = torch.zeros(1, 2)
    lab_lp_full[0] = lab_lp64.float()
    lab_ids_full = torch.tensor([[3, 5]], dtype=torch.int32)
    nt_lp = torch.nested.nested_tensor([lab_lp_full], layout=torch.jagged)
    nt_id = torch.nested.nested_tensor([lab_ids_full], layout=torch.jagged)
    cfg = types.SimpleNamespace(
        distillation_loss=types.SimpleNamespace(chunked_topk_chunk_size=4096))

    x = logits_bf16.reshape(1, 1, V).clone().requires_grad_(True)
    out = fsdp_losses_ns["compute_tg_soft_ce"](
        x, nt_lp, nt_id, torch.ones(1, 1, dtype=torch.bool), cfg)

    # float64 reference on the SAME bf16-quantised logits. The shipped path
    # computes in fp32, so allow the fp32-vs-fp64 tolerance at this magnitude
    # (~1e-6 KL); the load-bearing claims are non-negativity + the right
    # order of magnitude, against the legacy round-trip's -1e-3 below.
    ref_lp = torch.log_softmax(logits_bf16.double(), dim=-1)[[3, 5]]
    ref = float((lab_lp64.exp() * (lab_lp64 - ref_lp)).sum())
    assert ref > 0
    assert out[0, 0].item() == pytest.approx(ref, rel=0.05)
    assert out[0, 0].item() >= 0.0
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    # The legacy behaviour (round-tripping q_lp through bf16) is genuinely
    # negative on this input — the defect the fp32 path removes.
    legacy_q_lp = torch.log_softmax(logits_bf16.float(), dim=-1)[[3, 5]]
    legacy_q_lp = legacy_q_lp.to(torch.bfloat16).float()
    legacy = float((lab_lp64.exp().float() * (lab_lp64.float() - legacy_q_lp)).sum())
    assert legacy < 0


def test_metric_min_max_aggregate_dp_accepts_ragged():
    """codex impl-review P0-3: conditionally-emitted MIN/MAX metrics (tg
    telemetry only on micro-batches holding a tg token; OPD mass extrema
    absent on all-tg micro-batches) legitimately give DP ranks different
    value counts under dynamic packing. Extrema are ragged-safe; SUM/MEAN
    keep the equal-length invariant."""
    src = (_VERL / "utils" / "metric" / "utils.py").read_text(encoding="utf-8")
    ns = {}
    exec(compile(src, "vendored_metric_utils.py", "exec"), ns)
    Metric, AggregationType = ns["Metric"], ns["AggregationType"]

    m_a = Metric(aggregation=AggregationType.MIN, value=0.5)
    m_b = Metric(aggregation=AggregationType.MIN, value=0.2)
    m_b.append(0.9)
    assert Metric.aggregate_dp([m_a, m_b]) == pytest.approx(0.2)

    x_a = Metric(aggregation=AggregationType.MAX, value=1.0)
    x_b = Metric(aggregation=AggregationType.MAX, value=3.0)
    x_b.append(2.0)
    assert Metric.aggregate_dp([x_a, x_b]) == pytest.approx(3.0)

    s_a = Metric(aggregation=AggregationType.SUM, value=1.0)
    s_b = Metric(aggregation=AggregationType.SUM, value=1.0)
    s_b.append(1.0)
    with pytest.raises(ValueError, match="same number of values"):
        Metric.aggregate_dp([s_a, s_b])


def test_config_tg_only_objective_is_legal(tmp_path):
    # codex impl-review P1-6: tg is a full objective — OPD 0 + tg > 0 loads.
    cfg = _load(tmp_path, "tgo", ["train.mix_opd_weight=0.0",
                                  "train.mix_tg_weight=1.0",
                                  "train.opd_lost_only=true"])
    assert cfg.train.mix_opd_weight == 0.0 and cfg.train.mix_tg_weight == 1.0
    # All four zero is still rejected.
    with pytest.raises(ValueError, match="no objective"):
        _load(tmp_path, "tgz", ["train.mix_opd_weight=0.0"])


def test_config_health_gate_rejects_inf(tmp_path):
    # codex impl-review P2-1: inf passes ">0" but never drops a row while the
    # fingerprint claims a configured gate.
    with pytest.raises(ValueError, match="finite"):
        _load(tmp_path, "hginf", ["train.mix_tg_weight=1.0",
                                  "train.opd_lost_only=true",
                                  "train.mix_tg_health_gate=.inf"])


def test_verl_config_objective_presence_includes_tg():
    src = (_VERL / "workers" / "config" / "distillation.py").read_text(encoding="utf-8")
    assert "or self.tg_loss_coef" in src


def _obs_stub(enabled=True, spy=None):
    mod = types.ModuleType("verl.trainer.distillation.observation")
    mod.observation_enabled = lambda: enabled
    mod._step_wanted = lambda step: True
    mod.collect_micro_batch = spy if spy is not None else (lambda **kw: None)
    mod._disable = lambda why: None
    return mod


def test_tg_soft_ce_obs_sink_ships_label_and_q(fsdp_losses_ns):
    """tg 行分布级观察(环1 显微):enabled 时 sink 收到三个 [1,T,K] 数组 —
    标签 p̄ / 其 id / q 在标签支持集上的 log-prob,tg 位置与输入逐位一致、
    非 tg 位置零;disabled 时 sink 保持空(零开销契约)。"""
    torch.manual_seed(1)
    T, V, K = 4, 12, 3
    logits = torch.randn(1, T, V)
    mask = torch.tensor([[True, True, False, False]])
    ref_lp = torch.log_softmax(torch.randn(T, V), dim=-1)
    top_lp, top_ids = torch.topk(ref_lp, k=K, dim=-1)
    lab_lp = top_lp - torch.logsumexp(top_lp, dim=-1, keepdim=True)
    lab_full = torch.zeros(T, K)
    ids_full = torch.zeros(T, K, dtype=torch.int32)
    lab_full[:2] = lab_lp[:2]
    ids_full[:2] = top_ids[:2].to(torch.int32)
    nt_lp = torch.nested.nested_tensor([lab_full], layout=torch.jagged)
    nt_id = torch.nested.nested_tensor([ids_full], layout=torch.jagged)
    cfg = types.SimpleNamespace(
        distillation_loss=types.SimpleNamespace(chunked_topk_chunk_size=4096))

    saved = sys.modules.get("verl.trainer.distillation.observation")
    sys.modules["verl.trainer.distillation.observation"] = _obs_stub(enabled=True)
    try:
        sink = {}
        fsdp_losses_ns["compute_tg_soft_ce"](logits, nt_lp, nt_id, mask, cfg,
                                             obs_sink=sink)
        assert set(sink) == {"tg_label_ids", "tg_label_lps", "tg_q_lps"}
        for v in sink.values():
            assert v.shape == (1, T, K)
        assert torch.equal(sink["tg_label_ids"][0, :2], top_ids[:2].to(torch.int32))
        assert torch.allclose(sink["tg_label_lps"][0, :2], lab_lp[:2])
        q_ref = torch.log_softmax(logits[0, :2].float(), dim=-1).gather(
            -1, top_ids[:2])
        assert torch.allclose(sink["tg_q_lps"][0, :2], q_ref, atol=1e-5)
        assert (sink["tg_label_ids"][0, 2:] == 0).all()
        assert (sink["tg_q_lps"][0, 2:] == 0).all()

        # disabled -> empty sink; no sink -> no side effects.
        sys.modules["verl.trainer.distillation.observation"] = _obs_stub(enabled=False)
        sink2 = {}
        fsdp_losses_ns["compute_tg_soft_ce"](logits, nt_lp, nt_id, mask, cfg,
                                             obs_sink=sink2)
        assert sink2 == {}
    finally:
        if saved is None:
            sys.modules.pop("verl.trainer.distillation.observation", None)
        else:
            sys.modules["verl.trainer.distillation.observation"] = saved


def test_collect_observation_tg_rows_use_their_own_pack(losses_ns):
    """collect 层(环1 显微落盘):真 tg 行的 teacher_* 字段来自 tg_label 包、
    student_support_lps 来自 tg_q_lps、token_loss 来自 tg 矩阵;OPD 行照旧;
    占位行(teacher_mode=tg_skip)跳过。"""
    ns, _ = losses_ns
    K = 2
    rows_seen = {}

    def spy(**kw):
        rows_seen.update(kw)

    stubs = {
        "verl.trainer.distillation.observation": _obs_stub(enabled=True, spy=spy),
        "verl.utils.ulysses": types.SimpleNamespace(
            get_ulysses_sequence_parallel_world_size=lambda: 1,
            get_ulysses_sequence_parallel_rank=lambda: 0),
    }
    saved = {k: sys.modules.get(k) for k in stubs}
    for k, v in stubs.items():
        mod = v if isinstance(v, types.ModuleType) else types.ModuleType(k)
        if not isinstance(v, types.ModuleType):
            for a, val in vars(v).items():
                setattr(mod, a, val)
        sys.modules[k] = mod
    try:
        nested = lambda rows_: torch.nested.nested_tensor(rows_, layout=torch.jagged)
        prompts = nested([torch.zeros(1, dtype=torch.long)] * 3)
        responses = nested([torch.tensor([5, 6])] * 3)
        T = 9  # packed: 3 rows x (P=1 + R=2)
        data = {
            "prompts": prompts, "responses": responses,
            "old_log_probs": nested([torch.zeros(2)] * 3),
            "response_mask": nested([torch.tensor([1, 1]), torch.tensor([1, 1]),
                                     torch.tensor([0, 0])]),
            "rm_scores": nested([torch.tensor([0.0, 0.0]),
                                 torch.tensor([1.0, 0.0]),
                                 torch.tensor([1.0, 0.0])]),
            "extra_fields": [
                {"sample_role": "bare", "won": 0, "task_id": "a",
                 "turn_spans": [[0, 2]]},
                {"sample_role": "tg", "teacher_mode": "tg", "won": 1,
                 "task_id": "b", "turn_spans": [[0, 2]]},
                {"sample_role": "tg", "teacher_mode": "tg_skip", "won": 1,
                 "task_id": "c"},
            ],
            "global_steps": [4, 4, 4],
        }
        obs = {
            "teacher_ids": torch.ones(1, T, K, dtype=torch.int32),
            "teacher_lps": torch.full((1, T, K), -1.0),
            "student_ids": torch.ones(1, T, K, dtype=torch.int32),
            "student_lps": torch.full((1, T, K), -9.0),
            "student_support_lps": torch.full((1, T, K), -3.0),
            "tg_label_ids": torch.full((1, T, K), 2, dtype=torch.int32),
            "tg_label_lps": torch.full((1, T, K), -2.0),
            "tg_q_lps": torch.full((1, T, K), -4.0),
        }
        ns["_collect_observation_rows"](
            data, obs, torch.full((1, T), 0.5),
            loss_mode="reverse_kl_topk", log_prob_min_clamp=-10.0,
            tg_token_losses=torch.full((1, T), 7.5))
        rows = rows_seen["rows"]
        assert [r["sample_role"] for r in rows] == ["bare", "tg"]
        bare, tg = rows
        assert float(bare["teacher_lps"][0, 0]) == -1.0
        assert float(bare["token_loss"][0]) == 0.5
        assert float(bare["student_support_lps"][0, 0]) == -3.0
        assert int(tg["teacher_ids"][0, 0]) == 2       # label ids, not stubs
        assert float(tg["teacher_lps"][0, 0]) == -2.0  # label p̄
        assert float(tg["student_support_lps"][0, 0]) == -4.0  # q on support
        assert float(tg["token_loss"][0]) == 7.5       # soft-CE matrix
        assert tg["turn_spans"] == [[0, 2]]
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class _FakeStack:
    """NonTensorStack 形态:list() 出来是包装对象,.tolist() 才解包成 dict。"""

    def __init__(self, items, decoded=True):
        self._items = items
        self._decoded = decoded

    def tolist(self):
        if self._decoded:
            return self._items
        return [object() for _ in self._items]  # 解包失败形态:非 dict

    def __iter__(self):
        # list(extras) 的旧路径拿到的是包装对象(非 dict)——正是 §7 的根因。
        return iter(object() for _ in self._items)

    def __len__(self):
        return len(self._items)


def _collect_data(extras):
    nested = lambda rows_: torch.nested.nested_tensor(rows_, layout=torch.jagged)
    n = len(extras)
    return {
        "prompts": nested([torch.zeros(1, dtype=torch.long)] * n),
        "responses": nested([torch.tensor([5, 6])] * n),
        "old_log_probs": nested([torch.zeros(2)] * n),
        "response_mask": nested([torch.tensor([1, 1])] * n),
        "rm_scores": nested([torch.tensor([1.0, 0.0])] * n),
        "extra_fields": extras,
        "global_steps": [4] * n,
    }


def test_collect_decodes_nontensorstack_extras(losses_ns):
    """CD-006 反馈 §7:extra_fields 是 NonTensorStack,collect 必须 .tolist()
    解包——否则所有元数据静默落默认值(won=0、task 空、turn_spans 空、tg 行
    伪装成 bare)。修复后:包装形态下元数据完整。"""
    ns, _ = losses_ns
    rows_seen = {}
    stub = _obs_stub(enabled=True, spy=lambda **kw: rows_seen.update(kw))
    ulysses = types.ModuleType("verl.utils.ulysses")
    ulysses.get_ulysses_sequence_parallel_world_size = lambda: 1
    ulysses.get_ulysses_sequence_parallel_rank = lambda: 0
    saved = {k: sys.modules.get(k) for k in
             ("verl.trainer.distillation.observation", "verl.utils.ulysses")}
    sys.modules["verl.trainer.distillation.observation"] = stub
    sys.modules["verl.utils.ulysses"] = ulysses
    try:
        T = 6  # 2 rows x (P=1 + R=2) packed
        extras = _FakeStack([
            {"sample_role": "bare", "won": 1, "task_id": "t-a",
             "turn_spans": [[0, 2]]},
            {"sample_role": "sft", "won": 1, "task_id": "t-b"},
        ])
        obs = {k: torch.zeros(1, T, 2) for k in
               ("teacher_lps", "student_lps", "student_support_lps")}
        obs["teacher_ids"] = torch.zeros(1, T, 2, dtype=torch.int32)
        obs["student_ids"] = torch.zeros(1, T, 2, dtype=torch.int32)
        ns["_collect_observation_rows"](
            _collect_data(extras), obs, torch.zeros(1, T),
            loss_mode="reverse_kl_topk", log_prob_min_clamp=-10.0)
        rows = rows_seen["rows"]
        # 解包生效:bare 行元数据完整;sft 行(零教师包语义)被跳过。
        assert len(rows) == 1
        assert rows[0]["task_id"] == "t-a" and rows[0]["won"] == 1
        assert rows[0]["turn_spans"] == [[0, 2]]
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_collect_disables_loud_on_undecodable_extras(losses_ns):
    """解包失败(全部非 dict)不再安静写默认值:raise 落入 _disable,原因
    可读,且不落任何行。"""
    ns, _ = losses_ns
    disabled, collected = [], []
    stub = _obs_stub(enabled=True, spy=lambda **kw: collected.append(kw))
    stub._disable = lambda why: disabled.append(why)
    ulysses = types.ModuleType("verl.utils.ulysses")
    ulysses.get_ulysses_sequence_parallel_world_size = lambda: 1
    ulysses.get_ulysses_sequence_parallel_rank = lambda: 0
    saved = {k: sys.modules.get(k) for k in
             ("verl.trainer.distillation.observation", "verl.utils.ulysses")}
    sys.modules["verl.trainer.distillation.observation"] = stub
    sys.modules["verl.utils.ulysses"] = ulysses
    try:
        extras = _FakeStack([{"sample_role": "bare"}], decoded=False)
        obs = {k: torch.zeros(1, 2, 2) for k in
               ("teacher_lps", "student_lps", "student_support_lps")}
        obs["teacher_ids"] = torch.zeros(1, 2, 2, dtype=torch.int32)
        obs["student_ids"] = torch.zeros(1, 2, 2, dtype=torch.int32)
        ns["_collect_observation_rows"](
            _collect_data(extras), obs, torch.zeros(1, 2),
            loss_mode="reverse_kl_topk", log_prob_min_clamp=-10.0)
        assert not collected
        assert disabled and "did not decode" in disabled[0]
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_tg_gradient_bounded_under_extreme_q():
    """User-requested extreme-case pin (2026-08-02): the tg soft-CE has NO
    1/q term — the log singularity cancels against the softmax Jacobian, so
    the per-logit gradient is bounded by 1 (and per-token L1 by 2) no matter
    how small q gets on the label support. A ratio-type estimator (p/q) would
    explode on the same input; the tg formula is structurally immune."""
    V = 50
    logits = torch.zeros(V, requires_grad=True)
    with torch.no_grad():
        logits[10] = 35.0  # q on the label support ~ 6e-16
    label_ids = torch.tensor([0, 1, 2])
    label_p = torch.tensor([0.99995, 2.5e-5, 2.5e-5])  # tail below e^-10
    log_q = torch.log_softmax(logits, dim=-1)[label_ids]
    loss = (label_p * (label_p.log() - log_q)).sum()
    loss.backward()
    assert loss.item() > 30            # the loss IS large — that is fine
    assert logits.grad.abs().max().item() <= 1.0 + 1e-6
    assert logits.grad.abs().sum().item() <= 2.0 + 1e-6
