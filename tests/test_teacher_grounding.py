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

    # Row layout is (P-1) zeros | R valid | 1 dummy for scored rows.
    priv_valid = lps[0][P - 1 : P - 1 + R]
    tg_valid = lps[1][P - 1 : P - 1 + R]
    # privileged: raw top-K logprobs — truncated mass < 1 with K=4 < V=7.
    assert (torch.logsumexp(priv_valid, dim=-1) < -1e-4).all()
    # tg: renormalised — a proper distribution over the K buckets.
    assert torch.logsumexp(tg_valid, dim=-1).abs().max().item() < 1e-5
    # tg_skip: all-zero dummy rows, nothing scored.
    assert data["teacher_logprobs"].unbind()[2].abs().sum().item() == 0.0
    assert data["teacher_ids"].unbind()[2].abs().sum().item() == 0.0


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


def _call_ppo(ns, tg_coef, agg_mode, data, losses=None):
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
    model_output = {"distillation_losses": losses}
    return ns["distillation_ppo_loss"](config, distillation_config, model_output, data)


def test_tg_call_site_fails_loud_without_tg_mask(losses_ns):
    ns, _ = losses_ns
    data = {"response_mask": torch.tensor([[True, True], [False, False]]),
            "batch_num_seqs": 1, "dp_size": 1}
    with pytest.raises(RuntimeError, match="tg_mask"):
        _call_ppo(ns, 1.0, "seq-mean-token-mean", data)


def test_tg_call_site_seq_mean_uses_engine_stamped_tg_count(losses_ns):
    ns, calls = losses_ns
    tg_mask = torch.tensor([[False, False], [True, True]])
    data = {"response_mask": torch.tensor([[True, True], [False, False]]),
            "tg_mask": tg_mask,
            "batch_num_seqs": 5, "dp_size": 2, "batch_num_tg_seqs": 3}
    _, metrics = _call_ppo(ns, 1.0, "seq-mean-token-mean", data)
    tg_calls = [c for c in calls if c.get("loss_mask") is not None
                and torch.equal(c["loss_mask"], tg_mask)]
    assert len(tg_calls) == 1
    assert tg_calls[0]["global_batch_size"] == 3.0
    assert tg_calls[0]["dp_size"] == 2.0
    assert "tg/loss" in metrics and "tg/rows" in metrics and "tg/tokens" in metrics


def test_tg_call_site_seq_mean_fails_loud_without_stamp(losses_ns):
    ns, _ = losses_ns
    data = {"response_mask": torch.tensor([[True, True], [False, False]]),
            "tg_mask": torch.tensor([[False, False], [True, True]]),
            "batch_num_seqs": 5, "dp_size": 2}
    with pytest.raises(RuntimeError, match="batch_num_tg_seqs"):
        _call_ppo(ns, 1.0, "seq-mean-token-mean", data)


def test_tg_call_site_token_mean_keeps_local_zero_safe_denominator(losses_ns):
    ns, calls = losses_ns
    tg_mask = torch.tensor([[False, False], [True, True]])
    data = {"response_mask": torch.tensor([[True, True], [False, False]]),
            "tg_mask": tg_mask}
    _call_ppo(ns, 1.0, "token-mean", data)
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


def test_tg_losses_are_clamped_like_the_opd_branch(losses_ns):
    # A negative per-token value (fp jitter) must be clamped to 0 before the
    # tg aggregation — the spy receives the clamped matrix.
    ns, calls = losses_ns
    tg_mask = torch.tensor([[True, True]])
    data = {"response_mask": torch.tensor([[False, False]]),
            "tg_mask": tg_mask, "batch_num_seqs": 1, "dp_size": 1,
            "batch_num_tg_seqs": 1}
    losses = torch.tensor([[-0.5, 2.0]])
    _call_ppo(ns, 1.0, "seq-mean-token-mean", data, losses=losses)
    tg_calls = [c for c in calls if torch.equal(c.get("loss_mask"), tg_mask)]
    assert (tg_calls[0]["loss_mat"] >= 0).all()


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


def test_config_tg_requires_forward_kl_topk_and_fsdp_backend(tmp_path):
    base = ["train.mix_tg_weight=1.0", "train.opd_lost_only=true"]
    with pytest.raises(ValueError, match="forward_kl_topk"):
        _load(tmp_path, "d", base + ["train.opd_kl_mode=reverse_kl_k3"])
    with pytest.raises(ValueError, match="fsdp"):
        _load(tmp_path, "e", base + ["train.teacher_scoring_backend=vllm"])


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
              "tg_role": 1, "tg_dropped_overflow": 0}
    on = build_curve_metrics(validate=False, research_enabled=False,
                             mixed_tg=True, fields=fields)
    assert on["tg_role"] == 1.0 and on["tg_dropped_nodoc"] == 0.0
    assert "tg_dropped_lostonly" in on
    off = build_curve_metrics(validate=False, research_enabled=False,
                              mixed_tg=False, fields=fields)
    assert "tg_role" not in off


# ---------------------------------------------------------------------------
# 7. tg rows leave the on-policy machinery (0d653b0 self-review fixes)
# ---------------------------------------------------------------------------
def test_tg_rows_excluded_from_metrics_grpo_group_and_padding_template():
    trainer = (_VERL / "trainer" / "ppo" / "v1" / "trainer_base.py").read_text(encoding="utf-8")
    # Behaviour metrics: a tg row is a WON trajectory whose share grows with
    # training — leaking it into critic/score/mean would fake improvement on
    # the very curve that gates the run.
    assert 'in ("sft", "research", "tg")' in trainer
    # Own GRPO group: under the won split the task's bare rows are all lost;
    # a reward-1.0 group-mate would hand each a negative advantage baseline.
    assert 'if role in ("sft", "research", "tg") else uid' in trainer
    padding = (_VERL / "trainer" / "ppo" / "padding_utils.py").read_text(encoding="utf-8")
    assert 'if "tg_mask" in template_sample:' in padding
    assert 'template_sample["tg_mask"] = torch.zeros_like(response_mask)' in padding
