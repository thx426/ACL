"""Credit-assignment unit of the distillation loss, tested against the VENDORED
sources (the math training actually runs, not a transcription).

Context (CD-005 review of the CD-008 loss_agg_mode switch, 2026-07-29):
seq-mean-* modes divide by a sequence count. Without a batch-GLOBAL count the
vendored ``agg_loss`` falls back to the LOCAL micro-batch row count — and the
balanced partitioner packs long rows into few-row micro-batches, handing long
rows back almost exactly the length-proportional weight seq-mean is chosen to
remove; an all-masked micro-batch divides by zero. The fix: the engine stamps
the all-reduced non-empty-row count (``batch_num_seqs``) and the distillation
call site reads it fail-loud. token-mean keeps its local clamped denominator
byte-for-byte — every historical run trained on that contract.
"""
import ast
import math
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_VERL = Path(__file__).resolve().parents[1] / "verl" / "verl"


# ---------------------------------------------------------------------------
# agg_loss itself — extracted from the vendored core_algos.py by name, executed
# with only its two dependencies (torch, verl_F.masked_sum) provided.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def agg_loss():
    src = (_VERL / "trainer" / "ppo" / "core_algos.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "agg_loss")
    fn.decorator_list = []
    mod = ast.Module(body=[fn], type_ignores=[])
    from typing import Any, Optional

    verl_F = types.SimpleNamespace(
        masked_sum=lambda mat, mask, axis=None: (mat * mask).sum())
    ns = {"torch": torch, "verl_F": verl_F, "Optional": Optional, "Any": Any}
    exec(compile(mod, "vendored_agg_loss.py", "exec"), ns)
    return ns["agg_loss"]


def test_seq_mean_with_global_count_weighs_rows_equally(agg_loss):
    # Row 0: 3 tokens of loss 2.0 (row mean 2.0). Row 1: 1 token of loss 8.0
    # (row mean 8.0). Equal per-sequence weight means the 3x longer row does
    # NOT count 3x — global count 4 and dp 2 give (2+8)/4*2 = 5.
    loss = torch.tensor([[2.0, 2.0, 2.0], [8.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    out = agg_loss(loss_mat=loss, loss_mask=mask,
                   loss_agg_mode="seq-mean-token-mean",
                   global_batch_size=4.0, dp_size=2.0)
    assert out.item() == pytest.approx((2.0 + 8.0) / 4.0 * 2.0, rel=1e-5)


def test_seq_mean_all_masked_rows_contribute_zero_with_global_count(agg_loss):
    loss = torch.tensor([[5.0, 5.0], [7.0, 7.0]])
    mask = torch.zeros(2, 2)
    out = agg_loss(loss_mat=loss, loss_mask=mask,
                   loss_agg_mode="seq-mean-token-mean",
                   global_batch_size=4.0, dp_size=1.0)
    assert out.item() == 0.0 and not math.isnan(out.item())


def test_seq_mean_local_fallback_is_the_defect_the_call_site_must_avoid(agg_loss):
    """Pin the upstream semantics the fix routes around: without a global
    count the denominator is the LOCAL non-empty row count, and an all-masked
    micro-batch is 0/0 = NaN. If upstream ever changes this, the call-site
    guard can be revisited — until then it must stay."""
    loss = torch.tensor([[2.0, 2.0, 2.0], [8.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    local = agg_loss(loss_mat=loss, loss_mask=mask,
                     loss_agg_mode="seq-mean-token-mean")
    assert local.item() == pytest.approx((2.0 + 8.0) / 2.0, rel=1e-5)  # /local rows

    all_masked = agg_loss(loss_mat=loss, loss_mask=torch.zeros(2, 3),
                          loss_agg_mode="seq-mean-token-mean")
    assert math.isnan(all_masked.item())


def test_token_mean_semantics_unchanged(agg_loss):
    loss = torch.tensor([[2.0, 2.0, 2.0], [8.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    out = agg_loss(loss_mat=loss, loss_mask=mask, loss_agg_mode="token-mean",
                   batch_num_tokens=mask.sum())
    assert out.item() == pytest.approx((2.0 * 3 + 8.0) / 4.0, rel=1e-5)


# ---------------------------------------------------------------------------
# The distillation call site — vendored trainer/distillation/losses.py executed
# with its verl imports stubbed and agg_loss replaced by a spy, so the test
# reads exactly which denominators the shipped wiring hands over.
# ---------------------------------------------------------------------------
@pytest.fixture()
def distillation_ns():
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


def _call_distillation(ns, loss_agg_mode, data):
    losses = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    ns["get_distillation_loss_fn"] = lambda mode: (
        lambda config, distillation_config, model_output, data: (losses, {}))
    config = types.SimpleNamespace(loss_agg_mode=loss_agg_mode,
                                   global_batch_info={})
    loss_cfg = types.SimpleNamespace(loss_mode="fake", loss_max_clamp=None,
                                     use_policy_gradient=False)
    distillation_config = types.SimpleNamespace(distillation_loss=loss_cfg)
    return ns["distillation_loss"](config, distillation_config, {}, data)


def test_call_site_hands_seq_mean_the_engine_stamped_global_count(distillation_ns):
    ns, calls = distillation_ns
    data = {"response_mask": torch.tensor([[True, True], [True, False]]),
            "batch_num_seqs": 7, "dp_size": 2}
    _call_distillation(ns, "seq-mean-token-mean", data)
    kw = calls[-1]
    assert kw["loss_agg_mode"] == "seq-mean-token-mean"
    assert kw["global_batch_size"] == 7.0
    assert kw["dp_size"] == 2.0


def test_call_site_fails_loud_when_the_engine_stamp_is_missing(distillation_ns):
    ns, _ = distillation_ns
    data = {"response_mask": torch.tensor([[True, True], [True, False]])}
    with pytest.raises(RuntimeError, match="batch_num_seqs"):
        _call_distillation(ns, "seq-mean-token-mean", data)


def test_call_site_token_mean_keeps_the_local_clamped_denominator(distillation_ns):
    """The historical contract: token-mean gets ONLY the local clamped token
    count — no global_batch_size, no dp_size — so every existing run's
    behaviour is byte-for-byte unchanged."""
    ns, calls = distillation_ns
    mask = torch.tensor([[True, True], [True, False]])
    data = {"response_mask": mask, "batch_num_seqs": 7, "dp_size": 2}
    _call_distillation(ns, "token-mean", data)
    kw = calls[-1]
    assert "global_batch_size" not in kw and "dp_size" not in kw
    assert kw["batch_num_tokens"].item() == mask.sum().item()


# ---------------------------------------------------------------------------
# The engine stamp itself — the vendored forward_backward_batch must compute
# batch_num_seqs next to batch_num_tokens (textual pin; the numeric semantics
# are covered above and by CD-006's production smoke).
# ---------------------------------------------------------------------------
def test_engine_stamps_batch_num_seqs_next_to_batch_num_tokens():
    src = (_VERL / "workers" / "engine" / "fsdp" / "transformer_impl.py").read_text(
        encoding="utf-8")
    assert "batch_num_seqs=batch_num_seqs.item()" in src
    body = src.split("def forward_backward_batch", 1)[1]
    assert body.index("batch_num_tokens=batch_num_tokens.item()") \
        < body.index("batch_num_seqs=batch_num_seqs.item()") \
        < body.index("prepare_micro_batches(")


# ---------------------------------------------------------------------------
# Fingerprint and launcher wiring (trace_opd side).
# ---------------------------------------------------------------------------
def test_fingerprint_baseline_is_legacy_token_mean_not_the_new_default(tmp_path):
    from test_mixed_objective import _TEMPLATE_YAML, _write

    from trace_opd.common.config import load_config, objective_fingerprint

    def fp(name, overrides=None):
        cfg = load_config(_write(tmp_path / name, "c.yaml", _TEMPLATE_YAML),
                          overrides=overrides)
        return objective_fingerprint(cfg)

    seq_default = fp("d")  # default is seq-mean-token-mean since 2026-07-29
    seq_explicit = fp("e", ["train.loss_agg_mode=seq-mean-token-mean"])
    token = fp("t", ["train.loss_agg_mode=token-mean"])
    # The value decides, not whether it was spelled out ...
    assert seq_default == seq_explicit
    # ... and the two credit-assignment units are different objectives.
    assert token != seq_default


def test_launcher_carries_loss_agg_mode_to_verl(tmp_path):
    import importlib.util

    from trace_opd.common.config import load_config

    launcher = Path(__file__).resolve().parents[1] / "scripts" / "launch_train.py"
    spec = importlib.util.spec_from_file_location("launch_train_agg", launcher)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from test_mixed_objective import _TEMPLATE_YAML, _write
    cfg = load_config(_write(tmp_path, "c.yaml", _TEMPLATE_YAML))
    cmd = mod.build_command(cfg, tmp_path / "run")
    assert "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean" in cmd

    cfg_t = load_config(_write(tmp_path / "t", "c.yaml", _TEMPLATE_YAML),
                        overrides=["train.loss_agg_mode=token-mean"])
    cmd_t = mod.build_command(cfg_t, tmp_path / "run2")
    assert "actor_rollout_ref.actor.loss_agg_mode=token-mean" in cmd_t


def test_call_site_fails_loud_when_dp_size_is_missing(distillation_ns):
    """P1-1: dp_size is stamped unconditionally by the engine — absence means
    a broken engine, and a silent 1.0 fallback would land the loss at 1/dp."""
    ns, _ = distillation_ns
    data = {"response_mask": torch.tensor([[True, True], [True, False]]),
            "batch_num_seqs": 7}
    with pytest.raises(RuntimeError, match="dp_size"):
        _call_distillation(ns, "seq-mean-token-mean", data)


class _TD(dict):
    """Just enough TensorDict surface for task_reward_pg."""
    def select(self, *fields):
        return _TD({k: self[k] for k in fields})

    def to_padded_tensor(self):
        return self


def _call_aggregator(ns, loss_agg_mode, *, rl_coef=0.0, sft_coef=0.0,
                     pg_snapshots=None):
    losses = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    ns["get_distillation_loss_fn"] = lambda mode: (
        lambda config, distillation_config, model_output, data: (losses, {}))

    def spy_policy_fn(**kw):
        if pg_snapshots is not None:
            pg_snapshots.append(dict(kw["config"].global_batch_info))
        return torch.tensor(0.5), {}

    ns["get_policy_loss_fn"] = lambda mode: spy_policy_fn
    config = types.SimpleNamespace(loss_agg_mode=loss_agg_mode,
                                   global_batch_info={},
                                   policy_loss={"loss_mode": "vanilla"})
    loss_cfg = types.SimpleNamespace(
        loss_mode="fake", loss_max_clamp=None, use_policy_gradient=False,
        use_task_rewards=False, rl_loss_coef=rl_coef, sft_loss_coef=sft_coef,
        research_loss_coef=0.0, opd_loss_coef=1.0)
    distillation_config = types.SimpleNamespace(distillation_loss=loss_cfg)
    mask = torch.tensor([[True, True], [True, False]])
    data = _TD({"response_mask": mask,
                "old_log_probs": torch.zeros(2, 2),
                "advantages": torch.tensor([[1.0, 0.0], [0.0, -1.0]]),
                "batch_num_seqs": 5, "dp_size": 2})
    model_output = {"log_probs": torch.zeros(2, 2)}
    return ns["distillation_ppo_loss"](config, distillation_config,
                                       model_output, data), config


def test_aggregator_stages_one_denominator_for_both_terms_under_seq_mean(distillation_ns):
    """P0-1: with RL on and seq-mean, the distillation agg_loss call AND the
    policy_loss_fn must see the SAME engine-stamped global sequence count —
    nominal 1:1 coefficients mean an actual 1:1."""
    ns, calls = distillation_ns
    snapshots = []
    (_, config) = _call_aggregator(ns, "seq-mean-token-mean", rl_coef=1.0,
                                   pg_snapshots=snapshots)
    distill_kw = calls[-1]
    assert distill_kw["global_batch_size"] == 5.0 and distill_kw["dp_size"] == 2.0
    assert snapshots == [{"global_batch_size": 5.0, "dp_size": 2.0}]
    # the staged window is restored afterwards — nothing leaks across batches
    assert config.global_batch_info == {}


def test_aggregator_token_mean_keeps_both_terms_on_the_local_contract(distillation_ns):
    ns, calls = distillation_ns
    snapshots = []
    _call_aggregator(ns, "token-mean", rl_coef=1.0, pg_snapshots=snapshots)
    distill_kw = calls[-1]
    assert "global_batch_size" not in distill_kw and "dp_size" not in distill_kw
    assert snapshots == [{}]


def test_aggregator_blocks_sft_arms_under_seq_mean(distillation_ns):
    """P0-2 guard: the SFT term's local expert-token mean is incommensurable
    with a global per-sequence distillation mean — refuse loud, don't drift."""
    ns, _ = distillation_ns
    with pytest.raises(RuntimeError, match="sft_loss_coef"):
        _call_aggregator(ns, "seq-mean-token-mean", sft_coef=0.5)


def test_config_rejects_the_unruled_aggregation_modes(tmp_path):
    """P1-2: only the two ruled-on units load; verl's -sum/-sum-norm are
    refused fail-early (the -norm scale factor is packing-dependent)."""
    from test_mixed_objective import _TEMPLATE_YAML, _write

    from trace_opd.common.config import load_config

    for bad in ("seq-mean-token-sum", "seq-mean-token-sum-norm"):
        with pytest.raises(ValueError, match="loss_agg_mode"):
            load_config(_write(tmp_path / bad, "c.yaml", _TEMPLATE_YAML),
                        overrides=[f"train.loss_agg_mode={bad}"])
