"""Vendored-verl export integrity — every trace-opd patch must be present in
the exported tree ON ITS ACTUAL EXECUTION PATH.

Lesson from the f8f26d5 incident (CD-006, 2026-07-24): the rollout_extra
curves patch sat only in the V0 trainer (ray_trainer.py) while training
actually runs the V1 trainer (TaskRunnerV1 -> v1/trainer_base.py), so the
curves silently never appeared. These assertions grep the vendored files the
launcher puts on PYTHONPATH, so a stale export or a patch landing on a dead
path fails loudly here.
"""
from pathlib import Path

_VERL = Path(__file__).resolve().parents[1] / "verl" / "verl"


def _read(rel):
    return (_VERL / rel).read_text(encoding="utf-8")


def test_rollout_extra_curves_on_v1_trainer_path():
    # e8f697c: import + call site must both be in the V1 trainer, which is
    # what training executes (V0 keeps its copy for parity, not sufficient).
    src = _read("trainer/ppo/v1/trainer_base.py")
    assert src.count("compute_rollout_extra_metrics") >= 2
    assert "def compute_rollout_extra_metrics" in _read("trainer/ppo/metric_utils.py")  # f8f26d5


def test_micro_batch_budget_is_packing_target_not_assert():
    # 6d77dc7: the old hard assert would kill the whole run on one long-tail
    # trajectory; the export must carry the clamp+warning form.
    assert "below longest sequence" in _read("utils/seqlen_balancing.py")


def test_teacher_scoring_keeps_sticky_routing_key():
    # 79d4252: self-teacher scoring shares the per-(step,task) route key.
    assert "teacher_routing_key" in _read("experimental/agent_loop/agent_loop.py")


def test_packing_budget_not_raised_to_longest_sequence():
    # c48568e (R1, amends 6d77dc7): raising the budget shrinks
    # num_micro_batches for the whole batch (~1.3x longest-sequence peak, the
    # step-5 OOM amplifier). The raise line must stay gone.
    src = _read("utils/seqlen_balancing.py")
    assert "max_token_len = int(max_seq_len)" not in src
    assert "packing budget stays" in src


def test_wake_up_hands_cache_back_before_resume_weights():
    # c48568e (R2): resume weights must be preceded by aggressive_empty_cache,
    # or vLLM's CuMemAllocator cannot get physical pages (cumem OOM, step-5).
    src = _read("workers/engine_workers.py")
    before, _, after = src.partition("# 1. resume rollout memory")
    assert after, "resume-weights anchor comment disappeared — re-check R2"
    assert "aggressive_empty_cache(force_sync=True)" in before.rsplit("def ", 1)[-1]


def test_agent_loop_hang_guard_on_tq_path():
    # 58cd07e (R3): per-prompt wall-clock timeout wraps the gather on the TQ
    # path (the executed one); a permanent vLLM hang must cost one prompt,
    # not deadlock the run.
    src = _read("trainer/ppo/v1/agent_loop_tq.py")
    assert "TRACE_AGENT_TASK_TIMEOUT_S" in src
    assert "asyncio.wait_for(asyncio.gather(*tasks)" in src


def test_fsdp_teacher_scoring_backend_present_and_wired():
    # ce96673 (R11a): the fsdp scoring module exists, update_actor calls it,
    # the rollout side skips the vLLM call, and the env switch rides the Ray
    # runtime env (workers do not inherit the process env).
    fsdp = _read("trainer/distillation/teacher_topk_fsdp.py")
    assert "TRACE_TEACHER_SCORING_BACKEND" in fsdp
    assert "teacher_mode" in fsdp  # codex H1 provenance, fail-loud
    assert "compute_teacher_topk_fsdp" in _read("workers/engine_workers.py")
    assert "TRACE_TEACHER_SCORING_BACKEND" in _read("experimental/agent_loop/agent_loop.py")
    assert "TRACE_TEACHER_SCORING_BACKEND" in _read("trainer/constants_ppo.py")


def test_identity_temperature_division_is_skipped():
    # f88db6c (617817c): logits / 1.0 allocated a second full-size fp32 tensor
    # (13.5GiB at 22k tokens) — the step-16 OOM trigger.
    src = _read("workers/engine/fsdp/transformer_impl.py")
    assert "logits = logits / temperature.clamp" not in src
    assert "logits = logits / _temp" in src


def test_distillation_diagnostics_survive_empty_batch():
    # 28a82d5 (01f0fae + 5f9a29d): both min()/max() diagnostic sites must be
    # guarded; a cancelled prompt's empty replicas must not kill the run.
    src = _read("trainer/distillation/losses.py")
    assert src.count(".numel() == 0") >= 2


def test_mixed_objective_terms_are_wired_and_default_off():
    # a61c654: the RL and SFT arms exist, default to 0 (byte-identical to pure
    # OPD), and the expert rows are excluded from the on-policy machinery at
    # every point they could leak into it.
    cfg = _read("workers/config/distillation.py")
    assert "rl_loss_coef: float = 0.0" in cfg
    assert "sft_loss_coef: float = 0.0" in cfg

    losses = _read("trainer/distillation/losses.py")
    assert "def sft_ce_loss" in losses
    assert "sft_mask" in losses

    # response_mask cleared at TQ put time — this is what keeps the OPD scale
    # unchanged when the SFT arm is switched on, so the ablation arms compare.
    tq = _read("trainer/ppo/v1/agent_loop_tq.py")
    assert 'field["sft_mask"]' in tq
    assert 'field["response_mask"] = torch.zeros_like(response_mask)' in tq

    trainer = _read("trainer/ppo/v1/trainer_base.py")
    assert ":sft" in trainer  # own GRPO group
    assert "on_policy_mask" in trainer  # 93a0109: metrics exclude expert rows
    assert "metrics_batch = batch.select_idxs(on_policy_mask)" in trainer


def test_microscope_skips_rows_with_no_distillation():
    # 93a0109: an all-zero mask row (expert sample, or a prompt the hang guard
    # cancelled) would make the offline span-vs-mask integrity check misfire.
    src = _read("trainer/distillation/losses.py")
    assert "if not bool(mask[i, :length].any()):" in src


def test_signal_enhancement_triplet_removed():
    # User decision 2026-07-26 (requirements 3-D): K1/K2/K3 are REMOVED, not
    # disabled — no implementation, no config fields, anywhere in the export.
    for rel in (
        "trainer/distillation/losses.py",
        "trainer/distillation/teacher_topk_fsdp.py",
        "workers/config/distillation.py",
        "experimental/agent_loop/agent_loop.py",
    ):
        src = _read(rel)
        for marker in ("loss_soft_cap", "kl_weight_alpha", "kl_weight_cap",
                       "teacher_block_starts", "teacher_doc_ids"):
            assert marker not in src, f"{marker} still present in {rel}"
