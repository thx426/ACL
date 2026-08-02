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


def test_padding_template_shrinks_the_sft_mask_with_the_sequence():
    # construct_minimal_padding_template clones every field off an arbitrary
    # real row and then overrides a fixed list. sft_mask is per-token like
    # response_mask, so leaving it out ships a full-length (possibly expert)
    # mask on a 1-token padding row: a shape mismatch in the SFT loss, or —
    # when the widths happen to agree — a silently inflated expert-token count.
    src = _read("trainer/ppo/padding_utils.py")
    assert 'if "sft_mask" in template_sample:' in src
    assert 'template_sample["sft_mask"] = torch.zeros_like(response_mask)' in src
    # ...and the loss refuses a mask that is out of step with the responses,
    # so any future field-synthesizing path fails loud instead of miscounting.
    assert "does not match the response layout" in _read("trainer/distillation/losses.py")


def test_rl_arm_shares_the_opd_normalisation():
    # 6de6693: ppo_loss refills global_batch_info from the batch, so its PG term
    # normalises over the global token count while the OPD term is pinned to the
    # per-micro-batch mean. Those differ by the micro-batch count (~60 here), so
    # mix_rl_weight=1.0 would really have been OPD:RL of 60:1. The RL term must
    # therefore go through task_reward_pg, not ppo_loss.
    src = _read("trainer/distillation/losses.py")
    assert "def task_reward_pg" in src
    assert "policy_loss, policy_metrics = task_reward_pg(config, model_output, data)" in src
    # ...and inside the cleared-snapshot block, i.e. the same convention as OPD.
    block = src[src.index("if rl_loss_coef > 0.0:"):src.index("def task_reward_pg")]
    assert "config.global_batch_info.clear()" in block
    assert "task_reward_pg(" in block
    # The share of the batch actually carrying an RL gradient is measured, not
    # assumed: with a binary reward most groups are all-success or all-failure.
    assert "rl/nonzero_advantage_frac" in src


def test_microscope_skips_rows_with_no_distillation():
    # 93a0109: an all-zero mask row (expert sample, or a prompt the hang guard
    # cancelled) would make the offline span-vs-mask integrity check misfire.
    src = _read("trainer/distillation/losses.py")
    assert "if not bool(mask[i, :length].any()):" in src


def test_kl_mode_family_wired_on_the_executed_paths():
    # 0a6edfb + 6fb4145 + 48c8743: the four opd_kl_mode values all resolve to
    # code that exists in the export, on the paths training actually runs.
    losses = _read("trainer/distillation/losses.py")
    # top-k direction dispatch in the logits-processor entry, both names on the
    # (direction-agnostic) outer post-processor, and the gated objective.
    assert 'loss_mode == "reverse_kl_topk"' in losses
    assert '"forward_kl_topk", "reverse_kl_topk"' in losses
    assert "def compute_distillation_loss_gated_logp_shift" in losses
    assert "def _gated_logp_shift" in losses
    assert 'names=["reverse_kl_gated"]' in losses

    fsdp = _read("trainer/distillation/fsdp/losses.py")
    assert "def compute_reverse_kl_topk" in fsdp
    assert "tail_kl" in fsdp  # the (K+1)-bucket tail term, not a bare truncation

    cfg = _read("workers/config/distillation.py")
    assert "gate_beta: float = 5.0" in cfg
    assert "gate_beta must be finite and > 0" in cfg


def test_single_point_scoring_mode_on_the_fsdp_backend():
    # 0a6edfb: teacher_topk == 0 (the estimator family) must not hit torch.topk,
    # and the dummy width must follow the vLLM max(k, 1) convention.
    fsdp = _read("trainer/distillation/teacher_topk_fsdp.py")
    assert "def _score_rows_realized" in fsdp
    assert "width = max(topk, 1)" in fsdp
    # hardened alignment sentinel: first rows, not row 0 alone (a single
    # saturated glue token can round to logprob 0.0 in fp32).
    assert "lps_i[: min(8, R)]" in fsdp


def test_estimator_family_reaches_the_microscope():
    # 48c8743: under k3/gated the offline analysis must not silently go dark —
    # forward, the estimator family, and the gated mode all dump rows.
    losses = _read("trainer/distillation/losses.py")
    assert losses.count("_dump_microscope_rows(") >= 3


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

def test_self_research_channel_wired_on_the_executed_paths():
    # 6e524ed + e3bd3bf + af6a8d3: every touchpoint of the research channel
    # must be present on the paths training actually runs.
    losses = _read("trainer/distillation/losses.py")
    assert "def research_pg_loss" in losses
    assert "research_num_tokens" in losses            # batch-global denominator
    assert 'data["research_only"]' in losses          # separate-mode short-circuit
    assert "research_loss_coef" in losses

    tq = _read("trainer/ppo/v1/agent_loop_tq.py")
    assert 'field["research_mask"] = response_mask' in tq
    assert '"research_arm"' in tq                     # uniform allocation stamp

    trainer = _read("trainer/ppo/v1/trainer_base.py")
    assert "_grouped_uid" in trainer                  # own GRPO group per role
    assert 'if role in ("sft", "research", "tg")' in trainer  # (+tg 0d653b0)
    assert "_research_channel_enabled" in trainer
    assert "research_update_mode" in trainer          # joint/separate routing
    assert "research_warmup_steps" in trainer
    assert "num_mini_batch" in trainer                # af6a8d3 subset sizing fix
    assert 'in ("sft", "research", "tg")' in trainer  # e3bd3bf metrics exclusion (+tg 0d653b0)

    engine = _read("workers/engine/fsdp/transformer_impl.py")
    assert "research_num_tokens" in engine            # all-reduced pre-micro-batch

    fsdp = _read("trainer/distillation/teacher_topk_fsdp.py")
    assert 'in ("sft", "research", "tg_skip")' in fsdp  # no scoring forward burned
    # (teacher grounding: tg_skip placeholders ride the same dummy channel)

    agent = _read("experimental/agent_loop/agent_loop.py")
    assert '"teacher_mode"] = "research"' in agent

    padding = _read("trainer/ppo/padding_utils.py")
    assert '"research_mask" in template_sample' in padding

    cfg = _read("workers/config/distillation.py")
    assert "research_loss_coef: float = 0.0" in cfg
    assert 'research_update_mode: str = "joint"' in cfg

def test_codex_review_fixes_present_on_the_executed_paths():
    # 13741a2: the three P0s and the P1 fixes from the codex review, pinned.
    fsdp = _read("trainer/distillation/teacher_topk_fsdp.py")
    # P0-1: the M4 structural gate must admit research rows, or the skip
    # branch behind it is unreachable and every research batch dies.
    # (608b6aa extends the whitelist with the teacher-grounding modes.)
    assert '("privileged", "bare", "sft", "research", "tg", "tg_skip")' in fsdp

    trainer = _read("trainer/ppo/v1/trainer_base.py")
    # P0-2: the memory shortcut must stay off while the research term is on
    # (it reads log_probs on every update, warmup exec sub-batches included).
    dist_only = trainer.split("distillation_only = (")[1][:1200]
    assert "research_loss_coef" in dist_only
    # P1-7/P1-8: the second update neither advances the scheduler nor runs on
    # a zero-signal split; P1-5: an all-research split falls back unsplit.
    assert '"advance_lr_scheduler"] = False' in trainer
    assert "skipped_zero_signal" in trainer
    assert "falling back to the" in trainer
    # P2-11: groups below two trainable candidates carry no gradient.
    assert "trainable = mask_padded.sum(dim=-1) > 0" in trainer

    workers = _read("workers/engine_workers.py")
    # P1-7: the last-mini-batch scheduler stamp is now gated.
    assert "advance_lr_scheduler" in workers
    assert "update_lr_scheduler=advance_lr_scheduler" in workers

    losses = _read("trainer/distillation/losses.py")
    # P0-3 + P2-12: dp compensation and the clamped log-ratio.
    assert "* dp_size" in losses.split("def research_pg_loss")[1].split("def ")[0]
    assert "clamp(min=-20.0, max=20.0)" in losses

def test_self_review_hardening_present():
    # 92f8ac9: vLLM scoring skip for sft/research rows, entropy off and the
    # divisibility guard on the separate research split.
    agent = _read("experimental/agent_loop/agent_loop.py")
    assert 'get("sample_role") in ("sft", "research")' in agent  # dummy columns, no forward

    trainer = _read("trainer/ppo/v1/trainer_base.py")
    assert '"calculate_entropy"] = False' in trainer
    assert "skipped_uneven_split" in trainer


def test_observation_packs_wired_on_the_executed_paths():
    # 5389497: the distribution-quartet collection points.
    obs = _read("trainer/distillation/observation.py")
    assert "TRACE_OBS_DIR" in obs
    assert "def collect_micro_batch" in obs
    assert "loss_mode" in obs                       # self-contained audit metadata

    fsdp = _read("trainer/distillation/fsdp/losses.py")
    assert fsdp.count("_observation_payload(") >= 3  # def + both topk functions
    assert '"obs": obs' in fsdp

    losses = _read("trainer/distillation/losses.py")
    assert "_collect_observation_rows(" in losses
    assert "log_prob_min_clamp=distillation_config" in losses

def test_observation_review_fixes_present():
    # 9172da4: the codex-review fixes on the observation packs.
    losses = _read("trainer/distillation/losses.py")
    assert "NonTensorStack" in losses or "tolist" in losses  # P0-1 unwrap
    assert "_step_wanted(global_step)" in losses             # P1-11 early gate

    obs = _read("trainer/distillation/observation.py")
    assert "atexit.register(flush_all)" in obs               # P1-9
    assert 'row.setdefault("sp_rank"' in obs                 # P2-4

    consts = _read("trainer/constants_ppo.py")
    assert '"TRACE_OBS_DIR", "TRACE_OBS_EVERY"' in consts    # P1-10

