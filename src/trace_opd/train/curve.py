"""Per-sample scalars for the swanlab curve channel (user request 2026-07-23:
richer training-time observability of the mechanism).

The dict returned here goes into ``extra_fields["reward_extra_info"]``; verl
turns every numeric key into a curve — ``rollout_extra/<key>/mean`` per train
step (verl patch f8f26d5) and ``val-aux/<data_source>/<key>/mean@1`` per val.

HOMOGENEITY CONTRACT: verl expands the batch using the FIRST sample's key set
and indexes every sample with it, so all samples of one generate_sequences
call must produce the same keys. The three shapes below (train+research /
train without research / val) are each internally homogeneous because
``validate`` and ``research_enabled`` are batch-wide constants.

Key curve reading (mechanism health): ``research_final_success`` (doc-guided
experiment success) vs ``critic/score/mean`` (bare on-policy success, free
from the rollout's own won signal — no extra eval pass) — the gap IS the
document's privileged gain; score rising while the gap narrows is the
internalization signature.
"""
from __future__ import annotations

from typing import Any

_ALWAYS = ("env_steps", "invalid_format", "truncated")
_RESEARCH = (
    "research_final_success",
    "doc_missing",
    "research_error",
    "doc_words",
    "research_experiments",
    "research_episodes",
)
# Self-research channel (docs/self-research-design.md). Research rows fill
# their own values; bare/expert rows fill the group aggregates stamped by
# _await_research_group (all_fail/all_pass/best_p) and 0 for the rest — the
# homogeneity contract holds because research_rl is a batch-wide constant.
_RESEARCH_RL = (
    "research_rl_role",          # 1 on research rows: denominator for ratios
    "research_rl_reward",        # p_i (doc's K-rollout success rate)
    "research_rl_finalized",     # session produced a document
    "research_rl_traj_dropped",  # sample degraded (empty/overlen), mask zeroed
    "research_rl_group_all_fail",  # group stats, stamped on bare rows
    "research_rl_group_all_pass",
    "research_rl_best_p",
)
# Mixed objective, SFT arm (docs/mixed-objective-design.md).
_MIXED_TG = (
    "tg_role",              # 该行是否为完整 tg 行(批均值 = won 转换率)
    "tg_dropped_overflow",  # doc 前缀溢出而降级的占位行
    "tg_dropped_nodoc",     # research 失败无 doc 的占位行
    "tg_dropped_lostonly",  # tg 臂关、仅 opd_lost_only 的占位行(T1' 形态)
    "tg_dropped_unhealthy", # 健康门拦截的膨胀 won 行(mix_tg_health_gate)
    # 标签质量(预注册锐判据在线版;仅完整 tg 行非零):
    #   主判读 = mean(tg_label_thought_cpt) ÷ mean(tg_role)
    #          = 标签行的 trajectory-mean cpt,与离线 won_traj_evolution
    #            同 estimand(codex impl-review P1-5,健康基线 284–315 可直用);
    #   chars ÷ turns 两曲线相除 = turn-weighted 总量视图,作辅助。
    "tg_label_thought_chars",
    "tg_label_thought_turns",
    "tg_label_thought_cpt",
)
_MIXED_SFT = (
    "sft_role", "sft_won", "sft_accepted", "sft_dropped_ref",
    # An expert slot that could not produce supervision. Kept as a slot (masked
    # to nothing) rather than re-used as a bare rollout: research failure
    # correlates with task difficulty, so converting it would feed the SFT arms
    # extra on-policy data on exactly the hard tasks.
    "sft_missing_no_doc", "sft_missing_env_error",
)


def build_curve_metrics(
    *,
    validate: bool,
    research_enabled: bool,
    fields: dict[str, Any],
    mixed_sft: bool = False,
    mixed_tg: bool = False,
    research_rl: bool = False,
) -> dict[str, float]:
    """Assemble the per-sample curve scalars from the sample's extra_fields.

    Call AFTER the teacher-prefix step so ``doc_missing`` reflects the final
    state (a doc overflowing the teacher context degrades to bare and flips
    doc_missing to 1 in extra_fields after research reported 0).

    ``mixed_sft`` is a batch-wide constant (the config's SFT weight), so adding
    its keys keeps the homogeneity contract. They are per-BATCH means over all
    replicas, expert and bare alike — read them as ratios: ``sft_won/sft_role``
    is the document-guided success rate (the PG-7 document-quality signal) and
    ``sft_accepted/sft_role`` is what actually reaches the SFT loss.
    """
    metrics = {key: float(fields[key]) for key in _ALWAYS}
    if not validate and research_enabled:
        for key in _RESEARCH:
            metrics[key] = float(fields.get(key, 0) or 0)
    if not validate and mixed_sft:
        for key in _MIXED_SFT:
            metrics[key] = float(fields.get(key, 0) or 0)
    if not validate and mixed_tg:
        for key in _MIXED_TG:
            metrics[key] = float(fields.get(key, 0) or 0)
    if not validate and research_rl:
        for key in _RESEARCH_RL:
            metrics[key] = float(fields.get(key, 0) or 0)
    return metrics
