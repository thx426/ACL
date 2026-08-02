# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator]) != 1:
            raise ValueError(
                f"Expected only one of use_estimator, use_topk, but got {self.use_estimator=}, {self.use_topk=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    if response_mask.is_nested:
        distillation_losses_response = distillation_losses[response_mask.bool().to_padded_tensor(False)]
    else:
        distillation_losses_response = distillation_losses[response_mask.bool()]
    # Same empty-batch hazard as the mass diagnostics below: once the agent-loop
    # guard cancels a stuck prompt, its replicas arrive with an empty response
    # and the mask selects nothing. min()/max() raise on that and kill the run,
    # so report the range as absent instead.
    if distillation_losses_response.numel() == 0:
        return {}
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
    }


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    loss_mode = distillation_config.distillation_loss.loss_mode
    # trace-opd teacher grounding (decoupling ruling 2026-08-02): build the
    # packed per-token tg flag (full P+R span per tg row) — it tells the
    # standalone tg formula which positions belong to tg rows. Zero-cost when
    # no row is a tg row.
    tg_token_mask = None
    extras = data.get("extra_fields", None)
    if extras is not None:
        extras_list = list(extras.tolist() if hasattr(extras, "tolist") else extras)
        # codex impl-review P0-1: the predicate must MATCH the scorer's
        # (teacher_mode == "tg"), not sample_role — dropped-won placeholders
        # (nodoc/overflow/health-gate) carry sample_role "tg" with
        # teacher_mode "tg_skip" and NO label; keying on sample_role makes a
        # placeholder-only shard demand a label channel the scorer never
        # shipped (RuntimeError inside the forward -> collective hang).
        if any(isinstance(e, dict) and e.get("teacher_mode") == "tg" for e in extras_list):
            prompts, responses = data["prompts"], data["responses"]
            if not (getattr(prompts, "is_nested", False) and getattr(responses, "is_nested", False)):
                raise RuntimeError(
                    "teacher-grounding rows require the nested (THD) batch layout - "
                    "the dense path does not carry the per-row tg span flags"
                )
            p_lens = prompts.offsets().diff().tolist()
            r_lens = responses.offsets().diff().tolist()
            flags = [
                torch.full((int(pl) + int(rl),),
                           bool(isinstance(e, dict) and e.get("teacher_mode") == "tg"),
                           dtype=torch.bool)
                for pl, rl, e in zip(p_lens, r_lens, extras_list)
            ]
            tg_token_mask = torch.cat(flags).unsqueeze(0).to(student_logits.device)
    match config.strategy:
        # VeOmni uses FSDP2 internally, so its loss computation is identical to FSDP.
        # fsdp2 shares the FSDP data layout as well (H3 pilot fix: the upstream
        # match missed it because the official OPD examples use k1+PG, which
        # never reaches the top-k logits-processor path).
        case "fsdp" | "fsdp2" | "veomni":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            if loss_mode == "reverse_kl_topk":
                distillation_loss_fn = fsdp_losses.compute_reverse_kl_topk
            else:
                distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
        case "megatron":
            if tg_token_mask is not None:
                raise NotImplementedError(
                    "teacher-grounding rows are implemented for the FSDP family only"
                )
            import verl.trainer.distillation.megatron.losses as megatron_losses

            if loss_mode == "reverse_kl_topk":
                raise NotImplementedError("reverse_kl_topk is implemented for the FSDP family only")
            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    outputs = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"],
        teacher_topk_ids=data["teacher_ids"],
        config=distillation_config,
        data_format=data_format,
    )
    # trace-opd teacher grounding (decoupling ruling 2026-08-02): the tg term
    # runs its OWN per-token formula on its OWN label channel — the OPD
    # formula above never sees a tg label, and the tg soft-CE never inherits
    # OPD clamps. A micro-batch without tg rows still ships a zero matrix so
    # the aggregation side keeps one uniform read path (its all-false tg_mask
    # contributes zero tokens over the batch-global denominator).
    tg_loss_coef = float(getattr(distillation_config.distillation_loss, "tg_loss_coef", 0.0) or 0.0)
    if tg_loss_coef > 0.0 and config.strategy in ("fsdp", "fsdp2", "veomni"):
        import verl.trainer.distillation.fsdp.losses as fsdp_losses

        if tg_token_mask is None:
            outputs["tg_losses"] = torch.zeros(
                student_logits.shape[:2], dtype=torch.float32, device=student_logits.device
            )
        else:
            for key in ("tg_label_ids", "tg_label_logprobs"):
                if key not in data.keys():
                    raise RuntimeError(
                        f"tg rows present but '{key}' is missing — the scorer did not ship "
                        "the decoupled tg label channel (teacher_topk_fsdp)"
                    )
            # The tg observation pack (label p̄ / q-on-support / soft-CE
            # matrix) rides the same obs stream — the microscope's tg mirror
            # of the OPD per-token quartet.
            tg_obs_sink = {}
            outputs["tg_losses"] = fsdp_losses.compute_tg_soft_ce(
                student_logits=student_logits,
                tg_label_log_probs=data["tg_label_logprobs"],
                tg_label_ids=data["tg_label_ids"],
                tg_token_mask=tg_token_mask,
                config=distillation_config,
                obs_sink=tg_obs_sink,
            )
            if tg_obs_sink:
                if outputs.get("obs") is None:
                    outputs["obs"] = {}
                outputs["obs"].update(tg_obs_sink)
    obs = outputs.pop("obs", None)
    if obs is not None:
        _collect_observation_rows(
            data, obs, outputs["distillation_losses"],
            loss_mode=loss_mode,
            log_prob_min_clamp=distillation_config.distillation_loss.log_prob_min_clamp,
            tg_token_losses=outputs.get("tg_losses"),
        )

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def _stamped_global_seq_info(data) -> dict:
    """Batch-global sequence-count info for seq-mean-* aggregation.

    Read from the engine's stamps (``forward_backward_batch`` all-reduces the
    non-empty response_mask row count next to batch_num_tokens), fail-loud on
    BOTH fields: without the global count the seq-mean denominator degrades to
    the LOCAL micro-batch row count (partitioner-dependent weighting, 0/0 on
    an all-masked micro-batch); without dp_size the loss lands at 1/dp of the
    true value under FSDP's gradient mean-allreduce (CD-005 review P1-1 —
    the engine stamps it unconditionally, absence means a broken engine)."""
    try:
        global_seqs = float(data["batch_num_seqs"])
    except KeyError as err:
        raise RuntimeError(
            "seq-mean aggregation needs the batch-global sequence count, but "
            "'batch_num_seqs' is missing — the engine did not stamp it "
            "(forward_backward_batch)"
        ) from err
    # trace-opd teacher grounding, frequency weighting (user ruling
    # 2026-08-02, codex tg-review P1-1): with the tg arm on, BOTH objective
    # terms divide by the TOTAL signal-row count N_lost + N_tg, so each term's
    # effective weight scales with its share of the batch —
    #   L = (N_lost/N)*mean_lost[OPD] + w_tg*(N_tg/N)*mean_won[tg]
    # — the won rate genuinely anneals OPD instead of leaving a lone lost row
    # carrying the full arm (the conditional-mean defect). tg off (no stamp):
    # byte-identical to the historical denominator.
    if "batch_num_tg_seqs" in data.keys():
        global_seqs += float(data["batch_num_tg_seqs"])
    try:
        dp_size = float(data["dp_size"])
    except KeyError as err:
        raise RuntimeError(
            "seq-mean aggregation needs 'dp_size' to compensate FSDP's "
            "gradient mean-allreduce, but the engine did not stamp it — "
            "without it the loss lands at 1/dp of the true value"
        ) from err
    return {"global_batch_size": max(global_seqs, 1.0), "dp_size": dp_size}


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # Called as logits processor
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    # Called as final policy loss
    distillation_loss_config = distillation_config.distillation_loss

    # trace-opd self-research, separate mode: the second update of a global
    # step carries research rows ONLY. Every executor-side term would divide by
    # an all-zero response_mask there (their masks were cleared at TQ put
    # time), so the aggregator short-circuits to the research term alone. The
    # flag rides batch.extra_info -> non-tensor key, set exclusively by the
    # trainer's second update_actor call.
    research_only = bool(data["research_only"]) if "research_only" in data.keys() else False
    if research_only:
        research_loss_coef = float(getattr(distillation_loss_config, "research_loss_coef", 0.0) or 0.0)
        if research_loss_coef <= 0.0:
            raise RuntimeError(
                "research_only update dispatched but research_loss_coef is 0 — the trainer "
                "split a research batch the loss is not configured to train"
            )
        research_loss, research_metrics = research_pg_loss(distillation_loss_config, model_output, data)
        return research_loss * research_loss_coef, research_metrics
    # trace-opd mixed objective: an explicitly weighted task-reward policy gradient
    # alongside the distillation term (config rejects combining it with
    # use_task_rewards, which is the same term at an implicit weight of 1).
    rl_loss_coef = float(getattr(distillation_loss_config, "rl_loss_coef", 0.0) or 0.0)

    if rl_loss_coef > 0.0:
        # Both trace-opd terms are evaluated under the SAME normalisation, and it
        # has to be the per-micro-batch token mean — the pure-OPD path's
        # behaviour, which is what the 4B mainline trains under today.
        #
        # Two things force this. First, ``ppo_loss`` fills
        # config.global_batch_info as a side effect and the config object
        # outlives the micro-batch loop, so the distillation term would
        # normalise per micro-batch on the first micro-batch and globally on
        # every one after it — a silent scale jump, and only in the arms that
        # enable RL. Second, and worse: the two normalisations differ by the
        # MICRO-BATCH COUNT. A per-micro-batch mean accumulates to N x mean over
        # the N micro-batches of a gradient-accumulation step; a global mean
        # accumulates to 1 x mean. At this run's shapes N is around 60, so a
        # nominal mix_rl_weight of 1.0 would in truth weight OPD against RL
        # about 60:1 — and the ablation would report "RL does not help" as an
        # artefact of arithmetic, not as a finding.
        #
        # So the RL term does not go through ``ppo_loss``: that function refills
        # the global counts from `data` regardless of what we clear here, and it
        # also folds the entropy and reference-KL regularisers into the same
        # scalar, neither of which should be scaled by an RL weight.
        saved_batch_info = dict(config.global_batch_info)
        config.global_batch_info.clear()
        if config.loss_agg_mode != "token-mean":
            # CD-005 review P0-1: under seq-mean-* BOTH terms must divide by
            # the SAME batch-global sequence count, or the nominal
            # mix_rl_weight ratio silently becomes ~N:1 (task_reward_pg
            # aggregates through policy_loss_fn's `**config.global_batch_info`
            # and, with the cleared window empty, agg_loss falls back to the
            # LOCAL micro-batch row count — the exact defect the distillation
            # branch routes around). Stage the engine-stamped info INSIDE the
            # cleared window: distillation_loss consumes it via its
            # `batch_info = dict(config.global_batch_info)` snapshot, the RL
            # term through policy_loss_fn — one denominator, so 1:1 means 1:1.
            # Under token-mean the window stays empty and both terms keep the
            # historical local per-micro-batch mean, byte-for-byte.
            config.global_batch_info.update(_stamped_global_seq_info(data))
        try:
            distill_loss, distill_metrics = distillation_loss(
                config, distillation_config, model_output, data
            )
            policy_loss, policy_metrics = task_reward_pg(config, model_output, data)
            policy_loss = policy_loss * rl_loss_coef
        finally:
            config.global_batch_info.clear()
            config.global_batch_info.update(saved_batch_info)
    else:
        distill_loss, distill_metrics = distillation_loss(
            config, distillation_config, model_output, data
        )
        if distillation_loss_config.use_task_rewards or distillation_loss_config.use_policy_gradient:
            policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)
            if not distillation_loss_config.use_task_rewards:
                policy_loss = 0.0
        else:
            # no need to compute policy loss
            policy_loss = 0.0
            policy_metrics = {}

    # Combine distillation with policy loss
    policy_metrics.update(distill_metrics)
    # Upstream's distillation_loss_coef is forced to 1 unless use_task_rewards is
    # on, because under pure distillation an overall scale is meaningless. In the
    # mixed objective it is not meaningless, so trace-opd carries its own
    # explicit weight on top — 1.0 by default, which leaves both behaviours
    # unchanged. All three arms then read the same way: a weight per term.
    distillation_loss_coef = (
        distillation_loss_config.distillation_loss_coef if distillation_loss_config.use_task_rewards else 1.0
    ) * float(getattr(distillation_loss_config, "opd_loss_coef", 1.0))
    policy_loss += distill_loss * distillation_loss_coef
    policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)

    # trace-opd mixed objective: cross-entropy over the expert tokens of the
    # privileged-guided rollouts. Those rows carry an all-zero response_mask
    # (cleared at TQ put time), so they contribute nothing above and nothing to
    # any denominator derived from it — the OPD/RL terms keep the exact scale
    # they have with the SFT arm off, and this term is a pure addition.
    sft_loss_coef = float(getattr(distillation_loss_config, "sft_loss_coef", 0.0) or 0.0)
    if sft_loss_coef > 0.0:
        if config.loss_agg_mode != "token-mean":
            # CD-005 review P0-2 guard: the SFT term is a LOCAL per-micro-batch
            # expert-token mean while the distillation term is a GLOBAL
            # per-sequence mean under seq-mean-* — sft_loss_coef would weigh
            # two differently-shaped means and its effective strength would
            # drift with how expert rows land in micro-batches. Blocked loud
            # until the SFT term's unit under seq-mean is ruled on; run mixed
            # SFT arms with loss_agg_mode: token-mean.
            raise RuntimeError(
                f"sft_loss_coef > 0 with loss_agg_mode={config.loss_agg_mode!r}: "
                "the SFT term's local token-mean is incommensurable with the "
                "global seq-mean distillation term — use token-mean for SFT "
                "arms until the seq-mean SFT unit is defined"
            )
        sft_loss, sft_metrics = sft_ce_loss(model_output, data)
        policy_loss = policy_loss + sft_loss * sft_loss_coef
        policy_metrics.update(sft_metrics)

    # trace-opd teacher grounding (docs/teacher-grounding-design.md): reverse
    # distillation over the tg rows — bare SUCCESS trajectories re-prompted
    # with the doc prefix, labelled by the bare policy's renormalised top-k.
    # Decoupled from the OPD family end to end (user ruling 2026-08-02): the
    # labels ride their own tg_label channel and the logits processor computed
    # their per-token soft-CE with the standalone compute_tg_soft_ce, so this
    # term reads ``tg_losses`` under ``tg_mask``. tg rows carry an all-zero
    # response_mask, so every term above is blind to them and this term is a
    # pure addition, exactly like the SFT arm.
    tg_loss_coef = float(getattr(distillation_loss_config, "tg_loss_coef", 0.0) or 0.0)
    if tg_loss_coef > 0.0:
        tg_mask = data.get("tg_mask", None)
        if tg_mask is None:
            # Fail loud: a silently dropped tg arm reads as "teacher grounding
            # does not help", the one conclusion this experiment must not fake.
            raise RuntimeError(
                "tg_loss_coef > 0 but the batch carries no 'tg_mask' field — the rollout "
                "side did not mark tg samples (expected sample_role == 'tg' rows)"
            )
        if tg_mask.is_nested:
            tg_mask = tg_mask.to_padded_tensor(0)
        tg_mask = tg_mask.to(torch.bool)
        # Decoupling ruling 2026-08-02: the tg term reads its OWN per-token
        # matrix (compute_tg_soft_ce over the tg_label channel), never the OPD
        # formula's. Absence is a broken invariant, not a fallback.
        if "tg_losses" not in model_output:
            raise RuntimeError(
                "tg_loss_coef > 0 but the logits processor shipped no 'tg_losses' — "
                "the decoupled tg formula did not run (compute_topk_loss dispatch)"
            )
        tg_losses = no_padding_2_padding(model_output["tg_losses"], data)
        if tg_losses.shape != tg_mask.shape:
            raise RuntimeError(
                f"tg_mask shape {tuple(tg_mask.shape)} does not match the per-token loss "
                f"layout {tuple(tg_losses.shape)} — a tg row is out of step with its responses"
            )
        # The standalone soft-CE is non-negative by construction (renormalised
        # label, no OPD clamps anywhere near it — codex tg-review P0-2 made
        # structural by the decoupling), so clamp_min(0) is genuinely lossless
        # fp hygiene. loss_max_clamp is deliberately NOT applied: a large tg
        # loss is legitimate supervision (q parking its mass outside the label
        # support) and clamping would zero its gradient exactly there.
        # Telemetry keeps the invariant observable - a materially negative
        # pre-clamp minimum means the formula broke.
        with torch.no_grad():
            tg_valid_losses = tg_losses[tg_mask]
            if tg_valid_losses.numel():
                policy_metrics["tg/loss_min_preclamp"] = Metric(
                    AggregationType.MIN, tg_valid_losses.min())
                policy_metrics["tg/loss_max"] = Metric(
                    AggregationType.MAX, tg_valid_losses.max())
        tg_losses = tg_losses.clamp_min(0.0)
        if config.loss_agg_mode == "token-mean":
            # Local per-micro-batch token mean with the SFT arm's zero-safe
            # denominator: a micro-batch may hold no tg token at all.
            tg_batch_info = {"batch_num_tokens": tg_mask.sum().clamp(min=1)}
        else:
            # seq-mean-*: frequency weighting (user ruling 2026-08-02) — the
            # tg term divides by the same TOTAL signal-row count the OPD term
            # uses (_stamped_global_seq_info adds batch_num_tg_seqs when
            # present), so the two shares sum to the batch. The tg stamp's own
            # presence stays fail-loud: a tg batch the engine did not count
            # must not silently fall back to the lost-only denominator.
            if "batch_num_tg_seqs" not in data.keys():
                raise RuntimeError(
                    "tg seq-mean aggregation needs the batch-global tg sequence count, "
                    "but 'batch_num_tg_seqs' is missing — the engine did not stamp it "
                    "(forward_backward_batch)"
                )
            tg_batch_info = dict(_stamped_global_seq_info(data))
        tg_loss = agg_loss(
            loss_mat=tg_losses,
            loss_mask=tg_mask,
            loss_agg_mode=config.loss_agg_mode,
            **tg_batch_info,
        )
        policy_loss = policy_loss + tg_loss * tg_loss_coef
        policy_metrics["tg/loss"] = Metric(value=tg_loss, aggregation=AggregationType.SUM)
        policy_metrics["tg/tokens"] = Metric(
            value=tg_mask.sum(), aggregation=AggregationType.SUM
        )
        policy_metrics["tg/rows"] = Metric(
            value=(tg_mask.sum(dim=-1) > 0).sum(), aggregation=AggregationType.SUM
        )

    # trace-opd self-research (docs/self-research-design.md 7.1, joint mode):
    # the research role's own PG term, over research_mask tokens only, with its
    # OWN batch-global denominator — a pure addition, exactly like the SFT arm.
    # Research rows ship an all-zero response_mask, so nothing above ever sees
    # them; and this term reads research_mask, so it never sees anything else.
    # In warmup the trainer drops research rows from the update batch entirely,
    # which leaves this term summing over an empty mask: 0, by construction.
    research_loss_coef = float(getattr(distillation_loss_config, "research_loss_coef", 0.0) or 0.0)
    if research_loss_coef > 0.0:
        research_loss, research_metrics = research_pg_loss(distillation_loss_config, model_output, data)
        policy_loss = policy_loss + research_loss * research_loss_coef
        policy_metrics.update(research_metrics)

    return policy_loss, policy_metrics


def task_reward_pg(
    config: ActorConfig, model_output: dict, data: TensorDict
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Task-reward policy gradient for the mixed objective's RL arm.

    The same clipped objective ``ppo_loss`` computes, over the same GRPO
    advantages, but aggregated the way the distillation term is — under whatever
    ``config.global_batch_info`` the caller has set up, rather than refilling it
    from the batch. That is the whole reason this exists rather than calling
    ``ppo_loss``: see the comment at the call site for why a mismatch there costs
    a factor of the micro-batch count. It also leaves the entropy and
    reference-KL regularisers out, since neither belongs under an RL weight.

    Expert rows contribute nothing: their response_mask is all zero, so they
    leave both the numerator and the denominator.
    """
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    padded = data.select(*fields).to_padded_tensor()
    response_mask = padded["response_mask"].to(bool)

    policy_loss_fn = get_policy_loss_fn(config.policy_loss.get("loss_mode", "vanilla"))
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=padded["old_log_probs"],
        log_prob=log_prob,
        advantages=padded["advantages"],
        response_mask=response_mask,
        loss_agg_mode=config.loss_agg_mode,
        config=config,
        rollout_is_weights=padded.get("rollout_is_weights", None),
    )
    metrics: dict[str, Any] = {f"rl/{k[len('actor/'):]}": v for k, v in pg_metrics.items()}
    metrics["rl/pg_loss"] = Metric(value=pg_loss, aggregation=AggregationType.SUM)
    # How much of the batch actually carries an RL gradient. With a binary
    # reward and G rollouts per task, a group whose members all succeed or all
    # fail has zero advantage throughout, and at a few percent success rate that
    # is most groups — so the arm's effective strength grows on its own as the
    # policy improves. Worth watching rather than inferring.
    nonzero = ((padded["advantages"] != 0) & response_mask).sum()
    total = response_mask.sum()
    metrics["rl/nonzero_advantage_frac"] = (
        (nonzero.float() / total).item() if total > 0 else 0.0
    )
    return pg_loss, metrics


def sft_ce_loss(model_output: dict, data: TensorDict) -> tuple[torch.Tensor, dict[str, Any]]:
    """Token-level cross-entropy over expert tokens (trace-opd mixed objective).

    ``sft_mask`` marks the assistant-generated tokens of a privileged-guided
    trajectory replayed under the BARE prompt — the supervision the policy should
    be able to produce without the document. Every other row, and every
    observation token inside an expert row, is masked out.

    Normalisation is a per-micro-batch token mean, deliberately the same shape
    the distillation term uses (``agg_loss`` token-mean with an empty
    global_batch_info, which is what the pure-OPD path has always run), so
    ``sft_loss_coef`` weights one mean against another and means the same thing
    in every ablation arm. NOTE (2026-07-29): that symmetry holds only while
    the distillation term runs token-mean. Under ``loss_agg_mode=seq-mean-*``
    the distillation term becomes a GLOBAL per-sequence mean while this term
    stays a LOCAL per-micro-batch token mean — ``sft_loss_coef`` then weighs
    two differently-shaped means, and its calibrated value does not carry over
    across aggregation modes. It is computed here rather than through ``agg_loss``
    for one reason: a micro-batch may hold no expert token at all, and
    ``agg_loss`` would divide by that zero. Such a micro-batch contributes
    exactly 0 instead.
    """
    sft_mask = data.get("sft_mask", None)
    if sft_mask is None:
        # Fail loud: a silently dropped SFT arm looks exactly like "SFT does not
        # help", which is the one conclusion this experiment must not fake.
        raise RuntimeError(
            "sft_loss_coef > 0 but the batch carries no 'sft_mask' field — the rollout "
            "side did not mark expert samples (expected extra_fields['sample_role'] == 'sft')"
        )
    if sft_mask.is_nested:
        sft_mask = sft_mask.to_padded_tensor(0)
    sft_mask = sft_mask.to(torch.bool)

    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    ce = -log_prob
    if sft_mask.shape != ce.shape:
        # ``no_padding_2_padding`` sizes off data["responses"], so a mismatch
        # means some row's sft_mask no longer matches its own response length —
        # e.g. a synthesized row that copied the field from a different sample.
        # Broadcasting would either raise deep inside the multiply or, worse,
        # quietly inflate the expert-token count that the SFT arm is judged by.
        raise RuntimeError(
            f"sft_mask shape {tuple(sft_mask.shape)} does not match the response layout "
            f"{tuple(ce.shape)} — a row's sft_mask is out of step with its responses"
        )
    n_expert_tokens = sft_mask.sum()
    sft_loss = (ce * sft_mask).sum() / n_expert_tokens.clamp(min=1)

    metrics = {
        "sft/loss": Metric(value=sft_loss, aggregation=AggregationType.SUM),
        "sft/expert_tokens": Metric(value=n_expert_tokens, aggregation=AggregationType.SUM),
    }
    return sft_loss, metrics


def research_pg_loss(
    distillation_loss_config, model_output: dict, data: TensorDict
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Clipped policy gradient over researcher-session tokens (self-research).

    ``research_mask`` marks the researcher-generated tokens of a research row
    (tool returns and every non-research row are zero). ``research_advantage``
    carries the group-relative document-utility advantage the trainer computed
    (mean-centred over the task's N documents, NOT divided by the group std,
    clamped to +/-research_adv_clip). The ratio term is the standard clipped
    importance form: ~1 under joint mode (gradients at the sampling weights;
    it still absorbs the rollout-vs-trainer logprob numerics), truly at work
    under separate mode (one update of staleness).

    Normalisation is the batch-GLOBAL research-token count, all-reduced before
    micro-batching (``research_num_tokens``): each micro-batch contributes
    numerator/T_global and the accumulation over micro-batches reconstructs the
    exact global mean — the term's value cannot depend on sequence packing.
    """
    research_mask = data.get("research_mask", None)
    if research_mask is None:
        # Fail loud, same contract as the SFT arm: a silently dropped research
        # channel reads as "research RL does not help", the one conclusion this
        # experiment must not fake.
        raise RuntimeError(
            "research_loss_coef > 0 but the batch carries no 'research_mask' field — the "
            "rollout side did not mark research samples (expected "
            "extra_fields['research_arm'] stamps and sample_role == 'research')"
        )
    research_adv = data.get("research_advantage", None)
    if research_adv is None:
        raise RuntimeError(
            "research rows arrived without 'research_advantage' — the trainer's advantage "
            "stage did not run its research branch (trainer_base._compute_advantage)"
        )
    if research_mask.is_nested:
        research_mask = research_mask.to_padded_tensor(0)
    research_mask = research_mask.to(torch.bool)
    if research_adv.is_nested:
        research_adv = research_adv.to_padded_tensor(0.0)

    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    old_log_prob = data["old_log_probs"]
    if old_log_prob.is_nested:
        old_log_prob = old_log_prob.to_padded_tensor(0.0)
    if research_mask.shape != log_prob.shape or research_adv.shape != log_prob.shape:
        raise RuntimeError(
            f"research fields out of step with the response layout: mask "
            f"{tuple(research_mask.shape)}, advantage {tuple(research_adv.shape)}, "
            f"log_prob {tuple(log_prob.shape)}"
        )

    try:
        t_res_global = float(data["research_num_tokens"])
    except KeyError as err:
        raise RuntimeError(
            "research rows present but 'research_num_tokens' missing — the engine did not "
            "compute the batch-global research-token count (forward_backward_batch)"
        ) from err
    # codex review P0-3: FSDP averages gradients across dp ranks, so a loss of
    # local_numerator / T_global lands at 1/dp_size of the true value — the
    # exact compensation upstream agg_loss applies (masked_sum / batch_num_tokens
    # * dp_size). Without it the effective research weight shrinks with the GPU
    # count and the packing/parallelism invariance claim is false.
    try:
        dp_size = float(data["dp_size"])
    except KeyError:
        dp_size = 1.0

    clip_low = float(getattr(distillation_loss_config, "clip_ratio_low", 0.2) or 0.2)
    clip_high = float(getattr(distillation_loss_config, "clip_ratio_high", 0.2) or 0.2)
    # codex review P2-12: clamp the log-ratio before exp (PPO main path does the
    # same) — an extreme stale ratio would overflow, and inf*0 on a masked or
    # zero-advantage token still poisons the sum with NaN.
    log_ratio = (log_prob - old_log_prob).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(min=1.0 - clip_low, max=1.0 + clip_high)
    per_token = -torch.minimum(ratio * research_adv, clipped_ratio * research_adv)
    denominator = max(t_res_global, 1.0)
    research_loss = (per_token * research_mask).sum() / denominator * dp_size

    n_tokens = research_mask.sum()
    with torch.no_grad():
        clipped_frac = (
            (((ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)) & research_mask).sum().float()
            / n_tokens.clamp(min=1)
        )
        adv_rows = research_adv[research_mask]
        adv_mean = adv_rows.mean() if adv_rows.numel() else torch.tensor(0.0)
        adv_absmax = adv_rows.abs().max() if adv_rows.numel() else torch.tensor(0.0)
    metrics = {
        # SUM aggregation: micro-batches each hold numerator/T_global, so the
        # sum over micro-batches is the true global value of the term.
        "research/pg_loss": Metric(value=research_loss, aggregation=AggregationType.SUM),
        "research/tokens": Metric(value=n_tokens, aggregation=AggregationType.SUM),
        "research/ratio_clipped_frac": clipped_frac.item(),
        "research/adv_mean": adv_mean.item(),
        "research/adv_absmax": adv_absmax.item(),
    }
    return research_loss, metrics


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    loss_agg_mode = config.loss_agg_mode

    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )
    if loss_config.loss_max_clamp is not None:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    if loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        saved_pg_info = dict(loss_config.global_batch_info)
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        if loss_agg_mode != "token-mean" and not loss_config.global_batch_info:
            # CD-005 review P0-3: with no upstream info this path would fall
            # back to the LOCAL micro-batch row count under seq-mean-* (and
            # 0/0 -> NaN on an all-masked micro-batch) — the same defect the
            # direct branch below routes around, same cure. Restored in the
            # finally: the config object outlives the micro-batch loop and
            # batch_num_seqs changes across mini-batches.
            loss_config.global_batch_info.update(_stamped_global_seq_info(data))
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = data["old_log_probs"].to_padded_tensor(0.0)
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        rollout_is_weights = data.get("rollout_is_weights", None)
        try:
            distillation_loss, pg_metrics = policy_loss_fn(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=-distillation_losses.detach(),
                response_mask=response_mask,
                loss_agg_mode=loss_agg_mode,
                config=loss_config,
                rollout_is_weights=rollout_is_weights,
            )
        finally:
            loss_config.global_batch_info.clear()
            loss_config.global_batch_info.update(saved_pg_info)
        pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        batch_info = dict(config.global_batch_info)
        if not batch_info:
            # trace-opd mixed objective: a micro-batch can legitimately hold no
            # on-policy token at all — expert rows carry an all-zero
            # response_mask, and the balanced partitioner groups by workload
            # without regard to role, so an all-expert micro-batch is reachable.
            # With no global count supplied, agg_loss's token-mean divides by
            # loss_mask.sum(), i.e. by that zero, and hands a NaN to backward.
            # Passing the clamped count keeps the denominator identical whenever
            # the mask is non-empty and makes such a micro-batch contribute
            # exactly 0, which is what it has to say.
            batch_info = {"batch_num_tokens": response_mask.sum().clamp(min=1)}
            if loss_agg_mode != "token-mean":
                # CD-005 review of the seq-mean switch (2026-07-29). Equal
                # weight per sequence means equal per GLOBAL sequence: without
                # a global count, agg_loss's seq-mean branches divide by the
                # LOCAL micro-batch row count, and the balanced partitioner
                # packs long rows into few-row micro-batches — a 28k-token row
                # riding alone would weigh ~10x a row sharing a 30-row
                # micro-batch, handing long rows back almost exactly the
                # length-proportional weight this mode is chosen to remove.
                # And an all-masked micro-batch (expert rows) would make that
                # local count zero -> 0/0 -> NaN into backward. The engine
                # stamps the all-reduced non-empty-row count next to
                # batch_num_tokens (forward_backward_batch); read it fail-loud
                # via _stamped_global_seq_info (both fields — dp_size too,
                # CD-005 review P1-1), same contract as research_num_tokens.
                # dp_size compensates FSDP's mean-allreduce of gradients,
                # exactly as agg_loss expects. token-mean keeps the local
                # denominator above, unchanged — every historical run trained
                # on that contract.
                batch_info.update(_stamped_global_seq_info(data))
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **batch_info,
        )

    return distillation_loss, distillation_metrics


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# trace-opd distillation microscope: sample per-token distillation quantities
# to TRACE_DISTILL_MICROSCOPE_DIR for offline analysis (joined with rollout
# dumps via a content hash over response token ids). The hash below MUST stay
# bit-identical with trace_opd.common.hashing.response_token_hash.
# ---------------------------------------------------------------------------
_MICROSCOPE_MOD = (1 << 63) - 1
_MICROSCOPE_MUL = 1000003


def _resp_hash(ids) -> int:
    h = 0
    for x in ids:
        h = (h * _MICROSCOPE_MUL + int(x)) & _MICROSCOPE_MOD
    return h


def _response_id_rows(data: TensorDict):
    responses = data.get("responses", None)
    if responses is None:
        return None
    if responses.is_nested:
        return [t.tolist() for t in responses.unbind()]
    prompts = data.get("prompts", None)
    attention_mask = data.get("attention_mask", None)
    if prompts is None or attention_mask is None:
        return None
    prompt_len = prompts.shape[1]
    lens = attention_mask[:, prompt_len:].sum(dim=1).tolist()
    return [responses[i, : int(n)].tolist() for i, n in enumerate(lens)]


def _collect_observation_rows(
    data: TensorDict, obs: dict, token_losses: torch.Tensor,
    *, loss_mode: str, log_prob_min_clamp, tg_token_losses: torch.Tensor = None,
) -> None:
    """trace-opd observation packs: cut each row's RESPONSE window out of this
    rank's packed-sequence segment and hand per-row numpy records to the
    collector (docs/microscope-design.md).

    Layout facts this function owns:
      * the packed THD stream concatenates prompt+response per row; the row's
        response PREDICTIONS live at [start + P - 1, start + P - 1 + R) — the
        same window no_padding_2_padding reads;
      * under sequence parallel the stream is padded to a multiple of sp_size
        and split evenly; this rank holds [sp_rank*parts, (sp_rank+1)*parts).
        Rows are cut to the intersection; token_loss here is the PRE-outer-
        clamp per-token KL (the audit's ground-truth anchor, noted as such).
    Best-effort by contract: any failure disables collection, never training.
    """
    try:
        from verl.trainer.distillation.observation import collect_micro_batch, observation_enabled
        from verl.utils.ulysses import (
            get_ulysses_sequence_parallel_rank,
            get_ulysses_sequence_parallel_world_size,
        )

        if not observation_enabled():
            return
        # codex review P0-1: global_steps is stamped per ROW at TQ put time, so
        # the micro-batch carries a NonTensorStack (list semantics) — int() on
        # it raises and the old code disabled collection forever on the very
        # first batch. Unwrap every observed shape; rows of one micro-batch
        # share the step, so the first element is the step.
        step_value = data.get("global_steps", None)
        step_value = getattr(step_value, "data", step_value)
        if hasattr(step_value, "tolist"):
            step_value = step_value.tolist()
        if isinstance(step_value, (list, tuple)):
            step_value = step_value[0] if step_value else 0
        step_value = getattr(step_value, "data", step_value)
        global_step = int(step_value or 0)
        # codex review P1-11: gate on the sampling schedule BEFORE any CPU
        # copy/assembly. (The student-own top-k GPU pass in the payload cannot
        # see the step and stays constant — ms-scale per micro-batch,
        # documented as such.)
        from verl.trainer.distillation.observation import _step_wanted

        if not _step_wanted(global_step):
            return

        prompts = data["prompts"]
        responses = data["responses"]
        prompt_lens = prompts.offsets().diff().tolist()
        resp_rows = [r.tolist() for r in responses.unbind()]
        old_lp = data["old_log_probs"]
        old_lp_rows = [r.float().cpu().numpy() for r in old_lp.unbind()]
        mask_rows = [m.cpu().numpy() for m in data["response_mask"].unbind()]
        rewards = data["rm_scores"].to_padded_tensor(0.0).sum(-1).cpu().tolist()
        extras = data.get("extra_fields", None)
        extras = list(extras) if extras is not None else [{}] * len(resp_rows)

        sp_size = get_ulysses_sequence_parallel_world_size()
        sp_rank = get_ulysses_sequence_parallel_rank() if sp_size > 1 else 0
        full_len = int(sum(prompt_lens) + sum(len(r) for r in resp_rows))
        padded = full_len
        if sp_size > 1 and full_len % sp_size:
            padded = full_len + (sp_size - full_len % sp_size)
        parts = padded // sp_size
        rank_lo, rank_hi = sp_rank * parts, (sp_rank + 1) * parts

        def seg_np(t: torch.Tensor, lo: int, hi: int):
            return t[0, lo:hi].detach().float().cpu().numpy() if t.dtype.is_floating_point else t[0, lo:hi].detach().cpu().numpy()

        # codex impl-review P1-3: resp_hash alone is NOT a row key — identical
        # short responses (8 rollouts of one task, or across tasks) collide
        # and the offline reader would merge distinct rows. A per-call batch
        # fingerprint + the row's batch index is stable across the SP group
        # (same micro-batch data, same enumeration) and unique across
        # micro-batches, so (mb_fp, i) identifies the row.
        mb_fp = _resp_hash([_resp_hash(r) for r in resp_rows])
        rows = []
        start = 0
        for i, resp in enumerate(resp_rows):
            P, R = int(prompt_lens[i]), len(resp)
            win_lo, win_hi = start + P - 1, start + P - 1 + R
            start += P + R
            lo, hi = max(win_lo, rank_lo), min(win_hi, rank_hi)
            if lo >= hi:
                continue
            extra = extras[i] if isinstance(extras[i], dict) else {}
            # trace-opd teacher grounding (decoupled channels): a REAL tg row
            # observes through its OWN pack — teacher_* fields carry the bare
            # label p̄, student_support_lps carries q on the label support, and
            # token_loss carries the soft-CE matrix (the tg mirror of the OPD
            # quartet; the row's student_* top-k IS q's own top-k since the tg
            # training forward runs under the doc prefix). Dropped-won
            # placeholders (sample_role "tg", teacher_mode "tg_skip") have no
            # label and nothing to observe.
            is_real_tg = extra.get("teacher_mode") == "tg"
            if extra.get("sample_role") == "tg" and not is_real_tg:
                continue
            if is_real_tg and (tg_token_losses is None or "tg_label_ids" not in obs):
                continue  # tg row but no tg pack shipped — skip, never mix
            if is_real_tg:
                t_ids_src, t_lps_src = obs["tg_label_ids"], obs["tg_label_lps"]
                sup_src, loss_src = obs["tg_q_lps"], tg_token_losses
            else:
                t_ids_src, t_lps_src = obs["teacher_ids"], obs["teacher_lps"]
                sup_src, loss_src = obs["student_support_lps"], token_losses
            seg_slice = (lo - rank_lo, hi - rank_lo)
            rows.append(
                {
                    "resp_hash": _resp_hash(resp),
                    "row_uid": f"{mb_fp:x}:{i}",
                    "task_id": extra.get("task_id", ""),
                    "sample_role": extra.get("sample_role", "bare"),
                    "won": extra.get("won", 0),
                    "reward": rewards[i] if i < len(rewards) else 0.0,
                    "turn_spans": extra.get("turn_spans") or [],
                    "response_ids": np.asarray(resp, dtype=np.int32),
                    "response_mask": mask_rows[i],
                    "realized_student_lp": old_lp_rows[i],
                    "seg_start": lo - win_lo,
                    "seg_len": hi - lo,
                    "teacher_ids": seg_np(t_ids_src, *seg_slice),
                    "teacher_lps": seg_np(t_lps_src, *seg_slice),
                    "student_ids": seg_np(obs["student_ids"], *seg_slice),
                    "student_lps": seg_np(obs["student_lps"], *seg_slice),
                    "student_support_lps": seg_np(sup_src, *seg_slice),
                    "token_loss": seg_np(loss_src.detach(), *seg_slice),
                }
            )
        if rows:
            for row in rows:
                row["loss_mode"] = loss_mode
                row["log_prob_min_clamp"] = float(log_prob_min_clamp if log_prob_min_clamp is not None else -1e9)
            collect_micro_batch(
                global_step=global_step, rows=rows, sp_rank=sp_rank, sp_size=sp_size
            )
    except Exception as err:  # noqa: BLE001
        from verl.trainer.distillation.observation import _disable

        _disable(f"orchestration failed: {type(err).__name__}: {err}")


def _dump_microscope_rows(data: TensorDict, kl, mask, extras: dict) -> None:
    out_dir = os.getenv("TRACE_DISTILL_MICROSCOPE_DIR")
    if not out_dir:
        return
    try:
        every = max(1, int(os.getenv("TRACE_DISTILL_SAMPLE_EVERY", "4")))
        rows = _response_id_rows(data)
        if rows is None:
            return
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        lines = []
        for i, ids in enumerate(rows):
            resp_hash = _resp_hash(ids)
            if resp_hash % every:
                continue
            length = len(ids)
            if not bool(mask[i, :length].any()):
                # No distillation happened on this row — an expert (SFT) sample
                # of the mixed objective, or a prompt the hang guard cancelled.
                # Dumping it would put an all-zero mask in front of the offline
                # join, whose integrity check compares the mask against the
                # trajectory's turn spans and would report a false mismatch.
                continue

            def row_of(t):
                return [round(float(v), 5) for v in t[i, :length].tolist()]

            record = {
                "resp_hash": resp_hash,
                "resp_len": length,
                "ids": [int(x) for x in ids],
                "mask": [int(v) for v in mask[i, :length].tolist()],
                "kl": row_of(kl),
            }
            for name, tensor in extras.items():
                record[name] = row_of(tensor) if tensor is not None else None
            lines.append(json.dumps(record))
        if lines:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"rank{rank}.jsonl"), "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
    except Exception:  # noqa: BLE001 — diagnostics must never kill training
        logger.warning("distillation microscope dump failed", exc_info=True)


@register_distillation_loss(
    DistillationLossSettings(names=["forward_kl_topk", "reverse_kl_topk"], use_topk=True)
)  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute top-k distillation loss and related metrics using top-k log probabilities.

    Registered for BOTH top-k modes: the direction-specific math runs in the
    logits processor (``compute_topk_loss`` dispatches on ``loss_mode`` to
    ``fsdp.losses.compute_forward_kl_topk`` / ``compute_reverse_kl_topk``);
    this outer function only post-processes what that stage computed, and every
    step of it — mass/overlap/entropy diagnostics, clamps, microscope dump —
    is direction-agnostic.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    overlap_count = model_output.get("overlap_count")
    overlap_token_advantage = model_output.get("overlap_token_advantage")
    if overlap_count is not None and overlap_token_advantage is not None:
        overlap_count = no_padding_2_padding(overlap_count, data)
        overlap_token_advantage = no_padding_2_padding(overlap_token_advantage, data)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    overlap_metrics = {}
    if overlap_count is not None and overlap_token_advantage is not None:
        assert overlap_count.shape == overlap_token_advantage.shape == response_mask_bool.shape
        valid_overlap_count = overlap_count[response_mask_bool]
        k = distillation_config.distillation_loss.topk
        assert k is not None
        # codex impl-review P1-1: an all-tg micro-batch has an empty
        # response_mask — mean() over the empty selection is NaN and would
        # poison the OPD diagnostics stream. No OPD tokens, no overlap
        # metrics for this micro-batch (same absent-not-zero contract as the
        # mass diagnostics below).
        if valid_overlap_count.numel():
            # Diagnostics for tracking teacher/student top-k overlap in OPD, following
            # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016):
            # overlap ratio and average teacher-token KL contribution on overlapped tokens.
            overlap_metrics["distillation/overlap_ratio"] = (valid_overlap_count.float().mean() / k).item()
            overlap_position_mask = response_mask_bool & (overlap_count > 0)
            if overlap_position_mask.any():
                overlap_metrics["distillation/overlap_token_advantage"] = (
                    overlap_token_advantage[overlap_position_mask].mean().item()
                )
            else:
                overlap_metrics["distillation/overlap_token_advantage"] = 0.0

    # Padded per-token views for the microscope dump, saved before the metric
    # code below flattens them through the response mask.
    teacher_topk_entropy = model_output.get("teacher_topk_entropy")
    if teacher_topk_entropy is not None:
        teacher_topk_entropy = no_padding_2_padding(teacher_topk_entropy, data)
    microscope_extras = {
        "s_mass": student_mass,
        "t_mass": teacher_mass,
        "overlap": overlap_count,
        "t_ent": teacher_topk_entropy,
    }

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    # A micro-batch can carry no valid response token at all -- the agent-loop
    # wall-clock guard cancels a stuck prompt and its replicas arrive with an
    # empty response. min()/max() raise on an empty tensor and take the whole
    # run down with them, which is how a single hung prompt killed step-17.
    # These are diagnostics only, so report them as absent and let training
    # proceed on the remaining samples.
    if student_mass.numel() == 0:
        distillation_metrics = dict(overlap_metrics)
    else:
        distillation_metrics = {
            "distillation/student_mass": student_mass.mean().item(),
            "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
            "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
            "distillation/teacher_mass": teacher_mass.mean().item(),
            "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
            "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
            **overlap_metrics,
        }

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)

    # Same clamped values the loss trains on — keep the dump loss-consistent.
    _dump_microscope_rows(data, distillation_losses, response_mask_bool, microscope_extras)

    return distillation_losses, distillation_metrics


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
    )
    # Since k1 can be negative, log the mean absolute loss. Guarded like every
    # other diagnostic: an all-cancelled/all-expert micro-batch selects nothing.
    valid_losses = distillation_losses[response_mask_bool]
    metrics = {}
    if valid_losses.numel() > 0:
        metrics["distillation/abs_loss"] = Metric(AggregationType.MEAN, valid_losses.abs().mean())
    # Estimator modes carry no top-k tensors, so the microscope row is just the
    # per-token estimate itself; the offline join (resp_hash, mask integrity)
    # works the same as under forward_kl_topk.
    _dump_microscope_rows(data, distillation_losses, response_mask_bool, {})
    return distillation_losses, metrics


def _gated_logp_shift(
    student_log_probs: torch.Tensor, teacher_log_probs: torch.Tensor, gate_beta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The SDAR/SEED math, dependency-free so tests can execute it in isolation.

    Returns (per-token losses, gate, teacher_gap); gradient flows only through
    the student log-probs — gap and gate are detached, teacher is data.
    """
    teacher_gap = (teacher_log_probs - student_log_probs).detach()
    gate = torch.sigmoid(float(gate_beta) * teacher_gap)
    losses = gate * (teacher_log_probs.detach() - student_log_probs)
    return losses, gate, teacher_gap


@register_distillation_loss(DistillationLossSettings(names=["reverse_kl_gated"], use_estimator=True))  # type: ignore[arg-type]
def compute_distillation_loss_gated_logp_shift(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Confidence-gated logp-shift on sampled tokens — the SDAR/SEED objective.

    The two 2026 agentic self-distillation lines use the mathematically
    identical term (SDAR ``sdar_utils.compute_sdar_loss``, SEED
    ``core_algos.compute_opd_loss``, both gate_beta=5.0 by default):

        delta_t = log pi_T(y_t) - log pi_S(y_t)          (sampled token, detached)
        gate_t  = sigmoid(beta * delta_t)                 (detached)
        loss_t  = gate_t * (log pi_T(y_t) - log pi_S(y_t))

    Gradient flows only through -log pi_S: every sampled token is pulled UP
    with strength gate_t — full pull where the teacher endorses it beyond the
    student's own confidence, exponentially attenuated where the teacher
    rejects it. It is not a KL estimator (the papers call it a reverse-KL
    surrogate); unlike k1, the teacher does shape the gradient — through the
    detached gate.

    Differences to the source repos, on purpose: their sample/step-level
    ``opd_step_mask`` (SEED hindsight skills cover only some steps) has no
    counterpart here because the privileged document conditions the whole
    trajectory — response_mask IS the step mask; and their loss coefficient
    (SDAR ``sdar_loss_coef``, code default 0.1, paper 0.01; SEED
    ``opd_loss_coef``) maps onto the caller's opd_loss_coef, not a second knob.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    gate_beta = float(getattr(loss_config, "gate_beta", 5.0))
    distillation_losses, gate, teacher_gap = _gated_logp_shift(
        student_log_probs, teacher_log_probs, gate_beta
    )

    metrics = {}
    valid_gate = gate[response_mask_bool]
    if valid_gate.numel() > 0:
        metrics = {
            "distillation/gate_mean": valid_gate.mean().item(),
            "distillation/gate_active_ratio": (valid_gate > 0.5).float().mean().item(),
            "distillation/teacher_gap_mean": teacher_gap[response_mask_bool].mean().item(),
            "distillation/abs_loss": Metric(
                AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()
            ),
        }
    _dump_microscope_rows(data, distillation_losses, response_mask_bool, {"gate": gate})
    return distillation_losses, metrics
