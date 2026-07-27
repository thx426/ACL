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

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


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
        sft_loss, sft_metrics = sft_ce_loss(model_output, data)
        policy_loss = policy_loss + sft_loss * sft_loss_coef
        policy_metrics.update(sft_metrics)

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
    in every ablation arm. It is computed here rather than through ``agg_loss``
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
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = data["old_log_probs"].to_padded_tensor(0.0)
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        rollout_is_weights = data.get("rollout_is_weights", None)
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=-distillation_losses.detach(),
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
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
