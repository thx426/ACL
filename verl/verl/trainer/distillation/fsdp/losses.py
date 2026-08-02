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


import torch
import torch.nn.functional as F

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def _chunked_topk_log_probs(
    logits: torch.Tensor,
    topk_ids: torch.Tensor,
    chunk_size: int = 4096,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    """Compute log_softmax(logits).gather(topk_ids) without materializing [B, T, V].

    Uses the identity:
        log_softmax(x).gather(idx) == x.gather(idx) - logsumexp(x, keepdim=True)
    Streams the reduction in chunks of `chunk_size` tokens along (B*T) with fp32
    logsumexp for numerical stability.

    Args:
        logits:    [B, T, V] student logits.
        topk_ids:  [B, T, K] indices to gather.
        chunk_size: number of tokens per chunk; only affects memory, not numerics.
        out_dtype: dtype of the returned tensor. Default None keeps the
            historical behaviour (logits.dtype — BF16 on the training path).
            codex impl-review P0-2: the tg soft-CE needs fp32 here — a BF16
            round-trip carries ~1e-3 absolute error on log-probs, which
            swamps the near-converged KL (~1e-5) and flips it negative, and
            clamp_min(0) then kills the loss AND the gradient exactly in the
            fine-tuning regime. OPD callers keep the default so their
            baselines stay byte-identical.

    Returns:
        [B, T, K] tensor in `out_dtype` (or `logits.dtype` if None).
    """
    out_dtype = out_dtype if out_dtype is not None else logits.dtype
    B, T, V = logits.shape
    K = topk_ids.shape[-1]
    flat_logits = logits.reshape(-1, V)  # [N, V]
    flat_topk = topk_ids.reshape(-1, K)  # [N, K]
    N = flat_logits.shape[0]

    # Edge case: empty input (e.g. fully-padded micro-batch).
    if N == 0:
        return torch.empty((B, T, K), dtype=out_dtype, device=logits.device)

    out = torch.empty((N, K), dtype=out_dtype, device=logits.device)
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        chunk_logits_fp32 = flat_logits[s:e].float()
        log_z = torch.logsumexp(chunk_logits_fp32, dim=-1, keepdim=True)  # [c, 1]
        chunk_topk_logits = torch.gather(chunk_logits_fp32, dim=-1, index=flat_topk[s:e])
        out[s:e] = (chunk_topk_logits - log_z).to(out_dtype)
    return out.reshape(B, T, K)


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def _observation_payload(
    student_logits, student_log_probs, student_topk_ids, student_topk_log_probs,
    teacher_topk_log_probs, teacher_topk_ids, loss_config, use_chunked_topk,
):
    """trace-opd observation packs (docs/microscope-design.md): the raw
    distribution quartet of this rank's sequence segment, collected ONLY when
    TRACE_OBS_DIR is set. student_topk_log_probs here is the student ON the
    teacher support (pre-clamp); the student's OWN top-k log-probs are the one
    piece the loss never needed, computed here on demand (a second chunked
    pass when chunking — memory-safe, obs-gated)."""
    try:
        from ..observation import observation_enabled
    except Exception:  # noqa: BLE001 — best-effort contract: no module, no packs
        return None

    if not observation_enabled():
        return None
    with torch.no_grad():
        if use_chunked_topk or student_log_probs is None:
            student_own_lps = _chunked_topk_log_probs(
                student_logits,
                student_topk_ids,
                chunk_size=getattr(loss_config, "chunked_topk_chunk_size", 4096),
            )
        else:
            student_own_lps = torch.gather(student_log_probs, dim=-1, index=student_topk_ids)
        return {
            "teacher_ids": teacher_topk_ids.detach(),
            "teacher_lps": teacher_topk_log_probs.detach(),
            "student_ids": student_topk_ids.detach(),
            "student_lps": student_own_lps.detach(),
            "student_support_lps": student_topk_log_probs.detach(),
        }


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups
    # ``use_chunked_topk`` (opt-in, default off) trades latency for memory:
    # the chunked path streams logsumexp + gather to avoid the [B, T, V]
    # log_softmax buffer, enabling long-context (>=64K) where the default
    # F.log_softmax path OOMs. See ``DistillationLossConfig.use_chunked_topk``
    # for trade-offs and benchmark numbers.
    loss_config: DistillationLossConfig = config.distillation_loss
    use_chunked_topk = getattr(loss_config, "use_chunked_topk", False)
    if use_chunked_topk:
        # log_softmax is monotonic, so topk(logits) == topk(log_softmax(logits)).
        student_topk_ids = torch.topk(student_logits, k=teacher_topk_ids.shape[-1], dim=-1).indices
        student_topk_log_probs = _chunked_topk_log_probs(
            student_logits,
            teacher_topk_ids,
            chunk_size=getattr(loss_config, "chunked_topk_chunk_size", 4096),
        )
    else:
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        student_topk_ids = torch.topk(student_log_probs, k=teacher_topk_ids.shape[-1], dim=-1).indices
        student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    obs = _observation_payload(
        student_logits, None if use_chunked_topk else student_log_probs,
        student_topk_ids, student_topk_log_probs,
        teacher_topk_log_probs, teacher_topk_ids, loss_config, use_chunked_topk,
    )
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)

    # Diagnostics for tracking teacher/student top-k overlap in OPD, following
    # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016).
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)
    token_kl = teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)
    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
    )

    # Truncated (top-k) teacher entropy — the decision-point/uncertainty signal
    # of the trace-opd distillation microscope (per-token, cheap to carry).
    teacher_topk_probs = teacher_topk_log_probs.exp()
    teacher_topk_entropy = -(teacher_topk_probs * teacher_topk_log_probs).sum(dim=-1)

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
        "teacher_topk_entropy": teacher_topk_entropy,
        "obs": obs,
    }


def compute_reverse_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> dict:
    """Reverse-KL mirror of ``compute_forward_kl_topk`` on the SAME top-k data.

    Per token the loss is the KL between the two (K+1)-bucket categorical
    distributions induced by the teacher's top-k support plus a tail bucket:

        sum_{i in topK} p_S(i) * (log p_S(i) - log p_T(i))
          + (1 - m_S) * (log(1 - m_S) - log(1 - m_T))

    with m_S / m_T the two distributions' top-k masses. The member sum alone is
    the truncated reverse KL; the tail bucket is what makes the whole thing a
    true KL (non-negative, and it is the term that penalises student mass
    parked OUTSIDE the teacher's top-k — exactly the mode-seeking pressure a
    support-restricted sum cannot express). Deliberately a standalone mirror
    rather than a refactor: ``compute_forward_kl_topk`` is the 4B mainline's
    hot path and stays byte-identical.
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    loss_config: DistillationLossConfig = config.distillation_loss
    use_chunked_topk = getattr(loss_config, "use_chunked_topk", False)
    if use_chunked_topk:
        student_topk_ids = torch.topk(student_logits, k=teacher_topk_ids.shape[-1], dim=-1).indices
        student_topk_log_probs = _chunked_topk_log_probs(
            student_logits,
            teacher_topk_ids,
            chunk_size=getattr(loss_config, "chunked_topk_chunk_size", 4096),
        )
    else:
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        student_topk_ids = torch.topk(student_log_probs, k=teacher_topk_ids.shape[-1], dim=-1).indices
        student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)

    # Masses come off the RAW log-probs (same order as the forward path: mass
    # first, clamp after) and stay in the graph — the tail term is where the
    # out-of-support gradient comes from.
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    obs = _observation_payload(
        student_logits, None if use_chunked_topk else student_log_probs,
        student_topk_ids, student_topk_log_probs,
        teacher_topk_log_probs, teacher_topk_ids, loss_config, use_chunked_topk,
    )
    if loss_config.log_prob_min_clamp is not None:
        clamped_student_lp = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        clamped_teacher_lp = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    else:
        clamped_student_lp = student_topk_log_probs
        clamped_teacher_lp = teacher_topk_log_probs
    # Member sum: q||p with the roles of kl_divergence's arguments swapped.
    member_kl = kl_divergence(log_q=clamped_teacher_lp, log_p=clamped_student_lp)

    # Tail bucket. exp/sum can carry the mass a hair past 1.0 in fp32 and
    # clamping members ABOVE the floor can only add mass, so cap strictly
    # below 1 before log1p; the same -10 floor as the members bounds the term.
    tail_floor = loss_config.log_prob_min_clamp if loss_config.log_prob_min_clamp is not None else -80.0
    student_tail = (1.0 - student_mass.clamp(min=0.0, max=1.0 - 1e-6)).clamp(min=1e-6)
    teacher_tail = (1.0 - teacher_mass.clamp(min=0.0, max=1.0 - 1e-6)).clamp(min=1e-6)
    log_student_tail = student_tail.log().clamp_min(tail_floor)
    log_teacher_tail = teacher_tail.log().clamp_min(tail_floor)
    tail_kl = student_tail * (log_student_tail - log_teacher_tail)

    distillation_losses = member_kl + tail_kl

    # Same overlap diagnostics as the forward path (the definition is
    # direction-agnostic: teacher-token forward-KL contribution on overlapped
    # tokens), so the 2604.13016 curves stay comparable across arms.
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)
    token_kl = clamped_teacher_lp.exp() * (clamped_teacher_lp - clamped_student_lp)
    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
    )

    teacher_topk_probs = clamped_teacher_lp.exp()
    teacher_topk_entropy = -(teacher_topk_probs * clamped_teacher_lp).sum(dim=-1)

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
        "teacher_topk_entropy": teacher_topk_entropy,
        "obs": obs,
    }


def compute_tg_soft_ce(
    student_logits: torch.Tensor,
    tg_label_log_probs: torch.Tensor,
    tg_label_ids: torch.Tensor,
    tg_token_mask: torch.Tensor,
    config: DistillationConfig,
) -> torch.Tensor:
    """trace-opd teacher grounding: per-token soft cross-entropy over tg rows.

    A tg row is a bare-policy SUCCESS trajectory whose TRAINING prompt was
    swapped to the doc prefix, so on these rows ``student_logits`` ARE the
    teacher q(.|doc, h). The label is the bare policy's top-k distribution,
    renormalised and detached at scoring time. Per token:

        loss_t = sum_k  label_k * (log label_k - log q_k)

    i.e. KL(sg[label] || q) restricted to the label support — non-negative by
    construction (label sums to 1, q's support mass <= 1), gradient q - label
    bounded by 2 per logit. Standalone by design (user decoupling ruling
    2026-08-02): NO OPD clamp applies here — log_prob_min_clamp would lift the
    renormalised label tail above its true probability and kill the gradient
    exactly where q parks mass outside the label support (codex tg-review
    P0-2), and loss_max_clamp would zero the gradient on legitimately large
    supervision. The gather runs chunked (logit - logsumexp), so no
    [.., V] log_softmax buffer and no duplicated OPD compute.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab) — SP-sliced like the OPD
            formulas' input.
        tg_label_log_probs / tg_label_ids: nested (rows, P+R, K); non-tg rows
            carry zero stubs that are never read (masked out below).
        tg_token_mask: (bsz, seqlen) packed bool, True on the full span of tg
            rows; sliced here for SP exactly like the label tensors.

    Returns:
        (bsz, seqlen/sp_size) fp32 per-token losses, zero outside tg positions.
    """
    assert tg_label_log_probs.is_nested and tg_label_ids.is_nested
    label_lps = tg_label_log_probs.values().unsqueeze(0)  # (1, total_nnz, K)
    label_ids = tg_label_ids.values().unsqueeze(0)  # (1, total_nnz, K)
    if get_ulysses_sequence_parallel_world_size() > 1:
        label_lps = slice_input_tensor(label_lps, dim=1)
        label_ids = slice_input_tensor(label_ids, dim=1)
        # bool pads to False (zero), so SP padding positions never enter.
        tg_token_mask = slice_input_tensor(tg_token_mask, dim=1)
    assert label_lps.shape[:2] == label_ids.shape[:2] == student_logits.shape[:2], (
        f"tg label/logits misalignment: labels {tuple(label_lps.shape[:2])} vs "
        f"logits {tuple(student_logits.shape[:2])}"
    )
    assert tg_token_mask.shape == student_logits.shape[:2], (
        f"tg_token_mask {tuple(tg_token_mask.shape)} vs logits {tuple(student_logits.shape[:2])}"
    )

    out = torch.zeros(student_logits.shape[:2], dtype=torch.float32, device=student_logits.device)
    mask = tg_token_mask[0].bool()
    if not mask.any():
        return out
    flat_logits = student_logits[0][mask]  # [nnz_tg, V] — keeps the autograd graph
    flat_ids = label_ids[0][mask].to(torch.long)  # [nnz_tg, K]
    flat_lps = label_lps[0][mask].float()  # [nnz_tg, K] renormalised label
    loss_config: DistillationLossConfig = config.distillation_loss
    # codex impl-review P0-2: q_lp stays fp32 end to end — a BF16 round-trip
    # would flip near-converged KL negative and clamp_min(0) would zero both
    # loss and gradient exactly where tg is doing its fine-grained work.
    q_lp = _chunked_topk_log_probs(
        flat_logits.unsqueeze(0),
        flat_ids.unsqueeze(0),
        chunk_size=getattr(loss_config, "chunked_topk_chunk_size", 4096),
        out_dtype=torch.float32,
    )[0]  # [nnz_tg, K]
    ce = (flat_lps.exp() * (flat_lps - q_lp)).sum(dim=-1)  # [nnz_tg]
    out[0][mask] = ce
    return out
