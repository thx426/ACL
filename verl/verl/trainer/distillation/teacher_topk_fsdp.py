# -*- coding: utf-8 -*-
"""R11a v2: self-teacher top-k scoring on the TRAINING engine (FSDP backend).

Replaces the vLLM prompt_logprobs path (vllm issue #5907: logprobs allocations
live outside the gpu_memory_utilization budget — root cause of the rollout-tail
VRAM spikes and the creeping vLLM-process baseline).

Row semantics are a strict replica of the vLLM single-call path
(agent_loop._compute_teacher_logprobs + extract_prompt_logprobs):
  * scored sequence = teacher_prefix_ids + response_ids (actual tokens)
  * autoregressive row i predicts token i+1; rows [T-1 : T+R-1] predict the R
    response tokens (T = teacher prefix length)
  * per-sample layout: (P-1) zero rows | R valid rows | 1 zero dummy row,
    P = student BARE prompt length (loss reads rows [P-1 : P+R-1])
  * top-K by logprob (rank order == vLLM ranks), fp32 logsumexp, temperature 1.0

v2 incorporates the codex adversarial review (task-ms0edrml-9d9z87):
  H1  explicit teacher_mode provenance ("privileged"|"bare") asserted here —
      a privileged sample whose prefix got lost in transit now fails loud
      instead of silently degrading to a bare (KL~0) teacher.
  H2  scoring runs inside engine.eval_mode() (offload/mode contract). Note on
      root reshard: with param_offload=False the ctx is a no-op by design and
      the root FSDP handle stays unsharded after our forward — the immediately
      following train forward reuses it, so we accept the transient residency
      (validated in smoke; revisit if peaks regress).
  H3  chunks compute topk+logsumexp directly on the fp32 chunk — the full
      [chunk, V] log_softmax output is never materialized; intermediates are
      freed per chunk; a hard sequence-budget guard (40960 = max_model_len).
  H4  enforced upstream: the agent-loop branch only activates for self_teacher.
  M1  dense NO_PADDING batches (all-equal lengths -> torch.stack) need no mask.
  M2  K comes from the rollout side (teacher_topk in extra_fields), asserted
      batch-consistent; no hard-coded 64.
  M3  weights-equality with the vLLM path holds only for sync training with a
      single optimizer update per generation batch (current profile). Guarded
      by the caller keeping this backend off otherwise.
  M4  all structural validation runs BEFORE the first forward, so a malformed
      batch fails on every rank prior to any FSDP collective.
  L1  R == 0 is rejected loudly.

Weights: update_actor runs BEFORE this step's optimizer update, so scoring
weights equal what vLLM would have used. Same model, same prefix, same tokens.

Single-point mode (trace-opd reverse-KL family): ``teacher_topk == 0`` mirrors
the vLLM convention (``prompt_logprobs=0`` -> one column holding the REALIZED
token's logprob; extract_prompt_logprobs pads dummies at ``max(k, 1)`` width).
The estimator losses (k3 / reverse_kl_gated) consume exactly that column, so
scoring skips torch.topk entirely and ships (rows, 1) tensors.
"""
import logging
import os

import torch

logger = logging.getLogger(__name__)

MAX_TEACHER_SEQ = 40960  # == rollout max_model_len; anything longer is a bug
_LOGGED_BACKEND = False


def teacher_scoring_backend() -> str:
    return os.environ.get("TRACE_TEACHER_SCORING_BACKEND", "vllm").strip().lower()


def _unwrap_module(engine):
    for attr in ("module", "model", "actor_module"):
        m = getattr(engine, attr, None)
        if m is not None and callable(getattr(m, "forward", None)):
            return m
    raise RuntimeError(
        f"teacher_topk_fsdp: cannot locate model module on {type(engine).__name__}"
    )


def _per_sample_actual(data, key: str):
    """list[1D LongTensor] of ACTUAL tokens for `key` (nested, dense-equal-length,
    or padded-with-mask)."""
    t = data[key]
    if getattr(t, "is_nested", False):
        return [x.to(torch.long) for x in t.unbind()]
    # dense: either NO_PADDING with all-equal actual lengths (no mask shipped —
    # codex M1), or a padded layout with an attention mask.
    am = data.get("attention_mask", None)
    if am is None:
        return [row.to(torch.long) for row in t]
    if key == "responses":
        plen = data["prompts"].shape[1]
        mask = am[:, plen:]
    elif key == "prompts":
        mask = am[:, : t.shape[1]]
    else:
        raise KeyError(key)
    return [row[m.bool()].to(torch.long) for row, m in zip(t, mask)]


def _score_rows_topk(rows: torch.Tensor, topk: int, chunk: int):
    """top-K ids/logprobs for each row WITHOUT materializing full log_softmax.
    logprob(top_j) = logit(top_j) - logsumexp(row)  (fp32, temperature 1.0)."""
    R = rows.shape[0]
    ids = torch.empty((R, topk), dtype=torch.int32)
    lps = torch.empty((R, topk), dtype=torch.float32)
    for s in range(0, R, chunk):
        e = min(s + chunk, R)
        ch = rows[s:e].float()                        # [c, V] fp32 (only copy)
        v_logit, ix = torch.topk(ch, k=topk, dim=-1)  # rank-ordered
        lz = torch.logsumexp(ch, dim=-1, keepdim=True)
        lps[s:e] = (v_logit - lz).cpu()
        ids[s:e] = ix.to(torch.int32).cpu()
        del ch, v_logit, ix, lz                       # codex H3: free per chunk
    return ids, lps


def _score_rows_realized(rows: torch.Tensor, resp: torch.Tensor, chunk: int):
    """Single-point mode (topk == 0): logprob of the REALIZED response token per
    row, (R, 1) tensors matching the vLLM ``prompt_logprobs=0`` layout. Row i
    predicts response token i, so the gather index is simply resp[i]."""
    R = rows.shape[0]
    ids = resp.to(torch.int32).unsqueeze(-1).cpu()
    lps = torch.empty((R, 1), dtype=torch.float32)
    gather_ix = resp.to(rows.device, torch.long).unsqueeze(-1)
    for s in range(0, R, chunk):
        e = min(s + chunk, R)
        ch = rows[s:e].float()                        # [c, V] fp32 (only copy)
        v_logit = torch.gather(ch, dim=-1, index=gather_ix[s:e])
        lz = torch.logsumexp(ch, dim=-1, keepdim=True)
        lps[s:e] = (v_logit - lz).cpu()
        del ch, v_logit, lz
    return ids, lps


@torch.no_grad()
def compute_teacher_topk_fsdp(data, engine, chunk: int = 2048):
    """Fill data["teacher_ids"/"teacher_logprobs"] by scoring on the FSDP engine.
    Fail-loud by design: a silently wrong teacher is worse than a lost step."""
    global _LOGGED_BACKEND
    tef = data.get("extra_fields", None)
    if tef is None:
        tef = data.get("tool_extra_fields", None)
    if tef is None:
        raise RuntimeError("teacher_topk_fsdp: extra_fields missing from batch — "
                           "TQ did not ship the rollout extras")
    if hasattr(tef, "tolist"):
        tef = tef.tolist()
    extras = list(tef)

    prompts = _per_sample_actual(data, "prompts")
    responses = _per_sample_actual(data, "responses")
    n = len(responses)
    if not (len(extras) == len(prompts) == n):
        raise RuntimeError(f"teacher_topk_fsdp: misaligned batch extras={len(extras)} "
                           f"prompts={len(prompts)} responses={n}")
    if n == 0:
        raise RuntimeError("teacher_topk_fsdp: empty batch — nothing to score")

    # ---- codex M4: full structural validation BEFORE any forward ----------
    topk = None
    plan = []  # (prefix_tensor, response_tensor, P, R)
    for i in range(n):
        ex = extras[i] or {}
        mode = ex.get("teacher_mode", None)
        if mode not in ("privileged", "bare", "sft", "research", "tg", "tg_skip"):
            raise RuntimeError(f"teacher_topk_fsdp: sample {i} has teacher_mode={mode!r} "
                               "— provenance lost in transit (codex H1)")
        k = ex.get("teacher_topk", None)
        if k is None:
            raise RuntimeError(f"teacher_topk_fsdp: sample {i} missing teacher_topk")
        k = int(k)
        if topk is None:
            topk = k
        elif topk != k:
            raise RuntimeError(f"teacher_topk_fsdp: inconsistent topk {topk} vs {k}")
        prefix = ex.get("teacher_prefix_ids", None)
        if mode == "privileged" and prefix is None:
            raise RuntimeError(f"teacher_topk_fsdp: sample {i} is 'privileged' but "
                               "teacher_prefix_ids is missing (codex H1)")
        # trace-opd teacher grounding: a tg row's TRAINING prompt is the
        # doc-prefixed sequence, so its label MUST be scored under the bare
        # prompt carried in teacher_prefix_ids — the exact mirror of the
        # privileged mode. A tg row whose bare prefix got lost cannot fall
        # back to its own (doc) prompt: that would make the label the
        # teacher's own distribution and the loss a silent no-op.
        if mode == "tg":
            if prefix is None:
                raise RuntimeError(f"teacher_topk_fsdp: sample {i} is 'tg' but "
                                   "teacher_prefix_ids (bare prompt) is missing")
            if topk <= 0:
                raise RuntimeError("teacher_topk_fsdp: 'tg' rows need topk > 0 "
                                   "(the soft label is a top-k distribution)")
        resp = responses[i]
        if mode in ("sft", "research", "tg_skip"):
            # trace-opd: an expert (SFT) row is supervised by the cross-entropy
            # alone, a research row by its own PG term over research_mask; both
            # have their response_mask cleared at TQ put time,
            # so whatever sits in teacher_ids/teacher_logprobs is masked out of
            # the distillation loss and its metrics — scoring it would burn a
            # full forward per expert rollout for a value nothing reads.
            plan.append((None, resp, int(prompts[i].numel()), int(resp.numel()), mode))
            continue
        if mode == "bare":
            # vLLM-path semantics: research failed => score under the BARE
            # student prompt (teacher == student => per-token KL ~ 0, neutral).
            prefix = prompts[i]
        else:
            prefix = torch.as_tensor(list(prefix), dtype=torch.long)
        P, R, T = int(prompts[i].numel()), int(resp.numel()), int(prefix.numel())
        if R <= 0:
            raise RuntimeError(f"teacher_topk_fsdp: sample {i} has empty response (codex L1)")
        if P <= 0 or T <= 0:
            raise RuntimeError(f"teacher_topk_fsdp: sample {i} degenerate P={P} T={T}")
        if T + R > MAX_TEACHER_SEQ:
            raise RuntimeError(f"teacher_topk_fsdp: sample {i} teacher seq {T + R} "
                               f"exceeds budget {MAX_TEACHER_SEQ} (codex H3 guard)")
        plan.append((prefix, resp, P, R, mode))

    module = _unwrap_module(engine)
    if not _LOGGED_BACKEND:
        # This line is the smoke-test proof that the fsdp path is actually live
        # on the trainer side (codex M5). The step-16 run showed neither this
        # probe nor vLLM's own stats reaching the driver log, so logging levels
        # cannot be trusted inside a Ray worker at all -- print to stdout, which
        # Ray does forward, and keep the logger call for structured consumers.
        print(f"teacher_topk_fsdp ACTIVE: backend=fsdp topk={topk} n={n} (codex M5 probe)",
              flush=True)
        logger.warning("teacher_topk_fsdp ACTIVE: backend=fsdp topk=%s n=%s (codex M5 probe)",
                       topk, n)
        _LOGGED_BACKEND = True

    was_training = module.training
    # vLLM parity: prompt_logprobs=0 still ships ONE column (the realized
    # token), and extract_prompt_logprobs pads dummies at max(k, 1) width.
    width = max(topk, 1)
    out_ids, out_lps = [], []
    # trace-opd teacher grounding (decoupling ruling 2026-08-02): tg labels
    # ship on their OWN channel. teacher_ids/teacher_logprobs stay pure OPD
    # (tg rows hold zero stubs there, like sft/research rows), so the OPD
    # loss family never sees a bare-policy distribution wearing a teacher
    # field name. The tg fields exist only when the batch has a tg row —
    # tg off means zero extra fields, zero extra transfer.
    has_tg = any(p[4] == "tg" for p in plan)
    tg_ids, tg_lps = [], []
    # codex H2: run under the engine's eval contract (offload load/unload, mode).
    with engine.eval_mode(zero_grad_on_exit=False):
        module.eval()
        try:
            device = next(module.parameters()).device
            for i, (prefix, resp, P, R, mode) in enumerate(plan):
                if prefix is None:  # sft/research/tg_skip row: no label, no forward
                    out_ids.append(torch.zeros((P + R, width), dtype=torch.int32))
                    out_lps.append(torch.zeros((P + R, width), dtype=torch.float32))
                    if has_tg:
                        tg_ids.append(torch.zeros((P + R, width), dtype=torch.int32))
                        tg_lps.append(torch.zeros((P + R, width), dtype=torch.float32))
                    continue
                T = int(prefix.numel())
                seq = torch.cat([prefix.cpu(), resp.cpu()]).to(device).unsqueeze(0)
                logits = module(input_ids=seq, use_cache=False).logits[0]
                rows = logits[T - 1 : T + R - 1]
                if rows.shape[0] != R:
                    raise RuntimeError(f"teacher_topk_fsdp: sample {i} rows "
                                       f"{rows.shape[0]} != R {R}")
                if topk > 0:
                    ids_i, lps_i = _score_rows_topk(rows, topk, chunk)
                else:
                    ids_i, lps_i = _score_rows_realized(rows, resp, chunk)
                if mode == "tg":
                    # trace-opd teacher grounding: the tg label is the bare
                    # policy's top-k RENORMALISED to a proper distribution
                    # (design ruling: the label semantics must be clean and the
                    # resulting soft-CE non-negative; the truncated mass is
                    # ~5e-4 on the policy's own trajectory). Done here, once,
                    # at the only point that knows which rows are tg rows —
                    # compute_tg_soft_ce consumes an already-proper label.
                    lps_i = lps_i - torch.logsumexp(lps_i, dim=-1, keepdim=True)
                del logits, rows
                # Alignment sentinel across the first few rows, not just row 0:
                # with a single column (topk == 0) one saturated glue token can
                # legitimately round its logprob to exactly 0.0 in fp32, but a
                # run of them cannot — while a misaligned window still reads the
                # all-zero dummy row and fails here on every row.
                if not (lps_i[: min(8, R)] < 0).any():
                    raise RuntimeError("teacher_topk_fsdp: first response rows look "
                                       "like dummy rows — alignment is off")
                zid = torch.zeros((P - 1, width), dtype=torch.int32)
                zlp = torch.zeros((P - 1, width), dtype=torch.float32)
                did = torch.zeros((1, width), dtype=torch.int32)
                dlp = torch.zeros((1, width), dtype=torch.float32)
                full_ids = torch.cat([zid, ids_i, did], dim=0)
                full_lps = torch.cat([zlp, lps_i, dlp], dim=0)
                if mode == "tg":
                    tg_ids.append(full_ids)
                    tg_lps.append(full_lps)
                    out_ids.append(torch.zeros((P + R, width), dtype=torch.int32))
                    out_lps.append(torch.zeros((P + R, width), dtype=torch.float32))
                else:
                    out_ids.append(full_ids)
                    out_lps.append(full_lps)
                    if has_tg:
                        tg_ids.append(torch.zeros((P + R, width), dtype=torch.int32))
                        tg_lps.append(torch.zeros((P + R, width), dtype=torch.float32))
        finally:
            if was_training:
                module.train()
            torch.cuda.empty_cache()

    data["teacher_ids"] = torch.nested.nested_tensor(out_ids, layout=torch.jagged)
    data["teacher_logprobs"] = torch.nested.nested_tensor(out_lps, layout=torch.jagged)
    if has_tg:
        data["tg_label_ids"] = torch.nested.nested_tensor(tg_ids, layout=torch.jagged)
        data["tg_label_logprobs"] = torch.nested.nested_tensor(tg_lps, layout=torch.jagged)
    return data
