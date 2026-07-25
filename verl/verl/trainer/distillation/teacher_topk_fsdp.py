# -*- coding: utf-8 -*-
"""R11a: self-teacher top-k scoring on the TRAINING engine (FSDP), replacing the
vLLM prompt_logprobs path (vLLM issue #5907: logprobs allocations live outside
the gpu_memory_utilization budget with no scheduler protection — the root cause
of both the rollout-tail VRAM spikes and the creeping baseline).

This module is imported by engine_workers.update_actor. Semantics are a strict
replica of the vLLM path (agent_loop._compute_teacher_logprobs single-call):

  * scored sequence  = teacher_prefix_ids + response_ids       (actual tokens)
  * autoregressive row i predicts token i+1  ==  vLLM's "row i is the
    distribution at position i predicting token i+1" (extract_prompt_logprobs
    drops the leading None row and appends one dummy row)
  * rows [T-1 : T+R-1] predict the R response tokens (T = teacher prefix len)
  * per-sample layout written to the batch:  (P-1) zero rows | R valid rows |
    1 zero dummy row, P = student BARE prompt length — byte-compatible with
    what the loss side expects (no_padding_2_padding reads rows [P-1 : P+R-1])
  * top-64 by logprob, fp32 log_softmax (vLLM computes logprobs on raw logits;
    scoring temperature is 1.0 so no temperature division here either)

Weights: update_actor runs BEFORE the optimizer step of this training step, so
the scoring weights equal the weights vLLM would have used (post update_weights
of the previous step). Same model, same prefix, same tokens => same math.
"""
import logging
import os

import torch

logger = logging.getLogger(__name__)

_K3_WARNED = False


def teacher_scoring_backend() -> str:
    return os.environ.get("TRACE_TEACHER_SCORING_BACKEND", "vllm").strip().lower()


def _unwrap_module(engine):
    """Best-effort handle to the forward-able (FSDP-wrapped) module."""
    for attr in ("module", "model", "actor_module", "module_fsdp"):
        m = getattr(engine, attr, None)
        if m is not None and callable(getattr(m, "forward", None)):
            return m
    raise RuntimeError(
        "teacher_topk_fsdp: cannot locate the model module on engine "
        f"{type(engine).__name__}; tried .module/.model/.actor_module"
    )


def _per_sample_actual(data, key: str):
    """Return list[1D LongTensor] of ACTUAL (unpadded) tokens for `key`.

    Supports both nested/jagged tensors (NO_PADDING pad mode) and padded
    tensors (falls back to the matching attention mask)."""
    t = data[key]
    if getattr(t, "is_nested", False):
        return [x.to(torch.long) for x in t.unbind()]
    # padded fallback: derive lengths from masks
    if key == "responses":
        mask = data.get("response_attention_mask")
        if mask is None:
            am = data["attention_mask"]  # [b, P+R]
            plen = data["prompts"].shape[1]
            mask = am[:, plen:]
        return [row[m.bool()].to(torch.long) for row, m in zip(t, mask)]
    if key == "prompts":
        am = data["attention_mask"]
        plen = t.shape[1]
        pm = am[:, :plen]
        return [row[m.bool()].to(torch.long) for row, m in zip(t, pm)]
    raise KeyError(key)


@torch.no_grad()
def compute_teacher_topk_fsdp(data, engine, topk: int = 64, chunk: int = 2048):
    """Fill data["teacher_ids"/"teacher_logprobs"] by scoring on the FSDP engine.

    Fail-loud by design (project discipline): any structural mismatch raises —
    a silently wrong teacher is worse than a lost step."""
    # TQ path: AgentLoopOutput.as_dict() ships the whole extra_fields dict per
    # sample (NonTensorStack under key "extra_fields"); the non-TQ DataProto
    # path ships it as "tool_extra_fields". Support both.
    tef = data.get("extra_fields", None)
    if tef is None:
        tef = data.get("tool_extra_fields", None)
    if tef is None:
        raise RuntimeError("teacher_topk_fsdp: extra_fields/tool_extra_fields missing from batch")
    # NonTensorStack / numpy object array / list — normalize to list[dict]
    if hasattr(tef, "tolist"):
        tef = tef.tolist()
    extras = list(tef)

    prompts = _per_sample_actual(data, "prompts")     # student BARE prompts (P_i)
    responses = _per_sample_actual(data, "responses")  # actual response tokens (R_i)
    n = len(responses)
    if not (len(extras) == len(prompts) == n):
        raise RuntimeError(
            f"teacher_topk_fsdp: batch misaligned extras={len(extras)} prompts={len(prompts)} responses={n}"
        )

    module = _unwrap_module(engine)
    device = next(module.parameters()).device
    was_training = module.training
    module.eval()

    global _K3_WARNED
    out_ids, out_lps = [], []
    try:
        for i in range(n):
            ex = extras[i] or {}
            prefix = ex.get("teacher_prefix_ids", None)
            resp = responses[i]
            P = int(prompts[i].numel())
            R = int(resp.numel())
            if prefix is None:
                # bare-teacher sample (research failed => neutral): zero rows,
                # same semantics as the vLLM path emitting nothing => KL≈0.
                out_ids.append(torch.zeros((P + R, topk), dtype=torch.int32))
                out_lps.append(torch.zeros((P + R, topk), dtype=torch.float32))
                continue
            if ex.get("teacher_block_starts") and not _K3_WARNED:
                logger.warning("teacher_topk_fsdp: K3 block starts present but K3 is "
                               "not implemented on the fsdp backend — single-prefix scoring")
                _K3_WARNED = True
            pfx = torch.as_tensor(list(prefix), dtype=torch.long)
            T = int(pfx.numel())
            seq = torch.cat([pfx, resp.cpu()]).to(device).unsqueeze(0)   # [1, T+R]
            logits = module(input_ids=seq, use_cache=False).logits[0]     # [T+R, V]
            rows = logits[T - 1 : T + R - 1]                              # predict R response tokens
            if rows.shape[0] != R:
                raise RuntimeError(f"teacher_topk_fsdp: sample {i} rows {rows.shape[0]} != R {R}")
            ids_i = torch.empty((R, topk), dtype=torch.int32)
            lps_i = torch.empty((R, topk), dtype=torch.float32)
            for s in range(0, R, chunk):
                e = min(s + chunk, R)
                lp = torch.log_softmax(rows[s:e].float(), dim=-1)         # fp32, raw logits (temp=1.0)
                v, ix = torch.topk(lp, k=topk, dim=-1)                    # rank-ordered == vLLM ranks
                ids_i[s:e] = ix.to(torch.int32).cpu()
                lps_i[s:e] = v.cpu()
            del logits, rows
            # sentinel parity with the vLLM path
            if R > 0 and not (lps_i[0] < 0).any():
                raise RuntimeError("teacher_topk_fsdp: first response row looks like a dummy row")
            zid = torch.zeros((P - 1, topk), dtype=torch.int32)
            zlp = torch.zeros((P - 1, topk), dtype=torch.float32)
            did = torch.zeros((1, topk), dtype=torch.int32)
            dlp = torch.zeros((1, topk), dtype=torch.float32)
            out_ids.append(torch.cat([zid, ids_i, did], dim=0))           # [P+R, topk]
            out_lps.append(torch.cat([zlp, lps_i, dlp], dim=0))
    finally:
        if was_training:
            module.train()
        torch.cuda.empty_cache()

    data["teacher_ids"] = torch.nested.nested_tensor(out_ids, layout=torch.jagged)
    data["teacher_logprobs"] = torch.nested.nested_tensor(out_lps, layout=torch.jagged)
    return data
