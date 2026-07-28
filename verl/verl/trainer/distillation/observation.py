# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""trace-opd distillation observation packs (docs/microscope-design.md).

The RAW, self-contained record of what OPD actually saw on every trained
sample: for each response position, the teacher's top-K distribution, the
student's OWN top-K, the student's log-probs ON the teacher support, and the
realized token's both-side log-probs — the minimal complete source from which
every derived quantity (KL, overlap, masses, entropies, rank shifts, and
views not invented yet) is recomputable offline. Scalars are deliberately NOT
stored (they are functions of the pack) with ONE exception: the per-token
training loss, kept as the ground-truth anchor the offline audit re-derives
against.

Storage: one .npz per (step, rank) under TRACE_OBS_DIR, float16 log-probs
(clamped domain [-10, 0] makes fp16 ~1e-2 accurate — audit tolerance),
variable-length rows as flat arrays + offsets. Under sequence parallel each
rank stores its OWN sequence segment plus (sp_rank, sp_size); the reader
reassembles. Collection is best-effort by contract: any failure disables
itself for the run with one warning — observation must never hurt training.
"""
from __future__ import annotations

import logging
import os
import threading

import numpy as np
import torch

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_STEP_BUFFERS: dict[int, list[dict]] = {}
_DISABLED = False
_LAST_FLUSHED_STEP: int | None = None


def observation_dir() -> str | None:
    return os.getenv("TRACE_OBS_DIR") or None


def observation_enabled() -> bool:
    if _DISABLED or not observation_dir():
        return False
    return True


def _step_wanted(step: int) -> bool:
    every = max(1, int(os.getenv("TRACE_OBS_EVERY", "1") or 1))
    return step % every == 0


def _disable(why: str) -> None:
    global _DISABLED
    if not _DISABLED:
        _DISABLED = True
        logger.warning("observation collection disabled: %s", why)


def _to_numpy(t: torch.Tensor, dtype) -> np.ndarray:
    return t.detach().to("cpu", non_blocking=False).numpy().astype(dtype, copy=False)


def collect_micro_batch(*, global_step: int, rows: list[dict], sp_rank: int, sp_size: int) -> None:
    """Buffer one micro-batch of observation rows.

    Each row dict carries numpy arrays already cut to THIS rank's segment of
    the row's RESPONSE window (the caller owns the packed-layout arithmetic):
      resp_hash task_id sample_role won reward turn_spans
      response_ids (R,) response_mask (R,) realized_student_lp (R,)   full-length
      seg_start seg_len                                     segment in [0, R)
      teacher_ids/teacher_lps/student_ids/student_lps/student_support_lps (seg_len, K)
      token_loss (seg_len,)
    """
    if not observation_enabled() or not _step_wanted(int(global_step)):
        return
    try:
        with _LOCK:
            _STEP_BUFFERS.setdefault(int(global_step), []).extend(rows)
            _maybe_flush_older(int(global_step))
    except Exception as err:  # noqa: BLE001 — observation must never hurt training
        _disable(f"collect failed: {type(err).__name__}: {err}")


def _maybe_flush_older(current_step: int) -> None:
    """A new step showing up means every older buffered step is complete."""
    global _LAST_FLUSHED_STEP
    stale = [s for s in _STEP_BUFFERS if s < current_step]
    for step in stale:
        records = _STEP_BUFFERS.pop(step)
        threading.Thread(
            target=_flush_step, args=(step, records), daemon=True
        ).start()
        _LAST_FLUSHED_STEP = step


def flush_all() -> None:
    """Flush everything buffered (checkpoint/exit hook)."""
    with _LOCK:
        steps = list(_STEP_BUFFERS)
        for step in steps:
            _flush_step(step, _STEP_BUFFERS.pop(step))


def _flush_step(step: int, rows: list[dict]) -> None:
    try:
        out_dir = observation_dir()
        if not out_dir:
            return
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        step_dir = os.path.join(out_dir, f"step_{step:05d}")
        os.makedirs(step_dir, exist_ok=True)

        mat_keys = ("teacher_ids", "teacher_lps", "student_ids", "student_lps",
                    "student_support_lps")
        arrays: dict[str, np.ndarray] = {}
        seg_offsets, resp_offsets, spans_offsets = [0], [0], [0]
        spans_flat: list[int] = []
        for row in rows:
            seg_offsets.append(seg_offsets[-1] + int(row["seg_len"]))
            resp_offsets.append(resp_offsets[-1] + len(row["response_ids"]))
            for a, b in row.get("turn_spans") or []:
                spans_flat.extend((int(a), int(b)))
            spans_offsets.append(len(spans_flat))
        if not rows:
            return
        for key in mat_keys:
            arrays[key] = np.concatenate([row[key] for row in rows], axis=0).astype(
                np.int32 if key.endswith("ids") else np.float16, copy=False
            )
        arrays["token_loss"] = np.concatenate(
            [row["token_loss"] for row in rows]
        ).astype(np.float16, copy=False)
        arrays["response_ids"] = np.concatenate(
            [row["response_ids"] for row in rows]
        ).astype(np.int32, copy=False)
        arrays["response_mask"] = np.concatenate(
            [row["response_mask"] for row in rows]
        ).astype(np.int8, copy=False)
        arrays["realized_student_lp"] = np.concatenate(
            [row["realized_student_lp"] for row in rows]
        ).astype(np.float16, copy=False)
        arrays["row_hash"] = np.array([row["resp_hash"] for row in rows], dtype=np.int64)
        arrays["row_won"] = np.array([int(row.get("won", 0)) for row in rows], dtype=np.int8)
        arrays["row_reward"] = np.array(
            [float(row.get("reward", 0.0)) for row in rows], dtype=np.float32
        )
        arrays["row_task"] = np.array([str(row.get("task_id", "")) for row in rows], dtype="U")
        arrays["row_role"] = np.array(
            [str(row.get("sample_role", "bare")) for row in rows], dtype="U"
        )
        arrays["seg_start"] = np.array([int(row["seg_start"]) for row in rows], dtype=np.int32)
        arrays["seg_len"] = np.array([int(row["seg_len"]) for row in rows], dtype=np.int32)
        arrays["resp_len"] = np.array(
            [len(row["response_ids"]) for row in rows], dtype=np.int32
        )
        arrays["seg_offsets"] = np.array(seg_offsets, dtype=np.int64)
        arrays["resp_offsets"] = np.array(resp_offsets, dtype=np.int64)
        arrays["turn_spans"] = np.array(spans_flat, dtype=np.int32)
        arrays["turn_spans_offsets"] = np.array(spans_offsets, dtype=np.int64)
        arrays["loss_mode"] = np.array([str(rows[0].get("loss_mode", "forward_kl_topk"))], dtype="U")
        arrays["log_prob_min_clamp"] = np.array(
            [float(rows[0].get("log_prob_min_clamp", -10.0))], dtype=np.float32
        )
        arrays["sp_info"] = np.array(
            [int(rows[0].get("sp_rank", 0)), int(rows[0].get("sp_size", 1))], dtype=np.int32
        )
        arrays["global_step"] = np.array([step], dtype=np.int64)

        path = os.path.join(step_dir, f"rank{rank}.npz")
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
        logger.info(
            "observation pack: step %s rank %s rows=%d seg_tokens=%d -> %s",
            step, rank, len(rows), int(arrays["seg_offsets"][-1]), path,
        )
    except Exception as err:  # noqa: BLE001
        _disable(f"flush failed: {type(err).__name__}: {err}")
