"""Content hash that joins a trajectory's rollout dump with its training-side
distillation-microscope rows (analysis-design.md §2).

The SAME polynomial rolling hash is implemented independently inside the
vendored verl patch (trainer/distillation/losses.py::_resp_hash) — the two
must stay bit-identical; test_hashing.py pins golden values for both sides.
"""
from __future__ import annotations

from typing import Iterable

_MOD = (1 << 63) - 1
_MUL = 1000003


def response_token_hash(ids: Iterable[int]) -> int:
    h = 0
    for x in ids:
        h = (h * _MUL + int(x)) & _MOD
    return h
