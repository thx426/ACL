"""Probe tool: exact full-vocab distribution geometry on a frozen probe set
(analysis-design.md §1 rows 1-5 — d_init / d_gain / d_frozen / alignment
fingerprint), run offline at every val point. Nothing here touches training.

Four conditions per sweep: pi_k bare, pi_k (+) current doc, pi_k (+) frozen
doc, pi_0 bare (cached once). Bare distributions are persisted per sweep so
the NEXT sweep can difference them into the training-move direction
Delta_train for the alignment fingerprint.

The model is abstracted behind ProbeModel; HFProbeModel (transformers, lazy
imports) is the real backend on the server. Probe-set construction needs the
executor scaffold, which only the environment side can provide — pass a
scaffold_provider (server-side wiring; see build_probe_set).
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .schema import ProbePoint

TOP_UPLIFT = 32  # candidate-set size for the robust fingerprint


# ------------------------------------------------------------------ probe set


@dataclass
class ProbeItem:
    probe_id: str
    task_id: str
    scaffold: str                    # bare executor system prompt
    dialogue: list[dict[str, str]]   # user/assistant turns up to the decision point
    doc_fix: str                     # frozen doc (step-1 snapshot), "" if none
    meta: dict[str, Any] = field(default_factory=dict)


def save_probe_set(path: Path, items: list[ProbeItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(asdict(i), ensure_ascii=False) for i in items),
        encoding="utf-8",
    )


def load_probe_set(path: Path) -> list[ProbeItem]:
    return [
        ProbeItem(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def messages_for(item: ProbeItem, doc: str | None) -> list[dict[str, str]]:
    from ..core.protocol import inject_guidance

    system = inject_guidance(item.scaffold, doc) if doc else item.scaffold
    return [{"role": "system", "content": system}, *item.dialogue]


def build_probe_set(
    run_dir: Path,
    scaffold_provider: Callable[[str], str],
    *,
    n_items: int = 64,
    turns_at: tuple[int, ...] = (1, 6),
    seed: int = 7,
) -> list[ProbeItem]:
    """Reconstruct decision-point prefixes from rollout dumps.

    Assistant turns are re-rendered from the protocol contract
    (<reasoning>/<action>); whitespace may differ from the original stream,
    which is fine — all four conditions share the exact same prefix, and the
    set is frozen once, so every cross-sweep comparison stays valid.
    """
    from ..viz.schema import read_json

    rng = random.Random(seed)
    candidates = sorted((run_dir / "rollouts").glob("step_*/*.json"))
    rng.shuffle(candidates)
    items: list[ProbeItem] = []
    for file in candidates:
        if len(items) >= n_items:
            break
        d = read_json(file)
        if not d:
            continue
        traj = d.get("trajectory") or {}
        steps = traj.get("steps") or []
        for turn in turns_at:
            if len(items) >= n_items or len(steps) < turn:
                continue
            dialogue: list[dict[str, str]] = [
                {"role": "user", "content": traj.get("initial_obs", "")}
            ]
            for s in steps[: turn - 1]:
                dialogue.append({
                    "role": "assistant",
                    "content": (
                        f"<reasoning>{s['thought']}</reasoning>\n"
                        f"<action>{s['action_raw'] or s['action_norm']}</action>"
                    ),
                })
                dialogue.append({"role": "user", "content": s["observation"]})
            task_id = str(d.get("task_id", ""))
            items.append(ProbeItem(
                probe_id=f"{file.stem[:40]}#t{turn}",
                task_id=task_id,
                scaffold=scaffold_provider(task_id),
                dialogue=dialogue,
                doc_fix="",
                meta={"split": d.get("split", ""), "source": file.name,
                      "turn": turn},
            ))
    return items


# ------------------------------------------------------------------- models


class ProbeModel(Protocol):
    def logprobs(self, messages: list[dict[str, str]]) -> "Any":
        """Full-vocab log-probs (1-D array) of the NEXT token at the prefix end."""
        ...


class HFProbeModel:
    """transformers backend (server-side; lazy imports keep this module light)."""

    def __init__(self, model_path: str, device: str = "cuda", dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=getattr(torch, dtype), device_map=device
        ).eval()

    def logprobs(self, messages):
        torch = self._torch
        ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            logits = self.model(ids).logits[0, -1].float()
        return torch.log_softmax(logits, dim=-1).cpu().numpy()


# ------------------------------------------------------------------ the math


def kl(logp, logq) -> float:
    import numpy as np

    return float(np.sum(np.exp(logp) * (logp - logq)))


def cosine(u, v) -> float:
    import numpy as np

    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(np.dot(u, v) / (nu * nv)) if nu > 0 and nv > 0 else float("nan")


def uplift_overlap(delta_a, delta_b, top: int = TOP_UPLIFT) -> float:
    """Jaccard of the top uplifted token sets — the robust fingerprint twin."""
    import numpy as np

    a = set(np.argsort(delta_a)[-top:].tolist())
    b = set(np.argsort(delta_b)[-top:].tolist())
    return len(a & b) / len(a | b)


def _bootstrap_ci(values, rng, n_boot: int = 500) -> tuple[float, float]:
    import numpy as np

    values = np.asarray(values, dtype=float)
    means = [
        float(np.mean(values[rng.integers(0, len(values), len(values))]))
        for _ in range(n_boot)
    ]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sweep(
    *,
    step: int,
    model: ProbeModel,
    probes: list[ProbeItem],
    current_docs: dict[str, str],
    base_logprobs: dict[str, Any],
    prev_bare: dict[str, Any] | None,
    out_dir: Path | None = None,
    seed: int = 0,
) -> tuple[ProbePoint, dict[str, Any]]:
    """One sweep at training step k. Returns the aggregated ProbePoint and the
    bare distributions of this sweep (feed them back as prev_bare next time).

    base_logprobs: probe_id -> pi_0 bare log-probs (cached once, first sweep).
    current_docs: task_id -> this step's researcher doc ("" -> skip d_gain).
    """
    import numpy as np

    d_init, d_gain, d_frozen, aligns, aligns_robust = [], [], [], [], []
    bare_out: dict[str, Any] = {}
    for item in probes:
        bare = model.logprobs(messages_for(item, None))
        bare_out[item.probe_id] = bare
        d_init.append(kl(bare, base_logprobs[item.probe_id]))
        if item.doc_fix:
            frozen = model.logprobs(messages_for(item, item.doc_fix))
            d_frozen.append(kl(frozen, bare))
        doc = current_docs.get(item.task_id, "")
        if doc:
            teacher = model.logprobs(messages_for(item, doc))
            d_gain.append(kl(teacher, bare))
            if prev_bare is not None and item.probe_id in prev_bare:
                delta_doc = np.exp(teacher) - np.exp(bare)
                delta_train = np.exp(bare) - np.exp(prev_bare[item.probe_id])
                aligns.append(cosine(delta_doc, delta_train))
                aligns_robust.append(uplift_overlap(delta_doc, delta_train))

    rng = np.random.default_rng(seed)
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
    point = ProbePoint(
        step=step,
        d_init=mean(d_init),
        d_gain=mean(d_gain),
        d_frozen=mean(d_frozen),
        align=mean(aligns),
        align_robust=mean(aligns_robust),
        ci={
            name: _bootstrap_ci(vals, rng)
            for name, vals in (
                ("d_init", d_init), ("d_gain", d_gain),
                ("d_frozen", d_frozen), ("align", aligns),
            )
            if len(vals) >= 2
        },
    )
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / f"bare_step{step:05d}.npz",
            **{pid: arr.astype(np.float16) for pid, arr in bare_out.items()},
        )
    return point, bare_out


def load_bare(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path) as z:
        return {pid: z[pid].astype(np.float32) for pid in z.files}
