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
    scaffold_provider: Callable[[str], str] | None = None,
    *,
    n_items: int = 64,
    turns_at: tuple[int, ...] = (1, 6),
    seed: int = 7,
    freeze_doc_step: int = 1,
) -> list[ProbeItem]:
    """Reconstruct ACTION decision points from rollout dumps.

    The prefix ends inside a partial assistant turn — history, then the
    current turn's real reasoning, then an open ``<action>`` tag — so the
    next-token distribution IS the action choice (that is where the doc's
    privileged signal lives, and what the action-flow/fork figures read).
    Assistant turns are re-rendered from the protocol contract; whitespace
    may differ from the original stream, which is fine: all four conditions
    share the exact same frozen prefix, so cross-sweep comparisons stay valid.

    scaffold_provider defaults to the protocol's task-agnostic executor
    scaffold. doc_fix is the task's earliest archived doc at or before
    ``freeze_doc_step`` (the frozen reference of analysis-design.md §1-2).
    """
    from ..core import protocol
    from ..viz.schema import read_json

    if scaffold_provider is None:
        scaffold_provider = lambda task_id: protocol.SYSTEM_PROMPT_BARE  # noqa: E731

    rng = random.Random(seed)
    candidates = sorted((run_dir / "rollouts").glob("step_*/*.json"))
    rng.shuffle(candidates)
    items: list[ProbeItem] = []
    seen_tasks: set[tuple[str, int]] = set()
    for file in candidates:
        if len(items) >= n_items:
            break
        d = read_json(file)
        if not d:
            continue
        traj = d.get("trajectory") or {}
        steps = traj.get("steps") or []
        task_id = str(d.get("task_id", ""))
        for turn in turns_at:
            if len(items) >= n_items or len(steps) < turn:
                continue
            if (task_id, turn) in seen_tasks:  # one probe per (task, depth)
                continue
            seen_tasks.add((task_id, turn))
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
            here = steps[turn - 1]
            dialogue.append({  # partial turn: next token = the action choice
                "role": "assistant",
                "content": f"<reasoning>{here['thought']}</reasoning>\n<action>",
            })
            items.append(ProbeItem(
                probe_id=f"{file.stem[:40]}#t{turn}",
                task_id=task_id,
                scaffold=scaffold_provider(task_id),
                dialogue=dialogue,
                doc_fix=doc_for(run_dir, task_id, step=freeze_doc_step,
                                earliest=True) or "",
                meta={"split": d.get("split", ""), "source": file.name,
                      "turn": turn},
            ))
    return items


# --------------------------------------------------------------- doc loaders


def doc_for(run_dir: Path, task_id: str, *, step: int, earliest: bool = False) -> str | None:
    """The task's archived doc at policy_version <= step — latest such
    session by default (the 'current doc' of a sweep), or the earliest
    (the frozen reference)."""
    from ..viz.schema import parse_research_dirname

    task_dir = run_dir / "research" / task_id.replace("/", "__").replace(" ", "_")
    if not task_dir.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for session in task_dir.iterdir():
        ver = parse_research_dirname(session.name)
        if ver is None or ver > step or not (session / "doc.md").is_file():
            continue
        if best is None or (ver < best[0] if earliest else ver > best[0]):
            best = (ver, session)
    return (best[1] / "doc.md").read_text(encoding="utf-8") if best else None


def current_docs_for(run_dir: Path, probes: list[ProbeItem], step: int) -> dict[str, str]:
    return {
        item.task_id: doc_for(run_dir, item.task_id, step=step) or ""
        for item in probes
    }


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
        # Decision-point prefixes end in a partial assistant turn (open
        # <action> tag) — continue it instead of opening a new turn.
        partial = messages[-1]["role"] == "assistant"
        ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=not partial,
            continue_final_message=partial,
            return_tensors="pt",
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
    teacher_out: dict[str, Any] = {}
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
            teacher_out[item.probe_id] = teacher
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
        if teacher_out:  # action-flow / fork figures read these back
            np.savez_compressed(
                out_dir / f"teacher_step{step:05d}.npz",
                **{pid: arr.astype(np.float16) for pid, arr in teacher_out.items()},
            )
    return point, bare_out


def load_bare(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path) as z:
        return {pid: z[pid].astype(np.float32) for pid in z.files}


# ---------------------------------------------------------------------- CLI


def main() -> int:
    import argparse

    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="freeze a probe set from rollout dumps")
    b.add_argument("--run-dir", required=True, type=Path)
    b.add_argument("--out", required=True, type=Path)
    b.add_argument("--n", type=int, default=64)
    b.add_argument("--turns", default="1,6",
                   help="comma-separated decision-point turn depths")

    s = sub.add_parser("sweep", help="four-condition sweep at one checkpoint")
    s.add_argument("--probe-set", required=True, type=Path)
    s.add_argument("--model", required=True, help="HF weights of pi_k")
    s.add_argument("--base-model", default=None,
                   help="HF weights of pi_0 (only needed until the base cache exists)")
    s.add_argument("--run-dir", required=True, type=Path,
                   help="run dir (research/ provides the current docs)")
    s.add_argument("--step", required=True, type=int)
    s.add_argument("--state-dir", required=True, type=Path,
                   help="bare/teacher npz cache + base cache + sweeps.jsonl")
    s.add_argument("--device", default="cuda")

    args = ap.parse_args()
    if args.cmd == "build":
        items = build_probe_set(
            args.run_dir, n_items=args.n,
            turns_at=tuple(int(t) for t in args.turns.split(",")),
        )
        save_probe_set(args.out, items)
        with_doc = sum(1 for i in items if i.doc_fix)
        print(f"froze {len(items)} probes ({with_doc} with a frozen doc) -> {args.out}")
        return 0

    probes = load_probe_set(args.probe_set)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    base_path = args.state_dir / "base_pi0.npz"
    if base_path.exists():
        base = load_bare(base_path)
    else:
        if not args.base_model:
            raise SystemExit("no base cache yet — pass --base-model for the first sweep")
        base_model = HFProbeModel(args.base_model, device=args.device)
        base = {p.probe_id: base_model.logprobs(messages_for(p, None)) for p in probes}
        np.savez_compressed(
            base_path, **{k: v.astype(np.float16) for k, v in base.items()}
        )
        del base_model
    prior = sorted(args.state_dir.glob("bare_step*.npz"))
    prev_bare = load_bare(prior[-1]) if prior else None

    model = HFProbeModel(args.model, device=args.device)
    point, _ = sweep(
        step=args.step, model=model, probes=probes,
        current_docs=current_docs_for(args.run_dir, probes, args.step),
        base_logprobs=base, prev_bare=prev_bare, out_dir=args.state_dir,
    )
    out = args.state_dir / "sweeps.jsonl"
    existing = [
        line for line in (out.read_text(encoding="utf-8").splitlines() if out.exists() else [])
        if line.strip() and json.loads(line).get("step") != args.step
    ]
    from dataclasses import asdict as _asdict

    out.write_text(
        "\n".join(existing + [json.dumps(_asdict(point))]) + "\n", encoding="utf-8"
    )
    print(json.dumps(_asdict(point)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
