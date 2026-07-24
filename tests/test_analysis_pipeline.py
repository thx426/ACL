"""Distillation-microscope data pipeline: episode turn spans, the two-sided
content hash, the offline collect join, and the probe-sweep math."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from fakes import FakeTokenizer, ScriptedAdapter, ScriptedGenerate, reply

from trace_opd.analysis import probe as probe_mod
from trace_opd.analysis.collect import collect
from trace_opd.analysis.probe import ProbeItem, load_bare, sweep
from trace_opd.analysis.schema import TurnProfile, read_jsonl
from trace_opd.common.hashing import response_token_hash
from trace_opd.core.episode import run_episode
from trace_opd.core.template_delta import TemplateDelta

TOK = FakeTokenizer()
DELTA = TemplateDelta(TOK)


# ------------------------------------------------------------ episode spans


@pytest.mark.asyncio
async def test_turn_spans_cover_exactly_mask_one():
    texts = [reply("look first", "look"), reply("now win", "do the thing")]
    result = await run_episode(
        adapter=ScriptedAdapter(), generate=ScriptedGenerate(TOK, texts),
        tokenizer=TOK, delta=DELTA, guidance="", episode_id="ep",
        response_budget=100000, sampling_params={},
    )
    spans = result.tokens.turn_spans
    mask = result.tokens.response_mask
    assert len(spans) == result.assistant_turns == 2
    in_span = [False] * len(mask)
    for s, e in spans:
        for t in range(s, e):
            in_span[t] = True
    assert [bool(m) for m in mask] == in_span  # spans == mask-1 positions


# ------------------------------------------------------------------ hashing


def test_hash_golden_and_vendored_verl_constants_match():
    assert response_token_hash([1, 2, 3]) == 1000008000018
    assert response_token_hash([]) == 0
    src = (
        Path(__file__).resolve().parents[1]
        / "verl" / "verl" / "trainer" / "distillation" / "losses.py"
    ).read_text(encoding="utf-8")
    assert "_MICROSCOPE_MUL = 1000003" in src
    assert "_MICROSCOPE_MOD = (1 << 63) - 1" in src
    assert "TRACE_DISTILL_MICROSCOPE_DIR" in src  # dump hook present in export


# ------------------------------------------------------------------ collect


def test_collect_joins_and_buckets_by_turn(tmp_path):
    ids = list(range(100, 110))
    spans = [[0, 4], [7, 10]]
    mask = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    kl = [0.1, 0.2, 0.3, 0.4, 9.0, 9.0, 9.0, 0.6, 0.6, 0.6]  # 9.0 = masked junk
    s_mass = [0.95, 0.5, 0.95, 0.5, 1, 1, 1, 0.5, 0.95, 0.95]
    run_dir = tmp_path / "run"
    step_dir = run_dir / "rollouts" / "step_00003"
    step_dir.mkdir(parents=True)
    (step_dir / "t_a.json").write_text(json.dumps({
        "task_id": "maze/room-1", "split": "train", "validate": False,
        "global_step": 3, "outcome": "FAILED", "won": False, "env_steps": 2,
        "trajectory": {"task_id": "maze/room-1", "steps": []},
        "response_token_len": 10, "response_token_hash": response_token_hash(ids),
        "turn_token_spans": spans,
    }), encoding="utf-8")
    micro = run_dir / "microscope"
    micro.mkdir(parents=True)
    (micro / "rank0.jsonl").write_text(json.dumps({
        "resp_hash": response_token_hash(ids), "resp_len": 10, "ids": ids,
        "mask": mask, "kl": kl, "s_mass": s_mass, "t_mass": [1.0] * 10,
        "overlap": [60] * 10, "t_ent": [0.5] * 10,
    }) + "\n" + json.dumps({  # orphan row: no dump carries this hash
        "resp_hash": 424242, "resp_len": 3, "ids": [1, 2, 3],
        "mask": [1, 1, 1], "kl": [1, 1, 1], "s_mass": [1, 1, 1],
        "t_mass": [1, 1, 1], "overlap": [1, 1, 1], "t_ent": [1, 1, 1],
    }) + "\n", encoding="utf-8")

    stats = collect(run_dir, micro, tmp_path / "out")
    assert stats == {
        "joined": 1, "orphans": 1, "mask_mismatch": 0,
        "turn_rows": 2, "token_rows": 7,
    }
    turns = read_jsonl(tmp_path / "out" / "turn_profiles.jsonl", TurnProfile)
    t1, t2 = sorted(turns, key=lambda r: r.turn)
    assert (t1.turn, t1.n_tokens) and t1.n_tokens == 4
    assert t1.g_mean == pytest.approx(0.25)     # masked junk excluded
    assert t1.g_gated == pytest.approx(0.3)     # only s_mass<0.9 positions
    assert t2.n_tokens == 3 and t2.g_mean == pytest.approx(0.6)
    bands = json.loads((tmp_path / "out" / "position_bands.json").read_text())
    assert bands[0]["step"] == 3 and bands[0]["n"] == 7


# ------------------------------------------------------------------- probe


class _FakeModel:
    """system prompt decides the condition: guidance header => teacher."""

    def __init__(self):
        self.bare = np.log(np.array([0.1, 0.2, 0.3, 0.4]))
        self.teacher = np.log(np.array([0.05, 0.1, 0.25, 0.6]))

    def logprobs(self, messages):
        return self.teacher if "Task guidance" in messages[0]["content"] else self.bare


def _kl(p_log, q_log):
    p = np.exp(p_log)
    return float(np.sum(p * (p_log - q_log)))


def test_probe_sweep_math(tmp_path):
    model = _FakeModel()
    probes = [
        ProbeItem(probe_id=f"p{i}", task_id="maze/room-1", scaffold="You act.",
                  dialogue=[{"role": "user", "content": "obs"}], doc_fix="frozen doc")
        for i in range(2)
    ]
    base = {p.probe_id: np.log(np.full(4, 0.25)) for p in probes}
    point, bare = sweep(
        step=20, model=model, probes=probes,
        current_docs={"maze/room-1": "go right"},
        base_logprobs=base, prev_bare=base, out_dir=tmp_path, seed=0,
    )
    assert point.d_init == pytest.approx(_kl(model.bare, base["p0"]), abs=1e-6)
    assert point.d_gain == pytest.approx(_kl(model.teacher, model.bare), abs=1e-6)
    assert point.d_frozen == pytest.approx(point.d_gain, abs=1e-6)  # same fake doc
    assert point.align > 0 and 0 < point.align_robust <= 1
    assert "d_init" in point.ci and point.ci["d_init"][0] <= point.d_init
    # bare distributions persisted for the next sweep's delta_train
    reloaded = load_bare(tmp_path / "bare_step00020.npz")
    assert np.allclose(reloaded["p0"], bare["p0"], atol=1e-3)


def test_probe_messages_inject_doc_via_protocol():
    item = ProbeItem(probe_id="p", task_id="t", scaffold="SYS",
                     dialogue=[{"role": "user", "content": "obs"}], doc_fix="")
    bare_msgs = probe_mod.messages_for(item, None)
    doc_msgs = probe_mod.messages_for(item, "the doc body")
    assert bare_msgs[0]["content"] == "SYS"
    assert "Task guidance" in doc_msgs[0]["content"]
    assert "the doc body" in doc_msgs[0]["content"]
    assert bare_msgs[1:] == doc_msgs[1:] == item.dialogue
