"""Researcher-SFT sampling pipeline: the protocol-conversion round-trip
contract, stratified sampling determinism, export filtering, and the config
section. The conversion tests are the load-bearing ones — they pin "the SFT
distribution matches the self-evolve wire format" as a checked property."""
import json
from pathlib import Path

import pytest

from trace_opd.common.config import load_config
from trace_opd.research.prompts import TOOL_SCHEMAS
from trace_opd.research.toolproto import TextProtocol
from trace_opd.common.llm import ChatResult
from trace_opd.sftgen.convert import fncall_messages_to_text, messages_to_text_protocol
from trace_opd.sftgen.strata import alfworld_stratum, stratified_sample

_ROOT = Path(__file__).resolve().parents[1]


def _fncall_session():
    return [
        {"role": "system", "content": "You are the researcher."},
        {"role": "user", "content": "Begin. Budget: 4 experiments."},
        {
            "role": "assistant",
            "content": "The task needs heating; test the microwave first.",
            "tool_calls": [
                {
                    "id": "call_0",
                    "function": {
                        "name": "run_experiment",
                        "arguments": json.dumps({"guidance": "heat it", "k": 4}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "2/4 succeeded. ..."},
        {
            "role": "assistant",
            "content": "Good enough; finalize.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "finalize",
                        "arguments": json.dumps({"document": "## Doc\ndo X"}),
                    },
                }
            ],
        },
    ]


# ---------------------------------------------------------------- conversion

def test_fncall_conversion_round_trips_through_the_text_protocol():
    """Every converted assistant turn must re-parse into the SAME calls the
    sampler made — otherwise SFT teaches a wire format the runtime cannot
    execute."""
    proto = TextProtocol()
    converted = fncall_messages_to_text(_fncall_session(), TOOL_SCHEMAS)
    assistants = [m for m in converted if m["role"] == "assistant"]
    calls_0 = proto.extract(ChatResult(content=assistants[0]["content"]))
    assert len(calls_0) == 1
    assert calls_0[0].name == "run_experiment"
    assert calls_0[0].arguments == {"guidance": "heat it", "k": 4}
    calls_1 = proto.extract(ChatResult(content=assistants[1]["content"]))
    assert calls_1[0].name == "finalize"
    assert calls_1[0].arguments == {"document": "## Doc\ndo X"}
    # analysis text survives, before the call block
    assert assistants[0]["content"].startswith("The task needs heating")


def test_conversion_rebuilds_the_exact_runtime_system_prompt_and_tool_turns():
    proto = TextProtocol()
    converted = fncall_messages_to_text(_fncall_session(), TOOL_SCHEMAS)
    want_system, api_tools = proto.prepare("You are the researcher.", TOOL_SCHEMAS)
    assert api_tools is None
    assert converted[0] == {"role": "system", "content": want_system}
    # the tool result becomes exactly the user turn the runtime would emit
    tool_turns = [
        m for m in converted
        if m["role"] == "user" and m.get("content", "").startswith("<tool_response>")
    ]
    assert len(tool_turns) == 1
    assert tool_turns[0]["role"] == "user"
    assert tool_turns[0]["content"] == "<tool_response>\n2/4 succeeded. ...\n</tool_response>"


def test_text_sessions_pass_through_unchanged():
    messages = [{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}]
    assert messages_to_text_protocol(messages, TOOL_SCHEMAS, "text") == messages


def test_conversion_fails_loud_on_corrupt_arguments():
    bad = [
        {"role": "system", "content": "s"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": "not json"}}],
        },
    ]
    with pytest.raises(json.JSONDecodeError):
        fncall_messages_to_text(bad, TOOL_SCHEMAS)


# ---------------------------------------------------------------- sampling

def _items():
    types = ["pick_and_place_simple", "pick_heat_then_place_in_recep", "look_at_obj_in_light"]
    return [
        {"task_id": f"{t}-Obj-None-Recep-{i}/trial_{i}"}
        for t in types
        for i in range(10)
    ]


def test_stratified_sample_is_deterministic_and_balanced():
    items = _items()
    a = stratified_sample(items, 9, seed=7, stratum_fn=alfworld_stratum)
    b = stratified_sample(items, 9, seed=7, stratum_fn=alfworld_stratum)
    assert [x["task_id"] for x in a] == [x["task_id"] for x in b]
    counts = {}
    for item in a:
        counts[alfworld_stratum(item)] = counts.get(alfworld_stratum(item), 0) + 1
    assert set(counts.values()) == {3}  # 9 across 3 strata -> 3 each
    c = stratified_sample(items, 9, seed=8, stratum_fn=alfworld_stratum)
    assert [x["task_id"] for x in a] != [x["task_id"] for x in c]


def test_oversampling_returns_the_whole_population():
    items = _items()
    assert len(stratified_sample(items, 10_000, seed=1, stratum_fn=alfworld_stratum)) == len(items)


# ------------------------------------------------------------------ export

def _write_archive(root: Path, task: str, *, finalized: bool, gain: float,
                   error=None, multi_call=False):
    d = root / "tasks" / task / "s0"
    d.mkdir(parents=True)
    calls = [
        {"id": "c0", "function": {"name": "finalize", "arguments": json.dumps({"document": "d"})}}
    ]
    if multi_call:
        calls.append({"id": "c1", "function": {"name": "fetch_raw", "arguments": "{}"}})
    session = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "a", "tool_calls": calls},
        ],
        "tool_protocol": "fncall",
    }
    (d / "session.json").write_text(json.dumps(session), encoding="utf-8")
    (d / "meta.json").write_text(json.dumps({
        "task_id": task, "finalized": finalized, "error": error, "gain": gain,
        "p_doc": max(gain, 0.0), "doc_words": 10, "researcher_model": "m",
    }), encoding="utf-8")
    (d / "DONE").write_text("ok", encoding="utf-8")


def test_export_filters_and_writes_the_messages_schema(tmp_path):
    pd = pytest.importorskip("pandas")
    archive = tmp_path / "arch"
    _write_archive(archive, "keep", finalized=True, gain=0.5)
    _write_archive(archive, "nofinal", finalized=False, gain=0.0)
    _write_archive(archive, "err", finalized=True, gain=0.5, error="boom")
    _write_archive(archive, "lowgain", finalized=True, gain=-0.25)
    _write_archive(archive, "multi", finalized=True, gain=0.5, multi_call=True)

    from trace_opd.sftgen.export import export

    out = tmp_path / "sft.parquet"
    manifest = export(archive, out, min_gain=0.0, drop_multi_call=True)
    assert manifest["kept"] == 1
    assert manifest["dropped_not_finalized"] == 1
    assert manifest["dropped_error"] == 1
    assert manifest["dropped_gain"] == 1
    assert manifest["dropped_multi_call"] == 1
    df = pd.read_parquet(out)
    assert list(df["task_id"]) == ["keep"]
    messages = list(df.iloc[0]["messages"])
    assert messages[0]["role"] == "system"
    assert "<tool_call>" in messages[1]["content"]  # converted to text protocol


# ------------------------------------------------------------------ config

def test_sft_gen_section_loads_and_stays_off_the_training_path(tmp_path):
    cfg = load_config(_ROOT / "configs" / "trace.yaml")
    assert cfg.sft_gen.benchmark == "alfworld"
    assert cfg.sft_gen.eval_k == 8
    # the section must never gate training: defaults are always legal
    from trace_opd.common.config import objective_fingerprint
    base = objective_fingerprint(cfg)
    cfg.sft_gen.tasks = 7
    assert objective_fingerprint(cfg) == base  # sampling knobs never enter the hash
