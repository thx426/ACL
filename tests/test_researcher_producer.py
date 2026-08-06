"""在线 researcher 文档源：配置、去重/等待、降级与归档契约。"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from trace_opd.common.aio import shared_task
from trace_opd.common.config import load_config, objective_fingerprint
from trace_opd.research.session import SessionResult


BASE = """
run: {name: t, runs_root: runs, tape: false}
alfworld: {data_root: /d, worker_python: /p}
train:
  model_path: /m
  train_file: /t
  val_files: [/v]
  train_batch_size: 16
  agent_num_workers: 16
  agent_task_timeout_s: 5400
eval: {}
steering:
  enabled: true
  doc_source: researcher
  doc_table: ""
  theta_d: 0.8
research:
  enabled: true
  backend: external
  tool_protocol: fncall
  gt: "off"
  n_exp: 12
  k_default: 4
  k_max: 8
  max_turns: 30
  wrapup_buffer: 2
  fetch_raw_limit: 20
  fetch_raw_max_chars: 16000
  doc_max_words: 3000
  episode_temperature: 1.0
  researcher_max_tokens: 8192
  session_timeout_s: 3600
narration: {backend: engine, max_words: 1000, temperature: 0.3}
roles:
  researcher:
    endpoint: "http://127.0.0.1:18888/v1,http://127.0.0.1:18889/v1"
    model: qwen3.6-35b-a3b
    api_key_env: TEST_RESEARCHER_KEY
    mode: external_pool
    sidecar_sleep: false
    dp_size: 0
    max_concurrency: 64
    temperature: 0.7
    max_tokens: 8192
    timeout_s: 600
    extra_body: {chat_template_kwargs: {enable_thinking: false}}
"""


def _load(tmp_path, text=BASE):
    import os

    os.environ.setdefault("TEST_RESEARCHER_KEY", "test-key")
    path = tmp_path / "c.yaml"
    path.write_text(text)
    return load_config(path)


def test_dynamic_contract_and_secret_not_in_raw(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_RESEARCHER_KEY", "top-secret")
    cfg = _load(tmp_path)
    assert cfg.steering.doc_source == "researcher"
    assert cfg.role("researcher").api_key == "top-secret"
    assert cfg.role("researcher").api_key_env == "TEST_RESEARCHER_KEY"
    assert "top-secret" not in json.dumps(cfg.raw)
    assert cfg.role("researcher").extra_body["chat_template_kwargs"]["enable_thinking"] is False


@pytest.mark.parametrize(
    "old,new,match",
    [
        ("backend: external", "backend: engine", "backend"),
        ("tool_protocol: fncall", "tool_protocol: text", "tool_protocol"),
        ('gt: "off"', 'gt: "on"', "gt"),
        ("backend: engine, max_words", "backend: none, max_words", "narration"),
        ("mode: external_pool", "mode: local_sidecar", "mode"),
        ("sidecar_sleep: false", "sidecar_sleep: true", "sidecar_sleep"),
        ("dp_size: 0", "dp_size: 1", "dp_size"),
        ("enable_thinking: false", "enable_thinking: true", "thinking"),
        ("agent_task_timeout_s: 5400", "agent_task_timeout_s: 3600", "agent_task_timeout"),
        ("train_batch_size: 16", "train_batch_size: 8", "train_batch_size"),
    ],
)
def test_dynamic_unsafe_combinations_fail_loud(tmp_path, old, new, match):
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, BASE.replace(old, new, 1))


def test_static_default_compatibility_and_fingerprint(tmp_path):
    static = BASE.replace("doc_source: researcher", "doc_source: static", 1).replace(
        'doc_table: ""', "doc_table: /tmp/docs.jsonl", 1
    ).replace("enabled: true\n  backend: external", "enabled: false\n  backend: external", 1)
    explicit = _load(tmp_path, static)
    implicit = _load(tmp_path, static.replace("  doc_source: static\n", ""))
    assert objective_fingerprint(explicit) == objective_fingerprint(implicit)


def test_dynamic_fingerprint_scientific_not_deployment_fields(tmp_path):
    cfg = _load(tmp_path)
    base = objective_fingerprint(cfg)
    model = _load(tmp_path, BASE.replace("qwen3.6-35b-a3b", "another-model"))
    assert objective_fingerprint(model) != base
    protocol = _load(tmp_path, BASE.replace("n_exp: 12", "n_exp: 11"))
    assert objective_fingerprint(protocol) != base
    endpoint = _load(tmp_path, BASE.replace("127.0.0.1:18888", "127.0.0.1:28888"))
    assert objective_fingerprint(endpoint) == base
    concurrency = _load(tmp_path, BASE.replace("max_concurrency: 64", "max_concurrency: 32"))
    assert objective_fingerprint(concurrency) == base


def test_shared_task_eight_waiters_one_per_step():
    async def scenario():
        cache = {}
        calls = []

        async def producer(step):
            calls.append(step)
            await asyncio.sleep(0)
            return f"doc-{step}"

        first = await asyncio.gather(
            *[shared_task(cache, (1, "task"), lambda: producer(1)) for _ in range(8)]
        )
        second = await asyncio.gather(
            *[shared_task(cache, (2, "task"), lambda: producer(2)) for _ in range(8)]
        )
        assert first == ["doc-1"] * 8 and second == ["doc-2"] * 8
        assert calls == [1, 2]

    asyncio.run(scenario())


def _session(doc="# doc"):
    return SessionResult(
        finalized=True,
        no_finalize=False,
        turns_used=1,
        experiments_used=1,
        episodes_consumed=4,
        doc=doc,
        self_assessment="ok",
        flags=[],
        last_exp_successes=2,
        last_exp_attempts=4,
        messages=[{"role": "system", "content": "s"}],
    )


def test_policy_version_evidence_is_exact_and_frozen():
    from trace_opd.core.types import OUTCOME_ENV_ERROR, OUTCOME_FAILED, Trajectory
    from trace_opd.research.tools import EpisodeRecord
    from trace_opd.research.versioning import (
        assert_rollout_policy_version,
        attach_episode_policy_range,
        research_policy_version,
    )

    def record(eid, lo=None, hi=None, outcome=OUTCOME_FAILED):
        meta = {}
        if lo is not None:
            meta["min_global_steps"] = lo
        if hi is not None:
            meta["max_global_steps"] = hi
        traj = Trajectory(
            episode_id=eid,
            task_id="task",
            guidance_hash="bare",
            sample_idx=0,
            initial_obs="obs",
            outcome=outcome,
            meta=meta,
        )
        return EpisodeRecord(
            trajectory=traj,
            fact_card=SimpleNamespace(render=lambda: "card"),
            narration=None,
        )

    rec = record("attached")
    attach_episode_policy_range(
        rec.trajectory, {"min_global_steps": 7, "max_global_steps": 7}
    )
    assert rec.trajectory.meta == {
        "min_global_steps": 7,
        "max_global_steps": 7,
    }

    assert research_policy_version([record("a", 7, 7), record("b", 7, 7)]) == 7
    assert research_policy_version([record("env", outcome=OUTCOME_ENV_ERROR)]) is None
    with pytest.raises(RuntimeError, match="has no rollout policy version"):
        research_policy_version([record("bad")])
    with pytest.raises(RuntimeError, match="crossed policy versions 7..8"):
        research_policy_version([record("bad", 7, 8)])
    with pytest.raises(RuntimeError, match=r"session crossed policy versions \[7, 8\]"):
        research_policy_version([record("a", 7, 7), record("b", 8, 8)])

    student_extra = {"min_global_steps": 7, "max_global_steps": 7}
    assert_rollout_policy_version(student_extra, 7, label="student rollout")
    student_extra["max_global_steps"] = 8
    with pytest.raises(RuntimeError, match="student rollout crossed"):
        assert_rollout_policy_version(student_extra, 7, label="student rollout")
    student_extra = {"min_global_steps": 8, "max_global_steps": 8}
    with pytest.raises(RuntimeError, match="research/student rollout policy mismatch"):
        assert_rollout_policy_version(student_extra, 7, label="student rollout")


def test_archive_failure_degrades_document(monkeypatch, tmp_path):
    # agent_loop 生产依赖 verl；测试环境缺 tensordict 时仅跳过该集成点。
    pytest.importorskip("tensordict")
    from trace_opd.train.agent_loop import TraceAgentLoop

    loop = object.__new__(TraceAgentLoop)
    loop.cfg = _load(tmp_path)
    loop.delta = object()
    loop.rollout_config = SimpleNamespace(response_length=4096)
    loop.run_dir = tmp_path
    loop.researcher_client = SimpleNamespace(model="m")
    loop.executor_model_name = "student"
    loop.archive = SimpleNamespace(save=lambda **_: (_ for _ in ()).throw(OSError("disk")))
    loop._researcher_chat = lambda task_id: None
    loop._narrate_fn = lambda route: None
    loop.server_manager = object()
    loop.tokenizer = object()

    class Router:
        def __init__(self, **kwargs):
            self.state = SimpleNamespace(episodes={})

    class Session:
        def __init__(self, **kwargs):
            pass

        async def run(self):
            return _session()

    monkeypatch.setattr("trace_opd.train.agent_loop.ToolRouter", Router)
    monkeypatch.setattr("trace_opd.train.agent_loop.ResearchSession", Session)
    adapter = SimpleNamespace(
        task_id="task", task_brief=lambda: "task", gt_text=lambda: None
    )
    doc, fields = asyncio.run(loop._research(adapter, 3, "route"))
    assert doc == ""
    assert fields["research_error"] == 1
    assert fields["research_doc_missing"] == 1
    assert fields["research_archive"] == ""


def test_archive_slug_is_collision_safe_and_claim_is_atomic(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from trace_opd.docstore.archive import ResearchArchive

    archive = ResearchArchive(tmp_path)
    a = archive.save(task_id="a/b", result=None, records=[], global_step=1)
    b = archive.save(task_id="a__b", result=None, records=[], global_step=1)
    assert a.parent != b.parent
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(
            lambda _: archive.save(task_id="same/task", result=None, records=[], global_step=2),
            range(4),
        ))
    assert len({str(x) for x in paths}) == 4
    assert all(x.is_dir() for x in paths)


def test_archive_contains_task_step_and_error(tmp_path):
    from trace_opd.docstore.archive import ResearchArchive

    out = ResearchArchive(tmp_path).save(
        task_id="task", result=_session(), records=[], policy_version=6, global_step=7,
        extra_meta={"researcher_model": "m", "tool_protocol": "fncall", "error": None},
    )
    meta = json.loads((out / "meta.json").read_text())
    assert meta["task_id"] == "task" and meta["global_step"] == 7
    assert meta["policy_version"] == 6
    assert meta["researcher_model"] == "m" and meta["tool_protocol"] == "fncall"
    assert (out / "session.jsonl").exists() and (out / "doc.md").exists()


def test_rollout_dump_carries_policy_version_evidence():
    import ast
    from pathlib import Path

    source_path = Path(__file__).resolve().parents[1] / "src/trace_opd/train/agent_loop.py"
    source = source_path.read_text()
    tree = ast.parse(source)
    run = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef)
        and n.name == "run"
        and any(isinstance(x, ast.arg) and x.arg == "sampling_params" for x in n.args.args)
    )
    text = ast.get_source_segment(source, run)
    assert '"policy_version"' in text
    assert 'result.gen_extra.get("min_global_steps")' in text
    assert 'result.gen_extra.get("max_global_steps")' in text


def test_validation_path_skips_dynamic_research():
    import ast

    source_path = __import__("pathlib").Path(__file__).resolve().parents[1] / "src/trace_opd/train/agent_loop.py"
    source = source_path.read_text()
    tree = ast.parse(source)
    run = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "run"
        and any(isinstance(x, ast.arg) and x.arg == "sampling_params" for x in n.args.args)
    )
    text = ast.get_source_segment(source, run)
    assert "if self.cfg.steering.enabled and not validate" in text
    assert 'doc_source == "researcher"' in text


def test_null_step_guard_present_in_dynamic_run_path():
    import ast

    source_path = __import__("pathlib").Path(__file__).resolve().parents[1] / "src/trace_opd/train/agent_loop.py"
    tree = ast.parse(source_path.read_text())
    run = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "run"
        and any(isinstance(x, ast.arg) and x.arg == "sampling_params" for x in n.args.args)
    )
    text = ast.get_source_segment(source_path.read_text(), run)
    assert "global_step is None" in text
    assert "dynamic researcher requires a non-null global_step" in text


def test_static_fingerprint_material_ignores_source_schema(tmp_path):
    from trace_opd.common.config import objective_fingerprint
    docs = tmp_path / "docs.jsonl"
    docs.write_text("{}\n")
    static = BASE.replace("doc_source: researcher", "doc_source: static", 1)
    static = static.replace('doc_table: ""', f"doc_table: {docs}", 1)
    static = static.replace(
        "enabled: true\n  backend: external",
        "enabled: false\n  backend: external", 1,
    )
    cfg = _load(tmp_path, static)
    fp = objective_fingerprint(cfg)
    cfg.research.n_exp = 999
    cfg.roles["researcher"].model = "deployment-only-unused"
    assert objective_fingerprint(cfg) == fp
