"""Evidence, isolation and fail-closed replacement contracts (no live model)."""
import asyncio
import json

import pytest
from fakes import FakeTokenizer, ScriptedAdapter, reply
from test_research import tc

from trace_opd.common.config import (
    NarrationCfg,
    ResearchCfg,
    load_config,
    objective_fingerprint,
)
from trace_opd.common.evolution_config import EvolutionCfg
from trace_opd.core.episode import GenResult
from trace_opd.core.template_delta import TemplateDelta
from trace_opd.core.types import Step, Trajectory
from trace_opd.docstore.archive import ResearchArchive
from trace_opd.research.errors import digest, error_record
from trace_opd.research.prompts import tool_schemas_for
from trace_opd.research.session import ResearchSession
from trace_opd.research.toolproto import FncallProtocol
from trace_opd.research.tools import ToolRouter

TOK = FakeTokenizer()
DELTA = TemplateDelta(TOK)


class ExactAdapter(ScriptedAdapter):
    supports_exact_replay = True


class HoldoutAdapter(ExactAdapter):
    def executor_scaffold(self):
        return super().executor_scaffold() + " HOLDOUT_TASK"


class GuidedGenerate:
    def __init__(self, mode="improve", calls=None):
        self.n = 0
        self.mode = mode
        self.calls = calls if calls is not None else []

    async def __call__(self, ids, params):
        text = TOK.decode(ids)
        self.calls.append({"text": text, "params": dict(params)})
        self.n += 1
        revised = "# Error-specific rules" in text
        holdout = "HOLDOUT_TASK" in text
        if holdout:
            action = "bad" if revised and self.mode == "regress_holdout" else "do the thing"
        elif "<action>look</action>" not in text:
            action = "look"
        elif revised and self.mode != "no_gain":
            action = "do the thing" if self.mode != "local_only" or self.n == 1 else "bad"
        else:
            action = "bad"
        output = reply("check then act", action) if self.n <= 2 else ""
        version = 8 if self.mode == "version_drift" and revised else 7
        extra = {"min_global_steps": version, "max_global_steps": version}
        if self.mode == "missing_version":
            extra = {}
        return GenResult(token_ids=TOK.encode(output), text=output, extra=extra)


def make_router(tmp_path, *, mode="improve", exact=True, holdout=True, **overrides):
    manifest = tmp_path / "holdout.jsonl"
    manifest.write_text(json.dumps({"task_id": "reserved/task"}) + "\n")
    cfg = EvolutionCfg(enabled=True, samples=2, holdout_tasks=str(manifest) if holdout else "",
                       **overrides)
    calls = []
    router = ToolRouter(
        adapter=(ExactAdapter if exact else ScriptedAdapter)(),
        research=ResearchCfg(evolution=cfg, n_exp=10), narration=NarrationCfg(),
        generate_factory=lambda: GuidedGenerate(mode, calls), tokenizer=TOK, delta=DELTA,
        episode_response_budget=10000, narrate_fn=None,
        adapter_factory=lambda row: HoldoutAdapter(task_id=row["task_id"]),
    )
    return router, calls


def proposal(error_ids):
    return {"error_type": "illegal_action", "rules": [{
        "rule_id": "check_action", "sources": error_ids,
        "trigger": "Before the next action.", "checks": "Read available actions.",
        "action": "Choose an available action that meets the task requirement.",
        "completion": "Observe the task success feedback.",
        "recovery": "After a rejection, inspect the prerequisites before retrying.",
    }]}


async def prepare(router):
    result = json.loads(await router.dispatch(tc("run_experiment", {})))
    error = result["errors"][0]
    assert error["status"] == "verified" and error["step"] == 2
    candidate = json.loads(await router.dispatch(tc("propose_revision", proposal([error["error_id"]]))))
    return error, candidate


@pytest.mark.asyncio
async def test_full_pipeline_promotes_only_after_independent_validation(tmp_path):
    router, calls = make_router(tmp_path, discovery_prompts=["DISCOVERY_A", "DISCOVERY_B"])
    error, candidate = await prepare(router)
    assert router.evolution.active_skill == "" and router.state.doc is None
    source = router.evolution._sources[error["episode_id"]].trajectory
    assert error["trajectory_sha256"] == digest(source.to_dict())
    assert error["observation_before"] == "An empty room."
    decision = json.loads(await router.dispatch(tc("validate_revision", {"candidate_id": candidate["candidate_id"]})))
    assert decision["status"] == "accepted", decision
    assert decision["local"][0]["new"]["correct_action_rate"] == 1
    assert decision["full"]["new"]["success_rate"] == 1
    assert len(decision["holdout"]) == 1
    assert router.evolution.active_skill == candidate["skill"]
    trajectories = router.evolution.trajectories
    discovery = [r for r in trajectories if r["meta"]["skill_evolution"]["phase"] == "discovery"]
    assert len(discovery) == 4
    discovery_seeds = {r["sample_idx"] for r in discovery}
    validation_seeds = {r["sample_idx"] for r in trajectories if r not in discovery}
    assert not discovery_seeds & validation_seeds
    formal_calls = [c for c in calls if c["params"]["seed"] in validation_seeds]
    assert all("DISCOVERY_" not in c["text"] for c in formal_calls)
    for phase in ("local", "full", "holdout"):
        arms = [[r for r in trajectories if r["meta"]["skill_evolution"]["phase"] == f"{phase}:{arm}"]
                for arm in ("old", "new")]
        assert [r["sample_idx"] for r in arms[0]] == [r["sample_idx"] for r in arms[1]]
    assert all(env.closed for env in router.adapter.envs)
    # Finalization cannot smuggle in an untested document, even on a second call.
    for _ in range(2):
        out = await router.dispatch(tc("finalize", {"experience_doc": "unvalidated replacement"}))
        assert "cannot replace" in out and not router.state.finalized
    await router.dispatch(tc("finalize", {"self_assessment": "tested"}))
    assert router.state.doc == candidate["skill"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,reason", [
    ("no_gain", "whole_task_success_gain_below_threshold"),
    ("local_only", "whole_task_success_gain_below_threshold"),
    ("regress_holdout", "holdout_regression:reserved/task"),
    ("version_drift", "executor_policy_not_frozen"),
])
async def test_failed_gates_preserve_old_skill(tmp_path, mode, reason):
    router, _ = make_router(tmp_path, mode=mode)
    _, candidate = await prepare(router)
    report = json.loads(await router.dispatch(tc("validate_revision", {"candidate_id": candidate["candidate_id"]})))
    assert report["status"] == "rejected" and reason in report["reasons"]
    assert router.evolution.active_skill == ""
    if mode == "local_only":
        assert report["local"][0]["new"]["correct_action_rate"] == 1


@pytest.mark.asyncio
async def test_uncertain_claim_not_accepted_as_first_error(tmp_path):
    router, _ = make_router(tmp_path)
    result = json.loads(await router.dispatch(tc("run_experiment", {})))
    error = result["errors"][0]
    rec = json.loads(await router.dispatch(tc("record_error", {
        "episode_id": error["episode_id"], "step": 1, "error_type": "wrong_choice",
        "evidence": "I think looking was wrong",
    })))
    assert rec["status"] == "uncertain" and rec["first_supported_step"] == 2
    out = await router.dispatch(tc("propose_revision", proposal([rec["error_id"]])))
    assert "sources must be verified" in out
    assert not router.evolution.candidates


def test_failed_terminal_step_is_not_automatically_first_error():
    adapter = ScriptedAdapter()
    trajectory = Trajectory("e", adapter.task_id, "bare", 0, adapter.initial_obs,
                            steps=[Step(1, "", "look", "look", "An empty room.", True)])
    record = error_record(adapter, trajectory, error_id="e1")
    assert record["status"] == "uncertain" and record["step"] is None


def test_repetition_without_state_evidence_stays_uncertain():
    adapter = ScriptedAdapter()
    trajectory = Trajectory("e", adapter.task_id, "bare", 0, adapter.initial_obs,
                            steps=[Step(t, "", "poll", "poll", "pending", True) for t in (1, 2, 3)])
    record = error_record(adapter, trajectory, error_id="e1")
    assert record["status"] == "uncertain" and record["error_type"] == "repeated_loop"
    assert record["step"] == 3


def test_semantic_diagnosis_requires_adapter_evidence_and_sees_no_future():
    class CheckedAdapter(ScriptedAdapter):
        def diagnose_error(self, trajectory, index):
            assert len(trajectory.steps) == index + 1
            return {"error_type": "missing_precondition", "reason": "Observed door remains closed"}
    adapter = CheckedAdapter()
    trajectory = Trajectory("e", adapter.task_id, "bare", 0, "door is closed",
                            steps=[Step(1, "", "enter", "enter", "blocked", False),
                                   Step(2, "", "leave", "leave", "failed", False)])
    record = error_record(adapter, trajectory, error_id="e1")
    assert record["status"] == "verified" and record["error_type"] == "missing_precondition"
    assert record["step"] == 1


@pytest.mark.asyncio
async def test_replay_drift_invalidates_local_comparison(tmp_path):
    router, _ = make_router(tmp_path)
    _, candidate = await prepare(router)
    router.adapter._responses["look"] = "The room changed."
    report = json.loads(await router.dispatch(tc("validate_revision", {"candidate_id": candidate["candidate_id"]})))
    assert report["status"] == "rejected"
    assert "incomplete_local_pairs_or_replay_drift" in report["reasons"]
    assert report["full"] is not None


@pytest.mark.asyncio
async def test_unsupported_replay_is_explicit_and_still_requires_full_runs(tmp_path):
    router, _ = make_router(tmp_path, exact=False)
    _, candidate = await prepare(router)
    report = json.loads(await router.dispatch(tc("validate_revision", {"candidate_id": candidate["candidate_id"]})))
    assert report["local"][0]["status"] == "unsupported"
    assert report["full"]["new"]["attempts"] == 2 and report["status"] == "accepted"


@pytest.mark.asyncio
async def test_missing_holdout_fails_closed(tmp_path):
    router, _ = make_router(tmp_path, holdout=False)
    _, candidate = await prepare(router)
    report = json.loads(await router.dispatch(tc("validate_revision", {"candidate_id": candidate["candidate_id"]})))
    assert report["status"] == "rejected"
    assert "held-out" in report["reasons"][0]


@pytest.mark.asyncio
async def test_one_error_type_per_round_and_immutable_candidate(tmp_path):
    router, _ = make_router(tmp_path)
    error, candidate = await prepare(router)
    assert "one candidate" in await router.dispatch(tc("propose_revision", proposal([error["error_id"]])))
    assert "pending candidate" in await router.dispatch(tc("run_experiment", {}))
    report = json.loads(await router.dispatch(tc("validate_revision", {"candidate_id": candidate["candidate_id"]})))
    assert report["status"] == "accepted"
    assert "pending" in await router.dispatch(tc("validate_revision", {"candidate_id": candidate["candidate_id"]}))
    stale = await router.dispatch(tc("propose_revision", proposal([error["error_id"]])))
    assert "current Skill" in stale


@pytest.mark.asyncio
async def test_interrupted_validation_preserves_audit_and_previous_skill(tmp_path, monkeypatch):
    router, _ = make_router(tmp_path)
    _, candidate = await prepare(router)
    started = asyncio.Event()

    async def blocked(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(router.evolution, "_pairs", blocked)
    pending = asyncio.create_task(router.evolution.validate({"candidate_id": candidate["candidate_id"]}))
    await started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert router.evolution.active_skill == ""
    assert router.evolution.validations[0]["status"] == "interrupted"


@pytest.mark.asyncio
async def test_session_snapshot_and_archive_include_evidence(tmp_path):
    router, _ = make_router(tmp_path)
    _, candidate = await prepare(router)
    await router.evolution.validate({"candidate_id": candidate["candidate_id"]})
    await router.dispatch(tc("finalize", {}))
    session = ResearchSession(researcher_chat=None, toolproto=FncallProtocol(), router=router,
                              system_prompt="Research", max_turns=10)
    snapshot = session.snapshot()
    assert "INDEPENDENT VALIDATION" in session.messages[0]["content"]
    out = ResearchArchive(tmp_path / "archive").save(task_id=router.adapter.task_id,
        result=snapshot, records=list(router.state.episodes.values()))
    audit = json.loads((out / "skill_evolution.json").read_text())
    assert audit["errors"] and audit["validations"][0]["status"] == "accepted"
    assert len(audit["trajectories"]) == len(router.state.episodes)
    assert (out / "errors.jsonl").exists() and (out / "skill_validations.jsonl").exists()


@pytest.mark.asyncio
async def test_researcher_session_runs_all_stages_with_existing_protocol(tmp_path):
    from test_research import ScriptedChat, assistant_with_call

    router, _ = make_router(tmp_path)
    chat = ScriptedChat([
        assistant_with_call("Observe the frozen executor.", tc("run_experiment", {})),
        assistant_with_call("Fix the first rejected action.", tc("propose_revision", proposal(["err0001"]))),
        assistant_with_call("Test independently.", tc("validate_revision", {"candidate_id": "candidate001"})),
        assistant_with_call("Release the accepted Skill.", tc("finalize", {})),
    ])
    session = ResearchSession(researcher_chat=chat, toolproto=FncallProtocol(), router=router,
                              system_prompt="Research the task.", max_turns=10)
    result = await session.run()
    assert result.finalized and result.error is None and result.turns_used == 4
    assert result.doc and result.evolution_audit["validations"][0]["status"] == "accepted"
    assert result.experiments_used == 2


@pytest.mark.asyncio
async def test_resume_inherits_remaining_environment_and_token_budget(tmp_path, monkeypatch):
    from fakes import ScriptedGenerate

    from trace_opd.core import protocol
    from trace_opd.core.episode import run_episode

    router, _ = make_router(tmp_path)
    error, _ = await prepare(router)
    source = router.evolution._sources[error["episode_id"]]
    monkeypatch.setattr(protocol, "MAX_ENV_STEPS", 2)
    gen = ScriptedGenerate(TOK, [reply("another look", "look"), reply("win", "do the thing")])
    budget = 10000
    result = await run_episode(adapter=router.adapter, generate=gen, tokenizer=TOK, delta=DELTA,
        guidance="revised", response_budget=budget, sampling_params={}, resume=(source, 2))
    assert not result.won
    assert len(gen.calls) == 1  # replayed look consumes the first of two environment steps
    assert router.adapter.envs[-1].commands == ["look", "look"]
    assert result.trajectory.meta["resume"]["env_steps_used"] == 1
    assert result.tokens.response_ids[:source.tokens.turn_spans[1][0]] == source.tokens.response_ids[:source.tokens.turn_spans[1][0]]
    assert len(result.tokens.response_ids) <= budget


@pytest.mark.asyncio
async def test_changed_error_evidence_and_sealed_candidate_are_rejected(tmp_path):
    router, _ = make_router(tmp_path)
    error, candidate = await prepare(router)
    router.evolution._sources[error["episode_id"]].trajectory.steps[0].observation = "tampered"
    report = json.loads(await router.evolution.validate({"candidate_id": candidate["candidate_id"]}))
    assert report["status"] == "rejected" and "source evidence changed" in report["reasons"][0]
    assert router.evolution.active_skill == ""


@pytest.mark.asyncio
async def test_other_error_type_rules_cannot_be_rewritten(tmp_path):
    router, _ = make_router(tmp_path)
    _, candidate = await prepare(router)
    report = json.loads(await router.evolution.validate({"candidate_id": candidate["candidate_id"]}))
    assert report["status"] == "accepted"
    accepted = router.evolution.active_skill
    router.adapter.diagnose_error = lambda trajectory, index: (
        {"error_type": "wrong_choice", "reason": "Task-specific choice check"} if index == 0 else None)
    discovery = json.loads(await router.dispatch(tc("run_experiment", {})))
    proposed = proposal([discovery["errors"][0]["error_id"]])
    proposed["error_type"] = "wrong_choice"
    out = await router.dispatch(tc("propose_revision", proposed))
    assert "different error type" in out and router.evolution.active_skill == accepted


@pytest.mark.asyncio
async def test_rejection_can_finalize_existing_nonempty_skill(tmp_path):
    router, _ = make_router(tmp_path, mode="no_gain", initial_skill="Keep the existing procedure.")
    _, candidate = await prepare(router)
    await router.evolution.validate({"candidate_id": candidate["candidate_id"]})
    await router.dispatch(tc("finalize", {}))
    assert router.state.doc == "Keep the existing procedure."


@pytest.mark.asyncio
async def test_finalize_abandons_unvalidated_proposal_without_replacement(tmp_path):
    router, _ = make_router(tmp_path, initial_skill="Existing Skill")
    await prepare(router)
    await router.dispatch(tc("finalize", {}))
    assert router.state.doc == "Existing Skill"
    assert router.evolution.candidates[0]["status"] == "abandoned"
    assert not router.evolution.validations


@pytest.mark.asyncio
async def test_empty_generation_keeps_observed_policy_version(tmp_path):
    from trace_opd.research.versioning import research_policy_version

    router, _ = make_router(tmp_path)

    async def empty(ids, params):
        return GenResult([], "", {"min_global_steps": 7, "max_global_steps": 7})

    router._generate_factory = lambda: empty
    await router.dispatch(tc("run_experiment", {}))
    assert research_policy_version(router.state.episodes.values()) == 7


def test_holdout_is_required_by_production_config(tmp_path, monkeypatch):
    from test_researcher_producer import BASE

    monkeypatch.setenv("TEST_RESEARCHER_KEY", "test-only-placeholder")
    path = tmp_path / "c.yaml"
    path.write_text(BASE)
    with pytest.raises(ValueError, match="holdout_tasks is required"):
        load_config(path, ["research.evolution.enabled=true"])
    holdout = tmp_path / "holdout.jsonl"
    holdout.write_text('{"task_id":"a"}\n')
    cfg = load_config(path, ["research.evolution.enabled=true",
                             f"research.evolution.holdout_tasks={holdout}"])
    first = objective_fingerprint(cfg)
    holdout.write_text('{"task_id":"b"}\n')
    assert objective_fingerprint(cfg) != first


def test_config_and_tools_preserve_legacy_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_RESEARCHER_KEY", "test-only-placeholder")
    from test_researcher_producer import BASE
    path = tmp_path / "cfg.yaml"
    path.write_text(BASE)
    legacy = load_config(path)
    before = objective_fingerprint(legacy)
    assert not legacy.research.evolution.enabled
    assert len(tool_schemas_for(legacy.research)) == 3
    enable = ["research.evolution.enabled=true", "research.evolution.require_holdout=false"]
    enabled = load_config(path, enable + ["research.evolution.samples=4"])
    assert len(tool_schemas_for(enabled.research)) == 6
    assert objective_fingerprint(enabled) != before
    enabled.research.evolution.min_success_gain = 0.2
    assert objective_fingerprint(enabled) != objective_fingerprint(
        load_config(path, enable))
    assert objective_fingerprint(legacy) == before


@pytest.mark.parametrize("kwargs", [
    {"samples": 0}, {"samples": True}, {"min_success_gain": 0},
    {"min_error_reduction": float("nan")}, {"max_holdout_regression": -1},
    {"discovery_prompts": ["a", "b", "c"]}, {"require_holdout": "false"},
])
def test_invalid_evolution_config_rejected(kwargs):
    with pytest.raises(ValueError):
        EvolutionCfg(**kwargs)
