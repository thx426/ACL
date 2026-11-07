"""Researcher-driven proposals with program-owned evidence and replacement gates.

The researcher still explores, reads raw logs and writes rules. It cannot
declare its own candidate verified, select validation seeds, or edit a sealed
candidate. The existing session/tool protocol remains the orchestration layer.
"""
from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import asdict
from pathlib import Path

from ..core.episode import run_episode
from ..core.types import OUTCOME_ENV_ERROR, Trajectory, guidance_hash
from .errors import ERROR_TYPES, digest, error_record, has_error, step_error
from .versioning import (
    _version_int,
    assert_rollout_policy_version,
    attach_episode_policy_range,
)

RULE_FIELDS = ("trigger", "checks", "action", "completion", "recovery")


class _PromptAdapter:
    def __init__(self, adapter, scaffold):
        self.adapter, self.scaffold = adapter, scaffold

    def __getattr__(self, key):
        return getattr(self.adapter, key)

    def executor_scaffold(self):
        return self.scaffold


def render_skill(base, rules):
    if not rules:
        return base.strip()
    blocks = [base.strip(), "# Error-specific rules"]
    for key, rule in sorted(rules.items()):
        blocks.append(f"## {key}")
        blocks.extend(f"{name.capitalize()}: {rule[name]}" for name in RULE_FIELDS)
    return "\n\n".join(x for x in blocks if x)


class SkillEvolution:
    def __init__(self, router, *, adapter_factory=None):
        self.router = router
        self.cfg = router.cfg.evolution
        self.adapter_factory = adapter_factory
        self.active_skill = self.cfg.initial_skill.strip()
        self.rules = {}
        self.errors = []
        self.candidates = []
        self.validations = []
        self.trajectories = []
        self.pending = None
        self._sources = {}
        self._seed = self.cfg.seed
        self._policy_version = None
        self._policy_faults = []
        self._scaffold = router.adapter.executor_scaffold()
        self._temperature = router.cfg.episode_temperature
        self._scaffolds = {router.adapter.task_id: self._scaffold}
        # Freeze the entire holdout manifest before the first proposal.
        self._holdout_bytes = (Path(self.cfg.holdout_tasks).read_bytes()
                               if self.cfg.holdout_tasks else b"")
        self._holdout_rows = [json.loads(line) for line in self._holdout_bytes.splitlines()
                              if line.strip()]
        self._settings = asdict(self.cfg)
        if len(self.active_skill.split()) > router.cfg.doc_max_words:
            raise ValueError("initial Skill exceeds doc_max_words")

    def snapshot(self):
        return copy.deepcopy({
            "schema_version": 1, "settings": self._settings,
            "active_skill": self.active_skill, "active_skill_hash": guidance_hash(self.active_skill),
            "policy_version": self._policy_version, "policy_faults": self._policy_faults,
            "execution_contract": {"temperature": self._temperature,
                                   "response_budget": self.router._episode_budget,
                                   "scaffold_sha256": digest(self._scaffold)},
            "holdout_manifest_sha256": digest(self._holdout_rows),
            "errors": self.errors, "candidates": self.candidates,
            "validations": self.validations, "trajectories": self.trajectories,
        })

    def _seeds(self):
        if self._seed + self.cfg.samples >= 2**31:
            raise ValueError("validation seed space exhausted")
        values = list(range(self._seed, self._seed + self.cfg.samples))
        self._seed += self.cfg.samples
        return values

    def _budget(self):
        if self.router.state.finalized:
            raise ValueError("session already finalized")
        if self.router.state.experiments_used >= self.router.cfg.n_exp:
            raise ValueError("experiment budget exhausted; finalize the current accepted Skill")

    async def _run(self, adapter, skill, prompt, seed, phase, *, resume=None):
        router, st = self.router, self.router.state
        eid = st.next_episode_id()
        if adapter.task_id not in self._scaffolds:
            self._scaffolds[adapter.task_id] = adapter.executor_scaffold()
        base = self._scaffolds[adapter.task_id]
        scaffold = base + ("\n\n" + prompt if prompt else "")
        generate = router._generate_factory()
        observed_version = None

        async def frozen_generate(ids, params):
            nonlocal observed_version
            result = await generate(ids, params)
            try:
                lo = _version_int(result.extra.get("min_global_steps"))
                if lo is None:
                    raise RuntimeError("executor did not report its policy version")
                expected = self._policy_version if self._policy_version is not None else lo
                assert_rollout_policy_version(result.extra, expected, label=eid)
                self._policy_version = expected
                observed_version = expected
            except RuntimeError as err:
                self._policy_faults.append({"episode_id": eid, "error": str(err)})
                raise
            return result

        result = await run_episode(
            adapter=_PromptAdapter(adapter, scaffold), generate=frozen_generate,
            tokenizer=router._tokenizer, delta=router._delta, guidance=skill,
            episode_id=eid, sample_idx=seed, response_budget=router._episode_budget,
            sampling_params={"temperature": self._temperature, "seed": seed},
            resume=resume,
        )
        traj = result.trajectory
        # The core stops before copying extras for an empty generation. Preserve
        # the version actually checked here, including zero-token attempts.
        if observed_version is not None:
            result.gen_extra.update(min_global_steps=observed_version,
                                    max_global_steps=observed_version)
        attach_episode_policy_range(traj, result.gen_extra)
        traj.meta["skill_evolution"] = {
            "phase": phase, "seed": seed, "skill_sha256": digest(skill),
            "scaffold_sha256": digest(scaffold), "response_budget": router._episode_budget,
            "temperature": self._temperature,
        }
        self.trajectories.append(traj.to_dict())
        if phase == "discovery":
            self._sources[eid] = result
        if traj.outcome != OUTCOME_ENV_ERROR:
            st.episodes_consumed += 1
            st.exp_attempts_total += 1
            st.exp_successes_total += int(traj.won)
        # Validation never calls the narrator. Its judgement is irrelevant to the gate.
        if phase == "discovery":
            record = await router._make_record(traj)
        else:
            from ..narrate.factcard import build_fact_card
            from .tools import EpisodeRecord
            record = EpisodeRecord(traj, build_fact_card(traj), None)
        st.episodes[eid] = record
        return result

    async def discover(self, args):
        self._budget()
        if self.pending is not None:
            raise ValueError("validate the pending candidate before collecting another round")
        if args.get("guidance", self.active_skill) != self.active_skill:
            raise ValueError("Skill is frozen during discovery; use propose_revision to change it")
        st = self.router.state
        st.experiments_used += 1
        tasks = []
        for prompt in self.cfg.discovery_prompts:
            tasks.extend(self._run(self.router.adapter, self.active_skill, prompt, seed, "discovery")
                         for seed in self._seeds())
        results = await asyncio.gather(*tasks)
        reports = []
        for result in results:
            rec = self._record_error(result.trajectory)
            reports.append(rec)
        attempts = [r for r in results if r.trajectory.outcome != OUTCOME_ENV_ERROR]
        st.last_exp_attempts, st.last_exp_successes = len(attempts), sum(r.won for r in attempts)
        st.guidance_log.append({"experiment_idx": st.experiments_used, "phase": "discovery",
                                "guidance": self.active_skill,
                                "guidance_hash": guidance_hash(self.active_skill),
                                "k": len(results), "new_version": False})
        # Even failed infrastructure consumes a call here: validation is fail-closed
        # and retries cannot create an unbounded evolution loop.
        return json.dumps({"phase": "error_localization", "errors": reports,
                           "episodes": [{"episode_id": r.trajectory.episode_id,
                                         "fact_card": r.fact_card.render(),
                                         "narration": r.narration.text if r.narration else None,
                                         "mechanical_log": r.mechanical_log}
                                        for r in (st.episodes[x.trajectory.episode_id] for x in results)],
                           "successes": st.last_exp_successes, "attempts": len(attempts),
                           "remaining_experiments": self.router.cfg.n_exp - st.experiments_used,
                           "next": "Inspect fetch_raw, then propose_revision for one verified error_type."},
                          ensure_ascii=False)

    def _record_error(self, trajectory, **kwargs):
        rec = error_record(self.router.adapter, trajectory,
                           error_id=f"err{len(self.errors) + 1:04d}", **kwargs)
        self.errors.append(rec)
        return rec

    def record_error(self, args):
        source = self._sources.get(args.get("episode_id"))
        if source is None:
            raise ValueError("error evidence must reference a discovery episode")
        rec = self._record_error(source.trajectory, claimed_step=args.get("step"),
                                 claimed_type=args.get("error_type"),
                                 researcher_note=str(args.get("evidence", "")))
        return json.dumps(rec, ensure_ascii=False)

    def propose(self, args):
        self._budget()
        if self.pending is not None:
            raise ValueError("one candidate per round; validate it before revising again")
        if len(self.candidates) >= self.cfg.max_rounds:
            raise ValueError("evolution round limit reached")
        kind = args.get("error_type")
        if kind not in ERROR_TYPES:
            raise ValueError("select exactly one supported error_type")
        updates = args.get("rules")
        if not isinstance(updates, list) or not 1 <= len(updates) <= 8:
            raise ValueError("provide between one and eight local rules")
        rules = copy.deepcopy(self.rules)
        ids, sources = set(), set()
        by_id = {e["error_id"]: e for e in self.errors}
        for raw in updates:
            if not isinstance(raw, dict) or set(raw) != {"rule_id", "sources", *RULE_FIELDS}:
                raise ValueError("each rule needs rule_id, sources and all five rule fields")
            key = raw["rule_id"]
            if not isinstance(key, str) or not key or not key.replace("_", "").replace("-", "").isalnum():
                raise ValueError("rule_id must contain only letters, numbers, underscores or hyphens")
            if key in ids or (key in rules and rules[key]["error_type"] != kind):
                raise ValueError("cannot duplicate a rule or change a different error type")
            ids.add(key)
            if any(not isinstance(raw[n], str) or not raw[n].strip() for n in RULE_FIELDS):
                raise ValueError("trigger/checks/action/completion/recovery must be non-empty strings")
            if not isinstance(raw["sources"], list) or not 1 <= len(raw["sources"]) <= 8:
                raise ValueError("each rule must cite one to eight error records")
            for source_id in raw["sources"]:
                rec = by_id.get(source_id)
                if (rec is None or rec["status"] != "verified" or rec["error_type"] != kind
                        or rec["skill_hash"] != guidance_hash(self.active_skill)):
                    raise ValueError("sources must be verified errors of this type under the current Skill")
                source = self._sources[rec["episode_id"]].trajectory
                if rec["trajectory_sha256"] != digest(source.to_dict()):
                    raise ValueError("source trajectory changed after diagnosis")
                sources.add(source_id)
            rules[key] = {**copy.deepcopy(raw), "error_type": kind}
        if len(sources) > 8:
            raise ValueError("a round may use at most eight error records")
        skill = render_skill(self.cfg.initial_skill, rules)
        if skill == self.active_skill:
            raise ValueError("revision must change the Skill")
        if len(skill.split()) > self.router.cfg.doc_max_words:
            raise ValueError("candidate exceeds doc_max_words")
        candidate = {
            "candidate_id": f"candidate{len(self.candidates) + 1:03d}",
            "round": len(self.candidates) + 1, "error_type": kind,
            "source_errors": sorted(sources), "base_skill": self.active_skill,
            "base_skill_sha256": digest(self.active_skill), "skill": skill,
            "skill_sha256": digest(skill), "rules": rules, "changed_rule_ids": sorted(ids),
            "status": "proposed", "replacement_conditions": {
                "min_error_reduction": self.cfg.min_error_reduction,
                "min_success_gain": self.cfg.min_success_gain,
                "max_holdout_regression": self.cfg.max_holdout_regression,
                "require_holdout": self.cfg.require_holdout,
                "samples": self.cfg.samples,
            },
        }
        self.candidates.append(candidate)
        self.pending = candidate
        return json.dumps({k: v for k, v in candidate.items() if k != "rules"}, ensure_ascii=False)

    async def _pairs(self, adapter, candidate, phase, *, resume=None):
        runs = []
        for seed in self._seeds():
            runs.extend(self._run(adapter, skill, self.cfg.validation_prompt, seed,
                                  f"{phase}:{arm}", resume=resume)
                        for arm, skill in (("old", candidate["base_skill"]),
                                           ("new", candidate["skill"])))
        results = await asyncio.gather(*runs)
        return results[::2], results[1::2]

    @staticmethod
    def _metrics(adapter, results, kind, *, measure_errors=True):
        valid = [r for r in results if r.trajectory.outcome != OUTCOME_ENV_ERROR]
        return {"episode_ids": [r.trajectory.episode_id for r in results],
                "attempts": len(valid), "requested": len(results),
                "success_rate": sum(r.won for r in valid) / len(valid) if valid else None,
                "target_error_rate": sum(has_error(adapter, r.trajectory, kind) for r in valid)
                                     / len(valid) if valid and measure_errors else None}

    async def validate(self, args):
        self._budget()
        c = self.pending
        if c is None or c["candidate_id"] != args.get("candidate_id") or c["status"] != "proposed":
            raise ValueError("validate exactly the pending, untested candidate")
        c["status"] = "validating"
        self.router.state.experiments_used += 1
        report = {"candidate_id": c["candidate_id"], "status": "running",
                  "local": [], "full": None, "holdout": [], "reasons": []}
        self.validations.append(report)
        reasons = report["reasons"]
        try:
            if asdict(self.cfg) != self._settings:
                raise ValueError("evolution settings changed after discovery started")
            if digest(self.active_skill) != c["base_skill_sha256"] or digest(c["skill"]) != c["skill_sha256"]:
                raise ValueError("Skill changed after the candidate was sealed")
            if self.router.adapter.executor_scaffold() != self._scaffold:
                raise ValueError("executor scaffold changed during the session")
            holdouts = []
            if self._holdout_rows:
                if self.adapter_factory is None:
                    raise ValueError("holdout tasks require an adapter_factory")
                holdouts = [self.adapter_factory(row) for row in self._holdout_rows]
                ids = [a.task_id for a in holdouts]
                if len(set(ids)) != len(ids) or self.router.adapter.task_id in ids:
                    raise ValueError("holdout tasks must be unique and exclude the discovery task")
            if self.cfg.require_holdout and not holdouts:
                raise ValueError("replacement requires configured held-out tasks")
            kind = c["error_type"]
            if getattr(self.router.adapter, "supports_exact_replay", False):
                for error_id in c["source_errors"]:
                    rec = next(e for e in self.errors if e["error_id"] == error_id)
                    source = self._sources[rec["episode_id"]]
                    if digest(source.trajectory.to_dict()) != rec["trajectory_sha256"]:
                        raise ValueError("source evidence changed")
                    pairs = await self._pairs(self.router.adapter, c, "local",
                                              resume=(source, rec["step"]))
                    local = {"error_id": error_id, "status": "measured"}
                    for arm, results in zip(("old", "new"), pairs):
                        m = self._metrics(self.router.adapter, results, kind, measure_errors=False)
                        correct = 0
                        for result in results:
                            if not result.trajectory.steps:
                                continue
                            context = Trajectory.from_dict(source.trajectory.to_dict())
                            next_step = copy.deepcopy(result.trajectory.steps[0])
                            next_step.t = rec["step"]
                            context.steps = context.steps[:rec["step"] - 1] + [next_step]
                            first = context.steps[-1]
                            diagnosis = step_error(self.router.adapter, context, len(context.steps) - 1)
                            # Correct *with respect to the selected error*, not a claim
                            # that every legal action is globally optimal.
                            correct += int(first.acted and first.valid and diagnosis is None)
                        m["correct_action_rate"] = correct / len(results)
                        local[arm] = m
                    report["local"].append(local)
                    old, new = local["old"], local["new"]
                    if any(m["attempts"] != self.cfg.samples for m in (old, new)):
                        reasons.append("incomplete_local_pairs_or_replay_drift")
                    elif (new["correct_action_rate"] <= old["correct_action_rate"] or
                          new["success_rate"] < old["success_rate"]):
                        reasons.append("local_correction_not_improved")
            else:
                report["local"].append({"status": "unsupported",
                                        "reason": "adapter does not declare exact replay support"})
            # Local results NEVER replace fresh whole-task validation.
            old_runs, new_runs = await self._pairs(self.router.adapter, c, "full")
            report["full"] = {arm: self._metrics(self.router.adapter, runs, kind)
                              for arm, runs in (("old", old_runs), ("new", new_runs))}
            old, new = report["full"]["old"], report["full"]["new"]
            if any(m["attempts"] != self.cfg.samples for m in (old, new)):
                reasons.append("incomplete_full_pairs")
            else:
                if old["target_error_rate"] - new["target_error_rate"] + 1e-12 < self.cfg.min_error_reduction:
                    reasons.append("target_error_reduction_below_threshold")
                if new["success_rate"] - old["success_rate"] + 1e-12 < self.cfg.min_success_gain:
                    reasons.append("whole_task_success_gain_below_threshold")
            for adapter in holdouts:
                old_runs, new_runs = await self._pairs(adapter, c, "holdout")
                measurements = {arm: self._metrics(adapter, runs, kind, measure_errors=False)
                                for arm, runs in (("old", old_runs), ("new", new_runs))}
                report["holdout"].append({"task_id": adapter.task_id, **measurements})
                old, new = measurements["old"], measurements["new"]
                if any(m["attempts"] != self.cfg.samples for m in (old, new)):
                    reasons.append(f"incomplete_holdout_pairs:{adapter.task_id}")
                elif old["success_rate"] - new["success_rate"] > self.cfg.max_holdout_regression + 1e-12:
                    reasons.append(f"holdout_regression:{adapter.task_id}")
            if self._policy_version is None or self._policy_faults:
                reasons.append("executor_policy_not_frozen")
        except asyncio.CancelledError:
            report["status"], c["status"] = "interrupted", "rejected"
            reasons.append("validation_interrupted")
            self.pending = None
            raise
        except Exception as err:  # noqa: BLE001 — faulty evaluation must never promote a Skill
            reasons.append(f"validation_error:{type(err).__name__}:{err}")
        accepted = not reasons
        report["status"] = c["status"] = "accepted" if accepted else "rejected"
        report["policy_version"] = self._policy_version
        report["replacement_conditions"] = copy.deepcopy(c["replacement_conditions"])
        if accepted:
            self.active_skill, self.rules = c["skill"], copy.deepcopy(c["rules"])
        if report["full"] is not None:
            current = report["full"]["new" if accepted else "old"]
            self.router.state.last_exp_attempts = current["attempts"]
            self.router.state.last_exp_successes = round(
                (current["success_rate"] or 0) * current["attempts"])
        report["active_skill_sha256"] = digest(self.active_skill)
        self.pending = None
        return json.dumps(report, ensure_ascii=False)

    def finalize(self, args):
        if self.router.state.finalized:
            return "Already finalized."
        doc = args.get("experience_doc", self.active_skill)
        if doc != self.active_skill:
            raise ValueError("finalize cannot replace a Skill; submit only the accepted active_skill")
        if self.pending is not None:
            # Ending a session may abandon a proposal, but may never promote it.
            self.pending["status"] = "abandoned"
            self.pending["reason"] = "finalized_without_validation"
            self.pending = None
        st = self.router.state
        st.finalized, st.doc = True, self.active_skill or None
        st.self_assessment = str(args.get("self_assessment", ""))
        return "Session complete. Only the accepted Skill is released; all proposals and decisions are archived."
