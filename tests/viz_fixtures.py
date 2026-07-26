"""Synthetic runs/ trees for viz tests — schema mirrors the real writers
(train/agent_loop.py::_dump_rollout, docstore/archive.py) exactly.

Two environments on purpose: `alfworld` exercises the plugin path, `gridworld`
(a made-up action vocabulary, no plugin) is the genericity acceptance fixture —
the kernel must be fully functional on it.
"""
from __future__ import annotations

import json
from pathlib import Path


def _step(t, thought, action, obs, valid=True, acted=True):
    return {
        "t": t, "thought": thought, "action_raw": f"<action>{action}</action>",
        "action_norm": action, "observation": obs, "valid": valid, "acted": acted,
    }


def _traj(task_id, steps, won, sample_idx=0, outcome=None):
    return {
        "episode_id": "student", "task_id": task_id, "guidance_hash": "bare",
        "sample_idx": sample_idx, "initial_obs": "You are in a room.",
        "steps": steps, "outcome": outcome or ("SUCCESS" if won else "FAILED"),
        "won": won, "error": None, "meta": {},
    }


def write_dump(
    run_dir: Path, *, step: int, validate: bool, task_id: str, traj: dict,
    split: str = "train", uid: str = "aaaa0000", archive: str = "",
    invalid_format: int = 0, truncated: bool = False, final_success: float = 0.0,
    sample_role: str = "bare",
) -> Path:
    step_dir = run_dir / "rollouts" / (f"step_{step:05d}" + ("_val" if validate else ""))
    step_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": task_id, "split": split, "validate": validate,
        "global_step": step, "outcome": traj["outcome"], "won": traj["won"],
        "env_steps": sum(1 for s in traj["steps"] if s["acted"]),
        "assistant_turns": len(traj["steps"]), "invalid_format": invalid_format,
        "truncated": truncated, "doc_words": 120,
        "research": {
            "research_error": 0, "research_experiments": 3, "research_episodes": 9,
            "research_turns": 12, "research_final_success": final_success,
            "doc_missing": 0, "research_archive": archive,
        },
        "trajectory": traj, "ts": 1000.0 + step,
        "sample_role": sample_role,
    }
    path = step_dir / f"{task_id.replace('/', '__')}_{uid}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_research(
    run_dir: Path, *, task_id: str, step: int, doc: str | None = "# Guidance\n\nGo directly.",
) -> str:
    rel = f"research/{task_id.replace('/', '__')}/20260724_00000{step}_v{step}"
    session = run_dir / rel
    session.mkdir(parents=True, exist_ok=True)
    meta = {
        "task_id": task_id, "ts": 2000.0 + step, "policy_version": step,
        "finalized": True, "no_finalize": False, "turns_used": 12,
        "experiments_used": 3, "episodes_consumed": 9,
        "doc_words": len(doc.split()) if doc else 0, "self_assessment": "solved",
        "flags": [], "last_exp_successes": 2, "last_exp_attempts": 2,
        "guidance_log": ["v1", "v2"], "session_error": None,
        "researcher_prompt_tokens": 1000, "researcher_completion_tokens": 200,
        "researcher_model": "m-27b", "tool_protocol": "native", "error": None,
    }
    (session / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if doc:
        (session / "doc.md").write_text(doc, encoding="utf-8")
    (session / "session.jsonl").write_text(
        json.dumps({"role": "user", "content": "task brief"}) + "\n", encoding="utf-8"
    )
    (session / "episodes.jsonl").write_text(
        json.dumps({"trajectory": _traj(task_id, [], True), "fact_card": "",
                    "narration": None, "narration_skip": None,
                    "narration_conflict": False}) + "\n",
        encoding="utf-8",
    )
    return rel


# ------------------------------------------------------------------ alfworld

ALF_TASK = "train/pick_and_place_simple-Soap-None-GarbageCan-5/trial_1"
ALF_TASK_HEAT = "train/pick_heat_then_place_in_recep-Egg-None-GarbageCan-2/trial_9"

WIN_STEPS = [
    _step(0, "find it", "go to cabinet 4", "You arrive at cabinet 4."),
    _step(1, "grab", "take soapbottle 1 from cabinet 4", "You pick up the soapbottle 1."),
    _step(2, "deliver", "go to garbagecan 1", "You arrive at garbagecan 1."),
    _step(3, "drop", "move soapbottle 1 to garbagecan 1", "You move the soapbottle 1."),
]
NEVER_MOVED_STEPS = [
    _step(0, "hmm", "examine drawer 1", "Nothing happens.", valid=False),
    _step(1, "hmm", "examine drawer 1", "Nothing happens.", valid=False),
    _step(2, "??", "look", "Nothing happens.", valid=False),
    _step(3, "", "examine drawer 1", "FORMAT reminder", valid=False, acted=False),
]
MOVED_NO_TAKE_STEPS = [
    _step(0, "go", "go to drawer 1", "You arrive at drawer 1."),
    _step(1, "open", "open drawer 1", "The drawer 1 is open."),
    _step(2, "empty", "go to drawer 2", "You arrive at drawer 2."),
]
TOOK_NO_FINISH_STEPS = [
    _step(0, "go", "go to cabinet 1", "You arrive at cabinet 1."),
    _step(1, "grab", "take egg 1 from cabinet 1", "You pick up the egg 1."),
    _step(2, "drop?", "move egg 1 to garbagecan 1", "Nothing happens.", valid=False),
    _step(3, "drop?", "move egg 1 to garbagecan 1", "Nothing happens.", valid=False),
]
EXHAUSTIVE_STEPS = [
    _step(i, "search", f"go to drawer {i + 1}", f"You arrive at drawer {i + 1}.")
    for i in range(6)
]


def make_alfworld_run(root: Path, run: str = "alf_run") -> Path:
    run_dir = root / run
    for step in (1, 2):
        arch = write_research(run_dir, task_id=ALF_TASK, step=step)
        arch_heat = write_research(run_dir, task_id=ALF_TASK_HEAT, step=step)
        win = 1 if step == 2 else 0  # success only appears at step-2
        write_dump(run_dir, step=step, validate=False, task_id=ALF_TASK,
                   traj=_traj(ALF_TASK, WIN_STEPS if win else NEVER_MOVED_STEPS, bool(win)),
                   uid=f"win{step}0000", archive=arch, final_success=1.0)
        write_dump(run_dir, step=step, validate=False, task_id=ALF_TASK,
                   traj=_traj(ALF_TASK, MOVED_NO_TAKE_STEPS, False, sample_idx=1),
                   uid=f"mnt{step}0000", archive=arch, final_success=1.0)
        write_dump(run_dir, step=step, validate=False, task_id=ALF_TASK_HEAT,
                   traj=_traj(ALF_TASK_HEAT, TOOK_NO_FINISH_STEPS, False),
                   uid=f"tnf{step}0000", archive=arch_heat, final_success=0.5,
                   invalid_format=1)
        write_dump(run_dir, step=step, validate=False, task_id=ALF_TASK_HEAT,
                   traj=_traj(ALF_TASK_HEAT, EXHAUSTIVE_STEPS, False, sample_idx=1),
                   uid=f"exh{step}0000", archive=arch_heat, final_success=0.5,
                   truncated=True)
    write_dump(run_dir, step=2, validate=True, task_id=ALF_TASK,
               traj=_traj(ALF_TASK, WIN_STEPS, True), split="valid_seen", uid="valseen0")
    write_dump(run_dir, step=2, validate=True, task_id=ALF_TASK_HEAT,
               traj=_traj(ALF_TASK_HEAT, NEVER_MOVED_STEPS, False),
               split="valid_unseen", uid="valunsn0")
    return run_dir


# ----------------------------------------------------------------- gridworld

GRID_TASK = "maze/room-3"
GRID_WIN = [
    _step(0, "walk", "north", "You step north."),
    _step(1, "walk", "east", "You step east."),
    _step(2, "grab", "grab key", "Key acquired."),
]
GRID_LOSE = [
    _step(0, "walk", "south", "A wall."),
    _step(1, "walk", "south", "A wall.", valid=False),
    _step(2, "walk", "south", "A wall.", valid=False),
    _step(3, "", "south", "FORMAT reminder", valid=False, acted=False),
]


def make_gridworld_run(root: Path, run: str = "grid_run") -> Path:
    run_dir = root / run
    arch = write_research(run_dir, task_id=GRID_TASK, step=1)
    write_dump(run_dir, step=1, validate=False, task_id=GRID_TASK,
               traj=_traj(GRID_TASK, GRID_WIN, True), uid="gwin0000",
               archive=arch, final_success=1.0)
    write_dump(run_dir, step=1, validate=False, task_id=GRID_TASK,
               traj=_traj(GRID_TASK, GRID_LOSE, False, sample_idx=1),
               uid="glose000", archive=arch)
    return run_dir
