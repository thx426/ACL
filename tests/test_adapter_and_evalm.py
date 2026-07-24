"""Adapter identity fields (review P1-5) and evaluator summary discipline
(review P1-9) — no engine, no server."""
import threading

from trace_opd.common.config import AlfworldCfg
from trace_opd.core.types import OUTCOME_ENV_ERROR, OUTCOME_FAILED, OUTCOME_SUCCESS
from trace_opd.envs.alfworld.adapter import AlfworldTaskAdapter
from trace_opd.evalm.run_eval import summarize


def _adapter(task_id: str) -> AlfworldTaskAdapter:
    return AlfworldTaskAdapter(
        {
            "task_id": task_id,
            "gamefile": "x/game.tw-pddl",
            "initial_obs": "You are in a kitchen. Your task is to: heat some egg.",
            "gt_text": "1. GotoLocation(fridge)",
        },
        AlfworldCfg(data_root="/d", worker_python="/p"),
        spawn_gate=threading.Semaphore(1),
    )


def test_task_type_skips_split_segment():
    a = _adapter("train/pick_heat_then_place_in_recep-Egg-None-Fridge-13/trial_T123")
    assert a.task_type == "pick_heat_then_place_in_recep"
    # ids without a split prefix still resolve
    b = _adapter("pick_and_place_simple-Vase-None-Shelf-1/trial_T9")
    assert b.task_type == "pick_and_place_simple"


def test_task_brief_carries_real_type_and_scene():
    a = _adapter("train/look_at_obj_in_light-Book-None-DeskLamp-308/trial_T1")
    brief = a.task_brief()
    assert "Task type: look_at_obj_in_light" in brief
    assert "heat some egg" in brief
    assert a.gt_text().startswith("1. GotoLocation")


def _row(split, outcome, won, steps=10):
    return {"split": split, "outcome": outcome, "won": won, "env_steps": steps}


def test_summarize_excludes_env_errors_from_denominator():
    rows = [
        _row("valid_seen", OUTCOME_SUCCESS, True),
        _row("valid_seen", OUTCOME_FAILED, False),
        _row("valid_seen", OUTCOME_ENV_ERROR, False),
        _row("valid_unseen", OUTCOME_SUCCESS, True),
    ]
    summary, incomplete = summarize(rows, "bare")
    seen = summary["splits"]["valid_seen"]
    assert seen["total"] == 3 and seen["attempts"] == 2
    assert seen["success_rate"] == 0.5          # 1/2, NOT 1/3
    assert seen["env_errors"] == 1
    assert incomplete and summary["incomplete"]


def test_summarize_clean_run_is_complete():
    rows = [_row("valid_unseen", OUTCOME_FAILED, False)]
    summary, incomplete = summarize(rows, "bare")
    assert not incomplete
    assert summary["splits"]["valid_unseen"]["success_rate"] == 0.0
