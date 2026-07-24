"""ALFWorld environment worker — a standalone one-episode subprocess CLI.

Lineage: experience_internalization_claude worker → h1_pilot (GT support)
→ trace_opd (verbatim; stdlib-only, loaded by file path, zero package
coupling). GT support:
  * optional GT support: with argv[3] == "--gt" the worker additionally
    requests ``policy_commands`` at registration and answers a ``__GT__``
    request line with {"event": "gt", "plan": [...] | null}. The GT switch
    is a host-side config; without the flag nothing GT-related is ever
    requested from the engine (firewall default).

WHY A SUBPROCESS CLI: the TextWorld/ALFWorld engine is NOT thread-safe (the
tatsu grammar parser keeps process-global state). Process isolation per
episode is the only safe concurrency model, and a plain subprocess with a
configurable interpreter decouples the orchestrator's venv from the
environment's (this file imports only the standard library at module level).

PROTOCOL (JSON lines):
  parent -> child stdin : one environment command per line;
                          ``__GT__`` requests the expert plan (gt mode only);
                          ``__CLOSE__`` terminates the episode.
  child  -> parent      : one JSON object per line on the child's ORIGINAL
                          stdout (duplicated before fd redirection):
      {"event": "ready", "obs": str}
      {"event": "step", "obs": str, "done": bool, "won": bool}
      {"event": "gt", "plan": list[str] | null}
      {"event": "fatal", "error": str}
"""
from __future__ import annotations

import importlib.util
import os
import sys
import traceback

_RT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker_runtime.py")
_rt_spec = importlib.util.spec_from_file_location("worker_runtime", _RT_PATH)
_runtime = importlib.util.module_from_spec(_rt_spec)
_rt_spec.loader.exec_module(_runtime)

CLOSE_SENTINEL = _runtime.CLOSE_SENTINEL
GT_SENTINEL = "__GT__"


def main() -> int:
    _runtime.harden()
    proto = _runtime.protocol_channel()
    gt_plan = None
    try:
        gamefile = sys.argv[1]
        max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        want_gt = len(sys.argv) > 3 and sys.argv[3] == "--gt"

        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

        if want_gt:
            request = textworld.EnvInfos(won=True, policy_commands=True)
        else:
            request = textworld.EnvInfos(won=True)
        env_id = textworld.gym.register_game(
            gamefile,
            request,
            max_episode_steps=max_steps,
            wrappers=[AlfredDemangler()],
        )
        env = textworld.gym.make(env_id)
        obs, infos = env.reset()
        if want_gt:
            try:
                pc = infos.get("policy_commands")
                gt_plan = [str(c) for c in pc] if pc else None
            except Exception:  # noqa: BLE001
                gt_plan = None
        _runtime.emit(proto, {"event": "ready", "obs": obs})
    except Exception:  # noqa: BLE001 — report through the protocol, then die
        _runtime.emit(proto, {"event": "fatal", "error": traceback.format_exc()})
        return 1

    for command in _runtime.iter_stdin_lines():
        try:
            if command == GT_SENTINEL:
                _runtime.emit(proto, {"event": "gt", "plan": gt_plan})
                continue
            obs, _score, done, infos = env.step(command)
            _runtime.emit(
                proto,
                {
                    "event": "step",
                    "obs": _runtime.truncate_middle(obs, 6000),
                    "done": bool(done),
                    "won": bool(infos.get("won", False)),
                },
            )
        except Exception:  # noqa: BLE001
            _runtime.emit(proto, {"event": "fatal", "error": traceback.format_exc()})
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
