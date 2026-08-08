"""ScienceWorld environment worker — a standalone one-episode subprocess CLI.

Lineage: envs/alfworld/worker.py(line 协议/hardening/GT 开关同款);引擎换
scienceworld pip 包(Scala JAR + py4j,每 worker 一个 JVM,本地实测 boot
0.6s + load 0.6s——冷启成本低,无需池化)。runtime 公共件跨目录复用
alfworld/worker_runtime.py(worker 以文件路径启动,零包耦合)。

PROTOCOL (JSON lines):
  parent -> child stdin : one environment command per line;
                          ``__GT__`` requests the gold action sequence (gt
                          mode only); ``__CLOSE__`` terminates the episode.
  child  -> parent      :
      {"event": "ready", "obs": str}            # obs = taskDesc + reset obs
      {"event": "step", "obs": str, "done": bool, "won": bool, "score": int}
      {"event": "gt", "plan": list[str] | null}
      {"event": "fatal", "error": str}

argv: task_name variation_idx simplifications env_step_limit [--gt]
  task_name 接受 ETO 长名(如 task-1-boil)——包内 infer_task 原生转换
  (本地实测确认)。won = isCompleted 且 score >= 100(严格版,终局二元
  reward 的判据;RLVMR 的宽松版 score>0 不采,差异随对照披露)。
"""
from __future__ import annotations

import importlib.util
import os
import sys
import traceback

_RT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "alfworld", "worker_runtime.py"
)
_rt_spec = importlib.util.spec_from_file_location("worker_runtime", _RT_PATH)
_runtime = importlib.util.module_from_spec(_rt_spec)
_rt_spec.loader.exec_module(_runtime)

CLOSE_SENTINEL = _runtime.CLOSE_SENTINEL
GT_SENTINEL = "__GT__"


def main() -> int:
    _runtime.harden()
    proto = _runtime.protocol_channel()
    gold_plan = None
    try:
        task_name = sys.argv[1]
        variation_idx = int(sys.argv[2])
        simplifications = sys.argv[3]
        env_step_limit = int(sys.argv[4])
        want_gt = len(sys.argv) > 5 and sys.argv[5] == "--gt"

        from scienceworld import ScienceWorldEnv

        env = ScienceWorldEnv("", envStepLimit=env_step_limit)
        env.load(
            task_name,
            variation_idx,
            simplificationStr=simplifications,
            generateGoldPath=want_gt,
        )
        obs, _info = env.reset()
        if want_gt:
            try:
                plan = env.get_gold_action_sequence()
                # 未生成时包返回单元素 ERROR 字符串(官方语义),归一为 None
                gold_plan = (
                    [str(c) for c in plan]
                    if plan and not str(plan[0]).startswith("ERROR")
                    else None
                )
            except Exception:  # noqa: BLE001
                gold_plan = None
        # 首观察口径:任务描述 + 起始房间观察(scene + goal,与 ALFWorld 的
        # initial_obs 语义对齐;taskDesc 不再单独下发)
        ready_obs = f"Task: {env.get_task_description()}\n\n{obs}"
        _runtime.emit(proto, {"event": "ready", "obs": ready_obs})
    except Exception:  # noqa: BLE001 — report through the protocol, then die
        _runtime.emit(proto, {"event": "fatal", "error": traceback.format_exc()})
        return 1

    for command in _runtime.iter_stdin_lines():
        try:
            if command == GT_SENTINEL:
                _runtime.emit(proto, {"event": "gt", "plan": gold_plan})
                continue
            obs, _reward, done, info = env.step(command)
            score = int(info.get("score", 0))
            _runtime.emit(
                proto,
                {
                    "event": "step",
                    "obs": _runtime.truncate_middle(obs, 6000),
                    # done 三来源(官方 step):真完成 / 超 envStepLimit /
                    # score<0(不可逆失败强制终局)。won 用严格判据。
                    "done": bool(done),
                    "won": bool(done and score >= 100),
                    "score": score,
                },
            )
        except Exception:  # noqa: BLE001
            _runtime.emit(proto, {"event": "fatal", "error": traceback.format_exc()})
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
