"""AppWorld environment worker — a standalone one-episode subprocess CLI.

自 skills_evolve css/envs/appworld/worker.py 移植,runtime 依赖内联自足化
(本文件模块级只 import 标准库;appworld 在 main() 内延迟加载,worker
解释器 = 配置的 py3.11 venv)。

WHY A SUBPROCESS(官方硬约束,原样继承):AppWorld unified mode 每 OS
进程只允许一个 live world(freezegun 进程级 mock 时钟,DB/内存管理假定
单 world)——每 episode 一个 worker 进程使隔离按构造成立;宿主进程永不
import appworld。

PROTOCOL(JSON lines;stdin 收请求,原始 stdout 复制为协议通道后被重定向
到 stderr,引擎噪声不污染协议):
  parent -> child : {"op": "execute", "code": str}
                    {"op": "evaluate"}
                    {"op": "gold"}
                    字面 ``__CLOSE__``(或 stdin EOF)结束 episode
  child  -> parent: {"event": "ready", "instruction", "supervisor", "metadata"}
                    {"event": "execute_result", "output", "completed"}
                    {"event": "evaluation", "success", "report"}
                    {"event": "gold", "solution_code", "answer"}
                    {"event": "fatal", "error"}

GROUND-TRUTH FIREWALL(继承):gold 只由宿主显式调用(现行 gt=off 从不
调);ready/execute_result 永不携带任何 gold 内容;metadata(难度)只进
数据构建侧,不进 executor。

argv: worker.py <task_id> <experiment_name> <ground_truth_mode>
              <max_interactions> <obs_max_chars>
env : APPWORLD_ROOT 指向数据 checkout(含 data/)。
"""
from __future__ import annotations

import json
import os
import sys
import traceback

CLOSE_SENTINEL = "__CLOSE__"


# ---- 内联 runtime(自足;Linux PDEATHSIG 尽力而为) ----------------------

def _harden() -> None:
    try:
        import ctypes  # noqa: PLC0415

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        import signal  # noqa: PLC0415

        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    except Exception:  # noqa: BLE001 — 非 Linux/失败仅失去自杀保护
        pass


def _protocol_channel():
    """复制原始 stdout 作协议通道;stdout 重定向 stderr(引擎打印隔离)。"""
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    return proto


def _emit(proto, obj: dict) -> None:
    proto.write(json.dumps(obj, ensure_ascii=False) + "\n")
    proto.flush()


def _iter_stdin_lines():
    for line in sys.stdin:
        if line.strip() == CLOSE_SENTINEL:
            return
        yield line


def _truncate_middle(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return text[:head] + "\n[... output truncated ...]\n" + text[-tail:]


# ---- 引擎对象序列化(移植) ----------------------------------------------

def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except Exception:  # noqa: BLE001
        pass
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict())
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _supervisor_dict(task) -> dict:
    sup = getattr(task, "supervisor", None)
    if sup is None:
        return {}
    if isinstance(sup, dict):
        return {str(k): str(v) for k, v in sup.items()}
    out = {}
    for key in ("first_name", "last_name", "email", "phone_number"):
        out[key] = str(getattr(sup, key, ""))
    return out


def _task_metadata(world) -> dict:
    for holder in (getattr(world, "task", None),
                   getattr(getattr(world, "task", None), "ground_truth", None)):
        md = getattr(holder, "metadata", None)
        if md is None:
            continue
        if isinstance(md, dict):
            return _jsonable(md)
        out = {}
        for key in ("difficulty", "num_apis", "num_apps",
                    "num_solution_code_lines"):
            if hasattr(md, key):
                out[key] = _jsonable(getattr(md, key))
        if out:
            return out
    return {}


def _tracker_report(tracker) -> dict:
    """per-requirement 评估报告(防御跨 appworld 版本;优化侧燃料,永不
    进 executor)。"""
    report: dict = {}
    for name in ("to_dict", "as_dict", "dict", "report"):
        fn = getattr(tracker, name, None)
        if callable(fn):
            try:
                out = fn()
                if isinstance(out, (dict, list, str)) and out:
                    report["detail"] = _jsonable(out)
                    return report
            except Exception:  # noqa: BLE001
                continue
    for attr in ("passes", "failures", "passed_tests", "failed_tests",
                 "test_results"):
        val = getattr(tracker, attr, None)
        if isinstance(val, (list, tuple)) and val:
            report[attr] = [_jsonable(v) for v in val]
    if not report:
        report["detail"] = str(tracker)
    return report


def main() -> int:
    _harden()
    proto = _protocol_channel()
    try:
        task_id = sys.argv[1]
        experiment_name = sys.argv[2]
        ground_truth_mode = sys.argv[3] if len(sys.argv) > 3 else "minimal"
        max_interactions = int(sys.argv[4]) if len(sys.argv) > 4 else 50
        obs_max_chars = int(sys.argv[5]) if len(sys.argv) > 5 else 3000

        from appworld import AppWorld  # noqa: PLC0415 — worker venv only

        root = os.environ.get("APPWORLD_ROOT", "")
        if root:
            try:
                from appworld import update_root  # noqa: PLC0415

                update_root(root)
            except Exception:  # noqa: BLE001 — env var alone may suffice
                pass

        world = AppWorld(
            task_id=task_id,
            experiment_name=experiment_name,
            ground_truth_mode=ground_truth_mode,
            max_interactions=max_interactions,
        )
        _emit(proto, {
            "event": "ready",
            "instruction": str(world.task.instruction),
            "supervisor": _supervisor_dict(world.task),
            "metadata": _task_metadata(world),
        })
    except Exception:  # noqa: BLE001 — 经协议报告后退场
        _emit(proto, {"event": "fatal", "error": traceback.format_exc()})
        return 1

    try:
        for line in _iter_stdin_lines():
            if not line.strip():
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                _emit(proto, {
                    "event": "fatal",
                    "error": "malformed request line: %r" % line[:200]})
                return 1
            op = request.get("op", "")
            if op == "execute":
                output = world.execute(str(request.get("code", "")))
                _emit(proto, {
                    "event": "execute_result",
                    "output": _truncate_middle(str(output), obs_max_chars),
                    "completed": bool(world.task_completed()),
                })
            elif op == "evaluate":
                tracker = world.evaluate(suppress_errors=True)
                _emit(proto, {
                    "event": "evaluation",
                    "success": bool(getattr(tracker, "success", False)),
                    "report": _tracker_report(tracker),
                })
            elif op == "gold":
                gt = getattr(world.task, "ground_truth", None)
                code = str(getattr(gt, "compiled_solution_code", "") or "")
                answer = getattr(gt, "answer", None)
                _emit(proto, {
                    "event": "gold",
                    "solution_code": code,
                    "answer": _jsonable(answer) if answer is not None else None,
                })
            else:
                _emit(proto, {"event": "fatal", "error": "unknown op: %r" % op})
                return 1
    except Exception:  # noqa: BLE001
        _emit(proto, {"event": "fatal", "error": traceback.format_exc()})
        return 1
    finally:
        try:
            world.close()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
