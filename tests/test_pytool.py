"""pytool 共享执行件:孤儿进程防线(math 域 A800 load 300+ 事故定式)。

真 spawn 孙进程验证:超时击杀整组、正常退出后清扫遗孤——朴素
subprocess.run(timeout=) 两者都做不到(只杀直接子进程,孙进程逃逸)。
"""
from __future__ import annotations

import os
import threading
import time

from trace_opd.envs.pytool import clip_middle, run_python_script

GATE = threading.Semaphore(2)

#: 中间父进程 spawn 一个 sleep 60 的孙进程并落盘其 pid,然后按 tail 行动。
_SPAWN_GRANDCHILD = (
    "import subprocess, sys, time\n"
    "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
    "open('grandchild.pid', 'w').write(str(p.pid))\n"
    "{parent_tail}\n"
)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover — 他人进程占用该 pid
        return True


def _wait_dead(pid: int, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


def test_timeout_kills_whole_process_group(tmp_path):
    code = _SPAWN_GRANDCHILD.format(parent_tail="time.sleep(60)")
    t0 = time.monotonic()
    out, rc = run_python_script(code, cwd=str(tmp_path), timeout_s=2.0, gate=GATE)
    assert rc == 124 and "timeout" in out
    # 孙进程持有 stdout 写端:朴素 run() 在这里会挂到孙进程退出(60s);
    # killpg 路线必须秒级返回。
    assert time.monotonic() - t0 < 15
    pid = int((tmp_path / "grandchild.pid").read_text())
    assert _wait_dead(pid), f"grandchild {pid} survived the group kill"


def test_normal_exit_sweeps_orphaned_grandchildren(tmp_path):
    code = _SPAWN_GRANDCHILD.format(parent_tail="print('done')")
    t0 = time.monotonic()
    out, rc = run_python_script(code, cwd=str(tmp_path), timeout_s=30.0, gate=GATE)
    # 孙进程持 stdout 写端、父正常退出:communicate 版会等满 30s 误判超时
    # 丢弃成功输出(本机复现);wait+drain 版必须秒级返回真实结果。
    assert time.monotonic() - t0 < 15
    assert rc == 0 and "done" in out
    pid = int((tmp_path / "grandchild.pid").read_text())
    assert _wait_dead(pid), f"orphaned grandchild {pid} survived the sweep"


def test_plain_script_semantics_unchanged(tmp_path):
    out, rc = run_python_script(
        "print('hi'); import sys; print('err', file=sys.stderr)",
        cwd=str(tmp_path), timeout_s=10.0, gate=GATE,
    )
    assert rc == 0 and "hi" in out and "err" in out
    out, rc = run_python_script(
        "import sys; sys.exit(3)", cwd=str(tmp_path), timeout_s=10.0, gate=GATE
    )
    assert rc == 3 and out == "(no output)"


def test_clip_middle():
    assert clip_middle("abc", 10) == "abc"
    clipped = clip_middle("a" * 50 + "b" * 50, 20)
    assert len(clipped) < 60 and "truncated" in clipped
