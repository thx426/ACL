"""Shared sandboxed-Python execution for tool-calling benchmarks.

SSB 与 math 共用的 run_python 执行件(自 spreadsheetbench/adapter 提取):
subprocess 隔离、净化环境(PYTHONHASHSEED=0 保 v4 重放确定性)、超时击杀、
空输出语义。并发门由调用方传入(各 benchmark 的 cfg 定并发额度并做
per-worker 折算;CPU 饥饿纪律:环境子进程不得饿死推理引擎)。

孤儿防线(2026-08-17,math 域 A800 load 300+ 事故定式移植):模型代码里的
multiprocessing/ProcessPoolExecutor 会 spawn 孙进程。朴素
``subprocess.run(timeout=)`` 有两个坑,本机皆已复现:
  1. 超时只杀直接子进程,孙进程逃逸成孤儿满核空转;
  2. 孙进程继承 stdout/stderr 写端 ⇒ 管道无 EOF,**父进程正常退出也会把
     communicate 挂满 timeout**,成功输出被误判成超时丢弃。
修法 = ``start_new_session=True``(独立进程组)+ drain 线程读管道 +
``wait()`` 只等进程退出 + 退出后(无论超时与否)``killpg`` 整组 SIGKILL
**杀两轮**(第一轮送达瞬间恰在 fork 的进程可能逃逸,短待后补刀);组灭
⇒ 写端全关 ⇒ EOF 自然到达,drain 线程收尾。
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time


def _kill_process_group(pgid: int) -> None:
    """整组 SIGKILL,杀两轮(清理定式:单轮击杀会漏正在 fork 的孙进程)。
    组已空 ⇒ ProcessLookupError 静默返回。"""
    for _ in range(2):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            return
        time.sleep(0.05)


def _drain(stream, key: str, buf: dict) -> None:
    try:
        buf[key] = stream.read()
    except Exception:  # noqa: BLE001 — 击杀竞态下的 IO 异常按空输出处理
        buf.setdefault(key, "")
    finally:
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            pass


def run_python_script(
    code: str, *, cwd: str, timeout_s: float, gate: threading.Semaphore
) -> tuple[str, int]:
    """Execute ``code`` with the current interpreter in ``cwd``;
    return (combined stdout+stderr, returncode). Timeout → rc=124."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": cwd,
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",   # 重放确定性
    }
    with gate:
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,   # 新会话 ⇒ pgid == pid,可整组击杀
        )
        buf: dict[str, str] = {}
        threads = [
            threading.Thread(
                target=_drain, args=(s, k, buf), daemon=True
            )
            for k, s in (("out", proc.stdout), ("err", proc.stderr))
        ]
        for t in threads:
            t.start()
        timed_out = False
        try:
            proc.wait(timeout=timeout_s)   # 只等进程退出,不等管道 EOF
        except subprocess.TimeoutExpired:
            timed_out = True
        # 超时 = 击杀;正常退出 = 清扫遗孤(daemon 化/双 fork 的后台孙进程
        # 同样是孤儿源)。两条路径都以组灭告终 ⇒ 写端全关,drain 收到 EOF。
        _kill_process_group(proc.pid)
        proc.wait()                        # reap(组已 SIGKILL,立即返回)
        for t in threads:
            t.join(timeout=5)
    if timed_out:
        # 超时反馈保持固定串(残余半截输出不夹带)。
        return f"[timeout after {timeout_s:.0f}s]", 124
    out = (buf.get("out") or "") + (buf.get("err") or "")
    return out if out.strip() else "(no output)", proc.returncode


def clip_middle(text: str, limit: int) -> str:
    """工具输出中部截断(保头尾;<tool_response> 包裹完整性由调用方的
    limit 选择保证)。"""
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return text[:head] + "\n[... output truncated ...]\n" + text[-tail:]
