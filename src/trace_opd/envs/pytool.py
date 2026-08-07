"""Shared sandboxed-Python execution for tool-calling benchmarks.

SSB 与 math 共用的 run_python 执行件(自 spreadsheetbench/adapter 提取):
subprocess 隔离、净化环境(PYTHONHASHSEED=0 保 v4 重放确定性)、超时击杀、
空输出语义。并发门由调用方传入(各 benchmark 的 cfg 定并发额度并做
per-worker 折算;CPU 饥饿纪律:环境子进程不得饿死推理引擎)。
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading


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
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return f"[timeout after {timeout_s:.0f}s]", 124
    out = (proc.stdout or "") + (proc.stderr or "")
    return out if out.strip() else "(no output)", proc.returncode


def clip_middle(text: str, limit: int) -> str:
    """工具输出中部截断(保头尾;<tool_response> 包裹完整性由调用方的
    limit 选择保证)。"""
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return text[:head] + "\n[... output truncated ...]\n" + text[-tail:]
