"""Answer equivalence for math tasks — math-verify (HF, RLVR community
standard) run in a **timeout-killed subprocess**.

实测封装策略(2026-08-07,本机验证;子进程化 2026-08-13):
  * 裸 latex 元组 ``\\left(3, \\frac{\\pi}{2}\\right)`` 直接 parse 会丢元素;
    包 ``$...$`` 定界后正确解析 ⇒ 两侧都做「原样 + $ 包裹」的组合 parse,
    任一组合 verify 为真即判对;
  * ``parse('$204$')`` vs 自由文本/boxed 均可抽取;答案本就是短字符串,
    歧义面最小;精确匹配走进程内快路径,不起子进程。

为什么必须子进程(2026-08-13 冒烟实锤):react 跑在 asyncio.to_thread 工作
线程,math-verify 的 signal 超时只能装在主线程 ⇒ 此前只能
``parsing_timeout=None`` 裸跑 + 2000 字符长度护栏。但**短表达式也能让
sympy 无限自旋**(temp 1.0 采样撞上病态式):GRPO 冒烟一个
AgentLoopWorker 286% CPU 满核自旋 35+ 分钟、7 个 task 43 个 episode 被拖
死、整臂无法收尾。线程杀不死、signal 装不上 ⇒ 唯一干净解 = 子进程 + 硬
超时击杀(与 run_python 同模式)。超时按不等价处理并告警(fail-closed,
与「任何异常按不等价」一致;病态式作为答案本就不该判对)。

失败闭合:任何异常/超时按不等价处理(评分绝不 crash episode);
math-verify 未安装在装配期 fail loud(工厂检查——静默降级成纯字符串比对
会把表达式答案题系统性判错,是评估语义漂移)。
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

#: 子进程侧完整判定逻辑(自包含;组合 parse + verify 语义与历史版逐字等价)。
_CHILD_SRC = r"""
import json, sys
try:
    from math_verify import parse, verify
except Exception:
    print("E:import"); raise SystemExit(0)
data = json.loads(sys.stdin.read())
gold, answer = data["gold"], data["answer"]
def variants(text):
    out, seen = [], set()
    for cand in (text, "$" + text + "$"):
        if cand in seen:
            continue
        seen.add(cand)
        try:
            p = parse(cand, parsing_timeout=None)
        except Exception:
            p = []
        if p:
            out.append(p)
    return out
ok = False
for g in variants(gold):
    for a in variants(answer):
        try:
            if verify(g, a, timeout_seconds=None):
                ok = True
                break
        except Exception:
            continue
    if ok:
        break
print("1" if ok else "0")
"""


def answers_equivalent(
    gold: str,
    answer: str,
    *,
    timeout_s: float = 20.0,
    gate=None,
) -> bool:
    """math-verify 等价判定(子进程 + 硬超时)。gold 来自数据集
    ground_truth,answer 来自终答通道(\\boxed 抽取或 submit 参数)。
    ``gate``(threading.Semaphore)限制并发子进程数——调用方通常复用
    run_python 的 exec 门。"""
    gold = (gold or "").strip()
    answer = (answer or "").strip()
    if not gold or not answer:
        return False
    if gold == answer:          # 进程内快路径(多数整数答案零开销)
        return True
    # 长度护栏保留:超长输入连子进程都不必起(合法短答案 <100 字符)。
    if len(gold) > 2000 or len(answer) > 2000:
        return False
    payload = json.dumps({"gold": gold, "answer": answer})
    ctx = gate if gate is not None else contextlib.nullcontext()
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    with ctx:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _CHILD_SRC],
                input=payload,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "math-verify subprocess timed out after %.0fs — graded "
                "not-equivalent (gold=%.60r answer=%.60r)",
                timeout_s, gold, answer,
            )
            return False
        except Exception as exc:  # noqa: BLE001 — 评分绝不 crash episode
            logger.warning("math-verify subprocess failed (%s) — graded not-equivalent", exc)
            return False
    out = (proc.stdout or "").strip()
    if out == "E:import":
        logger.error(
            "math-verify not importable in verifier subprocess — expression "
            "answers will be systematically misgraded"
        )
        return False
    return out.endswith("1")


def ensure_math_verify() -> None:
    """装配期依赖检查(工厂调用;fail loud)。"""
    try:
        import math_verify  # noqa: F401, PLC0415
    except ImportError as err:
        raise ValueError(
            "math benchmark requires the 'math-verify' package for answer "
            "equivalence grading (pip install math-verify) — running without "
            "it would silently misgrade expression answers"
        ) from err
