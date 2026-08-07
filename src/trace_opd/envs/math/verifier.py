"""Answer equivalence for math tasks — math-verify (HF, RLVR community
standard) with an exact-match fallback.

实测封装策略(2026-08-07,本机验证):
  * 裸 latex 元组 ``\\left(3, \\frac{\\pi}{2}\\right)`` 直接 parse 会丢元素;
    包 ``$...$`` 定界后正确解析 ⇒ 两侧都做「原样 + $ 包裹」的组合 parse,
    任一组合 verify 为真即判对;
  * ``parse('$204$')`` vs 自由文本/boxed 均可抽取;submit(answer) 的参数
    本就是短答案字符串,歧义面最小;
  * math-verify 解析失败(空 parse)时回退精确字符串比较(strip 后相等)。

失败闭合:任何异常按不等价处理(评分绝不 crash episode);math-verify
未安装在装配期 fail loud(工厂检查——静默降级成纯字符串比对会把表达式
答案题系统性判错,是评估语义漂移)。
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _parse_variants(text: str):
    from math_verify import parse  # noqa: PLC0415 — heavy import stays lazy

    variants = []
    seen: set[str] = set()
    for candidate in (text, f"${text}$"):
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = parse(candidate)
        except Exception:  # noqa: BLE001 — 解析崩溃按无结果
            parsed = []
        if parsed:
            variants.append(parsed)
    return variants


def answers_equivalent(gold: str, answer: str) -> bool:
    """math-verify 组合判定 + 精确匹配 fallback。gold 来自数据集
    ground_truth,answer 来自 submit(answer) 参数。"""
    gold = (gold or "").strip()
    answer = (answer or "").strip()
    if not gold or not answer:
        return False
    if gold == answer:
        return True
    try:
        from math_verify import verify  # noqa: PLC0415

        for g in _parse_variants(gold):
            for a in _parse_variants(answer):
                try:
                    if verify(g, a):
                        return True
                except Exception:  # noqa: BLE001 — 单组合崩溃继续
                    continue
    except ImportError:
        logger.error(
            "math-verify not installed — falling back to exact match only "
            "(expression answers will be systematically misgraded)"
        )
    return False


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
