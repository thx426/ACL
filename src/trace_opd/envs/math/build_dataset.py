"""Math dataset builder — DAPO-Math-17k(train)+ 六竞赛评测源(valid)。

训练源坑(2026-08-07 实查):HF `BytedTsinghua-SIA/DAPO-Math-17k` 实际
1,791,700 行 = 每题重复展开版 ⇒ **按题面文本 hash 去重**(extra_info.index
不可信)→ ~17k 唯一题;prompt 是固定三段模板,剥离锚点不匹配即 fail loud
(防数据版本漂移)。用户裁定:17k 全量。

valid 六源(逐 schema 适配,`data_source=math/<bench>` 分桶 val 曲线;
用户裁定:全量 greedy@1 口径,终局 avg@K 走 run_eval --n):

  aime24    math-ai/aime24        problem / solution("\\boxed{204}" 剥箱)
  aime25    math-ai/aime25        同 aime24 系(载入时自适应)
  aime26    MathArena/aime_2026   problem / answer(split=train)
  amc23     math-ai/amc23         question / answer
  math500   HuggingFaceH4/MATH-500  problem / answer(latex 表达式)
  olympiad  math-ai/olympiadbench question / final_answer(列表;多元素
                                  答案题跳过并计数——submit 单串语义)

运行(需 datasets 库与网络/本地缓存;由 CD-006 在数据环境执行):
  python -m trace_opd.envs.math.build_dataset --out <dir> \
    [--train-src BytedTsinghua-SIA/DAPO-Math-17k] [--skip-train]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from ...core import protocol
from .adapter import SYSTEM_PROMPT, initial_obs_of

_DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: $Answer (without quotes) where "
    "$Answer is the answer to the problem.\n\n"
)
_DAPO_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'

_BOXED_RE = re.compile(r"\\boxed\{(.+)\}\s*$", re.DOTALL)


def strip_dapo_template(content: str) -> str:
    """剥固定三段模板取题面;锚点不匹配 fail loud(数据版本漂移可见)。"""
    if not (content.startswith(_DAPO_PREFIX) and content.endswith(_DAPO_SUFFIX)):
        raise SystemExit(
            "FATAL: DAPO prompt template anchors did not match — the dataset "
            f"version drifted. Got head={content[:80]!r} tail={content[-60:]!r}"
        )
    return content[len(_DAPO_PREFIX):-len(_DAPO_SUFFIX)].strip()


def _row(problem: str, ground_truth: str, task_id: str, task_type: str, split: str) -> dict:
    extra_info = {
        "task_id": task_id,
        "task_type": task_type,
        "split": split,
        "problem": problem,
        "ground_truth": ground_truth,
    }
    extra_info["initial_obs"] = initial_obs_of(extra_info)
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": protocol.truncate_obs(extra_info["initial_obs"])},
        ],
        "data_source": "math" if split == "train" else f"math/{task_type}",
        "agent_name": "trace_opd",
        "reward_model": {"style": "env", "ground_truth": ""},
        "extra_info": extra_info,
    }


def build_train(src: str) -> list[dict]:
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(src, split="train", streaming=True)
    rows: list[dict] = []
    seen: set[str] = set()
    scanned = 0
    for rec in ds:
        scanned += 1
        problem = strip_dapo_template(rec["prompt"][0]["content"])
        h = hashlib.sha1(problem.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        gt = str(rec["reward_model"]["ground_truth"]).strip()
        if not gt:
            continue
        rows.append(_row(problem, gt, f"dapo/{h[:12]}", "math", "train"))
    print(f"[math] train: scanned {scanned} rows -> {len(rows)} unique problems")
    return rows


#: (bench, hf_id, split, problem_field, answer_extractor)
_VALID_SOURCES = [
    ("aime24", "math-ai/aime24", "test", None, None),
    ("aime25", "math-ai/aime25", "test", None, None),
    ("aime26", "MathArena/aime_2026", "train", None, None),
    ("amc23", "math-ai/amc23", "test", None, None),
    ("math500", "HuggingFaceH4/MATH-500", "test", None, None),
    ("olympiad", "math-ai/olympiadbench", "test", None, None),
]


def _extract_answer(bench: str, rec: dict) -> str | None:
    """逐源答案抽取;None = 跳过该题(记录计数)。"""
    if "final_answer" in rec and rec["final_answer"] is not None:   # olympiad
        fa = rec["final_answer"]
        if isinstance(fa, list):
            if len(fa) != 1:
                return None            # 多元素答案:submit 单串语义,跳过
            return str(fa[0]).strip()
        return str(fa).strip()
    if rec.get("answer") is not None and str(rec.get("answer")).strip():
        return str(rec["answer"]).strip()
    sol = str(rec.get("solution") or "").strip()   # aime24/25:\boxed{...}
    m = _BOXED_RE.search(sol)
    if m:
        return m.group(1).strip()
    return None


def build_valid() -> list[dict]:
    from datasets import load_dataset  # noqa: PLC0415

    rows: list[dict] = []
    for bench, hf_id, split, _pf, _ax in _VALID_SOURCES:
        ds = load_dataset(hf_id, split=split)
        kept, skipped = 0, 0
        for i, rec in enumerate(ds):
            problem = str(rec.get("problem") or rec.get("question") or "").strip()
            if not problem:
                skipped += 1
                continue
            gt = _extract_answer(bench, dict(rec))
            if gt is None or not gt:
                skipped += 1
                continue
            rows.append(_row(problem, gt, f"{bench}/{i}", bench, "valid"))
            kept += 1
        print(f"[math] valid/{bench}: kept {kept}, skipped {skipped}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-src", default="BytedTsinghua-SIA/DAPO-Math-17k")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        train_rows = build_train(args.train_src)
        pd.DataFrame(train_rows).to_parquet(out / "train.parquet")
        print(f"[math] train -> {out / 'train.parquet'} ({len(train_rows)})")
    valid_rows = build_valid()
    pd.DataFrame(valid_rows).to_parquet(out / "valid.parquet")
    print(f"[math] valid -> {out / 'valid.parquet'} ({len(valid_rows)})")
    (out / "manifest.json").write_text(json.dumps({
        "train_src": None if args.skip_train else args.train_src,
        "dedupe": "sha1(problem) unique",
        "valid_sources": [(b, h, s) for b, h, s, _, _ in _VALID_SOURCES],
        "counts": {"valid": len(valid_rows)},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
