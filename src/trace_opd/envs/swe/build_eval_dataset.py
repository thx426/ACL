"""SWE-bench Verified(评测集)数据集构建 —— HF → verl parquet + 特权分离。

数据源:`R2E-Gym/SWE-Bench-Verified`(500,test split)。选它而非
`princeton-nlp/SWE-bench_Verified` 的理由(实测 2026-08-11):字段是标准
SWE-bench 全套(repo/base_commit/patch/test_patch/F2P/P2P/version/
environment_setup_commit),**另带两个便利字段**——`docker_image`
(`slimshetty/swebench-verified:sweb.eval.x86_64.<instance_id>`,单仓 500 tag,
比官方 500 个独立 repo 更易镜像到 ACR)与 `run_tests`(旧版 eval 脚本,
仅作回退)。instance_id 与官方逐字一致 ⇒ 导出的 predictions 可直接喂官方
harness。

**prompt 与训练侧同一真源**(prompts.py):任务形态相同,prompt 漂移会让
评测读数不可归因。

信息状态红线:`patch`(GT 修复)、`test_patch`(F2P 测试本体!)、F2P/P2P
清单**绝不进 parquet**,落 privileged/<instance_id>.json。Verified 的评测
语义本就要求 agent 看不到测试(harness 判分前才打入 test_patch),这条红线
同时也是评测口径本身。

运行:
  python -m trace_opd.envs.swe.build_eval_dataset \
    --out experiments/bench_env/data/swe/eval [--limit 50]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .build_dataset import strip_issue
from .prompts import SYSTEM_PROMPT, build_instruction

HF_PARQUET = (
    "https://huggingface.co/api/datasets/R2E-Gym/SWE-Bench-Verified/"
    "parquet/default/test/0.parquet"
)
#: Verified 官方规模(SWE-bench Verified = OpenAI 人工复核的 500 条)。
EXPECTED_N = 500


def make_row(row: dict) -> tuple[dict, dict]:
    iid = str(row["instance_id"])
    issue = strip_issue(str(row["problem_statement"]), iid)
    extra_info = {
        "instance_id": iid,
        "repo": str(row["repo"]),
        "repo_name": str(row["repo"]).split("/")[-1],  # family(留一检验口径)
        "base_commit": str(row["base_commit"]),
        "version": str(row["version"]),
        "docker_image": str(row["docker_image"]),
        "source": "swebench",     # 显式数据源(禁止镜像名嗅探,深读 R1)
        "split": "test",
    }
    parquet_row = {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_instruction(issue)},
        ],
        "data_source": "swe_verified",
        "agent_name": "trace_opd",
        "reward_model": {"style": "env", "ground_truth": ""},
        "extra_info": extra_info,
    }
    privileged = {
        "instance_id": iid,
        "patch": str(row["patch"]),                 # GT 修复(特权 L4)
        "test_patch": str(row["test_patch"]),       # F2P 测试本体(判分侧)
        "FAIL_TO_PASS": str(row["FAIL_TO_PASS"]),
        "PASS_TO_PASS": str(row["PASS_TO_PASS"]),
        "environment_setup_commit": str(row["environment_setup_commit"]),
        "hints_text": str(row.get("hints_text", "") or ""),  # PR 前真人讨论
        "run_tests": str(row.get("run_tests", "") or ""),    # 旧版脚本(回退)
    }
    return parquet_row, privileged


def _leak_check(pairs: list[tuple[dict, dict]]) -> list[str]:
    """agent 可见面不得含 GT patch 正文 / 测试本体 / 测试清单。

    **不能**按 `diff --git` 关键词粗判:实测 500 条里有 5 条的 issue 原文
    就带作者自己贴的补丁提议(如 django__django-13410),那是 SWE-bench
    Verified 的固有属性、官方口径的一部分,剥掉反而偏离评测口径。故按
    per-instance 精确比对 GT patch / test_patch 正文。返回"issue 自带 diff"
    的实例清单(记入 manifest,供分析时知情)。"""
    self_diff: list[str] = []
    for row, priv in pairs:
        iid = row["extra_info"]["instance_id"]
        prompt = json.dumps(row["prompt"], ensure_ascii=False)
        for field in ("patch", "test_patch"):
            body = str(priv.get(field, ""))
            if len(body) > 200 and body[:200] in prompt:
                raise AssertionError(f"LEAK: {iid} prompt contains {field} body")
        for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            if key in prompt:
                raise AssertionError(f"LEAK: {iid} prompt mentions {key}")
        if "diff --git" in prompt:
            self_diff.append(iid)   # 非泄漏:issue 作者自贴的补丁提议
    return self_diff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default="", help="本地 parquet(默认从 HF 拉)")
    ap.add_argument("--limit", type=int, default=0,
                    help=">0 时只取前 N 条(按 instance_id 排序;pilot 用)")
    args = ap.parse_args()

    src = args.raw or HF_PARQUET
    df = pd.read_parquet(src)
    print(f"[swe-eval] source: {src}\n[swe-eval] rows: {len(df)}")
    if not args.raw and len(df) != EXPECTED_N:
        raise SystemExit(
            f"FATAL: SWE-bench Verified 应为 {EXPECTED_N} 条,拿到 {len(df)} "
            "—— 数据源漂移,停(评测集规模是报表口径的一部分)"
        )
    rows = sorted((dict(r) for _, r in df.iterrows()),
                  key=lambda r: str(r["instance_id"]))
    if args.limit > 0:
        rows = rows[: args.limit]
        print(f"[swe-eval] limit={args.limit}")

    out = Path(args.out)
    priv_dir = out / "privileged"
    priv_dir.mkdir(parents=True, exist_ok=True)
    parquet_rows, fams, pairs = [], {}, []
    for r in rows:
        prow, priv = make_row(r)
        parquet_rows.append(prow)
        pairs.append((prow, priv))
        (priv_dir / f"{priv['instance_id']}.json").write_text(
            json.dumps(priv, ensure_ascii=False), encoding="utf-8")
        fam = prow["extra_info"]["repo_name"]
        fams[fam] = fams.get(fam, 0) + 1
    self_diff = _leak_check(pairs)

    path = out / "test.parquet"
    pd.DataFrame(parquet_rows).to_parquet(path)
    (out / "manifest.json").write_text(json.dumps({
        "source": "R2E-Gym/SWE-Bench-Verified (500, test)",
        "n": len(parquet_rows),
        "by_repo": fams,
        "images": "slimshetty/swebench-verified:sweb.eval.x86_64.<instance_id>",
        "grading": ("官方口径 F2P∧P2P → resolved;默认 deferred(导出 "
                    "predictions.jsonl 交官方 harness),inline 可选"),
        "privileged_dir": "privileged/<instance_id>.json (patch/test_patch/F2P/P2P)",
        "prompt_source": "envs/swe/prompts.py(与训练侧同一真源)",
        "issue_contains_own_diff": self_diff,   # issue 原文自带补丁提议(官方固有)
        "sha256_instances": hashlib.sha256(
            "\n".join(r["extra_info"]["instance_id"] for r in parquet_rows).encode()
        ).hexdigest(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[swe-eval] test: {len(parquet_rows)} -> {path}")
    print(f"[swe-eval]   by_repo(top5): "
          f"{dict(sorted(fams.items(), key=lambda kv: -kv[1])[:5])}")
    print(f"[swe-eval]   privileged -> {priv_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
