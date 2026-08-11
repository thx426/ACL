"""SWE 域(R2E-Gym-Subset)数据集构建 —— HF parquet → verl parquet + 特权分离。

数据源:R2E-Gym-Subset(arXiv:2504.07164,Apache-2.0),4,578 实例 / 10 repos,
上游已 repo 级去污染(剔除 SWE-bench 测试仓,sympy=0 物证)。每实例携带:
  - docker_image:环境镜像(= 数据源分派唯一依据,但我们改用显式 source 字段,
    禁止镜像名字符串嗅探 —— 深读 R1)
  - problem_statement:issue(带 [ISSUE]..[/ISSUE] 壳,构建期剥;剥壳正则失配
    则 fail-loud,不沿用上游"原样返回含标签"的静默行为)
  - expected_output_json:判分基准({test: status},judge 侧读,不进 agent 面)
  - parsed_commit_content:GT patch(特权 L4)/ modified_* / relevant_files:定位特权

信息状态红线(SSB/SRL 同款):判分基准与 GT patch **不进 parquet 的 agent 可见
字段**,落 privileged/<instance_id>.json;parquet 的 prompt 只含 system + issue。
`prompt` 列(反译 issue 用的提示词)训练不用,丢弃。

pilot 子集化(用户裁定范围):5 个小镜像 repo(coveragepy/tornado/scrapy/
pyramid/aiohttp),每 repo 取前 N 条(按 instance_id 排序,确定性)。

运行(本机):
  experiments/bench_env/.venv-webarena/bin/python -m trace_opd.envs.swe.build_dataset \
    --raw experiments/bench_env/data/swe/raw/subset_all.parquet \
    --out experiments/bench_env/data/swe/dataset \
    --pilot --per-repo 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from .prompts import SYSTEM_PROMPT, build_instruction

_ISSUE_RE = re.compile(r"\[ISSUE\](.*)\[/ISSUE\]", re.DOTALL)

#: pilot 子集:5 个最小镜像 repo(压缩 0.31-0.50GB,纯 Python 无 xvfb)。
#: orange3 需 xvfb 排除;pandas/numpy 体积大排除(扩量期收编)。
PILOT_REPOS = ("coveragepy", "tornado", "scrapy", "pyramid", "aiohttp")

def strip_issue(problem_statement: str, instance_id: str) -> str:
    """剥 [ISSUE]..[/ISSUE] 壳,与官方 get_task_instruction 同语义
    (docker.py:1003-1009):有标签取内层,无标签用整段。

    实测(2026-08-11 全量 4,578):4,518 带标签、60 不带(1.3%,分散 9 repo)——
    不带标签者内容仍是合法 issue(`**Title:**..**Description:**`),官方对其
    原样使用。故无标签**不是**数据异常,不 fail-loud;但仍防真空:剥出的
    正文非空且不含未配对标签残留。"""
    m = _ISSUE_RE.search(problem_statement)
    issue = m.group(1).strip() if m else problem_statement.strip()
    if not issue:
        raise ValueError(f"instance {instance_id}: empty problem_statement")
    if "[ISSUE]" in issue or "[/ISSUE]" in issue:
        raise ValueError(
            f"instance {instance_id}: unpaired ISSUE tag leaked into prompt"
        )
    return issue


def instance_id_of(row: dict) -> str:
    """稳定实例 id = docker_image 尾段(与 R2E trajectory.instance_name 同源:
    `namanjain12/orange3_final:<hash>` → `<repo>_final:<hash>` 的 `.` split
    尾段;这里直接取 image 冒号后 hash 前拼 repo,保证跨机稳定+可读)。"""
    img = str(row["docker_image"])
    tag = img.split(":")[-1]
    return f"{row['repo_name']}__{tag}"


def _to_list(v) -> list:
    if v is None:
        return []
    try:
        return [str(x) for x in list(v)]
    except TypeError:
        return [str(v)]


def make_row(row: dict, split: str) -> tuple[dict, dict]:
    """(parquet 行, 特权 json)。特权(GT patch/期望输出/定位)绝不进 parquet
    的 agent 可见字段。"""
    iid = instance_id_of(row)
    issue = strip_issue(str(row["problem_statement"]), iid)
    instruction = build_instruction(issue)
    extra_info = {
        "instance_id": iid,
        "repo_name": str(row["repo_name"]),
        "docker_image": str(row["docker_image"]),
        "commit_hash": str(row["commit_hash"]),
        "source": "r2e",  # 显式数据源(禁止镜像名嗅探,深读 R1)
        "split": split,
        "num_non_test_files": int(row["num_non_test_files"]),
        "num_non_test_func_methods": int(row["num_non_test_func_methods"]),
        "num_non_test_lines": int(row["num_non_test_lines"]),
    }
    parquet_row = {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ],
        "data_source": "swe_r2e" if split == "train" else f"swe_r2e/{split}",
        "agent_name": "trace_opd",
        "reward_model": {"style": "env", "ground_truth": ""},
        "extra_info": extra_info,
    }
    privileged = {
        "instance_id": iid,
        "expected_output_json": str(row["expected_output_json"]),  # judge 用
        "parsed_commit_content": str(row["parsed_commit_content"]),  # GT patch (L4)
        "modified_files": _to_list(row["modified_files"]),  # 定位特权
        "modified_entity_summaries": (
            json.loads(row["modified_entity_summaries"])
            if isinstance(row["modified_entity_summaries"], str)
            else _entity_to_jsonable(row["modified_entity_summaries"])
        ),
        "relevant_files": _to_list(row["relevant_files"]),  # L1
    }
    return parquet_row, privileged


def _entity_to_jsonable(v):
    """modified_entity_summaries 是 ndarray[dict];转可 json 化 list。
    (ndarray 不能用 `v or []` 判真值 —— 歧义;显式 None 检查。)"""
    if v is None:
        return []
    out = []
    for e in list(v):
        out.append({k: (v2.item() if hasattr(v2, "item") else v2)
                    for k, v2 in dict(e).items()})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="subset_all.parquet(合并后)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pilot", action="store_true", help="仅 PILOT_REPOS 子集")
    ap.add_argument("--per-repo", type=int, default=0,
                    help=">0 时每 repo 取前 N 条(按 instance_id 排序)")
    args = ap.parse_args()

    df = pd.read_parquet(args.raw)
    print(f"[swe] raw: {len(df)} instances, repos={df.repo_name.nunique()}")
    if args.pilot:
        df = df[df.repo_name.isin(PILOT_REPOS)].copy()
        print(f"[swe] pilot repos {PILOT_REPOS}: {len(df)}")

    # 稳定排序 + 每 repo 截断
    rows = [dict(r) for _, r in df.iterrows()]
    rows.sort(key=instance_id_of)
    if args.per_repo > 0:
        by_repo: dict[str, list] = {}
        for r in rows:
            by_repo.setdefault(r["repo_name"], []).append(r)
        rows = []
        for repo in sorted(by_repo):
            rows.extend(by_repo[repo][: args.per_repo])
        print(f"[swe] per-repo={args.per_repo}: {len(rows)} instances")

    out = Path(args.out)
    priv_dir = out / "privileged"
    priv_dir.mkdir(parents=True, exist_ok=True)
    parquet_rows = []
    fams: dict[str, int] = {}
    for r in rows:
        prow, priv = make_row(r, "train")
        parquet_rows.append(prow)
        (priv_dir / f"{priv['instance_id']}.json").write_text(
            json.dumps(priv, ensure_ascii=False), encoding="utf-8"
        )
        fams[r["repo_name"]] = fams.get(r["repo_name"], 0) + 1

    path = out / "train.parquet"
    pd.DataFrame(parquet_rows).to_parquet(path)
    # 校验:agent 可见字段绝无 golden 关键词
    _leak_check(parquet_rows)
    manifest = {
        "source": "R2E-Gym-Subset (arXiv:2504.07164, Apache-2.0)",
        "n": len(parquet_rows),
        "by_repo": fams,
        "pilot": bool(args.pilot),
        "per_repo": args.per_repo,
        "privileged_dir": "privileged/<instance_id>.json (GT patch/expected/locate)",
        "decontam": "上游 repo 级(SWE-bench 测试仓已剔);测试集 Verified 12 repo 与本 10 repo 零重叠",
        "sha256_instances": hashlib.sha256(
            "\n".join(sorted(p["extra_info"]["instance_id"] for p in parquet_rows)).encode()
        ).hexdigest(),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[swe] train: {len(parquet_rows)} -> {path}")
    print(f"[swe]   by_repo: {fams}")
    print(f"[swe]   privileged: {len(parquet_rows)} files -> {priv_dir}")
    print(f"[swe] manifest -> {out / 'manifest.json'}")
    return 0


def _leak_check(parquet_rows: list[dict]) -> None:
    """golden 红线自检:parquet 的 prompt/extra_info 里不得含 GT patch 迹象。"""
    banned = ("parsed_commit_content", "expected_output_json", "old_file_content",
              "FAIL_TO_PASS", "PASSED\",")
    for p in parquet_rows:
        blob = json.dumps(p, ensure_ascii=False)
        for b in banned:
            if b in blob:
                raise AssertionError(
                    f"LEAK: {p['extra_info']['instance_id']} parquet contains {b!r}"
                )


if __name__ == "__main__":
    raise SystemExit(main())
