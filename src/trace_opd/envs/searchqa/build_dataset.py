"""SearchQA dataset builder — Search-R1 官方数据 → 我们的 parquet schema。

源(HF `PeterJinGo/nq_hotpotqa_train`,2026-08-12 实测 schema:顶层含干净
的 question/golden_answers 列,无需剥其 <think>/<search> prompt 模板):
  train.parquet  169,615 行(nq 79,168 + hotpotqa 90,447)
  test.parquet    51,713 行(nq 3,610 / triviaqa 11,313 / popqa 14,267 /
                  hotpotqa 7,405 / 2wikimultihopqa 12,576 / musique 2,417 /
                  bamboogle 125)——七源与 arXiv 2503.09516 评测集逐字吻合

输出:
  train.parquet — data_source="searchqa"
  valid.parquet — data_source="searchqa/<src>"(--val-per-source 分层抽样,
                  seed 固定;val 是训练期曲线用,全量评测另用 test 输出)
  test.parquet  — 七源全量(--emit-test 时;终评对拍 Search-R1 Table 1 用)
  manifest.json — 行数/抽样参数/源 parquet 指纹

用法:
  python -m trace_opd.envs.searchqa.build_dataset \
      --src-train <nq_hotpotqa>/train.parquet --src-test <...>/test.parquet \
      --out <dir> [--val-per-source 200] [--emit-test] [--skip-train]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ...core import protocol
from .adapter import SYSTEM_PROMPT, initial_obs_of

_TRAIN_SOURCES = {"nq", "hotpotqa"}
_TEST_SOURCES = {
    "nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique",
    "bamboogle",
}


def _record(row: dict, split: str) -> dict:
    src = str(row["data_source"])
    golden = [str(g) for g in list(row["golden_answers"]) if str(g).strip()]
    if not golden:
        raise SystemExit(
            f"FATAL: empty golden_answers for id={row['id']!r} — dataset "
            "schema drifted; refusing to emit unscorable tasks."
        )
    extra_info = {
        "task_id": f"{src}_{row['id']}",
        "task_type": "searchqa" if split == "train" else src,
        "split": split,
        "question": str(row["question"]),
        "golden_answers": golden,
    }
    extra_info["initial_obs"] = initial_obs_of(extra_info)
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": protocol.truncate_obs(extra_info["initial_obs"])},
        ],
        "data_source": "searchqa" if split == "train" else f"searchqa/{src}",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(golden, ensure_ascii=False)},
        "extra_info": extra_info,
    }


def _fingerprint(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        h.update(str(path.stat().st_size).encode())
    return h.hexdigest()[:12]


def _check_sources(df: pd.DataFrame, expected: set[str], which: str) -> None:
    got = set(df["data_source"].unique())
    if got != expected:
        raise SystemExit(
            f"FATAL: {which} data_source set drifted — got {sorted(got)}, "
            f"expected {sorted(expected)}."
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-train", required=False, default="")
    ap.add_argument("--src-test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-per-source", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--emit-test", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"seed": args.seed, "val_per_source": args.val_per_source}

    if not args.skip_train:
        if not args.src_train:
            raise SystemExit("FATAL: --src-train required unless --skip-train")
        src = Path(args.src_train)
        df = pd.read_parquet(src)
        _check_sources(df, _TRAIN_SOURCES, "train")
        recs = [_record(row, "train") for row in df.to_dict("records")]
        pd.DataFrame(recs).to_parquet(out / "train.parquet")
        manifest["train"] = {
            "rows": len(recs),
            "by_source": df["data_source"].value_counts().to_dict(),
            "src_fingerprint": _fingerprint(src),
        }
        print(f"[searchqa] train: {len(recs)} rows")

    src = Path(args.src_test)
    df = pd.read_parquet(src)
    _check_sources(df, _TEST_SOURCES, "test")

    val_parts = []
    for name, gdf in df.groupby("data_source"):
        n = min(args.val_per_source, len(gdf))
        val_parts.append(gdf.sample(n=n, random_state=args.seed))
    vdf = pd.concat(val_parts)
    recs = [_record(row, "valid") for row in vdf.to_dict("records")]
    pd.DataFrame(recs).to_parquet(out / "valid.parquet")
    manifest["valid"] = {
        "rows": len(recs),
        "by_source": vdf["data_source"].value_counts().to_dict(),
        "src_fingerprint": _fingerprint(src),
    }
    print(f"[searchqa] valid: {len(recs)} rows ({args.val_per_source}/source cap)")

    if args.emit_test:
        recs = [_record(row, "test") for row in df.to_dict("records")]
        pd.DataFrame(recs).to_parquet(out / "test.parquet")
        manifest["test"] = {
            "rows": len(recs),
            "by_source": df["data_source"].value_counts().to_dict(),
        }
        print(f"[searchqa] test: {len(recs)} rows (full seven-benchmark set)")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[searchqa] manifest -> {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
