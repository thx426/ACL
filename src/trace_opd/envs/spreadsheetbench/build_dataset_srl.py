"""Spreadsheet-RL dataset builder — excelforum 5,925 → verl parquet.

数据源:Spreadsheet-RL(arXiv:2605.22642,CC BY-SA 4.0)`spreadsheets.zip`
解包后的 `excelforum/<subforum>/<task-job-id>/` 目录树,每任务
`instruction.json`(instruction/type/answer_position)+ `input.xlsx`(起点)
+ `target.xlsx`(golden,Excel 重算后带缓存值——官方 sample_meta.py 只认
target)+ `output.xlsx`(oracle 中间产物,判分不用,本构建器忽略)。

与 SSB(verified-400,测试集)共用同一 adapter/evaluator/recalc —— 训练
(本数据)与测试(SSB)同一把尺子;单 test-case 下 soft 口径天然退化为
{0,1} 二值 reward。

剔除链(全部记入 manifest,顺序固定):
  1. probe 门:`srl_probe.py` 产出的 g1_fail(golden 自评不过 ⇒ 判分
     机制对该任务不健康)与 g2_trivial(input 即满足判分 ⇒ 退化任务,
     GRPO 组内全 1 零梯度/虚高 val);
  2. oracle 门:`srl_oracle_smoke.py --all` 产出的生产链 unreliable
     (golden 过完整评分链——含 LibreOffice 重算——自比 ≠1.0;SSB
     unreliable_tasks 同款纪律);
  3. decontam 门:对 SSB(测试集)的泄漏剔除清单(工作簿内容指纹 +
     指令文本三信号检测,`srl_decontam.py` 产出 + 人工裁定)。
剔除发生在**划分之后**(与 SSB build_dataset 同纪律:保持 seed42 落位
稳定,后续增删清单时 train/valid 仍是原划分子集,曲线可比)。

运行(数据所在机器上;openpyxl 版本必须与训练运行一致——initial_obs
逐字节红线):
  python -m trace_opd.envs.spreadsheetbench.build_dataset_srl \
    --data-root experiments/bench_env/data/spreadsheet_rl/task-data \
    --probe experiments/bench_env/runs/spreadsheet_rl_probe/golden-selfcheck-<ts>.json \
    --exclusions experiments/bench_env/data/spreadsheet_rl/decontam/exclusions.json \
    --out experiments/bench_env/data/spreadsheet_rl/dataset
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

from .build_dataset import make_row

_MISSING_LEAD_QUOTE = re.compile(r"^([^'!]+)'!(.+)$")

#: valid 的 in-domain held-out 大小(监控「学到 excelforum 分布」;
#: 跨分布主读数另由 SSB valid 提供,verl data.val_files 配置层组合)
DEFAULT_HELD_OUT = 128


def normalize_answer_position(ap: str) -> str:
    """修数据侧系统性 typo:段首缺前引号(实测 2026-08-11 全量 5,925:
    828 段级修复 = 668 单区域 `Sheet1'!C3:C545` + 128 多区域首段 + 少量
    多区域后段)。以 `'` 开头的段(含 sheet 名带合法撇号,如
    `'RYAN'S ORDERS'!M6:M42`)不动——那不是 typo,靠 evaluator 的奇引号
    回退解析。多区域按裸逗号切:缺引号段的 sheet 名实测均不含逗号,
    含逗号 sheet 名的段必有完好引号对,裸切安全(probe G1 兜底)。"""
    parts = []
    for seg in ap.split(","):
        s = seg.strip()
        if s and not s.startswith("'"):
            m = _MISSING_LEAD_QUOTE.match(s)
            if m:
                s = f"'{m.group(1)}'!{m.group(2)}"
        parts.append(s)
    return ",".join(parts)


def load_items(data_root: Path) -> tuple[list[dict], list[str]]:
    """excelforum 目录树 → (item 列表, bad_schema 剔除 ids)。
    id = task-job 目录名,实测全局唯一(冲突 fail loud)。instruction 必须
    是非空 str(实测 2026-08-11:task-job-bd4a53b230f1c2cf 的 instruction
    是 bool true——指令缺失的脏样本,探针只查判分抓不到,在此剔)。"""
    items: list[dict] = []
    bad_schema: list[str] = []
    seen: dict[str, str] = {}
    for ins_path in sorted((data_root / "excelforum").rglob("instruction.json")):
        task_dir = ins_path.parent
        rel = task_dir.relative_to(data_root).as_posix()
        tid = task_dir.name
        if tid in seen:
            raise SystemExit(
                f"FATAL: duplicate task id {tid}: {seen[tid]} vs {rel}"
            )
        seen[tid] = rel
        d = json.loads(ins_path.read_text(encoding="utf-8"))
        instr = d.get("instruction")
        ap = d.get("answer_position")
        if (not isinstance(instr, str) or not instr.strip()
                or not isinstance(ap, str) or not ap.strip()):
            print(f"[srl] bad schema, drop {tid}: "
                  f"instruction={type(instr).__name__} "
                  f"answer_position={type(ap).__name__}")
            bad_schema.append(tid)
            continue
        items.append(
            {
                "id": tid,
                "instruction": instr,
                "instruction_type": str(d.get("type", "")),
                "answer_position": normalize_answer_position(ap),
                "spreadsheet_path": rel,
            }
        )
    if not items:
        raise SystemExit(f"FATAL: no tasks under {data_root}/excelforum")
    return items, bad_schema


def load_probe(path: Path) -> tuple[set[str], set[str]]:
    """srl_probe 结果 → (g1_fail ids, g2_trivial ids);task 字段是
    `excelforum/<subforum>/<task-job-id>` 相对路径,取尾段。"""
    doc = json.loads(path.read_text(encoding="utf-8"))
    g1 = {r["task"].rsplit("/", 1)[-1] for r in doc.get("g1_fail", [])}
    g2 = {r["task"].rsplit("/", 1)[-1] for r in doc.get("g2_trivial", [])}
    return g1, g2


def load_oracle(path: Path) -> set[str]:
    """srl_oracle_smoke --all 结果 → 生产链 unreliable ids(golden 过完整
    评分链自比 ≠1.0:LibreOffice 重算漂移/pivot 重建/#VALUE 传播等——
    判分器在这些任务上不可信;SSB unreliable_tasks 同款纪律,用户裁定
    2026-08-09「全部剔除」的同构延伸)。"""
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {r["task"] for r in doc.get("fails", [])}


def load_exclusions(paths: list[Path]) -> dict[str, str]:
    """去污染清单(可多份合并):{train_task_id: reason}。文件格式
    {"exclusions": [{"train_id": ..., "reason": ...}, ...]}。"""
    merged: dict[str, str] = {}
    for p in paths:
        doc = json.loads(p.read_text(encoding="utf-8"))
        for e in doc["exclusions"]:
            merged[str(e["train_id"])] = str(e.get("reason", ""))
    return merged


def main() -> int:
    apar = argparse.ArgumentParser()
    apar.add_argument("--data-root", required=True)
    apar.add_argument("--out", required=True)
    apar.add_argument("--probe", required=True,
                      help="srl_probe.py 的 golden-selfcheck json(健康门)")
    apar.add_argument("--oracle", required=True,
                      help="srl_oracle_smoke.py --all 的 oracle-full json"
                           "(生产链 unreliable 门)")
    apar.add_argument("--exclusions", nargs="*", default=[],
                      help="去污染剔除清单 json(可多份;srl_decontam 产出)")
    apar.add_argument("--held-out", type=int, default=DEFAULT_HELD_OUT)
    args = apar.parse_args()

    data_root = Path(args.data_root)
    items, bad_schema = load_items(data_root)
    print(f"[srl] items: {len(items)} (bad_schema dropped: {len(bad_schema)})")

    # 划分先行(seed42 落位与**清单类**剔除解耦,SSB 同纪律;bad_schema
    # 是 load 层固定过滤、不迭代,在 pool 之前剔)
    pool = sorted(items, key=lambda it: str(it["id"]))
    random.Random(42).shuffle(pool)
    valid_part = pool[: args.held_out]
    train_part = pool[args.held_out:]

    g1_fail, g2_trivial = load_probe(Path(args.probe))
    oracle_bad = load_oracle(Path(args.oracle))
    decontam = load_exclusions([Path(p) for p in args.exclusions])
    print(f"[srl] 剔除清单: g1_fail={len(g1_fail)} g2_trivial={len(g2_trivial)} "
          f"oracle_unreliable={len(oracle_bad)} decontam={len(decontam)}")

    dropped: dict[str, list[str]] = {
        "g1_fail": [], "g2_trivial": [], "oracle_unreliable": [], "decontam": [],
    }

    def keep(it: dict) -> bool:
        tid = str(it["id"])
        if tid in g1_fail:
            dropped["g1_fail"].append(tid)
            return False
        if tid in g2_trivial:
            dropped["g2_trivial"].append(tid)
            return False
        if tid in oracle_bad:
            dropped["oracle_unreliable"].append(tid)
            return False
        if tid in decontam:
            dropped["decontam"].append(tid)
            return False
        return True

    train = [it for it in train_part if keep(it)]
    valid = [it for it in valid_part if keep(it)]
    print(f"[srl] train {len(train_part)}→{len(train)}, "
          f"valid {len(valid_part)}→{len(valid)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, part in (("train", train), ("valid", valid)):
        rows = [
            make_row(it, split, str(data_root), source_name="spreadsheet_rl")
            for it in part
        ]
        fams: dict[str, int] = {}
        for r in rows:
            t = r["extra_info"]["instruction_type"]
            fams[t] = fams.get(t, 0) + 1
        path = out / f"{split}.parquet"
        pd.DataFrame(rows).to_parquet(path)
        print(f"[srl] {split}: {len(rows)} tasks -> {path}")
        print(f"[srl]   instruction_type: {fams}")

    manifest = {
        "source": "Spreadsheet-RL (arXiv:2605.22642) spreadsheets.zip / excelforum",
        "license": "CC BY-SA 4.0(ShareAlike:再分发衍生数据须同协议;内部训练无碍)",
        "method": (
            f"walk excelforum -> sort(id) -> random.Random(42).shuffle -> "
            f"valid 前 {args.held_out} / train 其余 -> 划分后剔除"
            "(probe g1_fail + g2_trivial + decontam)"
        ),
        "counts": {"train": len(train), "valid": len(valid)},
        "dropped": {
            "bad_schema": sorted(bad_schema),
            **{k: sorted(v) for k, v in dropped.items()},
        },
        "probe": str(args.probe),
        "oracle": str(args.oracle),
        "exclusions": [str(p) for p in args.exclusions],
        "golden": "target.xlsx(官方 sample_meta.py 同款;output.xlsx 不参与判分)",
        "reward": "SSB evaluator soft 口径;单 case ⇒ 二值 {0,1}",
    }
    (out / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[srl] manifest -> {out / 'split_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
