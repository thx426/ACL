#!/usr/bin/env python
"""SearchQA 训练内 val 集:七源等额分层抽样(2026-08-23)。

动机(用户观察 + 实测证实):旧 val_shuf2048 是 test 池化 shuffle 按比例抽,
bamboogle 只剩 4 条(总量本就 125)、musique 108、nq 131 —— 小源分项曲线
纯噪声,无法体现训练效果;且比例抽样优化的是 micro 估计,而主表口径是
SEED 同构的 per-source 等权 macro-avg,本质失配。

规则(极简可复现):每源目标配额 --per-source(默认 320),不足取全量
(bamboogle=125 全进),不跨源补配;seed 固定;全局重洗后落单文件——
verl 按 data_source 分组上报,分项曲线不依赖文件拆分(ALF seen/unseen 同机制)。
默认产出 6×320+125 = 2045 条 ≈ 旧 2048,单次 val 成本不变。

★ 2026-08-26 事故补丁:首版直接从原始 test.parquet 抽样落盘,**没有补 verl trainer
需要的两个字段**(agent_name='trace_opd' / reward_model.style='env',见
probe/sqa_fix_parquet.py 与记忆 verl-parquet-trainer-fields),SQA7B 第三发在 step0
val 判分处全军 NotImplementedError(七源 ×557 episode)。现把 fix 内建 + 落盘前自检,
从源头堵死;prompt/extra_info 原样不动(判分读 extra_info.golden_answers)。

用法(数据宿主上执行):
  python make_sqa_val7.py --data-dir $TRACE_DATA/classic_searchqa_data \
      [--per-source 320] [--seed 42] [--out val_eq320.parquet]
"""
import argparse
from pathlib import Path

import pandas as pd


def _trainer_schema(d: pd.DataFrame) -> pd.DataFrame:
    """与 probe/sqa_fix_parquet.py 的 fix() 逐字同义:补 verl trainer 两字段 + 列序对齐 ALFWorld。"""
    d = d.copy()
    d["agent_name"] = "trace_opd"
    d["reward_model"] = d["reward_model"].map(lambda rm: {**dict(rm), "style": "env"})
    return d[["prompt", "data_source", "agent_name", "reward_model", "extra_info"]]


def _self_check(d: pd.DataFrame) -> None:
    """落盘前硬校验(fail-loud):任何一项不满足都不允许产出文件。"""
    assert set(d["agent_name"]) == {"trace_opd"}, "agent_name 必须全为 trace_opd"
    assert all(rm["style"] == "env" for rm in d["reward_model"]), "reward_model.style 必须全为 env"
    need = {"task_id", "task_type", "split", "question", "golden_answers", "initial_obs"}
    assert need <= set(d["extra_info"].iloc[0].keys()), f"extra_info 缺 {need - set(d['extra_info'].iloc[0].keys())}"
    assert d["data_source"].str.startswith("searchqa/").all(), "val 的 data_source 须带源后缀(分项曲线依赖)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--test-file", default="test.parquet")
    ap.add_argument("--per-source", type=int, default=320)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="val_eq320.parquet")
    a = ap.parse_args()

    d = Path(a.data_dir)
    df = pd.read_parquet(d / a.test_file)
    assert "data_source" in df.columns, "test parquet 缺 data_source 列"

    parts, report = [], []
    for src, g in sorted(df.groupby("data_source"), key=lambda kv: kv[0]):
        n = min(a.per_source, len(g))
        parts.append(g.sample(n=n, random_state=a.seed))
        report.append((src, n, len(g)))
    out = (
        pd.concat(parts)
        .sample(frac=1, random_state=a.seed)  # 全局重洗,防按源成块
        .reset_index(drop=True)
    )
    out = _trainer_schema(out)
    _self_check(out)
    out_path = d / a.out
    out.to_parquet(out_path, index=False)

    print(f"wrote {out_path}  total={len(out)}  (per-source cap={a.per_source} seed={a.seed})")
    for src, n, total in report:
        tag = " (全量)" if n == total else ""
        print(f"  {src:28s} {n:4d} / {total}{tag}")


if __name__ == "__main__":
    main()
