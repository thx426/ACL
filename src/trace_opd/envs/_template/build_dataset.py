"""Template dataset builder — the minimal verl-parquet field contract.

每个 benchmark 自带数据集构建脚本(本文件即示范;ALFWorld 的对应物是
train/build_datasets.py)。产出的 parquet 行契约(训练 agent_loop、评估
run_eval、SFT 采样 sftgen_researcher 共同消费):

  prompt        [{role: system, content: <adapter.executor_scaffold()>},
                 {role: user,   content: <truncate_obs(initial_obs)>}]
                —— system 必须与 adapter 的 scaffold 同源(勿手写第二份);
                   user 是捕获的初始观察。
  data_source   train 用裸名("template");val 用 "template/<split>"——
                verl 按它分桶 val 曲线(seen/unseen 对比即核心曲线)。
  agent_name    恒为 "trace_opd" —— 部署硬要求:verl 按它路由到本项目的
                agent loop,写错=静默跑别的 loop。
  reward_model  {"style": "env", "ground_truth": ""} —— reward 全部来自
                agent loop 的环境终态(rm_scores),不走 verl 的
                data_source 路由 reward 函数;占位结构保持原样。
  extra_info    工厂契约:``factory.adapter_from_item(extra_info)`` 是所有
                装配点的唯一取用方式,故 extra_info 必须自足——本 benchmark
                的身份(task_id/task_type)、确定性锚点(initial_obs)、
                任务参数(spec)、特权参考(gt_text)全在其中。

运行:python -m trace_opd.envs._template.build_dataset --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ...core import protocol
from .adapter import TemplateTaskAdapter

# 确定性任务网格(真实 benchmark 在此处枚举/捕获它的任务源)。
# 两个 family 示范 task_type 分层:passthrough(单连线)/crosslink(双连线)。
_GRID: list[tuple[str, dict[str, str]]] = [
    ("passthrough", {"s1": "out1"}),
    ("passthrough", {"s1": "out2"}),
    ("passthrough", {"s2": "out1"}),
    ("crosslink", {"s1": "out2", "s2": "out1"}),
    ("crosslink", {"s1": "out1", "s2": "out2"}),
    ("crosslink", {"s2": "out2", "s3": "out1"}),
]


def _initial_obs_of(spec: dict[str, str]) -> str:
    """确定性初始观察 —— 与 open_episode 的 env 逐字节同源(确定性红线:
    dataset 捕获值 = live 值,episode 开局与 v4 前缀重放都比对它)。"""
    ports = sorted(set(spec) | set(spec.values()))
    return (
        f"You stand before a signal patchboard. Ports: {', '.join(ports)}. "
        "Your task is to: route every signal to its matching output, then "
        "run diagnostics."
    )


def _gt_text_of(spec: dict[str, str]) -> str:
    """特权 walkthrough(researcher-only;信息状态红线见 adapter.gt_text)。"""
    lines = [f"{i}. link {s} {o}" for i, (s, o) in enumerate(sorted(spec.items()), 1)]
    lines.append(f"{len(lines) + 1}. run diagnostics")
    return "\n".join(lines)


def make_rows(split: str) -> list[dict]:
    rows = []
    for i, (family, spec) in enumerate(_GRID):
        initial_obs = _initial_obs_of(spec)
        extra_info = {
            "task_id": f"{split}/{family}-{i}/case_{i}",
            "task_type": family,
            "split": split,
            "initial_obs": initial_obs,
            "spec": dict(spec),
            "gt_text": _gt_text_of(spec),
        }
        scaffold = TemplateTaskAdapter(extra_info).executor_scaffold()
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": scaffold},
                    {"role": "user", "content": protocol.truncate_obs(initial_obs)},
                ],
                "data_source": "template" if split == "train" else f"template/{split}",
                "agent_name": "trace_opd",
                "reward_model": {"style": "env", "ground_truth": ""},
                "extra_info": extra_info,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid"):
        df = pd.DataFrame(make_rows(split))
        path = out / f"{split}.parquet"
        df.to_parquet(path)
        print(f"[template] {split}: {len(df)} tasks -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
