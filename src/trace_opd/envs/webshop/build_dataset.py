"""WebShop dataset builder — 经运行中的 webshop 服务固化 goal → parquet。

划分口径(本仓裁定,WebShop 原版语义;与 GiGPO/RAGEN 的差异见 adapter
docstring):env 内建 seed(233) shuffle 后的 goal 列表上
  test  = [0, 500)
  valid = [500, 1500)   (--val-size 截取其前缀,训练期曲线用)
  train = [1500, num_goals)

initial_obs 与 instruction 逐 goal 经服务端 /reset 固化进 parquet(WebShop
reset(session=goal_idx) 幂等确定 ⇒ 无 drift;约 6,910 goals 分钟级)。

用法(服务端须先起):
  python -m trace_opd.envs.webshop.build_dataset \
      --service-url http://10.77.110.162:8310 --out <dir> \
      [--val-size 500] [--splits train,valid,test]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
import pandas as pd

from ...core import protocol
from .adapter import _render_actions, system_prompt_text

_TEST_END = 500
_VAL_END = 1500


def _capture(http: httpx.Client, base: str, goal_idx: int) -> tuple[str, str]:
    r = http.post(f"{base}/reset", json={"goal_idx": goal_idx}, timeout=60)
    r.raise_for_status()
    d = r.json()
    obs, instruction, sid = d["observation"], d["instruction"], d["sid"]
    actions = d.get("actions") or {}
    http.post(f"{base}/release", json={"sid": sid}, timeout=30)
    # ★ 动作面渲染必须与 adapter.open_episode 走**同一个函数** —— 否则
    # parquet 里的 initial_obs 与 runtime 现取的观察会分叉,而 episode 的
    # mismatch 校验正是拿这两者对拍(校验用的两份若不同源,校验就是假的)。
    # 旧实现是复制粘贴的第二份,在"无 search bar 且无 clickables"时与
    # _render_actions 输出不同("" vs "(no actions available)");reset 后
    # 恒有 search bar 所以从未触发,但那是运气不是设计。
    rendered = f"{obs}\n\n[Available actions]\n{_render_actions(actions)}"
    return rendered, instruction


def _record(goal_idx: int, split: str, initial_obs: str, instruction: str) -> dict:
    extra_info = {
        "task_id": f"webshop_{goal_idx:05d}",
        "task_type": "webshop",
        "split": split,
        "goal_idx": goal_idx,
        "instruction": instruction,
        "initial_obs": initial_obs,
    }
    return {
        "prompt": [
            # 构建期快照(runtime 一律从 adapter.executor_scaffold() 取生效
            # scaffold);默认档 = text,与 adapter 默认一致。
            {"role": "system", "content": system_prompt_text()},
            {"role": "user", "content": protocol.truncate_obs(initial_obs)},
        ],
        "data_source": "webshop" if split == "train" else f"webshop/{split}",
        # ★ 恒为 "trace_opd":verl trainer 按它路由到本项目的 agent loop,
        # 缺失会在 reward 分派处炸 NotImplementedError。2026-08-13 SearchQA
        # 撞过同坑(探针走独立 evalm 路径不经 trainer,所以从未暴露);
        # 本次(2026-08-16)按当时的全域扫描结论补齐。
        "agent_name": "trace_opd",
        "reward_model": {"style": "env", "ground_truth": ""},
        "extra_info": extra_info,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--service-url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--splits", default="train,valid,test")
    args = ap.parse_args()

    base = args.service_url.rstrip("/")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    http = httpx.Client()

    meta = http.get(f"{base}/meta", timeout=30).json()
    num_goals = int(meta["num_goals"])
    if num_goals <= _VAL_END:
        raise SystemExit(
            f"FATAL: service reports only {num_goals} goals (≤{_VAL_END}) — "
            "expected the ~6,910-goal small-set; wrong data mounted?"
        )
    print(f"[webshop] service reports {num_goals} goals")

    # 两分口径(用户裁定 2026-08-12,对齐 verl-agent/GiGPO:服务端 seed=0
    # shuffle 后 eval=前 500、train=其余全部;训练中 val 直接用 test 全量,
    # 不再切独立 valid/缓冲带)。--val-size 保留仅为兼容,valid=test 别名。
    ranges = {
        "test": range(0, _TEST_END),
        "valid": range(0, _TEST_END),
        "train": range(_TEST_END, num_goals),
    }
    manifest: dict = {"num_goals": num_goals, "val_size": args.val_size,
                      "split_rule": "test=valid=[0,500) train=[500,N) (verl-agent aligned, service seed=0)"}

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        idxs = ranges[split]
        recs = []
        for i, gi in enumerate(idxs):
            obs, instruction = _capture(http, base, gi)
            recs.append(_record(gi, split, obs, instruction))
            if (i + 1) % 500 == 0:
                print(f"[webshop] {split}: {i + 1}/{len(idxs)} captured")
        pd.DataFrame(recs).to_parquet(out / f"{split}.parquet")
        manifest[split] = {"rows": len(recs), "goal_range": [idxs[0], idxs[-1] + 1]}
        print(f"[webshop] {split}: {len(recs)} rows -> {out / f'{split}.parquet'}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[webshop] manifest -> {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
