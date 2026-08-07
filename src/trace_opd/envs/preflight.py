"""统一环境预检 — 把 GO 清单的文字前置变成可执行断言(fail-loud 同向)。

用法(部署机上,起跑前):
  python -m trace_opd.envs.preflight --config configs/experiments/<run>.yaml
  python -m trace_opd.envs.preflight --config ... --parquet <valid.parquet>

三级检查,逐级收紧:
  1. config   load_config(schema/校验/env 展开——env 变量缺失在这里死);
  2. factory  make_env_factory(各环境装配检查:data_root 在位、soffice、
              math-verify、appworld worker 配置……全部 fail-loud 构造项);
  3. probe    (给了 --parquet 时)取首行任务开一个真实 episode:引擎
              boot / worker 往返 / preview 渲染真实发生,并执行 initial_obs
              漂移比对——这就是最小真冒烟。

退出码 0 = 全过;非 0 = 第一处失败已打印(带下一步提示)。
"""
from __future__ import annotations

import argparse
import sys


def _ok(step: str, detail: str = "") -> None:
    print(f"[preflight] OK   {step}" + (f" — {detail}" if detail else ""))


def _fail(step: str, err: BaseException) -> int:
    print(f"[preflight] FAIL {step}\n  {type(err).__name__}: {err}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--parquet", default="",
                    help="valid/train parquet;给了就做真实 episode probe")
    args = ap.parse_args()

    # 1. config ------------------------------------------------------------
    try:
        from ..common.config import load_config

        cfg = load_config(args.config)
        from ..core import protocol

        protocol.configure_episode_caps(
            cfg.episode.max_env_steps, cfg.episode.max_assistant_turns
        )
    except BaseException as err:  # noqa: BLE001
        return _fail("config load (check env vars referenced by the yaml)", err)
    _ok("config", f"environment={cfg.environment.name}")

    # 2. factory -----------------------------------------------------------
    try:
        from .registry import make_env_factory

        factory = make_env_factory(cfg)
    except BaseException as err:  # noqa: BLE001
        return _fail(f"factory build for {cfg.environment.name!r}", err)
    _ok("factory", type(factory).__name__)

    # 3. probe -------------------------------------------------------------
    if not args.parquet:
        print("[preflight] note: no --parquet given — skipped the live episode "
              "probe (pass the valid parquet for the full smoke).")
        return 0
    try:
        import pandas as pd

        df = pd.read_parquet(args.parquet)
        item = dict(df.iloc[0]["extra_info"])
        adapter = factory.adapter_from_item(item)
        _ok("adapter", f"task_id={adapter.task_id} task_type={adapter.task_type}")
        env = adapter.open_episode()
        try:
            live = env.initial_obs.strip()
            captured = adapter.initial_obs.strip()
            if captured and live != captured:
                raise RuntimeError(
                    "initial_obs drift: live episode does not reproduce the "
                    "dataset capture — data/version mismatch between build "
                    "and this runtime (rebuild the parquet in THIS env, or "
                    "fix the asset paths). This is the sentinel working, "
                    "not a bug."
                )
        finally:
            env.close()
    except BaseException as err:  # noqa: BLE001
        return _fail("live episode probe", err)
    _ok("probe", "episode opened, initial_obs matched, closed cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
