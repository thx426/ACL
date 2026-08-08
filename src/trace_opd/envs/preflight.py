"""统一环境预检 — 把 GO 清单的文字前置变成可执行断言(fail-loud 同向)。

用法(部署机上,起跑前):
  python -m trace_opd.envs.preflight --config configs/experiments/<run>.yaml
  python -m trace_opd.envs.preflight --config ... --parquet <valid.parquet>

四级检查,逐级收紧:
  1. config   load_config(schema/校验/env 展开——env 变量缺失在这里死);
  2. model    模型资产(model_path 在本机时):tokenizer 可载、轮末停止符
              集非空、TemplateDelta 模板差分硬断言前置;并按 sglang 的
              引擎默认停止集推导语义(generation_config 优先,缺则 config/
              text_config)做 eos 一致性告警——Qwen3.5 系资产坑的固化
              (docs/qwen35-9b-eos-stop-bug.md §2.1);
  3. factory  make_env_factory(各环境装配检查:data_root 在位、soffice、
              math-verify、appworld worker 配置……全部 fail-loud 构造项);
  4. probe    (给了 --parquet 时)取首行任务开一个真实 episode:引擎
              boot / worker 往返 / preview 渲染真实发生,并执行 initial_obs
              漂移比对——这就是最小真冒烟。

退出码 0 = 全过;非 0 = 第一处失败已打印(带下一步提示)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


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

    # 2. model assets -------------------------------------------------------
    # 2026-08-08 Qwen3.5 EOS 事故的固化:eos 三处一致性 + 模板差分断言前置。
    # 目录不在本机时跳过(config-only 核对场景);部署机上目录必在 ⇒ 必查。
    # FC 协议环境登记表(react 走 protocol.parse_tool_calls 的环境;新 FC
    # 环境接入必须在此登记,M1 后 alfworld 加入)——FC 门只对它们启用。
    _FC_ENVIRONMENTS = {"spreadsheetbench", "math"}
    mp = Path(cfg.train.model_path)
    if not mp.is_dir():
        print(f"[preflight] note: model_path {mp} absent on this host — asset "
              "check skipped (run on the deploy host for the full check).")
    else:
        try:
            import json as _json

            from transformers import AutoTokenizer

            from ..core.template_delta import (
                TemplateDelta,
                fc_roundtrip_probe,
                turn_stop_token_ids,
            )

            tok = AutoTokenizer.from_pretrained(str(mp), trust_remote_code=True)
            stop_ids = turn_stop_token_ids(tok)
            if not stop_ids:
                raise RuntimeError(
                    "turn_stop_token_ids() is empty — tokenizer has neither eos "
                    "nor <|im_end|>; rollout requests would carry no turn-final "
                    "stop and every turn would overrun"
                )
            delta = TemplateDelta(tok)  # glue/gen_prefix 差分硬断言,起跑前即暴露模板漂移

            # FC 门(2026-08-08 家族化):FC 协议环境要求模板可渲染 tool_calls
            # 且我方文本层解析器能逐值还原(训练通道无服务端 parser,两端形态
            # 必须同族——Qwen3 hermes 与 Qwen3.5 qwen35_xml 互不可解)。
            if cfg.environment.name in _FC_ENVIRONMENTS:
                if delta.fc_dialect == "none":
                    raise RuntimeError(
                        f"environment {cfg.environment.name!r} uses the FC protocol "
                        "but this chat template does not render tool_calls "
                        "(fc_dialect=none) — the executor could never emit a "
                        "parseable call"
                    )
                fc_roundtrip_probe(tok, delta.fc_dialect)
            _ok("model", f"stop_ids={stop_ids}; fc_dialect={delta.fc_dialect}; "
                         "template delta probes pass")

            # 引擎默认停止集(镜像 sglang ModelConfig._get_hf_eos_token_id 语义:
            # generation_config 优先,缺则 config/text_config)与轮末符对拍。
            # 只告警不拦——本仓库 rollout 通道由 EngineGenerate 显式下发 stop,
            # 告警的是"这份权重在任何绕过该通道的部署里会越界续写"这一事实。
            gcfg = mp / "generation_config.json"
            if gcfg.is_file():
                raw = _json.load(open(gcfg)).get("eos_token_id")
            else:
                c = _json.load(open(mp / "config.json"))
                raw = c.get("eos_token_id") or (c.get("text_config") or {}).get(
                    "eos_token_id"
                )
            engine_eos = {raw} if isinstance(raw, int) else set(raw or [])
            missing = set(stop_ids) - engine_eos
            if missing:
                why = ("no generation_config.json" if not gcfg.is_file()
                       else "generation_config incomplete")
                print(f"[preflight] WARN model — engine-default stop set "
                      f"{sorted(engine_eos)} misses turn-final ids {sorted(missing)} "
                      f"({why}); Qwen3.5-series pitfall, see "
                      "docs/qwen35-9b-eos-stop-bug.md — any serving of these "
                      "weights outside this repo's rollout channel must set "
                      "stop explicitly.")
        except BaseException as err:  # noqa: BLE001
            return _fail("model assets (tokenizer / stop ids / template delta)", err)

    # 3. factory -----------------------------------------------------------
    try:
        from .registry import make_env_factory

        factory = make_env_factory(cfg)
    except BaseException as err:  # noqa: BLE001
        return _fail(f"factory build for {cfg.environment.name!r}", err)
    _ok("factory", type(factory).__name__)

    # 4. probe -------------------------------------------------------------
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
