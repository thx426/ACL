"""阶段D ckpt 筛选判据合同(用户裁定 2026-08-07)。

红线:① 协议层/产出层是硬门,不达标必须 NO-GO 而不是"挑个最好的";
② "文档产出率相当"时必须取更靠前的 ckpt;③ 相当性用 Wilson 下界判,
不用点估计(20 样本下点估计噪声会盖过真实差异)。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_sft_ckpts.py"
_spec = importlib.util.spec_from_file_location("probe_sft_ckpts", _SCRIPT)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def _session(finalized=True, doc="a b c", turns=5, exps=4, calls=4, bad_calls=0):
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "go"}]
    for _ in range(calls):
        msgs.append({"role": "assistant",
                     "content": 'think\n<tool_call>\n{"name": "run_experiment", '
                                '"arguments": {"guidance": "", "k": 4}}\n</tool_call>'})
        msgs.append({"role": "user", "content": "<tool_response>ok</tool_response>"})
    for _ in range(bad_calls):
        msgs.append({"role": "assistant", "content": "I will now run an experiment."})
    return {"messages": msgs, "doc": doc, "finalized": finalized,
            "error": None, "turns_used": turns, "experiments_used": exps}


def _mk(root: Path, step: int, n=20, fin=20, bad_calls=0):
    d = root / f"step_{step}"
    for i in range(n):
        p = d / f"task{i}"
        p.mkdir(parents=True, exist_ok=True)
        (p / "s0.json").write_text(json.dumps(
            _session(finalized=i < fin, doc="w " * 300 if i < fin else "",
                     bad_calls=bad_calls)), encoding="utf-8")
    return d


def test_tool_call_parsing_counts_only_valid_json_calls():
    ok, tot = probe._tool_call_stats(_session(calls=3, bad_calls=2)["messages"], "text")
    assert (ok, tot) == (3, 5)          # 纯文本"我将要调用"不算合法调用


def test_wilson_lower_is_conservative_on_small_n():
    # 20/20 的点估计是 1.0,但下界必须显著低于 1 —— 这正是不能用点估计的理由
    assert probe.wilson_lower(20, 20) < 0.90
    assert probe.wilson_lower(200, 200) > 0.98
    assert probe.wilson_lower(0, 20) == 0.0


def test_picks_earliest_ckpt_when_finalize_rate_is_comparable(tmp_path, capsys):
    _mk(tmp_path, 6, n=20, fin=18)     # 0.90
    _mk(tmp_path, 12, n=20, fin=19)    # 0.95
    _mk(tmp_path, 18, n=20, fin=20)    # 1.00 最高
    args = probe.argparse.Namespace(probe_root=str(tmp_path), ref="", protocol="text",
                                    tool_ok_min=0.5, finalize_min=0.5, tol=0.10)
    assert probe.cmd_select(args) == 0
    out = capsys.readouterr().out
    assert "GO: 选 step 6" in out, out   # 三者相当 ⇒ 取最靠前,不是取最高


def test_clearly_worse_early_ckpt_is_not_picked(tmp_path, capsys):
    _mk(tmp_path, 6, n=20, fin=6)      # 0.30 明显更差
    _mk(tmp_path, 12, n=20, fin=19)    # 0.95
    args = probe.argparse.Namespace(probe_root=str(tmp_path), ref="", protocol="text",
                                    tool_ok_min=0.5, finalize_min=0.5, tol=0.10)
    assert probe.cmd_select(args) == 0
    assert "GO: 选 step 12" in capsys.readouterr().out


def test_no_go_when_protocol_not_learned(tmp_path, capsys):
    """工具调用学不会 ⇒ 必须 NO-GO,而不是在坏 ckpt 里挑最好的。"""
    d = tmp_path / "step_6"
    for i in range(20):
        p = d / f"task{i}"
        p.mkdir(parents=True)
        (p / "s0.json").write_text(json.dumps(
            _session(calls=0, bad_calls=5, finalized=False, doc="")), encoding="utf-8")
    args = probe.argparse.Namespace(probe_root=str(tmp_path), ref="", protocol="text",
                                    tool_ok_min=0.80, finalize_min=0.50, tol=0.10)
    assert probe.cmd_select(args) == 2
    assert "NO-GO" in capsys.readouterr().out
