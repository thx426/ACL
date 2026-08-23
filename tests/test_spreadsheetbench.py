"""SpreadsheetBench 环境:评估语义锁、function-call react、端到端与隔离红线。

评估语义 = 旧 skills_evolve 移植(官方 evaluation.py 规则),用行为快照锁定
(round-2 量化/类型严格/空值等价/datetime/bool);episode 端到端跑真
subprocess(run_python)驱动真实机制循环 —— function-calling 协议在非
文本动作环境上成立的机器证明。
"""
from __future__ import annotations

import asyncio
import datetime
import json
import threading
import re
from pathlib import Path
from types import SimpleNamespace

import openpyxl
import pytest

from fakes import FakeTokenizer, ScriptedGenerate
from trace_opd.common.config import (
    SpreadsheetbenchCfg,
    load_config,
    objective_fingerprint,
)
from trace_opd.core.episode import run_episode
from trace_opd.core.template_delta import TemplateDelta
from trace_opd.envs import registry
from trace_opd.envs.spreadsheetbench.adapter import (
    _format_reminder,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_TEXT,
    TOOLS,
    SpreadsheetbenchEnvFactory,
    SpreadsheetbenchTaskAdapter,
    initial_obs_of,
    parse_text_action,
    text_extra_note,
    text_format_reminder,
)
from trace_opd.envs.spreadsheetbench.evaluator import (
    _compare_cell_value,
    evaluate,
    find_test_cases,
    preview_workbook,
)
from trace_opd.envs.spreadsheetbench.recalc import (
    find_soffice,
    workbook_has_formulas,
)

TOK = FakeTokenizer()
DELTA = TemplateDelta(TOK)


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def _wb(path: Path, cells: dict[str, object]) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    for coord, v in cells.items():
        ws[coord] = v
    wb.save(path)
    wb.close()


def _make_task(tmp_path: Path, *, answers: list[dict[str, object]]) -> dict:
    """构造任务目录(N 个 case)+ 返回自足 extra_info item。
    input: A1=1, A2=2;答案单元格 B1 由 answers[i] 给出。"""
    task_dir = tmp_path / "spreadsheet" / "T1"
    task_dir.mkdir(parents=True)
    for i, ans in enumerate(answers, 1):
        _wb(task_dir / f"{i}_t1_input.xlsx", {"A1": 1, "A2": 2})
        _wb(task_dir / f"{i}_t1_answer.xlsx", {"A1": 1, "A2": 2, **ans})
    item = {
        "task_id": "T1",
        "instruction": "Sum A1:A2 into B1.",
        "instruction_type": "Cell-Level Manipulation",
        "answer_position": "B1",
        "spreadsheet_dir": "spreadsheet/T1",
        "split": "train",
        "n_cases": len(answers),
    }
    item["initial_obs"] = initial_obs_of(
        item, preview_workbook(str(task_dir / "1_t1_input.xlsx"))
    )
    return item


def _adapter(
    tmp_path: Path, item: dict, protocol: str = "fncall"
) -> SpreadsheetbenchTaskAdapter:
    cfg = SpreadsheetbenchCfg(data_root=str(tmp_path), protocol=protocol)
    return SpreadsheetbenchTaskAdapter(item, cfg, threading.Semaphore(4))


def _call(name: str, **arguments) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"working on it\n<tool_call>\n{payload}\n</tool_call>"


SUM_CODE = (
    "import openpyxl\n"
    "wb = openpyxl.load_workbook('input.xlsx')\n"
    "ws = wb.active\n"
    "ws['B1'] = ws['A1'].value + ws['A2'].value\n"
    "wb.save('output.xlsx')\n"
    "print('saved')\n"
)


# ------------------------------------------------------- 评估语义锁(官方移植)

def test_compare_cell_value_official_semantics():
    assert _compare_cell_value(3.001, 3.0)            # round-2 量化
    assert not _compare_cell_value(3.01, 3.0)
    assert _compare_cell_value("3.00", 3)             # 数值字符串→float
    assert _compare_cell_value("", None)              # 空值等价
    assert not _compare_cell_value("3", "three")
    assert not _compare_cell_value(1, "x")            # 类型严格
    assert _compare_cell_value(True, 1)               # bool 提升
    assert _compare_cell_value(
        datetime.datetime(2024, 1, 2, 3), datetime.datetime(2024, 1, 2, 9)
    )                                                 # 序列日取整(同日等价)


def test_evaluate_and_case_discovery(tmp_path):
    item = _make_task(tmp_path, answers=[{"B1": 3}])
    task_dir = tmp_path / "spreadsheet" / "T1"
    cases = find_test_cases(str(task_dir))
    assert len(cases) == 1 and cases[0][0] == "1"
    pred = tmp_path / "pred.xlsx"
    _wb(pred, {"A1": 1, "A2": 2, "B1": 3})
    assert evaluate(str(pred), cases[0][2], item["answer_position"])["ok"]
    _wb(pred, {"A1": 1, "A2": 2, "B1": 4})
    verdict = evaluate(str(pred), cases[0][2], item["answer_position"])
    assert not verdict["ok"]
    assert "pred=4" in verdict["reason"]              # GT SAFETY:只报 pred
    assert "3" not in verdict["reason"].split("pred=")[1]  # 不泄 golden 值


def test_tools_schema_native_rendering_contract(tmp_path):
    """FC 三层契约(用户裁定 2026-08-07):定义=OpenAI 格式单一真源;
    渲染=chat template 原生(scaffold 绝不手写工具块);解析=文本层。"""
    assert [t["function"]["name"] for t in TOOLS] == ["run_python", "submit"]
    for t in TOOLS:
        assert t["type"] == "function"
        assert set(t["function"]) == {"name", "description", "parameters"}
        assert t["function"]["parameters"]["type"] == "object"
    assert TOOLS[0]["function"]["parameters"]["required"] == ["code"]
    # scaffold 零手写渲染块(工具块由模板按 tools() 渲染)
    assert "<tools>" not in SYSTEM_PROMPT and "<tool_call>" not in SYSTEM_PROMPT
    item = _make_task(tmp_path, answers=[{"B1": 3}])
    adapter = _adapter(tmp_path, item)
    assert adapter.tools() is TOOLS
    # prompt 渲染确实带上了 tools(fake 模板的确定性等价渲染)
    from trace_opd.core.template_delta import render_ids

    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "obs"}]
    with_tools = render_ids(TOK, msgs, True, tools=TOOLS)
    without = render_ids(TOK, msgs, True)
    assert with_tools != without and len(with_tools) > len(without)


# ------------------------------------------------------- episode 端到端(真执行)

def _episode(item, tmp_path, texts, protocol: str = "fncall"):
    adapter = _adapter(tmp_path, item, protocol=protocol)
    return _run(run_episode(
        adapter=adapter,
        generate=ScriptedGenerate(TOK, texts),
        tokenizer=TOK,
        delta=DELTA,
        guidance="",
        episode_id="ep-ssb",
        response_budget=200000,
        sampling_params={"temperature": 0.5},
    ))


def test_episode_win_via_function_calls(tmp_path):
    item = _make_task(tmp_path, answers=[{"B1": 3}])
    result = _episode(item, tmp_path, [
        _call("run_python", code=SUM_CODE),
        _call("submit"),
    ])
    traj = result.trajectory
    assert traj.won and traj.score == 1.0
    assert traj.steps[0].valid                       # exit 0
    assert "saved" in traj.steps[0].observation
    assert traj.steps[0].observation.startswith("<tool_response>")
    # 信息状态红线:submit 反馈零评估信息
    assert "Submitted." in traj.steps[1].observation
    assert "pass" not in traj.steps[1].observation.lower()
    # action_norm 可 json 解析(replay 契约)
    assert json.loads(traj.steps[0].action_norm)["name"] == "run_python"


def test_episode_partial_credit_multi_case(tmp_path):
    # case1 期望 B1=3(agent 能对),case2 期望 B1=99(同一 output 必挂)
    item = _make_task(tmp_path, answers=[{"B1": 3}, {"B1": 99}])
    result = _episode(item, tmp_path, [
        _call("run_python", code=SUM_CODE),
        _call("submit"),
    ])
    traj = result.trajectory
    assert traj.won is False
    assert traj.score == 0.5                          # soft 保真


def test_episode_format_failure_and_premature_submit(tmp_path):
    item = _make_task(tmp_path, answers=[{"B1": 3}])
    result = _episode(item, tmp_path, [
        "let me think, no call yet",                  # 格式失败
        _call("submit"),                              # output 未写:不终局
        _call("run_python", code=SUM_CODE),
        _call("submit"),
    ])
    traj = result.trajectory
    assert traj.steps[0].acted is False
    assert traj.steps[0].observation == _format_reminder()
    assert traj.steps[1].acted and not traj.steps[1].valid
    assert "not found" in traj.steps[1].observation
    assert traj.won and traj.score == 1.0
    assert result.invalid_format == 1


# ------------------------------------------------- 文本协议(2026-08-17 裁定)

def _py_block(code: str) -> str:
    return f"let me work\n```python\n{code}```\n"


def test_parse_text_action_priority():
    # 代码块提取(标注必需;裸 ``` 不收)
    call = parse_text_action(_py_block("print(1)\n"))
    assert call == {"name": "run_python", "arguments": {"code": "print(1)"}}
    assert parse_text_action("```\nprint(1)\n```") is None
    # 多块取最后(ToRL 同款)
    two = "```python\nprint(1)\n```\nthen\n```py\nprint(2)\n```"
    assert parse_text_action(two)["arguments"]["code"] == "print(2)"
    # submit 识别(独立回复;fence 内空白/换行宽容)
    assert parse_text_action("done.\n```submit\n```")["name"] == "submit"
    assert parse_text_action("```submit```")["name"] == "submit"
    # 同现优先级:代码 > submit(乐观断言防线)
    both = _py_block("print(3)\n") + "\n```submit\n```"
    assert parse_text_action(both)["name"] == "run_python"
    # void turn
    assert parse_text_action("I think the answer is 42.") is None
    # 思考正文里提到 submit 单词不触发
    assert parse_text_action("I will submit soon.") is None


def test_text_extra_note_states_ignored_submit():
    both = _py_block("print(3)\n") + "\n```submit\n```"
    note = text_extra_note(both)
    assert "ignored" in note and "submit" in note
    # 多块 + submit:两条 note 都在
    tri = "```python\nprint(1)\n```\n```python\nprint(2)\n```\n```submit\n```"
    note2 = text_extra_note(tri)
    assert "earlier code block" in note2 and "```submit```" in note2
    assert text_extra_note(_py_block("print(1)\n")) == ""


def test_text_mode_contract(tmp_path):
    item = _make_task(tmp_path, answers=[{"B1": 3}])
    ad = _adapter(tmp_path, item, protocol="text")
    # 文本协议不渲染 tools(与围栏协议打架);scaffold 换 one-shot 版
    assert ad.tools() is None
    assert ad.executor_scaffold() is SYSTEM_PROMPT_TEXT
    assert "```submit" in SYSTEM_PROMPT_TEXT and "```python" in SYSTEM_PROMPT_TEXT
    assert "Worked example" in SYSTEM_PROMPT_TEXT      # one-shot(用户裁定)
    assert ad.format_reminder() == text_format_reminder()
    brief = ad.task_brief()
    assert "```python" in brief and "```submit" in brief
    assert "run_python(code)" not in brief
    # fncall 档回归:契约不变
    ad_fc = _adapter(tmp_path, item)
    assert ad_fc.tools() is TOOLS
    assert ad_fc.executor_scaffold() is SYSTEM_PROMPT
    assert "run_python(code)" in ad_fc.task_brief()


def test_episode_text_protocol_win(tmp_path):
    item = _make_task(tmp_path, answers=[{"B1": 3}])
    result = _episode(item, tmp_path, [
        _py_block(SUM_CODE),
        "verified.\n```submit\n```",
    ], protocol="text")
    traj = result.trajectory
    assert traj.won and traj.score == 1.0
    # 观察 = ```output 围栏,不是 <tool_response>
    assert traj.steps[0].observation.startswith("```output\n")
    assert "saved" in traj.steps[0].observation
    assert "<tool_response>" not in traj.steps[0].observation
    assert "Submitted." in traj.steps[1].observation
    # action_norm 与 FC 同构(replay 契约共用)
    assert json.loads(traj.steps[0].action_norm)["name"] == "run_python"


def test_episode_text_void_premature_and_cosubmit(tmp_path):
    item = _make_task(tmp_path, answers=[{"B1": 3}])
    result = _episode(item, tmp_path, [
        "thinking out loud, no action",               # void turn
        "done!\n```submit\n```",                      # output 未写:不终局
        _py_block(SUM_CODE) + "\n```submit\n```",     # 同现:执行代码,不终局
        "now finish.\n```submit\n```",
    ], protocol="text")
    traj = result.trajectory
    assert traj.steps[0].acted is False
    assert traj.steps[0].observation == text_format_reminder()
    assert traj.steps[1].acted and not traj.steps[1].valid
    assert "not found" in traj.steps[1].observation
    # 同现轮:代码执行了,note 在围栏外说明 submit 被忽略
    obs2 = traj.steps[2].observation
    assert obs2.startswith("```output\n") and "saved" in obs2
    assert obs2.rstrip().endswith(")")                # note 在闭栏之后
    assert "ignored" in obs2.split("```", 2)[-1]
    assert traj.won and traj.score == 1.0
    assert result.invalid_format == 1


def test_initial_obs_drift_detected(tmp_path):
    """审阅修复回归:episode 的 initial_obs 由 live 重算(preview 同源)
    ——数据集捕获后 xlsx 被改动必须 ENV_ERROR(此前回传捕获值恒真)。"""
    from trace_opd.core.types import OUTCOME_ENV_ERROR

    item = _make_task(tmp_path, answers=[{"B1": 3}])
    _wb(tmp_path / "spreadsheet" / "T1" / "1_t1_input.xlsx", {"A1": 999})  # 漂移
    result = _episode(item, tmp_path, [_call("submit")])
    assert result.trajectory.outcome == OUTCOME_ENV_ERROR
    assert "initial_obs mismatch" in (result.trajectory.error or "")


def test_replay_action_deterministic_and_isolated(tmp_path):
    item = _make_task(tmp_path, answers=[{"B1": 3}])
    adapter = _adapter(tmp_path, item)
    env = adapter.open_episode()
    try:
        # 隔离红线:沙箱里只有 input.xlsx,golden 不可达
        assert sorted(p.name for p in Path(env.workdir).iterdir()) == ["input.xlsx"]
        norm = json.dumps(
            {"name": "run_python", "arguments": {"code": "print(open('input.xlsx','rb').read(4).hex())"}},
            sort_keys=True,
        )
        fb1, ended1 = adapter.replay_action(env, norm)
        fb2, ended2 = adapter.replay_action(env, norm)
        assert fb1 == fb2 and not ended1 and not ended2   # 重放确定性
    finally:
        env.close()
    assert not Path(env.workdir).exists()                 # close 清理沙箱


# ------------------------------------------ 公式重算(官方 open_spreadsheet 语义)

def test_workbook_has_formulas_zip_scan(tmp_path):
    plain = tmp_path / "plain.xlsx"
    _wb(plain, {"A1": 1, "B1": 2})
    assert workbook_has_formulas(plain.read_bytes()) is False
    withf = tmp_path / "withf.xlsx"
    _wb(withf, {"A1": 1, "B1": "=A1+1"})
    assert workbook_has_formulas(withf.read_bytes()) is True
    assert workbook_has_formulas(b"not a zip") is False


FORMULA_CODE = (
    "import openpyxl\n"
    "wb = openpyxl.load_workbook('input.xlsx')\n"
    "ws = wb.active\n"
    "ws['B1'] = '=A1+A2'\n"
    "wb.save('output.xlsx')\n"
    "print('formula written')\n"
)


@pytest.mark.skipif(find_soffice() is None, reason="no LibreOffice backend")
def test_formula_solution_passes_via_recalc(tmp_path):
    """核心回归:写公式是合法解法——评分前 LibreOffice 重算,公式解法
    与写值解法同分。(第一代移植漏掉重算 ⇒ 公式必挂,是历史环境缺陷。)"""
    item = _make_task(tmp_path, answers=[{"B1": 3}])
    result = _episode(item, tmp_path, [
        _call("run_python", code=FORMULA_CODE),
        _call("submit"),
    ])
    traj = result.trajectory
    assert traj.won and traj.score == 1.0


# ------------------------------------------------------------- 装配与配置

@pytest.fixture
def fake_soffice(monkeypatch, tmp_path):
    """factory 装配护栏测试专用:注入假 soffice 路径(find_soffice 只查
    文件存在;lru_cache 前后清空)。"""
    exe = tmp_path / "soffice-fake"
    exe.write_text("#!/bin/sh\n")
    find_soffice.cache_clear()
    monkeypatch.setenv("SSB_SOFFICE", str(exe))
    yield str(exe)
    find_soffice.cache_clear()


def test_registry_dispatch_and_dataroot_required(tmp_path, fake_soffice):
    cfg = SimpleNamespace(
        environment=SimpleNamespace(name="spreadsheetbench"),
        spreadsheetbench=SpreadsheetbenchCfg(data_root=str(tmp_path)),
    )
    factory = registry.make_env_factory(cfg)
    assert isinstance(factory, SpreadsheetbenchEnvFactory)
    bad = SimpleNamespace(
        environment=SimpleNamespace(name="spreadsheetbench"),
        spreadsheetbench=SpreadsheetbenchCfg(data_root=""),
    )
    with pytest.raises(ValueError, match="data_root"):
        registry.make_env_factory(bad)


def test_factory_fails_loud_without_soffice(monkeypatch, tmp_path):
    """无重算后端(soffice 与 SSB_RECALC_URL 皆无)= 公式解法静默判 0 =
    reward 语义漂移 ⇒ 装配期即死。"""
    find_soffice.cache_clear()
    monkeypatch.setenv("SSB_SOFFICE", str(tmp_path / "nonexistent"))
    monkeypatch.delenv("SSB_RECALC_URL", raising=False)
    try:
        with pytest.raises(ValueError, match="no formula-recalc backend"):
            SpreadsheetbenchEnvFactory(SpreadsheetbenchCfg(data_root=str(tmp_path)))
    finally:
        find_soffice.cache_clear()


@pytest.fixture
def deploy_env(monkeypatch, tmp_path):
    for k, v in {
        "TRACE_MODEL_PATH": str(tmp_path / "model"),
        "SSB_TRAIN_FILE": str(tmp_path / "t.parquet"),
        "SSB_VAL_FILE": str(tmp_path / "v.parquet"),
        "SSB_DATA": str(tmp_path / "d"),
        "TRACE_RESEARCHER_API_KEY": "dummy-for-validation",
    }.items():
        monkeypatch.setenv(k, v)


def test_ssb_yaml_pair_loads_and_fingerprints(deploy_env):
    """SSB 的两个正式实验臂:v4 自体研究员 vs 纯 GRPO 对照。

    2026-08-09 迁移:原先指向 configs/experiments/a800-ssb-s1-{selfres,extres}.yaml,
    那两份成文于 v3.2 时代(gate_mode=always_on / w_beh=per_token / 无 M1M2M4 键),
    机制定型到 v3.3 后已过期,按用户裁定删除。实验配置改为一实验一目录。
    """
    root = Path(__file__).resolve().parents[1]
    selfres = load_config(root / "experiments/ssb-9b-selfres-a800/config.yaml")
    grpo = load_config(root / "experiments/ssb-9b-grpo-baseline-a800/config.yaml")
    assert selfres.environment.name == "spreadsheetbench"
    # 自体研究员臂 = v3.3 全套 + v4 双目标
    assert selfres.research_rl.enabled and selfres.research.backend == "engine"
    assert selfres.steering.enabled and selfres.steering.adv_mode == "asym"
    assert selfres.train.adv_estimator == "pgs_asym_grpo"
    assert selfres.steering.gate_mode == "entropy"     # M4 熵门(v3.3),非 always_on
    assert selfres.steering.w_beh_mode == "off"        # M3 新默认
    # 对照臂 = 机制全关,且守卫强制的联动到位
    assert not (grpo.steering.enabled or grpo.research_rl.enabled or grpo.research.enabled)
    assert grpo.train.adv_estimator == "grpo_loo"
    assert grpo.steering.adv_mode in ("", "legacy_shared")
    fp_self, fp_grpo = objective_fingerprint(selfres), objective_fingerprint(grpo)
    assert fp_self != fp_grpo
    assert "environment=spreadsheetbench" not in (fp_self + fp_grpo)   # 指纹是哈希非明文
    # 换环境进指纹:同配置改回 alfworld 指纹必变
    selfres.environment.name = "alfworld"
    assert objective_fingerprint(selfres) != fp_self


def test_ssb_external_researcher_path_still_expressible(deploy_env, tmp_path):
    """方案裁定 4:researcher 双模式必须都保留(自体 + 外部大模型端点)。

    不再用一份静态 yaml 占位 —— 改为从自体臂就地改三个键构造 external 变体,
    证明该路径在**当前机制**下仍可表达(旧做法会让模板随机制演进而腐化)。
    """
    root = Path(__file__).resolve().parents[1]
    raw = (root / "experiments/ssb-9b-selfres-a800/config.yaml").read_text(encoding="utf-8")
    raw = raw.replace("  backend: engine", "  backend: external", 1)
    raw = raw.replace("  tool_protocol: text", "  tool_protocol: fncall", 1)
    raw = re.sub(r"^(research_rl:\s*\n)  enabled: true", r"\1  enabled: false", raw, count=1, flags=re.M)
    f = tmp_path / "ssb_extres.yaml"
    f.write_text(raw, encoding="utf-8")
    cfg = load_config(f)
    assert cfg.research.backend == "external" and not cfg.research_rl.enabled
    assert cfg.steering.enabled and cfg.steering.adv_mode == "asym"   # 机制仍是 v3.3


# ---------- Spreadsheet-RL 引入件(2026-08-11) ----------

def test_find_test_cases_srl_naming(tmp_path):
    """SRL 目录形态 input/target ⇒ 单 case;output.xlsx(oracle 中间产物)
    不得干扰发现;SSB 三种存量命名优先级不受影响。"""
    td = tmp_path / "excelforum" / "sub" / "task-job-x"
    td.mkdir(parents=True)
    _wb(td / "input.xlsx", {"A1": 1})
    _wb(td / "target.xlsx", {"A1": 1, "B1": 2})
    _wb(td / "output.xlsx", {"A1": 1})
    cases = find_test_cases(str(td))
    assert len(cases) == 1
    no, ip, ap = cases[0]
    assert no == "1" and ip.endswith("input.xlsx") and ap.endswith("target.xlsx")


def test_normalize_answer_position_forms():
    """实测三形态:A 单区域缺前引号 / B 多区域首段缺 / D 合法撇号不动;
    完好输入恒等。"""
    from trace_opd.envs.spreadsheetbench.build_dataset_srl import (
        normalize_answer_position as norm,
    )
    assert norm("Sheet1'!C3:C545") == "'Sheet1'!C3:C545"                    # A
    assert norm("Total'!G2,'Total'!H2") == "'Total'!G2,'Total'!H2"          # B
    assert norm("'RYAN'S ORDERS'!M6:M42") == "'RYAN'S ORDERS'!M6:M42"       # D
    assert norm("'Sheet1'!K2:K71") == "'Sheet1'!K2:K71"                     # 恒等
    assert norm("B2:C160") == "B2:C160"                                     # 无 sheet 段


def test_preview_workbook_chartsheet(tmp_path):
    """含 Chartsheet 的 workbook(SRL Charting 任务实测形态)preview 不炸,
    输出占位行;普通 sheet 输出不变。"""
    from openpyxl.chart import BarChart, Reference

    p = tmp_path / "chart.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws["A1"] = 7
    cs = wb.create_chartsheet("MyChart")
    chart = BarChart()  # 空 chartsheet openpyxl 自己写完读不回,挂真 chart
    chart.add_data(Reference(ws, min_col=1, min_row=1, max_row=1))
    cs.add_chart(chart)
    wb.save(p)
    wb.close()
    text = preview_workbook(str(p))
    assert "A1=7" in text
    assert "MyChart" in text and "no cell grid" in text


# ---------- Windows Excel 远程重算后端(2026-08-11,用户裁定) ----------

def _mock_recalc_server(response_bytes: bytes | None, status: int = 200):
    """本地 HTTP mock:POST /recalc → status+bytes。返回 (server, url)。"""
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            self.rfile.read(n)
            body = response_bytes if response_bytes is not None else b"boom"
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _formula_wb_bytes() -> bytes:
    import io

    wb = openpyxl.Workbook()
    wb.active["A1"] = "=1+1"
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_remote_recalc_preferred_over_soffice(monkeypatch):
    """SSB_RECALC_URL 设置 ⇒ 走 remote,soffice 不被调用。"""
    from trace_opd.envs.spreadsheetbench import recalc as R

    payload = b"X" * 200  # >100 字节即视为合法 workbook 响应
    srv, url = _mock_recalc_server(payload)
    monkeypatch.setenv("SSB_RECALC_URL", url)
    called = []
    monkeypatch.setattr(R, "recalculate", lambda *a, **k: called.append(1) or b"")
    out = R.recalculated_for_grading(
        _formula_wb_bytes(), soffice="/fake/soffice", timeout_s=10, task_id="t"
    )
    srv.shutdown()
    assert out == payload and not called


def test_remote_recalc_failure_no_fallback(monkeypatch):
    """remote 失败 ⇒ 原字节评分,绝不回落 soffice(引擎混用=语义漂移)。"""
    from trace_opd.envs.spreadsheetbench import recalc as R

    srv, url = _mock_recalc_server(None, status=500)
    monkeypatch.setenv("SSB_RECALC_URL", url)
    called = []
    monkeypatch.setattr(R, "recalculate", lambda *a, **k: called.append(1) or b"")
    data = _formula_wb_bytes()
    out = R.recalculated_for_grading(
        data, soffice="/fake/soffice", timeout_s=10, task_id="t"
    )
    srv.shutdown()
    assert out == data and not called


def test_factory_boots_with_remote_url_only(monkeypatch, tmp_path):
    """无 soffice 但有 SSB_RECALC_URL ⇒ factory 可装配(装配门放行)。"""
    from trace_opd.envs.spreadsheetbench import adapter as A

    monkeypatch.setenv("SSB_RECALC_URL", "http://127.0.0.1:1")
    monkeypatch.setattr(A, "find_soffice", lambda: None)
    cfg = SpreadsheetbenchCfg(data_root=str(tmp_path))
    A.SpreadsheetbenchEnvFactory(cfg)  # 不抛即过

    monkeypatch.delenv("SSB_RECALC_URL")
    with pytest.raises(ValueError, match="no formula-recalc backend"):
        A.SpreadsheetbenchEnvFactory(cfg)
