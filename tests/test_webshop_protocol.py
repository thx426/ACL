"""WebShop 两协议档的语义锁(2026-08-16 新建)。

该 benchmark 此前**零测试覆盖** —— 加 text 档时全量测试仍全绿,等于协议
改动无人守门(SearchQA 2026-08-16 同款情况)。本文件补上最小但真实的锁:

  * text scaffold 的协议说明与 one-shot 必须用**同一个**运行时推理标签;
  * 两档的 ``action_norm`` 必须都是**环境原生动作字符串**(replay 与轨迹
    渲染两档同形 —— 这是 continuation 前缀续写的前提);
  * 两档共用同一份任务面措辞(角色/工作风格/判分),差异只在动作发出方式
    —— 否则两档读数无法归因到协议;
  * text 档不做语法修复:``<action>`` 内容原样交环境(非法动作由 WebShop
    自己回 "Invalid action!");
  * 协议档进 objective_fingerprint。

不碰网络:``_post`` 被替换成假服务。
"""
from __future__ import annotations

import json

import pytest

from trace_opd.common.config import WebshopCfg
from trace_opd.core import protocol as P
from trace_opd.envs.webshop.adapter import (
    SYSTEM_PROMPT_FC,
    WebshopEpisode,
    WebshopTaskAdapter,
    _call_to_action,
    system_prompt_text,
)

ITEM = {
    "task_id": "webshop_00007",
    "task_type": "webshop",
    "split": "test",
    "goal_idx": 7,
    "instruction": "Buy a stainless steel bottle under 30 dollars",
    "initial_obs": "WebShop\nInstruction:\n...",
}


@pytest.fixture(autouse=True)
def _restore_tag():
    """任何用例改了全局标签都必须还原 —— 它是模块级状态。"""
    old = P.REASONING_TAG
    yield
    P.configure_reasoning_tag(old)


class _FakeService:
    """记录收到的 (动作, 分片),按脚本回响应。"""

    def __init__(self, script=None):
        self.seen: list[str] = []
        self.bases: list[str | None] = []
        self.script = script or {}

    def __call__(self, path, payload, timeout, base=None):
        assert path == "/step"
        action = payload["action"]
        self.seen.append(action)
        self.bases.append(base)
        return self.script.get(action, {
            "observation": "Page 1 (Total results: 3)\n[B01] Bottle $19.99",
            "reward": 0.0,
            "done": False,
            "actions": {"has_search_bar": False, "clickables": ["b01", "next >"]},
        })


def _adapter(protocol_name: str, service: _FakeService,
             cfg: WebshopCfg | None = None) -> WebshopTaskAdapter:
    cfg = cfg or WebshopCfg(protocol=protocol_name, preflight=False)
    ad = WebshopTaskAdapter(ITEM, cfg, http=None, gate=None)
    ad._post = service  # noqa: SLF001 — 测试替身,不碰网络
    return ad


#: 一个"搜索页"的可用动作:动作合法性判据的输入(见 action_is_available)
_PAGE = {"has_search_bar": True,
         "clickables": ["b01", "next >", "buy now", "black", "x-large"]}


def _episode(base: str = "http://127.0.0.1:8310",
             actions: dict | None = None) -> WebshopEpisode:
    return WebshopEpisode("sid-1", "obs", http=None,
                          cfg=WebshopCfg(preflight=False), base=base,
                          actions=dict(_PAGE if actions is None else actions))


# ---- scaffold 面 --------------------------------------------------------

def test_text_scaffold_uses_runtime_tag_in_both_places():
    s = system_prompt_text()
    ro, rc = P.rtag_open(), P.rtag_close()
    assert ro in s and rc in s
    assert "<think>" not in s
    # 协议说明一次 + one-shot 一次:标签必须同一个,否则 one-shot 在教错格式
    assert s.count(ro) >= 2 and s.count(rc) >= 2
    assert "<action>" in s and "</action>" in s
    # 动作字面是 WebShop 原生语法(与 GiGPO/verl-agent 同款)
    assert "search[" in s and "click[" in s

    P.configure_reasoning_tag("thought")
    s2 = system_prompt_text()
    assert "<thought>" in s2 and "<reasoning>" not in s2


def test_both_protocols_share_the_task_wording():
    """两档只应差在"动作怎么发",任务本身逐字相同 —— 否则读数不可归因。"""
    for line in ("You are a shopping agent in a text-based e-commerce",
                 "graded on product match, selected options, and price",
                 "do not browse aimlessly"):
        assert line in SYSTEM_PROMPT_FC
        assert line in system_prompt_text()
    # FC 档不得出现文本协议的标签(否则模型会两种格式混着写)
    assert "<action>" not in SYSTEM_PROMPT_FC
    assert P.rtag_open() not in SYSTEM_PROMPT_FC


def test_tools_and_reminder_follow_the_protocol():
    svc = _FakeService()
    text_ad, fc_ad = _adapter("text", svc), _adapter("fncall", svc)
    assert text_ad.tools() is None
    assert [t["function"]["name"] for t in fc_ad.tools()] == ["search", "click"]
    assert "<action>" in text_ad.format_reminder()
    assert "<action>" not in fc_ad.format_reminder()
    # task_brief 的动作面必须与 executor 实际所见一致
    assert "search[query]" in text_ad.task_brief()
    assert "search(query)" in fc_ad.task_brief()


# ---- react 面 -----------------------------------------------------------

def test_text_action_is_passed_through_verbatim():
    """不做语法修复:原样交环境,非法由 WebShop 自己判(与 ALFWorld 同纪律)。"""
    svc = _FakeService()
    ad = _adapter("text", svc)
    out = ad.react(_episode(),
                   "<reasoning>need a bottle</reasoning>\n"
                   "<action>search[stainless steel bottle]</action>")
    assert svc.seen == ["search[stainless steel bottle]"]
    assert out.acted and out.consumed_env_step
    assert out.thought == "need a bottle"
    assert out.action_norm == "search[stainless steel bottle]"
    # text 档反馈是裸文本,不包 tool 响应
    assert "<tool_response>" not in out.feedback
    assert "[Available actions]" in out.feedback


def test_text_missing_action_block_does_not_consume_a_step():
    svc = _FakeService()
    out = _adapter("text", svc).react(_episode(), "I think I should search.")
    assert svc.seen == []
    assert not out.acted and not out.consumed_env_step and not out.valid
    # 提醒里只讲格式,不伪造环境语义 —— WebShop 从不说 "Invalid action!"
    assert "<action>" in out.feedback and "Invalid action!" not in out.feedback


def test_action_validity_uses_the_page_not_the_observation():
    """★ 2026-08-16 修正(验证套件 F1 抓出):WebShop 对非法动作**静默无操作**,
    返回的仍是当前页面,从不吐 "Invalid action!" —— 旧判据"观察里含 invalid
    action"恒为 False ⇒ valid 恒 True,诊断指标全假。正确判据 = 动作是否在
    **发出前**的 clickables 里(与 web_agent_text_env.step 逐条等价)。"""
    from trace_opd.envs.webshop.adapter import action_is_available

    assert action_is_available("click[b01]", _PAGE)
    assert action_is_available("search[hat]", _PAGE)
    assert not action_is_available("click[nonexistent]", _PAGE)
    assert not action_is_available("search[  ]", _PAGE)          # 空 query
    # 商品页没有搜索框 ⇒ search 非法(环境里 browser.search 无效果)
    assert not action_is_available("search[hat]",
                                   {"has_search_bar": False, "clickables": []})
    # 大小写精确比对:环境对 radio 选项的 key 未做 lower,归一化会造成不等价
    assert not action_is_available("click[B01]", _PAGE)
    # ★ click[search] 必须判非法:初始页的 clickables 里**确实含 "search"**
    #   (搜索按钮),但环境显式排除它(`and action_arg != 'search'`)。
    #   只查 "in clickables" 会假 PASS —— 验证套件的 F1' 踩过。
    assert not action_is_available(
        "click[search]", {"has_search_bar": True, "clickables": ["search"]})

    # 端到端:非法动作仍打到环境(不做本地拦截 —— 让它可见地失败),
    # 消耗步,但 valid=False
    svc = _FakeService()
    out = _adapter("text", svc).react(
        _episode(), "<reasoning>try</reasoning><action>click[nonexistent]</action>")
    assert svc.seen == ["click[nonexistent]"]
    assert out.acted and out.consumed_env_step   # 消耗步:非法动作有成本
    assert not out.valid


def test_valid_action_is_marked_valid_and_page_state_advances():
    svc = _FakeService()
    ep = _episode()
    ad = _adapter("text", svc)
    out = ad.react(ep, "<reasoning>go</reasoning><action>click[b01]</action>")
    assert out.valid
    # 页面状态必须跟着推进 —— 否则下一步的合法性判据用的是过期页面
    assert ep.actions == {"has_search_bar": False,
                          "clickables": ["b01", "next >"]}
    out2 = ad.react(ep, "<reasoning>x</reasoning><action>search[hat]</action>")
    assert not out2.valid          # 新页面没有搜索框


def test_both_protocols_emit_the_same_action_norm():
    """★ replay/渲染两档同形的硬前提。"""
    svc_t, svc_f = _FakeService(), _FakeService()
    _adapter("text", svc_t).react(
        _episode(), "<reasoning>r</reasoning><action>click[b01]</action>")
    _adapter("fncall", svc_f).react(
        _episode(),
        '<tool_call>{"name":"click","arguments":{"target":"b01"}}</tool_call>')
    assert svc_t.seen == svc_f.seen == ["click[b01]"]
    # ⚠ 两档在**大小写**上并不同形:text 档经 parse_action 会 lower,FC 档
    # 保留原样。对 WebShop 无害(环境不敏感,见 test_action_is_lowercased),
    # 但跨档比对 action_norm 字面时须知。
    svc_t2, svc_f2 = _FakeService(), _FakeService()
    _adapter("text", svc_t2).react(
        _episode(), "<reasoning>r</reasoning><action>click[Buy Now]</action>")
    _adapter("fncall", svc_f2).react(
        _episode(),
        '<tool_call>{"name":"click","arguments":{"target":"Buy Now"}}</tool_call>')
    assert svc_t2.seen == ["click[buy now]"]
    assert svc_f2.seen == ["click[Buy Now]"]


def test_fncall_bad_call_consumes_a_step_but_never_reaches_env():
    svc = _FakeService()
    out = _adapter("fncall", svc).react(
        _episode(),
        '<tool_call>{"name":"buy","arguments":{}}</tool_call>')
    assert svc.seen == []                       # 没投给环境
    assert out.acted and out.consumed_env_step  # 但消耗步(防零成本烧轮次)
    assert not out.valid
    assert "unknown tool" in out.feedback


def test_call_to_action_mapping():
    assert _call_to_action({"name": "search", "arguments": {"query": "hat"}}) == (
        "search[hat]", None)
    assert _call_to_action({"name": "click", "arguments": {"target": "Buy Now"}}) == (
        "click[Buy Now]", None)
    for bad in ({"name": "search", "arguments": {"query": "  "}},
                {"name": "click", "arguments": {}},
                {"name": "checkout", "arguments": {}}):
        action, err = _call_to_action(bad)
        assert action == "" and err


def test_action_is_lowercased_and_that_is_safe():
    """``protocol.parse_action`` 会 lower() —— 对 WebShop 无害,两条实测依据
    (2026-08-16,162 真实服务,scratchpad/ws_case_probe.py):
      ① 页面 clickables 字面**本身全是小写**(`buy now`/`back to search`/
         `x-large`/`b09qqp3356`),模型照抄即小写;
      ② 环境**大小写不敏感**:click[Buy Now] / click[buy now] / click[BUY NOW]
         三种都成功购买(done=True,reward 0.6/0.6/0.43)。
    这条锁住"lower 是已知且已验证安全的",避免以后有人看到大小写丢失误判为 bug。
    """
    svc = _FakeService()
    out = _adapter("text", svc).react(
        _episode(), "<reasoning>r</reasoning><action>click[Buy Now]</action>")
    assert svc.seen == ["click[buy now]"]
    assert out.action_norm == "click[buy now]"


def test_done_hides_the_score():
    """信息状态红线:分数绝不回到 executor 的上下文。"""
    svc = _FakeService({"click[buy now]": {
        "observation": "Thank you for shopping", "reward": 0.73, "done": True,
        "actions": {}}})
    out = _adapter("text", svc).react(
        _episode(), "<reasoning>buy</reasoning><action>click[Buy Now]</action>")
    assert out.done and out.score == pytest.approx(0.73)
    assert "0.73" not in out.feedback and "score" not in out.feedback.lower()


def test_replay_uses_action_norm_directly():
    svc = _FakeService()
    ad = _adapter("text", svc)
    feedback, done = ad.replay_action(_episode(), "search[hat]")
    assert svc.seen == ["search[hat]"] and not done
    assert "<tool_response>" not in feedback


# ---- 指纹面 -------------------------------------------------------------

def test_fingerprint_tracks_the_protocol(monkeypatch, tmp_path):
    from dataclasses import replace
    from pathlib import Path

    from trace_opd.common.config import load_config, objective_fingerprint

    for k, v in {"TRACE_MODEL_PATH": str(tmp_path / "m"),
                 "TRACE_RESEARCHER_API_KEY": "dummy",
                 "TRACE_35B_KEY": "dummy"}.items():
        monkeypatch.setenv(k, v)
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/classic/webshop.yaml")
    assert cfg.environment.name == "webshop"
    assert cfg.webshop.protocol == "text"   # 2026-08-16 用户裁定的默认档

    def with_proto(p):
        return objective_fingerprint(
            replace(cfg, webshop=replace(cfg.webshop, protocol=p)))

    assert with_proto("text") != with_proto("fncall")


def test_config_rejects_unknown_protocol(monkeypatch, tmp_path):
    from trace_opd.common.config import load_config

    monkeypatch.setenv("TRACE_MODEL_PATH", str(tmp_path / "m"))
    monkeypatch.setenv("TRACE_RESEARCHER_API_KEY", "dummy")
    monkeypatch.setenv("TRACE_35B_KEY", "dummy")
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "run:\n  name: t\nenvironment:\n  name: webshop\n"
        "webshop:\n  protocol: json\n"
        "train:\n  model_path: /tmp/m\n", encoding="utf-8")
    with pytest.raises(ValueError, match="webshop.protocol"):
        load_config(bad)


# ---- 分片与退避面(2026-08-16:修 2026-08-12 的池泄漏事故) ---------------

def test_endpoints_parsing_and_deterministic_shard_binding():
    from trace_opd.envs.webshop.adapter import endpoints_of

    cfg = WebshopCfg(service_url="http://a:1, http://b:2/ ,http://c:3",
                     preflight=False)
    assert endpoints_of(cfg) == ["http://a:1", "http://b:2", "http://c:3"]
    with pytest.raises(ValueError):
        endpoints_of(WebshopCfg(service_url="  ", preflight=False))

    # goal_idx 取模 ⇒ 同一 goal 恒落同一分片(可复现),且连续 idx 均匀铺开
    seen = []
    for gi in range(6):
        item = dict(ITEM, goal_idx=gi)
        ad = WebshopTaskAdapter(item, cfg, http=None, gate=None)
        calls = {}

        def fake(path, payload, timeout, base=None, _c=calls):
            _c["base"] = base
            return {"sid": "s", "observation": "o", "actions": {}}

        ad._post = fake  # noqa: SLF001
        ep = ad.open_episode()
        assert ep.base == calls["base"]        # episode 记住自己的分片
        seen.append(ep.base)
    assert seen == ["http://a:1", "http://b:2", "http://c:3"] * 2


def test_episode_steps_stay_on_its_own_shard():
    """sid 是分片本地的 —— step 打错分片 = 404,必须跟着 episode 走。"""
    svc = _FakeService()
    cfg = WebshopCfg(service_url="http://a:1,http://b:2", preflight=False)
    ad = _adapter("text", svc, cfg)
    ad.react(_episode("http://b:2"),
             "<reasoning>r</reasoning><action>search[x]</action>")
    assert svc.bases == ["http://b:2"]


def test_429_is_retried_with_backoff_not_treated_as_failure(monkeypatch):
    """服务端池满回 429 = 忙,不是坏。客户端必须退避重试。"""
    import trace_opd.envs.webshop.adapter as A

    slept: list[float] = []
    monkeypatch.setattr(A.time, "sleep", lambda s: slept.append(s))

    class _Resp:
        def __init__(self, code, body=None):
            self.status_code = code
            self._body = body or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"http {self.status_code}")

        def json(self):
            return self._body

    class _Http:
        def __init__(self, codes):
            self.codes = list(codes)
            self.n = 0

        def post(self, url, json=None, timeout=None):
            c = self.codes[self.n]
            self.n += 1
            return _Resp(c, {"ok": True} if c == 200 else None)

    import contextlib
    cfg = WebshopCfg(preflight=False, retry_backoff_s=1.0,
                     retry_backoff_max_s=8.0, max_retries=4)
    ad = WebshopTaskAdapter(ITEM, cfg, http=_Http([429, 429, 200]),
                            gate=contextlib.nullcontext())
    assert ad._post("/reset", {}, timeout=1.0) == {"ok": True}
    assert slept == [1.0, 2.0]           # 指数退避

    # 一直 429 ⇒ 耗尽重试后 fail loud(静默零分会伪装成能力差)
    ad2 = WebshopTaskAdapter(ITEM, cfg, http=_Http([429] * 4),
                             gate=contextlib.nullcontext())
    with pytest.raises(RuntimeError, match="unreachable"):
        ad2._post("/reset", {}, timeout=1.0)


def test_preflight_checks_every_shard_and_goal_count_agreement():
    from trace_opd.envs.webshop.adapter import WebshopEnvFactory

    class _Http:
        def __init__(self, metas):
            self.metas = metas

        def get(self, url, timeout=None):
            base = url.rsplit("/meta", 1)[0]
            meta = self.metas.get(base)

            class R:
                status_code = 200 if meta is not None else 500

                def raise_for_status(_self):
                    if meta is None:
                        raise RuntimeError("down")

                def json(_self):
                    return meta
            return R()

    cfg = WebshopCfg(service_url="http://a:1,http://b:2", preflight=False)
    f = WebshopEnvFactory.__new__(WebshopEnvFactory)
    f._cfg = cfg
    OK = {"num_goals": 6910, "option_fix": True}

    # 一个分片挂了 ⇒ fail loud(否则落在它上面的 goal 全灭且静默)
    f._http = _Http({"http://a:1": OK})
    with pytest.raises(RuntimeError, match="shard not reachable"):
        f._preflight()

    # 两分片 goals 数不同 ⇒ goal_idx 语义跨分片错位,拒绝启动
    f._http = _Http({"http://a:1": OK,
                     "http://b:2": {"num_goals": 6900, "option_fix": True}})
    with pytest.raises(RuntimeError, match="num_goals 不一致"):
        f._preflight()

    f._http = _Http({"http://a:1": OK, "http://b:2": OK})
    f._preflight()   # 一致 ⇒ 通过


def test_build_dataset_record_carries_agent_name():
    """★ 缺 agent_name = verl trainer 在 reward 分派处炸(SearchQA 撞过)。"""
    from trace_opd.envs.webshop.build_dataset import _record

    rec = _record(7, "train", "obs text", "buy a bottle")
    assert rec["agent_name"] == "trace_opd"
    assert rec["reward_model"]["style"] == "env"
    assert rec["data_source"] == "webshop"
    assert _record(7, "test", "o", "i")["data_source"] == "webshop/test"
    assert rec["extra_info"]["goal_idx"] == 7
    # 构建期快照用 text 档 scaffold(与 adapter 默认一致)
    assert json.dumps(rec["prompt"], ensure_ascii=False).count("action") >= 1


# ---- 报数口径面(2026-08-16 用户要求:训练/测试都要 Score + Succ.)------

def test_evalm_reports_both_score_and_success_rate():
    """★ WebShop 是**连续 score** 环境,只报 SR 会丢掉"部分分进展" ——
    SR 还是 0 的时候 Score 已经在涨,那是早期最灵敏的信号。文献表头也是
    Score / Succ. 两列(Score 按惯例 ×100 显示)。"""
    from trace_opd.evalm.run_eval import summarize

    rows = [
        {"split": "test", "outcome": "SUCCESS", "won": True, "score": 1.0,
         "env_steps": 5},
        {"split": "test", "outcome": "FAILED", "won": False, "score": 0.714,
         "env_steps": 8},
        {"split": "test", "outcome": "FAILED", "won": False, "score": 0.0,
         "env_steps": 15},
    ]
    summ, incomplete = summarize(rows, "bare")
    s = summ["splits"]["test"]
    assert not incomplete
    # summarize 内部 round(...,4),容差按之
    assert s["success_rate"] == pytest.approx(1 / 3, abs=1e-4)        # Succ.
    assert s["mean_score"] == pytest.approx((1.0 + 0.714) / 3, abs=1e-4)  # Score
    # 内部值保持 [0,1] 与环境 reward 同尺度;×100 只在显示层
    assert 0.0 <= s["mean_score"] <= 1.0
    # 二值环境下两者必须恒等(ALFWorld/SearchQA 无解读负担)
    binary = [{"split": "t", "outcome": "SUCCESS", "won": True, "score": 1.0,
               "env_steps": 1},
              {"split": "t", "outcome": "FAILED", "won": False, "score": 0.0,
               "env_steps": 1}]
    b = summarize(binary, "bare")[0]["splits"]["t"]
    assert b["mean_score"] == pytest.approx(b["success_rate"], abs=1e-4)


def test_training_side_reward_extra_info_carries_score():
    """★ 训练与 val 的 Score 曲线都来自 `reward_extra_info["score"]`:
    verl 对该 dict 的**每个 key** 生成 `rollout_extra/<k>/mean`(训练)与
    `val-aux/<data_source>/<k>/{mean@N,best@N,worst@N}`(验证)。
    删掉这个键 = WebShop 训练全程看不到部分分进展(A800 生产树的旧版
    就缺它,2026-08-16 发现)。此处锁住键名不被误删。"""
    # ★ 读**文件**而不是 import:agent_loop 依赖 verl,而本仓的 verl 副本
    # 需要 py3.11+(enum.StrEnum),本地 3.10 环境下 import 会炸 —— 这个锁
    # 只需要看源码文本,不必加载模块。
    from pathlib import Path as _P

    src = (_P(__file__).resolve().parents[1]
           / "src/trace_opd/train/agent_loop.py").read_text(encoding="utf-8")
    # 两处 reward_extra_info(主 rollout 行 + research 行)都必须带 score,
    # 否则 per-key append 会产生"锯齿字典"(见 test_rollout_extra_metrics_ragged)
    blocks = src.count('output.extra_fields["reward_extra_info"]')
    assert blocks >= 2, f"reward_extra_info 装配点只有 {blocks} 处"
    assert src.count('"score": float(') >= 2, "score 未在全部装配点出现"


# ---- option 修复开关(2026-08-16 用户裁定:修掉,默认开)------------------

def test_option_fix_switch_is_part_of_the_reward_definition(monkeypatch, tmp_path):
    """★ 开关**无条件**进指纹:它把 6.63% 的任务从"恒 won=False"变成
    "可达满分" —— 那是 reward 定义本身的改变,两种口径的 run 必须指纹不同,
    否则 resume 守卫会让一个 run 前后半段用两套 reward。"""
    from dataclasses import replace
    from pathlib import Path

    from trace_opd.common.config import load_config, objective_fingerprint

    for k, v in {"TRACE_MODEL_PATH": str(tmp_path / "m"),
                 "TRACE_RESEARCHER_API_KEY": "dummy",
                 "TRACE_35B_KEY": "dummy"}.items():
        monkeypatch.setenv(k, v)
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/classic/webshop.yaml")
    assert cfg.webshop.fix_option_matching is True      # 默认打开修复

    def fp(on: bool):
        return objective_fingerprint(
            replace(cfg, webshop=replace(cfg.webshop, fix_option_matching=on)))

    assert fp(True) != fp(False)


def test_preflight_rejects_option_fix_mismatch_and_stale_service():
    """开关与服务端不一致 = 一种口径训练、另一种口径记指纹,且**完全静默** ——
    必须装配期拦下。旧版服务(无 option_fix 字段)要单独报,否则 None 会被
    当成 False 报成"不一致",把人指向错误的修法。"""
    from trace_opd.envs.webshop.adapter import WebshopEnvFactory

    class _Http:
        def __init__(self, metas):
            self.metas = metas

        def get(self, url, timeout=None):
            meta = self.metas.get(url.rsplit("/meta", 1)[0])

            class R:
                def raise_for_status(_s):
                    if meta is None:
                        raise RuntimeError("down")

                def json(_s):
                    return meta
            return R()

    def _factory(cfg):
        f = WebshopEnvFactory.__new__(WebshopEnvFactory)
        f._cfg = cfg
        return f

    cfg_on = WebshopCfg(service_url="http://a:1", preflight=False,
                        fix_option_matching=True)
    cfg_off = WebshopCfg(service_url="http://a:1", preflight=False,
                         fix_option_matching=False)

    # 旧版服务:没有 option_fix 字段 ⇒ 指向"部署新版服务端"
    f = _factory(cfg_on)
    f._http = _Http({"http://a:1": {"num_goals": 6910}})
    with pytest.raises(RuntimeError, match="没有 option_fix 字段"):
        f._preflight()

    # 配置要修复、服务端没修 ⇒ 拦下
    f._http = _Http({"http://a:1": {"num_goals": 6910, "option_fix": False}})
    with pytest.raises(RuntimeError, match="fix_option_matching=True 与服务端不一致"):
        f._preflight()

    # 反向:配置要原始口径、服务端却修了 ⇒ 同样拦下
    f2 = _factory(cfg_off)
    f2._http = _Http({"http://a:1": {"num_goals": 6910, "option_fix": True}})
    with pytest.raises(RuntimeError, match="fix_option_matching=False 与服务端不一致"):
        f2._preflight()

    # 一致 ⇒ 通过(两个方向都测,避免判据写成恒真)
    for cfg, srv in ((cfg_on, True), (cfg_off, False)):
        fx = _factory(cfg)
        fx._http = _Http({"http://a:1": {"num_goals": 6910, "option_fix": srv}})
        fx._preflight()


def test_service_option_fix_patch_is_correct_and_idempotent():
    """服务端 patch 的语义锁:把 goal 侧的 (key,value) 元组换成 value,
    其余一字不动;重复调用幂等(服务重启脚本可能多次触发)。"""
    import importlib.util
    import sys
    import types
    from pathlib import Path

    # webshop_service.py 顶层不 import web_agent_site(那是 _build_pool 才做),
    # 但 _apply_option_fix 会 —— 用一个假模块顶替,只验 patch 的行为。
    fake_goal = types.ModuleType("web_agent_site.engine.goal")
    seen: list = []

    def _orig_get_option_reward(purchased, goal_opts):
        seen.append(list(goal_opts))
        return 1.0, len(list(goal_opts))

    fake_goal.get_option_reward = _orig_get_option_reward
    pkg = types.ModuleType("web_agent_site")
    eng = types.ModuleType("web_agent_site.engine")
    eng.goal = fake_goal
    pkg.engine = eng
    for name, mod in (("web_agent_site", pkg), ("web_agent_site.engine", eng),
                      ("web_agent_site.engine.goal", fake_goal)):
        sys.modules[name] = mod
    try:
        path = (Path(__file__).resolve().parents[2]
                / "experiments/bench_env/classic_bench/webshop_service.py")
        spec = importlib.util.spec_from_file_location("_ws_service", path)
        svc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(svc)

        svc._apply_option_fix()
        patched = fake_goal.get_option_reward
        assert patched is not _orig_get_option_reward
        # 幂等:再打一次不应叠加
        svc._apply_option_fix()
        assert fake_goal.get_option_reward is patched

        # 语义:tuple → 取 value;非 tuple 原样透传
        patched(["black"], [("color", "black1901"), ("size", "9")])
        assert seen[-1] == ["black1901", "9"]
        patched(["black"], ["black1901", "9"])
        assert seen[-1] == ["black1901", "9"]
    finally:
        for name in ("web_agent_site.engine.goal", "web_agent_site.engine",
                     "web_agent_site"):
            sys.modules.pop(name, None)
