"""WebShop TaskAdapter — 文本电商购物(经典转向主线,2026-08-12 裁定)。

社区口径(arXiv 2207.01206;代码级核实见
experiments/bench_env/docs/classic-bench-env-buildout.md §5.2):近期 RL
工作(GiGPO/verl-agent)全跑 1000 商品 + 合成 goal(~6,910)小集;15 步/
episode 惯例;报数双口径 Score(100×平均 reward)+ SR(reward==1 比例)。
我们的 won=score>=1.0 派生 ⇒ score 透传连续 reward 即天然双口径。

划分口径(2026-08-12 用户裁定④,对齐 verl-agent):服务以 env.seed=0 洗牌
后的 goal 列表上,**test=valid=[0,500) / train=[500,~6910)**,训练期每次
eval 全量 500 不抽样(build_dataset 的 split_rule 是同一真源)。与 GiGPO
的"各任务槽自排列前/后 500 切分"为同分布不同样本(见 OPS-LOG §9 代码级
对拍),其 66.1 锚点按带抽样误差解读。

★ goals 确定性(2026-08-16 CD-011 根因修复):vendored `engine.py:190` 的
`price = random.uniform(*pricing[:2])` 无 seed,污染 product_prices ⇒
`get_synthetic_goals` 里 price_range 候选集合漂移 ⇒ 同一 goal_idx 在不同
实例上 price_upper 不同(实测 16.4% goals 受影响),直接污染判分。修复=
SimServer.__init__ 的 load_products 之前补 `random.seed(0)`。实证:跨实例
400 次 reset 零漂移、独立进程构建与服务内存 120/120 逐字一致。

环境 = 162 上的薄 HTTP 服务(experiments/bench_env/classic_bench/
webshop_service.py;按 goal_idx 精确 reset,实例池并发)。两个协议档,
动作空间逐字相同(环境原生 `search[q]` / `click[t]`):

  text(默认,2026-08-16 用户裁定)— <reasoning>…</reasoning> +
       <action>search[q]</action>;与 ALFWorld/SearchQA/Math 三域对齐,
       也与 GiGPO/verl-agent 的 WebShop 动作字面同款
  fncall — OpenAI 格式两工具 search(query) / click(target)

终局 = click["Buy Now"] ⇒ 环境返回 done + [0,1] reward(属性/选项/价格
模糊匹配)。信息状态红线:gold attributes 不进任何提示词/反馈(评分在
服务端 env 内部);gt_text=None。

重放:WebShop 是确定性模拟器(同 goal 同动作序列同结果)⇒ replay =
重放工具调用。episode 句柄持远端 sid,close() 必须 release(池资源)。
"""
from __future__ import annotations

import json
import threading
import time

import httpx

from ...common.config import WebshopCfg
from ...core import protocol
from ...core.adapter import ActionOutcome
from ..pytool import clip_middle


def _format_reminder_fc() -> str:
    return protocol.fc_format_reminder("search", "click")


def _format_reminder_text() -> str:
    """裸文本提醒(text 档下 assistant 未发出合法动作,反馈不是 tool 响应)。

    ★ 不加环境习语前缀:WebShop 对非法动作静默无操作,**从不吐
    "Invalid action!"**(该字符串只在我们不用的 web_agent_site_env.py 里,
    且是 print)。加一个环境根本不会说的前缀 = 向模型伪造环境语义。
    ALFWorld 的 "Nothing happens. " 前缀成立,是因为那是它真实的响应。
    """
    return protocol.DEFAULT_FORMAT_REMINDER


def _no_wrap(body: str) -> str:
    """text 档的反馈形态 = 裸文本(user 轮),不包 tool 响应。"""
    return body


def action_is_available(env_action: str, actions: dict) -> bool:
    """动作在**当前页面**是否合法 —— 与 vendored `WebAgentTextEnv.step` 的
    判据逐条等价。

    ★ 2026-08-16 修正(验证套件 F1 抓出):WebShop 对非法动作是
    ``status = dict(reward=0, done=False)`` —— **静默无操作、返回当前页面**,
    从不吐 "Invalid action!"(该字符串只存在于我们不用的
    `web_agent_site_env.py` 且是 print)。原判据"观察里含 invalid action"
    ⇒ **恒为 False ⇒ valid 恒为 True**,诊断指标完全失真(此前所有 WebShop
    探针的 invalid 率读数都是假的 0)。

    环境的真实判据(`web_agent_text_env.py:101-110`):
      search — action_name=='search' 且 arg 非空(不检查 has_search_bar,
               但无搜索框的页面上 browser.search 无效果 ⇒ 一并要求它)
      click  — arg in self.text_to_clickable.keys() **且 arg != 'search'**
    而 `get_available_actions()` 返回的 ``clickables`` **就是**
    `text_to_clickable.keys()`(:136-146)⇒ 用它判定与环境逐条一致。

    ★ `!= 'search'` 这一条不能漏:初始页面的 clickables 里**确实含
    "search"**(那是搜索按钮),但环境显式排除了 `click[search]` —— 只查
    "in clickables" 会把它判成合法(2026-08-16 验证套件的 F1' 曾因此假 PASS)。
    大小写不做归一:环境对 buttons/links 的 key 已 lower、对 radio 选项的
    key 未 lower,精确比对才与之等价。
    """
    if env_action.startswith("search[") and env_action.endswith("]"):
        return bool(actions.get("has_search_bar")) and bool(
            env_action[len("search["):-1].strip())
    if env_action.startswith("click[") and env_action.endswith("]"):
        target = env_action[len("click["):-1]
        return (target != "search"
                and target in (actions.get("clickables") or []))
    return False


def _call_to_action(call: dict) -> tuple[str, str | None]:
    """FC 工具调用 → 环境原生动作字符串;失败返回 (\"\", 错误说明)。"""
    name = call.get("name")
    if name == "search":
        arg = str(call.get("arguments", {}).get("query", "") or "").strip()
        if not arg:
            return "", "search: empty query."
        return f"search[{arg}]", None
    if name == "click":
        arg = str(call.get("arguments", {}).get("target", "") or "").strip()
        if not arg:
            return "", ("click: empty target — pass the exact clickable "
                        "text as 'target'.")
        return f"click[{arg}]", None
    return "", f"unknown tool: {name!r}. Available: search, click."


TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the shop's catalog. Only legal when the current "
                "page has a search bar (the action list shows "
                "has_search_bar). Returns a results page listing products "
                "with their ids and prices. Use focused keyword queries "
                "drawn from the instruction (product type, key attributes); "
                "over-long queries match poorly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword query for the product search.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": (
                "Click one element on the current page. The target must be "
                "copied VERBATIM from the current clickables list (a "
                "product id like 'B09P5CRVQ6', an option value like "
                "'large' or 'green stripe', 'Description', 'Back to "
                "Search', '< Prev', or 'Buy Now'). Select every required "
                "option BEFORE clicking 'Buy Now' — buying ends the "
                "episode immediately and the purchase is graded as-is; "
                "there is no undo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Exact clickable text from the current page.",
                    },
                },
                "required": ["target"],
            },
        },
    },
]

#: 两档共用的任务面措辞(角色 / 工作风格 / 判分语义)—— 协议档之间**只有
#: 动作发出方式不同**,任务本身逐字相同,这样两档读数才可归因到协议。
_ROLE_AND_STYLE = (
    "You are a shopping agent in a text-based e-commerce environment. "
    "Your goal is to find and buy a product matching the instruction "
    "(product type, attributes, options, and price limit).\n\n"
    "Working style:\n"
    "- Search with focused keywords first; open a promising product id "
    "to inspect it.\n"
    "- On the product page, select every option the instruction asks "
    "for (click each option value), then buy it.\n"
    "- The purchase is graded on product match, selected options, and "
    "price — buying a wrong or unconfigured product scores poorly. Act "
    "within the step budget; do not browse aimlessly."
)

SYSTEM_PROMPT_FC = (
    _ROLE_AND_STYLE + "\n- Reason as normal text before each tool call."
)


def system_prompt_text() -> str:
    """text 档 scaffold —— **必须是调用时求值**:含推理标签,而
    ``protocol.configure_reasoning_tag()`` 在入口注入后才定标签,模块常量
    会把旧值绑死在导入方名字上(2026-08-16 标签统一时踩过,见
    docs/notes/2026-08-16-reasoning-tag-unification.md §"实现陷阱")。

    动作字面 ``search[q]`` / ``click[t]`` 是 WebShop 原生语法,与
    GiGPO/verl-agent 的 prompt 逐字同款 —— 动作空间不因协议档而变。
    """
    ro, rc = protocol.rtag_open(), protocol.rtag_close()
    return (
        _ROLE_AND_STYLE + "\n\n"
        "Actions (exactly one per turn):\n"
        "  search[query]  — keyword search over the catalog; only legal when "
        "the page has a search bar\n"
        "  click[target]  — click an element; the target must be copied "
        "verbatim from the page's clickable list. click[Buy Now] purchases "
        "the currently configured product and ends the episode.\n\n"
        "On every turn respond with exactly this format and nothing else:\n"
        f"{ro}brief reasoning about the current page and what to do next{rc}\n"
        "<action>one action</action>\n\n"
        # one-shot 格式示例:Qwen2.5 系 zero-shot 对自定义 XML 标签不服从
        # (ALFWorld 上 274/274 全灭 DEGENERATE:format,单条 system 内示例即
        # 0%→99.9%,见记忆 qwen25-format-compliance-oneshot)。示例为单步
        # 通用常识、与任何具体 goal 无关 —— 不触信息状态红线,两臂共用。
        "Example of a correctly formatted reply (for an unrelated "
        "instruction):\n"
        f"{ro}I need a stainless steel water bottle under 30 dollars. "
        f"Nothing is on screen yet, so I will search with the key "
        f"attributes.{rc}\n"
        "<action>search[stainless steel water bottle]</action>"
    )


def initial_obs_of(item: dict) -> str:
    """初始观察 = reset 后的首页 observation(含 Instruction;build_dataset
    经服务端固化——WebShop reset(session=goal_idx) 幂等确定)。"""
    return str(item["initial_obs"])


def _render_actions(actions: dict) -> str:
    has_search = bool(actions.get("has_search_bar"))
    clickables = list(actions.get("clickables") or [])
    parts = []
    if has_search:
        parts.append("search bar: available")
    if clickables:
        parts.append("clickables: " + json.dumps(clickables, ensure_ascii=False))
    return "\n".join(parts) if parts else "(no actions available)"


def endpoints_of(cfg: WebshopCfg) -> list[str]:
    """``service_url`` 支持逗号分隔多个分片(多进程绕开单进程 GIL 吞吐墙)。"""
    eps = [u.strip().rstrip("/") for u in str(cfg.service_url).split(",") if u.strip()]
    if not eps:
        raise ValueError("webshop.service_url 为空")
    return eps


class WebshopEpisode:
    """EnvHandle:远端服务上的一个活跃 session(实例池资源)。close()
    幂等 release——泄漏会耗干池并发。

    ``base`` 是本 episode 绑定的**分片**:sid 是分片本地的,reset/step/release
    必须打同一个分片(借鉴 verl-agent 的"槽位静态绑定")。"""

    def __init__(self, sid: str, initial_obs: str, http: httpx.Client,
                 cfg: WebshopCfg, base: str,
                 actions: dict | None = None) -> None:
        self.sid = sid
        self.base = base
        self._initial_obs = initial_obs
        self._http = http
        self._cfg = cfg
        self.closed = False
        #: 当前页面的可用动作(reset/step 每次刷新)。**动作合法性判据的唯一
        #: 依据** —— 见 `action_is_available` 的说明。
        self.actions: dict = dict(actions or {})

    @property
    def initial_obs(self) -> str:
        return self._initial_obs

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                self._http.post(
                    f"{self.base}/release",
                    json={"sid": self.sid},
                    timeout=self._cfg.step_timeout_s,
                )
            except Exception:  # noqa: BLE001 — release 尽力;done 时服务端已自动归还
                pass


class WebshopTaskAdapter:
    """One adapter per goal. item = parquet extra_info(goal_idx/task_id/
    instruction/initial_obs/split)。"""

    def __init__(
        self, item: dict, cfg: WebshopCfg,
        http: httpx.Client, gate: threading.Semaphore,
    ) -> None:
        self._item = dict(item)
        self._cfg = cfg
        self._http = http
        self._gate = gate

    # ---- 身份面 ----------------------------------------------------------

    @property
    def task_id(self) -> str:
        return str(self._item["task_id"])

    @property
    def task_type(self) -> str:
        return str(self._item.get("task_type") or "webshop")

    @property
    def initial_obs(self) -> str:
        return str(self._item["initial_obs"])

    # ---- researcher 面 ---------------------------------------------------

    def task_brief(self) -> str:
        """给 researcher 的环境说明 —— 动作面必须与 executor 实际所见一致
        (否则 doc 会教一套 executor 用不了的语法)。"""
        if self._cfg.protocol == "fncall":
            action_face = (
                "The executor reasons in text and may call tools (one call "
                "per turn):\n"
                "  search(query) — keyword search over the catalog (only when "
                "the page has a search bar)\n"
                "  click(target) — click an element, verbatim from the "
                "current clickables list; click('Buy Now') ends the episode\n"
            )
        else:
            action_face = (
                "The executor emits one action per turn as "
                "<action>…</action> after a reasoning block:\n"
                "  search[query] — keyword search over the catalog (only when "
                "the page has a search bar)\n"
                "  click[target] — click an element, verbatim from the "
                "current clickables list; click[Buy Now] ends the episode\n"
            )
        return (
            "Environment: a text-based e-commerce site (search page, "
            "results page, product page). " + action_face +
            "Grading: the purchased product is scored in [0,1] on type, "
            "attribute, option, and price match against the instruction; "
            "success means a full match (score 1.0). The executor never "
            "sees the score.\n"
            f"Instruction: {self._item.get('instruction', '')}\n"
            f"Budget: the executor has {protocol.MAX_ENV_STEPS} actions "
            "per episode."
        )

    def executor_scaffold(self) -> str:
        if self._cfg.protocol == "fncall":
            return SYSTEM_PROMPT_FC
        return system_prompt_text()

    def tools(self) -> list[dict] | None:
        """fncall:两工具;text:无 tools(动作走 <action> 文本)。"""
        return TOOLS if self._cfg.protocol == "fncall" else None

    def gt_text(self) -> str | None:
        """gold attributes/options 在服务端评分内部——绝不进 researcher。"""
        return None

    def domain_notes(self) -> str:
        """喂给 narrator 的"该关注什么"。

        ★ WebShop 比别的域更需要这段:非法动作**静默无操作**(环境不吐任何
        提示,页面原样返回),narrator 若不知道,会把"点了但没反应"如实叙述成
        "点了" —— researcher 就永远学不到"target 必须逐字来自 clickables"
        这个最高频的失败模式。空 domain_notes 是基类默认,不是本域的答案。
        """
        return (
            "Key events in this environment: the search queries tried and how "
            "relevant the returned products were; what the product page "
            "revealed (attributes, available options, price vs the "
            "instruction's limit); which options were selected before buying; "
            "and — most importantly — clicks that had no effect. WebShop "
            "silently ignores an invalid action and re-shows the same page, "
            "with no error message: if the page did not change after a click, "
            "the target was not in the page's clickable list (it must be "
            "copied verbatim). On the very first page the only legal move is "
            "a search — the word 'search' appears in the clickable list but "
            "clicking it does nothing. Note whether the purchase was made "
            "with all requested options set."
        )

    # ---- 环境面 ----------------------------------------------------------

    def open_episode(self) -> WebshopEpisode:
        # 确定性分片路由:同一 goal 恒落同一分片 ⇒ 可复现,且 goal_idx 连续
        # 取模天然均匀。
        eps = endpoints_of(self._cfg)
        base = eps[int(self._item["goal_idx"]) % len(eps)]
        resp = self._post(
            "/reset", {"goal_idx": int(self._item["goal_idx"])},
            timeout=self._cfg.reset_timeout_s, base=base,
        )
        actions = resp.get("actions") or {}
        live_obs = (
            f"{resp['observation']}\n\n[Available actions]\n"
            f"{_render_actions(actions)}"
        )
        return WebshopEpisode(resp["sid"], live_obs, self._http, self._cfg,
                              base, actions)

    # ---- 交互面 ----------------------------------------------------------

    def react(self, env: WebshopEpisode, assistant_text: str) -> ActionOutcome:
        """协议分派。两档的 ``action_norm`` 统一为**环境原生动作字符串**
        (``search[q]`` / ``click[t]``)⇒ replay 与轨迹渲染两档同形。"""
        if self._cfg.protocol == "fncall":
            return self._react_fncall(env, assistant_text)
        return self._react_text(env, assistant_text)

    def _react_text(self, env: WebshopEpisode, assistant_text: str) -> ActionOutcome:
        """text 档:``<action>`` 内容**原样**交环境判定 —— 不做语法修复,
        非法动作由 WebShop 自己回 "Invalid action!"(与 ALFWorld 的
        "让引擎可见地失败" 同纪律,也与 GiGPO 直接投文本给 env 同口径)。"""
        command = protocol.parse_action(assistant_text)
        if command is None:
            return ActionOutcome(
                thought=assistant_text.strip(),
                action_raw="",
                action_norm="",
                feedback=_format_reminder_text(),
                done=False,
                score=0.0,
                acted=False,
                valid=False,
                consumed_env_step=False,
            )
        thought = protocol.parse_reasoning(assistant_text) or assistant_text.strip()
        norm = self.normalize_action(command)
        feedback, done, score, valid = self._dispatch(env, norm, "", _no_wrap)
        return ActionOutcome(
            thought=thought,
            action_raw=command,
            action_norm=norm,
            feedback=feedback,
            done=done,
            score=score,
            acted=True,
            valid=valid,
            consumed_env_step=True,
        )

    def _react_fncall(self, env: WebshopEpisode, assistant_text: str) -> ActionOutcome:
        calls = protocol.parse_tool_calls(assistant_text, tools=self.tools())
        thought = protocol.strip_tool_calls(assistant_text)
        if not calls:
            return ActionOutcome(
                thought=assistant_text.strip(),
                action_raw="",
                action_norm="",
                feedback=_format_reminder_fc(),
                done=False,
                score=0.0,
                acted=False,
                valid=False,
                consumed_env_step=False,
            )
        call = calls[0]
        extra_note = (
            f"\n(note: {len(calls) - 1} extra tool call(s) in the same reply "
            "were ignored — one call per turn)"
            if len(calls) > 1 else ""
        )
        env_action, err = _call_to_action(call)
        if err is not None:
            # 发起了调用即消耗环境步(与 ALFWorld/Math/SSB 同语义:否则
            # unknown/空调用可零成本烧满轮次,且 acted=True 才不重置
            # DEGENERATE breaker)。
            return ActionOutcome(
                thought=thought or assistant_text.strip(),
                action_raw=json.dumps(call, ensure_ascii=False),
                action_norm="",
                feedback=protocol.tool_response_wrap(err + extra_note),
                done=False,
                score=0.0,
                acted=True,
                valid=False,
                consumed_env_step=True,
            )
        feedback, done, score, valid = self._dispatch(
            env, env_action, extra_note, protocol.tool_response_wrap)
        return ActionOutcome(
            thought=thought or assistant_text.strip(),
            action_raw=json.dumps(call, ensure_ascii=False),
            action_norm=env_action,
            feedback=feedback,
            done=done,
            score=score,
            acted=True,
            valid=valid,
            consumed_env_step=True,
        )

    def replay_action(self, env: WebshopEpisode, action_norm: str) -> tuple[str, bool]:
        wrap = (protocol.tool_response_wrap
                if self._cfg.protocol == "fncall" else _no_wrap)
        feedback, done, _score, _valid = self._dispatch(env, action_norm, "", wrap)
        return feedback, done

    def _dispatch(
        self, env: WebshopEpisode, env_action: str, extra_note: str, wrap,
    ) -> tuple[str, bool, float, bool]:
        """把一个环境原生动作投给服务并组装反馈。``wrap`` 决定反馈形态:
        FC 档包成 tool 响应,text 档裸文本(与推理引擎的渲染同形)。"""
        # ★ 合法性必须用**发出动作前**的可用动作集判定(环境在 step 内先
        # 刷新 text_to_clickable 再判,那份就是上一次返回给我们的 actions)。
        valid = action_is_available(env_action, env.actions)
        resp = self._post(
            "/step", {"sid": env.sid, "action": env_action},
            timeout=self._cfg.step_timeout_s, base=env.base,
        )
        env.actions = dict(resp.get("actions") or {})
        done = bool(resp.get("done"))
        score = float(resp.get("reward") or 0.0)
        obs = str(resp.get("observation") or "")
        if done:
            # 信息状态红线:分数不反馈(won/score 只入 trajectory)。
            body = "Purchase completed. The episode has ended."
        else:
            body = (
                f"{clip_middle(obs, self._cfg.obs_max_chars)}\n\n"
                f"[Available actions]\n{_render_actions(resp.get('actions') or {})}"
            )
        return wrap(body + extra_note), done, score, valid

    def _post(self, path: str, payload: dict, timeout: float,
              base: str | None = None) -> dict:
        """服务不可达按环境故障 raise(fail loud);瞬断重试一次。

        **429 单独处理**:服务端池满时快速回 429(不是坏,只是忙)。这里指数
        退避重试而不是当故障 —— 2026-08-12 的池泄漏事故里,服务端曾以 120s
        阻塞代替 429,导致"客户端超时放弃 → handler 事后占槽 → 池永久干涸"。
        服务端改成快速 429 后,退避重试就是配套的另一半。
        """
        target = (base or endpoints_of(self._cfg)[0]) + path
        last_err: Exception | None = None
        backoff = self._cfg.retry_backoff_s
        for attempt in range(self._cfg.max_retries):
            try:
                with self._gate:
                    resp = self._http.post(target, json=payload, timeout=timeout)
                if resp.status_code == 429:
                    last_err = RuntimeError("pool busy (429)")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, self._cfg.retry_backoff_max_s)
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt + 1 < self._cfg.max_retries:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, self._cfg.retry_backoff_max_s)
        raise RuntimeError(
            f"webshop service unreachable at {target}: {last_err}"
        )

    # ---- 文本动作协议件(平凡实现) --------------------------------------

    def normalize_action(self, raw: str) -> str:
        """不做语义修复(与 ALFWorld 同纪律:非法动作必须可见地失败)。

        注:text 档下 ``protocol.parse_action`` 已把命令 lower(),对 WebShop
        无害 —— 2026-08-16 在 162 真实服务上实测(scratchpad/ws_case_probe.py):
        ① 页面 clickables 字面本身全小写(``buy now``/``back to search``/
        ``x-large``/asin ``b09qqp3356``),模型照抄即小写;② 环境大小写不敏感,
        ``click[Buy Now]``/``click[buy now]``/``click[BUY NOW]`` 都成功购买。
        """
        return raw.strip()

    def observation_indicates_invalid(self, observation: str) -> bool:
        """★ WebShop **无法**从观察判定动作是否合法。

        环境对非法动作静默无操作(`web_agent_text_env.py:109-110` 的
        `status = dict(reward=0, done=False)`),返回的仍是当前页面 —— 没有
        任何 "Nothing happens." 之类的习语可查(ALFWorld 有,所以那边这个
        接口成立)。合法性一律走 `action_is_available(动作, 发出前的 actions)`。

        本方法保留是因为它是 TaskAdapter 协议的一部分(通用文本动作协议件
        `protocol.text_action_react` 会调),恒返回 False 表示"观察不携带该
        信息",而不是"动作都合法"。WebShop 的 react 自实现,不经该协议件。
        """
        return False

    def format_reminder(self) -> str:
        if self._cfg.protocol == "fncall":
            return _format_reminder_fc()
        return _format_reminder_text()


class WebshopEnvFactory:
    """EnvFactory:进程级一个;共享 httpx 连接池 + 请求并发门;装配期
    真实 ping 服务(fail loud)。"""

    def __init__(self, cfg: WebshopCfg) -> None:
        self._cfg = cfg
        self._http = httpx.Client(
            limits=httpx.Limits(
                max_connections=cfg.request_concurrency,
                max_keepalive_connections=cfg.request_concurrency,
            )
        )
        self._gate = threading.Semaphore(cfg.request_concurrency)
        if cfg.preflight:
            self._preflight()

    def _preflight(self) -> None:
        """★ 每个分片都要验:一个分片没起来 = 落到它上面的那批 goal 全灭,
        而且是静默的(其余分片照常工作)⇒ 必须装配期 fail loud。同时核对
        各分片 num_goals 一致 —— 不一致意味着它们加载了不同的 goals(正是
        2026-08-16 修掉的那类不一致),goal_idx 语义会跨分片错位。"""
        metas: dict[str, int] = {}
        fixes: dict[str, object] = {}
        for base in endpoints_of(self._cfg):
            try:
                resp = self._http.get(f"{base}/meta",
                                      timeout=self._cfg.step_timeout_s)
                resp.raise_for_status()
                meta = resp.json()
                if int(meta.get("num_goals") or 0) <= 0:
                    raise RuntimeError(f"service reports no goals: {meta}")
                metas[base] = int(meta["num_goals"])
                fixes[base] = meta.get("option_fix")
            except Exception as exc:
                raise RuntimeError(
                    "webshop preflight failed — shard not reachable/valid at "
                    f"{base}: {exc}"
                ) from exc
        if len(set(metas.values())) > 1:
            raise RuntimeError(
                f"webshop 分片的 num_goals 不一致 {metas} —— goal_idx 语义会"
                "跨分片错位,拒绝启动"
            )
        # ★ option 修复开关必须与服务端一致 —— 它定义 reward,两边不一致
        # 就会用一种口径训练、另一种口径记指纹,而且**完全静默**。
        # 先查缺失(旧版服务)再查不一致 —— 否则 None 会被当成 False 报成
        # "不一致",指向错误的修法。
        stale = [b for b, v in fixes.items() if v is None]
        if stale:
            raise RuntimeError(
                f"webshop 分片 {stale[:3]} 的 /meta 没有 option_fix 字段 —— "
                "是旧版服务端,请部署 classic_bench/webshop_service.py 的当前"
                "版本后重启(该字段是 reward 口径的自证)"
            )
        want = bool(self._cfg.fix_option_matching)
        bad = {b: v for b, v in fixes.items() if bool(v) is not want}
        if bad:
            raise RuntimeError(
                f"webshop.fix_option_matching={want} 与服务端不一致:{bad}"
                " —— 服务端用 `--option-fix on|off` 对齐后重启(它改变 reward"
                " 定义,不允许两边分歧)"
            )

    def adapter_from_item(self, item: dict) -> WebshopTaskAdapter:
        return WebshopTaskAdapter(item, self._cfg, self._http, self._gate)
