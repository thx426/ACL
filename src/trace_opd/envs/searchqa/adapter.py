"""SearchQA TaskAdapter — Search-R1-style wiki-search QA(经典转向主线,
2026-08-12 用户裁定:Search-R1 风格,非字面 SearchQA 数据集)。

社区口径(arXiv 2503.09516,代码级核实见
experiments/bench_env/docs/classic-bench-env-buildout.md §5.1):训练
NQ+HotpotQA 混合 169,615;评测七集 EM;e5-base-v2 + wiki-18 + topk=3;
reward 纯二值 EM;max_turns=4。社区原版是 <search>/<answer> tag 文本协议,
我们统一 FC(PG-3 评测训练同构),两工具:

  search(query)   — 远端稠密检索服务(HTTP POST /retrieve),返回 topk 文档
  submit(answer)  — 终局:EM 判分(normalize 后与 golden_answers 任一精确匹配)

判分内嵌本文件(normalize/em 语义与 verl qa_em.py 一致),**不走其
extract_solution**——那条路径要求 `<answer>` 匹配数>1(prompt 模板示例占位
陷阱),FC 形态下 submit 参数直接判,天然绕开。

信息状态红线:golden_answers 随 extra_info 携带,仅评分路径读取,零进
提示词/反馈(submit 反馈恒 "Submitted.");gt_text=None(答案即评测本体;
特权信号=gold docs,由数据/researcher 层供给,不经本 adapter)。

环境侧无状态:检索服务是只读索引,episode 无资产 ⇒ initial_obs 即 live,
重放 = 重放工具调用(检索确定性:faiss Flat 精确检索,同 query 同 topk
同结果)。
"""
from __future__ import annotations

import json
import queue
import re
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from ...common.config import SearchQACfg
from ...core import protocol
from ...core.adapter import ActionOutcome
from ..pytool import clip_middle


def _format_reminder() -> str:
    return protocol.fc_format_reminder("search", "submit")


TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search a fixed 2018 Wikipedia snapshot with a dense "
                "retriever and return the top matching passages (title + "
                "text). The index is static: identical queries always "
                "return identical passages, and information newer than the "
                "snapshot does not exist. Issue focused queries — one "
                "entity or fact per call works far better than a long "
                "compound question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The search query — a short, specific phrase "
                            "naming the entity or fact to look up."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "Submit the final answer and end the episode. Grading is "
                "exact-match after normalization (lowercase, strip "
                "articles and punctuation), so answer with the minimal "
                "span only — the entity, name, date, or number itself, "
                "not a sentence. No grading feedback is ever returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": (
                            "The final answer span only — no reasoning, no "
                            "'Answer:' prefix, no surrounding sentence."
                        ),
                    },
                },
                "required": ["answer"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a research assistant answering factual questions with the "
    "help of a Wikipedia search tool.\n\n"
    "Working style:\n"
    "- Reason as normal text before any tool call; decide what fact is "
    "missing, then search for exactly that.\n"
    "- Multi-hop questions need multiple searches: resolve one entity "
    "first, then search for the next fact using what you found.\n"
    "- Passages are evidence, not truth — cross-check when passages "
    "conflict.\n"
    "- When the answer is established, call submit with the minimal "
    "answer span (entity / name / date / number). Submitting ends the "
    "episode.\n\n"
    # 2026-08-13 用户裁定加格式提醒。7B bare 探针实测 18 条样本中 4 条
    # (22%)出现格式失误,失误模式单一:把调用写成纯文本(如
    # `submit({"answer": "Sonnac"})`),白费一轮。ALFWorld 可用 30 轮吸收,
    # SearchQA 只有 4 步预算 ⇒ 一次失误即 25% 预算。此处**只用自然语言**
    # 提醒走工具通道,绝不手写调用语法(语法由 chat template 按 TOOLS 原生
    # 渲染,手写会在 hermes/qwen35_xml 之间串味)。
    "Every action must go through an actual tool call. Never write a call "
    "as plain text in your reply (e.g. typing submit(...) as text does "
    "nothing and wastes one of your few turns)."
)


# --- 文本标签协议(2026-08-13 用户裁定回退)-------------------------------
# 依据:Search-R1(scripts/data_process/nq_search.py::make_prefix)与 SEED
# (agent_system/environments/prompts/search.py)**都用纯文本标签,无 FC**:
#   <think>推理</think> → <search>查询</search> → 环境回 <information>…</information>
#   → <answer>答案</answer>;且两家 prompt 内都自带 one-shot 示例
#   ("For example, <answer> Beijing </answer>")。
# 动因:Qwen2.5 系(2024)对 function-call 的遵循不足——FC 档实测 11.8% 轨迹
# 因连续格式违规被判 DEGENERATE:format(平均每条违规 3 次),而 4 步预算下
# 一次失误即 25%。改文本协议同时消除该损失并与两家口径对齐。
# 推理标签(2026-08-16 用户裁定):原本沿用 Search-R1 原生的 <think> —— 依据
# 是"实测 Qwen2.5 的 chat template 不剥离历史轮 <think> 块"。该依据只对
# Qwen2.5 成立:thinking-family 模板(Qwen3.5/3.6,含拟用作分布干预教师的
# 35B)**会**把历史 assistant 轮截断到最后一个 </think> 之后,推理整段丢失。
# 故统一改用 protocol.REASONING_TAG(默认 <reasoning>,与 ALFWorld 一致)。
# ⚠ 代价记账:与 Search-R1 / SEED 的 prompt 不再逐字同源(其余措辞未动,
#   仍是两家的 <search>/<answer>/<information> 协议与 one-shot 结构)。
# ★ 必须是**调用时求值**的函数:模块常量会被 `from … import` 绑定旧值,
#   configure_reasoning_tag 之后不更新。
def system_prompt_text() -> str:
    ro, rc = protocol.rtag_open(), protocol.rtag_close()
    return (
        "You answer factual questions with the help of a Wikipedia search "
        "engine, reasoning step by step.\n\n"
        f"Every turn you must first reason inside {ro} and {rc}. "
        "After reasoning, choose exactly one of the following actions "
        "(never both in one reply):\n"
        "(1) If you lack some knowledge, call the search engine as "
        "<search> your query </search>. It returns the top passages between "
        "<information> and </information>.\n"
        "(2) If you already know enough, give the final answer inside "
        "<answer> and </answer>, with no explanation — just the minimal "
        "span (entity / name / date / number). For example, "
        "<answer> Beijing </answer>.\n\n"
        "Worked example of one turn:\n"
        f"{ro}I need the birth year of the director, but I do not know who "
        f"directed the film yet, so I will look that up first.{rc}\n"
        "<search> director of the film Inception </search>\n\n"
        "Notes: multi-hop questions need several searches — resolve one entity "
        "first, then search the next fact using what you found. Passages are "
        "evidence, not truth. Answering ends the episode; there is no partial "
        "credit."
    )


_TEXT_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)
_TEXT_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def parse_text_action(assistant_text: str) -> dict | None:
    """把文本标签解析成与 FC 同构的 call dict —— 下游 _dispatch 零改动。

    取**最后一个**标签(模型可能先复述格式再给真动作);search 与 answer
    同现时以 answer 优先(它是终局动作,Search-R1 同序)。
    """
    t = assistant_text or ""
    ans = _TEXT_ANSWER_RE.findall(t)
    if ans:
        return {"name": "submit", "arguments": {"answer": ans[-1].strip()}}
    q = _TEXT_SEARCH_RE.findall(t)
    if q:
        return {"name": "search", "arguments": {"query": q[-1].strip()}}
    return None


def text_thought(assistant_text: str) -> str:
    """thought 记账:优先推理块,缺失则取动作标签之前的正文。"""
    t = assistant_text or ""
    th = protocol.reasoning_block_re().findall(t)
    if th:
        return th[-1].strip()
    head = re.split(r"<search>|<answer>", t, maxsplit=1, flags=re.IGNORECASE)[0]
    return head.strip()


def text_format_reminder() -> str:
    """格式提醒(Search-R1 同款措辞:重述两个动作的确切写法)。"""
    return (
        "\n(Your reply contained neither a <search> nor an <answer> block. "
        "To search, put the query between <search> and </search>. To finish, "
        "put the final answer between <answer> and </answer>, "
        "e.g. <answer> Beijing </answer>. Let me try again.)\n"
    )


def text_obs_wrap(body: str) -> str:
    """环境观察包裹:Search-R1/SEED 均用 <information>…</information>。"""
    return f"\n\n<information>{body}</information>\n\n"


# ---- EM 判分(verl qa_em.py 同语义,内嵌避其 extract_solution 陷阱) ----

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_check(answer: str, golden_answers: list[str]) -> bool:
    norm = normalize_answer(answer)
    return any(normalize_answer(str(g)) == norm for g in golden_answers if str(g).strip())


def initial_obs_of(item: dict) -> str:
    """初始观察 = 纯问题文本(确定性锚点;答题纪律在 scaffold)。"""
    return str(item["question"])


class SearchQAEpisode:
    """EnvHandle:无状态占位(检索服务只读共享,无 per-episode 资产)。"""

    def __init__(self, initial_obs: str) -> None:
        self._initial_obs = initial_obs
        self.closed = False

    @property
    def initial_obs(self) -> str:
        return self._initial_obs

    def close(self) -> None:
        self.closed = True


class SearchQATaskAdapter:
    """One adapter per question. item = parquet extra_info(自足:question/
    golden_answers/task_id/task_type/split)。"""

    def __init__(
        self, item: dict, cfg: SearchQACfg,
        http: httpx.Client, gate: threading.Semaphore,
        batcher: "_SearchBatcher",
    ) -> None:
        self._item = dict(item)
        self._cfg = cfg
        self._http = http
        self._gate = gate
        self._batcher = batcher

    # ---- 身份面 ----------------------------------------------------------

    @property
    def task_id(self) -> str:
        return str(self._item["task_id"])

    @property
    def task_type(self) -> str:
        """train = 常量 "searchqa";valid = 源 benchmark 名(nq/hotpotqa/...,
        build_dataset 写入),供 by-family 曲线。"""
        return str(self._item.get("task_type") or "searchqa")

    @property
    def initial_obs(self) -> str:
        return str(self._item["initial_obs"])

    # ---- researcher 面 ---------------------------------------------------

    def task_brief(self) -> str:
        return (
            "Environment: one open-domain factual question over a fixed "
            "2018 Wikipedia snapshot. The executor reasons in text and may "
            "call tools (one call per turn):\n"
            "  search(query) — dense retrieval over the snapshot, returns "
            "top passages (title + text); deterministic for identical "
            "queries\n"
            "  submit(answer) — submit the final answer span and end the "
            "episode\n"
            "Grading: exact match after normalization (lowercase, strip "
            "articles/punctuation) against reference answers. Score is "
            "binary. The executor never sees any grading feedback "
            "(\"Submitted.\" is all it gets back).\n"
            f"Question: {self._item['question']}\n"
            f"Budget: the executor has {protocol.MAX_ENV_STEPS} tool calls "
            "per episode; reasoning between calls is free-form text."
        )

    def executor_scaffold(self) -> str:
        if getattr(self._cfg, "protocol", "fncall") == "text":
            return system_prompt_text()
        return SYSTEM_PROMPT

    def tools(self) -> list[dict] | None:
        # 文本协议不渲染 tools(否则 chat template 会同时教模型 FC 语法,
        # 与 <search>/<answer> 打架)。
        if getattr(self._cfg, "protocol", "fncall") == "text":
            return None
        return TOOLS

    def gt_text(self) -> str | None:
        """golden_answers 即评测本体——绝不进 researcher。"""
        return None

    def domain_notes(self) -> str:
        return ""

    # ---- 环境面 ----------------------------------------------------------

    def open_episode(self) -> SearchQAEpisode:
        return SearchQAEpisode(self.initial_obs)

    # ---- 交互面 ----------------------------------------------------------

    def react(self, env: SearchQAEpisode, assistant_text: str) -> ActionOutcome:
        if getattr(self._cfg, "protocol", "fncall") == "text":
            return self._react_text(env, assistant_text)
        return self._react_fncall(env, assistant_text)

    def _react_text(self, env: SearchQAEpisode, assistant_text: str) -> ActionOutcome:
        """文本标签协议(Search-R1/SEED 同构)。判分与执行完全复用
        _dispatch —— 只是把 <search>/<answer> 翻译成同构的 call dict。"""
        call = parse_text_action(assistant_text)
        thought = text_thought(assistant_text)
        if call is None:
            return ActionOutcome(
                thought=assistant_text.strip(),
                action_raw="",
                action_norm="",
                feedback=text_format_reminder(),
                done=False,
                score=0.0,
                acted=False,
                valid=False,
                consumed_env_step=False,
            )
        feedback, done, score, valid = self._dispatch(call, "", text_mode=True)
        return ActionOutcome(
            thought=thought or assistant_text.strip(),
            action_raw=json.dumps(call, ensure_ascii=False),
            action_norm=json.dumps(call, ensure_ascii=False, sort_keys=True),
            feedback=feedback,
            done=done,
            score=score,
            acted=True,
            valid=valid,
            consumed_env_step=True,
        )

    def _react_fncall(self, env: SearchQAEpisode, assistant_text: str) -> ActionOutcome:
        calls = protocol.parse_tool_calls(assistant_text, tools=self.tools())
        thought = protocol.strip_tool_calls(assistant_text)
        if not calls:
            return ActionOutcome(
                thought=assistant_text.strip(),
                action_raw="",
                action_norm="",
                feedback=_format_reminder(),
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
        feedback, done, score, valid = self._dispatch(call, extra_note)
        return ActionOutcome(
            thought=thought or assistant_text.strip(),
            action_raw=json.dumps(call, ensure_ascii=False),
            action_norm=json.dumps(call, ensure_ascii=False, sort_keys=True),
            feedback=feedback,
            done=done,
            score=score,
            acted=True,
            valid=valid,
            consumed_env_step=True,
        )

    def replay_action(self, env: SearchQAEpisode, action_norm: str) -> tuple[str, bool]:
        call = json.loads(action_norm)
        feedback, done, _score, _valid = self._dispatch(
            call, "", text_mode=(getattr(self._cfg, "protocol", "fncall") == "text")
        )
        return feedback, done

    def _dispatch(self, call: dict, extra_note: str,
                  text_mode: bool = False) -> tuple[str, bool, float, bool]:
        # 观察包裹随协议切换:FC 用 <tool_response>,文本协议用
        # <information>(Search-R1/SEED 同款),其余逻辑与判分完全共用。
        wrap = text_obs_wrap if text_mode else protocol.tool_response_wrap
        name = call.get("name")
        if name == "search":
            query = str(call.get("arguments", {}).get("query", "") or "")
            if not query.strip():
                return (
                    wrap("search: empty query." + extra_note),
                    False, 0.0, False,
                )
            body, ok = self._do_search(query.strip())
            return (
                wrap(clip_middle(body, self._cfg.obs_max_chars) + extra_note),
                False, 0.0, ok,
            )
        if name == "submit":
            answer = str(call.get("arguments", {}).get("answer", "") or "")
            if not answer.strip():
                return (
                    wrap("submit: empty answer — give the final answer "
                         "inside <answer> and </answer>." + extra_note),
                    False, 0.0, False,
                )
            # pandas/parquet 给出的 golden_answers 是 numpy array:`or []` 的
            # 真值判断在多元素 array 上抛 ValueError(单元素不炸——冒烟
            # 测不出),str 也不能直接 list 化,逐型处理。
            golden = self._item.get("golden_answers")
            if golden is None:
                golden = []
            elif isinstance(golden, str):
                golden = [golden]
            elif not isinstance(golden, (list, tuple)):
                golden = list(golden)
            score = 1.0 if em_check(answer, list(golden)) else 0.0
            # 信息状态红线:评分结果不反馈。
            return wrap("Submitted." + extra_note), True, score, True
        return (
            wrap(f"unknown action: {name!r}. Available: search, submit." + extra_note),
            False, 0.0, False,
        )

    def _do_search(self, query: str) -> tuple[str, bool]:
        """检索失败按环境故障 raise(fail loud:静默降级会把基建故障伪装成
        模型能力差 ⇒ 恒 0 分投毒 reward)。

        走进程级 _SearchBatcher 合批(2026-08-13:单 query 单请求吞吐封顶
        14.6 q/s,合批后 498 q/s);重试与 fail-loud 语义由 batcher 保持。"""
        return self._passages_to_string(self._batcher.search(query)), True

    @staticmethod
    def _passages_to_string(passages: list) -> str:
        """Search-R1 同款渲染:`Doc N(Title: ...) text`。corpus 行 contents
        首行是带引号的 title,余下为正文;兼容 {document:..} 包裹与
        title/text 直出两种形状。"""
        out = []
        for i, p in enumerate(passages):
            doc = p.get("document", p) if isinstance(p, dict) else p
            contents = str(doc.get("contents") or "")
            if contents:
                first, _, rest = contents.partition("\n")
                title = first.strip().strip('"')
                text = rest.strip()
            else:
                title = str(doc.get("title") or "")
                text = str(doc.get("text") or "")
            out.append(f"Doc {i + 1}(Title: {title}) {text}")
        return "\n".join(out) if out else "(no results)"

    # ---- 文本动作协议件(平凡实现) --------------------------------------

    def normalize_action(self, raw: str) -> str:
        return raw.strip()

    def observation_indicates_invalid(self, observation: str) -> bool:
        return False

    def format_reminder(self) -> str:
        if getattr(self._cfg, "protocol", "fncall") == "text":
            return text_format_reminder()
        return _format_reminder()


class _SearchBatcher:
    """跨 episode 的检索请求聚合器(micro-batching)。

    为什么需要:服务端 /retrieve 原生收 `queries` 数组,而每 episode 单
    query 单请求会被请求开销封死吞吐——2026-08-13 实测 162 retriever:
    并发 16 → 14.6 q/s、并发 256 → 15.7 q/s(**拉并发完全无用**);合批后
    batch64 × 在途16 → **498.1 q/s(34×)**。RL rollout 下 1024 episodes ×
    ~2.7 次检索 = 2765 次/步,单 query 模式检索独占 184s/步。

    语义保持与原 `_do_search` 一致:整批失败 ⇒ 批内每个调用者都收到异常
    (fail loud——静默降级会把基建故障伪装成模型能力差,投毒 reward);
    批级重试一次容瞬断。调用方仍是同步阻塞语义。
    """

    def __init__(self, http: httpx.Client, cfg: SearchQACfg) -> None:
        self._http = http
        self._url = f"{cfg.retriever_url.rstrip('/')}/retrieve"
        self._topk = cfg.topk
        self._timeout = cfg.search_timeout_s
        self._max_batch = max(1, int(cfg.search_max_batch))
        self._wait = max(0.0, float(cfg.search_batch_wait_ms) / 1000.0)
        self._q: queue.Queue = queue.Queue()
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, int(cfg.search_concurrency)),
            thread_name_prefix="sqa-batch",
        )
        # 可观测性(2026-08-13:冒烟出现 120s 超时,而离线压测显示服务端
        # 即便在 35B 满负荷争抢下仍有 836 q/s / 1.17s ⇒ 瓶颈不在服务端,
        # 必须让 batcher 自报队列深度与等待时长,而不是继续猜)
        self._stat_lock = threading.Lock()
        self._stat = {"submitted": 0, "flushed": 0, "batches": 0,
                      "wait_sum": 0.0, "wait_max": 0.0,
                      "flush_sum": 0.0, "flush_max": 0.0, "inflight": 0}
        self._loop_t = threading.Thread(
            target=self._collect_loop, name="sqa-batcher", daemon=True
        )
        self._loop_t.start()
        self._stat_t = threading.Thread(
            target=self._stat_loop, name="sqa-batcher-stat", daemon=True
        )
        self._stat_t.start()

    def _stat_loop(self) -> None:
        """每 30s 打一行摘要:队列深度是判断"客户端排队 vs 服务端慢"的关键。"""
        import logging
        log = logging.getLogger(__name__)
        while True:
            time.sleep(30)
            with self._stat_lock:
                st = dict(self._stat)
                self._stat.update({"submitted": 0, "flushed": 0, "batches": 0,
                                   "wait_sum": 0.0, "wait_max": 0.0,
                                   "flush_sum": 0.0, "flush_max": 0.0})
            if st["submitted"] == 0 and st["flushed"] == 0 and self._q.qsize() == 0:
                continue
            nb = max(st["batches"], 1)
            nf = max(st["flushed"], 1)
            log.warning(
                "[sqa-batcher/30s] queued=%d inflight=%d | in=%d out=%d batches=%d "
                "avg_batch=%.0f | wait avg=%.2fs max=%.2fs | flush avg=%.2fs max=%.2fs",
                self._q.qsize(), st["inflight"], st["submitted"], st["flushed"],
                st["batches"], st["flushed"] / nb,
                st["wait_sum"] / nf, st["wait_max"],
                st["flush_sum"] / nb, st["flush_max"],
            )

    def search(self, query: str) -> list:
        """阻塞直到本 query 的 passages 返回;失败 raise(与原语义同)。"""
        cell: dict = {"q": query, "ev": threading.Event(), "res": None,
                      "err": None, "t0": time.monotonic()}
        with self._stat_lock:
            self._stat["submitted"] += 1
        self._q.put(cell)
        # 等待上限给足:排队 + 批级重试(2×timeout)+ 余量
        if not cell["ev"].wait(timeout=self._timeout * 3 + 30.0):
            raise RuntimeError(
                f"searchqa batcher timed out waiting for retriever at {self._url}"
            )
        if cell["err"] is not None:
            raise cell["err"]
        return cell["res"]

    def _collect_loop(self) -> None:
        """攒批:拿到首个请求后,最多再等 wait 窗口或攒满 max_batch 即发。"""
        while True:
            batch = [self._q.get()]           # 阻塞等首个
            deadline = time.monotonic() + self._wait
            while len(batch) < self._max_batch:
                remain = deadline - time.monotonic()
                if remain <= 0:
                    break
                try:
                    batch.append(self._q.get(timeout=remain))
                except queue.Empty:
                    break
            # 发批交线程池(池大小=在途批上限),聚合循环立即回去攒下一批
            self._pool.submit(self._flush, batch)

    def _flush(self, batch: list) -> None:
        t_flush = time.monotonic()
        with self._stat_lock:
            self._stat["inflight"] += 1
            self._stat["batches"] += 1
        try:
            self._flush_inner(batch)
        finally:
            dt = time.monotonic() - t_flush
            with self._stat_lock:
                self._stat["inflight"] -= 1
                self._stat["flush_sum"] += dt
                self._stat["flush_max"] = max(self._stat["flush_max"], dt)
                for c in batch:
                    w = time.monotonic() - c.get("t0", t_flush)
                    self._stat["flushed"] += 1
                    self._stat["wait_sum"] += w
                    self._stat["wait_max"] = max(self._stat["wait_max"], w)

    def _flush_inner(self, batch: list) -> None:
        queries = [c["q"] for c in batch]
        last_err: Exception | None = None
        for _attempt in range(2):            # 批级重试一次(原语义)
            try:
                resp = self._http.post(
                    self._url,
                    # **return_scores=True 是必需的,不是可选项**(2026-08-13
                    # 冒烟实证):服务端在 return_scores=False 且 len(queries)>1
                    # 时返回**转置**的结果矩阵 —— queries=2/topk=3 时给出
                    # len=3 的 [2,2,2] 而非 len=2 的 [3,3];单 query 恰好不受
                    # 影响,所以原逐条实现从未暴露。带 scores 的路径形状正确
                    # (实测 queries=64 → len=64,各 [3,3,…])。score 本身我们
                    # 不消费(_passages_to_string 只读 document.contents);
                    # 已验证批量与单发的**文档内容逐字一致**,仅 score 有第 4
                    # 位小数级浮点抖动(batch 编码差异),不影响判分与观察。
                    json={"queries": queries, "topk": self._topk,
                          "return_scores": True},
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                results = resp.json()["result"]
                # 形状硬校验:外层长度必须等于 query 数,且每项是 list
                # (转置形态会在这里被抓住,而不是把错文档喂给模型)
                if len(results) != len(batch) or not all(
                    isinstance(r, list) for r in results
                ):
                    raise RuntimeError(
                        f"retriever returned malformed result: len={len(results)} "
                        f"for {len(batch)} queries "
                        f"(elem lens={[len(r) if isinstance(r, list) else 'NA' for r in results[:8]]})"
                    )
                for cell, res in zip(batch, results):
                    cell["res"] = res
                    cell["ev"].set()
                return
            except Exception as exc:  # noqa: BLE001 — 重试后统一分发异常
                last_err = exc
        err = RuntimeError(
            f"searchqa retriever unreachable at {self._url}: {last_err}"
        )
        for cell in batch:
            cell["err"] = err
            cell["ev"].set()


class SearchQAEnvFactory:
    """EnvFactory:进程级一个;共享 httpx 连接池 + 检索批量聚合器。装配期
    真实 ping 检索服务(fail loud:服务不可达时零分会伪装成能力问题)。"""

    def __init__(self, cfg: SearchQACfg) -> None:
        self._cfg = cfg
        self._http = httpx.Client(
            limits=httpx.Limits(
                max_connections=cfg.search_concurrency,
                max_keepalive_connections=cfg.search_concurrency,
            )
        )
        self._gate = threading.Semaphore(cfg.search_concurrency)
        if cfg.preflight:
            self._preflight()
        # preflight 之后再起聚合器:服务不可达时先 fail loud,不留后台线程
        self._batcher = _SearchBatcher(self._http, cfg)

    def _preflight(self) -> None:
        try:
            resp = self._http.post(
                f"{self._cfg.retriever_url.rstrip('/')}/retrieve",
                json={"queries": ["preflight"], "topk": 1},
                timeout=self._cfg.search_timeout_s,
            )
            resp.raise_for_status()
            _ = resp.json()["result"]
        except Exception as exc:
            raise RuntimeError(
                "searchqa preflight failed — retriever not reachable/valid "
                f"at {self._cfg.retriever_url}: {exc}"
            ) from exc

    def adapter_from_item(self, item: dict) -> SearchQATaskAdapter:
        return SearchQATaskAdapter(
            item, self._cfg, self._http, self._gate, self._batcher
        )
