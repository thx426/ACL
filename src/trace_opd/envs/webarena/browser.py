"""Playwright browser session for the WebArena domain (sync API, one session
per episode, thread-confined).

Interaction semantics are ported 1:1 from WebServ's ``rl_web_agent/env.py``
(deep-read: experiments/bench_env/codebases/webserv, Apache-2.0): semantic-ID
selectors (``[data-semantic-id]``) with fail-fast scroll + force actions, the
``initscript.js`` XHR/fetch activity counter driving custom network-idle
detection, and the ``parser.js`` DOM stripper producing the observation dict
``{html, clickable_elements, hoverable_elements, input_elements,
select_elements}``. Both JS files are byte-identical copies under ``assets/``
(provenance: WebServ ``rl_web_agent/javascript/``).

WebArena-Verified additions (proposal §8.5 rulings):
* HAR recording on the TASK context with ``record_har_content="embed"`` —
  31/812 tasks' NetworkEventEvaluator asserts on response bodies, so bodies
  must be embedded, not omitted.
* Login happens in a SEPARATE context (no HAR); only the resulting
  ``storage_state`` is carried into the task context. Keeps login requests out
  of the evaluated trace (mirrors the package's ``.storage_state.json``
  pre-login design).
* ``shopping_admin`` uses header auto-login (``X-M2-Admin-Auto-Login:
  <username>``, native to the am1n3e verified images) instead of the admin UI
  form; the header rides on the task context.

Threading:sync Playwright 对象绑定其创建线程,而 ``asyncio.to_thread`` 不
保证同一 episode 的两次 react 落在同一线程——所以 BrowserSession 内置一条
专属 owner 线程,所有 Playwright 调用经命令队列编组到该线程执行(公共方法
线程安全,可从任意调用线程进入;单 episode 内在 owner 线程上天然串行)。
"""
from __future__ import annotations

import concurrent.futures
import logging
import queue
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_ASSETS = Path(__file__).resolve().parent / "assets"
_PARSER_JS = (_ASSETS / "parser.js").read_text(encoding="utf-8")
_INITSCRIPT_JS = (_ASSETS / "initscript.js").read_text(encoding="utf-8")

#: WebServ conf/base.yaml browser.launch_options.args(逐项照搬)。
_LAUNCH_ARGS = [
    "--disable-autofill-keyboard-accessory-view",
    "--disable-autofill-profile-update",
    "--disable-autofill",
    "--disable-extensions",
    "--disable-background-networking",
    "--safebrowsing-disable-auto-update",
    "--disable-component-update",
    "--disable-default-apps",
    "--no-first-run",
    "--no-default-browser-check",
    "--metrics-recording-only",
    "--disable-sync",
    "--disable-translate",
    "--disable-client-side-phishing-detection",
    "--disable-domain-reliability",
    "--disable-crash-reporter",
]
_VIEWPORT = {"width": 1440, "height": 900}   # WebServ context_options.viewport

#: 原版 WebArena 公共账号(WebServ conf/base.yaml accounts;am1n3e verified
#: 镜像内置同一批账号)。站点无登录:map / wikipedia。
ACCOUNTS: dict[str, dict[str, str]] = {
    "shopping": {"username": "emma.lopez@gmail.com", "password": "Password.123"},
    "shopping_admin": {"username": "admin", "password": "admin12341"},
    "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
}
#: webarena_verified.environments.MAGENTO_ADMIN_AUTO_LOGIN_HEADER
MAGENTO_ADMIN_AUTO_LOGIN_HEADER = "X-M2-Admin-Auto-Login"

# 超时组(ms)。page_load_* 对齐 WebServ 的两段等待结构;数值为保守缺省,
# 生产值经 pilot 标定后走 WebArenaCfg 覆盖。
_T_DOMCONTENT_MS = 20_000
_T_NETWORKIDLE_MS = 10_000
_T_CUSTOM_IDLE_MS = 500
_T_ELEMENT_MS = 500        # scroll_into_view fail-fast(WebServ 同值)
_T_LOGIN_MS = 30_000


class BrowserError(RuntimeError):
    """Infrastructure-level browser failure (launch/context/HAR)."""


def format_llm_observation(obs: dict) -> str:
    """WebServ ``format_llm_observation`` 逐结构照搬(观测文本口径一致,
    便于与其 44.8% 参考读数可比)。"""
    if not obs:
        return "No observation data available"
    parts: list[str] = []
    tabs = obs.get("tabs") or []
    active = next((t for t in tabs if t["is_active"]), tabs[0] if tabs else None)
    if active:
        parts.append(f"URL: {active['url']}")
        parts.append(f"Title: {active['title']}")
        parts.append("")
    parts.append("HTML:")
    parts.append(obs.get("html", ""))
    parts.append("")
    if obs.get("clickable_elements"):
        elems = obs["clickable_elements"]
        parts.append(f"CLICKABLE ELEMENTS ({len(elems)})")
        parts.append("-" * 40)
        for i, elem_id in enumerate(elems, 1):
            parts.append(f"  {i:2d}. {elem_id}")
        parts.append("")
    if obs.get("input_elements"):
        inputs = obs["input_elements"]
        parts.append(f"INPUT ELEMENTS ({len(inputs)})")
        parts.append("-" * 40)
        for i, inp in enumerate(inputs, 1):
            status_parts = []
            if inp.get("isFocused"):
                status_parts.append("focused")
            if not inp.get("canEdit"):
                status_parts.append("read-only")
            status = " " + " ".join(status_parts) if status_parts else ""
            parts.append(f"  {i:2d}. {inp['id']} [{inp['type']}]{status}")
            if inp.get("value"):
                parts.append(f"      Value: '{inp['value']}'")
        parts.append("")
    if obs.get("select_elements"):
        sels = obs["select_elements"]
        parts.append(f"SELECT ELEMENTS ({len(sels)})")
        parts.append("-" * 40)
        for i, sel in enumerate(sels, 1):
            parts.append(f"  {i:2d}. {sel['id']} - {sel['value']}")
        parts.append("")
    parts.append(f"TABS ({len(tabs)})")
    parts.append("-" * 40)
    for tab in tabs:
        state = "ACTIVE" if tab["is_active"] else "inactive"
        parts.append(f"  {tab['id']:2d}. {state} - {tab['title']} - {tab['url']}")
    parts.append("")
    return "\n".join(parts)


class _OwnerThread:
    """Single dedicated thread executing submitted callables in order.
    Playwright objects live and die on this thread only."""

    def __init__(self, name: str) -> None:
        self._q: queue.Queue = queue.Queue()
        self._t = threading.Thread(target=self._run, name=name, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            fn, fut = item
            if fut.set_running_or_notify_cancel():
                try:
                    fut.set_result(fn())
                except BaseException as err:  # noqa: BLE001
                    fut.set_exception(err)

    def call(self, fn, timeout_s: float):
        if not self._t.is_alive():
            raise BrowserError("browser owner thread is dead")
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._q.put((fn, fut))
        try:
            return fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError as err:
            # owner 线程可能仍卡在 playwright 内部;标记会话报废,由 close 兜底
            raise BrowserError(
                f"browser call timed out after {timeout_s:.0f}s "
                "(owner thread wedged)"
            ) from err

    def stop(self) -> None:
        self._q.put(None)


class BrowserSession:
    """One live browser episode: login → HAR-recorded task context → actions.

    Lifecycle: ``start(sites, start_urls)`` → action methods / ``observation()``
    → ``finalize()`` (flushes HAR) → ``close()``. close() 幂等且吞异常;
    finalize() 失败 fail-loud(HAR 缺失 = 评测不可进行)。公共方法可从任意
    线程调用(内部编组到 owner 线程)。
    """

    #: 单次编组调用的墙钟上限(页面等待链最长 ~40s,留裕量;卡死→报废会话)
    CALL_TIMEOUT_S = 120.0

    def __init__(
        self,
        *,
        base_urls: dict[str, str],
        har_path: Path,
        headless: bool = True,
        slow_site_grace_s: float = 0.0,
    ) -> None:
        self._base_urls = dict(base_urls)
        self._har_path = Path(har_path)
        self._headless = headless
        self._grace_s = slow_site_grace_s
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._closed = False
        self._owner = _OwnerThread(f"browser-{id(self):x}")

    # ---------------------------------------------------- public (any thread)

    def start(self, sites: list[str], start_urls: list[str]) -> None:
        self._owner.call(lambda: self._do_start(sites, start_urls),
                         self.CALL_TIMEOUT_S * 2)   # 登录+多 tab 打开较长

    def observation(self) -> dict:
        return self._owner.call(self._do_observation, self.CALL_TIMEOUT_S)

    def act(self, fn_name: str, *args) -> None:
        """Dispatch an action method (``click``/``type_text``/…) by name on
        the owner thread."""
        fn = getattr(self, f"_do_{fn_name}")
        self._owner.call(lambda: fn(*args), self.CALL_TIMEOUT_S)

    def finalize(self) -> None:
        self._owner.call(self._do_finalize, self.CALL_TIMEOUT_S)

    def close(self) -> None:
        """Idempotent full cleanup; never raises. Blocks briefly to let the
        owner thread release the browser process."""
        if self._closed:
            return
        self._closed = True
        try:
            self._owner.call(self._do_close, 30.0)
        except Exception:  # noqa: BLE001 — owner 卡死时只能弃线程(daemon)
            pass
        self._owner.stop()

    # ------------------------------------------------- impl (owner thread)

    def _do_start(self, sites: list[str], start_urls: list[str]) -> None:
        """Launch browser, log into ``sites``, open ``start_urls`` (already
        rendered to real URLs; first in the main tab, rest in new tabs)."""
        from playwright.sync_api import sync_playwright

        try:
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(
                headless=self._headless, args=_LAUNCH_ARGS,
            )
            storage_state = self._login_all(sites)
            self._har_path.parent.mkdir(parents=True, exist_ok=True)
            self._context = self._browser.new_context(
                viewport=_VIEWPORT,
                storage_state=storage_state,
                record_har_path=str(self._har_path),
                record_har_content="embed",   # 31/812 任务断言响应体,必须内嵌
                record_har_mode="full",
            )
            if "shopping_admin" in sites:
                self._context.set_extra_http_headers({
                    MAGENTO_ADMIN_AUTO_LOGIN_HEADER:
                        ACCOUNTS["shopping_admin"]["username"],
                })
            self._context.add_init_script(_INITSCRIPT_JS)
            if not start_urls:
                raise BrowserError("task has no start_urls")
            self._page = self._context.new_page()
            self._goto(self._page, start_urls[0])
            for url in start_urls[1:]:
                page = self._context.new_page()
                self._goto(page, url)
            self._page.bring_to_front()
        except BrowserError:
            self._do_close()      # owner 线程内直调,绝不能经编组(自死锁)
            raise
        except Exception as err:  # noqa: BLE001
            self._do_close()
            raise BrowserError(f"browser start failed: {err}") from err

    def _goto(self, page, url: str) -> None:
        page.goto(url, wait_until="domcontentloaded",
                  timeout=_T_DOMCONTENT_MS + self._grace_s * 1000)

    def _login_all(self, sites: list[str]) -> dict:
        """UI login on a throwaway context (never HAR-recorded); returns the
        storage_state dict carried into the task context. shopping_admin 走
        header 自动登录,不在这里做。"""
        ctx = self._browser.new_context(viewport=_VIEWPORT)
        try:
            for site in sites:
                if site not in ACCOUNTS or site == "shopping_admin":
                    continue
                creds = ACCOUNTS[site]
                base = self._base_urls[site]
                last_err: Exception | None = None
                for attempt in range(3):        # WebServ login max_retries=3
                    page = ctx.new_page()
                    try:
                        self._login_attempt(page, site, base,
                                            creds["username"], creds["password"])
                        last_err = None
                        break
                    except Exception as err:  # noqa: BLE001
                        last_err = err
                        logger.warning("login %s attempt %d failed: %s",
                                       site, attempt + 1, err)
                        time.sleep(1.0 * (attempt + 1))
                    finally:
                        page.close()
                if last_err is not None:
                    raise BrowserError(
                        f"login to {site} failed after 3 attempts: {last_err}"
                    ) from last_err
            return ctx.storage_state()
        finally:
            ctx.close()

    def _login_attempt(self, page, site: str, base: str,
                       username: str, password: str) -> None:
        """WebServ ``_login_attempt`` 四分支的 sync 版(admin 分支不需要——
        header 登录),表单定位逐项照搬。"""
        if site == "shopping":
            page.goto(f"{base}/customer/account/login/",
                      wait_until="networkidle", timeout=_T_LOGIN_MS)
            page.get_by_label("Email", exact=True).fill(username)
            page.get_by_label("Password", exact=True).fill(password)
            page.get_by_role("button", name="Sign In").click()
            page.wait_for_load_state("networkidle", timeout=_T_LOGIN_MS)
        elif site == "reddit":
            page.goto(f"{base}/login", wait_until="networkidle",
                      timeout=_T_LOGIN_MS)
            page.get_by_label("Username").fill(username)
            page.get_by_label("Password").fill(password)
            page.get_by_role("button", name="Log in").click()
            page.wait_for_load_state("networkidle", timeout=_T_LOGIN_MS)
        elif site == "gitlab":
            page.goto(f"{base}/users/sign_in", wait_until="networkidle",
                      timeout=_T_LOGIN_MS)
            page.locator("#user_login").fill(username)
            page.locator("#user_password").fill(password)
            page.locator('button[type="submit"], input[type="submit"]').first.click()
            page.wait_for_load_state("networkidle", timeout=_T_LOGIN_MS)
        else:
            raise BrowserError(f"no login recipe for site {site!r}")
        # 登录后仍停在登录页 = 凭据被拒(表单站点共同特征:成功必跳转)。
        if ("/login" in page.url or "/sign_in" in page.url
                or "customer/account/login" in page.url):
            raise BrowserError(f"{site} login did not leave the login page "
                               f"(landed on {page.url})")

    def _do_finalize(self) -> None:
        """Close the task context, flushing the HAR to disk. Fail-loud:评测
        依赖 HAR,冲不出来必须暴露。"""
        if self._context is not None:
            ctx, self._context = self._context, None
            ctx.close()
            if not self._har_path.is_file():
                raise BrowserError(f"HAR not written: {self._har_path}")

    def _do_close(self) -> None:
        """Idempotent full cleanup on the owner thread; never raises."""
        for closer in (
            lambda: self._context.close() if self._context else None,
            lambda: self._browser.close() if self._browser else None,
            lambda: self._pw.stop() if self._pw else None,
        ):
            try:
                closer()
            except Exception:  # noqa: BLE001
                pass
        self._context = self._browser = self._pw = self._page = None

    # ------------------------------------------------------------ observation

    def _do_observation(self) -> dict:
        """Parsed page content via parser.js + tabs(WebServ observation()
        的环境侧部分;评测不在这里跑——Verified 是终局评测)。"""
        page = self._page
        try:
            page.wait_for_load_state("domcontentloaded",
                                     timeout=_T_DOMCONTENT_MS)
            try:
                page.wait_for_load_state("networkidle",
                                         timeout=_T_NETWORKIDLE_MS)
            except Exception:  # noqa: BLE001 — networkidle 超时是常态,非错误
                pass
            self._wait_custom_network_idle(page)
            page.wait_for_selector("body", timeout=5_000)
        except Exception as err:  # noqa: BLE001
            logger.warning("page-load wait timed out: %s", err)
        try:
            content = page.evaluate(_PARSER_JS)
        except Exception as err:  # noqa: BLE001
            logger.error("parser script failed: %s", err)
            content = {"html": page.content()}
        content["tabs"] = self._tabs_info()
        return content

    def _tabs_info(self) -> list[dict]:
        pages = self._context.pages
        infos = []
        for i, p in enumerate(pages):
            try:
                title = p.title()
            except Exception:  # noqa: BLE001
                title = ""
            infos.append({"id": i, "url": p.url, "title": title,
                          "is_active": p is self._page})
        return infos

    def _wait_custom_network_idle(self, page,
                                  timeout_ms: int = _T_NETWORKIDLE_MS,
                                  idle_ms: int = _T_CUSTOM_IDLE_MS) -> None:
        """Python-side poll of the initscript ``window.__networkActivity``
        counter (sync equivalent of WebServ's JS-promise wait)."""
        deadline = time.monotonic() + timeout_ms / 1000
        while time.monotonic() < deadline:
            try:
                state = page.evaluate(
                    "window.__networkActivity ? "
                    "{active: window.__networkActivity.activeRequests, "
                    "idle: Date.now() - window.__networkActivity.lastActivity}"
                    " : null"
                )
            except Exception:  # noqa: BLE001 — navigation in flight
                time.sleep(0.1)
                continue
            if state is None or (state["active"] <= 0 and state["idle"] >= idle_ms):
                return
            time.sleep(0.1)

    # ---------------------------------------------------------------- actions
    # 语义照搬 WebServ:semantic-id locator + 500ms fail-fast scroll + force。
    # 动作异常向上抛(adapter 捕获后作为环境反馈给模型,不终局)。

    def _locate(self, semantic_id: str):
        el = self._page.locator(f'[data-semantic-id="{semantic_id}"]')
        el.scroll_into_view_if_needed(timeout=_T_ELEMENT_MS)
        return el

    def _do_click(self, semantic_id: str) -> None:
        self._locate(semantic_id).click(force=True)

    def _do_type_text(self, semantic_id: str, text: str,
                      press_enter: bool = False) -> None:
        el = self._locate(semantic_id)
        el.fill(text, force=True)
        if press_enter:
            el.press("Enter")

    def _do_hover(self, semantic_id: str) -> None:
        self._locate(semantic_id).hover(force=True)

    def _do_select(self, semantic_id: str, value: str) -> None:
        self._locate(semantic_id).select_option(value, force=True)

    def _do_clear(self, semantic_id: str) -> None:
        self._locate(semantic_id).clear(force=True)

    def _do_key_press(self, key: str, semantic_id: str | None = None) -> None:
        if semantic_id:
            self._locate(semantic_id).press(key)
        else:
            self._page.keyboard.press(key)

    def _do_goto_url(self, url: str) -> None:
        self._page.goto(url, wait_until="domcontentloaded",
                        timeout=_T_DOMCONTENT_MS)

    def _do_back(self) -> None:
        self._page.go_back(wait_until="domcontentloaded",
                           timeout=_T_DOMCONTENT_MS)

    def _do_forward(self) -> None:
        self._page.go_forward(wait_until="domcontentloaded",
                              timeout=_T_DOMCONTENT_MS)

    def _do_refresh(self) -> None:
        self._page.reload(wait_until="domcontentloaded",
                          timeout=_T_DOMCONTENT_MS)

    def _do_new_tab(self, url: str | None = None) -> int:
        page = self._context.new_page()
        if url:
            self._goto(page, url)
        self._page = page
        return len(self._context.pages) - 1

    def _do_switch_tab(self, tab_id: int) -> None:
        pages = self._context.pages
        if not 0 <= tab_id < len(pages):
            raise ValueError(f"tab_id {tab_id} out of range "
                             f"(0..{len(pages) - 1})")
        self._page = pages[tab_id]
        self._page.bring_to_front()

    def _do_close_tab(self, tab_id: int) -> None:
        pages = self._context.pages
        if not 0 <= tab_id < len(pages):
            raise ValueError(f"tab_id {tab_id} out of range "
                             f"(0..{len(pages) - 1})")
        if len(pages) == 1:
            raise ValueError("cannot close the last remaining tab")
        victim = pages[tab_id]
        was_active = victim is self._page
        victim.close()
        if was_active:
            self._page = self._context.pages[0]
            self._page.bring_to_front()


_PLACEHOLDER_RE = None


def render_start_urls(start_urls: list[str], base_urls: dict[str, str]) -> list[str]:
    """Replace ``__SHOPPING__``-style placeholders with the episode's real
    base URLs(占位符集与 verified 数据集一致;未知占位符 fail-loud——注意
    检测用 ``__NAME__`` 模式而非裸 ``__``,wiki 路径天然可含下划线)。"""
    import re

    global _PLACEHOLDER_RE
    if _PLACEHOLDER_RE is None:
        _PLACEHOLDER_RE = re.compile(r"__[A-Z][A-Z_]*__")
    rendered = []
    for url in start_urls:
        out = url
        for site, base in base_urls.items():
            out = out.replace(f"__{site.upper()}__", base.rstrip("/"))
        leftover = _PLACEHOLDER_RE.search(out)
        if leftover:
            raise BrowserError(
                f"unresolved placeholder {leftover.group(0)} in start_url "
                f"{url!r} (have sites: {sorted(base_urls)})"
            )
        rendered.append(out)
    return rendered
