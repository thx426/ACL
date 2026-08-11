"""SWE 域 TaskAdapter(R2E-Gym-Subset × ACS 云沙箱)。

设计源:experiments/bench_env/docs/2026-08-11-swe-env-proposal.md +
deep-reads/{swe-master-r2egym.md, swe-harness-comparison.md}。交互面对齐
R2E-Gym / SWE-Master 的 openhands scaffold(三工具 FC),但执行走 ACS 沙箱
(复用 cloudbox);编辑动作在 tools.py 内联翻译成容器命令(不拷 R2E 的 python
工具脚本,避免 /root/.venv 软链依赖)。

判分(judge.py 移植 R2E _calculate_reward_r2e,二值):episode 结束(submit
或预算耗尽)后无条件对当前仓库状态跑 r2e_tests,逐 key 比对 expected_output。
预算耗尽的 reward×0.5 与环境故障 mask=0 的分层由训练侧终止原因分类处理
(深读 §5.2);adapter 只报 score∈[0,1] 与 infra_failure。

信息状态红线(SSB/SRL 同款):
  * expected_output_json / GT patch 只经 privileged 文件进 judge,绝不进
    executor 可见流(submit 反馈恒 "Submitted.");
  * gt_text() = privileged 的 GT patch(仅 researcher，GT 开关控制);
  * git 全禁 + .git/ 拦截(防从历史偷答案,裸信息状态前提)。

确定性:沙箱镜像固定(docker_image = commit hash tag),动作流经 tools 翻译
无绝对随机;replay 契约由 react/replay 共用 _dispatch 保证。
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from ...common.config import SweCfg
from ...core import protocol
from ...core.adapter import ActionOutcome
from ..cloudbox import CloudSandbox, CloudboxError
from . import tools as T
from .judge import run_tests_command, score_from_logs

logger = logging.getLogger(__name__)

TESTBED = "/testbed"

#: openhands system prompt(SWE-Master config/openhands/edit_fn_calling.yaml:1
#: 逐字;工具块由 chat template 按 tools() 原生渲染,不手写)。
SYSTEM_PROMPT = (
    "You are a helpful assistant that can interact with a computer to solve "
    "tasks. You are provided with a local git repository and a GitHub issue. "
    "Your goal is to identify, implement, and verify a solution to the issue "
    "while documenting your reasoning process. You have access to tools for "
    "viewing and editing files, executing bash commands, and submitting your "
    "final solution."
)

_OBS_HEAD = 5000
_OBS_TAIL = 5000
_OBS_TOO_LONG_HINT = (
    "The output of your last command was too long. Try a command that "
    "produces less output — use head/tail/sed to view fewer lines, a more "
    "selective grep/find pattern, or redirect to a file and search in it."
)


def truncate_observation(output: str, max_chars: int) -> str:
    """head/tail 截断(mini-swe-agent swebench.yaml:136-158 同构)。"""
    if len(output) <= max_chars:
        return output
    head = min(_OBS_HEAD, max_chars // 2)
    tail = min(_OBS_TAIL, max_chars // 2)
    elided = len(output) - head - tail
    return (
        f"{_OBS_TOO_LONG_HINT}\n\n"
        f"---- HEAD ({head} chars) ----\n{output[:head]}\n"
        f"---- {elided} chars elided ----\n"
        f"---- TAIL ({tail} chars) ----\n{output[-tail:]}"
    )


def _format_reminder() -> str:
    return protocol.fc_format_reminder("str_replace_editor", "execute_bash")


class SweEpisode:
    """One live ACS sandbox episode. score∈{0.0,1.0}, 仅在评分路径产生。

    基建故障(Pod 死/exec 通道丢)抛 CloudboxError,由 core 归 ENV_ERROR
    (审阅 P0-5 纪律:不返回 done+0,否则基建压力被当模型失败入训)。"""

    def __init__(self, sandbox: CloudSandbox, item: dict, cfg: SweCfg) -> None:
        self._sbx = sandbox
        self._item = item
        self._cfg = cfg
        self._initial_obs = str(item["initial_obs"])
        self.infra_failure = ""
        self.closed = False
        if not cfg.tests_visible:
            self._hide_tests()

    def _hide_tests(self) -> None:
        """对齐 SWE-bench Verified 口径:移除 r2e_tests 对 agent 的可见软链
        (判分脚本引用 /root 路径,不受影响)。默认 tests_visible=True 不走此。"""
        try:
            self._sbx.run("rm -f /testbed/r2e_tests", self._cfg.command_timeout_s)
        except CloudboxError as err:  # noqa: BLE001 — 非致命,记账
            logger.warning("hide_tests failed for %s: %s",
                           self._item.get("instance_id"), err)

    @property
    def initial_obs(self) -> str:
        return self._initial_obs

    def exec_in_testbed(self, command: str) -> tuple[str, int]:
        """在 /testbed 下跑一条命令,返回 (合并输出, exit_code)。
        无状态子 shell(cd/env 不持久),故显式 cd /testbed。"""
        wrapped = f"cd {TESTBED} && ( {command} )"
        r = self._sbx.run(wrapped, self._cfg.command_timeout_s)
        body = r.stdout if not r.stderr else f"{r.stdout}\n{r.stderr}".strip()
        return body, r.returncode

    def collect_patch(self) -> str:
        """episode 末收 patch(harness 侧执行,不经 agent 工具/不受 git 封禁)。
        与 R2E docker.py:1329-1337 一致:git add -A && git diff --cached。"""
        r = self._sbx.run(
            f"cd {TESTBED} && git add -A && git diff --cached",
            self._cfg.command_timeout_s,
        )
        return r.stdout

    def run_and_score(self, expected_output_json: str, repo_name: str) -> float:
        """无条件对当前仓库状态判分(R2E _calculate_reward_r2e 语义)。"""
        r = self._sbx.run(
            f"cd {TESTBED} && {run_tests_command(repo_name)}",
            self._cfg.test_timeout_s,
            grace_s=self._cfg.test_grace_s,
        )
        log = r.stdout if not r.stderr else f"{r.stdout}\n{r.stderr}"
        return score_from_logs(log, expected_output_json)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._sbx.kill()


class SweTaskAdapter:
    """One adapter per dataset row. item = parquet extra_info(自足)+ 特权
    从 privileged 文件懒读(judge/researcher 用,不进 executor)。"""

    def __init__(self, item: dict, cfg: SweCfg, factory: "SweEnvFactory") -> None:
        self._item = dict(item)
        self._cfg = cfg
        self._factory = factory
        self._priv: dict | None = None

    # ---- 身份面 ----------------------------------------------------------

    @property
    def task_id(self) -> str:
        return str(self._item["instance_id"])

    @property
    def task_type(self) -> str:
        return str(self._item["repo_name"])  # family = repo(留一检验用)

    @property
    def initial_obs(self) -> str:
        return str(self._item["initial_obs"])

    # ---- 特权(懒读 privileged 文件)------------------------------------

    def _privileged(self) -> dict:
        if self._priv is None:
            path = Path(self._cfg.data_root) / "privileged" / f"{self.task_id}.json"
            if not path.is_file():
                raise RuntimeError(f"privileged file missing: {path}")
            self._priv = json.loads(path.read_text(encoding="utf-8"))
        return self._priv

    def gt_text(self) -> str | None:
        """GT patch(特权 L4);仅 researcher,GT 开关控制,绝不进 executor。"""
        return self._privileged().get("parsed_commit_content")

    # ---- researcher 面 ---------------------------------------------------

    def task_brief(self) -> str:
        it = self._item
        return (
            "Environment: SWE (R2E-Gym) — fix a bug in a real Python "
            f"repository ({it['repo_name']}) described by a GitHub issue. "
            "The executor works inside /testbed (the repo checked out at a "
            "buggy commit, dependencies pre-installed).\n"
            "Executor tools (one call per turn):\n"
            "  str_replace_editor(command, path, ...) — view/create/str_replace/"
            "insert on files\n"
            "  execute_bash(command) — run a shell command in /testbed "
            "(git disabled; fresh shell per call)\n"
            "  submit() — end the episode; the repo state is then graded\n"
            "Grading: the hidden test suite is run; the task is solved (score "
            "1.0) only if every expected test status matches — binary, no "
            "partial credit. The executor never sees grading feedback.\n"
            f"Issue:\n{self.initial_obs}"
        )

    def executor_scaffold(self) -> str:
        return SYSTEM_PROMPT

    def tools(self) -> list[dict] | None:
        return T.TOOLS

    def domain_notes(self) -> str:
        return ""

    # ---- 环境面 ----------------------------------------------------------

    def open_episode(self) -> SweEpisode:
        image = self._factory.image_for(self._item)
        sbx = self._factory.spawn(image, self.task_id)
        return SweEpisode(sbx, self._item, self._cfg)

    # ---- 交互面(function-calling react)--------------------------------

    def react(self, env: SweEpisode, assistant_text: str) -> ActionOutcome:
        calls = protocol.parse_tool_calls(assistant_text, tools=self.tools())
        thought = protocol.strip_tool_calls(assistant_text)
        if not calls:
            return ActionOutcome(
                thought=assistant_text.strip(),
                action_raw="", action_norm="",
                feedback=_format_reminder(),
                done=False, score=0.0, acted=False, valid=False,
                consumed_env_step=False,
            )
        call = calls[0]
        extra = (
            f"\n(note: {len(calls) - 1} extra tool call(s) ignored — one per turn)"
            if len(calls) > 1 else ""
        )
        feedback, done, score, valid = self._dispatch(env, call, extra)
        return ActionOutcome(
            thought=thought or assistant_text.strip(),
            action_raw=json.dumps(call, ensure_ascii=False),
            action_norm=_norm_call(call),
            feedback=feedback, done=done, score=score,
            acted=True, valid=valid, consumed_env_step=True,
        )

    def replay_action(self, env: SweEpisode, action_norm: str) -> tuple[str, bool]:
        call = json.loads(action_norm)
        feedback, done, _score, _valid = self._dispatch(env, call, "")
        return feedback, done

    def _dispatch(self, env: SweEpisode, call: dict, extra: str):
        """(feedback, done, score, valid) — react/replay 共用执行核。"""
        name = call.get("name")
        args = call.get("arguments", {}) or {}
        if name == "submit":
            score = self._score(env)
            return protocol.tool_response_wrap("Submitted." + extra), True, score, True
        if name == "str_replace_editor":
            err = self._validate_editor(args)
            if err:
                return protocol.tool_response_wrap(err + extra), False, 0.0, False
            pblock = T.path_is_blocked(str(args.get("path", "")))
            if pblock:
                return protocol.tool_response_wrap(pblock + extra), False, 0.0, False
            command = T.editor_to_command(args)
            return self._run(env, command, extra)
        if name == "execute_bash":
            command = str(args.get("command", "") or "")
            if not command.strip():
                return protocol.tool_response_wrap(
                    "execute_bash: empty command." + extra), False, 0.0, False
            if not self._cfg.allow_git:
                block = T.bash_is_blocked(command)
                if block:
                    return protocol.tool_response_wrap(block + extra), False, 0.0, False
            return self._run(env, command, extra)
        return protocol.tool_response_wrap(
            f"unknown tool: {name!r}. Available: str_replace_editor, "
            f"execute_bash, submit." + extra), False, 0.0, False

    def _run(self, env: SweEpisode, command: str, extra: str):
        out, rc = env.exec_in_testbed(command)  # 可抛 CloudboxError → core ENV_ERROR
        body = out if rc == 0 else f"[exit {rc}]\n{out}"
        clipped = truncate_observation(body, self._cfg.max_obs_chars)
        return protocol.tool_response_wrap(clipped + extra), False, 0.0, rc == 0

    def _score(self, env: SweEpisode) -> float:
        priv = self._privileged()
        return env.run_and_score(
            priv["expected_output_json"], str(self._item["repo_name"])
        )

    @staticmethod
    def _validate_editor(args: dict) -> str | None:
        """必参校验(可读错误,不炸)。"""
        cmd = args.get("command")
        if cmd not in T.ALLOWED_EDITOR_COMMANDS:
            return (f"str_replace_editor: invalid command {cmd!r}. "
                    f"Allowed: {', '.join(T.ALLOWED_EDITOR_COMMANDS)}.")
        if not str(args.get("path", "")).strip():
            return "str_replace_editor: `path` is required."
        if cmd == "create" and "file_text" not in args:
            return "str_replace_editor: `file_text` is required for `create`."
        if cmd == "str_replace" and "old_str" not in args:
            return "str_replace_editor: `old_str` is required for `str_replace`."
        if cmd == "insert" and ("insert_line" not in args or "new_str" not in args):
            return "str_replace_editor: `insert_line` and `new_str` required for `insert`."
        return None

    # ---- 文本动作协议件(SWE 不经 text_action_react;平凡实现)-----------

    def normalize_action(self, raw: str) -> str:
        return raw.strip()

    def observation_indicates_invalid(self, observation: str) -> bool:
        return False

    def format_reminder(self) -> str:
        return _format_reminder()


def _norm_call(call: dict) -> str:
    return json.dumps(call, ensure_ascii=False, sort_keys=True)


class SweEnvFactory:
    """EnvFactory:进程级一实例;持沙箱创建并发门。"""

    def __init__(self, cfg: SweCfg) -> None:
        if not cfg.data_root:
            raise ValueError("swe config incomplete: data_root required (SWE_DATA)")
        cfg._validate()
        self._cfg = cfg
        self._gate = threading.Semaphore(cfg.spawn_concurrency)

    def image_for(self, item: dict) -> str:
        """镜像全名:有 image_registry ⇒ 转推 ACR 的 <registry>/<repo>_final:
        <hash>;否则用数据集原 docker_image(dockerhub,仅本地/162 直连)。"""
        if not self._cfg.image_registry:
            return str(item["docker_image"])
        repo, tag = str(item["docker_image"]).split("/")[-1].split(":")
        return f"{self._cfg.image_registry.rstrip('/')}/{repo}:{tag}"

    def spawn(self, image: str, task_id: str) -> CloudSandbox:
        with self._gate:
            return CloudSandbox(
                kubeconfig=self._cfg.kubeconfig,
                namespace=self._cfg.namespace,
                image=image,
                image_pull_secret=self._cfg.image_pull_secret,
                cpu=self._cfg.pod_cpu,
                memory=self._cfg.pod_memory,
                task_id=task_id,
                create_timeout_s=self._cfg.create_timeout_s,
                run_label=self._cfg.run_label,
            )

    def adapter_from_item(self, item: dict) -> SweTaskAdapter:
        return SweTaskAdapter(item, self._cfg, self)
