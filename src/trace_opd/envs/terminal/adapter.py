"""Terminal-CLI TaskAdapter (TMax-15K corpus, ACS cloud sandbox execution).

Design source: experiments/bench_env/docs/deep-reads/terminal-tmax-seta.md
(§3 protocol, §6.5 episode flow) + 2026-08-10-terminal-env-proposal.md (user
rulings). The interaction surface is byte-faithful to TMax's RL harness
(``swerl_vanillux_sandbox.py``, Apache-2.0): the single ``bash`` tool schema,
the persistent-shell wrapper (same /tmp paths), observation truncation and
hints, the submit marker, and the binary submit-then-verify reward. Documented
deviations (proposal §6.5 table): runtime ``setup.sh`` on a shared fat base
image instead of per-task prebaked images; format errors get a feedback
message instead of silently ending the episode; scoring outer wall > inner
600s (TMax 120<600 inversion fix); ACS Pod backend instead of Podman/Daytona.

Data contract: one adapter per dataset row ``{task_id, user_prompt}`` —
``user_prompt`` is the parquet ``messages[1]`` verbatim (already rendered
through TMax's vendored mini-swe-agent instance template; we never re-render).
Task assets (``setup.sh``, ``tests/test.sh``, ``instruction.md``) are read
from ``task_data_root/<task_id>/`` at episode time. ``tests/`` is uploaded to
the sandbox ONLY at scoring time (anti-peeking, TMax design).

Reward semantics (TMax train semantics, deep-read §7.3): the verifier runs
ONLY when the model submits via ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``;
unsubmitted / budget-exhausted episodes score 0 without a verifier run
("overlong negatives" are kept as negatives). ``terminal.score_at_budget``
flips to Terminal-Bench-style verify-at-end (eval-side approximation only).
"""
from __future__ import annotations

import logging
import shlex
import threading
from pathlib import Path

from ...common.config import TerminalCfg
from ...core import protocol
from ..cloudbox import CloudSandbox, CloudboxError

logger = logging.getLogger(__name__)

# --- TMax harness literals(swerl_vanillux_sandbox.py / vanillux_prompts.yaml
#     逐字节照搬;路径都保留原名——沙箱内表面与 TMax 完全同构)-------------
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

#: 快照在容器内的挂载点(只读;provision 期解开后不再需要)
_SNAP_IN_CONTAINER = "/tmp/.trace_snapshot.tgz"

_BASH_WRAPPER_PATH = "/tmp/.swerl_vanillux_bash_wrapper.sh"
_BASH_WRAPPER_PATH_QUOTED = shlex.quote(_BASH_WRAPPER_PATH)
_BASH_CWD_PATH = "/tmp/.swerl_vanillux_cwd"
_BASH_ENV_PATH = "/tmp/.swerl_vanillux_env"
_BASH_WRAPPER = f"""#!/bin/bash
set -a
source {shlex.quote(_BASH_ENV_PATH)} 2>/dev/null || true
set +a
_cwd="$(cat {shlex.quote(_BASH_CWD_PATH)} 2>/dev/null || echo /app)"
cd "$_cwd" 2>/dev/null || cd /workspace || exit 1
eval "$1"
_exit_code=$?
export -p > {shlex.quote(_BASH_ENV_PATH)}
pwd > {shlex.quote(_BASH_CWD_PATH)}
exit $_exit_code
"""

TOOLS_FC: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a bash command in a persistent shell. "
                "Working directory and environment variables are preserved "
                "between calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    },
                },
                "required": ["command"],
            },
        },
    },
]

#: 训练语料的 system prompt(tmax15k.parquet messages[0],全 14,601 行唯一;
#: build_terminal_dataset.py 逐行断言与此常量相等——数据漂移即 fail)。
SYSTEM_PROMPT = (
    "You are a helpful coding assistant. You have access to a bash terminal. "
    "Use it to explore the codebase, understand the problem, implement a "
    "solution, and verify it works. When you are confident your solution is "
    "correct, submit by running: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
)

_OBS_MAX_CHARS = 10_000
_OBS_HEAD_CHARS = 5_000
_OBS_TAIL_CHARS = 5_000
_OBS_TOO_LONG_HINT = """The output of your last command was too long.
Please try a different command that produces less output.
If you're looking at a file you can try use head, tail or sed to view a
smaller number of lines selectively. If you're using grep or find and it
produced too much output, you can use a more selective search pattern.
If you really need to see something from the full command's output, you
can redirect output to a file and then search in that file.
"""

_FORMAT_ERROR_TEMPLATE = """Format error: {{error}}

Please always provide EXACTLY ONE call to the `bash` tool. If you want to
end the task, please issue the command `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
via the `bash` tool, with no other content in the command.
"""

TOOL_CALL_FORMAT_ERROR_MESSAGE = (
    "Format error: Your last response did not include a valid `bash` tool call.\n\n"
    "Please always provide EXACTLY ONE call to the `bash` tool. If you want to\n"
    "end the task, please issue the command `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\n"
    "via the `bash` tool, with no other content in the command."
)


def truncate_observation(output: str) -> str:
    """mini-swe-agent head/tail truncation(TMax 同名函数逐字节)。"""
    if len(output) <= _OBS_MAX_CHARS:
        return output
    elided = len(output) - _OBS_HEAD_CHARS - _OBS_TAIL_CHARS
    return (
        f"{_OBS_TOO_LONG_HINT}\n\n"
        f"---- HEAD ({_OBS_HEAD_CHARS} chars) ----\n"
        f"{output[:_OBS_HEAD_CHARS]}\n"
        f"---- {elided} chars elided ----\n"
        f"---- TAIL ({_OBS_TAIL_CHARS} chars) ----\n"
        f"{output[-_OBS_TAIL_CHARS:]}"
    )


def _parse_env(spec: str) -> dict | None:
    """``K=V,K2=V2`` → dict;空串/无等号项忽略。值里允许含 ``=``(只切首个)。"""
    env = {}
    for kv in (spec or "").split(","):
        kv = kv.strip()
        if "=" in kv:
            k, v = kv.split("=", 1)
            if k.strip():
                env[k.strip()] = v.strip()
    return env or None


def format_error_message(error: str) -> str:
    return _FORMAT_ERROR_TEMPLATE.replace("{{error}}", error)


class TerminalEnvHandle:
    """One live ACS sandbox episode.

    step() → (obs, done, score);score∈{0.0,1.0} 且仅在提交(或
    score_at_budget 的末步)时经 verifier 产生。沙箱基建故障(Pod 死/exec
    通道丢)**抛 CloudboxError**,由 core 统一归 OUTCOME_ENV_ERROR(训练侧
    重试/丢弃)——不返回 done+0,否则基建压力会被当模型失败入训(2026-08-11
    审阅 P0-5 修正;此前的 done+0 语义废弃)。infra 失败率经 infra_failure
    记账单列监控。
    """

    def __init__(
        self,
        sandbox: CloudSandbox,
        *,
        assets_dir: Path,
        initial_obs: str,
        cfg: TerminalCfg,
        max_env_steps: int,
    ) -> None:
        self._sbx = sandbox
        self._assets = assets_dir
        self._initial_obs = initial_obs
        self._cfg = cfg
        self._max_env_steps = max_env_steps
        self._steps = 0
        self.infra_failure = ""       # 非空 = episode 因基建故障终局(可观测)

    @property
    def initial_obs(self) -> str:
        return self._initial_obs

    def step(self, command: str) -> tuple[str, bool, float]:
        self._steps += 1
        try:
            r = self._sbx.run(
                f"bash {_BASH_WRAPPER_PATH_QUOTED} {shlex.quote(command)}",
                self._cfg.command_timeout_s,
            )
        except CloudboxError as err:
            self.infra_failure = str(err)[:500]
            raise
        output = r.stdout or ""
        if r.stderr:
            output += f"\n{r.stderr}" if output else r.stderr
        if r.exit_code == 124:
            timed_out = f"Command timed out after {int(self._cfg.command_timeout_s)}s."
            output = f"{timed_out}\n{output}" if output else timed_out

        if SUBMIT_MARKER in output:
            return self._run_tests()

        truncated = truncate_observation(output) if output else "(no output)"
        obs = f"{truncated}\n\n(exit_code={r.exit_code})"
        if self._steps >= self._max_env_steps and self._cfg.score_at_budget:
            # TB-style verify-at-end(评测近似口径;训练默认关)。
            test_obs, _done, score = self._run_tests()
            return f"{obs}\n\n{test_obs}", True, score
        return obs, False, 0.0

    def _run_tests(self) -> tuple[str, bool, float]:
        """判分:tests 此刻才上传(防偷看)→ bash /tests/test.sh →
        reward.txt。观测文本与 TMax _run_tests 同构。"""
        tests_dir = self._assets / "tests"
        try:
            for attempt in range(2):
                self._sbx.upload_dir(str(tests_dir), "/tests")
                self._sbx.run("chmod +x /tests/test.sh 2>/dev/null || true", 30)
                check = self._sbx.run("test -f /tests/test.sh && echo EXISTS", 30)
                if check.stdout.strip() == "EXISTS":
                    break
                logger.warning(
                    "[%s] /tests/test.sh missing after upload attempt %d",
                    self._sbx.name, attempt + 1,
                )
            else:
                raise CloudboxError("tests upload never materialized test.sh")
            # run_polled:判分最长 600s,同样不可依赖长活 websocket。
            result = self._sbx.run_polled(
                "bash /tests/test.sh",
                self._cfg.test_timeout_s,
                grace_s=self._cfg.test_grace_s,
            )
            reward = self._parse_reward()
        except CloudboxError as err:
            self.infra_failure = str(err)[:500]
            raise
        stdout = truncate_observation(result.stdout) if result.stdout else ""
        stderr = truncate_observation(result.stderr) if result.stderr else ""
        obs = (
            f"Test execution complete.\n"
            f"Exit code: {result.exit_code}\n"
            f"[STDOUT]\n{stdout}\n"
            f"[STDERR]\n{stderr}\n"
            f"Reward: {reward}"
        )
        return obs, True, reward

    def _parse_reward(self) -> float:
        try:
            r = self._sbx.run("cat /logs/verifier/reward.txt", 30)
            if r.exit_code == 0 and r.stdout.strip():
                return max(0.0, min(1.0, float(r.stdout.strip())))
        except (ValueError, TypeError, CloudboxError):
            pass
        return 0.0

    def close(self) -> None:
        self._sbx.kill()


class TerminalTaskAdapter:
    """TaskAdapter bound to one TMax-15K row."""

    def __init__(
        self,
        item: dict,
        cfg: TerminalCfg,
        *,
        spawn_gate: threading.Semaphore,
        factory=None,
    ) -> None:
        self._factory = factory
        self._item = item
        self._cfg = cfg
        self._gate = spawn_gate

    # ------------------------------------------------------------- identity

    @property
    def task_id(self) -> str:
        return str(self._item["task_id"])

    @property
    def task_type(self) -> str:
        return "terminal"

    @property
    def image(self) -> str:
        """该任务的容器镜像。

        两种语料的环境交付方式不同,故镜像来源分两档:
        * **TMax-15K**(训练语料):共享胖基底 + per-task 增量快照 ⇒ 全语料同一
          个 `cfg.image`,差异在快照里。
        * **Terminal-Bench**(测试集):官方就是逐任务一个镜像
          (`environment/Dockerfile`,基底各异),没有共享基底可言 ⇒ 镜像名随
          数据行走(`extra_info.image`),由 build_tb_dataset 写入。
        缺省回落 `cfg.image`,故 TMax 侧零行为变化。
        """
        return str(self._item.get("image") or self._cfg.image)

    @property
    def initial_obs(self) -> str:
        return str(self._item["user_prompt"])

    # ------------------------------------------------------------- protocol

    def task_brief(self) -> str:
        """给 researcher 的任务简介。

        **代词消歧是刚需(2026-08-11 实测)**:TMax 的任务说明书是**以第二
        人称写给 executor 的**("You are tasked with fixing…"),而终端域的
        说明书长达 3.4-6.5k 字符 —— 实测占 researcher system prompt 的
        **44%**,是角色段(4.9%)的 9 倍,且位置更靠后(recency)。裸嵌进去
        会让两个 "you" 打架:角色段说"你是研究员,从不亲自操作环境",说明书
        说"你要去修这个 pipeline"。后果实测可见:9B 每个会话都有 1-2 轮去调
        不存在的 `bash` 工具(6 会话 10 个废轮),把自己当成了 executor。
        ALFWorld 不暴露此问题(任务描述仅一句话),故这是**终端域特有的接入
        适配**,在 adapter 侧处理,通用 prompts.py 零改动。
        """
        return (
            "Environment: a Linux terminal sandbox (Ubuntu). The executor "
            "solves a coding/ops task by issuing bash commands through the "
            "bash(command) tool, one command per turn; cd/export persist "
            "between turns (persistent shell). It submits by running "
            f"`echo {SUBMIT_MARKER}`.\n\n"
            "The block below is the task statement **handed to the executor**, "
            "quoted verbatim. It is written in the second person addressing "
            "the EXECUTOR: every \"you\"/\"your\" inside the block refers to "
            "the executor, never to you. You do not carry out any of these "
            "instructions yourself — you study them, then dispatch the "
            "executor to attempt them.\n"
            "<<<EXECUTOR_TASK_STATEMENT\n"
            f"{self.initial_obs}\n"
            "EXECUTOR_TASK_STATEMENT>>>\n\n"
            "(End of the executor's task statement. You are the RESEARCHER "
            "again: your only actions are the research instruments — you have "
            "no shell of your own.)\n\n"
            "Evaluation: binary — hidden tests run only after the executor "
            "submits; no submission means score 0 (no partial credit). "
            f"Episode step budget: {protocol.MAX_ENV_STEPS} tool calls; "
            f"each command has a {int(self._cfg.command_timeout_s)}s timeout; "
            "long observations are truncated to head+tail."
        )

    def executor_scaffold(self) -> str:
        return SYSTEM_PROMPT

    def _snapshot_path(self):
        """该任务的环境快照路径(**宿主侧**;有才返回)。

        存在性判定走 factory 预载的索引 —— snapshot_root 是 docker 宿主的
        路径,客户端(A800)看不到;而 docker 对不存在的 bind 源会自动建**空
        目录**,会静默把还原弄坏。索引缺省时退回本地检查(客户端==宿主)。"""
        root = self._cfg.snapshot_root
        if not root or self._cfg.backend != "docker":
            return None
        idx = self._factory.snapshot_ids if self._factory is not None else None
        if idx is not None:
            return (Path(root) / f"{self.task_id}.tgz") if self.task_id in idx else None
        p = Path(root) / f"{self.task_id}.tgz"
        return p if p.is_file() else None

    def _spawn_sandbox(self):
        """按 cfg.backend 造沙箱 —— acs(阿里云)/ docker(162 环境池)。
        两者方法面与错误类型完全一致(dockerbox 复用 ExecResult/CloudboxError),
        故 episode 交互语义零变化。"""
        cfg = self._cfg
        if cfg.backend == "docker":
            from ..dockerbox import DockerSandbox  # noqa: PLC0415

            vols = None
            snap = self._snapshot_path()
            if snap is not None:
                # 快照与 docker 同在宿主 ⇒ bind-mount,客户端零传输
                vols = {str(snap): {"bind": _SNAP_IN_CONTAINER, "mode": "ro"}}
            return DockerSandbox(
                image=self.image,
                task_id=self.task_id,
                cpus=cfg.pod_cpu,
                memory=cfg.pod_memory,
                docker_host=cfg.docker_host,
                create_timeout_s=cfg.create_timeout_s,
                run_label=cfg.run_label,
                pool_size=max(16, cfg.spawn_concurrency),
                volumes=vols,
                # 逐 episode 注入的环境变量(terminal.container_env)。TB 判分
                # 用 `uvx -p 3.13 -w pytest…`,uv 默认从 github.com 取 Python
                # standalone 与包 —— GitHub 在 162 不可达,不注入国内镜像则**每个
                # TB 任务 reward 恒 0**(2026-08-12 golden 实测:官方参考解也 0 分;
                # 注入后同一任务 reward=1)。TMax 任务不消费,置空即原行为。
                environment=_parse_env(cfg.container_env),
                # github.com → 127.0.0.1:让注定失败的 curl 秒退而非等 129s 超时
                extra_hosts=_parse_env(cfg.container_hosts),
            )
        return CloudSandbox(
            kubeconfig=cfg.kubeconfig,
            namespace=cfg.namespace,
            image=self.image,
            image_pull_secret=cfg.image_pull_secret,
            cpu=cfg.pod_cpu,
            memory=cfg.pod_memory,
            task_id=self.task_id,
            create_timeout_s=cfg.create_timeout_s,
            run_label=cfg.run_label,
        )

    def open_episode(self) -> TerminalEnvHandle:
        assets = Path(self._cfg.task_data_root) / self.task_id
        if not (assets / "tests" / "test.sh").is_file():
            raise CloudboxError(
                f"task assets incomplete for {self.task_id}: expected "
                f"{assets}/[setup.sh, tests/test.sh] (check terminal.task_data_root)"
            )
        with self._gate:
            sbx = self._spawn_sandbox()
        try:
            self._provision(sbx, assets)
        except Exception:
            sbx.kill()
            raise
        return TerminalEnvHandle(
            sbx,
            assets_dir=assets,
            initial_obs=self.initial_obs,
            cfg=self._cfg,
            max_env_steps=protocol.MAX_ENV_STEPS,
        )

    def _provision(self, sbx: CloudSandbox, assets: Path) -> None:
        """Deviation #1(runtime setup)+ TMax reset 骨架(逐字节命令)。
        顺序忠实于原镜像生命周期:setup(≙镜像 %post)→ 骨架(≙harness
        reset)。setup 失败 fail-loud——静默继续会把基建残破伪装成任务失败。"""
        snap = self._snapshot_path()
        if snap is not None:
            # ---- 快照还原路径(方案 A:复刻 TMax 预建;实测 107s→~3s)----
            # 语义等价性已实测:文件树/md5/useradd 用户身份与原生 setup 逐一致。
            r = sbx.run(f"cd / && tar -xzf {_SNAP_IN_CONTAINER}",
                        self._cfg.command_timeout_s)
            if r.exit_code != 0:
                raise CloudboxError(
                    f"snapshot restore failed for {self.task_id} "
                    f"(rc={r.exit_code}): {(r.stderr or r.stdout)[-400:]}"
                )
        else:
            setup_sh = (assets / "setup.sh").read_text(encoding="utf-8")
            sbx.write_file("/tmp/.trace_setup.sh", setup_sh)
        # run_polled:setup 是分钟级长命令,长活 websocket 会中途掉线
        # (2026-08-11 瘦镜像 setup 门 3/5 实测);分离执行 + 短轮询根治。
            r = sbx.run_polled(
                "cd / && bash /tmp/.trace_setup.sh",
                self._cfg.setup_timeout_s,
                grace_s=120.0,
            )
            if r.exit_code != 0:
                raise CloudboxError(
                    f"setup.sh failed for {self.task_id} (rc={r.exit_code}): "
                    f"{(r.stderr or r.stdout)[-800:]}"
                )
        sbx.run("mkdir -p /workspace /output /logs/verifier", 30)
        sbx.run(
            "mkdir -p /workspace /root && "
            "cd /workspace && "
            '[ -d /app ] || { _P="$(pwd)"; [ "$_P" != "/" ] && ln -sf "$_P" /app; } && '
            f"printf '%s\\n' /app > {shlex.quote(_BASH_CWD_PATH)} && "
            f": > {shlex.quote(_BASH_ENV_PATH)}",
            30,
        )
        sbx.write_file(_BASH_WRAPPER_PATH, _BASH_WRAPPER)
        sbx.run(f"chmod +x {_BASH_WRAPPER_PATH_QUOTED}", 30)

    def gt_text(self) -> str | None:
        return None                    # TMax 无可执行 oracle(深读 §4.1);
                                       # 特权文档走 steering.doc_table 通道

    def domain_notes(self) -> str:
        return (
            "Key events in this environment: which files/dirs the agent "
            "explored; compile/test command outcomes (exit codes); edits made "
            "(sed/heredoc) and whether later runs confirmed them; command "
            "timeouts or truncated outputs and how the agent adapted; whether "
            "and when it submitted."
        )

    def normalize_action(self, raw: str) -> str:
        return raw                     # bash 命令大小写/空白敏感,零改写

    def observation_indicates_invalid(self, observation: str) -> bool:
        return False                   # 终端无"无效动作"环境语义(非零退出
                                       # 码也是有信息的真实执行)

    def format_reminder(self) -> str:
        return TOOL_CALL_FORMAT_ERROR_MESSAGE

    def tools(self) -> list[dict] | None:
        return TOOLS_FC

    def react(self, env: TerminalEnvHandle, assistant_text: str):
        """FC react(与 alfworld/SSB 同框架 + TMax 错误文案):每回复恰一个
        bash 调用;no-call → TMax 格式错文案(裸文本,不耗步——偏离 TMax
        RL 的静默终止,proposal 裁定 #3);unknown tool / 缺 command → TMax
        文案包 tool 响应且耗步(与 TMax step 计数一致)。"""
        from ...core.adapter import ActionOutcome

        calls = protocol.parse_tool_calls(assistant_text, tools=self.tools())
        thought = protocol.strip_tool_calls(assistant_text)
        if not calls:
            return ActionOutcome(
                thought=assistant_text.strip(),
                action_raw="",
                action_norm="",
                feedback=self.format_reminder(),
                done=False,
                score=0.0,
                acted=False,
                valid=False,
                consumed_env_step=False,
            )
        extra_note = ""
        if len(calls) > 1:
            extra_note = (
                f"\n(note: {len(calls) - 1} extra tool call(s) in the same reply "
                "were ignored — make exactly one call per turn.)"
            )
        call = calls[0]
        if call["name"] != "bash":
            return ActionOutcome(
                thought=thought,
                action_raw=call["name"],
                action_norm="",
                feedback=protocol.tool_response_wrap(
                    format_error_message(
                        f"Unknown tool '{call['name']}'. The only available "
                        "tool is `bash`."
                    ) + extra_note
                ),
                done=False,
                score=0.0,
                acted=True,
                valid=False,
                consumed_env_step=True,
            )
        command = str(call["arguments"].get("command", "") or "")
        if not command:
            return ActionOutcome(
                thought=thought,
                action_raw=command,
                action_norm="",
                feedback=protocol.tool_response_wrap(
                    format_error_message("'command' parameter is required.")
                    + extra_note
                ),
                done=False,
                score=0.0,
                acted=True,
                valid=False,
                consumed_env_step=True,
            )
        obs, done, score = env.step(command)
        return ActionOutcome(
            thought=thought,
            action_raw=command,
            action_norm=command,
            feedback=protocol.tool_response_wrap(obs + extra_note),
            done=done,
            score=score,
            acted=True,
            valid=True,
            consumed_env_step=True,
        )

    def replay_action(self, env: TerminalEnvHandle, action_norm: str) -> tuple[str, bool]:
        obs, done, _score = env.step(action_norm)
        return obs, done


class TerminalEnvFactory:
    """Mints per-task adapters sharing one sandbox-create gate (per process)."""

    def __init__(self, cfg: TerminalCfg) -> None:
        # 必填键按后端分:acs 需 kubeconfig(k8s API);docker 走 DOCKER_HOST
        # (空则取环境变量,故不强制填 cfg.docker_host)。
        required = ["task_data_root", "image"]
        if cfg.backend == "acs":
            required.append("kubeconfig")
        missing = [k for k in required if not getattr(cfg, k)]
        if missing:
            raise ValueError(
                f"terminal config incomplete (backend={cfg.backend}): {missing} "
                "required (set TERMINAL_TASK_DATA / TERMINAL_KUBECONFIG / "
                "TERMINAL_IMAGE; with backend=docker set DOCKER_HOST instead)"
            )
        if not Path(cfg.task_data_root).is_dir():
            raise ValueError(
                f"terminal.task_data_root not a directory: {cfg.task_data_root!r} "
                "(unpack tmax-task-data.tar.gz there first)"
            )
        self._cfg = cfg
        self._gate = threading.Semaphore(cfg.spawn_concurrency)
        #: 快照索引(客户端可见的 task_id 集合;None = 未提供,退回本地检查)。
        #: snapshot_root 是 **docker 宿主侧**路径,客户端看不到那个目录,而
        #: docker 对不存在的 bind 源会自动建空目录 ⇒ 必须靠索引判存在。
        self.snapshot_ids = None
        if cfg.snapshot_index:
            idx = Path(cfg.snapshot_index)
            if not idx.is_file():
                raise ValueError(
                    f"terminal.snapshot_index not found: {idx} (generate it on the docker "
                    "host with `ls <snapshot_root>/*.tgz | xargs -n1 basename | "
                    "sed s/.tgz//` and copy it to the client)"
                )
            self.snapshot_ids = {
                ln.strip() for ln in idx.read_text().splitlines() if ln.strip()
            }
            logger.info("terminal snapshot index: %d tasks", len(self.snapshot_ids))

    def adapter_from_item(self, item: dict) -> TerminalTaskAdapter:
        return TerminalTaskAdapter(item, self._cfg, spawn_gate=self._gate,
                                   factory=self)
