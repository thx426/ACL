"""Template benchmark: a deterministic toy "signal patchboard" environment.

这是引入新 benchmark 的**参考实现**,不是生产环境。它与 ALFWorld 零共享
(独立动作文法、独立环境习语、部分分评估),用来示范:机制核心从不假定
任何任务协议——所有任务内容都经 ``core/adapter.py`` 的三个契约进入:

  EnvFactory     — dataset item(= parquet 行的 ``extra_info``)→ TaskAdapter
  TaskAdapter    — 身份/资产面 + 交互面(react / replay_action)
  ActionOutcome  — 每轮交互的唯一回传结构(动作语法、反馈、评估全在其后)

玩具域:一块信号接线板。若干信号端口(s1, s2, ...)与输出口(out1, ...),
任务要求把指定的信号接到指定输出,然后运行诊断验收。动作文法:

    probe <port>          查看一个端口
    link <a> <b>          接一根线
    run diagnostics       验收并结束(终局动作)

评估是**部分分**:score = 接对的线 / 要求的线(∈[0,1])——示范
ActionOutcome.score 不被二元假定;当前 GRPO 口径的 won 由 score>=1.0 派生。
"""
from __future__ import annotations

from ...core import protocol
from ...core.adapter import ActionOutcome, EnvHandle  # noqa: F401 (contract ref)

# --------------------------------------------------------------------------
# 1) 环境本体(任务私有,core 永不 import)
#
# 确定性红线:同一 dataset item 每次 open_episode 必须逐字节复现同样的
# initial_obs 与同样的 (动作→反馈) 映射。机制在两处依赖它:
#   * episode 开局把 live initial_obs 与 dataset 里捕获的比对,漂移 = 基建故障;
#   * v4 前缀续写把既往动作重放到新 episode 并逐步比对反馈,漂移 = 样本作废。
# 有真实引擎的 benchmark 要为此固定 seed/版本;本玩具是纯状态机,天然确定。
# --------------------------------------------------------------------------

#: 无效输入的环境响应(本 benchmark 的习语——故意不同于任何现有 benchmark,
#: 证明 core 不依赖具体字面量;判定经 adapter.observation_indicates_invalid)
REJECT_OBS = "The console rejects that input."


class PatchboardEnv:
    """One live episode. 满足 core 的 EnvHandle 契约(initial_obs/close);
    ``step`` 是本实现的私有细节,仅被本文件的 react/replay_action 消费——
    core 从不直接驱动环境,所以这里想用什么执行接口都行(字符串命令、
    (tool, args) 元组、进程 RPC ...)。"""

    #: 信号名(确定性分配;探索结构:executor 需 probe 两端才能推出配对)
    SIGNALS = ("alpha", "beta", "gamma", "delta", "epsilon")

    def __init__(self, spec: dict[str, str], initial_obs: str) -> None:
        self._required = dict(spec)          # {signal_port: output_port}
        self._links: set[tuple[str, str]] = set()
        self._initial_obs = initial_obs
        self._ports = sorted(set(self._required) | set(self._required.values()))
        self._signal_of: dict[str, str] = {}
        for i, (s, o) in enumerate(sorted(self._required.items())):
            self._signal_of[s] = self.SIGNALS[i % len(self.SIGNALS)]
            self._signal_of[o] = self.SIGNALS[i % len(self.SIGNALS)]
        self.closed = False

    @property
    def initial_obs(self) -> str:
        return self._initial_obs

    def step(self, command: str) -> tuple[str, bool, float]:
        """(observation, done, score)。score 只在 done 时有意义。
        注意返回的是标量分,不是布尔——终局评估形态由 benchmark 自定,
        react 负责把它装进 ActionOutcome.score。"""
        parts = command.split()
        if parts[:1] == ["probe"] and len(parts) == 2:
            port = parts[1]
            if port not in self._ports:
                return REJECT_OBS, False, 0.0
            sig = self._signal_of[port]
            role = "carries" if port in self._required else "expects"
            wired = sorted(b for a, b in self._links if a == port)
            tail = f"; wired to {', '.join(wired)}" if wired else ""
            return f"Port {port} {role} signal {sig}{tail}.", False, 0.0
        if parts[:1] == ["link"] and len(parts) == 3:
            a, b = parts[1], parts[2]
            if a not in self._ports or b not in self._ports:
                return REJECT_OBS, False, 0.0
            self._links.add((a, b))
            return f"A cable now joins {a} and {b}.", False, 0.0
        if command.strip() == "run diagnostics":
            good = sum(1 for s, o in self._required.items() if (s, o) in self._links)
            total = len(self._required)
            score = good / total
            if good == total:
                return "Diagnostics pass. All signals routed.", True, score
            return (
                f"Diagnostics report {good} of {total} links correct.",
                True,
                score,
            )
        return REJECT_OBS, False, 0.0

    def close(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------
# 2) TaskAdapter 实现(核心交付;逐成员为一个契约点)
# --------------------------------------------------------------------------


class TemplateTaskAdapter:
    """One adapter per task. 构造输入 item = parquet 行的 ``extra_info``
    dict——这是装配契约:训练(agent_loop)、评估(run_eval)、SFT 采样
    (sftgen_researcher)都只把 extra_info 交给工厂,所以 extra_info 必须
    自足(本 benchmark 需要的一切:身份、场景、任务参数、GT)。"""

    def __init__(self, item: dict) -> None:
        self._item = dict(item)
        self._spec: dict[str, str] = dict(item["spec"])

    # ---- 身份面 ----------------------------------------------------------

    @property
    def task_id(self) -> str:
        return str(self._item["task_id"])

    @property
    def task_type(self) -> str:
        """类别/family:SFT 分层抽样与分组统计的键。无自然分类的
        benchmark 返回常量即可。"""
        return str(self._item["task_type"])

    @property
    def initial_obs(self) -> str:
        """dataset 构建时捕获的初始观察(确定性锚点,见文件头红线)。"""
        return str(self._item["initial_obs"])

    # ---- researcher 面(特权侧) -----------------------------------------

    def task_brief(self) -> str:
        """researcher 的任务面板:目标、环境机制、**动作文法**、评估口径。
        通用 researcher 模板对交互形态零假定——executor 怎么动、能发什么,
        全靠这里说清楚。绝不包含 GT。"""
        ports = sorted(set(self._spec) | set(self._spec.values()))
        return (
            "Environment: a signal patchboard with ports "
            f"{', '.join(ports)}. Signal ports carry a named signal; output "
            "ports expect one. Which signal a port carries or expects is "
            "only revealed by probing it.\n"
            "Goal: connect every signal port to the output port expecting "
            "its signal, then run diagnostics to finish.\n"
            "Executor actions (one per turn, plain text):\n"
            "  probe <port>       reveal what the port carries/expects and "
            "its current wiring\n"
            "  link <a> <b>       connect two ports with a cable\n"
            "  run diagnostics    score the board and end the episode\n"
            f"Any other input gets: \"{REJECT_OBS}\"\n"
            "Scoring at diagnostics time: fraction of required links present "
            "(1.0 = success; partial credit for partially correct boards)."
        )

    def executor_scaffold(self) -> str:
        """executor 的 system prompt(锁定,researcher 不可改;guidance 经
        types.inject_guidance 以统一 header 追加)。声明动作文法与本任务的
        输出协议——这里用 core 的文本单动作协议,故按其格式契约书写。"""
        return (
            "You are a technician at a signal patchboard. Connect every "
            "signal port to the output port expecting its signal, then run "
            "diagnostics to finish.\n\n"
            "Available commands:\n"
            "  probe <port>       inspect a port (what it carries or "
            "expects, and its wiring)\n"
            "  link <a> <b>       connect two ports with a cable\n"
            "  run diagnostics    score the board and end the episode\n\n"
            "On every turn respond with exactly this format and nothing "
            "else:\n<reasoning>brief reasoning about the current "
            "situation</reasoning>\n<action>one command</action>"
        )

    # ---- 环境面 ----------------------------------------------------------

    def open_episode(self) -> PatchboardEnv:
        """新开一局(确定性)。可阻塞(子进程/网络)——调用方一律放线程池;
        重资源 benchmark 在 Factory 持共享闸门(参照 alfworld 的
        spawn_gate 模式),本玩具无需。"""
        return PatchboardEnv(self._spec, self._item["initial_obs"])

    def gt_text(self) -> str | None:
        """特权参考(walkthrough/gold answer)。信息状态红线:只在 GT 开关
        打开时进 researcher 提示词;永不进 executor、永不进 SFT 数据。"""
        gt = self._item.get("gt_text") or ""
        return gt or None

    def domain_notes(self) -> str:
        """给 narrator 的可选领域提示(什么算关键事件);通用契约够用时
        返回空串。"""
        return ""

    def tools(self) -> list[dict] | None:
        """本玩具是文本动作协议 ⇒ 无 tools。原生 function-calling 型
        benchmark 在此返回 OpenAI 格式 schema 列表(单一真源),由
        chat template 原生渲染进 prompt——绝不在 scaffold 里手写渲染块
        (参照 envs/spreadsheetbench 的 TOOLS)。"""
        return None

    # ---- 交互面(react 抽象:机制核心唯一消费的执行接口) ----------------

    def react(self, env: PatchboardEnv, assistant_text: str) -> ActionOutcome:
        """本 benchmark 是文本单动作协议 ⇒ 一行委托 core 的标准实现件
        (它组合下面三个协议件:normalize / invalid 判定 / 格式提醒)。

        ⚠️ 正式方向是统一 Function Call(2026-08-08 用户裁定):新 benchmark
        一律 FC——tools() 返回 OpenAI schema、react 自写,解析用
        ``protocol.parse_tool_calls(text, tools=self.tools())``(家族分派 +
        schema 类型还原)、观察经 ``protocol.tool_response_wrap`` 包裹、提醒
        用 ``protocol.fc_format_reminder``,并在 envs/preflight.py 的 FC 判定
        里登记环境名。范本:envs/alfworld(单工具)、envs/spreadsheetbench 与
        envs/math(多工具)。FC react 骨架::

            def react(self, env, assistant_text):
                calls = protocol.parse_tool_calls(assistant_text,
                                                  tools=self.tools())
                if not calls:
                    return ActionOutcome(thought=assistant_text.strip(),
                                         action_raw="", action_norm="",
                                         feedback=self.format_reminder(),
                                         done=False, score=0.0, acted=False,
                                         valid=False, consumed_env_step=False)
                result = env.execute(calls)                # 任意执行接口
                return ActionOutcome(
                    thought=calls.rationale, action_raw=calls.raw,
                    action_norm=calls.canonical, feedback=result.text,
                    done=result.finished, score=result.score,
                    acted=True, valid=result.recognized, consumed_env_step=True)

        唯一硬约束:坏的模型输出返回 acted=False 的 ActionOutcome(不抛),
        异常只留给基建故障。

        本环境 step 的第三个返回值是 [0,1] 部分分——标准实现件按 float
        保真透传(二元 benchmark 返回 bool 同样成立)。"""
        return protocol.text_action_react(self, env, assistant_text)

    def replay_action(self, env: PatchboardEnv, action_norm: str) -> tuple[str, bool]:
        """确定性重放钩子(v4 前缀续写的硬前提):把一条已录制动作
        (Step.action_norm)重新作用于新 episode,返回 (反馈, 是否终局)。
        调用方逐步比对反馈,漂移即作废样本。"""
        obs, done, _score = env.step(action_norm)
        return obs, done

    # ---- 文本动作协议件(text_action_react 的消费面;非文本协议的
    #      benchmark 实现为平凡值即可) ------------------------------------

    def normalize_action(self, raw: str) -> str:
        """保守、语义免修复的规范化(空白折叠;绝不猜测/纠正命令——
        不合文法的动作必须触达环境并可见地失败)。"""
        return " ".join(raw.split()).lower()

    def observation_indicates_invalid(self, observation: str) -> bool:
        """环境无效响应判定(观测字段,永不进 loss)。"""
        return observation.strip() == REJECT_OBS

    def format_reminder(self) -> str:
        """格式失败提醒 = 本环境习语前缀 + core 的通用协议提醒,使它读
        起来像一次正常环境响应。"""
        return REJECT_OBS + " " + protocol.DEFAULT_FORMAT_REMINDER


# --------------------------------------------------------------------------
# 3) EnvFactory 实现(装配契约:进程级一个实例,持共享资源)
# --------------------------------------------------------------------------


class TemplateEnvFactory:
    """``core.adapter.EnvFactory``:dataset item(= parquet 行的
    ``extra_info``)→ TaskAdapter。装配层经 ``envs/registry.py`` 的
    builder 构造本类——机制代码从不 import 本模块。

    本玩具无重资源;要起真实引擎/连接池的 benchmark 在这里持共享闸门
    (参照 alfworld:threading.Semaphore(spawn_concurrency) 传给每个
    adapter,boot 并发受控)。benchmark 专属配置照 AlfworldCfg 模式加
    自己的 config section,由 registry builder 取用并做 per-worker 折算。"""

    def adapter_from_item(self, item: dict) -> TemplateTaskAdapter:
        return TemplateTaskAdapter(item)
