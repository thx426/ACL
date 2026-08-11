"""SWE 域交互协议:三工具 FC schema + 动作→容器命令映射 + git 封禁。

对照来源(experiments/bench_env/codebases/,全部实读):
  - 工具 schema:R2E-Gym `tools/__init__.py:83-125,162-178,260-271`(逐字);
    openhands/sweagent scaffold 用 str_replace_editor+execute_bash+submit
    (SWE-Master RL 实际配置,`agent/agent.py:225-232`)。
  - 动作映射:R2E-Gym `action/action.py:109-152` to_bashcmd(shlex.quote)。
  - git 封禁:R2E `tools/execute_bash.py:13` BLOCKED_BASH_COMMANDS —— 意图
    = agent 完全无 git(patch 收集在 harness 侧不经 agent);SWE-Master 加
    子串拦截补漏仍可绕(`git --no-pager log`)。我们改 shlex 解析式拦截。

三处相对上游的修正(deep-read §3.5 / harness-comparison §7):
  1. execute_bash.required 补 ["command"](上游空,模型可发空调用);
  2. git 全禁改 shlex 解析式(argv 任何位置的 git 可执行名即拒)+ .git/ 路径
     读取双侧拦截;
  3. 工具本体不入容器 /usr/local/bin —— 我们不拷 R2E 的 python 工具脚本,
     而是在 adapter 里把 FC 直接翻译成容器命令(str_replace_editor 的四个
     子命令用 python 内联实现,execute_bash 直接跑 command)。理由:ACS
     沙箱 commands.run 是无状态子 shell,拷脚本+shebang 的路线依赖 /root/.venv
     软链,增加每 episode 往返;内联翻译更薄更可控。
"""
from __future__ import annotations

import re
import shlex

#: 子命令替换 `$(git …)` / 反引号 `` `git …` `` 里的 git(shlex 会把整段当
#: 一个 token,词法层看不到,故对原串正则兜底)。
_GIT_IN_SUBSHELL = re.compile(r"(?:\$\(|`)\s*git\b")

# ---- 工具 schema(R2E openhands 套件;单一真源)------------------------------

_STR_REPLACE_EDITOR_DESCRIPTION = (
    "Custom editing tool for viewing, creating and editing files\n"
    "* State is persistent across command calls\n"
    "* If `path` is a file, `view` displays the result of `cat -n`. If `path` "
    "is a directory, `view` lists non-hidden files up to 2 levels deep\n"
    "* The `create` command cannot be used if `path` already exists as a file\n"
    "* If a `command` generates long output, it is truncated and marked "
    "`<response clipped>`\n\n"
    "Notes for `str_replace`:\n"
    "* `old_str` must match EXACTLY one or more consecutive lines from the "
    "original file (mind the whitespace!)\n"
    "* If `old_str` is not unique in the file, no replacement is made — include "
    "enough context to make it unique\n"
    "* `new_str` contains the edited lines that replace `old_str`"
)

_EXECUTE_BASH_DESCRIPTION = (
    "Execute a bash command in the /testbed working directory and return its "
    "combined stdout and stderr with the exit code. Each call runs in a fresh "
    "non-interactive shell (directory and shell-variable changes do NOT persist "
    "between calls — chain with `&&` or `cd ... &&` within a single command). "
    "git is disabled. Long output is truncated in the middle."
)

_SUBMIT_DESCRIPTION = (
    "Signal that the task is complete. Grading runs on the current repository "
    "state after submission; no grading feedback is returned. Call it only "
    "once you believe the issue is fixed."
)

ALLOWED_EDITOR_COMMANDS = ["view", "create", "str_replace", "insert"]

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": _STR_REPLACE_EDITOR_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "description": (
                            "The command to run. Allowed: "
                            + ", ".join(f"`{c}`" for c in ALLOWED_EDITOR_COMMANDS)
                            + "."
                        ),
                        "enum": ALLOWED_EDITOR_COMMANDS,
                        "type": "string",
                    },
                    "path": {
                        "description": "Absolute path, e.g. `/testbed/pkg/mod.py`.",
                        "type": "string",
                    },
                    "file_text": {
                        "description": "Required for `create`: full file content.",
                        "type": "string",
                    },
                    "old_str": {
                        "description": "Required for `str_replace`: the exact "
                        "string in `path` to replace.",
                        "type": "string",
                    },
                    "new_str": {
                        "description": "Replacement for `str_replace` (optional) "
                        "or the string to insert for `insert` (required).",
                        "type": "string",
                    },
                    "insert_line": {
                        "description": "Required for `insert`: `new_str` is "
                        "inserted AFTER this 1-indexed line.",
                        "type": "integer",
                    },
                    "view_range": {
                        "description": "Optional for `view` on a file: "
                        "[start, end] 1-indexed line range; [start, -1] to end.",
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
                "required": ["command", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": _EXECUTE_BASH_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute, e.g. "
                        "`python -m pytest r2e_tests -x`.",
                    },
                },
                "required": ["command"],  # 上游为空 → 补(修正 1)
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": _SUBMIT_DESCRIPTION,
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# ---- git 封禁(shlex 解析式;替代可绕的子串匹配)----------------------------

#: git 可执行名(含常见别名);解析 argv 后逐 token 比对,`git --no-pager
#: log` / `cd x && git log` / `g=git;$g log` 均被切成 token 后命中或落到
#: 无法解析分支(见 bash_is_blocked)。
_GIT_NAMES = frozenset({"git", "/usr/bin/git", "/usr/local/bin/git"})
_BLOCKED_FIRST = frozenset({"ipython", "jupyter", "nohup"})


#: 命令分隔符(git 若出现在这些之后的**命令位置**才是执行,不是参数)。
_CMD_SEP = re.compile(r"(?:&&|\|\||[;&|\n])")
_GIT_MSG = (
    "git is disabled in this environment (to prevent reading the fix from "
    "history). Use a different command or tool."
)


def bash_is_blocked(command: str) -> str | None:
    """返回拒绝原因(str)或 None(放行)。

    git 封禁按**命令位置**精确拦(区分 `git log`=执行 与 `grep git`=参数):
    把命令按 `&& || ; | & \\n` 切成子命令,每个子命令跳过前导 `VAR=val`
    赋值后取首词,首词是 git 可执行名即拒。另兜底:①`$(git…)`/反引号里的
    git;②变量别名(`g=git;$g log`)—— 尽力拦但 shlex 词法有残余面(见下),
    与 R2E/SWE-Master 上游同类(它们仅首词/子串,更弱)。③读 `.git/` 目录。

    残余面(诚实标注,与上游一致或更强):`eval`/`base64|sh` 等间接执行、
    动态构造的 git 调用无法在命令层完全堵死;真正的兜底是环境侧(episode
    期 .git 只读 + tests_visible 计入 baseline),命令拦截是纵深防御第一层。"""
    stripped = command.strip()
    if not stripped:
        return "empty command"
    try:
        shlex.split(stripped)  # 语法可解析性检查(不配对引号 = 可疑)
    except ValueError:
        return "command could not be parsed (unbalanced quotes?)"
    # 按分隔符切子命令,逐个查命令位置首词
    for seg in _CMD_SEP.split(stripped):
        try:
            toks = shlex.split(seg.strip())
        except ValueError:
            continue
        # 跳过前导 VAR=val 赋值(bash 允许 `A=1 cmd`)
        i = 0
        while i < len(toks) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", toks[i]):
            rhs = toks[i].split("=", 1)[1].strip("'\"").rstrip(";")
            if rhs.rsplit("/", 1)[-1] == "git":  # 别名赋值 `g=git`
                return _GIT_MSG + " (aliasing git does not bypass this)"
            i += 1
        if i < len(toks):
            head = toks[i].rsplit("/", 1)[-1]
            if head in _BLOCKED_FIRST:
                return f"`{head}` is not allowed. Use a different command or tool."
            if toks[i] in _GIT_NAMES or head == "git":
                return _GIT_MSG
    # ①c 子命令替换里的 git
    if _GIT_IN_SUBSHELL.search(stripped):
        return _GIT_MSG
    # ③ .git/ 路径读取
    if ".git/" in stripped or stripped.endswith(".git"):
        return "reading the .git directory is not allowed."
    return None


def path_is_blocked(path: str) -> str | None:
    """str_replace_editor 的 path 拦截:禁止 .git/ 内部。"""
    if ".git/" in path or path.rstrip("/").endswith(".git"):
        return "accessing the .git directory is not allowed."
    return None


# ---- 动作 → 容器命令映射(str_replace_editor 内联实现)----------------------
# ACS 沙箱是无状态子 shell:不拷 R2E 的 python 工具脚本,而是把四个 editor
# 子命令翻译成自足的 python -c 片段(view 用 cat -n / sed;编辑用 python
# 就地读改写)。execute_bash 直接跑 command。所有值经 shlex.quote。


def editor_to_command(args: dict) -> str:
    """把 str_replace_editor 的 FC 参数翻译成一条容器内 bash 命令。
    未知/缺参在此不校验(adapter 层先做必参校验并回可读错误)。"""
    cmd = args.get("command")
    path = str(args.get("path", ""))
    q = shlex.quote
    if cmd == "view":
        vr = args.get("view_range")
        if isinstance(vr, (list, tuple)) and len(vr) == 2:
            start, end = int(vr[0]), int(vr[1])
            if end == -1:
                return f"sed -n {q(f'{start},$p')} {q(path)} | cat -n"
            return (f"sed -n {q(f'{start},{end}p')} {q(path)} | "
                    f"nl -ba -v {q(str(start))}")
        # 目录 or 整文件:目录 → find 两层;文件 → cat -n
        return (f"if [ -d {q(path)} ]; then find {q(path)} -maxdepth 2 "
                f"-not -path '*/.*' | sort; else cat -n {q(path)}; fi")
    if cmd == "create":
        text = str(args.get("file_text", ""))
        # heredoc 写入(引号定界符防变量展开);父目录自动建
        return (f"mkdir -p \"$(dirname {q(path)})\" && "
                f"cat > {q(path)} <<'__TRACE_EOF__'\n{text}\n__TRACE_EOF__")
    if cmd == "str_replace":
        return _py_edit(path, "str_replace",
                        old=str(args.get("old_str", "")),
                        new=str(args.get("new_str", "")))
    if cmd == "insert":
        return _py_edit(path, "insert",
                        line=int(args.get("insert_line", 0)),
                        new=str(args.get("new_str", "")))
    return f"echo {q(f'unknown editor command: {cmd}')} >&2; exit 2"


def _py_edit(path: str, op: str, **kw) -> str:
    """就地编辑用 python -c(base64 传参避免任何引号/换行注入)。
    str_replace:old_str 必须唯一命中一次,否则不改并报错(R2E 语义)。"""
    import base64
    import json as _json

    payload = base64.b64encode(
        _json.dumps({"path": path, "op": op, **kw}).encode()
    ).decode()
    script = (
        "import base64,json,sys\n"
        f"a=json.loads(base64.b64decode('{payload}'))\n"
        "p=a['path']\n"
        "s=open(p,encoding='utf-8').read()\n"
        "if a['op']=='str_replace':\n"
        "  n=s.count(a['old'])\n"
        "  if n==0: sys.exit('str_replace: old_str not found')\n"
        "  if n>1: sys.exit(f'str_replace: old_str not unique ({n} matches); add context')\n"
        "  s=s.replace(a['old'],a['new'],1)\n"
        "else:\n"
        "  lines=s.splitlines(keepends=True)\n"
        "  i=a['line']\n"
        "  ins=a['new'] if a['new'].endswith('\\n') else a['new']+'\\n'\n"
        "  lines.insert(i,ins)\n"
        "  s=''.join(lines)\n"
        "open(p,'w',encoding='utf-8').write(s)\n"
        "print('OK')\n"
    )
    return f"python3 -c {shlex.quote(script)}"
