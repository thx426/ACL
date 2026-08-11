"""SWE 域 prompt 单一真源(训练与评测共用 —— 口径一致是硬要求)。

来源:SWE-Master `config/openhands/edit_fn_calling.yaml`(RL 实际配置)的
system_prompt / instance_prompt,逐字。工具块不在此手写 —— 由 chat template
按 adapter.tools() 原生渲染(分布内形态)。

训练(R2E-Gym)与评测(SWE-bench Verified)用**同一份** prompt:两侧任务
形态相同(改 /testbed 里的非测试文件修 issue),prompt 漂移会让评测读数
不可归因。注意 instance_prompt 里"测试文件已处理好"这句在两侧含义一致
(agent 都不该改测试):R2E 侧测试在仓库里可见;Verified 侧 F2P 测试由
harness 在判分前才打入,agent 本就碰不到。
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a helpful assistant that can interact with a computer to solve "
    "tasks. You are provided with a local git repository and a GitHub issue. "
    "Your goal is to identify, implement, and verify a solution to the issue "
    "while documenting your reasoning process. You have access to tools for "
    "viewing and editing files, executing bash commands, and submitting your "
    "final solution."
)


def build_instruction(issue: str) -> str:
    """instance_prompt(openhands 套件语义;{problem_statement} 已剥 [ISSUE] 壳)。"""
    return (
        "I have uploaded a python code repository in the /testbed directory.\n\n"
        "Now consider the following GitHub issue:\n\n"
        f"<github_issue>\n{issue}\n</github_issue>\n\n"
        "Can you help me implement the necessary changes to the repository to "
        "fix the <github_issue>? I have already taken care of all changes to "
        "any of the test files described in the <github_issue>. This means you "
        "DON'T have to modify the testing logic or any of the tests in any "
        "way! Your task is to make changes to non-test files in the /testbed "
        "directory to ensure the <github_issue> is resolved."
    )
