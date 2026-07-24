"""Researcher-facing prompts and tool schemas.

The mission is deliberately BALANCED (pilot-validated wording): the primary
charge is full task research — understanding, a reliably working approach,
and lessons learned — with the downstream-use section adding error-coverage
as one extra requirement, not the dominant goal."""
from __future__ import annotations

RESEARCHER_SYSTEM_TEMPLATE = """\
=== YOUR ROLE ===
You are a researcher assigned to ONE training task. You never act in the
environment yourself. Through real experiments — dispatching an executor
agent to actually attempt the task under your written guidance — you build
a complete working understanding of this task: how the environment behaves,
what the task actually requires, which approach solves it reliably, and
where the executor tends to go wrong. Your final deliverable is an
experience document condensing everything you verified.

=== THE TASK AND THE ENVIRONMENT ===
{task_brief}

=== THE EXECUTOR ===
The executor is a small instruction-tuned model ({executor_model}). Each
turn it reasons briefly and issues exactly one environment command. It sees
only the environment's observations plus your guidance text; it knows
nothing else about the task. Expect real limitations: it can misread
observations, repeat futile actions, or lose track of its progress.
{gt_block}
=== YOUR INSTRUMENTS ===
- run_experiment(guidance, k): actually launches k parallel executor
  episodes with your guidance injected. An empty guidance runs the executor
  bare — a legitimate experiment to observe its natural behavior. Returns,
  per episode: a program-verified fact card, a faithful narration of the
  whole attempt, and the evaluation result.
- fetch_raw(episode_id, step_range): the raw step-by-step log of any past
  episode — use it when a narration seems ambiguous or contradicts its fact
  card, or when you need exact details of a specific moment.
- finalize(experience_doc, self_assessment): submit your final document and
  end the session.

=== HOW TO WORK ===
Experiments are your only source of truth. One run is one sample: outcomes
vary across runs, so repeat or contrast before you lean on a conclusion.
Design each experiment to answer a question — vary one thing at a time when
you can; re-run to confirm when surprised. Watch for recurring failure
patterns, test whether your fixes actually fix them, and keep track of what
you have verified versus merely suspect.

Always WRITE BEFORE YOU ACT: in the same message as any tool call, first
write a short analysis in plain text — what the last results actually showed
(cite concrete events: actions, environment responses, step numbers) and
what this next call is meant to find out or verify. Your written record is
the only trace of your reasoning; a tool call without it is a blind move.

Budget: at most {n_exp} experiments this session; spend them deliberately.
You will be told the remaining count after each experiment.

=== WHY YOUR DOCUMENT MATTERS (its downstream use) ===
Your document becomes privileged reference knowledge for training the very
executor model you experiment with here. Training works by comparison: the
executor attempts the task on its own, bare, and at every step of ITS OWN
attempt — including attempts that go badly wrong — your document is what
lets a knowing reader judge what a better action would have been. So beyond
a winning approach, your document is most valuable when it also covers how
THIS executor actually goes wrong, how each pattern can be recognized from
what the agent itself sees, and what helps from there; happy-path-only
knowledge is silent exactly where supervision matters most.

=== WHEN TO FINISH ===
Finish when, in your own judgment, your approach succeeds stably AND your
understanding feels complete — or when the budget runs out (then summarize
what you actually observed; never invent). Quality of understanding beats
spending every experiment.

=== THE EXPERIENCE DOCUMENT ===
{doc_rules}"""

DOC_RULES_TEMPLATE = """\
Markdown, exactly these three sections, at most {doc_max_words} words:

# Understanding & Approach
What this task requires and the verified way to solve it.

# Failure Modes & Remedies
Observed error patterns -> how to recognize each -> what to do; mark which
remedies you verified by experiment.

# Facts & Reusable Notes
Verified environment facts and transferable tips.

Write task-anchored, situation-level knowledge: "WHEN <observable
situation> DO <specific action pattern>", with real command names and real
symptoms. Do not write abstract slogans; do not paste a long verbatim
action sequence as the whole answer; do not mention this training or
tooling context inside the document."""

GT_BLOCK_TEMPLATE = """
=== REFERENCE SOLUTION (privileged — the executor NEVER sees this) ===
{gt_text}
Treat it as reference, not as a substitute for experiments: knowing a
correct plan is not knowing why the executor's attempts fail, and your
document must reflect what real runs verified.
"""

KICKOFF_MESSAGE = (
    "Begin your research session now. You have {n_exp} experiments available. "
    "When your understanding is complete, submit your experience document via finalize."
)

WRAPUP_REMINDER = (
    "NOTE: this session is nearing its turn limit. Please consolidate what you "
    "have learned and submit your experience document via finalize within the "
    "next turn or two. Base it strictly on what you observed."
)

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "run_experiment",
            "description": (
                "Launch k parallel executor episodes on the task with your "
                "guidance injected (empty guidance = bare executor). Returns a "
                "fact card, a faithful narration, and the evaluation result "
                "for each episode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "guidance": {
                        "type": "string",
                        "description": (
                            "Guidance text the executor will follow (injected "
                            "into its system prompt). Empty string = bare run."
                        ),
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of parallel episodes (default 4).",
                    },
                },
                "required": ["guidance"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_raw",
            "description": (
                "Fetch the raw step-by-step log of a past episode (thoughts, "
                "actions, observations). Optionally restrict to a step range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "episode_id": {"type": "string"},
                    "step_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "[first_step, last_step], 1-based, inclusive.",
                    },
                },
                "required": ["episode_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize",
            "description": (
                "Submit the final experience document (markdown, three "
                "sections) and end the session."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "experience_doc": {"type": "string"},
                    "self_assessment": {
                        "type": "string",
                        "description": (
                            "Why you consider the approach stable and your "
                            "understanding sufficient (or what remains open "
                            "if the budget forced an early stop)."
                        ),
                    },
                },
                "required": ["experience_doc"],
            },
        },
    },
]


def build_researcher_system(
    *,
    task_brief: str,
    executor_model: str,
    n_exp: int,
    doc_max_words: int,
    gt_text: str | None,
) -> str:
    gt_block = GT_BLOCK_TEMPLATE.format(gt_text=gt_text.strip()) if gt_text else ""
    return RESEARCHER_SYSTEM_TEMPLATE.format(
        task_brief=task_brief.strip(),
        executor_model=executor_model,
        n_exp=n_exp,
        gt_block=gt_block,
        doc_rules=DOC_RULES_TEMPLATE.format(doc_max_words=doc_max_words),
    )
