"""Optional researcher instruments for the approved evolution workflow."""

EVOLUTION_INSTRUCTIONS = """
=== ERROR LOCALIZATION — LOCAL REVISION — INDEPENDENT VALIDATION ===
This mode overrides the experiment and document-submission instructions above.
Keep your researcher role: review the task, inspect experiments, fetch raw
evidence, and explain your reasoning before calling tools.

run_experiment() samples the frozen executor and CURRENT Skill using the
configured discovery prompts (four samples each by default). You cannot
change the Skill by supplying experimental guidance. Multiple prompts serve
failure discovery only. Formal validation always uses ONE fixed prompt.

Read the first verifiable error record. Check the task requirement, preceding
observation, action and feedback. Never infer the first error from the terminal
failure alone. record_error can attach a diagnosis and evidence to a trace;
unsupported diagnoses remain uncertain. Missing-precondition and wrong-choice
diagnoses need an environment-specific independent verifier.

Choose ONE major error_type for a round: illegal_action, missing_precondition,
wrong_choice or repeated_loop. propose_revision supplies only local rules.
Each rule must include trigger, checks, action, completion and recovery, plus
source error IDs. Existing rules of other error types cannot be changed.
Do not invent facts unavailable to the executor. New findings must be expressed
as checks the executor can perform, not as hidden answers.

validate_revision runs a sealed candidate against the old Skill with fresh,
paired seeds. It checks exact-state local correction when supported, then full
episodes from the original start, and held-out tasks. The program decides
replacement using thresholds fixed before the proposal. Local improvement
alone is insufficient. Rejection keeps the old Skill; inspect the recorded
reason and collect new discovery evidence or propose another local correction.

finalize(self_assessment=...) releases ONLY the current accepted Skill.
Do not write a replacement experience_doc at finalize. The ordinary document
layout instructions apply to legacy mode only; this mode renders the five rule
fields mechanically. Stop at the budget limit and report unresolved errors.
"""


def evolution_schemas():
    def tool(name, description, props, required=()):
        return {"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": props,
                           "required": list(required), "additionalProperties": False}}}

    kind = {"type": "string", "enum": ["illegal_action", "missing_precondition",
                                         "wrong_choice", "repeated_loop"]}
    strings = {name: {"type": "string"} for name in
               ("rule_id", "trigger", "checks", "action", "completion", "recovery")}
    strings["sources"] = {"type": "array", "items": {"type": "string"}, "minItems": 1}
    return [
        tool("record_error", "Attach a trace-based diagnosis. The program independently checks "
             "the earliest supported error; unconfirmed claims are recorded as uncertain.",
             {"episode_id": {"type": "string"}, "step": {"type": "integer", "minimum": 1},
              "error_type": kind, "evidence": {"type": "string"}},
             ("episode_id", "step", "error_type", "evidence")),
        tool("propose_revision", "Propose local rules for exactly one error type. Each rule "
             "cites verified discovery error IDs and supplies all five operational fields.",
             {"error_type": kind, "rules": {"type": "array", "minItems": 1, "maxItems": 8,
              "items": {"type": "object", "properties": strings,
                        "required": list(strings), "additionalProperties": False}}},
             ("error_type", "rules")),
        tool("validate_revision", "Independently evaluate the sealed candidate using fixed "
             "prompts, fresh seeds, full episodes and held-out tasks. Returns replacement decision.",
             {"candidate_id": {"type": "string"}}, ("candidate_id",)),
    ]
