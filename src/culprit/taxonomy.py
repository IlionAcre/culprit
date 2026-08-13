"""The 21-class failure taxonomy L3 adjudicates against. Frozen as a StrEnum
with one-line descriptions rendered into the adjudication prompt via
`describe_all()`, so the prompt and the code that parses its output can never
drift out of sync: a class the model is offered is always a class
`Adjudication.failure_class` can validly hold, and vice versa.

Seven groups, matching the plan's grouping exactly: specification/planning
(3), tool/environment (5), information handling (3), control flow (3),
multi-agent (3), verification (2), output (1), plus `unknown` as the
explicit escape hatch for a class this taxonomy has not yet named.
"""

from enum import StrEnum


class FailureClass(StrEnum):
    # Specification / planning
    TASK_MISINTERPRETATION = "task_misinterpretation"
    PLAN_OMISSION = "plan_omission"
    CONSTRAINT_VIOLATION = "constraint_violation"

    # Tool / environment
    WRONG_TOOL_SELECTED = "wrong_tool_selected"
    MALFORMED_TOOL_INPUT = "malformed_tool_input"
    TOOL_FAILURE_UNHANDLED = "tool_failure_unhandled"
    SILENT_EMPTY_RESULT_MISREAD = "silent_empty_result_misread"
    HALLUCINATED_TOOL_OR_PARAMETER = "hallucinated_tool_or_parameter"

    # Information handling
    RETRIEVAL_MISS = "retrieval_miss"
    CONTEXT_LOSS = "context_loss"
    INFORMATION_FABRICATION = "information_fabrication"

    # Control flow
    INFINITE_LOOP_OR_OSCILLATION = "infinite_loop_or_oscillation"
    PREMATURE_TERMINATION = "premature_termination"
    STEP_BUDGET_EXHAUSTED = "step_budget_exhausted"

    # Multi-agent
    HANDOFF_INFORMATION_LOSS = "handoff_information_loss"
    ROLE_VIOLATION = "role_violation"
    CONFLICTING_SUBRESULTS_UNRECONCILED = "conflicting_subresults_unreconciled"

    # Verification
    MISSING_VERIFICATION = "missing_verification"
    INCORRECT_VERIFICATION = "incorrect_verification"

    # Output
    OUTPUT_SCHEMA_VIOLATION = "output_schema_violation"

    # Escape hatch
    UNKNOWN = "unknown"


_DESCRIPTIONS: dict[FailureClass, str] = {
    FailureClass.TASK_MISINTERPRETATION: (
        "The agent pursued a goal other than the one the user actually asked "
        "for, from the first planning step onward."
    ),
    FailureClass.PLAN_OMISSION: (
        "The agent's plan skipped a step that later turned out to be "
        "necessary for a correct outcome."
    ),
    FailureClass.CONSTRAINT_VIOLATION: (
        "The agent violated an explicit constraint stated in the task or "
        "system prompt, such as a budget, scope, or policy limit."
    ),
    FailureClass.WRONG_TOOL_SELECTED: (
        "The agent called a tool that could not accomplish the current "
        "subgoal when a more suitable tool was available."
    ),
    FailureClass.MALFORMED_TOOL_INPUT: (
        "The agent called a real tool with arguments that were syntactically "
        "or semantically invalid for that tool."
    ),
    FailureClass.TOOL_FAILURE_UNHANDLED: (
        "A tool call returned an explicit error and the agent proceeded "
        "without recovering from or acknowledging it."
    ),
    FailureClass.SILENT_EMPTY_RESULT_MISREAD: (
        "A tool returned successfully but with no usable content, and the "
        "agent treated the empty result as a positive finding."
    ),
    FailureClass.HALLUCINATED_TOOL_OR_PARAMETER: (
        "The agent invoked a tool that does not exist, or fabricated a "
        "parameter value with no provenance in the task or prior results."
    ),
    FailureClass.RETRIEVAL_MISS: (
        "A retrieval step returned documents that did not contain the "
        "information the task actually required."
    ),
    FailureClass.CONTEXT_LOSS: (
        "Information the agent needed was present earlier in the run but "
        "was dropped or truncated from context before it was used."
    ),
    FailureClass.INFORMATION_FABRICATION: (
        "The agent asserted a fact, number, or identifier that appears "
        "nowhere in the task, tool results, or prior context."
    ),
    FailureClass.INFINITE_LOOP_OR_OSCILLATION: (
        "The agent repeated the same action or an alternating pair of "
        "actions without making forward progress."
    ),
    FailureClass.PREMATURE_TERMINATION: (
        "The agent stopped and returned a result before the task's actual "
        "requirements were satisfied."
    ),
    FailureClass.STEP_BUDGET_EXHAUSTED: (
        "The run consumed its available steps or turns without reaching a "
        "conclusion, forcing a truncated or absent result."
    ),
    FailureClass.HANDOFF_INFORMATION_LOSS: (
        "A delegation between agents dropped information the receiving "
        "agent needed to complete its part of the task."
    ),
    FailureClass.ROLE_VIOLATION: (
        "An agent acted outside the role or authority it was assigned in a "
        "multi-agent run."
    ),
    FailureClass.CONFLICTING_SUBRESULTS_UNRECONCILED: (
        "Two sub-agents or steps produced contradictory results and the run "
        "proceeded without reconciling the contradiction."
    ),
    FailureClass.MISSING_VERIFICATION: (
        "The run produced a result that needed verification against a "
        "source of truth and no verification step occurred."
    ),
    FailureClass.INCORRECT_VERIFICATION: (
        "A verification step ran but reached the wrong conclusion, passing "
        "a result that was actually incorrect."
    ),
    FailureClass.OUTPUT_SCHEMA_VIOLATION: (
        "The final or an intermediate structured output did not conform to "
        "the schema the task required."
    ),
    FailureClass.UNKNOWN: (
        "None of the other classes describe this failure, or there is not "
        "enough evidence in the trace to assign a more specific class."
    ),
}


def describe_all() -> str:
    """Rendered into the L3 prompt. Built by iterating the enum, rather than
    hand-written as prose, so a class added to `FailureClass` without a
    matching `_DESCRIPTIONS` entry raises `KeyError` here instead of
    silently reaching the model as an unexplained option."""
    return "\n".join(f"- {m.value}: {_DESCRIPTIONS[m]}" for m in FailureClass)
