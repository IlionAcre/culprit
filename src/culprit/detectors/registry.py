"""Detector name -> callable registry. Every `culprit.synth_inject`
injection kind has a detector of the same name, so the definition of done
("every detector fires on its matching injection") stays checkable.
`instruction_noncompliance` is the one entry with no injection: it carries
its false-positive check in `tests/test_detectors_compliance.py` instead.
Owned solely by WS-C per CLAUDE.md's "one owner per plugin registry" rule.
"""

from culprit.detectors.base import Detector
from culprit.detectors.compliance import instruction_noncompliance
from culprit.detectors.context import context_overflow, silent_history_truncation, step_budget_exhausted
from culprit.detectors.flow import duplicate_delegation, missing_verification, premature_termination
from culprit.detectors.loops import oscillation, repeated_identical_action, retry_storm
from culprit.detectors.provenance import parameter_drift
from culprit.detectors.reasoning import reasoning_turn_defect
from culprit.detectors.retrieval import goal_token_drift, low_score_retrieval, unused_retrieval
from culprit.detectors.schema import hallucinated_tool, output_schema_violation, tool_arg_malformed
from culprit.detectors.timing import stall_timeout
from culprit.detectors.tool_errors import empty_tool_result, error_swallowed, tool_error

DETECTORS: dict[str, Detector] = {
    "tool_error": tool_error,
    "empty_tool_result": empty_tool_result,
    "error_swallowed": error_swallowed,
    "instruction_noncompliance": instruction_noncompliance,
    "reasoning_turn_defect": reasoning_turn_defect,
    "oscillation": oscillation,
    "repeated_identical_action": repeated_identical_action,
    "retry_storm": retry_storm,
    "context_overflow": context_overflow,
    "silent_history_truncation": silent_history_truncation,
    "step_budget_exhausted": step_budget_exhausted,
    "output_schema_violation": output_schema_violation,
    "tool_arg_malformed": tool_arg_malformed,
    "hallucinated_tool": hallucinated_tool,
    "premature_termination": premature_termination,
    "missing_verification": missing_verification,
    "duplicate_delegation": duplicate_delegation,
    "unused_retrieval": unused_retrieval,
    "low_score_retrieval": low_score_retrieval,
    "goal_token_drift": goal_token_drift,
    "parameter_drift": parameter_drift,
    "stall_timeout": stall_timeout,
}
