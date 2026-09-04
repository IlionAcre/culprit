"""reasoning_turn_defect: detects defects in agent reasoning turns.

Who&When places 98.9% of its annotated ground truth on agent reasoning turns
rather than tool calls. This detector evaluates agent reasoning steps
(SpanKind.AGENT) for two deterministic failure patterns:
1. Unverified assumptions or simulated data shortcuts (e.g. "simulate the",
   "let's assume", "placeholder values") instead of real retrieval or execution.
2. Malformed or unparseable code blocks embedded in reasoning turns where
   syntax fails ast.parse.
"""

import ast
import re

from culprit.detectors.base import DetectorContext
from culprit.schemas import AgentPayload, SpanKind
from culprit.signals import Evidence, Signal
from culprit.taxonomy import FailureClass

_ASSUMPTION_RE = re.compile(
    r"\b(?:we can simulate|simulate the|simulated dataset|simulated data|let'?s assume|assume that|assuming that|make an assumption|an assumption|dummy data|mock data|placeholder values?|without verifying)\b",
    re.IGNORECASE,
)

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

_SEVERITY = 0.65


def _fails_ast(code: str) -> bool:
    try:
        ast.parse(code)
        return False
    except SyntaxError:
        return True


def reasoning_turn_defect(ctx: DetectorContext) -> list[Signal]:
    """Flag agent reasoning turns containing unverified assumptions or broken code blocks."""
    signals: list[Signal] = []

    for step in ctx.steps:
        if step.kind != SpanKind.AGENT:
            continue
        span = ctx.spans_by_id.get(step.span_id)
        if span is None or not isinstance(span.payload, AgentPayload):
            continue

        text = span.payload.output_text or ""
        if not text:
            continue

        assumption_match = _ASSUMPTION_RE.search(text)
        if assumption_match:
            start = max(0, assumption_match.start() - 30)
            end = min(len(text), assumption_match.end() + 30)
            signals.append(
                Signal(
                    detector="reasoning_turn_defect",
                    step_index=step.step_index,
                    span_id=step.span_id,
                    severity=_SEVERITY,
                    category=FailureClass.INFORMATION_FABRICATION.value,
                    message=f"Agent reasoning introduces unverified assumption or simulation: {assumption_match.group(0)!r}",
                    evidence=[
                        Evidence(
                            span_id=step.span_id,
                            step_index=step.step_index,
                            field="payload.output_text",
                            excerpt=text[start:end],
                        )
                    ],
                )
            )
            continue

        bad_code = next((b for b in _CODE_BLOCK_RE.findall(text) if _fails_ast(b)), None)
        if bad_code is not None:
            signals.append(
                Signal(
                    detector="reasoning_turn_defect",
                    step_index=step.step_index,
                    span_id=step.span_id,
                    severity=_SEVERITY,
                    category=FailureClass.MALFORMED_TOOL_INPUT.value,
                    message="Agent reasoning contains malformed or unparseable code block",
                    evidence=[
                        Evidence(
                            span_id=step.span_id,
                            step_index=step.step_index,
                            field="payload.output_text",
                            excerpt=bad_code[:120].strip(),
                        )
                    ],
                )
            )

    return signals
