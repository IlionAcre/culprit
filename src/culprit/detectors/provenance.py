"""parameter_drift, one of the four detectors the product exists to catch
(CLAUDE.md's "L1 detectors" section). The trace looks entirely healthy at
the point of damage: a tool call with status OK, whose arguments happen to
contain a fabricated identifier that no earlier step or the task goal ever
mentioned.
"""

import re

from culprit.detectors.base import DetectorContext
from culprit.schemas import SpanKind, ToolPayload
from culprit.signals import Evidence, Signal
from culprit.taxonomy import FailureClass

# An "identifier-shaped" token: made only of id-safe characters, mixes
# letters and digits, and is long enough to plausibly name something
# specific (an order id, a path, a url) rather than being an ordinary word
# or a bare number.
_IDENTIFIER_RE = re.compile(r"^(?=[A-Za-z0-9_./:-]{5,}$)(?=.*[A-Za-z])(?=.*\d).+$")


def _identifier_tokens(value: object) -> list[str]:
    if isinstance(value, str):
        return [tok for tok in re.split(r"[\s,]+", value) if _IDENTIFIER_RE.match(tok)]
    if isinstance(value, dict):
        out: list[str] = []
        for v in value.values():
            out.extend(_identifier_tokens(v))
        return out
    if isinstance(value, list):
        out = []
        for v in value:
            out.extend(_identifier_tokens(v))
        return out
    return []


def parameter_drift(ctx: DetectorContext) -> list[Signal]:
    """Every identifier-shaped token in a TOOL step's arguments must appear
    verbatim in the task goal or in some earlier step's own text (result,
    arguments, message, retrieved content). A token with no such provenance
    was invented by the model rather than carried forward from real data."""
    signals = []
    for step in ctx.steps:
        if step.kind != SpanKind.TOOL:
            continue
        span = ctx.spans_by_id.get(step.span_id)
        if span is None or not isinstance(span.payload, ToolPayload):
            continue
        for token in _identifier_tokens(span.payload.arguments):
            if not ctx.had_provenance_before(token, step.step_index):
                signals.append(Signal(
                    detector="parameter_drift", step_index=step.step_index, span_id=step.span_id,
                    severity=0.7, category=FailureClass.HALLUCINATED_TOOL_OR_PARAMETER.value,
                    message=f"Argument {token!r} has no provenance in the goal or any prior step",
                    evidence=[Evidence(
                        span_id=step.span_id, step_index=step.step_index, field="payload.arguments",
                        excerpt=token,
                    )],
                ))
    return signals
