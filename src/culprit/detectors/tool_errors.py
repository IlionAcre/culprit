"""tool_error, empty_tool_result, error_swallowed.

The latter two are two of the four detectors the product exists to catch
(CLAUDE.md's "L1 detectors" section): the trace looks entirely healthy right
where the damage happens (status OK, no exception raised), and the
consequence only becomes visible looking at what happens next.
"""

from culprit.detectors.base import DetectorContext, truncate
from culprit.schemas import SpanKind, SpanStatus, ToolPayload
from culprit.signals import Evidence, Signal
from culprit.taxonomy import FailureClass

_EMPTY_RESULTS = {"", "[]", "{}", "null", "no results found"}
_FAILURE_WORDS = (
    "error", "fail", "unable", "could not", "couldn't", "sorry", "apolog", "not found", "denied",
)
_SUCCESS_WORDS = ("success", "done", "complete", "resolved", "issued", "processed", "confirmed")


def _is_empty_result(text: str) -> bool:
    return text.strip().lower() in _EMPTY_RESULTS


def tool_error(ctx: DetectorContext) -> list[Signal]:
    """A TOOL step whose status or payload says it failed. The loud,
    easy-to-notice case; `empty_tool_result` and `error_swallowed` exist
    because most failures do not look like this one."""
    signals = []
    for step in ctx.steps:
        span = ctx.spans_by_id.get(step.span_id)
        if step.kind != SpanKind.TOOL or span is None or not isinstance(span.payload, ToolPayload):
            continue
        p = span.payload
        if span.status != SpanStatus.ERROR and not p.is_error:
            continue
        signals.append(Signal(
            detector="tool_error", step_index=step.step_index, span_id=step.span_id,
            severity=0.7, category=FailureClass.TOOL_FAILURE_UNHANDLED.value,
            message=f"{p.tool_name} returned an error",
            evidence=[Evidence(
                span_id=step.span_id, step_index=step.step_index, field="payload.error_message",
                excerpt=truncate(p.error_message or p.result_text or "<no error message>"),
            )],
        ))
    return signals


def empty_tool_result(ctx: DetectorContext) -> list[Signal]:
    """Status OK, no exception, but the result is `{"", "[]", "{}", "null",
    "No results found"}`. Evidence includes the next LLM step's own text so
    a reader sees what the agent did with nothing, per the plan's catalogue
    entry for this detector."""
    signals = []
    for step in ctx.steps:
        span = ctx.spans_by_id.get(step.span_id)
        if step.kind != SpanKind.TOOL or span is None or not isinstance(span.payload, ToolPayload):
            continue
        p = span.payload
        if span.status != SpanStatus.OK or p.is_error or not _is_empty_result(p.result_text):
            continue
        evidence = [Evidence(
            span_id=step.span_id, step_index=step.step_index, field="payload.result_text",
            excerpt=truncate(p.result_text or "<empty>"),
        )]
        next_llm = next((s for s in ctx.steps if s.step_index > step.step_index and s.kind == SpanKind.LLM), None)
        if next_llm is not None:
            evidence.append(Evidence(
                span_id=next_llm.span_id, step_index=next_llm.step_index, field="payload.response_messages",
                excerpt=truncate(ctx.step_text[next_llm.step_index], 200),
            ))
        signals.append(Signal(
            detector="empty_tool_result", step_index=step.step_index, span_id=step.span_id,
            severity=0.6, category=FailureClass.SILENT_EMPTY_RESULT_MISREAD.value,
            message=f"{p.tool_name} returned an empty result with status OK",
            evidence=evidence,
        ))
    return signals


def _asserts_success_without_acknowledging_failure(text: str) -> bool:
    lowered = text.lower()
    if any(w in lowered for w in _FAILURE_WORDS):
        return False
    return any(w in lowered for w in _SUCCESS_WORDS)


def _response_text(ctx: DetectorContext, step_index: int) -> str:
    """The LLM step's own *response* text only, not its request/instruction
    text. A tool-calling LLM step's request text ("Check whether this
    refund was already issued") can coincidentally contain a success-shaped
    word ("issued") despite asserting nothing; only what the model actually
    said back is evidence of the model claiming success."""
    span = ctx.spans_by_id.get(ctx.steps[step_index].span_id)
    payload = span.payload if span is not None else None
    messages = getattr(payload, "response_messages", None) or []
    return " ".join(m.content or "" for m in messages)


def error_swallowed(ctx: DetectorContext) -> list[Signal]:
    """A tool error or empty result followed within 2 steps by an LLM step
    whose own response asserts a positive result with no failure language:
    the exact silent-failure pattern the product exists to catch. Because
    the base condition (tool error or empty result) never occurs in a clean
    run, this detector cannot misfire on clean data regardless of how the
    follow-up response is worded."""
    signals = []
    for step in ctx.steps:
        span = ctx.spans_by_id.get(step.span_id)
        if step.kind != SpanKind.TOOL or span is None or not isinstance(span.payload, ToolPayload):
            continue
        p = span.payload
        troubled = (span.status == SpanStatus.ERROR or p.is_error) or (
            span.status == SpanStatus.OK and not p.is_error and _is_empty_result(p.result_text)
        )
        if not troubled:
            continue
        for j in (step.step_index + 1, step.step_index + 2):
            if j >= len(ctx.steps) or ctx.steps[j].kind != SpanKind.LLM:
                continue
            text = _response_text(ctx, j)
            if _asserts_success_without_acknowledging_failure(text):
                signals.append(Signal(
                    detector="error_swallowed", step_index=step.step_index, span_id=step.span_id,
                    severity=0.85, category=FailureClass.TOOL_FAILURE_UNHANDLED.value,
                    message=f"{p.tool_name} failed or returned nothing, and step {j} reports success anyway",
                    evidence=[
                        Evidence(span_id=step.span_id, step_index=step.step_index, field="payload.result_text",
                                 excerpt=truncate(p.result_text or "<empty>")),
                        Evidence(span_id=ctx.steps[j].span_id, step_index=j, field="payload.response_messages",
                                 excerpt=truncate(text, 200)),
                    ],
                ))
                break
    return signals
