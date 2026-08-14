"""output_schema_violation, tool_arg_malformed, hallucinated_tool: three ways
a step's own declared contract (a response schema, a JSON arguments string,
a real tool name) diverges from what actually happened.
"""

import json

from culprit.detectors.base import DetectorContext, truncate
from culprit.schemas import LlmPayload, SpanKind, ToolPayload
from culprit.signals import Evidence, Signal
from culprit.taxonomy import FailureClass


def output_schema_violation(ctx: DetectorContext) -> list[Signal]:
    """An LLM step that declared a `response_format_schema` whose response
    is not valid JSON satisfying that schema's required keys. Clean synth
    steps never set `response_format_schema`, so this can only fire when a
    step actually opts into structured output and then violates it."""
    signals = []
    for step in ctx.steps:
        if step.kind != SpanKind.LLM:
            continue
        span = ctx.spans_by_id.get(step.span_id)
        if span is None or not isinstance(span.payload, LlmPayload):
            continue
        schema = span.payload.response_format_schema
        if not schema:
            continue
        text = next((m.content for m in span.payload.response_messages if m.content), None)
        required = schema.get("required", [])
        violated = True
        if text:
            try:
                parsed = json.loads(text)
                violated = not (isinstance(parsed, dict) and all(k in parsed for k in required))
            except (json.JSONDecodeError, TypeError):
                violated = True
        if violated:
            signals.append(Signal(
                detector="output_schema_violation", step_index=step.step_index, span_id=step.span_id,
                severity=0.6, category=FailureClass.OUTPUT_SCHEMA_VIOLATION.value,
                message="Response does not satisfy its declared response_format_schema",
                evidence=[Evidence(
                    span_id=step.span_id, step_index=step.step_index, field="payload.response_messages",
                    excerpt=truncate(text or "<no content>"),
                )],
            ))
    return signals


def tool_arg_malformed(ctx: DetectorContext) -> list[Signal]:
    """A TOOL step whose raw `arguments_json` fails to parse. Clean synth
    args are always produced by `json.dumps`, so this can only fire on
    genuinely malformed input."""
    signals = []
    for step in ctx.steps:
        if step.kind != SpanKind.TOOL:
            continue
        span = ctx.spans_by_id.get(step.span_id)
        if span is None or not isinstance(span.payload, ToolPayload):
            continue
        raw = span.payload.arguments_json
        if not raw:
            continue
        try:
            json.loads(raw)
        except json.JSONDecodeError:
            signals.append(Signal(
                detector="tool_arg_malformed", step_index=step.step_index, span_id=step.span_id,
                severity=0.65, category=FailureClass.MALFORMED_TOOL_INPUT.value,
                message=f"{span.payload.tool_name} arguments_json failed to parse as JSON",
                evidence=[Evidence(
                    span_id=step.span_id, step_index=step.step_index, field="payload.arguments_json",
                    excerpt=truncate(raw),
                )],
            ))
    return signals


def hallucinated_tool(ctx: DetectorContext) -> list[Signal]:
    """An LLM step calls a tool name that never appears as an actually
    executed TOOL step anywhere in this trace. `tool_universe` is built from
    every real TOOL-kind step across the whole trace (see `base.py`), so any
    tool that genuinely gets executed is safe even if its execution step
    comes later in the trace than the call requesting it."""
    signals = []
    for step in ctx.steps:
        if step.kind != SpanKind.LLM:
            continue
        span = ctx.spans_by_id.get(step.span_id)
        if span is None or not isinstance(span.payload, LlmPayload):
            continue
        for call in span.payload.tool_calls:
            if call.tool_name not in ctx.tool_universe:
                signals.append(Signal(
                    detector="hallucinated_tool", step_index=step.step_index, span_id=step.span_id,
                    severity=0.8, category=FailureClass.HALLUCINATED_TOOL_OR_PARAMETER.value,
                    message=f"Calls {call.tool_name!r}, which never executes as a real tool step in this trace",
                    evidence=[Evidence(
                        span_id=step.span_id, step_index=step.step_index, field="payload.tool_calls",
                        excerpt=call.tool_name,
                    )],
                ))
    return signals
