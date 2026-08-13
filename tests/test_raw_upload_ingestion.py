"""End-to-end check that the third fixture (direct JSON upload, not OTLP)
ingests through normalize_span + linearize the same way the OTLP fixtures do.
CLAUDE.md: this fixture "represents a third, distinct ingestion path" and
"does not need to match the other two fixtures' step sequence" - it encodes a
different scenario on purpose (a refund agent silently acting on an empty
tool result), which is exactly what this test asserts survives normalization.
"""

import json
from pathlib import Path

from culprit.linearize import linearize
from culprit.normalize import normalize_span
from culprit.schemas import SpanKind

_FIXTURE = Path(__file__).parent / "fixtures" / "otlp" / "raw_upload_sample.json"


def _load():
    return json.loads(_FIXTURE.read_bytes())


def test_raw_upload_spans_all_normalize_without_error():
    data = _load()

    spans = [normalize_span(raw, data["trace_id"]) for raw in data["spans"]]

    assert len(spans) == 4
    assert all(s.normalize_error is None for s in spans)
    assert all(s.vocabulary == "raw_upload" for s in spans)


def test_raw_upload_linearizes_to_four_semantic_steps_in_order():
    data = _load()
    spans = [normalize_span(raw, data["trace_id"]) for raw in data["spans"]]

    steps = linearize(spans)

    assert [s.kind for s in steps] == [SpanKind.AGENT, SpanKind.LLM, SpanKind.TOOL, SpanKind.LLM]
    assert [s.step_index for s in steps] == [0, 1, 2, 3]


def test_raw_upload_captures_the_silent_failure_pattern_the_fixture_exists_for():
    """The tool step's empty result and the following LLM step's confident,
    unqualified success claim both have to survive normalization intact -
    this is the exact evidence L1's empty_tool_result / error_swallowed
    detectors will need downstream."""
    data = _load()
    spans = [normalize_span(raw, data["trace_id"]) for raw in data["spans"]]
    steps = linearize(spans)

    tool_step = next(s for s in steps if s.kind == SpanKind.TOOL)
    final_step = steps[-1]

    tool_span = next(s for s in spans if s.span_id == tool_step.span_id)
    final_span = next(s for s in spans if s.span_id == final_step.span_id)

    assert tool_span.payload.result_text == ""
    assert tool_span.payload.is_error is False
    assert "refund" in final_span.payload.response_messages[0].content.lower()
    assert "processed" in final_span.payload.response_messages[0].content.lower()
