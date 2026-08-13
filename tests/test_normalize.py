from pathlib import Path

from culprit.normalize import normalize_span
from culprit.otlp import decode
from culprit.schemas import LlmPayload, SpanKind, SpanStatus

_FIXTURE = Path(__file__).parent / "fixtures" / "otlp" / "otel_genai_sample.json"


def _raw_spans() -> list[dict]:
    return decode(_FIXTURE.read_bytes(), "application/json")


def test_normalize_span_dispatches_a_recognized_vocabulary_to_its_module():
    raw = next(s for s in _raw_spans() if s["attributes"].get("gen_ai.operation.name") == "chat")

    span = normalize_span(raw, "t1")

    assert span.trace_id == "t1"
    assert span.kind == SpanKind.LLM
    assert span.vocabulary == "otel_genai"
    assert isinstance(span.payload, LlmPayload)
    assert span.normalize_error is None


def test_normalize_span_with_an_unrecognized_vocabulary_sets_normalize_error_and_keeps_attributes():
    """The core guardrail (CLAUDE.md): raw attributes are never discarded,
    even for a span whose vocabulary is unrecognized."""
    raw = {
        "span_id": "s1", "parent_span_id": None, "name": "mystery",
        "start_ns": 10, "end_ns": 20, "status_code": "STATUS_CODE_OK", "status_message": None,
        "attributes": {"vendor.weird.key": 42},
    }

    span = normalize_span(raw, "t1")

    assert span.payload is None
    assert span.kind == SpanKind.UNKNOWN
    assert span.normalize_error == "no vocabulary matched"
    assert span.attributes == {"vendor.weird.key": 42}
    assert span.span_id == "s1"
    assert span.start_ns == 10
    assert span.end_ns == 20


def test_normalize_span_isolates_a_vocab_module_that_raises_without_propagating():
    """A recognized vocabulary whose payload data is malformed (e.g. bad
    JSON in an attribute) must degrade to normalize_error rather than take
    down the whole ingestion batch - per-item error isolation."""
    raw = {
        "span_id": "s1", "parent_span_id": None, "name": "execute_tool",
        "start_ns": 10, "end_ns": 20, "status_code": "STATUS_CODE_OK", "status_message": None,
        "attributes": {
            "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "x",
            "gen_ai.tool.call.arguments": "{not valid json",
        },
    }

    span = normalize_span(raw, "t1")

    assert span.payload is None
    assert span.kind == SpanKind.UNKNOWN
    assert span.normalize_error is not None
    assert "not valid JSON" in span.normalize_error
    # The raw attributes survive intact even though normalization failed.
    assert span.attributes["gen_ai.tool.call.arguments"] == "{not valid json"


def test_normalize_span_fallback_maps_error_status_code():
    raw = {
        "span_id": "s1", "parent_span_id": None, "name": "mystery",
        "start_ns": 0, "end_ns": 1, "status_code": "STATUS_CODE_ERROR", "status_message": "boom",
        "attributes": {},
    }

    span = normalize_span(raw, "t1")

    assert span.status == SpanStatus.ERROR
    assert span.status_message == "boom"


def test_normalize_span_preserves_the_given_trace_id_regardless_of_source():
    raw = next(s for s in _raw_spans() if s["attributes"].get("gen_ai.operation.name") == "invoke_agent")

    span = normalize_span(raw, "overridden-trace-id")

    assert span.trace_id == "overridden-trace-id"
