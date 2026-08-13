import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from culprit.schemas import AgentPayload, LlmPayload, SpanKind, SpanStatus, ToolPayload
from culprit.vocab import raw_upload

_FIXTURE = Path(__file__).parent / "fixtures" / "otlp" / "raw_upload_sample.json"


def _raw_spans() -> list[dict]:
    return json.loads(_FIXTURE.read_bytes())["spans"]


def _by_span_id(span_id: str) -> dict:
    return next(s for s in _raw_spans() if s["span_id"] == span_id)


def test_matches_true_for_the_direct_upload_shape():
    assert raw_upload.matches(_by_span_id("s-1")) is True


def test_matches_false_for_an_otlp_derived_raw_dict():
    """An OTLP-decoded span dict has start_ns/end_ns, not start_time/
    end_time, and its "kind" (when present at all) is the OTel transport
    kind (SPAN_KIND_INTERNAL, ...), not one of culprit's own SpanKind
    values - matches() must not accidentally claim it."""
    otlp_like = {
        "span_id": "s1", "start_ns": 0, "end_ns": 1,
        "attributes": {"gen_ai.operation.name": "chat"},
    }

    assert raw_upload.matches(otlp_like) is False


def test_to_span_builds_agent_payload():
    span = raw_upload.to_span(_by_span_id("s-1"), "raw-upload-7f2c19")

    assert span.kind == SpanKind.AGENT
    assert span.vocabulary == "raw_upload"
    assert span.status == SpanStatus.UNSET
    assert isinstance(span.payload, AgentPayload)
    assert span.payload.agent_name == "refund_agent"


def test_to_span_parses_iso_timestamps_to_nanoseconds_preserving_order():
    start_span = raw_upload.to_span(_by_span_id("s-1"), "t1")
    child_span = raw_upload.to_span(_by_span_id("s-2"), "t1")

    assert start_span.start_ns < child_span.start_ns
    assert child_span.end_ns > child_span.start_ns
    # 2026-08-10T14:31:58.102Z, computed independently via stdlib datetime
    # rather than by reusing raw_upload's own conversion. Uses integer
    # days/seconds/microseconds, not total_seconds() * 1e9, which loses
    # precision at nanosecond scale through float rounding.
    expected = datetime(2026, 8, 10, 14, 31, 58, 102_000, tzinfo=UTC)
    delta = expected - datetime(1970, 1, 1, tzinfo=UTC)
    expected_ns = delta.days * 86_400_000_000_000 + delta.seconds * 1_000_000_000 + delta.microseconds * 1000
    assert start_span.start_ns == expected_ns


def test_to_span_builds_llm_payload_with_tool_call_arguments_already_a_dict():
    """Unlike the OTLP vocabularies, this format's tool_calls[].arguments is
    already a parsed dict, not a JSON string - to_span must not try to
    json.loads() it again."""
    span = raw_upload.to_span(_by_span_id("s-2"), "t1")

    assert isinstance(span.payload, LlmPayload)
    assert span.payload.model == "claude-3-5-haiku"
    assert span.payload.prompt_tokens == 356
    assert span.payload.tool_calls[0].tool_name == "get_order_history"
    assert span.payload.tool_calls[0].arguments == {"order_id": "55210"}


def test_to_span_builds_tool_payload_capturing_the_empty_result():
    """This is the exact silent-failure scenario the fixture exists to
    exercise (CLAUDE.md): the tool returns an empty result but is not
    marked is_error, so downstream L1's empty_tool_result detector is what
    has to catch it, not normalization."""
    span = raw_upload.to_span(_by_span_id("s-3"), "t1")

    assert isinstance(span.payload, ToolPayload)
    assert span.payload.tool_name == "get_order_history"
    assert span.payload.result_text == ""
    assert span.payload.result_len == 0
    assert span.payload.is_error is False


def test_to_span_retains_every_raw_attribute():
    raw = _by_span_id("s-2")
    span = raw_upload.to_span(raw, "t1")

    assert span.attributes == raw["attributes"]


def test_to_span_treats_a_missing_status_as_unset():
    raw = dict(_by_span_id("s-1"))
    del raw["status"]

    span = raw_upload.to_span(raw, "t1")

    assert span.status == SpanStatus.UNSET


def test_to_span_raises_on_an_unparseable_status_value():
    raw = dict(_by_span_id("s-1"))
    raw["status"] = "not-a-real-status"

    with pytest.raises(ValueError):
        raw_upload.to_span(raw, "t1")
