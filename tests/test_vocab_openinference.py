from pathlib import Path

import pytest

from culprit.otlp import decode
from culprit.schemas import AgentPayload, LlmPayload, RetrievalPayload, SpanKind, SpanStatus, ToolPayload
from culprit.vocab import openinference

_FIXTURE = Path(__file__).parent / "fixtures" / "otlp" / "openinference_sample.json"


def _raw_spans() -> list[dict]:
    return decode(_FIXTURE.read_bytes(), "application/json")


def _by_kind(kind: str, *, nth: int = 0) -> dict:
    matches = [s for s in _raw_spans() if s["attributes"].get("openinference.span.kind") == kind]
    return matches[nth]


def test_matches_true_for_a_span_carrying_openinference_span_kind():
    assert openinference.matches(_by_kind("AGENT")) is True


def test_matches_false_for_a_span_without_openinference_attributes():
    assert openinference.matches({"attributes": {"gen_ai.operation.name": "chat"}}) is False


def test_to_span_builds_agent_payload_from_input_and_output_value():
    span = openinference.to_span(_by_kind("AGENT"), "t1")

    assert span.kind == SpanKind.AGENT
    assert span.vocabulary == "openinference"
    assert span.status == SpanStatus.OK
    assert isinstance(span.payload, AgentPayload)
    assert span.payload.agent_name == "order_support_agent"
    assert "48291" in span.payload.input_text


def test_to_span_builds_llm_payload_reconstructing_flattened_indexed_messages():
    span = openinference.to_span(_by_kind("LLM", nth=0), "t1")

    assert isinstance(span.payload, LlmPayload)
    assert span.payload.provider == "openai"
    assert span.payload.model == "gpt-4o-mini"
    assert span.payload.prompt_tokens == 412
    assert span.payload.completion_tokens == 28
    assert span.payload.finish_reason == "tool_calls"
    assert len(span.payload.request_messages) == 2
    assert span.payload.request_messages[0].role == "system"
    assert span.payload.request_messages[1].role == "user"
    assert span.payload.temperature == pytest.approx(0.2)


def test_to_span_builds_llm_payload_with_flattened_tool_calls():
    span = openinference.to_span(_by_kind("LLM", nth=0), "t1")

    assert len(span.payload.tool_calls) == 1
    assert span.payload.tool_calls[0].call_id == "call_7f3e9a2b"
    assert span.payload.tool_calls[0].tool_name == "search_orders"
    assert span.payload.tool_calls[0].arguments == {"order_id": "48291"}


def test_to_span_builds_tool_payload():
    span = openinference.to_span(_by_kind("TOOL"), "t1")

    assert isinstance(span.payload, ToolPayload)
    assert span.payload.tool_name == "search_orders"
    assert span.payload.call_id == "call_7f3e9a2b"
    assert span.payload.arguments == {"order_id": "48291"}
    assert span.payload.is_error is False


def test_to_span_builds_retrieval_payload_reconstructing_flattened_indexed_documents():
    span = openinference.to_span(_by_kind("RETRIEVER"), "t1")

    assert isinstance(span.payload, RetrievalPayload)
    assert span.payload.top_k == 2
    assert span.payload.documents[0].doc_id == "policy-042"
    assert span.payload.documents[0].score == pytest.approx(0.91)
    assert span.payload.documents[1].doc_id == "policy-017"


def test_to_span_retains_every_raw_attribute():
    raw = _by_kind("TOOL")
    span = openinference.to_span(raw, "t1")

    assert span.attributes == raw["attributes"]


def test_to_span_maps_an_unrecognized_span_kind_to_unknown_without_raising():
    raw = {
        "span_id": "s1", "parent_span_id": None, "name": "mystery", "start_ns": 0, "end_ns": 1,
        "status_code": "STATUS_CODE_OK", "status_message": None,
        "attributes": {"openinference.span.kind": "SOME_FUTURE_KIND"},
    }

    span = openinference.to_span(raw, "t1")

    assert span.kind == SpanKind.UNKNOWN
    assert span.payload is None
    assert span.normalize_error is None


def test_to_span_builds_tool_payload_when_input_value_is_not_valid_json():
    raw = {
        "span_id": "s1", "parent_span_id": None, "name": "tool", "start_ns": 0, "end_ns": 1,
        "status_code": "STATUS_CODE_OK", "status_message": None,
        "attributes": {"openinference.span.kind": "TOOL", "tool.name": "x", "input.value": "{not json"},
    }

    span = openinference.to_span(raw, "t1")

    assert span.kind == SpanKind.TOOL
    assert isinstance(span.payload, ToolPayload)
    assert span.payload.arguments == {}
    assert span.payload.arguments_json == "{not json"
    assert span.payload.tool_name == "x"
    assert span.normalize_error is None
