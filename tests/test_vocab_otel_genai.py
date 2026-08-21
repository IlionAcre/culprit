from pathlib import Path

import pytest

from culprit.otlp import decode
from culprit.schemas import AgentPayload, LlmPayload, RetrievalPayload, SpanKind, SpanStatus, ToolPayload
from culprit.vocab import otel_genai

_FIXTURE = Path(__file__).parent / "fixtures" / "otlp" / "otel_genai_sample.json"


def _raw_spans() -> list[dict]:
    return decode(_FIXTURE.read_bytes(), "application/json")


def _by_operation(op: str) -> dict:
    return next(s for s in _raw_spans() if s["attributes"].get("gen_ai.operation.name") == op)


def test_matches_true_for_a_span_carrying_gen_ai_operation_name():
    assert otel_genai.matches(_by_operation("invoke_agent")) is True


def test_matches_false_for_a_span_without_gen_ai_attributes():
    assert otel_genai.matches({"attributes": {"openinference.span.kind": "LLM"}}) is False


def test_to_span_builds_agent_payload_from_invoke_agent():
    span = otel_genai.to_span(_by_operation("invoke_agent"), "t1")

    assert span.kind == SpanKind.AGENT
    assert span.vocabulary == "otel_genai"
    assert span.status == SpanStatus.OK
    assert span.normalize_error is None
    assert isinstance(span.payload, AgentPayload)
    assert span.payload.agent_name == "order_support_agent"


def test_to_span_builds_llm_payload_with_tool_call_from_the_decision_message():
    raw = _by_operation("chat")  # the first "chat" span calls search_orders
    span = otel_genai.to_span(raw, "t1")

    assert isinstance(span.payload, LlmPayload)
    assert span.payload.provider == "openai"
    assert span.payload.model == "gpt-4o-mini-2024-07-18"
    assert span.payload.prompt_tokens == 412
    assert span.payload.completion_tokens == 28
    assert span.payload.total_tokens == 440
    assert span.payload.finish_reason == "tool_calls"
    assert len(span.payload.tool_calls) == 1
    assert span.payload.tool_calls[0].tool_name == "search_orders"
    assert span.payload.tool_calls[0].arguments == {"order_id": "48291"}
    assert span.payload.request_messages[0].role == "system"


def test_to_span_builds_tool_payload_with_parsed_arguments_and_result():
    span = otel_genai.to_span(_by_operation("execute_tool"), "t1")

    assert isinstance(span.payload, ToolPayload)
    assert span.payload.tool_name == "search_orders"
    assert span.payload.call_id == "call_7f3e9a2b"
    assert span.payload.arguments == {"order_id": "48291"}
    assert span.payload.is_error is False
    assert "delivered" in span.payload.result_text
    assert span.payload.result_len == len(span.payload.result_text)


def test_to_span_builds_retrieval_payload_with_two_documents():
    span = otel_genai.to_span(_by_operation("retrieve_documents"), "t1")

    assert isinstance(span.payload, RetrievalPayload)
    assert span.payload.top_k == 2
    assert len(span.payload.documents) == 2
    assert span.payload.documents[0].doc_id == "policy-042"
    assert span.payload.documents[0].score == pytest.approx(0.91)


def test_to_span_retains_every_raw_attribute_regardless_of_kind():
    raw = _by_operation("execute_tool")
    span = otel_genai.to_span(raw, "t1")

    assert span.attributes == raw["attributes"]


def test_to_span_maps_an_unrecognized_operation_name_to_unknown_kind_without_raising():
    raw = {
        "span_id": "s1", "parent_span_id": None, "name": "mystery", "start_ns": 0, "end_ns": 1,
        "status_code": "STATUS_CODE_OK", "status_message": None,
        "attributes": {"gen_ai.operation.name": "some_future_operation"},
    }

    span = otel_genai.to_span(raw, "t1")

    assert span.kind == SpanKind.UNKNOWN
    assert span.payload is None
    assert span.normalize_error is None


def test_to_span_builds_tool_payload_when_tool_call_arguments_are_not_valid_json():
    raw = {
        "span_id": "s1", "parent_span_id": None, "name": "execute_tool", "start_ns": 0, "end_ns": 1,
        "status_code": "STATUS_CODE_OK", "status_message": None,
        "attributes": {
            "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "x",
            "gen_ai.tool.call.arguments": "{not json",
        },
    }

    span = otel_genai.to_span(raw, "t1")

    assert span.kind == SpanKind.TOOL
    assert isinstance(span.payload, ToolPayload)
    assert span.payload.arguments == {}
    assert span.payload.arguments_json == "{not json"
    assert span.payload.tool_name == "x"
    assert span.normalize_error is None


def test_vocab_dataclass_instance_matches_and_normalizes_the_same_as_the_module_functions():
    raw = _by_operation("invoke_agent")

    assert otel_genai.VOCAB.matches(raw) == otel_genai.matches(raw)
    assert otel_genai.VOCAB.to_span(raw, "t1") == otel_genai.to_span(raw, "t1")
