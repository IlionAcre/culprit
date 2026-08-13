from culprit.schemas import (
    AgentPayload,
    LlmPayload,
    Message,
    Outcome,
    RetrievalPayload,
    RetrievedDoc,
    Span,
    SpanKind,
    SpanStatus,
    Step,
    ToolCallRequest,
    ToolPayload,
    Trace,
)


def test_span_with_an_unrecognized_vocabulary_keeps_raw_attributes_and_records_the_error():
    """Normalization must be reversible: a span we cannot interpret today has
    to survive intact so a later vocabulary module can backfill it."""
    span = Span(
        trace_id="t1", span_id="s1", parent_span_id=None, name="mystery",
        kind=SpanKind.UNKNOWN, status=SpanStatus.UNSET, status_message=None,
        start_ns=0, end_ns=1, vocabulary="unknown",
        attributes={"vendor.weird.key": 42}, payload=None,
        normalize_error="no vocabulary matched",
    )

    assert span.attributes == {"vendor.weird.key": 42}
    assert span.payload is None
    assert span.normalize_error == "no vocabulary matched"


def test_tool_payload_records_result_length_separately_from_result_text():
    """result_len is stored rather than derived because empty_tool_result must
    still fire after attributes are pruned by the retention policy."""
    payload = ToolPayload(
        tool_name="search_orders", call_id="c1", arguments_json="{}",
        arguments={}, result_text="", result_len=0, is_error=False,
        error_message=None,
    )

    assert payload.result_len == 0
    assert payload.is_error is False


def test_trace_has_no_task_embedding_field():
    """task_embedding is a 384-float storage concern that lives only in
    Postgres; it must never ride along on every API response and log line."""
    assert "task_embedding" not in Trace.model_fields


def test_span_payload_is_a_plain_union_not_a_discriminated_one():
    """The discriminant lives on the sibling `kind` field, not inside any
    payload member, so payload cannot be declared with Field(discriminator=).
    Each payload kind must construct into the same span without a tag."""
    tool_span = Span(
        trace_id="t1", span_id="s2", parent_span_id="s1", name="search",
        kind=SpanKind.TOOL, status=SpanStatus.OK, status_message=None,
        start_ns=0, end_ns=1, vocabulary="openinference",
        attributes={},
        payload=ToolPayload(
            tool_name="search", call_id="c1", arguments_json="{}",
            arguments={}, result_text="ok", result_len=2, is_error=False,
        ),
    )
    llm_span = Span(
        trace_id="t1", span_id="s3", parent_span_id="s1", name="call",
        kind=SpanKind.LLM, status=SpanStatus.OK, status_message=None,
        start_ns=1, end_ns=2, vocabulary="otel_genai",
        attributes={},
        payload=LlmPayload(
            provider="openai", model="gpt-4o",
            request_messages=[Message(role="user", content="hi")],
            response_messages=[Message(role="assistant", content="hello")],
        ),
    )

    assert isinstance(tool_span.payload, ToolPayload)
    assert isinstance(llm_span.payload, LlmPayload)


def test_retrieval_and_agent_payloads_construct_from_their_documented_fields():
    retrieval = RetrievalPayload(
        query="refund policy",
        documents=[RetrievedDoc(doc_id="d1", text="policy text", score=0.9)],
        top_k=3,
    )
    agent = AgentPayload(
        agent_name="planner", role="orchestrator",
        input_text="refund order 88213", output_text="delegating to billing",
        delegated_to="billing_agent",
    )

    assert retrieval.top_k == 3
    assert retrieval.documents[0].doc_id == "d1"
    assert agent.delegated_to == "billing_agent"


def test_llm_payload_carries_tool_calls_as_typed_requests():
    payload = LlmPayload(
        provider="openai", model="gpt-4o",
        request_messages=[Message(role="user", content="refund it")],
        response_messages=[Message(role="assistant", content="")],
        tool_calls=[
            ToolCallRequest(call_id="c1", tool_name="refund_order", arguments={"order_id": "88213"})
        ],
        finish_reason="tool_calls",
        prompt_tokens=120,
        completion_tokens=15,
    )

    assert payload.tool_calls[0].tool_name == "refund_order"
    assert payload.finish_reason == "tool_calls"


def test_step_collapses_framework_spans_into_collapsed_span_ids():
    """CHAIN/EMBEDDING/GUARDRAIL/UNKNOWN spans never get their own step
    index; their ids fold into the enclosing semantic step instead, per the
    linearization rule that keeps step numbers meaningful to a human."""
    step = Step(
        trace_id="t1", step_index=3, span_id="s7", kind=SpanKind.TOOL,
        actor="billing_agent", depth=2, tree_path="0.1.2",
        signature="tool:billing_agent:refund_order:ok",
        summary="called refund_order", start_ns=10, end_ns=20,
        duration_ms=10.0, collapsed_span_ids=["s5", "s6"],
    )

    assert step.collapsed_span_ids == ["s5", "s6"]
    assert step.step_index == 3


def test_trace_metadata_defaults_to_an_empty_dict_for_orphan_span_tracking():
    """Orphan spans are re-parented to root and flagged in
    metadata["orphan_span_ids"], never silently dropped; metadata must
    default to an empty dict rather than None so callers can always index
    into it without a None check."""
    trace = Trace(
        trace_id="t1", source="otlp", outcome=Outcome.FAILURE,
        span_count=10, step_count=6,
    )

    assert trace.metadata == {}
