"""OTel GenAI semantic-convention vocabulary: `gen_ai.*` attributes on
ordinary OTel spans. See CLAUDE.md's OTLP ingestion decision for why this and
OpenInference are both supported rather than betting on one - GenAI *agent*
spans were still Development status as of mid-2026 while OpenInference is
richer today and already dual-emits.
"""

import json
from dataclasses import dataclass

from culprit.schemas import (
    AgentPayload,
    LlmPayload,
    Message,
    RetrievalPayload,
    RetrievedDoc,
    Span,
    SpanKind,
    SpanStatus,
    ToolCallRequest,
    ToolPayload,
)
from culprit.vocab.base import otlp_status, parse_json_attr

_OPERATION_TO_KIND = {
    "invoke_agent": SpanKind.AGENT,
    "create_agent": SpanKind.AGENT,
    "chat": SpanKind.LLM,
    "text_completion": SpanKind.LLM,
    "generate_content": SpanKind.LLM,
    "embeddings": SpanKind.EMBEDDING,
    "execute_tool": SpanKind.TOOL,
    # Not (yet) an official gen_ai.operation.name in the OTel semantic
    # conventions, but the value this fixture and several early real-world
    # instrumentations use for retrieval steps; RETRIEVER is the closest
    # canonical kind and the alternative (leaving it UNKNOWN) would silently
    # collapse a semantic step into framework noise.
    "retrieve_documents": SpanKind.RETRIEVER,
}


def _messages_from(items: list[dict]) -> list[Message]:
    return [
        Message(
            role=item.get("role", "unknown"), content=item.get("content"),
            tool_call_id=item.get("tool_call_id"), name=item.get("name"),
        )
        for item in items
    ]


def _tool_calls_from(items: list[dict]) -> list[ToolCallRequest]:
    calls = []
    for item in items:
        for tc in item.get("tool_calls") or []:
            function = tc.get("function", {})
            args_json = function.get("arguments") or "{}"
            try:
                args = json.loads(args_json) if args_json else {}
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCallRequest(
                call_id=tc.get("id", ""), tool_name=function.get("name", ""),
                arguments=args, arguments_json=args_json,
            ))
    return calls


def _llm_payload(attrs: dict) -> LlmPayload:
    response_items = parse_json_attr(attrs, "gen_ai.output.messages", [])
    finish_reasons = attrs.get("gen_ai.response.finish_reasons") or []
    prompt_tokens = attrs.get("gen_ai.usage.input_tokens")
    completion_tokens = attrs.get("gen_ai.usage.output_tokens")
    total_tokens = (
        prompt_tokens + completion_tokens
        if prompt_tokens is not None and completion_tokens is not None
        else None
    )
    return LlmPayload(
        provider=attrs.get("gen_ai.provider.name", "unknown"),
        model=attrs.get("gen_ai.response.model") or attrs.get("gen_ai.request.model", "unknown"),
        request_messages=_messages_from(parse_json_attr(attrs, "gen_ai.input.messages", [])),
        response_messages=_messages_from(response_items),
        tool_calls=_tool_calls_from(response_items),
        finish_reason=finish_reasons[0] if finish_reasons else None,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        total_tokens=total_tokens, max_tokens=attrs.get("gen_ai.request.max_tokens"),
        temperature=attrs.get("gen_ai.request.temperature"),
    )


def _tool_payload(attrs: dict, is_error: bool, error_message: str | None) -> ToolPayload:
    arguments_json = attrs.get("gen_ai.tool.call.arguments") or "{}"
    try:
        arguments = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as e:
        raise ValueError(f"tool call arguments are not valid JSON: {e}") from e
    result_text = attrs.get("gen_ai.tool.call.result", "")
    return ToolPayload(
        tool_name=attrs.get("gen_ai.tool.name", "unknown"),
        call_id=attrs.get("gen_ai.tool.call.id", ""),
        arguments_json=arguments_json, arguments=arguments,
        result_text=result_text, result_len=len(result_text),
        is_error=is_error, error_message=error_message,
    )


def _retrieval_payload(attrs: dict) -> RetrievalPayload:
    documents = parse_json_attr(attrs, "gen_ai.retrieval.documents", [])
    docs = [
        RetrievedDoc(
            doc_id=doc.get("id", f"doc-{n}"), content_len=len(doc.get("content", "")),
            content_preview=doc.get("content", "")[:200], score=doc.get("score"),
        )
        for n, doc in enumerate(documents)
    ]
    top_k = attrs.get("gen_ai.retrieval.document.count", len(docs))
    return RetrievalPayload(query=attrs.get("gen_ai.retrieval.query", ""), documents=docs, top_k=top_k)


def _agent_payload(attrs: dict) -> AgentPayload:
    return AgentPayload(
        agent_name=attrs.get("gen_ai.agent.name", "unknown"),
        role=attrs.get("gen_ai.agent.description") or "",
        # Neither the invoke_agent operation nor its sibling GenAI attributes
        # carry the agent's full input/output text - that content lives on
        # the LLM child spans instead. Left blank rather than guessed.
        input_text="", output_text="", delegated_to=[],
    )


def matches(raw: dict) -> bool:
    return "gen_ai.operation.name" in raw.get("attributes", {})


def to_span(raw: dict, trace_id: str) -> Span:
    attrs = raw.get("attributes", {})
    operation = attrs.get("gen_ai.operation.name", "")
    kind = _OPERATION_TO_KIND.get(operation, SpanKind.UNKNOWN)
    status = otlp_status(raw.get("status_code"))
    is_error = status == SpanStatus.ERROR

    payload = None
    if kind == SpanKind.LLM:
        payload = _llm_payload(attrs)
    elif kind == SpanKind.TOOL:
        payload = _tool_payload(attrs, is_error, raw.get("status_message") if is_error else None)
    elif kind == SpanKind.RETRIEVER:
        payload = _retrieval_payload(attrs)
    elif kind == SpanKind.AGENT:
        payload = _agent_payload(attrs)

    return Span(
        trace_id=trace_id, span_id=raw["span_id"], parent_span_id=raw.get("parent_span_id"),
        name=raw.get("name", ""), kind=kind, status=status,
        status_message=raw.get("status_message"), start_ns=int(raw["start_ns"]),
        end_ns=int(raw["end_ns"]), vocabulary="otel_genai", attributes=dict(attrs),
        payload=payload,
    )


@dataclass(frozen=True)
class _OtelGenaiVocab:
    name: str = "otel_genai"

    def matches(self, raw: dict) -> bool:
        return matches(raw)

    def to_span(self, raw: dict, trace_id: str) -> Span:
        return to_span(raw, trace_id)


VOCAB = _OtelGenaiVocab()
