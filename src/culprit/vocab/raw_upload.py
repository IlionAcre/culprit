"""Direct-upload vocabulary: a trace posted as a plain JSON document (a
`spans` list of already near-canonical span dicts) rather than as an OTLP
export. See CLAUDE.md: `raw_upload_sample.json` is "a third, distinct
ingestion path (direct JSON upload, not OTLP)" - `otlp.py` never sees this
shape, but it still has to become `Span` objects through the same
`normalize_span` entry point as everything else, which is why it gets a
vocab module of its own rather than a special case somewhere else.

Distinguishing marker: this shape's `kind`/`status` values are already the
canonical `SpanKind`/`SpanStatus` strings and it carries ISO-8601
`start_time`/`end_time` instead of `start_ns`/`end_ns` - no OTLP-derived raw
dict has either of those.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime

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

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _parse_ns(value: str) -> int:
    """ISO-8601 (optionally 'Z'-suffixed) to epoch nanoseconds. Only
    microsecond precision survives the round trip - `datetime` itself has no
    finer resolution - which is enough for the millisecond-grained
    timestamps every uploader in practice sends."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    delta = dt - _EPOCH
    return delta.days * 86_400_000_000_000 + delta.seconds * 1_000_000_000 + delta.microseconds * 1000


def _tool_calls_from(messages: list[dict]) -> list[ToolCallRequest]:
    calls = []
    for item in messages:
        for tc in item.get("tool_calls") or []:
            arguments = tc.get("arguments", {})
            calls.append(ToolCallRequest(
                call_id=tc.get("id", ""), tool_name=tc.get("name", ""),
                arguments=arguments, arguments_json=json.dumps(arguments),
            ))
    return calls


def _llm_payload(attrs: dict) -> LlmPayload:
    request_messages = attrs.get("request_messages", [])
    response_messages = attrs.get("response_messages", [])
    prompt_tokens = attrs.get("prompt_tokens")
    completion_tokens = attrs.get("completion_tokens")
    total_tokens = (
        prompt_tokens + completion_tokens
        if prompt_tokens is not None and completion_tokens is not None
        else None
    )
    return LlmPayload(
        provider=attrs.get("provider", "unknown"), model=attrs.get("model", "unknown"),
        request_messages=[
            Message(role=m.get("role", "unknown"), content=m.get("content"),
                    tool_call_id=m.get("tool_call_id"), name=m.get("name"))
            for m in request_messages
        ],
        response_messages=[
            Message(role=m.get("role", "unknown"), content=m.get("content"),
                    tool_call_id=m.get("tool_call_id"), name=m.get("name"))
            for m in response_messages
        ],
        tool_calls=_tool_calls_from(response_messages),
        finish_reason=attrs.get("finish_reason"),
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        total_tokens=total_tokens, max_tokens=attrs.get("max_tokens"),
        temperature=attrs.get("temperature"),
    )


def _tool_payload(attrs: dict) -> ToolPayload:
    arguments = attrs.get("arguments", {})
    result_text = attrs.get("result", "")
    is_error = bool(attrs.get("is_error", False))
    return ToolPayload(
        tool_name=attrs.get("tool_name", "unknown"), call_id=attrs.get("call_id", ""),
        arguments_json=json.dumps(arguments), arguments=arguments,
        result_text=result_text, result_len=len(result_text),
        is_error=is_error, error_message=attrs.get("error_message") if is_error else None,
    )


def _retrieval_payload(attrs: dict) -> RetrievalPayload:
    documents = attrs.get("documents", [])
    docs = [
        RetrievedDoc(
            doc_id=doc.get("doc_id", f"doc-{n}"), content_len=len(doc.get("content", "")),
            content_preview=doc.get("content", "")[:200], score=doc.get("score"),
        )
        for n, doc in enumerate(documents)
    ]
    return RetrievalPayload(query=attrs.get("query", ""), documents=docs, top_k=len(docs))


def _agent_payload(attrs: dict, actor: str) -> AgentPayload:
    return AgentPayload(
        agent_name=attrs.get("agent.name", actor), role=attrs.get("agent.role") or "",
        input_text=attrs.get("input_text", ""), output_text=attrs.get("output_text", ""),
        delegated_to=attrs.get("delegated_to", []),
    )


def matches(raw: dict) -> bool:
    return (
        raw.get("kind") in {k.value for k in SpanKind}
        and "start_time" in raw
        and "end_time" in raw
    )


def to_span(raw: dict, trace_id: str) -> Span:
    attrs = raw.get("attributes", {})
    kind = SpanKind(raw["kind"])
    status = SpanStatus(raw.get("status", "unset"))
    actor = raw.get("actor", "unknown")

    payload = None
    if kind == SpanKind.LLM:
        payload = _llm_payload(attrs)
    elif kind == SpanKind.TOOL:
        payload = _tool_payload(attrs)
    elif kind == SpanKind.RETRIEVER:
        payload = _retrieval_payload(attrs)
    elif kind == SpanKind.AGENT:
        payload = _agent_payload(attrs, actor)

    return Span(
        trace_id=trace_id, span_id=raw["span_id"], parent_span_id=raw.get("parent_span_id"),
        name=raw.get("name", ""), kind=kind, status=status, status_message=None,
        start_ns=_parse_ns(raw["start_time"]), end_ns=_parse_ns(raw["end_time"]),
        vocabulary="raw_upload", attributes=dict(attrs), payload=payload,
    )


@dataclass(frozen=True)
class _RawUploadVocab:
    name: str = "raw_upload"

    def matches(self, raw: dict) -> bool:
        return matches(raw)

    def to_span(self, raw: dict, trace_id: str) -> Span:
        return to_span(raw, trace_id)


VOCAB = _RawUploadVocab()
