"""OpenInference vocabulary: `openinference.span.kind` plus its flattened,
index-suffixed `llm.input_messages.0.message.role`-style attributes on
ordinary OTel spans. See CLAUDE.md's OTLP ingestion decision: OpenInference is
richer than OTel GenAI today and already dual-emits, so it is not safe to
skip in favor of GenAI alone.
"""

import json
import re
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

_KIND_MAP = {
    "AGENT": SpanKind.AGENT,
    "LLM": SpanKind.LLM,
    "TOOL": SpanKind.TOOL,
    "RETRIEVER": SpanKind.RETRIEVER,
    "CHAIN": SpanKind.CHAIN,
    "EMBEDDING": SpanKind.EMBEDDING,
    "GUARDRAIL": SpanKind.GUARDRAIL,
}

# Matches OpenInference's flattened "prefix.<index>.rest" attribute keys,
# e.g. "llm.input_messages.1.message.content" or
# "retrieval.documents.0.document.id". OpenInference flattens repeated
# structures into indexed attribute names for OTLP transport instead of
# nesting them, since OTel attributes have no native list-of-object type.
_INDEXED_KEY_RE = re.compile(r"^(?P<index>\d+)\.(?P<rest>.+)$")


def _grouped_by_index(attrs: dict, prefix: str) -> dict[int, dict[str, object]]:
    groups: dict[int, dict[str, object]] = {}
    prefix_dot = prefix + "."
    for key, value in attrs.items():
        if not key.startswith(prefix_dot):
            continue
        m = _INDEXED_KEY_RE.match(key[len(prefix_dot):])
        if not m:
            continue
        groups.setdefault(int(m.group("index")), {})[m.group("rest")] = value
    return groups


_TOOL_CALL_RE = re.compile(
    r"^llm\.output_messages\.(\d+)\.message\.tool_calls\.(\d+)\.tool_call\.(.+)$"
)


def _output_tool_calls(attrs: dict) -> list[ToolCallRequest]:
    """All tool calls across every output message, in (message, call) index
    order. LlmPayload.tool_calls is a flat list (see schemas.py), so the
    message boundary that OpenInference's flattened keys preserve is not
    needed downstream."""
    by_msg: dict[int, dict[int, dict[str, object]]] = {}
    for key, value in attrs.items():
        m = _TOOL_CALL_RE.match(key)
        if not m:
            continue
        msg_i, call_i, field = int(m.group(1)), int(m.group(2)), m.group(3)
        by_msg.setdefault(msg_i, {}).setdefault(call_i, {})[field] = value

    calls = []
    for msg_i in sorted(by_msg):
        for call_i in sorted(by_msg[msg_i]):
            c = by_msg[msg_i][call_i]
            args_json = c.get("function.arguments") or "{}"
            try:
                args = json.loads(args_json) if args_json else {}
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCallRequest(
                call_id=c.get("id", ""), tool_name=c.get("function.name", ""),
                arguments=args, arguments_json=args_json,
            ))
    return calls


def _messages(attrs: dict, prefix: str) -> list[Message]:
    groups = _grouped_by_index(attrs, prefix)
    return [
        Message(
            role=groups[i].get("message.role", "unknown"),
            content=groups[i].get("message.content"),
        )
        for i in sorted(groups)
    ]


def _llm_payload(attrs: dict) -> LlmPayload:
    params = parse_json_attr(attrs, "llm.invocation_parameters", {})
    return LlmPayload(
        provider=attrs.get("llm.system", "unknown"),
        model=attrs.get("llm.model_name", "unknown"),
        request_messages=_messages(attrs, "llm.input_messages"),
        response_messages=_messages(attrs, "llm.output_messages"),
        tool_calls=_output_tool_calls(attrs),
        finish_reason=attrs.get("llm.finish_reason"),
        prompt_tokens=attrs.get("llm.token_count.prompt"),
        completion_tokens=attrs.get("llm.token_count.completion"),
        total_tokens=attrs.get("llm.token_count.total"),
        max_tokens=attrs.get("llm.token_count.max"),
        temperature=params.get("temperature"),
    )


def _tool_payload(attrs: dict, is_error: bool, error_message: str | None) -> ToolPayload:
    arguments_json = attrs.get("input.value") or "{}"
    try:
        arguments = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as e:
        raise ValueError(f"tool input.value is not valid JSON: {e}") from e
    result_text = attrs.get("output.value", "")
    return ToolPayload(
        tool_name=attrs.get("tool.name", "unknown"), call_id=attrs.get("tool.id", ""),
        arguments_json=arguments_json, arguments=arguments,
        result_text=result_text, result_len=len(result_text),
        is_error=is_error, error_message=error_message,
    )


def _retrieval_payload(attrs: dict) -> RetrievalPayload:
    groups = _grouped_by_index(attrs, "retrieval.documents")
    docs = [
        RetrievedDoc(
            doc_id=groups[i].get("document.id", f"doc-{i}"),
            content_len=len(groups[i].get("document.content", "")),
            content_preview=groups[i].get("document.content", "")[:200],
            score=groups[i].get("document.score"),
        )
        for i in sorted(groups)
    ]
    return RetrievalPayload(query=attrs.get("input.value", ""), documents=docs, top_k=len(docs))


def _agent_payload(attrs: dict) -> AgentPayload:
    metadata = parse_json_attr(attrs, "metadata", {})
    return AgentPayload(
        agent_name=metadata.get("agent_name", "unknown"), role="",
        input_text=attrs.get("input.value", ""), output_text=attrs.get("output.value", ""),
        delegated_to=[],
    )


def matches(raw: dict) -> bool:
    return "openinference.span.kind" in raw.get("attributes", {})


def to_span(raw: dict, trace_id: str) -> Span:
    attrs = raw.get("attributes", {})
    kind = _KIND_MAP.get(attrs.get("openinference.span.kind", ""), SpanKind.UNKNOWN)
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
        end_ns=int(raw["end_ns"]), vocabulary="openinference", attributes=dict(attrs),
        payload=payload,
    )


@dataclass(frozen=True)
class _OpenInferenceVocab:
    name: str = "openinference"

    def matches(self, raw: dict) -> bool:
        return matches(raw)

    def to_span(self, raw: dict, trace_id: str) -> Span:
        return to_span(raw, trace_id)


VOCAB = _OpenInferenceVocab()
