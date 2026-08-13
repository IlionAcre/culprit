"""OTLP payload decoding: turns a protobuf or JSON OTLP trace export into a
flat list of per-span raw dicts, one shape regardless of wire format, so
`vocab/` and `normalize.py` never need to know which transport a trace
arrived over.

See CLAUDE.md's "Known gotcha: OTLP JSON encodes int64 fields and timestamps
as strings" - `startTimeUnixNano`/`endTimeUnixNano` and any `intValue`
attribute are JSON *strings* holding an integer, not JSON numbers, because a
JS/JSON double cannot represent a 64-bit integer exactly; `doubleValue` is an
ordinary JSON number. `traceId`/`spanId` are lowercase hex strings in real
OTLP JSON, even though they are protobuf `bytes` fields and strict
protobuf-JSON mapping would base64-encode them - decoding both wire formats
by hand (rather than via `google.protobuf.json_format`, which applies that
strict mapping) is what keeps the two paths in agreement.
"""

import base64
import json
import logging
from typing import Any

from opentelemetry.proto.collector.trace.v1 import trace_service_pb2
from opentelemetry.proto.trace.v1 import trace_pb2

from culprit.logging_config import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

_JSON_CONTENT_TYPES = {"application/json"}
_PROTOBUF_CONTENT_TYPES = {"application/x-protobuf", "application/protobuf"}


class OtlpDecodeError(Exception):
    """Raised when the whole payload cannot be parsed as OTLP at all
    (malformed bytes, unsupported content type). Per-span malformed data is
    normalize.py's concern (normalize_error on that one Span); this is a
    whole-payload parse failure with no per-item fallback available."""


def decode(payload: bytes, content_type: str) -> list[dict]:
    content_type = content_type.split(";")[0].strip().lower()
    if content_type in _JSON_CONTENT_TYPES:
        return _decode_json(payload)
    if content_type in _PROTOBUF_CONTENT_TYPES:
        return _decode_protobuf(payload)
    raise OtlpDecodeError(f"unsupported OTLP content type {content_type!r}")


# --- protobuf path -----------------------------------------------------

def _decode_protobuf(payload: bytes) -> list[dict]:
    request = trace_service_pb2.ExportTraceServiceRequest()
    try:
        request.ParseFromString(payload)
    except Exception as e:
        raise OtlpDecodeError(f"invalid OTLP protobuf payload: {e}") from e

    return [
        _span_from_pb(span)
        for resource_spans in request.resource_spans
        for scope_spans in resource_spans.scope_spans
        for span in scope_spans.spans
    ]


def _span_from_pb(span) -> dict:
    return {
        "trace_id": span.trace_id.hex(),
        "span_id": span.span_id.hex(),
        "parent_span_id": span.parent_span_id.hex() or None,
        "name": span.name,
        "otel_kind": trace_pb2.Span.SpanKind.Name(span.kind),
        "start_ns": span.start_time_unix_nano,
        "end_ns": span.end_time_unix_nano,
        "status_code": trace_pb2.Status.StatusCode.Name(span.status.code),
        "status_message": span.status.message or None,
        "attributes": {kv.key: _any_value_from_pb(kv.value) for kv in span.attributes},
    }


def _any_value_from_pb(value) -> Any:
    which = value.WhichOneof("value")
    if which == "string_value":
        return value.string_value
    if which == "bool_value":
        return value.bool_value
    if which == "int_value":
        return value.int_value
    if which == "double_value":
        return value.double_value
    if which == "bytes_value":
        return value.bytes_value
    if which == "array_value":
        return [_any_value_from_pb(v) for v in value.array_value.values]
    if which == "kvlist_value":
        return {kv.key: _any_value_from_pb(kv.value) for kv in value.kvlist_value.values}
    return None


# --- JSON path -----------------------------------------------------------

def _decode_json(payload: bytes) -> list[dict]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        raise OtlpDecodeError(f"invalid OTLP JSON payload: {e}") from e

    try:
        return [
            _span_from_json(span)
            for resource_spans in data.get("resourceSpans", [])
            for scope_spans in resource_spans.get("scopeSpans", [])
            for span in scope_spans.get("spans", [])
        ]
    except (KeyError, TypeError, ValueError) as e:
        raise OtlpDecodeError(f"malformed OTLP JSON payload: {e}") from e


def _span_from_json(span: dict) -> dict:
    status = span.get("status", {})
    return {
        "trace_id": span["traceId"],
        "span_id": span["spanId"],
        "parent_span_id": span.get("parentSpanId") or None,
        "name": span.get("name", ""),
        "otel_kind": span.get("kind", "SPAN_KIND_UNSPECIFIED"),
        # Known gotcha: int64 fields are JSON strings, not numbers.
        "start_ns": int(span["startTimeUnixNano"]),
        "end_ns": int(span["endTimeUnixNano"]),
        "status_code": status.get("code", "STATUS_CODE_UNSET"),
        "status_message": status.get("message") or None,
        "attributes": {
            kv["key"]: _any_value_from_json(kv["value"]) for kv in span.get("attributes", [])
        },
    }


def _any_value_from_json(value: dict) -> Any:
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "intValue" in value:
        # Known gotcha: intValue is a JSON string, unlike doubleValue.
        return int(value["intValue"])
    if "doubleValue" in value:
        return value["doubleValue"]
    if "bytesValue" in value:
        return base64.b64decode(value["bytesValue"])
    if "arrayValue" in value:
        return [_any_value_from_json(v) for v in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return {
            kv["key"]: _any_value_from_json(kv["value"])
            for kv in value["kvlistValue"].get("values", [])
        }
    return None
