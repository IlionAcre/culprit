import json
from pathlib import Path

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1 import trace_pb2

from culprit.otlp import OtlpDecodeError, decode

_FIXTURES = Path(__file__).parent / "fixtures" / "otlp"


def _set_any_value(pb_value, value: dict) -> None:
    """Hand-mirrors otlp.py's own AnyValue decoding in the opposite
    direction, deliberately not via google.protobuf.json_format: that helper
    applies the strict protobuf-JSON mapping (base64 for bytes fields), but
    real OTLP JSON exporters - and these fixtures - use lowercase hex for
    trace/span ids instead (see CLAUDE.md's OTLP JSON gotcha). Using it here
    would silently corrupt the very field this test needs to trust."""
    if "stringValue" in value:
        pb_value.string_value = value["stringValue"]
    elif "boolValue" in value:
        pb_value.bool_value = value["boolValue"]
    elif "intValue" in value:
        pb_value.int_value = int(value["intValue"])
    elif "doubleValue" in value:
        pb_value.double_value = value["doubleValue"]
    elif "arrayValue" in value:
        for item in value["arrayValue"].get("values", []):
            _set_any_value(pb_value.array_value.values.add(), item)


def _protobuf_bytes_from_json_fixture(data: dict) -> bytes:
    """Builds a real protobuf payload carrying the exact same logical
    content as the given OTLP JSON fixture dict, via the raw protobuf API
    (not otlp.py's own code), so a test comparing the two decode paths is an
    independent check rather than round-tripping through the code under
    test."""
    request = ExportTraceServiceRequest()
    for rs in data["resourceSpans"]:
        pb_rs = request.resource_spans.add()
        for ss in rs["scopeSpans"]:
            pb_ss = pb_rs.scope_spans.add()
            for span in ss["spans"]:
                pb_span = pb_ss.spans.add()
                pb_span.trace_id = bytes.fromhex(span["traceId"])
                pb_span.span_id = bytes.fromhex(span["spanId"])
                if span.get("parentSpanId"):
                    pb_span.parent_span_id = bytes.fromhex(span["parentSpanId"])
                pb_span.name = span.get("name", "")
                pb_span.kind = trace_pb2.Span.SpanKind.Value(
                    span.get("kind", "SPAN_KIND_UNSPECIFIED")
                )
                pb_span.start_time_unix_nano = int(span["startTimeUnixNano"])
                pb_span.end_time_unix_nano = int(span["endTimeUnixNano"])
                status = span.get("status", {})
                pb_span.status.code = trace_pb2.Status.StatusCode.Value(
                    status.get("code", "STATUS_CODE_UNSET")
                )
                if status.get("message"):
                    pb_span.status.message = status["message"]
                for attr in span.get("attributes", []):
                    kv = pb_span.attributes.add()
                    kv.key = attr["key"]
                    _set_any_value(kv.value, attr["value"])
    return request.SerializeToString()


@pytest.mark.parametrize("fixture_name", ["otel_genai_sample.json", "openinference_sample.json"])
def test_protobuf_and_json_paths_of_the_same_payload_decode_identically(fixture_name):
    """The strongest form of the parity requirement: build a real protobuf
    payload carrying the same content as the JSON fixture (independently of
    otlp.py's own encoder, since there isn't one), and confirm both wire
    formats decode to byte-for-byte identical raw span dicts."""
    json_bytes = (_FIXTURES / fixture_name).read_bytes()
    data = json.loads(json_bytes)
    protobuf_bytes = _protobuf_bytes_from_json_fixture(data)

    json_spans = decode(json_bytes, "application/json")
    protobuf_spans = decode(protobuf_bytes, "application/x-protobuf")

    assert json_spans == protobuf_spans
    assert len(json_spans) == 5


def test_decode_json_parses_int64_timestamp_strings_not_numbers():
    """CLAUDE.md gotcha: startTimeUnixNano is a JSON string holding a
    nanosecond epoch integer, not a JSON number - a JS/JSON double cannot
    represent a 64-bit integer exactly."""
    spans = decode((_FIXTURES / "otel_genai_sample.json").read_bytes(), "application/json")

    assert spans[0]["start_ns"] == 1755000000000000000
    assert isinstance(spans[0]["start_ns"], int)


def test_decode_json_parses_int_value_attribute_from_a_json_string():
    """CLAUDE.md gotcha: intValue is also a JSON string, unlike doubleValue
    which is an ordinary JSON number - the two are easy to conflate."""
    spans = decode((_FIXTURES / "otel_genai_sample.json").read_bytes(), "application/json")

    llm_span = next(s for s in spans if "gen_ai.usage.input_tokens" in s["attributes"])

    assert llm_span["attributes"]["gen_ai.usage.input_tokens"] == 412
    assert isinstance(llm_span["attributes"]["gen_ai.usage.input_tokens"], int)
    assert llm_span["attributes"]["gen_ai.request.temperature"] == 0.2


def test_decode_json_keeps_trace_and_span_ids_as_lowercase_hex_not_base64():
    """CLAUDE.md gotcha: traceId/spanId are lowercase hex strings in real
    OTLP JSON, even though strict protobuf-JSON mapping would base64-encode
    the underlying bytes field."""
    spans = decode((_FIXTURES / "otel_genai_sample.json").read_bytes(), "application/json")

    assert spans[0]["trace_id"] == "a1b2c3d4e5f60718293a4b5c6d7e8f90"
    assert spans[0]["span_id"] == "aa11bb22cc33dd44"


def test_decode_rejects_an_unsupported_content_type():
    with pytest.raises(OtlpDecodeError, match="unsupported OTLP content type"):
        decode(b"{}", "text/plain")


def test_decode_json_raises_on_malformed_json():
    with pytest.raises(OtlpDecodeError, match="invalid OTLP JSON payload"):
        decode(b"{not valid json", "application/json")


def test_decode_protobuf_raises_on_garbage_bytes():
    with pytest.raises(OtlpDecodeError, match="invalid OTLP protobuf payload"):
        decode(b"\xff\xff\xff\xff\xff\xff not a protobuf message at all", "application/x-protobuf")


def test_decode_json_root_span_has_no_parent():
    spans = decode((_FIXTURES / "otel_genai_sample.json").read_bytes(), "application/json")

    root = next(s for s in spans if s["parent_span_id"] is None)

    assert root["name"] == "invoke_agent order_support_agent"
