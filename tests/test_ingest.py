"""Integration task I2: proves ingest.ingest_payload really composes
otlp.decode/normalize_span/linearize into a persisted Trace, for both
input shapes it has to handle - real OTLP-JSON and the direct "raw_upload"
envelope (CLAUDE.md's "third, distinct ingestion path"). write_trace is
monkeypatched throughout so no real Postgres is needed; the assembly
itself (not the SQL) is what this file proves.
"""

import json
from pathlib import Path

from culprit.ingest import ingest_payload
from culprit.schemas import Outcome

_FIXTURES = Path(__file__).parent / "fixtures" / "otlp"


def _read(name: str) -> bytes:
    return (_FIXTURES / name).read_bytes()


def test_ingest_payload_assembles_a_trace_from_otlp_json(monkeypatch):
    """otel_genai_sample.json is real OTLP-JSON (resourceSpans/scopeSpans),
    the shape otlp.decode expects; proves that path end to end."""
    written = {}
    monkeypatch.setattr(
        "culprit.ingest.write_trace",
        lambda conn_fn, trace, spans, steps, embedding=None: written.update(
            trace=trace, spans=spans, steps=steps, embedding=embedding
        ),
    )

    trace_id = ingest_payload(_read("otel_genai_sample.json"), "application/json", conn_fn=lambda: None)

    assert trace_id == written["trace"].trace_id
    assert written["trace"].source == "otlp"
    assert written["trace"].span_count == len(written["spans"]) > 0
    assert written["trace"].step_count == len(written["steps"]) > 0
    assert written["trace"].agent_key
    assert written["trace"].task_key
    assert written["embedding"] is None  # no embed_fn supplied


def test_ingest_payload_assembles_a_trace_from_the_raw_upload_envelope(monkeypatch):
    """raw_upload_sample.json carries trace_id/source/framework/task_goal/
    outcome explicitly at the top level; the ingest path must read them as
    given rather than re-deriving them with the OTLP heuristics."""
    written = {}
    monkeypatch.setattr(
        "culprit.ingest.write_trace",
        lambda conn_fn, trace, spans, steps, embedding=None: written.update(trace=trace, spans=spans),
    )
    envelope = json.loads(_read("raw_upload_sample.json"))

    trace_id = ingest_payload(_read("raw_upload_sample.json"), "application/json", conn_fn=lambda: None)

    assert trace_id == envelope["trace_id"] == written["trace"].trace_id
    assert written["trace"].source == "manual_upload"
    assert written["trace"].framework == "custom_refund_bot"
    assert written["trace"].task_goal == envelope["task_goal"]
    assert written["trace"].outcome == Outcome.FAILURE
    assert len(written["spans"]) == len(envelope["spans"])


def test_ingest_payload_calls_embed_fn_on_the_task_goal_when_supplied(monkeypatch):
    monkeypatch.setattr(
        "culprit.ingest.write_trace",
        lambda conn_fn, trace, spans, steps, embedding=None: captured.update(embedding=embedding),
    )
    captured = {}

    ingest_payload(
        _read("raw_upload_sample.json"), "application/json", conn_fn=lambda: None,
        embed_fn=lambda texts: [[0.5] * 384 for _ in texts],
    )

    assert captured["embedding"] == [0.5] * 384


def test_ingest_payload_distinguishes_otlp_json_from_the_raw_upload_envelope_by_shape(monkeypatch):
    """Both real shapes are posted as content_type application/json; only
    the body (resourceSpans vs a bare spans list) tells them apart, so an
    OTLP-JSON payload must not be mistaken for a raw_upload envelope."""
    written = {}
    monkeypatch.setattr(
        "culprit.ingest.write_trace",
        lambda conn_fn, trace, spans, steps, embedding=None: written.update(source=trace.source),
    )

    ingest_payload(_read("openinference_sample.json"), "application/json", conn_fn=lambda: None)

    assert written["source"] == "otlp"
