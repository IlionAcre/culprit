from datetime import UTC, datetime

import pytest

from culprit.jobs import (
    PersistenceNotWiredError,
    TraceNotFoundError,
    diagnose_trace_job,
    ingest_trace,
    read_diagnoses_for_trace,
    recluster_job,
)
from culprit.schemas import Outcome, Trace
from culprit.synth_results import make_diagnosis


def _trace(trace_id: str = "trace-1") -> Trace:
    return Trace(
        trace_id=trace_id,
        source="synth",
        outcome=Outcome.FAILURE,
        span_count=5,
        step_count=5,
        ingested_at=datetime.now(UTC),
    )


def test_diagnose_trace_job_loads_diagnoses_and_persists_with_injected_seams(monkeypatch):
    """The job must resolve trace_id to a Trace, call pipeline.diagnose, and
    persist the result, using only the injected fakes - no real Postgres,
    Redis, or litellm involved."""
    trace = _trace()
    diagnosis = make_diagnosis(trace_id=trace.trace_id)
    written = []

    def fake_trace_loader(conn_fn, trace_id):
        assert trace_id == trace.trace_id
        return trace

    def fake_diagnosis_writer(conn_fn, d):
        written.append(d)

    monkeypatch.setattr("culprit.jobs.diagnose", lambda t, **kw: diagnosis)

    result = diagnose_trace_job(
        trace.trace_id,
        conn_fn=lambda: None,
        call_fn=lambda model, prompt: ("out", 1.0, 0.0),
        embed_fn=lambda texts: [[0.0] * 384 for _ in texts],
        trace_loader=fake_trace_loader,
        diagnosis_writer=fake_diagnosis_writer,
    )

    assert result == diagnosis.diagnosis_id
    assert written == [diagnosis]


def test_diagnose_trace_job_passes_model_to_pipeline_diagnose(monkeypatch):
    """pipeline.diagnose now requires a model string (Foundation amendment,
    see pipeline.py); the job must resolve one (from CULPRIT_MODEL, falling
    back to DEFAULT_MODEL) and forward it rather than dropping it."""
    trace = _trace()
    diagnosis = make_diagnosis(trace_id=trace.trace_id)
    seen = {}

    def fake_diagnose(t, **kw):
        seen.update(kw)
        return diagnosis

    monkeypatch.setattr("culprit.jobs.diagnose", fake_diagnose)
    monkeypatch.delenv("CULPRIT_MODEL", raising=False)

    diagnose_trace_job(
        trace.trace_id,
        conn_fn=lambda: None,
        call_fn=lambda model, prompt: ("out", 1.0, 0.0),
        embed_fn=lambda texts: [[0.0] * 384 for _ in texts],
        trace_loader=lambda conn_fn, trace_id: trace,
        diagnosis_writer=lambda conn_fn, d: None,
    )

    assert seen["model"] == "gemini/gemini-2.5-flash-lite"

    seen.clear()
    monkeypatch.setenv("CULPRIT_MODEL", "gpt-4o-mini")

    diagnose_trace_job(
        trace.trace_id,
        conn_fn=lambda: None,
        call_fn=lambda model, prompt: ("out", 1.0, 0.0),
        embed_fn=lambda texts: [[0.0] * 384 for _ in texts],
        trace_loader=lambda conn_fn, trace_id: trace,
        diagnosis_writer=lambda conn_fn, d: None,
    )

    assert seen["model"] == "gpt-4o-mini"


def test_diagnose_trace_job_propagates_trace_not_found(monkeypatch):
    """A trace_id with no persisted trace must fail the RQ job loudly, not
    silently return a placeholder result."""

    def raising_loader(conn_fn, trace_id):
        raise TraceNotFoundError(f"no persisted trace with id {trace_id!r}")

    with pytest.raises(TraceNotFoundError):
        diagnose_trace_job(
            "missing-trace",
            conn_fn=lambda: None,
            call_fn=lambda model, prompt: ("out", 1.0, 0.0),
            embed_fn=lambda texts: [],
            trace_loader=raising_loader,
            diagnosis_writer=lambda conn_fn, d: None,
        )


def test_diagnose_trace_job_default_trace_loader_raises_persistence_not_wired():
    """store_traces.py does not exist yet (WS-B). Calling the real default
    loader (rather than injecting a fake) must fail with a clearly-named
    error, not an opaque ModuleNotFoundError."""
    with pytest.raises(PersistenceNotWiredError, match="store_traces"):
        diagnose_trace_job(
            "trace-1",
            conn_fn=lambda: None,
            call_fn=lambda model, prompt: ("out", 1.0, 0.0),
            embed_fn=lambda texts: [],
        )


def test_recluster_job_reads_all_diagnoses_and_clusters_and_writes(monkeypatch):
    """The batch reclustering pass must read every diagnosis (not scoped to
    one trace), call cluster.cluster_diagnoses, and persist the assignment
    via the injected writer."""
    diagnoses = [make_diagnosis(diagnosis_id="d1"), make_diagnosis(diagnosis_id="d2")]
    written = {}

    def fake_reader(conn_fn):
        return diagnoses

    def fake_writer(conn_fn, assignment):
        written.update(assignment)

    monkeypatch.setattr(
        "culprit.jobs.cluster_diagnoses",
        lambda diags, *, embed_fn: {d.diagnosis_id: 0 for d in diags},
    )

    result = recluster_job(
        conn_fn=lambda: None,
        embed_fn=lambda texts: [[0.0] * 384 for _ in texts],
        diagnosis_reader=fake_reader,
        cluster_writer=fake_writer,
    )

    assert result == {"d1": 0, "d2": 0}
    assert written == {"d1": 0, "d2": 0}


def test_recluster_job_default_reader_raises_persistence_not_wired():
    """store_diagnoses.py has no documented bulk reader (see jobs.py's
    module docstring). Exercising the real default without injecting a
    fake must fail clearly rather than crash with an import error."""
    with pytest.raises(PersistenceNotWiredError, match="store_diagnoses"):
        recluster_job(conn_fn=lambda: None, embed_fn=lambda texts: [])


def test_ingest_trace_delegates_to_injected_ingest_fn_and_returns_trace_id():
    """api.py and cli.py both call this; the real otlp/normalize/linearize/
    store_traces pipeline is Integration's job, so this only has to prove
    the seam is wired: content_type and payload reach ingest_fn intact."""
    captured = {}

    def fake_ingest_fn(payload, content_type, conn_fn):
        captured["payload"] = payload
        captured["content_type"] = content_type
        return "trace-abc"

    result = ingest_trace(
        b"raw-bytes", "application/x-protobuf", conn_fn=lambda: None, ingest_fn=fake_ingest_fn
    )

    assert result == "trace-abc"
    assert captured == {"payload": b"raw-bytes", "content_type": "application/x-protobuf"}


def test_ingest_trace_default_ingest_fn_raises_persistence_not_wired():
    """No single WS-A/WS-B function assembles OTLP bytes into a persisted
    Trace yet; the default must say so clearly rather than guess."""
    with pytest.raises(PersistenceNotWiredError, match="OTLP ingestion pipeline"):
        ingest_trace(b"{}", "application/json", conn_fn=lambda: None)


def test_read_diagnoses_for_trace_delegates_to_injected_reader():
    diagnoses = [make_diagnosis(trace_id="trace-1")]

    result = read_diagnoses_for_trace(
        "trace-1",
        conn_fn=lambda: None,
        diagnoses_reader=lambda conn_fn, trace_id: diagnoses,
    )

    assert result == diagnoses
