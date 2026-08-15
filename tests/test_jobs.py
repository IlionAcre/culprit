from datetime import UTC, datetime
from pathlib import Path

import pytest

from culprit.jobs import (
    TraceNotFoundError,
    diagnose_trace_job,
    ingest_trace,
    read_diagnoses_for_trace,
    recluster_job,
)
from culprit.schemas import Outcome, Trace
from culprit.synth_results import make_diagnosis

_RAW_UPLOAD_FIXTURE = Path(__file__).parent / "fixtures" / "otlp" / "raw_upload_sample.json"


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


def test_diagnose_trace_job_default_trace_loader_wires_to_store_traces_read_trace(monkeypatch):
    """store_traces.py now exists (WS-B): the default loader's `_seam()` call
    must actually resolve `culprit.store_traces.read_trace` and use its
    result, rather than raising `PersistenceNotWiredError` forever. Proven
    here by monkeypatching the real function (not the seam) and checking its
    `(None, [], [])` "no such trace" return is what turns into
    `TraceNotFoundError` - the same contract `_default_trace_loader`
    documents. This test is the stale side of the WS-B/WS-G handshake:
    it asserted absence before WS-B landed, and now asserts wiring instead,
    the same resolution `tests/test_synth_results.py` already used for the
    analysis-layer stubs."""

    def fake_read_trace(conn_fn, trace_id):
        return None, [], []

    monkeypatch.setattr("culprit.store_traces.read_trace", fake_read_trace)

    with pytest.raises(TraceNotFoundError):
        diagnose_trace_job(
            "missing-trace",
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


def test_recluster_job_default_reader_and_writer_wire_to_store_diagnoses(monkeypatch):
    """store_diagnoses.py now exists (WS-B) with the exact reader name
    jobs.py's docstring guessed (`read_all_diagnoses`); the writer,
    `write_cluster_assignments`, lives in the sibling `store_clusters.py`
    (INTEGRATION_ITEMS.md item 4 - jobs.py's seam was repointed there
    directly once the temporary store_diagnoses re-export was deleted).
    Both default seams must resolve to the real functions rather than
    raising `PersistenceNotWiredError` forever. Proven by monkeypatching the
    real functions (not the seams) and checking the job's return value and
    the writer's captured call both come from that real wiring path."""
    diagnoses = [make_diagnosis(diagnosis_id="d1"), make_diagnosis(diagnosis_id="d2")]
    written = {}

    monkeypatch.setattr("culprit.store_diagnoses.read_all_diagnoses", lambda conn_fn: diagnoses)
    monkeypatch.setattr(
        "culprit.store_clusters.write_cluster_assignments",
        lambda conn_fn, assignment: written.update(assignment),
    )
    monkeypatch.setattr(
        "culprit.jobs.cluster_diagnoses",
        lambda diags, *, embed_fn: {d.diagnosis_id: 0 for d in diags},
    )

    result = recluster_job(conn_fn=lambda: None, embed_fn=lambda texts: [])

    assert result == {"d1": 0, "d2": 0}
    assert written == {"d1": 0, "d2": 0}


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


def test_ingest_trace_default_ingest_fn_wires_to_ingest_payload(monkeypatch):
    """INTEGRATION_ITEMS.md / this module's own former docstring gap,
    resolved (task I2): the default ingest_fn now composes otlp/normalize/
    linearize/store_traces for real via ingest.ingest_payload, rather than
    raising PersistenceNotWiredError forever. Proven with the raw_upload
    fixture (CLAUDE.md's "third, distinct ingestion path") and
    store_traces.write_trace monkeypatched so no real Postgres is needed."""
    written = {}
    monkeypatch.setattr(
        "culprit.ingest.write_trace",
        lambda conn_fn, trace, spans, steps, embedding=None: written.update(trace_id=trace.trace_id),
    )
    # Real embed_texts loads the ~130MB fastembed ONNX model; stub it so
    # this test stays offline like every other test in the suite.
    monkeypatch.setattr("culprit.jobs.embed_texts", lambda texts: [[0.0] * 384 for _ in texts])

    result = ingest_trace(
        _RAW_UPLOAD_FIXTURE.read_bytes(), "application/json", conn_fn=lambda: None,
    )

    assert result == written["trace_id"] == "raw-upload-7f2c19"


def test_read_diagnoses_for_trace_delegates_to_injected_reader():
    diagnoses = [make_diagnosis(trace_id="trace-1")]

    result = read_diagnoses_for_trace(
        "trace-1",
        conn_fn=lambda: None,
        diagnoses_reader=lambda conn_fn, trace_id: diagnoses,
    )

    assert result == diagnoses
