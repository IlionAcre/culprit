import asyncio
import inspect

from fastapi.testclient import TestClient
from rq.exceptions import NoSuchJobError

from culprit.api import app
from culprit.jobs import PersistenceNotWiredError
from culprit.synth_results import make_diagnosis

client = TestClient(app)


def test_zero_async_def_routes_in_the_service_surface():
    """Load-bearing project guardrail (see CLAUDE.md): the whole codebase
    is sync, including FastAPI routes, so the established
    monkeypatch.setattr mocking pattern keeps working. Only checks routes
    this module defines; FastAPI's own built-in /openapi.json route is
    async internally and is not ours to control."""
    culprit_routes = [
        route
        for route in app.routes
        if getattr(route, "endpoint", None) is not None
        and route.endpoint.__module__ == "culprit.api"
    ]
    assert culprit_routes, "expected at least one culprit.api route to check"
    for route in culprit_routes:
        assert not asyncio.iscoroutinefunction(route.endpoint)
        assert not inspect.iscoroutinefunction(route.endpoint)


def test_ingest_endpoint_passes_json_content_type_through_to_ingest_trace(monkeypatch):
    """The practical promise of culprit is that anyone already emitting
    OTel just points a collector at this endpoint; it must accept
    application/json."""
    captured = {}

    def fake_ingest_trace(payload, content_type):
        captured["payload"] = payload
        captured["content_type"] = content_type
        return "trace-json-1"

    monkeypatch.setattr("culprit.api.ingest_trace", fake_ingest_trace)

    response = client.post(
        "/v1/traces", content=b'{"resourceSpans": []}', headers={"content-type": "application/json"}
    )

    assert response.status_code == 200
    assert response.json() == {"trace_id": "trace-json-1"}
    assert captured["content_type"] == "application/json"
    assert captured["payload"] == b'{"resourceSpans": []}'


def test_ingest_endpoint_passes_protobuf_content_type_through_to_ingest_trace(monkeypatch):
    """Same endpoint must also accept application/x-protobuf, the other
    OTLP wire format."""
    captured = {}

    def fake_ingest_trace(payload, content_type):
        captured["payload"] = payload
        captured["content_type"] = content_type
        return "trace-pb-1"

    monkeypatch.setattr("culprit.api.ingest_trace", fake_ingest_trace)

    response = client.post(
        "/v1/traces", content=b"\x0a\x00", headers={"content-type": "application/x-protobuf"}
    )

    assert response.status_code == 200
    assert response.json() == {"trace_id": "trace-pb-1"}
    assert captured["content_type"] == "application/x-protobuf"
    assert captured["payload"] == b"\x0a\x00"


def test_ingest_endpoint_maps_persistence_not_wired_to_501(monkeypatch):
    def raising(payload, content_type):
        raise PersistenceNotWiredError("OTLP ingestion pipeline is not wired yet")

    monkeypatch.setattr("culprit.api.ingest_trace", raising)

    response = client.post("/v1/traces", content=b"{}", headers={"content-type": "application/json"})

    assert response.status_code == 501
    assert "not wired yet" in response.json()["detail"]


def test_ingest_endpoint_maps_value_error_to_400(monkeypatch):
    def raising(payload, content_type):
        raise ValueError("malformed OTLP payload")

    monkeypatch.setattr("culprit.api.ingest_trace", raising)

    response = client.post("/v1/traces", content=b"{}", headers={"content-type": "application/json"})

    assert response.status_code == 400
    assert response.json()["detail"] == "malformed OTLP payload"


def test_diagnose_trace_endpoint_returns_job_id(monkeypatch):
    monkeypatch.setattr("culprit.api.enqueue_diagnosis", lambda trace_id: "job-123")

    response = client.post("/traces/trace-1/diagnose")

    assert response.status_code == 200
    assert response.json() == {"job_id": "job-123", "trace_id": "trace-1"}


def test_get_diagnoses_returns_summary_views(monkeypatch):
    diagnosis = make_diagnosis(trace_id="trace-1")
    monkeypatch.setattr("culprit.api.read_diagnoses_for_trace", lambda trace_id: [diagnosis])

    response = client.get("/traces/trace-1/diagnoses")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["diagnosis_id"] == diagnosis.diagnosis_id
    assert "signals" not in body[0]


def test_get_diagnoses_maps_empty_result_to_404(monkeypatch):
    monkeypatch.setattr("culprit.api.read_diagnoses_for_trace", lambda trace_id: [])

    response = client.get("/traces/trace-unknown/diagnoses")

    assert response.status_code == 404


def test_get_diagnoses_maps_persistence_not_wired_to_501(monkeypatch):
    def raising(trace_id):
        raise PersistenceNotWiredError("store_diagnoses.read_diagnoses is not available yet")

    monkeypatch.setattr("culprit.api.read_diagnoses_for_trace", raising)

    response = client.get("/traces/trace-1/diagnoses")

    assert response.status_code == 501


def test_get_job_returns_job_status_view(monkeypatch):
    status = {"job_id": "job-1", "status": "finished", "result": "diag-1", "error": None}
    monkeypatch.setattr("culprit.api.fetch_job_status", lambda job_id: status)

    response = client.get("/jobs/job-1")

    assert response.status_code == 200
    assert response.json() == status


def test_get_job_maps_no_such_job_error_to_404(monkeypatch):
    def raising(job_id):
        raise NoSuchJobError(f"no such job {job_id}")

    monkeypatch.setattr("culprit.api.fetch_job_status", raising)

    response = client.get("/jobs/does-not-exist")

    assert response.status_code == 404


def test_recluster_endpoint_returns_job_id(monkeypatch):
    monkeypatch.setattr("culprit.api.enqueue_recluster", lambda: "job-recluster-1")

    response = client.post("/recluster")

    assert response.status_code == 200
    assert response.json() == {"job_id": "job-recluster-1"}
