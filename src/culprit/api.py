"""culprit's FastAPI service surface: OTLP ingestion, diagnosis job
enqueue/status, diagnosis read-back. Sync `def` routes throughout, no
`APIRouter`, no `Depends()`, matching Litmus's `api.py` exactly (see
Litmus/CLAUDE.md for why); the whole codebase is sync by guardrail (see
culprit's own CLAUDE.md) and the established `monkeypatch.setattr` mocking
pattern depends on it. Every route follows the same per-route try/except
shape: a `logger.warning(..., extra={"event": "api_error", ...})` right
before `raise HTTPException(...) from e`, so a domain failure is always
both logged and turned into the right HTTP status.
"""

import json
import logging
from typing import Any

from fastapi import Body, FastAPI, Header, HTTPException
from rq.exceptions import NoSuchJobError

from culprit.jobs import PersistenceNotWiredError, ingest_trace, read_diagnoses_for_trace
from culprit.logging_config import LOGGER_NAME
from culprit.queue import enqueue_diagnosis, enqueue_recluster, fetch_job_status
from culprit.views import diagnosis_summary_view, job_status_view

app = FastAPI(title="culprit")
logger = logging.getLogger(LOGGER_NAME)


@app.post("/v1/traces")
def ingest(
    payload: Any = Body(...),
    content_type: str = Header(default="application/json", alias="content-type"),
) -> dict:
    """Accepts an OTLP payload, protobuf or JSON, at the same endpoint:
    the practical promise of culprit is that anyone already emitting OTel
    just points a collector here. `Content-Type` decides how the payload
    is decoded (see `culprit.jobs.ingest_trace`).

    `payload` is typed `Any`, not `bytes`, because FastAPI/Starlette
    JSON-decode the request body ahead of route dispatch whenever
    Content-Type's subtype is "json" (see `fastapi.routing.
    request_body_to_args`), regardless of the declared Body() type - a
    `bytes`-typed parameter 422s on exactly the OTLP JSON payloads this
    endpoint most needs to accept. Re-encoding an already-parsed JSON body
    back to bytes below is the only way to get raw bytes to `ingest_trace`
    for both wire formats without an `async def` route."""
    raw_payload = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(payload).encode()
    logger.info(
        "POST /v1/traces",
        extra={"event": "api_request", "route": "/v1/traces", "content_type": content_type},
    )
    try:
        trace_id = ingest_trace(raw_payload, content_type)
    except PersistenceNotWiredError as e:
        logger.warning(
            str(e),
            extra={"event": "api_error", "route": "/v1/traces", "status_code": 501},
        )
        raise HTTPException(status_code=501, detail=str(e)) from e
    except ValueError as e:
        logger.warning(
            str(e),
            extra={"event": "api_error", "route": "/v1/traces", "status_code": 400},
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"trace_id": trace_id}


@app.post("/traces/{trace_id}/diagnose")
def diagnose_trace(trace_id: str) -> dict:
    """Enqueue analysis for an already-ingested trace and return the RQ
    job id immediately; analysis takes minutes, so it never runs inline
    (see CLAUDE.md)."""
    logger.info(
        "POST /traces/{trace_id}/diagnose",
        extra={
            "event": "api_request",
            "route": "/traces/{trace_id}/diagnose",
            "trace_id": trace_id,
        },
    )
    job_id = enqueue_diagnosis(trace_id)
    return {"job_id": job_id, "trace_id": trace_id}


@app.get("/traces/{trace_id}/diagnoses")
def get_diagnoses(trace_id: str) -> list[dict]:
    """Every persisted diagnosis for a trace (diagnoses.trace_id is
    deliberately not unique; re-running analysis adds a new row alongside
    older ones, see CLAUDE.md)."""
    logger.info(
        "GET /traces/{trace_id}/diagnoses",
        extra={
            "event": "api_request",
            "route": "/traces/{trace_id}/diagnoses",
            "trace_id": trace_id,
        },
    )
    try:
        diagnoses = read_diagnoses_for_trace(trace_id)
    except PersistenceNotWiredError as e:
        logger.warning(
            str(e),
            extra={
                "event": "api_error",
                "route": "/traces/{trace_id}/diagnoses",
                "status_code": 501,
            },
        )
        raise HTTPException(status_code=501, detail=str(e)) from e
    if not diagnoses:
        logger.warning(
            "no diagnoses for trace",
            extra={
                "event": "api_error",
                "route": "/traces/{trace_id}/diagnoses",
                "status_code": 404,
                "trace_id": trace_id,
            },
        )
        raise HTTPException(
            status_code=404, detail=f"no diagnoses for trace {trace_id!r}"
        )
    return [diagnosis_summary_view(d) for d in diagnoses]


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """Poll one job's status; the caller learns about job completion here,
    not via a webhook or long-lived connection."""
    logger.info(
        "GET /jobs/{job_id}",
        extra={"event": "api_request", "route": "/jobs/{job_id}", "job_id": job_id},
    )
    try:
        status = fetch_job_status(job_id)
    except NoSuchJobError as e:
        logger.warning(
            str(e),
            extra={
                "event": "api_error",
                "route": "/jobs/{job_id}",
                "status_code": 404,
                "job_id": job_id,
            },
        )
        raise HTTPException(status_code=404, detail=f"no job with id {job_id!r}") from e
    return job_status_view(status)


@app.post("/recluster")
def recluster() -> dict:
    """Enqueue the scheduled batch reclustering pass. Deliberately not
    triggered as a side effect of `/traces/{id}/diagnose`: clustering runs
    over the whole accumulated corpus, not one trace (see
    `culprit.jobs.recluster_job`)."""
    logger.info("POST /recluster", extra={"event": "api_request", "route": "/recluster"})
    job_id = enqueue_recluster()
    return {"job_id": job_id}
