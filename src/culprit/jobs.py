"""RQ job functions run by the worker process, plus the trace-ingestion
entrypoint api.py and cli.py both call. Business logic lives in
pipeline.diagnose and cluster.cluster_diagnoses (the analysis-layer stubs
from Foundation task 8) and, for ingestion, in otlp/normalize/linearize/
store_traces (owned by WS-A and WS-B); every external seam here is
injectable so this whole module is fully testable today, before any of
those workstreams exist.

**Integration gap, flagged for task I2.** The five analysis layers were
Foundation-stubbed at a fixed signature so callers could compile against
them from day one; there is no equivalent stub for "assemble one OTLP
payload into a persisted Trace", "load a Trace by id", "persist a
Diagnosis", "read every diagnosis for reclustering", or "write a cluster
assignment back". `_seam()` lazily imports the real function by name and
raises a clearly-named error if it is not there yet, so `api.py`'s OTLP
endpoint and both RQ jobs are fully built and tested now; Integration only
supplies the real callables, not new orchestration.
"""

import logging
import os
from collections.abc import Callable

from culprit.cluster import cluster_diagnoses
from culprit.db import ConnFn
from culprit.embed import EmbedFn, embed_texts
from culprit.llm import CallFn, litellm_call
from culprit.logging_config import LOGGER_NAME
from culprit.pipeline import diagnose
from culprit.schemas import Trace
from culprit.signals import Diagnosis

logger = logging.getLogger(LOGGER_NAME)

DEFAULT_DATABASE_URL = "postgresql://localhost:5432/culprit"

TraceLoaderFn = Callable[[ConnFn, str], Trace]
DiagnosisWriterFn = Callable[[ConnFn, Diagnosis], None]
DiagnosesReaderFn = Callable[[ConnFn, str], list[Diagnosis]]
AllDiagnosesReaderFn = Callable[[ConnFn], list[Diagnosis]]
ClusterWriterFn = Callable[[ConnFn, dict], None]
IngestFn = Callable[[bytes, str, ConnFn], str]


class TraceNotFoundError(Exception):
    """Raised when a job or the ingest endpoint is given a trace_id that
    does not resolve to a persisted trace."""


class PersistenceNotWiredError(Exception):
    """Raised by `_seam` when the WS-A/WS-B module a default resolver needs
    is not importable yet: "not wired up" (expected during Phase 1) rather
    than an actual bug. Integration task I2 is "inject the real callable",
    not "add exception handling"."""


def _seam(module: str, attr: str, note: str = ""):
    """Import `attr` from `culprit.<module>` at call time, not at module
    import time, so `import culprit.jobs` never depends on a module owned
    by another still-in-flight workstream."""
    try:
        mod = __import__(f"culprit.{module}", fromlist=[attr])
        return getattr(mod, attr)
    except (ImportError, AttributeError) as e:
        raise PersistenceNotWiredError(
            f"culprit.{module}.{attr} is not available yet.{(' ' + note) if note else ''}"
        ) from e


def _conn_fn_from_env() -> ConnFn:
    """Lazily builds a psycopg ConnectionPool against CULPRIT_DATABASE_URL
    (or the hardcoded default), built on first call so `import culprit.jobs`
    never dials Postgres. Reads an env var rather than culprit.config
    directly (only cli.py/plugin registries read config, see CLAUDE.md);
    cli.py's callback sets this env var from CulpritConfig.database_url."""
    from culprit.db import make_pool

    dsn = os.environ.get("CULPRIT_DATABASE_URL", DEFAULT_DATABASE_URL)
    pool = make_pool(dsn)
    pool.open()
    return pool.getconn


def _default_trace_loader(conn_fn: ConnFn, trace_id: str) -> Trace:
    read_trace = _seam("store_traces", "read_trace", "(WS-B)")
    trace, _spans, _steps = read_trace(conn_fn, trace_id)
    if trace is None:
        raise TraceNotFoundError(f"no persisted trace with id {trace_id!r}")
    return trace


def _default_diagnosis_writer(conn_fn: ConnFn, diagnosis: Diagnosis) -> None:
    _seam("store_diagnoses", "write_diagnosis", "(WS-B)")(conn_fn, diagnosis)


def _default_diagnoses_for_trace_reader(
    conn_fn: ConnFn, trace_id: str
) -> list[Diagnosis]:
    return _seam("store_diagnoses", "read_diagnoses", "(WS-B)")(conn_fn, trace_id)


def _default_all_diagnoses_reader(conn_fn: ConnFn) -> list[Diagnosis]:
    """WS-B's documented contract only specifies `read_diagnoses`
    (per-trace); reclustering needs every diagnosis across the whole
    corpus, undocumented anywhere in the plan. `read_all_diagnoses` is
    this module's best guess, flagged for Integration to confirm/rename."""
    note = "(WS-B's contract has no bulk reader; inject diagnosis_reader)"
    return _seam("store_diagnoses", "read_all_diagnoses", note)(conn_fn)


def _default_cluster_writer(conn_fn: ConnFn, assignment: dict) -> None:
    """Same caveat: persisting a cluster_id per diagnosis_id has no
    documented WS-B function name either."""
    note = "(not part of WS-B's contract either; inject cluster_writer)"
    _seam("store_diagnoses", "write_cluster_assignments", note)(conn_fn, assignment)


def _default_ingest(payload: bytes, content_type: str, conn_fn: ConnFn) -> str:
    """No single WS-A/WS-B function assembles one OTLP payload into a
    persisted Trace end to end: needs otlp.decode, normalize.normalize_span
    per raw span, linearize.linearize, a Trace built from the result, and
    store_traces.write_trace, nowhere composed behind one callable in the
    plan. Left as an explicit gap rather than a guess at unspecified
    plumbing: inject `ingest_fn` until Integration wires this (task I2)."""
    raise PersistenceNotWiredError(
        "OTLP ingestion pipeline is not wired yet: needs otlp.decode, "
        "normalize.normalize_span, linearize.linearize, and "
        "store_traces.write_trace composed into one flow, which is "
        "Integration's job (task I2). Inject ingest_fn explicitly until then."
    )


def diagnose_trace_job(
    trace_id: str,
    *,
    conn_fn: ConnFn | None = None,
    call_fn: CallFn | None = None,
    embed_fn: EmbedFn | None = None,
    trace_loader: TraceLoaderFn = _default_trace_loader,
    diagnosis_writer: DiagnosisWriterFn = _default_diagnosis_writer,
) -> str:
    """RQ entrypoint for one trace's full diagnosis: resolve trace_id to a
    Trace, run pipeline.diagnose (Integration-filled L0-L3 cascade),
    persist the result, and return the new diagnosis_id. Every seam is
    keyword-injectable and defaulted, so this is testable with pipeline
    monkeypatched and no Postgres, Redis, or litellm dependency."""
    conn_fn = conn_fn or _conn_fn_from_env()
    call_fn = call_fn or litellm_call
    embed_fn = embed_fn or embed_texts

    logger.info("diagnose job started", extra={"event": "diagnose_job_started", "trace_id": trace_id})
    try:
        trace = trace_loader(conn_fn, trace_id)
        diagnosis = diagnose(trace, conn_fn=conn_fn, call_fn=call_fn, embed_fn=embed_fn)
        diagnosis_writer(conn_fn, diagnosis)
    except Exception:
        logger.exception("diagnose job failed", extra={"event": "diagnose_job_failed", "trace_id": trace_id})
        raise
    logger.info(
        "diagnose job completed",
        extra={"event": "diagnose_job_completed", "trace_id": trace_id, "diagnosis_id": diagnosis.diagnosis_id},
    )
    return diagnosis.diagnosis_id


def recluster_job(
    *,
    conn_fn: ConnFn | None = None,
    embed_fn: EmbedFn | None = None,
    diagnosis_reader: AllDiagnosesReaderFn = _default_all_diagnoses_reader,
    cluster_writer: ClusterWriterFn = _default_cluster_writer,
) -> dict:
    """RQ entrypoint for the scheduled batch reclustering pass (L5).
    Deliberately not on the per-trace path: clustering needs an
    accumulated corpus of diagnoses to be meaningful, so this runs on a
    schedule or on demand, never as a side effect of one diagnosis."""
    conn_fn = conn_fn or _conn_fn_from_env()
    embed_fn = embed_fn or embed_texts

    logger.info("recluster job started", extra={"event": "recluster_job_started"})
    try:
        diagnoses = diagnosis_reader(conn_fn)
        assignment = cluster_diagnoses(diagnoses, embed_fn=embed_fn)
        cluster_writer(conn_fn, assignment)
    except Exception:
        logger.exception("recluster job failed", extra={"event": "recluster_job_failed"})
        raise
    logger.info(
        "recluster job completed",
        extra={
            "event": "recluster_job_completed",
            "diagnosis_count": len(diagnoses),
            "cluster_count": len(set(assignment.values())),
        },
    )
    return assignment


def ingest_trace(
    payload: bytes,
    content_type: str,
    *,
    conn_fn: ConnFn | None = None,
    ingest_fn: IngestFn = _default_ingest,
) -> str:
    """Decode and persist one OTLP payload (protobuf or JSON, same
    endpoint either way), returning the new trace_id. Called from both
    api.py's `POST /v1/traces` and cli.py's `ingest` command so the two
    surfaces share one code path."""
    conn_fn = conn_fn or _conn_fn_from_env()
    logger.info(
        "ingest started",
        extra={"event": "ingest_started", "content_type": content_type, "byte_count": len(payload)},
    )
    trace_id = ingest_fn(payload, content_type, conn_fn)
    logger.info("ingest completed", extra={"event": "ingest_completed", "trace_id": trace_id})
    return trace_id


def read_diagnoses_for_trace(
    trace_id: str,
    *,
    conn_fn: ConnFn | None = None,
    diagnoses_reader: DiagnosesReaderFn = _default_diagnoses_for_trace_reader,
) -> list[Diagnosis]:
    """Every persisted diagnosis for one trace, newest alongside older
    ones (diagnoses.trace_id is deliberately not unique, see CLAUDE.md),
    used by `GET /traces/{id}/diagnoses` and `culprit show`."""
    conn_fn = conn_fn or _conn_fn_from_env()
    return diagnoses_reader(conn_fn, trace_id)
