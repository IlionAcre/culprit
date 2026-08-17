"""RQ job functions run by the worker process, plus the trace-ingestion
entrypoint api.py and cli.py both call. Business logic lives in
pipeline.diagnose and cluster.cluster_diagnoses, and, for ingestion, in
ingest.ingest_payload (Integration task I2, composing otlp/normalize/
linearize/store_traces, owned by WS-A and WS-B); every external seam here
is injectable so this whole module is fully testable with no Postgres,
Redis, or litellm dependency.

**Integration task I2, resolved.** The five analysis layers and the
persistence functions were Foundation/WS-B-stubbed or -built at a fixed
signature so callers could compile against them from day one; `_seam()`
lazily imports the real function by name and raises a clearly-named error
if it is not there yet. Every default seam below now resolves to a real
implementation: `pipeline.diagnose` (I1), `store_traces.read_trace`/
`write_diagnosis`/`read_all_diagnoses` (WS-B), `store_clusters.
write_cluster_assignments` (WS-F, repointed here per INTEGRATION_ITEMS.md
item 4), and `ingest.ingest_payload` (this task, `_default_ingest` below).
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
# Mirrors CulpritConfig.model's own hardcoded default (config.py). Read from
# an env var rather than culprit.config directly, same reasoning as
# _conn_fn_from_env: only cli.py/plugin registries import config, and
# pipeline.diagnose now requires a model string (see pipeline.py's
# Foundation-amendment docstring).
DEFAULT_MODEL = "gemini/gemini-2.5-flash-lite"

TraceLoaderFn = Callable[[ConnFn, str], Trace]
DiagnosisWriterFn = Callable[[ConnFn, Diagnosis], None]
DiagnosesReaderFn = Callable[[ConnFn, str], list[Diagnosis]]
AllDiagnosesReaderFn = Callable[[ConnFn], list[Diagnosis]]
# Returns `label -> cluster_id`, not None (INTEGRATION_ITEMS.md item 1): the
# stable identity store_clusters.resolve_cluster_identity resolved this pass,
# which cluster_labeler below needs to read/write the right `clusters` rows.
ClusterWriterFn = Callable[[ConnFn, dict], dict]
ClusterLabelerFn = Callable[..., list]
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


def _conn_fn_from_env() -> tuple[ConnFn, Callable[[], None]]:
    """Lazily builds a psycopg ConnectionPool against CULPRIT_DATABASE_URL
    (or the hardcoded default), built on first call so `import culprit.jobs`
    never dials Postgres. Reads an env var rather than culprit.config
    directly (only cli.py/plugin registries read config, see CLAUDE.md);
    cli.py's callback sets this env var from CulpritConfig.database_url.

    Returns `(conn_fn, close)`, mirroring `bench.pooled_conn_fn`'s fix for
    the identical bug: store-layer functions check a connection OUT via
    `conn_fn()` and never return it (store_traces.py's "callers own
    pooling" contract), and with this project's psycopg_pool version
    `conn.close()` does not return a checked-out connection to the pool -
    only `pool.putconn(conn)` does (verified empirically in bench.py; see
    its docstring). Unlike `bench.py`, which amortizes one pool across an
    entire run and recycles per-trace, each job call here builds its own
    fresh pool for just that one call, so there is nothing to recycle
    mid-job - `close()` simply putconns every connection this job issued
    once its work is done (success or exception), then closes the pool.
    `max_size=32` mirrors `bench.pooled_conn_fn`'s override of `make_pool`'s
    default of 10: `diagnose_trace_job` alone can issue ~28 checkouts via
    L2's up-to-`contrast.MAX_REFERENCES` reference loads plus its own write
    and read paths, so the default is too small for one job's worst case."""
    from culprit.db import make_pool

    dsn = os.environ.get("CULPRIT_DATABASE_URL", DEFAULT_DATABASE_URL)
    pool = make_pool(dsn, max_size=32)
    pool.open()
    issued: list = []

    def conn_fn():
        conn = pool.getconn()
        # See pooled_conn_fn: autocommit avoids putconn rolling back an
        # INTRANS read connection noisily on close; explicit `with
        # conn.transaction():` write blocks (store_traces.py) are
        # unaffected by autocommit.
        conn.autocommit = True
        issued.append(conn)
        return conn

    def close() -> None:
        for conn in issued:
            pool.putconn(conn)
        issued.clear()
        pool.close()

    return conn_fn, close


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


def _default_cluster_writer(conn_fn: ConnFn, assignment: dict) -> dict:
    """INTEGRATION_ITEMS.md item 4: the real function lives in
    store_clusters.py (split out of store_diagnoses.py once that module hit
    the line ceiling), not store_diagnoses.py. This seam now points there
    directly; store_diagnoses.write_cluster_assignments was a temporary
    re-export kept only so this seam name kept resolving, now deleted.

    Returns the `label -> cluster_id` map (item 1's fix: identity is now
    resolved, not re-minted every pass), which `recluster_job` forwards to
    the cluster labeler."""
    note = "(not part of WS-B's original contract; inject cluster_writer)"
    return _seam("store_clusters", "write_cluster_assignments", note)(conn_fn, assignment)


def _default_cluster_labeler(
    conn_fn: ConnFn,
    diagnoses: list[Diagnosis],
    assignment: dict,
    cluster_ids: dict,
    *,
    embed_fn: EmbedFn,
    model: str,
) -> list:
    """INTEGRATION_ITEMS.md item 3: nothing wrote `clusters.label`/
    `description`/`suggested_fix`/`medoid_diagnosis_id`/`labeled_size`
    anywhere in the codebase before this. Wired here, not part of any
    workstream's original contract."""
    note = "(item 3: cluster label persistence; inject cluster_labeler)"
    return _seam("store_cluster_labels", "label_and_persist_clusters", note)(
        conn_fn, diagnoses, assignment, cluster_ids, embed_fn=embed_fn, model=model
    )


def _default_ingest(payload: bytes, content_type: str, conn_fn: ConnFn) -> str:
    """Integration task I2, resolved: composes otlp.decode (or the direct
    JSON upload envelope), normalize.normalize_span, linearize.linearize,
    and store_traces.write_trace via ingest.ingest_payload - see that
    module's docstring for why the composition needed its own module
    rather than living inline here. embed_texts is passed through so a
    real ingest also gets `task_embedding` populated, not just spans/steps.
    """
    from culprit.ingest import ingest_payload

    return ingest_payload(payload, content_type, conn_fn, embed_fn=embed_texts)


def diagnose_trace_job(
    trace_id: str,
    *,
    conn_fn: ConnFn | None = None,
    call_fn: CallFn | None = None,
    embed_fn: EmbedFn | None = None,
    model: str | None = None,
    trace_loader: TraceLoaderFn = _default_trace_loader,
    diagnosis_writer: DiagnosisWriterFn = _default_diagnosis_writer,
) -> str:
    """RQ entrypoint for one trace's full diagnosis: resolve trace_id to a
    Trace, run pipeline.diagnose (Integration-filled L0-L3 cascade),
    persist the result, and return the new diagnosis_id. Every seam is
    keyword-injectable and defaulted, so this is testable with pipeline
    monkeypatched and no Postgres, Redis, or litellm dependency.

    `model` defaults from CULPRIT_MODEL (set by cli.py's callback from
    CulpritConfig.model, same pattern as CULPRIT_DATABASE_URL/
    CULPRIT_REDIS_URL), falling back to DEFAULT_MODEL so a worker started
    without going through the CLI callback still has a usable default."""
    owns_conn = conn_fn is None
    close_conn = None
    if owns_conn:
        conn_fn, close_conn = _conn_fn_from_env()
    call_fn = call_fn or litellm_call
    embed_fn = embed_fn or embed_texts
    model = model or os.environ.get("CULPRIT_MODEL", DEFAULT_MODEL)

    logger.info("diagnose job started", extra={"event": "diagnose_job_started", "trace_id": trace_id})
    try:
        trace = trace_loader(conn_fn, trace_id)
        diagnosis = diagnose(
            trace, conn_fn=conn_fn, call_fn=call_fn, embed_fn=embed_fn, model=model
        )
        diagnosis_writer(conn_fn, diagnosis)
    except Exception:
        # logger.exception(...): JsonFormatter now reads record.exc_info
        # (logging_config.py's Foundation amendment), so the traceback lands
        # in the log line without hand-formatting it here. RQ's own
        # exc_string (see queue.py) is the fallback if a job dies before
        # reaching this except block.
        logger.exception(
            "diagnose job failed",
            extra={"event": "diagnose_job_failed", "trace_id": trace_id},
        )
        raise
    finally:
        # Releases every connection this job checked out (see
        # _conn_fn_from_env's docstring), success or exception alike, so a
        # raising job still returns its connections to the pool instead of
        # leaking them - not run when the caller injected its own conn_fn,
        # since that caller owns that connection's lifecycle.
        if close_conn is not None:
            close_conn()
    logger.info(
        "diagnose job completed",
        extra={"event": "diagnose_job_completed", "trace_id": trace_id, "diagnosis_id": diagnosis.diagnosis_id},
    )
    return diagnosis.diagnosis_id


def recluster_job(
    *,
    conn_fn: ConnFn | None = None,
    embed_fn: EmbedFn | None = None,
    model: str | None = None,
    diagnosis_reader: AllDiagnosesReaderFn = _default_all_diagnoses_reader,
    cluster_writer: ClusterWriterFn = _default_cluster_writer,
    cluster_labeler: ClusterLabelerFn = _default_cluster_labeler,
) -> dict:
    """RQ entrypoint for the scheduled batch reclustering pass (L5).
    Deliberately not on the per-trace path: clustering needs an
    accumulated corpus of diagnoses to be meaningful, so this runs on a
    schedule or on demand, never as a side effect of one diagnosis.

    Labeling (INTEGRATION_ITEMS.md item 3) now runs as the last step of this
    same job rather than nowhere: `cluster_writer` resolves stable
    `cluster_id`s (item 1), and `cluster_labeler` reads each cluster's prior
    state through that same stable identity so `needs_relabel`'s skip has
    something real to compare against on the next pass."""
    owns_conn = conn_fn is None
    close_conn = None
    if owns_conn:
        conn_fn, close_conn = _conn_fn_from_env()
    embed_fn = embed_fn or embed_texts
    model = model or os.environ.get("CULPRIT_MODEL", DEFAULT_MODEL)

    logger.info("recluster job started", extra={"event": "recluster_job_started"})
    try:
        diagnoses = diagnosis_reader(conn_fn)
        assignment = cluster_diagnoses(diagnoses, embed_fn=embed_fn)
        cluster_ids = cluster_writer(conn_fn, assignment)
        labels = cluster_labeler(
            conn_fn, diagnoses, assignment, cluster_ids, embed_fn=embed_fn, model=model
        )
    except Exception:
        # See diagnose_trace_job's except block: logger.exception now works
        # (JsonFormatter reads record.exc_info), no manual traceback needed.
        logger.exception(
            "recluster job failed",
            extra={"event": "recluster_job_failed"},
        )
        raise
    finally:
        # See diagnose_trace_job's finally block.
        if close_conn is not None:
            close_conn()
    logger.info(
        "recluster job completed",
        extra={
            "event": "recluster_job_completed",
            "diagnosis_count": len(diagnoses),
            "cluster_count": len(set(assignment.values())),
            "labeled_count": len(labels),
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
    owns_conn = conn_fn is None
    close_conn = None
    if owns_conn:
        conn_fn, close_conn = _conn_fn_from_env()
    logger.info(
        "ingest started",
        extra={"event": "ingest_started", "content_type": content_type, "byte_count": len(payload)},
    )
    try:
        trace_id = ingest_fn(payload, content_type, conn_fn)
    finally:
        # See diagnose_trace_job's finally block.
        if close_conn is not None:
            close_conn()
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
    owns_conn = conn_fn is None
    close_conn = None
    if owns_conn:
        conn_fn, close_conn = _conn_fn_from_env()
    try:
        return diagnoses_reader(conn_fn, trace_id)
    finally:
        # See diagnose_trace_job's finally block.
        if close_conn is not None:
            close_conn()
