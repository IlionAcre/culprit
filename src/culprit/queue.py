"""Redis + RQ queue wiring: one queue, `enqueue_diagnosis`/
`enqueue_recluster` to put work on it, `fetch_job_status` to read it back.
Analysis takes minutes per trace and cannot run inside a request (see
CLAUDE.md), so this module is the whole handoff between api.py/cli.py and
the separate `worker.py` process. `fakeredis` (a declared dev dependency)
lets every test in this file run with no real Redis, matching the
project's offline-tests guarantee: `get_queue`/`fetch_job_status` both take
an injectable connection for exactly that reason.

**Information disclosure, closed here.** A failed job's `exc_string` is a
full Python traceback: filesystem paths, module structure, and whatever a
failing call embedded in its exception text - in this system that plausibly
includes a Postgres DSN from a psycopg connection error, a Redis URL from a
connection failure, or request/response fragments from a litellm error.
`GET /jobs/{id}` (api.py) returns `fetch_job_status`'s dict verbatim, so
that traceback would otherwise reach any caller of the API. `fetch_job_status`
never puts `exc_string` in its return value; the full detail is logged once,
correlated by job_id, and the caller gets a generic message plus the job_id
it already has, enough to know the job failed and to quote in a support
request. Mirrors Litmus's own convention in both directions: its CLI tests
assert `"Traceback" not in result.stdout`, and `Portfolio-WebPage/main.py`
pairs `logger.exception(...)` with a generic user-facing message.
"""

import logging
import os

import redis
from rq import Queue
from rq.job import Job

from culprit.jobs import diagnose_trace_job, recluster_job
from culprit.logging_config import LOGGER_NAME

QUEUE_NAME = "culprit"
DEFAULT_REDIS_URL = "redis://localhost:6379/0"
logger = logging.getLogger(LOGGER_NAME)


def _redis_connection() -> redis.Redis:
    """Reads CULPRIT_REDIS_URL rather than culprit.config directly, per
    CLAUDE.md's rule that only cli.py and plugin registries read config;
    cli.py's callback sets this env var from CulpritConfig.redis_url
    before any command runs, mirroring Litmus's LITMUS_RUNS_DIR pattern."""
    url = os.environ.get("CULPRIT_REDIS_URL", DEFAULT_REDIS_URL)
    return redis.from_url(url)


def get_queue(*, connection: redis.Redis | None = None) -> Queue:
    return Queue(QUEUE_NAME, connection=connection or _redis_connection())


def enqueue_diagnosis(trace_id: str, *, queue: Queue | None = None) -> str:
    """Enqueue one trace's full diagnosis. Returns the RQ job id, which
    `fetch_job_status` (or `GET /jobs/{id}`) can then poll."""
    q = queue or get_queue()
    job = q.enqueue(diagnose_trace_job, trace_id)
    return job.id


def enqueue_recluster(*, queue: Queue | None = None) -> str:
    """Enqueue one batch reclustering pass. Deliberately not enqueued as a
    side effect of `enqueue_diagnosis` (see jobs.recluster_job's
    docstring): callers trigger this on a schedule or on demand."""
    q = queue or get_queue()
    job = q.enqueue(recluster_job)
    return job.id


def fetch_job_status(job_id: str, *, connection: redis.Redis | None = None) -> dict:
    """Raises `rq.exceptions.NoSuchJobError` for an unknown job id;
    api.py maps that straight to a 404 rather than this module wrapping
    it in a second exception type. Uses `return_value()`/`latest_result()`
    rather than the deprecated `job.result`/`job.exc_info` properties.

    Deliberately never returns `latest.exc_string` (see module docstring):
    that's a full traceback that can embed a DSN, a Redis URL, or a
    litellm request/response fragment, and this dict is what `GET
    /jobs/{id}` sends straight to an HTTP caller. The real detail is
    logged once here, correlated by job_id, for an operator working from a
    support request; the caller only learns that the job failed."""
    job = Job.fetch(job_id, connection=connection or _redis_connection())
    latest = job.latest_result()
    error = None
    if latest is not None and latest.exc_string:
        logger.error(
            "job failed",
            extra={"event": "job_failed", "job_id": job.id, "traceback": latest.exc_string},
        )
        error = f"job {job.id} failed; see server logs for the underlying error"
    return {
        "job_id": job.id,
        "status": job.get_status(),
        "result": job.return_value(refresh=True),
        "error": error,
    }
