"""Typer CLI: `culprit ingest|diagnose|show|job-status|recluster|serve|
worker`. Mirrors Litmus/src/litmus/cli.py's shape (single flat module,
config-driven option defaults, structured logging around every command,
`serve` launching uvicorn against the module-level FastAPI app) - see
Litmus/CLAUDE.md for why. This is the one module the plan explicitly
allows to run past the ~200 line ceiling if needed (Litmus's own cli.py is
383 lines), because `views.py` absorbs output shaping and `jobs.py`/
`queue.py` absorb orchestration.
"""

import logging
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
import typer
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from culprit.config import load_config
from culprit.jobs import PersistenceNotWiredError, ingest_trace, read_diagnoses_for_trace
from culprit.logging_config import LOGGER_NAME, configure_logging
from culprit.queue import enqueue_diagnosis, enqueue_recluster, fetch_job_status
from culprit.render import render_diagnosis, render_no_diagnoses
from culprit.views import diagnosis_detail_view, job_status_view

load_dotenv()
CONFIG = load_config()

app = typer.Typer()
logger = logging.getLogger(LOGGER_NAME)

# Quiet psycopg and psycopg_pool loggers so driver noise does not clutter
# terminal output on connection failure.
logging.getLogger("psycopg").setLevel(logging.CRITICAL)
logging.getLogger("psycopg.pool").setLevel(logging.CRITICAL)

# Shorten connection timeouts for CLI database operations so failures surface
# quickly (under 10 seconds) instead of waiting 30 seconds.
CLI_CONNECT_TIMEOUT_SECONDS = 4.0


def _default_env(key: str, default: str) -> None:
    """Like os.environ.setdefault, but an empty or whitespace-only value
    counts as unset too - see main()'s comment below for why."""
    if not os.environ.get(key, "").strip():
        os.environ[key] = default


_default_env("PGCONNECT_TIMEOUT", "4")

_orig_pool_init = ConnectionPool.__init__


def _cli_pool_init(self, *args, **kwargs):
    if "timeout" not in kwargs or kwargs["timeout"] == 30.0:
        kwargs["timeout"] = CLI_CONNECT_TIMEOUT_SECONDS
    return _orig_pool_init(self, *args, **kwargs)


ConnectionPool.__init__ = _cli_pool_init

# psycopg.OperationalError covers psycopg_pool.PoolTimeout (its subclass),
# raised when a pool cannot open against Postgres; the two redis errors
# cover a refused or timed-out connection to Redis.
_CONNECTION_ERRORS = (psycopg.OperationalError, RedisConnectionError, RedisTimeoutError)


def _redact_dsn(dsn: str) -> str:
    """Drop the password and any query string from a connection string, so
    what reaches a terminal or a log file names the destination without the
    credential that opens it. A DSN carries `user:password@host`, and this
    message is echoed by every command below, so an unredacted one prints
    the database password to whoever is watching a failed run. A value that
    does not parse as a URL (psycopg also accepts `host=... password=...`
    keyword form) is reported by shape rather than by content, so a
    malformed string cannot leak through the fallback either."""
    try:
        parts = urlsplit(dsn)
    except ValueError:
        return "<unparseable connection string>"
    if not parts.hostname:
        return "<unparseable connection string>"
    netloc = parts.hostname
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    if parts.username or parts.password:
        user = parts.username or ""
        netloc = f"{user}:***@{netloc}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _connection_error_message(e: Exception) -> str:
    """One line naming the destination culprit resolved, with the password
    redacted, and the two places to check against a connection failure:
    .env (is the value actually set) and compose.yaml (is the service
    actually running). Stands in for psycopg's or redis's own exception,
    which for a Postgres pool that cannot connect is a bare PoolTimeout
    after a short wait, with a driver traceback underneath it that
    names nothing a reader can act on."""
    if isinstance(e, psycopg.OperationalError):
        dsn = os.environ.get("CULPRIT_DATABASE_URL", CONFIG.database_url)
        return f"could not reach Postgres at {_redact_dsn(dsn)}. Check .env and compose.yaml."
    dsn = os.environ.get("CULPRIT_REDIS_URL", CONFIG.redis_url)
    return f"could not reach Redis at {_redact_dsn(dsn)}. Check .env and compose.yaml."


def _check_gemini_api_key() -> None:
    """Ensure GEMINI_API_KEY is present and non-empty.

    Commands reaching an LLM call this check at the CLI boundary. If the key is
    missing, exit nonzero naming the variable and .env.example.
    """
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        typer.echo(
            "Error: GEMINI_API_KEY is unset or empty. "
            "Set GEMINI_API_KEY in .env before running this command. "
            "See .env.example."
        )
        raise typer.Exit(code=1)


@app.callback()
def main(
    log_file: Path = typer.Option(
        CONFIG.log_file,
        "--log-file",
        envvar="CULPRIT_LOG_FILE",
        help="Structured JSON-lines log file path",
    ),
    log_level: str = typer.Option(
        CONFIG.log_level, "--log-level", envvar="CULPRIT_LOG_LEVEL", help="Logging level"
    ),
) -> None:
    """culprit: not where it failed, where it broke."""
    configure_logging(
        log_file,
        log_level,
        max_bytes=CONFIG.log_max_bytes,
        backup_count=CONFIG.log_backup_count,
    )
    # Propagated via env var, not a direct import, into jobs.py/queue.py:
    # both read CULPRIT_DATABASE_URL/CULPRIT_REDIS_URL rather than
    # importing culprit.config, per CLAUDE.md's rule that only cli.py and
    # plugin registries read config. An explicit, non-empty value (or a
    # test's monkeypatch.setenv) is never clobbered; an unset, empty, or
    # whitespace-only value falls back to the config default instead - the
    # README's `cp .env.example .env && set -a && . ./.env` step exports
    # every key present but blank, and plain setdefault would have honored
    # that blank as "explicitly set to nothing" rather than falling back.
    _default_env("CULPRIT_DATABASE_URL", CONFIG.database_url)
    _default_env("CULPRIT_REDIS_URL", CONFIG.redis_url)
    _default_env("CULPRIT_MODEL", CONFIG.model)
    _default_env("PGCONNECT_TIMEOUT", "4")
    logging.getLogger("psycopg").setLevel(logging.CRITICAL)
    logging.getLogger("psycopg.pool").setLevel(logging.CRITICAL)


@app.command(short_help="Ingest one OTLP trace file into Postgres.")
def ingest(
    trace_file: Path = typer.Argument(..., help="OTLP JSON or protobuf file to ingest"),
) -> None:
    """Ingest one OTLP trace file directly, without going through the HTTP
    endpoint. Content type is inferred from the file extension: `.json` is
    application/json, anything else is treated as application/x-protobuf."""
    content_type = (
        "application/json" if trace_file.suffix == ".json" else "application/x-protobuf"
    )
    payload = trace_file.read_bytes()
    try:
        trace_id = ingest_trace(payload, content_type)
    except PersistenceNotWiredError as e:
        logger.error("ingest failed", extra={"event": "ingest_failed", "error": str(e)})
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1) from e
    except _CONNECTION_ERRORS as e:
        typer.echo(f"Error: {_connection_error_message(e)}")
        raise typer.Exit(code=1) from e
    typer.echo(f"Ingested trace {trace_id}")


def _step_context_for(view: dict) -> list[dict] | None:
    """The root-cause step and its neighbours, or None if they cannot be
    read. A verdict is still worth printing when the surrounding steps are
    gone (pruned trace, missing rows, an unreachable pool after the
    diagnoses were already fetched), so every failure here degrades to None
    rather than turning a readable answer into an error."""
    if view.get("root_cause_step_index") is None:
        return None
    from culprit.bench import pooled_conn_fn
    from culprit.store_traces_query import step_context

    close = None
    try:
        conn_fn, _recycle, close = pooled_conn_fn()
        return step_context(conn_fn, view["trace_id"], view["root_cause_step_index"])
    except Exception:  # noqa: BLE001 - context is a nicety, never the answer
        return None
    finally:
        if close is not None:
            close()


@app.command(
    short_help="Ingest, diagnose, and print the verdict in one command. No worker needed."
)
def run(
    trace_file: Path | None = typer.Argument(
        None, help="OTLP JSON or protobuf file to ingest first"
    ),
    trace_id: str | None = typer.Option(
        None, "--trace-id", help="Diagnose a trace that is already ingested"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show every signal rather than the first few"
    ),
) -> None:
    """Ingest a trace file (or take one already ingested), diagnose it
    inline, and print the verdict.

    The same work `culprit diagnose` queues, run in this process instead.
    That removes Redis and the worker from the path, which matters because
    `culprit worker` cannot run on Windows at all (RQ forks; see CLAUDE.md),
    and because a queued job that nobody is listening for looks identical to
    a job that is about to run. Needs Postgres and GEMINI_API_KEY.
    """
    if (trace_file is None) == (trace_id is None):
        typer.echo("Error: pass either a trace file or --trace-id, not both or neither.")
        raise typer.Exit(code=1)

    _check_gemini_api_key()

    if trace_file is not None:
        content_type = (
            "application/json" if trace_file.suffix == ".json" else "application/x-protobuf"
        )
        try:
            trace_id = ingest_trace(trace_file.read_bytes(), content_type)
        except PersistenceNotWiredError as e:
            typer.echo(f"Error: {e}")
            raise typer.Exit(code=1) from e
        except _CONNECTION_ERRORS as e:
            typer.echo(f"Error: {_connection_error_message(e)}")
            raise typer.Exit(code=1) from e
        typer.echo(f"Ingested trace {trace_id}")

    from culprit.jobs import diagnose_trace_job

    typer.echo(f"Diagnosing {trace_id}...")
    try:
        diagnosis_id = diagnose_trace_job(trace_id)
    except PersistenceNotWiredError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1) from e
    except _CONNECTION_ERRORS as e:
        typer.echo(f"Error: {_connection_error_message(e)}")
        raise typer.Exit(code=1) from e

    diagnoses = read_diagnoses_for_trace(trace_id)
    fresh = [d for d in diagnoses if d.diagnosis_id == diagnosis_id] or diagnoses[-1:]
    for diagnosis in fresh:
        view = diagnosis_detail_view(diagnosis)
        render_diagnosis(view, context=_step_context_for(view), verbose=verbose)


@app.command(short_help="Queue a diagnosis for an ingested trace (needs a worker).")
def diagnose(trace_id: str = typer.Argument(...)) -> None:
    """Enqueue a diagnosis job for an already-ingested trace and print the
    job id. Poll it with `culprit job-status <job_id>` or read the result
    with `culprit show <trace_id>` once it completes."""
    _check_gemini_api_key()
    try:
        job_id = enqueue_diagnosis(trace_id)
    except _CONNECTION_ERRORS as e:
        typer.echo(f"Error: {_connection_error_message(e)}")
        raise typer.Exit(code=1) from e
    logger.info(
        "diagnose enqueued",
        extra={"event": "diagnose_enqueued", "trace_id": trace_id, "job_id": job_id},
    )
    typer.echo(f"Enqueued job {job_id} for trace {trace_id}")
    # Enqueueing succeeds whether or not anything is listening, so without
    # this the command looks like it worked and then nothing ever happens.
    typer.echo(
        "A worker has to be running to pick this up: `culprit worker` "
        "(Linux or WSL; see CLAUDE.md).\n"
        f"To diagnose without a worker or Redis: `culprit run --trace-id {trace_id}`."
    )


@app.command(short_help="Print the diagnoses culprit has for a trace.")
def show(
    trace_id: str = typer.Argument(...),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show every signal rather than the first few"
    ),
) -> None:
    """Print every persisted diagnosis for a trace: the verdict, the
    candidate shortlist L3 adjudicated, and the L1 signals behind it."""
    try:
        diagnoses = read_diagnoses_for_trace(trace_id)
    except PersistenceNotWiredError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1) from e
    except _CONNECTION_ERRORS as e:
        typer.echo(f"Error: {_connection_error_message(e)}")
        raise typer.Exit(code=1) from e
    if not diagnoses:
        render_no_diagnoses(trace_id)
        raise typer.Exit(code=1)
    for diagnosis in diagnoses:
        view = diagnosis_detail_view(diagnosis)
        render_diagnosis(view, context=_step_context_for(view), verbose=verbose)


@app.command("job-status", short_help="Poll one queued job.")
def job_status(job_id: str = typer.Argument(...)) -> None:
    """Poll one job's status (queued/started/finished/failed)."""
    try:
        status = fetch_job_status(job_id)
    except _CONNECTION_ERRORS as e:
        typer.echo(f"Error: {_connection_error_message(e)}")
        raise typer.Exit(code=1) from e
    view = job_status_view(status)
    typer.echo(
        f"[{view['status']}] {view['job_id']}: "
        f"result={view['result']} error={view['error']}"
    )


@app.command(short_help="Queue the batch reclustering pass over accumulated diagnoses.")
def recluster() -> None:
    """Enqueue the scheduled batch reclustering pass. Clustering runs over
    accumulated diagnoses, never per trace, so this is triggered on
    demand or by an external scheduler rather than automatically after
    every diagnosis (see `culprit.jobs.recluster_job`)."""
    try:
        job_id = enqueue_recluster()
    except _CONNECTION_ERRORS as e:
        typer.echo(f"Error: {_connection_error_message(e)}")
        raise typer.Exit(code=1) from e
    typer.echo(f"Enqueued recluster job {job_id}")


@app.command(short_help="Score a benchmark (trail or who_and_when) end to end.")
def bench(
    benchmark: str = typer.Argument(..., help="Benchmark name: trail or who_and_when"),
    data: Path = typer.Option(..., "--data", help="Merged benchmark JSON file"),
    sample: int | None = typer.Option(
        None, "--sample", help="Score only N evenly-spaced traces (deterministic)"
    ),
    ablate: str | None = typer.Option(None, "--ablate", help="Disable a layer: l2"),
) -> None:
    """Run the benchmark harness end to end (Integration task I5): persist
    each benchmark trace through the production tables, diagnose inline (no
    RQ worker, which cannot run on Windows), and print the scored report -
    every tolerance band, never only the flattering one."""
    _check_gemini_api_key()
    from culprit.bench import pooled_conn_fn, run_benchmark
    from culprit.embed import embed_texts
    from culprit.llm import litellm_call

    try:
        conn_fn, recycle, close = pooled_conn_fn()
    except _CONNECTION_ERRORS as e:
        typer.echo(f"Error: {_connection_error_message(e)}")
        raise typer.Exit(code=1) from e
    try:
        run = run_benchmark(
            benchmark, data, sample=sample, ablate=ablate,
            conn_fn=conn_fn, call_fn=litellm_call, embed_fn=embed_texts,
            model=os.environ.get("CULPRIT_MODEL", CONFIG.model),
            recycle_fn=recycle,
        )
    finally:
        close()
    for label, report in (("all annotated errors", run.report_all), ("earliest error only", run.report_primary)):
        typer.echo(f"== {label} (n_cases={report.n_cases}, traces={run.n_traces}) ==")
        typer.echo(f"  abstention_rate={report.abstention_rate:.3f}")
        typer.echo(f"  exact_accuracy={report.exact_accuracy:.3f}  joint_accuracy={report.joint_accuracy:.3f}")
        bands = "  ".join(f"@{k}={v:.3f}" for k, v in sorted(report.tolerance_accuracy.items()))
        typer.echo(f"  tolerance_accuracy: {bands}")
        typer.echo(
            f"  class_accuracy={report.class_accuracy if report.class_accuracy is None else f'{report.class_accuracy:.3f}'}"
            f"  span_accuracy={report.span_accuracy if report.span_accuracy is None else f'{report.span_accuracy:.3f}'}"
        )
        typer.echo(f"  earliness_error={report.earliness_error:.3f}")
        recall = "  ".join(f"@{k}={v:.3f}" for k, v in sorted(report.candidate_recall_at_k.items()))
        typer.echo(f"  candidate_recall: {recall}")
        typer.echo(
            f"  brier={report.brier_score if report.brier_score is None else f'{report.brier_score:.3f}'}"
            f"  ece={report.ece if report.ece is None else f'{report.ece:.3f}'}"
        )
    typer.echo(f"total LLM cost: ${run.total_cost_usd:.4f}")
    if run.per_trace_errors:
        typer.echo(f"traces failed (scored as abstained): {len(run.per_trace_errors)}")
        for trace_id, error in run.per_trace_errors.items():
            typer.echo(f"  {trace_id}: {error}")


@app.command(short_help="Run the HTTP service.")
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Launch the FastAPI service (OTLP ingest, diagnose, job status)."""
    import uvicorn

    uvicorn.run("culprit.api:app", host=host, port=port)


@app.command(short_help="Run one queue worker (Linux or WSL only).")
def worker() -> None:
    """Launch one RQ worker process, pulling from the same queue
    `culprit diagnose`/`culprit recluster` enqueue onto."""
    _check_gemini_api_key()
    from culprit.worker import run_worker

    run_worker()


if __name__ == "__main__":
    app()
