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

import typer
from dotenv import load_dotenv

from culprit.config import load_config
from culprit.jobs import PersistenceNotWiredError, ingest_trace, read_diagnoses_for_trace
from culprit.logging_config import LOGGER_NAME, configure_logging
from culprit.queue import enqueue_diagnosis, enqueue_recluster, fetch_job_status
from culprit.views import diagnosis_detail_view, job_status_view

load_dotenv()
CONFIG = load_config()

app = typer.Typer()
logger = logging.getLogger(LOGGER_NAME)


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
    # plugin registries read config. setdefault so an explicit env var
    # (or a test's monkeypatch.setenv) is never clobbered.
    os.environ.setdefault("CULPRIT_DATABASE_URL", CONFIG.database_url)
    os.environ.setdefault("CULPRIT_REDIS_URL", CONFIG.redis_url)


@app.command()
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
    typer.echo(f"Ingested trace {trace_id}")


@app.command()
def diagnose(trace_id: str = typer.Argument(...)) -> None:
    """Enqueue a diagnosis job for an already-ingested trace and print the
    job id. Poll it with `culprit job-status <job_id>` or read the result
    with `culprit show <trace_id>` once it completes."""
    job_id = enqueue_diagnosis(trace_id)
    logger.info(
        "diagnose enqueued",
        extra={"event": "diagnose_enqueued", "trace_id": trace_id, "job_id": job_id},
    )
    typer.echo(f"Enqueued job {job_id} for trace {trace_id}")


@app.command()
def show(trace_id: str = typer.Argument(...)) -> None:
    """Print every persisted diagnosis for a trace."""
    try:
        diagnoses = read_diagnoses_for_trace(trace_id)
    except PersistenceNotWiredError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1) from e
    if not diagnoses:
        typer.echo(f"No diagnoses yet for trace {trace_id}")
        raise typer.Exit(code=1)
    for diagnosis in diagnoses:
        view = diagnosis_detail_view(diagnosis)
        typer.echo(
            f"[{view['diagnosis_id']}] step={view['root_cause_step_index']} "
            f"class={view['failure_class']} "
            f"confidence={view['calibrated_confidence']:.2f} "
            f"abstained={view['abstained']}"
        )
        typer.echo(f"  {view['rationale']}")


@app.command("job-status")
def job_status(job_id: str = typer.Argument(...)) -> None:
    """Poll one job's status (queued/started/finished/failed)."""
    status = fetch_job_status(job_id)
    view = job_status_view(status)
    typer.echo(
        f"[{view['status']}] {view['job_id']}: "
        f"result={view['result']} error={view['error']}"
    )


@app.command()
def recluster() -> None:
    """Enqueue the scheduled batch reclustering pass. Clustering runs over
    accumulated diagnoses, never per trace, so this is triggered on
    demand or by an external scheduler rather than automatically after
    every diagnosis (see `culprit.jobs.recluster_job`)."""
    job_id = enqueue_recluster()
    typer.echo(f"Enqueued recluster job {job_id}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Launch the FastAPI service (OTLP ingest, diagnose, job status)."""
    import uvicorn

    uvicorn.run("culprit.api:app", host=host, port=port)


@app.command()
def worker() -> None:
    """Launch one RQ worker process, pulling from the same queue
    `culprit diagnose`/`culprit recluster` enqueue onto."""
    from culprit.worker import run_worker

    run_worker()


if __name__ == "__main__":
    app()
