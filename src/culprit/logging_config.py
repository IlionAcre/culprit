import json
import logging
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "culprit"

# Computed from a blank LogRecord rather than hand-maintained, so it stays
# correct across Python versions (e.g. 3.12 added "taskName").
_STANDARD_LOG_RECORD_ATTRS = frozenset(logging.makeLogRecord({}).__dict__.keys())


class JsonFormatter(logging.Formatter):
    """Structured JSON Lines formatter: one JSON object per log line, with
    whatever extra fields the call site passed via `extra={...}` merged in.

    **Foundation amendment (post-Phase-0): captures `record.exc_info`.**
    Before this, `logger.exception(...)` (and any `logger.error(...,
    exc_info=True)`) appeared to succeed - the log line was written - but the
    traceback was silently absent, because this formatter never read
    `exc_info` at all. Found empirically by WS-G while verifying a security
    fix actually worked. This is a strict addition (a new `traceback` field
    only appears when `exc_info` is present) and changes no existing
    contract. See `AI_docs/PHASES.md`'s "TRAP every remaining workstream will
    hit" note and `test_logging_config.py` for the pinning test.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_LOG_RECORD_ATTRS
        }
        payload.update(extra)
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        # default=str: extra fields aren't guaranteed JSON-native (e.g. a
        # Path passed as trace_dir) - a log call must never crash a run.
        return json.dumps(payload, default=str)


def configure_logging(
    log_file: Path | str,
    level: str = "INFO",
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
) -> None:
    """(Re)configure the shared "culprit" logger to write structured JSON
    Lines to a rotating file.

    Safe to call repeatedly within the same process - the CLI's top-level
    callback re-runs this on every invocation, including every test in a
    pytest session - because it always closes and removes any handlers
    already attached first. Without that, handlers would accumulate
    (duplicate log lines) and stale handlers would keep file handles open
    on already-torn-down tmp_path directories (a real problem on Windows).
    """
    logger = logging.getLogger(LOGGER_NAME)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    logger.setLevel(level)
    logger.propagate = False

    log_file = Path(log_file)
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    except OSError as e:
        # Logging failures must never abort the actual ingest/diagnose/serve
        # operation - degrade to no file logging instead.
        import typer

        typer.echo(f"WARNING: could not set up log file at {log_file}: {e}", err=True)
