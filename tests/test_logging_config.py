import json
import logging
import threading
from pathlib import Path

from culprit.logging_config import LOGGER_NAME, JsonFormatter, configure_logging


def _make_record(**extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name=LOGGER_NAME,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_produces_valid_json_with_an_event_key():
    record = _make_record(event="test_event", custom_field="custom_value")

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == LOGGER_NAME
    assert payload["message"] == "hello"
    assert payload["event"] == "test_event"
    assert payload["custom_field"] == "custom_value"
    assert "timestamp" in payload


def test_json_formatter_captures_exc_info_traceback():
    """Regression pin: JsonFormatter must read record.exc_info and emit a
    `traceback` field, or logger.exception(...)/logger.error(...,
    exc_info=True) silently drops the stack trace (the log line still
    appears, just with no traceback in it) - see AI_docs/PHASES.md's "TRAP
    every remaining workstream will hit"."""
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _make_record(event="test_event", exc_info=sys.exc_info())

    payload = json.loads(JsonFormatter().format(record))

    assert "traceback" in payload
    assert "ValueError: boom" in payload["traceback"]
    assert "Traceback (most recent call last)" in payload["traceback"]


def test_json_formatter_omits_traceback_field_when_no_exception():
    """The new field is additive only: a normal log line (no exc_info) must
    not gain a spurious `traceback` key."""
    record = _make_record(event="test_event")

    payload = json.loads(JsonFormatter().format(record))

    assert "traceback" not in payload


def test_configure_logging_writes_traceback_for_logger_exception(tmp_path):
    """End-to-end pin at the level jobs.py actually calls it: inside a real
    except block, logger.exception(...) must produce a log line whose
    `traceback` field contains the stack trace, not just the message."""
    log_file = tmp_path / "culprit.jsonl"
    configure_logging(log_file, "INFO")
    logger = logging.getLogger(LOGGER_NAME)

    try:
        raise RuntimeError("kaboom")
    except RuntimeError:
        logger.exception("job failed", extra={"event": "job_failed"})
    for handler in logger.handlers:
        handler.flush()

    lines = log_file.read_text().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "job_failed"
    assert "RuntimeError: kaboom" in payload["traceback"]


def test_json_formatter_handles_non_json_native_extra_fields():
    """A stray Path (or any non-JSON-native extra field, e.g. a datetime)
    must not crash the log call - default=str is the safety net."""
    record = _make_record(event="trace_ingested", trace_dir=Path("traces/example"))

    payload = json.loads(JsonFormatter().format(record))

    assert payload["trace_dir"] == str(Path("traces/example"))


def test_configure_logging_writes_structured_lines_to_file(tmp_path):
    log_file = tmp_path / "culprit.jsonl"
    configure_logging(log_file, "INFO")

    logger = logging.getLogger(LOGGER_NAME)
    logger.info("did a thing", extra={"event": "did_thing", "foo": "bar"})
    for handler in logger.handlers:
        handler.flush()

    lines = log_file.read_text().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "did_thing"
    assert payload["foo"] == "bar"


def test_configure_logging_is_idempotent_and_does_not_accumulate_handlers(tmp_path):
    log_file = tmp_path / "culprit.jsonl"
    configure_logging(log_file, "INFO")
    configure_logging(log_file, "INFO")
    configure_logging(log_file, "INFO")

    logger = logging.getLogger(LOGGER_NAME)
    assert len(logger.handlers) == 1

    logger.info("only once", extra={"event": "once"})
    for handler in logger.handlers:
        handler.flush()

    lines = log_file.read_text().splitlines()
    assert len(lines) == 1


def test_configure_logging_actually_rotates(tmp_path):
    log_file = tmp_path / "culprit.jsonl"
    configure_logging(log_file, "INFO", max_bytes=200, backup_count=2)

    logger = logging.getLogger(LOGGER_NAME)
    for i in range(50):
        logger.info("x" * 50, extra={"event": "pad", "i": i})

    assert log_file.exists()
    assert (tmp_path / "culprit.jsonl.1").exists()


def test_concurrent_logging_from_multiple_threads_produces_no_corrupt_lines(tmp_path):
    """RotatingFileHandler's write+rollover happens inside the inherited
    Handler.lock, so concurrent emit() calls should be fully serialized, not
    interleaved. This exercises that guarantee for real instead of leaving it
    as an unverified claim: many threads log concurrently, and every line
    written must be independently parseable and none may be lost."""
    log_file = tmp_path / "culprit.jsonl"
    configure_logging(log_file, "INFO")
    logger = logging.getLogger(LOGGER_NAME)

    thread_count = 8
    logs_per_thread = 50

    def _log_many(thread_id: int) -> None:
        for i in range(logs_per_thread):
            logger.info(
                "concurrent",
                extra={"event": "concurrent_test", "thread_id": thread_id, "i": i},
            )

    threads = [
        threading.Thread(target=_log_many, args=(t,)) for t in range(thread_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for handler in logger.handlers:
        handler.flush()

    lines = log_file.read_text().splitlines()
    assert len(lines) == thread_count * logs_per_thread

    seen = set()
    for line in lines:
        payload = json.loads(line)  # raises if any line got interleaved/corrupted
        seen.add((payload["thread_id"], payload["i"]))
    assert len(seen) == thread_count * logs_per_thread


def test_configure_logging_degrades_gracefully_when_log_dir_cannot_be_created(tmp_path, capsys):
    """Logging failures must never abort the actual ingest/diagnose/serve
    operation - this proves the OSError fallback path actually works, not
    just that the code reads that way."""
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory - mkdir(parents=True) through this fails")
    log_file = blocker / "sub" / "culprit.jsonl"

    configure_logging(log_file, "INFO")  # must not raise

    captured = capsys.readouterr()
    assert "WARNING: could not set up log file" in captured.err

    logger = logging.getLogger(LOGGER_NAME)
    logger.info("must not crash even though no file handler exists")
