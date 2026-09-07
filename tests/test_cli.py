import json
import os

import psycopg
from typer.testing import CliRunner

from culprit.cli import _redact_dsn, app
from culprit.jobs import PersistenceNotWiredError
from culprit.synth_results import make_diagnosis

runner = CliRunner()

_BOX_CHARS = "─│┌┐└┘├┤┬┴┼━┃╭╮╰╯"


def _unwrapped(output: str) -> str:
    """Rich draws borders and wraps to the console width, so a value can be
    split across lines by padding that carries no meaning. Strip the box
    characters and collapse whitespace before asserting on content, so these
    tests check what was rendered rather than how wide the terminal was."""
    stripped = "".join(" " if ch in _BOX_CHARS else ch for ch in output)
    return " ".join(stripped.split())


def test_ingest_reads_the_file_and_prints_the_new_trace_id(tmp_path, monkeypatch):
    trace_file = tmp_path / "trace.json"
    trace_file.write_text(json.dumps({"resourceSpans": []}))
    captured = {}

    def fake_ingest_trace(payload, content_type):
        captured["payload"] = payload
        captured["content_type"] = content_type
        return "trace-1"

    monkeypatch.setattr("culprit.cli.ingest_trace", fake_ingest_trace)

    result = runner.invoke(app, ["ingest", str(trace_file)])

    assert result.exit_code == 0
    assert "trace-1" in result.output
    assert captured["content_type"] == "application/json"


def test_ingest_treats_non_json_extension_as_protobuf(tmp_path, monkeypatch):
    trace_file = tmp_path / "trace.pb"
    trace_file.write_bytes(b"\x0a\x00")
    captured = {}

    def fake_ingest_trace(payload, content_type):
        captured["content_type"] = content_type
        return "trace-2"

    monkeypatch.setattr("culprit.cli.ingest_trace", fake_ingest_trace)

    result = runner.invoke(app, ["ingest", str(trace_file)])

    assert result.exit_code == 0
    assert captured["content_type"] == "application/x-protobuf"


def test_ingest_exits_nonzero_and_prints_error_when_persistence_not_wired(tmp_path, monkeypatch):
    trace_file = tmp_path / "trace.json"
    trace_file.write_text("{}")

    def raising(payload, content_type):
        raise PersistenceNotWiredError("OTLP ingestion pipeline is not wired yet")

    monkeypatch.setattr("culprit.cli.ingest_trace", raising)

    result = runner.invoke(app, ["ingest", str(trace_file)])

    assert result.exit_code == 1
    assert "not wired yet" in result.output


def test_diagnose_enqueues_and_prints_the_job_id(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("culprit.cli.enqueue_diagnosis", lambda trace_id: "job-1")

    result = runner.invoke(app, ["diagnose", "trace-1"])

    assert result.exit_code == 0
    assert "job-1" in result.output
    assert "trace-1" in result.output


def test_diagnose_says_a_worker_is_needed_and_names_the_inline_alternative(monkeypatch):
    """Enqueueing succeeds whether or not anything is listening, so the old
    output looked identical whether the job was about to run or would sit
    in the queue forever."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("culprit.cli.enqueue_diagnosis", lambda trace_id: "job-1")

    result = runner.invoke(app, ["diagnose", "trace-1"])

    assert result.exit_code == 0
    assert "culprit worker" in result.output
    assert "culprit run --trace-id trace-1" in _unwrapped(result.output)


def test_run_ingests_then_diagnoses_inline_and_renders_the_verdict(tmp_path, monkeypatch):
    trace_file = tmp_path / "trace.json"
    trace_file.write_text(json.dumps({"resourceSpans": []}))
    diagnosis = make_diagnosis(trace_id="trace-1", root_cause_step_index=4)

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("culprit.cli.ingest_trace", lambda payload, content_type: "trace-1")
    monkeypatch.setattr(
        "culprit.jobs.diagnose_trace_job", lambda trace_id: diagnosis.diagnosis_id
    )
    monkeypatch.setattr("culprit.cli.read_diagnoses_for_trace", lambda trace_id: [diagnosis])
    monkeypatch.setattr("culprit.cli._step_context_for", lambda view: None)

    result = runner.invoke(app, ["run", str(trace_file)])

    assert result.exit_code == 0
    assert "Ingested trace trace-1" in result.output
    assert "step 4" in result.output


def test_run_accepts_an_already_ingested_trace_id(monkeypatch):
    diagnosis = make_diagnosis(trace_id="trace-1", root_cause_step_index=4)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        "culprit.jobs.diagnose_trace_job", lambda trace_id: diagnosis.diagnosis_id
    )
    monkeypatch.setattr("culprit.cli.read_diagnoses_for_trace", lambda trace_id: [diagnosis])
    monkeypatch.setattr("culprit.cli._step_context_for", lambda view: None)

    result = runner.invoke(app, ["run", "--trace-id", "trace-1"])

    assert result.exit_code == 0
    assert "step 4" in result.output


def test_run_rejects_both_a_file_and_a_trace_id(tmp_path, monkeypatch):
    trace_file = tmp_path / "trace.json"
    trace_file.write_text("{}")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    result = runner.invoke(app, ["run", str(trace_file), "--trace-id", "trace-1"])

    assert result.exit_code == 1
    assert "not both or neither" in result.output


def test_run_rejects_neither_a_file_nor_a_trace_id(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 1
    assert "not both or neither" in result.output


def test_run_fails_before_any_work_when_the_api_key_is_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def unreachable(trace_id):
        raise AssertionError("diagnose_trace_job ran without an API key")

    monkeypatch.setattr("culprit.jobs.diagnose_trace_job", unreachable)

    result = runner.invoke(app, ["run", "--trace-id", "trace-1"])

    assert result.exit_code == 1
    assert "GEMINI_API_KEY" in result.output


def test_show_names_the_command_that_fills_the_gap_when_there_are_no_diagnoses(monkeypatch):
    monkeypatch.setattr("culprit.cli.read_diagnoses_for_trace", lambda trace_id: [])

    result = runner.invoke(app, ["show", "trace-unknown"])

    assert result.exit_code == 1
    assert "culprit run --trace-id trace-unknown" in _unwrapped(result.output)


def test_show_prints_every_diagnosis_for_the_trace(monkeypatch):
    diagnosis = make_diagnosis(trace_id="trace-1", root_cause_step_index=4)
    monkeypatch.setattr("culprit.cli.read_diagnoses_for_trace", lambda trace_id: [diagnosis])
    monkeypatch.setattr("culprit.cli._step_context_for", lambda view: None)

    result = runner.invoke(app, ["show", "trace-1"])

    assert result.exit_code == 0
    assert "trace-1" in result.output
    assert "step 4" in result.output
    # The id identifies which diagnosis this is when a trace has several,
    # so it stays reachable even though the verdict leads.
    assert diagnosis.diagnosis_id in _unwrapped(result.output)


def test_show_exits_nonzero_when_no_diagnoses_exist(monkeypatch):
    monkeypatch.setattr("culprit.cli.read_diagnoses_for_trace", lambda trace_id: [])

    result = runner.invoke(app, ["show", "trace-unknown"])

    assert result.exit_code == 1
    assert "No diagnoses" in result.output


def test_job_status_prints_status_and_result(monkeypatch):
    status = {"job_id": "job-1", "status": "finished", "result": "diag-1", "error": None}
    monkeypatch.setattr("culprit.cli.fetch_job_status", lambda job_id: status)

    result = runner.invoke(app, ["job-status", "job-1"])

    assert result.exit_code == 0
    assert "finished" in result.output
    assert "diag-1" in result.output


def test_recluster_enqueues_and_prints_the_job_id(monkeypatch):
    monkeypatch.setattr("culprit.cli.enqueue_recluster", lambda: "job-recluster-1")

    result = runner.invoke(app, ["recluster"])

    assert result.exit_code == 0
    assert "job-recluster-1" in result.output


def test_main_callback_sets_database_and_redis_url_env_vars_from_config(monkeypatch):
    """cli.py must propagate config into the env vars queue.py/jobs.py read,
    per CLAUDE.md's rule that only cli.py reads culprit.config directly."""
    monkeypatch.delenv("CULPRIT_DATABASE_URL", raising=False)
    monkeypatch.delenv("CULPRIT_REDIS_URL", raising=False)
    monkeypatch.setattr("culprit.cli.enqueue_recluster", lambda: "job-1")

    runner.invoke(app, ["recluster"])

    assert os.environ.get("CULPRIT_DATABASE_URL") == "postgresql://localhost:5432/culprit"
    assert os.environ.get("CULPRIT_REDIS_URL") == "redis://localhost:6379/0"


def test_main_callback_treats_an_empty_database_url_env_var_as_unset(monkeypatch):
    """The README's `cp .env.example .env && set -a && . ./.env` step
    exports CULPRIT_DATABASE_URL present but blank; that must still fall
    back to CulpritConfig.database_url rather than becoming a DSN of ''."""
    monkeypatch.setenv("CULPRIT_DATABASE_URL", "")
    monkeypatch.setattr("culprit.cli.enqueue_recluster", lambda: "job-1")

    runner.invoke(app, ["recluster"])

    assert os.environ.get("CULPRIT_DATABASE_URL") == "postgresql://localhost:5432/culprit"


def test_main_callback_keeps_an_explicit_non_empty_database_url_env_var(monkeypatch):
    monkeypatch.setenv("CULPRIT_DATABASE_URL", "postgresql://example.com:5432/mine")
    monkeypatch.setattr("culprit.cli.enqueue_recluster", lambda: "job-1")

    runner.invoke(app, ["recluster"])

    assert os.environ.get("CULPRIT_DATABASE_URL") == "postgresql://example.com:5432/mine"


def test_main_callback_treats_empty_redis_url_and_model_env_vars_as_unset(monkeypatch):
    monkeypatch.setenv("CULPRIT_REDIS_URL", "   ")
    monkeypatch.setenv("CULPRIT_MODEL", "")
    monkeypatch.setattr("culprit.cli.enqueue_recluster", lambda: "job-1")

    runner.invoke(app, ["recluster"])

    assert os.environ.get("CULPRIT_REDIS_URL") == "redis://localhost:6379/0"
    assert os.environ.get("CULPRIT_MODEL") == "gemini/gemini-2.5-flash-lite"


def test_show_exits_nonzero_and_prints_message_when_postgres_is_unreachable(monkeypatch):
    monkeypatch.setenv("CULPRIT_DATABASE_URL", "postgresql://localhost:5432/culprit")

    def raising(trace_id):
        raise psycopg.OperationalError("connection failed")

    monkeypatch.setattr("culprit.cli.read_diagnoses_for_trace", raising)

    result = runner.invoke(app, ["show", "trace-1"])

    assert result.exit_code == 1
    assert "postgresql://localhost:5432/culprit" in result.output
    assert ".env" in result.output
    assert "compose.yaml" in result.output
    assert "Traceback" not in result.output


def test_show_redacts_the_database_password_from_the_unreachable_message(monkeypatch):
    monkeypatch.setenv(
        "CULPRIT_DATABASE_URL",
        "postgresql://culprit_user:hunter2@db.internal:5432/culprit?sslmode=require",
    )

    def raising(trace_id):
        raise psycopg.OperationalError("connection failed")

    monkeypatch.setattr("culprit.cli.read_diagnoses_for_trace", raising)

    result = runner.invoke(app, ["show", "trace-1"])

    assert result.exit_code == 1
    assert "hunter2" not in result.output
    assert "sslmode" not in result.output
    assert "culprit_user:***@db.internal:5432" in result.output


def test_redact_dsn_reports_a_keyword_form_dsn_by_shape_rather_than_content():
    redacted = _redact_dsn("host=db.internal port=5432 password=hunter2")

    assert "hunter2" not in redacted
    assert redacted == "<unparseable connection string>"


def test_redact_dsn_leaves_a_credential_free_url_unchanged():
    assert (
        _redact_dsn("postgresql://localhost:5432/culprit")
        == "postgresql://localhost:5432/culprit"
    )


def test_diagnose_fails_when_gemini_api_key_is_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = runner.invoke(app, ["diagnose", "trace-1"])

    assert result.exit_code == 1
    assert "GEMINI_API_KEY" in result.output
    assert ".env.example" in result.output


def test_diagnose_fails_when_gemini_api_key_is_empty(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "   ")

    result = runner.invoke(app, ["diagnose", "trace-1"])

    assert result.exit_code == 1
    assert "GEMINI_API_KEY" in result.output
    assert ".env.example" in result.output


def test_planted_secret_in_dsn_is_never_printed_in_error_output(monkeypatch):
    planted_secret = "PLANTED_SUPER_SECRET_VALUE_98765"
    monkeypatch.setenv(
        "CULPRIT_DATABASE_URL",
        f"postgresql://culprit_user:{planted_secret}@db.internal:5432/culprit?secret_param={planted_secret}",
    )

    def raising(trace_id):
        raise psycopg.OperationalError("connection failed")

    monkeypatch.setattr("culprit.cli.read_diagnoses_for_trace", raising)

    result = runner.invoke(app, ["show", "trace-1"])

    assert result.exit_code == 1
    assert planted_secret not in result.output
    assert "culprit_user:***@db.internal:5432" in result.output


def test_planted_secret_in_redis_dsn_is_never_printed_in_error_output(monkeypatch):
    planted_secret = "REDIS_SUPER_SECRET_XYZ"
    monkeypatch.setenv(
        "CULPRIT_REDIS_URL",
        f"redis://:{planted_secret}@localhost:6379/0",
    )
    from redis.exceptions import ConnectionError as RedisConnectionError

    def raising(trace_id):
        raise RedisConnectionError("redis connection failed")

    monkeypatch.setattr("culprit.cli.enqueue_diagnosis", raising)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    result = runner.invoke(app, ["diagnose", "trace-1"])

    assert result.exit_code == 1
    assert planted_secret not in result.output
    assert ":***@localhost:6379" in result.output


def test_connection_failure_returns_quickly_without_driver_noise(monkeypatch):
    import time

    monkeypatch.setenv("CULPRIT_DATABASE_URL", "postgresql://localhost:54321/culprit")

    t0 = time.perf_counter()
    result = runner.invoke(app, ["show", "trace-1"])
    elapsed = time.perf_counter() - t0

    assert result.exit_code == 1
    assert elapsed < 10.0
    assert "could not reach Postgres at postgresql://localhost:54321/culprit" in result.output
    assert "couldn't stop thread" not in result.output
    assert "Traceback" not in result.output
    assert "error connecting in" not in result.output
