import json
import os

import psycopg
from typer.testing import CliRunner

from culprit.cli import _redact_dsn, app
from culprit.jobs import PersistenceNotWiredError
from culprit.synth_results import make_diagnosis

runner = CliRunner()


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
    monkeypatch.setattr("culprit.cli.enqueue_diagnosis", lambda trace_id: "job-1")

    result = runner.invoke(app, ["diagnose", "trace-1"])

    assert result.exit_code == 0
    assert "job-1" in result.output
    assert "trace-1" in result.output


def test_show_prints_every_diagnosis_for_the_trace(monkeypatch):
    diagnosis = make_diagnosis(trace_id="trace-1", root_cause_step_index=4)
    monkeypatch.setattr("culprit.cli.read_diagnoses_for_trace", lambda trace_id: [diagnosis])

    result = runner.invoke(app, ["show", "trace-1"])

    assert result.exit_code == 0
    assert diagnosis.diagnosis_id in result.output
    assert "step=4" in result.output


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
