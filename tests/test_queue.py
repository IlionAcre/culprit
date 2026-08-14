import fakeredis
import pytest
from rq import SimpleWorker
from rq.exceptions import NoSuchJobError
from rq.job import Job

from culprit.logging_config import configure_logging
from culprit.queue import QUEUE_NAME, enqueue_diagnosis, enqueue_recluster, fetch_job_status, get_queue


def _fake_connection() -> fakeredis.FakeStrictRedis:
    """A fresh in-memory fake per test, so enqueued jobs never leak between
    tests sharing a process; this is what keeps the whole suite offline."""
    return fakeredis.FakeStrictRedis()


def test_enqueue_diagnosis_returns_a_job_id_and_puts_diagnose_trace_job_on_the_queue():
    conn = _fake_connection()
    queue = get_queue(connection=conn)

    job_id = enqueue_diagnosis("trace-1", queue=queue)

    assert job_id
    job = Job.fetch(job_id, connection=conn)
    assert job.func_name == "culprit.jobs.diagnose_trace_job"
    assert job.args == ("trace-1",)
    assert job.origin == QUEUE_NAME


def test_enqueue_recluster_returns_a_job_id_and_puts_recluster_job_on_the_queue():
    conn = _fake_connection()
    queue = get_queue(connection=conn)

    job_id = enqueue_recluster(queue=queue)

    job = Job.fetch(job_id, connection=conn)
    assert job.func_name == "culprit.jobs.recluster_job"
    assert job.args == ()


def test_fetch_job_status_reports_queued_before_a_worker_runs_it():
    conn = _fake_connection()
    queue = get_queue(connection=conn)
    job_id = enqueue_diagnosis("trace-1", queue=queue)

    status = fetch_job_status(job_id, connection=conn)

    assert status["job_id"] == job_id
    assert status["status"] == "queued"
    assert status["result"] is None
    assert status["error"] is None


def test_fetch_job_status_reports_finished_and_result_after_a_worker_runs_it(monkeypatch):
    """Proves the queue -> worker -> status round trip end to end, with the
    real diagnose_trace_job monkeypatched so no analysis layer or
    persistence module needs to exist yet."""
    conn = _fake_connection()
    queue = get_queue(connection=conn)
    monkeypatch.setattr("culprit.jobs.diagnose_trace_job", lambda trace_id: f"diagnosis-for-{trace_id}")
    job_id = enqueue_diagnosis("trace-1", queue=queue)

    SimpleWorker([queue], connection=conn).work(burst=True)

    status = fetch_job_status(job_id, connection=conn)
    assert status["status"] == "finished"
    assert status["result"] == "diagnosis-for-trace-1"
    assert status["error"] is None


def test_fetch_job_status_reports_failed_and_a_generic_error_when_the_job_raises(monkeypatch):
    """A job that raises must surface as status=failed with a non-empty
    error string, not silently look identical to a successful run - but
    (see the next test) that string must not be the raw exception text."""
    conn = _fake_connection()
    queue = get_queue(connection=conn)

    def boom(trace_id):
        raise ValueError("no persisted trace")

    monkeypatch.setattr("culprit.jobs.diagnose_trace_job", boom)
    job_id = enqueue_diagnosis("trace-1", queue=queue)

    SimpleWorker([queue], connection=conn).work(burst=True)

    status = fetch_job_status(job_id, connection=conn)
    assert status["status"] == "failed"
    assert status["error"] is not None
    assert job_id in status["error"]


def test_fetch_job_status_never_leaks_a_traceback_but_logs_it_for_operators(monkeypatch, tmp_path):
    """Pins the information-disclosure fix: GET /jobs/{id} (api.py passes
    fetch_job_status's dict straight through) must never hand a caller a
    failed job's traceback - it can embed a Postgres DSN, a Redis URL, or a
    litellm request/response fragment, none of which should cross the HTTP
    boundary. The detail must still reach an operator, correlated by
    job_id, via the structured log. Mirrors Litmus's own
    `assert "Traceback" not in result.stdout` precedent."""
    log_file = tmp_path / "culprit-test.jsonl"
    configure_logging(str(log_file), "INFO")
    conn = _fake_connection()
    queue = get_queue(connection=conn)
    secret_dsn = "postgresql://culprit:s3cr3t-password@internal-db.example:5432/culprit"

    def boom(trace_id):
        raise RuntimeError(f"could not connect: {secret_dsn}")

    monkeypatch.setattr("culprit.jobs.diagnose_trace_job", boom)
    job_id = enqueue_diagnosis("trace-1", queue=queue)

    SimpleWorker([queue], connection=conn).work(burst=True)
    status = fetch_job_status(job_id, connection=conn)

    assert status["status"] == "failed"
    assert "Traceback" not in status["error"]
    assert secret_dsn not in status["error"]
    assert "RuntimeError" not in status["error"]

    logged = log_file.read_text()
    assert secret_dsn in logged
    assert job_id in logged


def test_fetch_job_status_raises_no_such_job_error_for_an_unknown_job_id():
    """api.py maps this straight to a 404; this test proves the exception
    type queue.py lets through un-wrapped."""
    conn = _fake_connection()

    with pytest.raises(NoSuchJobError):
        fetch_job_status("does-not-exist", connection=conn)
