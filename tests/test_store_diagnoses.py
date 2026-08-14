"""Tests for `store_diagnoses.py` (the `Diagnosis` write/read round trip).
Cluster-assignment-bridge tests (`write_cluster_assignments`) live in
`test_store_clusters.py`, mirroring the `store_diagnoses.py` /
`store_clusters.py` module split. Same two-tier structure: offline tests
against hand-shaped rows and a fake connection, plus `@requires_db` tests
proving the real thing, skipping cleanly when `CULPRIT_TEST_DSN` is unset.
"""

import contextlib
import datetime as dt

import pytest

from culprit.signals import Evidence
from culprit.store_diagnoses import (
    _adjudication_from_row,
    _divergence_from_row,
    _signal_from_row,
    read_all_diagnoses,
    read_diagnoses,
    write_diagnosis,
)
from culprit.synth_results import make_diagnosis, make_divergence, make_signal
from tests.conftest import requires_db

# --- offline: fake connection/cursor -----------------------------------


class _FakeCursor:
    def __init__(self, results: list[list[dict]] | None = None):
        self._results = list(results or [])
        self._current: list[dict] = []
        self.calls: list[tuple[str, object]] = []
        self.rowcount = 0

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        self._current = self._results.pop(0) if self._results else []
        self.rowcount = len(self._current)
        return self

    def executemany(self, sql, seq):
        seq = list(seq)
        self.calls.append((sql, seq))
        self.rowcount = len(seq)
        return self

    def fetchone(self):
        return self._current[0] if self._current else None

    def fetchall(self):
        return self._current

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConnection:
    def __init__(self, results: list[list[dict]] | None = None):
        self.cur = _FakeCursor(results)

    def cursor(self, row_factory=None):
        return self.cur

    def execute(self, sql, params=None):
        return self.cur.execute(sql, params)

    def transaction(self):
        return contextlib.nullcontext()


def _conn_fn(conn):
    return lambda: conn


# --- offline: row -> dataclass builders ----------------------------------


def test_signal_from_row_reconstructs_evidence_list():
    signal = make_signal(
        step_index=3,
        evidence=[Evidence(span_id="s3", step_index=3, field="payload.result_text", excerpt="empty")],
    )
    row = {
        "detector": signal.detector,
        "step_index": signal.step_index,
        "span_id": signal.span_id,
        "severity": signal.severity,
        "category": signal.category,
        "message": signal.message,
        "evidence": [{"span_id": "s3", "step_index": 3, "field": "payload.result_text", "excerpt": "empty", "numeric": None}],
        "error": signal.error,
    }

    rebuilt = _signal_from_row(row)

    assert rebuilt.evidence == signal.evidence
    assert rebuilt.detector == signal.detector


def test_divergence_from_row_reconstructs_expected_signatures_as_tuples():
    divergence = make_divergence(expected_signatures=[("tool:agent:x:ok", 0.7), ("tool:agent:y:ok", 0.3)])
    row = {
        "step_index": divergence.step_index,
        "span_id": divergence.span_id,
        "divergence_score": divergence.divergence_score,
        "cliff_delta": divergence.cliff_delta,
        "surprisal": divergence.surprisal,
        "profile_surprisal": divergence.profile_surprisal,
        "reference_count": divergence.reference_count,
        "observed_signature": divergence.observed_signature,
        "expected_signatures": [list(p) for p in divergence.expected_signatures],  # JSONB round trip: tuple -> list
        "nearest_reference_trace_ids": divergence.nearest_reference_trace_ids,
        "alignment_op": divergence.alignment_op,
        "error": divergence.error,
    }

    rebuilt = _divergence_from_row(row)

    assert rebuilt.expected_signatures == divergence.expected_signatures
    assert all(isinstance(p, tuple) for p in rebuilt.expected_signatures)


def test_adjudication_from_row_defaults_null_rationale_and_counterfactual():
    row = {
        "step_index": 2,
        "span_id": "s2",
        "is_root_cause": True,
        "failure_class": "tool_error",
        "confidence": 0.8,
        "calibrated_confidence": 0.7,
        "rationale": None,
        "counterfactual": None,
        "cited_step_indices": None,
        "abstained": False,
        "model": "gemini/flash",
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "cost_usd": 0.001,
        "error": None,
    }

    adjudication = _adjudication_from_row(row)

    assert adjudication.rationale == ""
    assert adjudication.counterfactual == ""
    assert adjudication.cited_step_indices == []


# --- offline: write_diagnosis SQL shape -----------------------------------


def test_write_diagnosis_writes_diagnoses_row_plus_all_three_child_tables():
    diagnosis = make_diagnosis(
        signals=[make_signal(step_index=1)],
        divergences=[make_divergence(step_index=1)],
        adjudications=[],
    )
    conn = _FakeConnection()

    write_diagnosis(_conn_fn(conn), diagnosis)

    sql_calls = [sql for sql, _ in conn.cur.calls]
    assert any("INSERT INTO diagnoses" in s for s in sql_calls)
    assert any("INSERT INTO signals" in s for s in sql_calls)
    assert any("INSERT INTO divergences" in s for s in sql_calls)
    # no adjudications passed, so no adjudications INSERT should fire at all
    assert not any("INSERT INTO adjudications" in s for s in sql_calls)


def test_write_diagnosis_skips_empty_child_tables_entirely():
    """A diagnosis with no signals/divergences/adjudications (e.g. a
    degraded-layer result) must not issue a zero-row executemany call."""
    diagnosis = make_diagnosis(signals=[], divergences=[], adjudications=[])
    conn = _FakeConnection()

    write_diagnosis(_conn_fn(conn), diagnosis)

    sql_calls = [sql for sql, _ in conn.cur.calls]
    assert len(sql_calls) == 1  # only the diagnoses row itself
    assert "INSERT INTO diagnoses" in sql_calls[0]


# --- offline: read assembly against a fake connection ---------------------


def test_read_diagnoses_assembles_signals_and_divergences_by_diagnosis_id():
    diagnosis_row = {
        "diagnosis_id": "d-1",
        "trace_id": "trace-1",
        "created_at": dt.datetime.now(dt.UTC),
        "root_cause_step_index": 2,
        "root_cause_span_id": "s2",
        "failure_class": "tool_error",
        "confidence": 0.8,
        "calibrated_confidence": 0.7,
        "abstained": False,
        "abstain_reason": None,
        "rationale": "r",
        "counterfactual": "c",
        "candidates_considered": 1,
        "layer_versions": {"l1": "0.1.0"},
        "degraded_layers": [],
        "error": None,
    }
    signal_row = {
        "diagnosis_id": "d-1",
        "detector": "empty_tool_result",
        "step_index": 2,
        "span_id": "s2",
        "severity": 0.9,
        "category": "tool_error",
        "message": "empty result",
        "evidence": [],
        "error": None,
    }

    conn = _FakeConnection(results=[[diagnosis_row], [signal_row], [], []])

    result = read_diagnoses(_conn_fn(conn), "trace-1")

    assert len(result) == 1
    assert result[0].diagnosis_id == "d-1"
    assert len(result[0].signals) == 1
    assert result[0].signals[0].detector == "empty_tool_result"
    assert result[0].divergences == []
    assert result[0].adjudications == []


def test_read_all_diagnoses_queries_without_a_trace_id_filter():
    conn = _FakeConnection(results=[[], [], [], []])

    result = read_all_diagnoses(_conn_fn(conn))

    sql, params = conn.cur.calls[0]
    assert "FROM diagnoses" in sql
    assert "WHERE" not in sql
    assert result == []


# --- requires_db: real round trips ---------------------------------------


@pytest.fixture
def db_conn_fn():
    import os

    import psycopg
    from pgvector.psycopg import register_vector

    dsn = os.environ["CULPRIT_TEST_DSN"]
    conn = psycopg.connect(dsn, autocommit=False)
    register_vector(conn)
    conn.execute(
        "TRUNCATE traces, spans, steps, diagnoses, signals, divergences, "
        "adjudications, clusters CASCADE"
    )
    conn.execute("INSERT INTO traces (trace_id, source, outcome) VALUES ('trace-db-1', 'test', 'failure')")
    conn.commit()
    yield lambda: conn
    conn.rollback()
    conn.close()


@requires_db
def test_round_trip_write_then_read_diagnosis_reproduces_every_list(db_conn_fn):
    diagnosis = make_diagnosis(
        trace_id="trace-db-1",
        signals=[make_signal(step_index=1), make_signal(step_index=2)],
        divergences=[make_divergence(step_index=1)],
        adjudications=[],
    )

    write_diagnosis(db_conn_fn, diagnosis)
    result = read_diagnoses(db_conn_fn, "trace-db-1")

    assert len(result) == 1
    assert result[0].diagnosis_id == diagnosis.diagnosis_id
    assert len(result[0].signals) == 2
    assert len(result[0].divergences) == 1
    assert result[0].layer_versions == diagnosis.layer_versions


@requires_db
def test_diagnoses_trace_id_is_not_unique_multiple_diagnoses_per_trace_round_trip(db_conn_fn):
    """CLAUDE.md: re-running analysis must produce a new diagnosis alongside
    the old, never overwrite it."""
    first = make_diagnosis(trace_id="trace-db-1")
    second = make_diagnosis(trace_id="trace-db-1")

    write_diagnosis(db_conn_fn, first)
    write_diagnosis(db_conn_fn, second)

    result = read_diagnoses(db_conn_fn, "trace-db-1")

    assert {d.diagnosis_id for d in result} == {first.diagnosis_id, second.diagnosis_id}


@requires_db
def test_read_all_diagnoses_returns_diagnoses_across_traces(db_conn_fn):
    with db_conn_fn().cursor() as cur:
        cur.execute("INSERT INTO traces (trace_id, source, outcome) VALUES ('trace-db-2', 'test', 'failure')")

    write_diagnosis(db_conn_fn, make_diagnosis(trace_id="trace-db-1"))
    write_diagnosis(db_conn_fn, make_diagnosis(trace_id="trace-db-2"))

    result = read_all_diagnoses(db_conn_fn)

    assert {d.trace_id for d in result} == {"trace-db-1", "trace-db-2"}
