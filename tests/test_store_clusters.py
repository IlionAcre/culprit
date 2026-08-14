"""Tests for `store_clusters.py` (the `cluster_diagnoses` int-label -> UUID
bridge). Split out of `test_store_diagnoses.py`, mirroring the module split.
Same two-tier structure: offline tests against a fake connection plus pure
logic, and `@requires_db` tests proving the real thing, skipping cleanly
when `CULPRIT_TEST_DSN` is unset.
"""

import contextlib
import uuid

import pytest

from culprit.store_clusters import _cluster_ids_for, _cluster_sizes, write_cluster_assignments
from culprit.store_diagnoses import write_diagnosis
from culprit.synth_results import make_diagnosis
from tests.conftest import requires_db

# --- offline: fake connection/cursor -----------------------------------


class _FakeCursor:
    def __init__(self, results: list[list] | None = None):
        self._results = list(results or [])
        self._current: list = []
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
    def __init__(self, results: list[list] | None = None):
        self.cur = _FakeCursor(results)

    def cursor(self, row_factory=None):
        return self.cur

    def execute(self, sql, params=None):
        return self.cur.execute(sql, params)

    def transaction(self):
        return contextlib.nullcontext()


def _conn_fn(conn):
    return lambda: conn


# --- offline: pure logic ---------------------------------------------------


def test_cluster_ids_for_maps_each_distinct_non_noise_label_deterministically():
    run_id = uuid.uuid4()
    assignment = {"d1": 0, "d2": 0, "d3": 1, "d4": -1}

    cluster_ids = _cluster_ids_for(run_id, assignment)

    assert set(cluster_ids.keys()) == {0, 1}
    assert cluster_ids[0] == uuid.uuid5(run_id, "0")
    # same run_id + label always yields the same cluster_id, deterministically
    assert _cluster_ids_for(run_id, assignment)[0] == cluster_ids[0]


def test_cluster_ids_for_excludes_the_noise_label():
    run_id = uuid.uuid4()
    assignment = {"d1": -1, "d2": -1}

    assert _cluster_ids_for(run_id, assignment) == {}


def test_cluster_sizes_counts_only_non_noise_labels():
    assignment = {"d1": 0, "d2": 0, "d3": 1, "d4": -1}

    assert _cluster_sizes(assignment) == {0: 2, 1: 1}


# --- offline: write_cluster_assignments SQL shape --------------------------


def test_write_cluster_assignments_is_a_noop_on_an_empty_assignment():
    conn = _FakeConnection()

    write_cluster_assignments(_conn_fn(conn), {})

    assert conn.cur.calls == []


def test_write_cluster_assignments_writes_cluster_rows_and_updates_diagnoses():
    conn = _FakeConnection()
    assignment = {"d1": 0, "d2": 0, "d3": -1}

    write_cluster_assignments(_conn_fn(conn), assignment)

    cluster_insert = next(c for c in conn.cur.calls if "INSERT INTO clusters" in c[0])
    diagnosis_update = next(c for c in conn.cur.calls if "UPDATE diagnoses" in c[0])

    assert len(cluster_insert[1]) == 1  # one distinct non-noise label (0)
    assert len(diagnosis_update[1]) == 3  # every diagnosis gets an update, including noise -> NULL
    noise_row = next(row for row in diagnosis_update[1] if row[1] == "d3")
    assert noise_row[0] is None


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
def test_write_cluster_assignments_round_trips_into_clusters_and_diagnoses_cluster_id(db_conn_fn):
    d1 = make_diagnosis(trace_id="trace-db-1")
    d2 = make_diagnosis(trace_id="trace-db-1")
    write_diagnosis(db_conn_fn, d1)
    write_diagnosis(db_conn_fn, d2)

    write_cluster_assignments(db_conn_fn, {d1.diagnosis_id: 0, d2.diagnosis_id: -1})

    with db_conn_fn().cursor() as cur:
        cur.execute(
            "SELECT diagnosis_id, cluster_id FROM diagnoses WHERE diagnosis_id = ANY(%s)",
            ([d1.diagnosis_id, d2.diagnosis_id],),
        )
        rows = {str(diag_id): cluster_id for diag_id, cluster_id in cur.fetchall()}

    assert rows[d1.diagnosis_id] is not None  # clustered (label 0)
    assert rows[d2.diagnosis_id] is None  # noise (label -1) -> no cluster

    with db_conn_fn().cursor() as cur:
        cur.execute("SELECT size FROM clusters")
        (size,) = cur.fetchone()
    assert size == 1
