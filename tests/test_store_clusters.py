"""Tests for `store_clusters.py` (the `cluster_diagnoses` int-label -> UUID
bridge, plus INTEGRATION_ITEMS.md item 1's identity-resolution fix). Split
out of `test_store_diagnoses.py`, mirroring the module split. Same two-tier
structure: offline tests against a fake connection plus pure logic, and
`@requires_db` tests proving the real thing, skipping cleanly when
`CULPRIT_TEST_DSN` is unset.
"""

import contextlib
import uuid

import pytest

from culprit.store_clusters import (
    _cluster_sizes,
    _group_by_label,
    resolve_cluster_identity,
    write_cluster_assignments,
)
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


def test_group_by_label_excludes_the_noise_label():
    assignment = {"d1": 0, "d2": 0, "d3": 1, "d4": -1}

    groups = _group_by_label(assignment)

    assert groups == {0: ["d1", "d2"], 1: ["d3"]}


def test_cluster_sizes_counts_only_non_noise_labels():
    assignment = {"d1": 0, "d2": 0, "d3": 1, "d4": -1}

    assert _cluster_sizes(assignment) == {0: 2, 1: 1}


# --- offline: resolve_cluster_identity (INTEGRATION_ITEMS.md item 1) -------


def test_resolve_cluster_identity_mints_fresh_uuids_with_no_prior_state():
    conn = _FakeConnection(results=[[]])  # no prior diagnoses.cluster_id rows
    assignment = {"d1": 0, "d2": 0, "d3": -1}

    resolved = resolve_cluster_identity(_conn_fn(conn), assignment)

    assert set(resolved.keys()) == {0}
    assert isinstance(resolved[0], uuid.UUID)


def test_resolve_cluster_identity_reuses_cluster_id_on_majority_overlap():
    """The crux of item 1: a new label whose members mostly already point at
    the same old cluster_id inherits that identity, regardless of what
    integer HDBSCAN happened to assign this pass."""
    old_cluster_id = uuid.uuid4()
    conn = _FakeConnection(results=[[("d1", old_cluster_id), ("d2", old_cluster_id)]])
    assignment = {"d1": 5, "d2": 5, "d3": 5}  # label "5" this pass, was "anything" before

    resolved = resolve_cluster_identity(_conn_fn(conn), assignment)

    assert resolved == {5: old_cluster_id}


def test_resolve_cluster_identity_mints_fresh_uuid_without_a_majority():
    """Two of three members point at two different old clusters: neither has
    a majority, so this is treated as a new cluster rather than guessing."""
    cluster_a, cluster_b = uuid.uuid4(), uuid.uuid4()
    conn = _FakeConnection(results=[[("d1", cluster_a), ("d2", cluster_b)]])
    assignment = {"d1": 0, "d2": 0, "d3": 0}

    resolved = resolve_cluster_identity(_conn_fn(conn), assignment)

    assert resolved[0] not in (cluster_a, cluster_b)


def test_resolve_cluster_identity_does_not_double_assign_a_split_cluster():
    """If HDBSCAN splits one old cluster into two new labels, only the label
    with the stronger overlap keeps the old identity; the other mints
    fresh - a persisted cluster_id is never handed to two live rows."""
    old_cluster_id = uuid.uuid4()
    conn = _FakeConnection(
        results=[[("d1", old_cluster_id), ("d2", old_cluster_id), ("d3", old_cluster_id)]]
    )
    assignment = {"d1": 0, "d2": 0, "d3": 0, "d4": 1, "d5": 1}

    resolved = resolve_cluster_identity(_conn_fn(conn), assignment)

    assert resolved[0] == old_cluster_id
    assert resolved[1] != old_cluster_id


# --- offline: write_cluster_assignments SQL shape --------------------------


def test_write_cluster_assignments_is_a_noop_on_an_empty_assignment():
    conn = _FakeConnection()

    result = write_cluster_assignments(_conn_fn(conn), {})

    assert conn.cur.calls == []
    assert result == {}


def test_write_cluster_assignments_writes_cluster_rows_and_returns_identity_map():
    conn = _FakeConnection(results=[[]])  # no prior membership
    assignment = {"d1": 0, "d2": 0, "d3": -1}

    result = write_cluster_assignments(_conn_fn(conn), assignment)

    cluster_insert = next(c for c in conn.cur.calls if "INSERT INTO clusters" in c[0])
    diagnosis_update = next(c for c in conn.cur.calls if "UPDATE diagnoses" in c[0])

    assert len(cluster_insert[1]) == 1  # one distinct non-noise label (0)
    assert len(diagnosis_update[1]) == 3  # every diagnosis gets an update, including noise -> NULL
    noise_row = next(row for row in diagnosis_update[1] if row[1] == "d3")
    assert noise_row[0] is None
    assert result == {0: cluster_insert[1][0][0]}  # returned map matches what got inserted


# --- INTEGRATION_ITEMS.md item 1: the skip fires across two real passes ----


class _FakeTables:
    """A minimal in-memory stand-in for the two tables this module and
    `store_cluster_labels.py` actually touch (`diagnoses.cluster_id` and
    `clusters`), stateful across calls unlike `_FakeConnection` above. This
    is the only way to prove the point of item 1's fix: that a second,
    independent `write_cluster_assignments` call resolves the *same*
    `cluster_id` as the first even though HDBSCAN handed it a different
    integer label, purely by reading back what the first call persisted."""

    def __init__(self):
        self.diagnosis_cluster_id: dict[str, uuid.UUID | None] = {}
        self.clusters: dict[uuid.UUID, dict] = {}

    def conn_fn(self):
        return _StatefulConnection(self)


class _StatefulCursor:
    def __init__(self, tables: _FakeTables):
        self.tables = tables
        self._current: list = []

    def execute(self, sql, params=None):
        if "SELECT diagnosis_id, cluster_id FROM diagnoses" in sql:
            (diagnosis_ids,) = params
            self._current = [
                (did, self.tables.diagnosis_cluster_id[did])
                for did in diagnosis_ids
                if self.tables.diagnosis_cluster_id.get(did) is not None
            ]
        else:
            raise AssertionError(f"unexpected SQL in fake cursor: {sql!r}")
        return self

    def executemany(self, sql, seq):
        seq = list(seq)
        if "INSERT INTO clusters" in sql:
            for cluster_id, run_id, size in seq:
                self.tables.clusters.setdefault(cluster_id, {})
                self.tables.clusters[cluster_id].update(run_id=run_id, size=size)
        elif "UPDATE diagnoses SET cluster_id" in sql:
            for cluster_id, diagnosis_id in seq:
                self.tables.diagnosis_cluster_id[diagnosis_id] = cluster_id
        else:
            raise AssertionError(f"unexpected SQL in fake cursor: {sql!r}")
        return self

    def fetchall(self):
        return self._current

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _StatefulConnection:
    def __init__(self, tables: _FakeTables):
        self.tables = tables

    def cursor(self, row_factory=None):
        return _StatefulCursor(self.tables)

    def transaction(self):
        return contextlib.nullcontext()


def test_cluster_identity_is_stable_across_two_passes_even_when_hdbscan_relabels():
    """The exact scenario INTEGRATION_ITEMS.md item 1 describes: pass one
    assigns {d1,d2,d3} to label 0; pass two clusters the very same three
    diagnoses (HDBSCAN's discovery order shifted, so this time they land on
    label "7" instead of "0"). Because identity is resolved by membership
    overlap against `diagnoses.cluster_id`, not by `uuid5(run_id, label)`,
    the second pass must resolve back to the *same* `cluster_id` - which is
    exactly what `cluster_label.needs_relabel`'s skip needs to see."""
    tables = _FakeTables()

    first_pass = write_cluster_assignments(tables.conn_fn, {"d1": 0, "d2": 0, "d3": 0})
    cluster_id_pass_one = first_pass[0]

    second_pass = write_cluster_assignments(tables.conn_fn, {"d1": 7, "d2": 7, "d3": 7})
    cluster_id_pass_two = second_pass[7]

    assert cluster_id_pass_two == cluster_id_pass_one
    # and the skip's other input, size, is also readable back unchanged
    assert tables.clusters[cluster_id_pass_one]["size"] == 3


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


@requires_db
def test_write_cluster_assignments_resolves_same_cluster_id_across_passes_against_real_db(db_conn_fn):
    """The DB-backed twin of the offline stateful test above: proves the
    majority-membership SQL actually behaves this way against real Postgres,
    not just against the fake cursor's hand-written dispatch."""
    d1 = make_diagnosis(trace_id="trace-db-1")
    d2 = make_diagnosis(trace_id="trace-db-1")
    d3 = make_diagnosis(trace_id="trace-db-1")
    for d in (d1, d2, d3):
        write_diagnosis(db_conn_fn, d)
    ids = [d1.diagnosis_id, d2.diagnosis_id, d3.diagnosis_id]

    first_pass = write_cluster_assignments(db_conn_fn, {i: 0 for i in ids})
    second_pass = write_cluster_assignments(db_conn_fn, {i: 9 for i in ids})

    assert second_pass[9] == first_pass[0]
