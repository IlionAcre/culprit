"""Tests for `store_traces_query.py` (`nearest_successful`, `prune_attributes`).
Split out of `test_store_traces.py`, mirroring the module split. Same two-tier
structure: offline SQL-shape assertions against a hand-rolled fake connection,
plus `@requires_db` tests proving the real thing, skipping cleanly when
`CULPRIT_TEST_DSN` is unset.
"""

import datetime as dt

import pytest

from culprit.schemas import Outcome, SpanKind
from culprit.store_traces import read_trace, write_trace
from culprit.store_traces_query import nearest_successful, prune_attributes
from culprit.synth import successful_run
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
        import contextlib

        return contextlib.nullcontext()


def _conn_fn(conn):
    return lambda: conn


# --- offline: SQL shape ----------------------------------------------------


def test_nearest_successful_filters_on_success_outcome_and_orders_by_cosine_distance():
    """Load-bearing per CLAUDE.md: dropping the `outcome = 'success'` filter
    would silently stop using the partial HNSW index once it exists
    (revision 0002), and this is the one query in the whole module where
    that regression would be invisible without a dedicated check."""
    conn = _FakeConnection(results=[[("trace-a",), ("trace-b",)]])

    result = nearest_successful(_conn_fn(conn), [0.1] * 384, 5)

    sql, params = conn.cur.calls[0]
    assert "outcome = %s" in sql
    assert "<=>" in sql
    assert "ORDER BY" in sql
    assert params[0] == Outcome.SUCCESS.value
    assert result == ["trace-a", "trace-b"]


def test_prune_attributes_only_touches_the_spans_table():
    """`prune_attributes` must never reference `payload`, `steps`,
    `signals`, `divergences`, or `diagnoses` - checked directly against the
    executed SQL text, not just by convention."""
    conn = _FakeConnection()

    rowcount = prune_attributes(_conn_fn(conn), 90)

    sql, params = conn.cur.calls[0]
    assert "UPDATE spans" in sql
    assert "attributes_pruned = true" in sql
    for forbidden in ("payload", "steps", "signals", "divergences", "diagnoses"):
        assert forbidden not in sql.lower()
    assert params == (90,)
    assert rowcount == 0


# --- requires_db: real queries ---------------------------------------------


@pytest.fixture
def db_conn_fn():
    import psycopg
    from pgvector.psycopg import register_vector

    dsn = pytest.importorskip("os").environ["CULPRIT_TEST_DSN"]
    conn = psycopg.connect(dsn, autocommit=False)
    register_vector(conn)
    conn.execute("TRUNCATE traces, spans, steps, diagnoses, signals, divergences, adjudications, clusters CASCADE")
    conn.commit()
    yield lambda: conn
    conn.rollback()
    conn.close()


@requires_db
def test_nearest_successful_returns_successes_ordered_by_distance_never_a_failure(db_conn_fn):
    query = [1.0] + [0.0] * 383

    def _write(seed, outcome, embedding):
        run = successful_run(seed=seed)
        trace = run.trace.model_copy(update={"trace_id": f"nn-{seed}", "outcome": outcome})
        write_trace(db_conn_fn, trace, [], [], embedding=embedding)

    _write(101, Outcome.SUCCESS, [1.0] + [0.0] * 383)  # identical: distance 0
    _write(102, Outcome.SUCCESS, [0.9] + [0.1] * 383)  # close
    _write(103, Outcome.FAILURE, [1.0] + [0.0] * 383)  # identical but a failure

    result = nearest_successful(db_conn_fn, query, k=5)

    assert result[0] == "nn-101"
    assert "nn-103" not in result
    assert set(result) <= {"nn-101", "nn-102"}


@requires_db
def test_prune_attributes_empties_attributes_but_never_payload_or_steps(db_conn_fn):
    run = successful_run(seed=8)
    write_trace(db_conn_fn, run.trace, run.spans, run.steps)
    with db_conn_fn().cursor() as cur:
        cur.execute(
            "UPDATE traces SET ingested_at = %s WHERE trace_id = %s",
            (dt.datetime.now(dt.UTC) - dt.timedelta(days=200), run.trace.trace_id),
        )

    updated = prune_attributes(db_conn_fn, older_than_days=90)

    _trace, spans, steps = read_trace(db_conn_fn, run.trace.trace_id)
    with db_conn_fn().cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM spans WHERE trace_id = %s AND NOT attributes_pruned",
            (run.trace.trace_id,),
        )
        (not_pruned_count,) = cur.fetchone()

    assert updated == len(run.spans)
    assert not_pruned_count == 0  # attributes_pruned is on the DB row, not the Span model
    assert all(s.attributes == {} for s in spans)
    assert all(s.payload is not None for s in spans if s.kind in (SpanKind.LLM, SpanKind.TOOL, SpanKind.AGENT))
    assert len(steps) == len(run.steps)
