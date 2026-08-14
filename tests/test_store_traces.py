"""Tests for `store_traces.py`.

Two tiers, same split `test_migrations.py` already established for this
project: pure/offline tests that need no database (row-mapping helpers
exercised directly with hand-shaped dicts, and SQL-shape assertions against a
hand-rolled fake connection/cursor), and `@requires_db` tests that prove the
real thing against a live Postgres + pgvector, skipping cleanly when
`CULPRIT_TEST_DSN` is unset so the default `uv run pytest` stays offline.
"""

import time
from datetime import UTC, datetime

import pytest

from culprit.schemas import Outcome, SpanKind
from culprit.store_traces import (
    _parse_payload,
    _span_from_row,
    _step_from_row,
    _trace_from_row,
    nearest_successful,
    prune_attributes,
    read_trace,
    write_trace,
)
from culprit.synth import successful_run
from tests.conftest import requires_db

# --- offline: fake connection/cursor -----------------------------------


class _FakeCursor:
    """Records every `execute`/`executemany` call and serves canned result
    sets in order. `_results` is a queue: one entry per `execute()` call
    that expects rows back (a list of dict rows), consumed front-to-back so
    a test can script "trace row, then span rows, then step rows" for a
    single reused cursor, mirroring how `read_trace` actually uses one."""

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
    """One shared `_FakeCursor` for every `cursor()`/`execute()` call, which
    is enough to assert on call order and SQL/parameter shape without
    reimplementing Postgres. `transaction()` is a no-op context manager."""

    def __init__(self, results: list[list[dict]] | None = None):
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


# --- offline: payload round trip via synth spans ------------------------


def test_parse_payload_round_trips_every_payload_kind_from_a_real_synth_run():
    """The tricky part of the read path: `Span.payload` is a plain union
    resolved by `kind` (schemas.py), so reconstructing it from a JSONB dict
    has to route on `kind` exactly the way `normalize.py` does on the write
    side. Exercised against every payload kind a real run actually
    produces, not just one hand-picked example."""
    run = successful_run(seed=3)
    kinds_seen = set()

    for span in run.spans:
        raw = span.payload.model_dump(mode="json") if span.payload is not None else None
        rebuilt = _parse_payload(span.kind.value, raw)
        assert rebuilt == span.payload
        kinds_seen.add(span.kind)

    assert {SpanKind.AGENT, SpanKind.LLM, SpanKind.TOOL} <= kinds_seen


def test_parse_payload_returns_none_for_a_null_payload():
    assert _parse_payload(SpanKind.LLM.value, None) is None


def test_parse_payload_returns_none_for_a_kind_with_no_payload_type():
    """CHAIN/EMBEDDING/GUARDRAIL/UNKNOWN spans never carry a typed payload."""
    assert _parse_payload(SpanKind.CHAIN.value, {"anything": "here"}) is None


# --- offline: row -> model builders --------------------------------------


def test_trace_from_row_reconstructs_a_valid_trace():
    run = successful_run(seed=1)
    trace = run.trace
    row = {
        "trace_id": trace.trace_id,
        "source": trace.source,
        "outcome": trace.outcome.value,
        "agent_key": trace.agent_key,
        "task_key": trace.task_key,
        "task_goal": trace.task_goal,
        "framework": trace.framework,
        "root_span_id": trace.root_span_id,
        "span_count": trace.span_count,
        "step_count": trace.step_count,
        "started_at": trace.started_at,
        "ended_at": trace.ended_at,
        "ingested_at": datetime.now(UTC),
        "metadata": trace.metadata,
        "ingest_error": trace.ingest_error,
    }

    rebuilt = _trace_from_row(row)

    assert rebuilt.trace_id == trace.trace_id
    assert rebuilt.outcome == Outcome.SUCCESS
    assert rebuilt.agent_key == trace.agent_key
    assert rebuilt.span_count == trace.span_count


def test_span_from_row_reconstructs_a_valid_span_with_its_typed_payload():
    run = successful_run(seed=1)
    span = next(s for s in run.spans if s.kind == SpanKind.TOOL)
    row = {
        "trace_id": span.trace_id,
        "span_id": span.span_id,
        "parent_span_id": span.parent_span_id,
        "name": span.name,
        "kind": span.kind.value,
        "status": span.status.value,
        "status_message": span.status_message,
        "start_ns": span.start_ns,
        "end_ns": span.end_ns,
        "vocabulary": span.vocabulary,
        "attributes": span.attributes,
        "payload": span.payload.model_dump(mode="json"),
        "normalize_error": span.normalize_error,
    }

    rebuilt = _span_from_row(row)

    assert rebuilt == span


def test_step_from_row_defaults_null_actor_and_summary_to_empty_string():
    """The steps DDL allows NULL actor/summary but the Step model requires
    `str`; a row `write_trace` itself would never produce, but the read path
    must not crash if one exists (e.g. hand-inserted test data)."""
    row = {
        "trace_id": "t1",
        "step_index": 0,
        "span_id": "t1-sp0",
        "kind": SpanKind.LLM.value,
        "actor": None,
        "depth": 0,
        "tree_path": "0",
        "signature": "llm:agent:plan",
        "summary": None,
        "start_ns": 0,
        "end_ns": 1,
        "duration_ms": 0.001,
        "collapsed_span_ids": None,
    }

    step = _step_from_row(row)

    assert step.actor == ""
    assert step.summary == ""
    assert step.collapsed_span_ids == []


# --- offline: SQL shape against a fake connection ------------------------


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


def test_write_trace_deletes_existing_spans_and_steps_before_reinserting():
    """Re-ingesting the same trace_id must be idempotent, not a PK
    violation: write_trace deletes the old spans/steps for this trace_id
    before bulk-inserting the new ones."""
    run = successful_run(seed=2)
    conn = _FakeConnection()

    write_trace(_conn_fn(conn), run.trace, run.spans, run.steps, embedding=[0.0] * 384)

    sql_calls = [sql for sql, _ in conn.cur.calls]
    delete_spans_idx = next(i for i, s in enumerate(sql_calls) if "DELETE FROM spans" in s)
    delete_steps_idx = next(i for i, s in enumerate(sql_calls) if "DELETE FROM steps" in s)
    insert_spans_idx = next(i for i, s in enumerate(sql_calls) if "INSERT INTO spans" in s)
    insert_steps_idx = next(i for i, s in enumerate(sql_calls) if "INSERT INTO steps" in s)

    assert delete_spans_idx < insert_spans_idx
    assert delete_steps_idx < insert_steps_idx


def test_write_trace_span_and_step_row_tuples_match_column_counts():
    """A cheap but real guard against the single most common bug in this
    kind of code: an INSERT with N columns fed a tuple of a different
    length. Checked against the actual executemany call, not by rereading
    the SQL by eye."""
    run = successful_run(seed=4)
    conn = _FakeConnection()

    write_trace(_conn_fn(conn), run.trace, run.spans, run.steps)

    span_call = next(c for c in conn.cur.calls if "INSERT INTO spans" in c[0])
    step_call = next(c for c in conn.cur.calls if "INSERT INTO steps" in c[0])
    span_placeholder_count = span_call[0].count("%s")
    step_placeholder_count = step_call[0].count("%s")

    assert all(len(row) == span_placeholder_count for row in span_call[1])
    assert all(len(row) == step_placeholder_count for row in step_call[1])
    assert len(span_call[1]) == len(run.spans)
    assert len(step_call[1]) == len(run.steps)


def test_read_trace_returns_none_and_empty_lists_for_a_missing_trace_id():
    conn = _FakeConnection(results=[[]])  # trace SELECT returns no row

    trace, spans, steps = read_trace(_conn_fn(conn), "no-such-trace")

    assert trace is None
    assert spans == []
    assert steps == []


# --- requires_db: real round trips ---------------------------------------


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
def test_round_trip_write_then_read_trace_reproduces_every_field(db_conn_fn):
    run = successful_run(seed=5)
    embedding = [0.01 * i for i in range(384)]

    write_trace(db_conn_fn, run.trace, run.spans, run.steps, embedding=embedding)
    trace, spans, steps = read_trace(db_conn_fn, run.trace.trace_id)

    assert trace.trace_id == run.trace.trace_id
    assert trace.outcome == run.trace.outcome
    assert trace.task_goal == run.trace.task_goal
    assert len(spans) == len(run.spans)
    assert len(steps) == len(run.steps)
    assert {s.span_id for s in spans} == {s.span_id for s in run.spans}
    by_id = {s.span_id: s for s in spans}
    for original in run.spans:
        assert by_id[original.span_id].payload == original.payload


@requires_db
def test_vector_384_round_trips_within_tight_tolerance(db_conn_fn):
    run = successful_run(seed=6)
    embedding = [(i % 97) / 97.0 for i in range(384)]

    write_trace(db_conn_fn, run.trace, run.spans, run.steps, embedding=embedding)

    with db_conn_fn().cursor() as cur:
        cur.execute("SELECT task_embedding FROM traces WHERE trace_id = %s", (run.trace.trace_id,))
        (stored,) = cur.fetchone()

    assert list(stored) == pytest.approx(embedding, abs=1e-6)


@requires_db
def test_bulk_insert_of_5000_spans_completes_in_under_5_seconds(db_conn_fn):
    """The stated performance budget (CLAUDE.md / plan def-of-done): a naive
    per-row insert loop takes roughly 30x this. Built from one seed run's
    steps repeated with rewritten span_ids, not 5,000 realistic distinct
    spans - the point is executemany's batching cost, not payload variety."""
    run = successful_run(seed=7)
    template = run.spans[0]
    spans = [
        template.model_copy(update={"span_id": f"bulk-sp-{i}", "parent_span_id": None})
        for i in range(5000)
    ]
    trace = run.trace.model_copy(update={"span_count": len(spans)})

    start = time.perf_counter()
    write_trace(db_conn_fn, trace, spans, [])
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0


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
    import datetime as dt

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
