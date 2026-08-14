"""Trace/Span/Step persistence: bulk write, full read, the pgvector
nearest-successful-neighbor query that backs WS-D's injected `NeighborFn`
(`signals.NeighborFn`), and the attributes retention policy.

Connection lifecycle: every function calls `conn_fn()` once and does all its
work on that connection, writes wrapped in `with conn.transaction():` so a
failure partway through never leaves a trace half-persisted. Never calls
`psycopg.connect`/`make_pool` directly (see `db.py`); callers own pooling.

**`nearest_successful(conn_fn, embedding, k)` is the concrete `NeighborFn`**
once `conn_fn` is bound, e.g. `functools.partial(nearest_successful, conn_fn)`.
It is deliberately not `Callable[[list[float], int], list[str]]` itself
(`ConnFn` is not part of that alias, see `signals.py`), so binding is the
caller's job, matching `contrast.py`'s injected `neighbor_fn` parameter.

**Ingestion composition (`otlp.decode` -> `normalize_span` -> `linearize` ->
`write_trace`) is deliberately NOT assembled here.** Not in this workstream's
contract; composing it needs `otlp.py`/`normalize.py`/`linearize.py`, which
the parallel-phase rule assigns to WS-A alone; and `tests/test_jobs.py`
(WS-G) already asserts `ingest_trace`'s default seam raises
`PersistenceNotWiredError` today, which is expected, tested behavior, not a
gap to silently close. Belongs in Integration task I2 (`pipeline.py`), the
one place already responsible for wiring cross-workstream callables.
"""

from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from culprit.db import ConnFn
from culprit.schemas import (
    AgentPayload,
    LlmPayload,
    Outcome,
    RetrievalPayload,
    Span,
    SpanKind,
    SpanStatus,
    Step,
    ToolPayload,
    Trace,
)

_PAYLOAD_TYPES: dict[SpanKind, type] = {
    SpanKind.LLM: LlmPayload,
    SpanKind.TOOL: ToolPayload,
    SpanKind.RETRIEVER: RetrievalPayload,
    SpanKind.AGENT: AgentPayload,
}


def _payload_json(span: Span) -> Jsonb | None:
    return Jsonb(span.payload.model_dump(mode="json")) if span.payload is not None else None


def _parse_payload(kind: str, raw: dict[str, Any] | None):
    """Reconstruct the correct payload submodel from `spans.kind` + the
    stored JSONB dict. Mirrors `normalize.py`'s kind-based resolution
    (`schemas.py`: `Span.payload` is a plain union, not a discriminated one,
    because the discriminant already lives on the sibling `kind` field)."""
    if raw is None:
        return None
    model = _PAYLOAD_TYPES.get(SpanKind(kind))
    return model.model_validate(raw) if model else None


def _trace_from_row(row: dict[str, Any]) -> Trace:
    return Trace(
        trace_id=row["trace_id"],
        source=row["source"],
        outcome=Outcome(row["outcome"]),
        agent_key=row["agent_key"],
        task_key=row["task_key"],
        task_goal=row["task_goal"],
        framework=row["framework"],
        root_span_id=row["root_span_id"],
        span_count=row["span_count"],
        step_count=row["step_count"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        ingested_at=row["ingested_at"],
        metadata=row["metadata"] or {},
        ingest_error=row["ingest_error"],
    )


def _span_from_row(row: dict[str, Any]) -> Span:
    return Span(
        trace_id=row["trace_id"],
        span_id=row["span_id"],
        parent_span_id=row["parent_span_id"],
        name=row["name"],
        kind=SpanKind(row["kind"]),
        status=SpanStatus(row["status"]),
        status_message=row["status_message"],
        start_ns=row["start_ns"],
        end_ns=row["end_ns"],
        vocabulary=row["vocabulary"],
        attributes=row["attributes"] or {},
        payload=_parse_payload(row["kind"], row["payload"]),
        normalize_error=row["normalize_error"],
    )


def _step_from_row(row: dict[str, Any]) -> Step:
    # actor/summary are NOT NULL on the Step model but nullable columns in
    # the steps DDL. Defaulted to "" defensively rather than raising a
    # pydantic ValidationError on a row write_trace itself would never
    # produce (schema/model looseness noted in the workstream report).
    return Step(
        trace_id=row["trace_id"],
        step_index=row["step_index"],
        span_id=row["span_id"],
        kind=SpanKind(row["kind"]),
        actor=row["actor"] or "",
        depth=row["depth"],
        tree_path=row["tree_path"],
        signature=row["signature"],
        summary=row["summary"] or "",
        start_ns=row["start_ns"],
        end_ns=row["end_ns"],
        duration_ms=row["duration_ms"],
        collapsed_span_ids=row["collapsed_span_ids"] or [],
    )


def write_trace(
    conn_fn: ConnFn,
    trace: Trace,
    spans: list[Span],
    steps: list[Step],
    embedding: list[float] | None = None,
) -> None:
    """Upserts one `traces` row, then replaces its `spans`/`steps` wholesale
    (delete-then-bulk-insert) so re-ingesting the same `trace_id` is
    idempotent rather than a PK violation. `embedding` is a separate
    parameter because `Trace` (schemas.py) deliberately carries no
    `task_embedding` field; it exists only as this Postgres column.

    Bulk inserts use one `executemany` call each for spans and steps (not a
    per-row loop): psycopg 3.1+'s `executemany` pipelines the batch over one
    round trip, which is what keeps a 5,000-span trace under the 5-second
    budget instead of the ~30x slower naive per-row form.
    """
    conn = conn_fn()
    with conn.transaction():
        conn.execute(
            """
            INSERT INTO traces (
                trace_id, source, outcome, agent_key, task_key, task_goal,
                task_embedding, framework, root_span_id, span_count,
                step_count, started_at, ended_at, ingested_at, metadata,
                ingest_error
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,COALESCE(%s, now()),%s,%s)
            ON CONFLICT (trace_id) DO UPDATE SET
                source = EXCLUDED.source,
                outcome = EXCLUDED.outcome,
                agent_key = EXCLUDED.agent_key,
                task_key = EXCLUDED.task_key,
                task_goal = EXCLUDED.task_goal,
                task_embedding = EXCLUDED.task_embedding,
                framework = EXCLUDED.framework,
                root_span_id = EXCLUDED.root_span_id,
                span_count = EXCLUDED.span_count,
                step_count = EXCLUDED.step_count,
                started_at = EXCLUDED.started_at,
                ended_at = EXCLUDED.ended_at,
                metadata = EXCLUDED.metadata,
                ingest_error = EXCLUDED.ingest_error
            """,
            (
                trace.trace_id,
                trace.source,
                trace.outcome.value,
                trace.agent_key,
                trace.task_key,
                trace.task_goal,
                embedding,
                trace.framework,
                trace.root_span_id,
                trace.span_count,
                trace.step_count,
                trace.started_at,
                trace.ended_at,
                trace.ingested_at,
                Jsonb(trace.metadata),
                trace.ingest_error,
            ),
        )
        conn.execute("DELETE FROM spans WHERE trace_id = %s", (trace.trace_id,))
        conn.execute("DELETE FROM steps WHERE trace_id = %s", (trace.trace_id,))

        if spans:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO spans (
                        trace_id, span_id, parent_span_id, name, kind, status,
                        status_message, start_ns, end_ns, vocabulary,
                        attributes, payload, normalize_error
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    [
                        (
                            s.trace_id,
                            s.span_id,
                            s.parent_span_id,
                            s.name,
                            s.kind.value,
                            s.status.value,
                            s.status_message,
                            s.start_ns,
                            s.end_ns,
                            s.vocabulary,
                            Jsonb(s.attributes),
                            _payload_json(s),
                            s.normalize_error,
                        )
                        for s in spans
                    ],
                )

        if steps:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO steps (
                        trace_id, step_index, span_id, kind, actor, depth,
                        tree_path, signature, summary, start_ns, end_ns,
                        duration_ms, collapsed_span_ids
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    [
                        (
                            st.trace_id,
                            st.step_index,
                            st.span_id,
                            st.kind.value,
                            st.actor,
                            st.depth,
                            st.tree_path,
                            st.signature,
                            st.summary,
                            st.start_ns,
                            st.end_ns,
                            st.duration_ms,
                            st.collapsed_span_ids,
                        )
                        for st in steps
                    ],
                )


def read_trace(conn_fn: ConnFn, trace_id: str) -> tuple[Trace | None, list[Span], list[Step]]:
    """`(None, [], [])` when no such trace exists, never a raise - matches
    `jobs.py`'s `_default_trace_loader`, which is the one thing that turns a
    `None` here into `TraceNotFoundError`."""
    conn = conn_fn()
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM traces WHERE trace_id = %s", (trace_id,))
        trow = cur.fetchone()
        if trow is None:
            return None, [], []
        trace = _trace_from_row(trow)

        cur.execute(
            "SELECT * FROM spans WHERE trace_id = %s ORDER BY start_ns, span_id",
            (trace_id,),
        )
        spans = [_span_from_row(r) for r in cur.fetchall()]

        cur.execute(
            "SELECT * FROM steps WHERE trace_id = %s ORDER BY step_index",
            (trace_id,),
        )
        steps = [_step_from_row(r) for r in cur.fetchall()]

    return trace, spans, steps


def nearest_successful(conn_fn: ConnFn, embedding: list[float], k: int) -> list[str]:
    """Successful traces nearest `embedding` by cosine distance
    (`vector <=> vector`), nearest first. `outcome = 'success'` is an
    explicit filter, not incidental: it is what lets this query use the
    partial HNSW index on `traces.task_embedding` (`WHERE outcome =
    'success'`, deferred to revision 0002 per CLAUDE.md) once it exists.
    Dropping the filter would still be correct today (no index exists yet)
    but would silently stop using the index the day it lands."""
    conn = conn_fn()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT trace_id FROM traces
            WHERE outcome = %s AND task_embedding IS NOT NULL
            ORDER BY task_embedding <=> %s
            LIMIT %s
            """,
            (Outcome.SUCCESS.value, embedding, k),
        )
        return [row[0] for row in cur.fetchall()]


def prune_attributes(conn_fn: ConnFn, older_than_days: int) -> int:
    """Empties `spans.attributes` and sets `attributes_pruned = true` for
    spans belonging to a trace ingested more than `older_than_days` ago.
    Returns the number of spans updated. Never touches `payload`, `steps`,
    `signals`, `divergences`, or `diagnoses` - those tables are not
    referenced by this query at all, by construction, not by care taken not
    to update them."""
    conn = conn_fn()
    with conn.transaction():
        cur = conn.execute(
            """
            UPDATE spans SET attributes = '{}'::jsonb, attributes_pruned = true
            WHERE attributes_pruned = false
              AND trace_id IN (
                  SELECT trace_id FROM traces
                  WHERE ingested_at < now() - make_interval(days => %s)
              )
            """,
            (older_than_days,),
        )
        return cur.rowcount
