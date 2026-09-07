"""Analytical queries over already-persisted traces: the pgvector
nearest-successful-neighbor lookup and the attributes retention policy.

Split out of `store_traces.py` (which owns the write/read round trip) once
that pair grew past the ~200-line module ceiling: these two functions share
a "read-only query over the `traces`/`spans` tables" concern with a
different reason to change (index strategy, retention policy) than the
round-trip concern does (the shape of `Trace`/`Span`/`Step`), so they get
their own module rather than a forced single file. Same connection-lifecycle
rule as `store_traces.py`: `conn_fn()` is called once per function, writes
wrapped in `with conn.transaction():`.

**`nearest_successful(conn_fn, embedding, k)` is the concrete `NeighborFn`**
once `conn_fn` is bound, e.g. `functools.partial(nearest_successful, conn_fn)`.
It is deliberately not `Callable[[list[float], int], list[str]]` itself
(`ConnFn` is not part of that alias, see `signals.py`), so binding is the
caller's job, matching `contrast.py`'s injected `neighbor_fn` parameter.
"""

from culprit.db import ConnFn
from culprit.schemas import Outcome


def step_context(
    conn_fn: ConnFn, trace_id: str, step_index: int, *, window: int = 1
) -> list[dict]:
    """The step at `step_index` plus `window` steps either side of it,
    ordered by step_index, as plain dicts.

    `culprit show` names a root-cause step by index, which on its own tells
    a reader nothing: "step 4" is only meaningful next to what step 4 did.
    This reads the few columns a verdict needs (`kind`, `actor`, `summary`,
    `duration_ms`) rather than rehydrating whole `Step` models through
    `read_trace`, because the renderer needs one line per step and the full
    round trip pulls spans and payloads it would immediately discard.

    Returns `[]` when the trace or the step is absent, so a diagnosis
    persisted against a pruned or missing trace still renders its verdict
    without the surrounding context instead of failing.
    """
    conn = conn_fn()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT step_index, kind, actor, summary, duration_ms
            FROM steps
            WHERE trace_id = %s AND step_index BETWEEN %s AND %s
            ORDER BY step_index
            """,
            (trace_id, step_index - window, step_index + window),
        )
        return [
            {
                "step_index": row[0],
                "kind": row[1],
                "actor": row[2],
                "summary": row[3],
                "duration_ms": row[4],
            }
            for row in cur.fetchall()
        ]


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
            ORDER BY task_embedding <=> %s::vector
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
