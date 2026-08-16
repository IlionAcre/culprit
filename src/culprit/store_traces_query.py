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
