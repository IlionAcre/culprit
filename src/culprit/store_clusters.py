"""Persists `cluster_diagnoses`'s output (`cluster.py`, WS-F, HDBSCAN integer
labels: `{diagnosis_id: label}`, `-1` = noise/unclustered) into the
`clusters`/`diagnoses.cluster_id` schema.

Split out of `store_diagnoses.py` once that module grew past the ~200-line
ceiling: this is a genuinely different concern from reading/writing a
`Diagnosis` (see that module's docstring) - a bridge over a real type
mismatch, not a straightforward persistence mapping.

**The mismatch this module exists to resolve.** `diagnoses.cluster_id` and
`clusters.cluster_id` are both `UUID` (revision 0001, frozen for the
parallel phase), but `cluster_diagnoses` returns plain `int` HDBSCAN labels
with no other metadata - no `run_id`, no cluster label text, no size history.
Nothing in the plan or in `cluster_diagnoses`'s own docstring says how an
`int` label is supposed to become a `UUID` row.

**What this module assumes, stated precisely because WS-F has not been built
yet and will make its own call here:**

1. **One fresh `run_id = uuid.uuid4()` is minted on every single call to
   `write_cluster_assignments`.** It is never looked up, never passed in,
   never derived from a prior `clusters.run_id`.
2. **Cluster identity is NOT stable across two different calls.** Because
   `run_id` changes every call, `uuid5(run_id, "0")` in one `recluster_job`
   run is unrelated to `uuid5(run_id, "0")` in the next - "cluster 0" today
   and "cluster 0" next week are different `clusters` rows with different
   `cluster_id`s, even if HDBSCAN's labeling is stable. Old `clusters` rows
   from a superseded run are not deleted or merged, only left behind and no
   longer pointed to by any diagnosis whose assignment changed.
3. **Label `-1` (noise) maps to no `clusters` row at all** - the matching
   diagnoses get `cluster_id = NULL`, not a shared "noise cluster" row.
4. **Only `cluster_id`, `run_id`, and `size` are ever written to `clusters`.**
   `label`, `description`, `suggested_fix`, and `medoid_diagnosis_id` are
   left `NULL` here always - `cluster_diagnoses` supplies none of that, and
   populating them is `cluster_label.py`'s job (WS-F, not built yet).

**What breaks if WS-F assumes otherwise.** If WS-F wants cross-run cluster
identity - e.g. "cluster 3 from last week's run is still cluster 3 this
week" for a stable dashboard, or to avoid re-labeling (and re-paying the one
LLM call per cluster, see CLAUDE.md's L5 cost-control invariant) every single
recluster pass - this module's behavior is wrong for that goal today and
needs to change: either `write_cluster_assignments` takes `run_id` as a
parameter so a caller can reuse one across calls, or cluster identity must be
resolved by looking up existing `clusters` rows (e.g. by medoid similarity)
instead of unconditionally minting new ones. Flagged as an Integration item
to confirm once WS-F lands, not resolved here as a guess.
"""

import uuid

from culprit.db import ConnFn


def _cluster_uuid(run_id: uuid.UUID, label: int) -> uuid.UUID:
    return uuid.uuid5(run_id, str(label))


def _cluster_ids_for(run_id: uuid.UUID, assignment: dict[str, int]) -> dict[int, uuid.UUID]:
    """HDBSCAN's `-1` (noise, see `cluster.py`) maps to no cluster row at
    all: it means "not clustered", not "the noise cluster"."""
    labels = {label for label in assignment.values() if label != -1}
    return {label: _cluster_uuid(run_id, label) for label in labels}


def _cluster_sizes(assignment: dict[str, int]) -> dict[int, int]:
    sizes: dict[int, int] = {}
    for label in assignment.values():
        if label != -1:
            sizes[label] = sizes.get(label, 0) + 1
    return sizes


def write_cluster_assignments(conn_fn: ConnFn, assignment: dict[str, int]) -> None:
    """Mints one fresh `run_id` and a deterministic `uuid5(run_id, str(label))`
    per distinct non-noise label (see module docstring for the assumptions
    this encodes), upserts a `clusters` row per label (`size` only), then
    points each diagnosis's `cluster_id` at the matching row (or `NULL` for
    noise)."""
    if not assignment:
        return
    run_id = uuid.uuid4()
    cluster_ids = _cluster_ids_for(run_id, assignment)
    sizes = _cluster_sizes(assignment)

    conn = conn_fn()
    with conn.transaction():
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO clusters (cluster_id, run_id, size)
                VALUES (%s,%s,%s)
                ON CONFLICT (cluster_id) DO UPDATE SET size = EXCLUDED.size
                """,
                [(cid, run_id, sizes[label]) for label, cid in cluster_ids.items()],
            )
            cur.executemany(
                "UPDATE diagnoses SET cluster_id = %s WHERE diagnosis_id = %s",
                [
                    (cluster_ids.get(label), diagnosis_id)
                    for diagnosis_id, label in assignment.items()
                ],
            )
