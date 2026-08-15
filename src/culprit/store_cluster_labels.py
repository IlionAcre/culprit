"""INTEGRATION_ITEMS.md item 3, resolved: persists `cluster_label.py`'s
output (`card_text`/`card_embedding` are handled separately - see below -
plus `label`, `description`, `suggested_fix`, `labeled_at`, `labeled_size`,
`medoid_diagnosis_id`) into the `clusters` rows `store_clusters.py` already
created this pass, and reads back the previous-pass state
`cluster_label.needs_relabel` needs to decide whether a cluster can skip
re-labeling.

Split from `store_clusters.py` for the same reason `store_diagnoses.py` and
`store_clusters.py` are split: reading/writing label text is one concern
(depends on `cluster_label.py`'s `ClusterLabel`/`ClusterLabelState` shapes),
resolving `int label -> UUID cluster_id` identity is a different one
(depends only on `diagnoses.cluster_id`). This module is the second half of
item 1's fix, not just item 3: `needs_relabel`'s skip only fires if
`read_cluster_label_states` can find a real previous row, which requires
`store_clusters.write_cluster_assignments` to have handed back a *stable*
`cluster_id` for a cluster that existed last pass - the identity resolution
this module's sibling now does.

**Why only relabeled clusters get written back.** `label_clusters` returns
one combined list mixing clusters that were actually re-labeled this pass
and clusters whose prior label was reused verbatim (the skip firing). If
`write_cluster_labels` blindly wrote every result, a reused label's `label`/
`description`/`suggested_fix` text would round-trip unchanged, but
`labeled_at` would advance to "now" every single pass regardless of whether
an LLM call happened - lying about when the cluster was actually last
summarized. `_was_relabeled` re-runs `needs_relabel` (a pure, free function)
against the same `previous` state `label_clusters` itself consulted, to
recover which results were fresh without threading an extra flag through
`ClusterLabel`.

**`card_text`/`card_embedding` are not written here.** Those columns live on
`diagnoses`, not `clusters` (see `store_diagnoses.py`'s docstring), and
nothing in the plan specifies a caller that renders and embeds a card purely
to persist it outside of the clustering pass that already consumes it
in-memory (`cluster.embed_cards`). Left as a known gap, not guessed at.
"""

import uuid

from psycopg.rows import dict_row

from culprit.cluster_embed import embed_cards
from culprit.cluster_label import ClusterLabel, ClusterLabelState, label_clusters, needs_relabel
from culprit.db import ConnFn
from culprit.embed import EmbedFn
from culprit.signals import Diagnosis

# Mirrors cluster_label.py's own defaults exactly (duplicated per CLAUDE.md's
# "duplicate small helpers rather than sharing them" convention, applied here
# to constants for the same reason: one small value is not worth a shared
# import surface between two modules with different reasons to change).
_DEFAULT_MODEL = "gemini/gemini-2.5-flash-lite"
_DEFAULT_MAX_WORKERS = 4


def read_cluster_label_states(
    conn_fn: ConnFn, cluster_ids: dict[int, uuid.UUID]
) -> dict[int, ClusterLabelState]:
    """`label -> ClusterLabelState` for every one of this pass's clusters
    that was actually labeled by some earlier pass. A `clusters` row with
    `label IS NULL` (brand new this pass, or matched to a pre-item-3 row
    that predates label persistence) is excluded, so `needs_relabel` sees
    `previous=None` for it and labels it - the same "no prior state" path a
    cluster's genuine first labeling pass takes."""
    if not cluster_ids:
        return {}
    label_by_cluster_id = {cid: label for label, cid in cluster_ids.items()}
    conn = conn_fn()
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT cluster_id, label, description, suggested_fix, "
            "medoid_diagnosis_id, labeled_size FROM clusters "
            "WHERE cluster_id = ANY(%s) AND label IS NOT NULL",
            (list(cluster_ids.values()),),
        )
        rows = cur.fetchall()
    return {
        label_by_cluster_id[row["cluster_id"]]: ClusterLabelState(
            medoid_diagnosis_id=row["medoid_diagnosis_id"],
            labeled_size=row["labeled_size"],
            label=row["label"],
            description=row["description"],
            suggested_fix=row["suggested_fix"],
        )
        for row in rows
        if row["cluster_id"] in label_by_cluster_id
    }


def _was_relabeled(result: ClusterLabel, previous: dict[int, ClusterLabelState]) -> bool:
    prior = previous.get(result.cluster_label_id)
    return needs_relabel(result.medoid_diagnosis_id, result.labeled_size, prior)


def write_cluster_labels(
    conn_fn: ConnFn,
    cluster_ids: dict[int, uuid.UUID],
    labels: list[ClusterLabel],
    previous: dict[int, ClusterLabelState] | None = None,
) -> None:
    """Writes only the results that were actually re-labeled this pass (see
    module docstring for why reused results are skipped) and carry no
    `error` (per-item error isolation: a failed labeling attempt must not
    wipe out a cluster's last good label)."""
    previous = previous or {}
    rows = [
        (
            result.label,
            result.description,
            result.suggested_fix,
            result.medoid_diagnosis_id,
            result.labeled_size,
            cluster_ids[result.cluster_label_id],
        )
        for result in labels
        if result.error is None
        and result.cluster_label_id in cluster_ids
        and _was_relabeled(result, previous)
    ]
    if not rows:
        return
    conn = conn_fn()
    with conn.transaction():
        with conn.cursor() as cur:
            cur.executemany(
                """
                UPDATE clusters
                SET label = %s, description = %s, suggested_fix = %s,
                    medoid_diagnosis_id = %s, labeled_size = %s, labeled_at = now()
                WHERE cluster_id = %s
                """,
                rows,
            )


def label_and_persist_clusters(
    conn_fn: ConnFn,
    diagnoses: list[Diagnosis],
    assignment: dict[str, int],
    cluster_ids: dict[int, uuid.UUID],
    *,
    embed_fn: EmbedFn,
    model: str = _DEFAULT_MODEL,
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> list[ClusterLabel]:
    """Orchestrates one recluster pass's labeling: embed only the clustered
    (non-noise) diagnoses, read back previous-pass label state for the
    stable `cluster_id`s `store_clusters.write_cluster_assignments` already
    resolved, run `cluster_label.label_clusters` (which skips re-labeling
    per `needs_relabel`), and persist whatever actually changed.

    A diagnosis with `assignment[id] == -1` never reaches `embed_cards`:
    noise is either genuinely unclustered or failed to render
    (`cluster.py`'s `_partition_renderable`), and a card that could not be
    rendered there cannot be rendered here either.
    """
    clustered = [d for d in diagnoses if assignment.get(d.diagnosis_id, -1) != -1]
    if not clustered:
        return []

    embeddings = dict(zip((d.diagnosis_id for d in clustered), embed_cards(clustered, embed_fn=embed_fn)))
    previous = read_cluster_label_states(conn_fn, cluster_ids)

    results = label_clusters(
        clustered, assignment, embeddings, model=model, max_workers=max_workers, previous=previous
    )
    write_cluster_labels(conn_fn, cluster_ids, results, previous)
    return results
