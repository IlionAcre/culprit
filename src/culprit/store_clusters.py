"""Persists `cluster_diagnoses`'s output (`cluster.py`, WS-F, HDBSCAN integer
labels: `{diagnosis_id: label}`, `-1` = noise/unclustered) into the
`clusters`/`diagnoses.cluster_id` schema.

Split out of `store_diagnoses.py` once that module grew past the ~200-line
ceiling: this is a genuinely different concern from reading/writing a
`Diagnosis` (see that module's docstring) - a bridge over a real type
mismatch, not a straightforward persistence mapping.

**INTEGRATION_ITEMS.md item 1, resolved: cluster identity is now resolved by
matching, not minted.** This module used to derive `cluster_id` as
`uuid5(run_id, str(label))` with a fresh `run_id` per call, which meant
cluster identity reset on every pass: `cluster_label.needs_relabel()`'s skip
(the L5 cost-control mechanism, CLAUDE.md) could never fire because it never
saw the same `cluster_id` twice.

**Why matching by membership, not threading a stable `run_id`.** The
obvious-looking fix - keep reusing one `run_id` across passes so
`uuid5(run_id, label)` stays constant - does not actually work: HDBSCAN's
integer labels are an artifact of cluster *discovery order* within one
`fit_predict` call, not a stable name for a cluster's content. A cluster
that was "label 0" last pass can easily come back as "label 3" this pass
even when its membership barely moved (a new diagnosis sorted earlier
changes discovery order), and two unrelated clusters can coincidentally both
be born "label 0" on different passes. Pinning `run_id` would make
`uuid5(run_id, label)` collide on unrelated clusters and miss real matches
just as often as it does today - it repairs the symptom (identity resets)
without repairing the cause (labels never meant identity).

**What actually identifies a cluster across passes is who is in it.**
`resolve_cluster_identity` reads which `cluster_id` each of this pass's
diagnoses currently points to (`diagnoses.cluster_id`, written by the
*previous* call to `write_cluster_assignments`) and, for each of this pass's
label groups, takes a majority vote: if more than half of a new group's
members already point at the same old `cluster_id`, that new group *is* the
old cluster and inherits its UUID. A previously persisted `cluster_id` can
only be claimed by one new group (the one with the strongest overlap) so a
cluster that HDBSCAN splits in two does not hand the same identity to both
halves. Groups with no majority match - a genuinely new cluster, or one so
reshuffled that no old cluster contributed most of its members - mint a
fresh `uuid.uuid4()`, same as every cluster did before this fix.

This is membership overlap, not "medoid similarity" by cosine distance:
membership is exact (diagnosis ids either match or they don't) and needs no
embeddings or similarity threshold at this layer, only the already-durable
`diagnoses.cluster_id` column. `cluster_label.needs_relabel` still makes its
own, separate medoid-drift judgement once identity is resolved - the two
checks answer different questions ("which old cluster is this" versus "has
this cluster's center moved enough to need a new summary") and conflating
them would make a bigger, harder-to-test function out of two simple ones.

**What this module still assumes**, unchanged from before this fix:

1. Label `-1` (noise) maps to no `clusters` row at all - the matching
   diagnoses get `cluster_id = NULL`, not a shared "noise cluster" row.
2. Only `cluster_id`, `run_id`, and `size` are ever written to `clusters`
   here. `label`, `description`, `suggested_fix`, `medoid_diagnosis_id`, and
   `labeled_size` are `store_cluster_labels.py`'s job (item 3), which runs
   after this module using the `dict[int, uuid.UUID]` this module now
   returns.
3. Old `clusters` rows that lose every diagnosis to reassignment are not
   deleted, only left unreferenced - pruning stale unlabeled cluster rows is
   a retention concern, not this module's.
"""

import uuid

from culprit.db import ConnFn


def _cluster_sizes(assignment: dict[str, int]) -> dict[int, int]:
    sizes: dict[int, int] = {}
    for label in assignment.values():
        if label != -1:
            sizes[label] = sizes.get(label, 0) + 1
    return sizes


def _group_by_label(assignment: dict[str, int]) -> dict[int, list[str]]:
    """HDBSCAN's `-1` (noise) is excluded here for the same reason it is
    excluded everywhere else in this codebase (see `cluster.py`,
    `cluster_label.py`): it means "not clustered", not "the noise cluster",
    so it never participates in identity resolution."""
    groups: dict[int, list[str]] = {}
    for diagnosis_id, label in assignment.items():
        if label != -1:
            groups.setdefault(label, []).append(diagnosis_id)
    return groups


def _previous_membership(conn_fn: ConnFn, diagnosis_ids: list[str]) -> dict[str, uuid.UUID]:
    """`diagnosis_id -> the cluster_id it currently points to`, for whichever
    of this pass's diagnoses were already clustered by an earlier pass. This
    is the only fact identity resolution needs; everything else (label text,
    medoid, size history) is `cluster_label_store.py`'s concern."""
    if not diagnosis_ids:
        return {}
    conn = conn_fn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT diagnosis_id, cluster_id FROM diagnoses "
            "WHERE diagnosis_id = ANY(%s) AND cluster_id IS NOT NULL",
            (diagnosis_ids,),
        )
        return {str(diagnosis_id): cluster_id for diagnosis_id, cluster_id in cur.fetchall()}


def resolve_cluster_identity(conn_fn: ConnFn, assignment: dict[str, int]) -> dict[int, uuid.UUID]:
    """Map this pass's HDBSCAN integer labels to stable `clusters.cluster_id`
    UUIDs: reuse a previously persisted `cluster_id` when a majority of a new
    group's members already pointed at it, otherwise mint a fresh one. See
    the module docstring for why majority-of-membership is the right test
    and threading a stable `run_id` is not. Deterministic tie-break
    (`-count, label`) so the same input always resolves the same way."""
    groups = _group_by_label(assignment)
    if not groups:
        return {}
    membership = _previous_membership(conn_fn, list(assignment.keys()))

    candidates: dict[int, tuple[uuid.UUID, int]] = {}
    for label, ids in groups.items():
        counts: dict[uuid.UUID, int] = {}
        for diagnosis_id in ids:
            prev_cluster_id = membership.get(diagnosis_id)
            if prev_cluster_id is not None:
                counts[prev_cluster_id] = counts.get(prev_cluster_id, 0) + 1
        if counts:
            best_cluster_id, best_count = max(counts.items(), key=lambda kv: (kv[1], str(kv[0])))
            if best_count * 2 > len(ids):  # strict majority
                candidates[label] = (best_cluster_id, best_count)

    # A persisted cluster_id can be claimed by only one new label - the one
    # with the strongest overlap - so a cluster HDBSCAN split in two does not
    # hand the same identity to both halves.
    winning_label_by_cluster_id: dict[uuid.UUID, int] = {}
    for label, (cluster_id, count) in sorted(
        candidates.items(), key=lambda kv: (-kv[1][1], kv[0])
    ):
        winning_label_by_cluster_id.setdefault(cluster_id, label)
    label_to_cluster_id = {label: cid for cid, label in winning_label_by_cluster_id.items()}

    return {
        label: label_to_cluster_id.get(label, uuid.uuid4())
        for label in groups
    }


def write_cluster_assignments(conn_fn: ConnFn, assignment: dict[str, int]) -> dict[int, uuid.UUID]:
    """Resolve each label to a stable `cluster_id` (see
    `resolve_cluster_identity`), upsert a `clusters` row per label (`size`
    and `run_id` only - `run_id` records which pass last touched the row,
    matching revision 0001's "which run is live" rationale), then point each
    diagnosis's `cluster_id` at the matching row (or `NULL` for noise).
    Returns the `label -> cluster_id` map so a caller can persist labels
    (`store_cluster_labels.py`) against the same stable identities."""
    if not assignment:
        return {}
    cluster_ids = resolve_cluster_identity(conn_fn, assignment)
    run_id = uuid.uuid4()
    sizes = _cluster_sizes(assignment)

    conn = conn_fn()
    with conn.transaction():
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO clusters (cluster_id, run_id, size)
                VALUES (%s,%s,%s)
                ON CONFLICT (cluster_id) DO UPDATE SET size = EXCLUDED.size, run_id = EXCLUDED.run_id
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
    return cluster_ids
