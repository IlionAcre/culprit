"""`Diagnosis` persistence: one `diagnoses` row plus its child `signals`,
`divergences`, and `adjudications` rows, written and reassembled together so
a `Diagnosis` round-trips whole.

Also home to the two functions `jobs.py`'s own module docstring names as its
best guess for operations no contract covered - `read_all_diagnoses` (batch
reclustering reader) and `write_cluster_assignments` (persists L5's output) -
matched here exactly, not renamed, so WS-G's `_seam()` resolves both without
an integration-time handshake.

`diagnoses.card_text` / `card_embedding` / `cluster_id` are storage-only
columns with no counterpart on the `Diagnosis` model, same pattern as
`traces.task_embedding` (`schemas.py`). `write_diagnosis` leaves all three
NULL; `write_cluster_assignments` is the only writer of `cluster_id`.
`card_text`/`card_embedding` have no writer here at all - rendering and
embedding the card is `cluster_embed.py`'s job (WS-F, not built yet),
flagged as an Integration item rather than guessed at.
"""

import uuid
from dataclasses import asdict
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from culprit.db import ConnFn
from culprit.signals import Adjudication, Diagnosis, DivergenceCandidate, Evidence, Signal


def write_diagnosis(conn_fn: ConnFn, diagnosis: Diagnosis) -> None:
    """One `diagnoses` row plus its `signals`/`divergences`/`adjudications`
    children, all in one transaction. `diagnoses.trace_id` deliberately has
    no unique constraint (CLAUDE.md): re-running analysis after improving a
    detector inserts a new row alongside any prior diagnosis for the same
    trace, it never overwrites one."""
    conn = conn_fn()
    with conn.transaction():
        conn.execute(
            """
            INSERT INTO diagnoses (
                diagnosis_id, trace_id, created_at, root_cause_step_index,
                root_cause_span_id, failure_class, confidence,
                calibrated_confidence, abstained, abstain_reason, rationale,
                counterfactual, candidates_considered, layer_versions,
                degraded_layers, error
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                diagnosis.diagnosis_id,
                diagnosis.trace_id,
                diagnosis.created_at,
                diagnosis.root_cause_step_index,
                diagnosis.root_cause_span_id,
                diagnosis.failure_class,
                diagnosis.confidence,
                diagnosis.calibrated_confidence,
                diagnosis.abstained,
                diagnosis.abstain_reason,
                diagnosis.rationale,
                diagnosis.counterfactual,
                diagnosis.candidates_considered,
                Jsonb(diagnosis.layer_versions),
                diagnosis.degraded_layers,
                diagnosis.error,
            ),
        )

        if diagnosis.signals:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO signals (
                        diagnosis_id, trace_id, detector, step_index, span_id,
                        severity, category, message, evidence, error
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    [
                        (
                            diagnosis.diagnosis_id,
                            diagnosis.trace_id,
                            s.detector,
                            s.step_index,
                            s.span_id,
                            s.severity,
                            s.category,
                            s.message,
                            Jsonb([asdict(e) for e in s.evidence]),
                            s.error,
                        )
                        for s in diagnosis.signals
                    ],
                )

        if diagnosis.divergences:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO divergences (
                        diagnosis_id, trace_id, step_index, span_id,
                        divergence_score, cliff_delta, surprisal,
                        profile_surprisal, reference_count, observed_signature,
                        expected_signatures, nearest_reference_trace_ids,
                        alignment_op, error
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    [
                        (
                            diagnosis.diagnosis_id,
                            diagnosis.trace_id,
                            d.step_index,
                            d.span_id,
                            d.divergence_score,
                            d.cliff_delta,
                            d.surprisal,
                            d.profile_surprisal,
                            d.reference_count,
                            d.observed_signature,
                            Jsonb([list(pair) for pair in d.expected_signatures]),
                            d.nearest_reference_trace_ids,
                            d.alignment_op,
                            d.error,
                        )
                        for d in diagnosis.divergences
                    ],
                )

        if diagnosis.adjudications:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO adjudications (
                        diagnosis_id, trace_id, step_index, span_id,
                        is_root_cause, failure_class, confidence,
                        calibrated_confidence, rationale, counterfactual,
                        cited_step_indices, abstained, model, prompt_tokens,
                        completion_tokens, cost_usd, error
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    [
                        (
                            diagnosis.diagnosis_id,
                            diagnosis.trace_id,
                            a.step_index,
                            a.span_id,
                            a.is_root_cause,
                            a.failure_class,
                            a.confidence,
                            a.calibrated_confidence,
                            a.rationale,
                            a.counterfactual,
                            a.cited_step_indices,
                            a.abstained,
                            a.model,
                            a.prompt_tokens,
                            a.completion_tokens,
                            a.cost_usd,
                            a.error,
                        )
                        for a in diagnosis.adjudications
                    ],
                )


def _diagnosis_from_row(
    row: dict[str, Any],
    signals: list[Signal],
    divergences: list[DivergenceCandidate],
    adjudications: list[Adjudication],
) -> Diagnosis:
    return Diagnosis(
        diagnosis_id=str(row["diagnosis_id"]),
        trace_id=row["trace_id"],
        created_at=row["created_at"],
        root_cause_step_index=row["root_cause_step_index"],
        root_cause_span_id=row["root_cause_span_id"],
        failure_class=row["failure_class"],
        confidence=row["confidence"],
        calibrated_confidence=row["calibrated_confidence"],
        abstained=row["abstained"],
        abstain_reason=row["abstain_reason"],
        rationale=row["rationale"] or "",
        counterfactual=row["counterfactual"] or "",
        candidates_considered=row["candidates_considered"],
        signals=signals,
        divergences=divergences,
        adjudications=adjudications,
        layer_versions=row["layer_versions"] or {},
        degraded_layers=row["degraded_layers"] or [],
        error=row["error"],
    )


def _signal_from_row(row: dict[str, Any]) -> Signal:
    return Signal(
        detector=row["detector"],
        step_index=row["step_index"],
        span_id=row["span_id"],
        severity=row["severity"],
        category=row["category"],
        message=row["message"],
        evidence=[Evidence(**e) for e in (row["evidence"] or [])],
        error=row["error"],
    )


def _divergence_from_row(row: dict[str, Any]) -> DivergenceCandidate:
    return DivergenceCandidate(
        step_index=row["step_index"],
        span_id=row["span_id"],
        divergence_score=row["divergence_score"],
        cliff_delta=row["cliff_delta"],
        surprisal=row["surprisal"],
        profile_surprisal=row["profile_surprisal"],
        reference_count=row["reference_count"],
        observed_signature=row["observed_signature"],
        expected_signatures=[tuple(p) for p in (row["expected_signatures"] or [])],
        nearest_reference_trace_ids=row["nearest_reference_trace_ids"] or [],
        alignment_op=row["alignment_op"],
        error=row["error"],
    )


def _adjudication_from_row(row: dict[str, Any]) -> Adjudication:
    return Adjudication(
        step_index=row["step_index"],
        span_id=row["span_id"],
        is_root_cause=row["is_root_cause"],
        failure_class=row["failure_class"],
        confidence=row["confidence"],
        calibrated_confidence=row["calibrated_confidence"],
        rationale=row["rationale"] or "",
        counterfactual=row["counterfactual"] or "",
        cited_step_indices=row["cited_step_indices"] or [],
        abstained=row["abstained"],
        model=row["model"],
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        cost_usd=row["cost_usd"],
        error=row["error"],
    )


def _fetch_children(cur, table: str, diagnosis_ids: list, builder) -> dict[str, list]:
    grouped: dict[str, list] = {str(did): [] for did in diagnosis_ids}
    if not diagnosis_ids:
        return grouped
    cur.execute(
        f"SELECT * FROM {table} WHERE diagnosis_id = ANY(%s) ORDER BY diagnosis_id, step_index",
        (diagnosis_ids,),
    )
    for row in cur.fetchall():
        grouped[str(row["diagnosis_id"])].append(builder(row))
    return grouped


def _assemble(cur, rows: list[dict[str, Any]]) -> list[Diagnosis]:
    ids = [r["diagnosis_id"] for r in rows]
    signals = _fetch_children(cur, "signals", ids, _signal_from_row)
    divergences = _fetch_children(cur, "divergences", ids, _divergence_from_row)
    adjudications = _fetch_children(cur, "adjudications", ids, _adjudication_from_row)
    return [
        _diagnosis_from_row(
            r,
            signals[str(r["diagnosis_id"])],
            divergences[str(r["diagnosis_id"])],
            adjudications[str(r["diagnosis_id"])],
        )
        for r in rows
    ]


def read_diagnoses(conn_fn: ConnFn, trace_id: str) -> list[Diagnosis]:
    """Every persisted diagnosis for one trace, oldest first. Deliberately
    not limited to one row: `diagnoses.trace_id` is not unique (see
    `write_diagnosis`)."""
    conn = conn_fn()
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT * FROM diagnoses WHERE trace_id = %s ORDER BY created_at", (trace_id,)
        )
        return _assemble(cur, cur.fetchall())


def read_all_diagnoses(conn_fn: ConnFn) -> list[Diagnosis]:
    """Every diagnosis in the corpus, for `recluster_job`'s batch pass
    (`jobs.py`). Not scoped to one trace, and not part of the plan's
    originally documented contract - `jobs.py`'s docstring names this
    function; matched here rather than renamed."""
    conn = conn_fn()
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM diagnoses ORDER BY created_at")
        return _assemble(cur, cur.fetchall())


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
    """Persists `cluster_diagnoses`'s `{diagnosis_id: label}` output (WS-F,
    HDBSCAN integer labels, `-1` = noise/unclustered).

    Neither the plan nor `cluster_diagnoses`'s contract documents this
    function, and a real type mismatch had to be resolved to write it:
    `diagnoses.cluster_id` and `clusters.cluster_id` are both `UUID`
    (revision 0001, frozen for the parallel phase), but `cluster_diagnoses`
    returns plain `int` labels with no other metadata. This mints one fresh
    `run_id` per call and a deterministic `uuid5(run_id, str(label))` per
    distinct non-noise label, upserts a `clusters` row per label (`size`
    only - `label`/`description`/`suggested_fix`/`medoid_diagnosis_id` are
    left NULL, since `cluster_diagnoses` supplies none of that and setting
    them is `cluster_label.py`'s job, WS-F, not built yet), then points each
    diagnosis's `cluster_id` at the matching row (or NULL for noise).
    Flagged as an Integration item to confirm once WS-F lands: this bridges
    the gap defensibly, but WS-F may want a different `run_id`/label
    lifecycle than "one new run per call, always"."""
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
