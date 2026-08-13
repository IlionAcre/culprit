"""Initial schema: traces, spans, steps, analysis, and benchmark tables.

Revision ID: 0001
Revises:
Create Date: 2026-08-12

Hand-written DDL, no `--autogenerate`: there are no SQLAlchemy ORM models in
this project, so every statement below is a literal `op.execute(...)` SQL
string rather than `op.create_table(...)` with SQLAlchemy Core types. That
keeps pgvector's `vector(384)` columns, JSONB, and native arrays exact
without fighting a type system this project otherwise avoids.

Ordering inside this revision is deliberate and matters:
1. `CREATE EXTENSION IF NOT EXISTS vector` first, so every later statement
   can use the `vector` type.
2. Every table, in FK-safe dependency order.
3. Only btree indexes.

HNSW indexes (on `traces.task_embedding` and `diagnoses.card_embedding`) are
deliberately **not** in this revision. They belong in revision 0002 so a
large initial backfill is not slowed by index maintenance on every insert.

Decisions carried over from the plan, restated here because they look wrong
until you know why:

- `spans` has a composite primary key `(trace_id, span_id)`. OpenTelemetry
  span ids are only unique within a trace, not globally; a single-column PK
  on `span_id` would silently collide across traces.
- `diagnoses.trace_id` is deliberately not unique (no UNIQUE constraint, no
  unique index). Re-running analysis after improving a detector must produce
  a new diagnosis alongside the old one, not overwrite it.
- `attributes` and `payload` (on `spans`) are JSONB because their shape is
  genuinely open across two attribute vocabularies plus vendor extensions.
  `kind`, `status`, and every timestamp are typed columns because every query
  filters on them. Homogeneous lists (`collapsed_span_ids`,
  `nearest_reference_trace_ids`, `cited_step_indices`, `degraded_layers`) use
  native Postgres arrays rather than JSONB for the same reason: they are
  typed and never heterogeneous.
- `spans.attributes_pruned boolean NOT NULL DEFAULT false` lives in this
  revision, not a later one, because the retention policy that flips it is
  part of the same storage contract as the column it prunes.
- `clusters.labeled_size` records the cluster's `size` at the moment it was
  last labeled, separate from the live `size`. The L5 re-label policy is "only
  re-label when the medoid changed or size grew more than 50% since it was
  last labeled," and that comparison needs the size *at labeling time*, not
  just the current count; without a stored `labeled_size`, the policy has
  nothing to diff against. `clusters.suggested_fix` holds the LLM-authored
  remediation text for the cluster, alongside the existing `label`. The prose
  column is named `description` rather than `summary`, matching this
  project's own precedent for "short code plus longer explanatory text"
  (`taxonomy.py`'s `_DESCRIPTIONS` dict for `FailureClass`), and pairs more
  naturally with `label` than `summary` does. `steps.summary` elsewhere in
  this file is unrelated (a one-line step recap, not a label's prose) and is
  intentionally left as `summary`.
- Foreign keys: only `benchmark_cases.trace_id` references `traces.trace_id`
  with an enforced constraint, matching the plan's explicit call-out of that
  one relationship. Every other `trace_id` / `diagnosis_id` cross-reference
  (spans, steps, diagnoses, signals, divergences, adjudications) is a plain
  indexed column without an FK constraint. The plan does not specify FK
  enforcement for those, and adding it here would be a guess about ingestion
  transaction ordering (bulk COPY-style inserts, deferred constraint
  semantics) that the persistence workstream (WS-B) owns, not this
  migration. If WS-B wants FK enforcement later, that is a straightforward
  revision 0002+ addition.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: Sequence[str] | str | None = None
depends_on: Sequence[str] | str | None = None


def upgrade() -> None:
    # --- 1. extension --------------------------------------------------
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # --- 2. tables -------------------------------------------------------

    op.execute(
        """
        CREATE TABLE traces (
            trace_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            outcome TEXT NOT NULL DEFAULT 'unknown',
            agent_key TEXT,
            task_key TEXT,
            task_goal TEXT,
            task_embedding vector(384),
            framework TEXT,
            root_span_id TEXT,
            span_count INTEGER NOT NULL DEFAULT 0,
            step_count INTEGER NOT NULL DEFAULT 0,
            started_at TIMESTAMPTZ,
            ended_at TIMESTAMPTZ,
            ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            ingest_error TEXT
        )
        """
    )

    op.execute(
        """
        CREATE TABLE spans (
            trace_id TEXT NOT NULL,
            span_id TEXT NOT NULL,
            parent_span_id TEXT,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            status_message TEXT,
            start_ns BIGINT NOT NULL,
            end_ns BIGINT NOT NULL,
            vocabulary TEXT NOT NULL,
            attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
            payload JSONB,
            normalize_error TEXT,
            attributes_pruned BOOLEAN NOT NULL DEFAULT false,
            PRIMARY KEY (trace_id, span_id)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE steps (
            trace_id TEXT NOT NULL,
            step_index INTEGER NOT NULL,
            span_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            actor TEXT,
            depth INTEGER NOT NULL,
            tree_path TEXT NOT NULL,
            signature TEXT NOT NULL,
            summary TEXT,
            start_ns BIGINT NOT NULL,
            end_ns BIGINT NOT NULL,
            duration_ms DOUBLE PRECISION NOT NULL,
            collapsed_span_ids TEXT[] NOT NULL DEFAULT '{}',
            PRIMARY KEY (trace_id, step_index)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE diagnoses (
            diagnosis_id UUID PRIMARY KEY,
            trace_id TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            root_cause_step_index INTEGER,
            root_cause_span_id TEXT,
            failure_class TEXT,
            confidence DOUBLE PRECISION,
            calibrated_confidence DOUBLE PRECISION,
            abstained BOOLEAN NOT NULL DEFAULT false,
            abstain_reason TEXT,
            rationale TEXT,
            counterfactual TEXT,
            candidates_considered INTEGER NOT NULL DEFAULT 0,
            card_text TEXT,
            card_embedding vector(384),
            layer_versions JSONB NOT NULL DEFAULT '{}'::jsonb,
            degraded_layers TEXT[] NOT NULL DEFAULT '{}',
            cluster_id UUID,
            error TEXT
        )
        """
    )

    op.execute(
        """
        CREATE TABLE signals (
            id BIGSERIAL PRIMARY KEY,
            diagnosis_id UUID NOT NULL,
            trace_id TEXT NOT NULL,
            detector TEXT NOT NULL,
            step_index INTEGER NOT NULL,
            span_id TEXT,
            severity DOUBLE PRECISION NOT NULL,
            category TEXT NOT NULL,
            message TEXT NOT NULL,
            evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
            error TEXT
        )
        """
    )

    op.execute(
        """
        CREATE TABLE divergences (
            id BIGSERIAL PRIMARY KEY,
            diagnosis_id UUID NOT NULL,
            trace_id TEXT NOT NULL,
            step_index INTEGER NOT NULL,
            span_id TEXT NOT NULL,
            divergence_score DOUBLE PRECISION NOT NULL,
            cliff_delta DOUBLE PRECISION NOT NULL,
            surprisal DOUBLE PRECISION NOT NULL,
            profile_surprisal DOUBLE PRECISION NOT NULL,
            reference_count INTEGER NOT NULL,
            observed_signature TEXT NOT NULL,
            expected_signatures JSONB NOT NULL DEFAULT '[]'::jsonb,
            nearest_reference_trace_ids TEXT[] NOT NULL DEFAULT '{}',
            alignment_op TEXT NOT NULL,
            error TEXT
        )
        """
    )

    op.execute(
        """
        CREATE TABLE adjudications (
            id BIGSERIAL PRIMARY KEY,
            diagnosis_id UUID NOT NULL,
            trace_id TEXT NOT NULL,
            step_index INTEGER NOT NULL,
            span_id TEXT,
            is_root_cause BOOLEAN NOT NULL,
            failure_class TEXT,
            confidence DOUBLE PRECISION,
            calibrated_confidence DOUBLE PRECISION,
            rationale TEXT,
            counterfactual TEXT,
            cited_step_indices INTEGER[] NOT NULL DEFAULT '{}',
            abstained BOOLEAN NOT NULL DEFAULT false,
            model TEXT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            cost_usd DOUBLE PRECISION,
            error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    # run_id distinguishes the current clustering pass from stale ones: a
    # recompute produces a whole new set of cluster rows rather than
    # mutating the previous run's, so "which run is live" has to be a
    # stored fact, not an inference.
    op.execute(
        """
        CREATE TABLE clusters (
            cluster_id UUID PRIMARY KEY,
            run_id UUID NOT NULL,
            label TEXT,
            description TEXT,
            suggested_fix TEXT,
            medoid_diagnosis_id UUID,
            size INTEGER NOT NULL DEFAULT 0,
            labeled_size INTEGER,
            failure_class_histogram JSONB NOT NULL DEFAULT '{}'::jsonb,
            labeled_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    op.execute(
        """
        CREATE TABLE benchmark_cases (
            case_id TEXT PRIMARY KEY,
            benchmark TEXT NOT NULL,
            trace_id TEXT NOT NULL REFERENCES traces (trace_id),
            ground_truth_step_index INTEGER,
            ground_truth_span_id TEXT,
            ground_truth_class TEXT,
            raw JSONB NOT NULL DEFAULT '{}'::jsonb
        )
        """
    )

    # --- 3. btree indexes --------------------------------------------------
    # traces: the partial HNSW index on task_embedding (WHERE outcome =
    # 'success') is deferred to revision 0002.
    op.execute("CREATE INDEX ix_traces_agent_task_outcome ON traces (agent_key, task_key, outcome)")
    op.execute("CREATE INDEX ix_traces_ingested_at ON traces (ingested_at DESC)")

    op.execute("CREATE INDEX ix_spans_trace_id ON spans (trace_id)")
    op.execute("CREATE INDEX ix_spans_trace_parent ON spans (trace_id, parent_span_id)")

    op.execute("CREATE INDEX ix_steps_signature ON steps (signature)")

    op.execute("CREATE INDEX ix_signals_trace_id ON signals (trace_id)")
    op.execute("CREATE INDEX ix_signals_diagnosis_id ON signals (diagnosis_id)")

    op.execute("CREATE INDEX ix_divergences_trace_id ON divergences (trace_id)")
    op.execute("CREATE INDEX ix_divergences_diagnosis_id ON divergences (diagnosis_id)")

    # diagnoses: the HNSW index on card_embedding is deferred to revision
    # 0002, same reasoning as traces.task_embedding above.
    op.execute("CREATE INDEX ix_diagnoses_trace_id ON diagnoses (trace_id)")
    op.execute("CREATE INDEX ix_diagnoses_failure_class ON diagnoses (failure_class)")
    op.execute("CREATE INDEX ix_diagnoses_cluster_id ON diagnoses (cluster_id)")

    op.execute("CREATE INDEX ix_adjudications_trace_id ON adjudications (trace_id)")
    op.execute("CREATE INDEX ix_adjudications_diagnosis_id ON adjudications (diagnosis_id)")

    op.execute("CREATE INDEX ix_clusters_run_id ON clusters (run_id)")

    op.execute("CREATE INDEX ix_benchmark_cases_trace_id ON benchmark_cases (trace_id)")
    op.execute("CREATE INDEX ix_benchmark_cases_benchmark ON benchmark_cases (benchmark)")


def downgrade() -> None:
    # Reverse dependency order. CASCADE drops each table's indexes with it.
    op.execute("DROP TABLE IF EXISTS benchmark_cases CASCADE")
    op.execute("DROP TABLE IF EXISTS clusters CASCADE")
    op.execute("DROP TABLE IF EXISTS adjudications CASCADE")
    op.execute("DROP TABLE IF EXISTS divergences CASCADE")
    op.execute("DROP TABLE IF EXISTS signals CASCADE")
    op.execute("DROP TABLE IF EXISTS diagnoses CASCADE")
    op.execute("DROP TABLE IF EXISTS steps CASCADE")
    op.execute("DROP TABLE IF EXISTS spans CASCADE")
    op.execute("DROP TABLE IF EXISTS traces CASCADE")
    op.execute("DROP EXTENSION IF EXISTS vector")
