"""HNSW indexes deferred from revision 0001, plus steps.actor/summary NOT NULL.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-15

Hand-written DDL, no `--autogenerate`, same reasoning as revision 0001: there
are no SQLAlchemy ORM models in this project. Revision 0001 is not amended;
this is a genuinely new revision per CLAUDE.md's "exactly one Alembic
revision during the parallel phase, new DDL is an Integration-phase revision
0002" rule.

Three independent pieces of DDL, in this order because the first two are
pure additions and the third mutates existing rows before constraining them:

1. **HNSW index on `traces.task_embedding`, partial `WHERE outcome =
   'success'`.** L2's reference pool (`reference.py`) only ever queries
   successful runs for nearest-neighbor traces - `nearest_successful`
   excludes failures by construction - so an index entry for a failed
   trace's embedding is pure write-and-maintenance cost with zero queries
   ever hitting it. `vector_cosine_ops` because L2's kNN lookup is cosine
   similarity, not L2 distance. `m = 16, ef_construction = 64` are
   pgvector's own documented defaults, made explicit here rather than
   implied so the tuning is a stated, deliberate choice.

2. **Ordinary (non-partial) HNSW index on `diagnoses.card_embedding`.** L5
   clustering and any future "find similar diagnoses" query read across
   every diagnosis regardless of the owning trace's outcome, so there is no
   analogous partial-index argument here. Same `vector_cosine_ops` and
   `m`/`ef_construction` as above for consistency - cosine is this
   project's similarity metric for both embedding columns
   (`cluster_embed.l2_normalize` exists specifically to make Euclidean
   distance over normalized vectors monotonic with cosine for HDBSCAN, which
   doesn't accept `metric="cosine"` directly; the stored embeddings
   themselves are still compared by cosine everywhere a raw kNN query is
   what's wanted, e.g. `nearest_successful`).

   Both indexes use a plain, transactional `CREATE INDEX` rather than
   `CREATE INDEX CONCURRENTLY`, matching revision 0001's style and keeping
   this migration a single transaction like every other Alembic revision in
   this project. `CONCURRENTLY` would avoid holding a lock during the index
   build, which matters more the larger the table already is at the moment
   this migration runs - worth revisiting as an operational follow-up if the
   corpus is large by the time this actually executes, but not a correctness
   concern for what revision 0002 is: written and reviewed, not yet run
   against a live database (no Postgres instance exists on this development
   machine - see AI_docs/INTEGRATION_ITEMS.md).

3. **`steps.actor` / `steps.summary` NOT NULL, after backfilling.**
   INTEGRATION_ITEMS.md item 5: both columns are nullable in revision 0001's
   DDL but required `str` on the `Step` Pydantic model
   (`schemas.py`), and `linearize.py` always produces a concrete, non-empty
   value for both - there is no code path that persists a `Step` with either
   field unset. `store_traces.py`'s `_step_from_row` defends with `or ""`
   only because the DDL let a NULL through in principle, never because
   linearize.py actually produces one. WS-B's own recorded recommendation
   (0001's docstring, item 5) is that the DDL was the wrong side of that
   mismatch, not the model - loosening `Step.actor`/`Step.summary` to
   `str | None` would just push the defensive `or ""` further downstream
   into every caller instead of removing it. The `UPDATE ... WHERE ... IS
   NULL` backfills are expected to be no-ops against any database this
   project has ever written to (every row was written by `linearize.py`),
   but are included anyway so this migration is correct against a
   hypothetical database with legacy or hand-inserted rows, not just the
   ones this codebase itself produced.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: Sequence[str] | str | None = None
depends_on: Sequence[str] | str | None = None


def upgrade() -> None:
    # --- 1. partial HNSW index: traces.task_embedding, successes only ----
    op.execute(
        """
        CREATE INDEX ix_traces_task_embedding_hnsw
        ON traces USING hnsw (task_embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        WHERE outcome = 'success'
        """
    )

    # --- 2. HNSW index: diagnoses.card_embedding, unrestricted -----------
    op.execute(
        """
        CREATE INDEX ix_diagnoses_card_embedding_hnsw
        ON diagnoses USING hnsw (card_embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )

    # --- 3. steps.actor / steps.summary: backfill, then NOT NULL ---------
    op.execute("UPDATE steps SET actor = '' WHERE actor IS NULL")
    op.execute("UPDATE steps SET summary = '' WHERE summary IS NULL")
    op.execute("ALTER TABLE steps ALTER COLUMN actor SET NOT NULL")
    op.execute("ALTER TABLE steps ALTER COLUMN summary SET NOT NULL")


def downgrade() -> None:
    # Reverse order. Backfilled '' values are left in place deliberately:
    # dropping the constraint does not need to (and cannot, without a second
    # guess at which rows were originally NULL) restore the pre-backfill
    # data.
    op.execute("ALTER TABLE steps ALTER COLUMN summary DROP NOT NULL")
    op.execute("ALTER TABLE steps ALTER COLUMN actor DROP NOT NULL")
    op.execute("DROP INDEX IF EXISTS ix_diagnoses_card_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_traces_task_embedding_hnsw")
