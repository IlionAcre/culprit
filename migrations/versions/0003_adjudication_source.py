"""Persist Candidate.source on adjudications.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-21

Hand-written DDL, no `--autogenerate`: there are no SQLAlchemy ORM models in
this project. This revision is additive only: it adds a nullable `source`
column to `adjudications` so `Candidate.source` ("l1", "l2", "both",
"fallback", "filler") round-trips through persistence instead of being
reconstructed by inference. No backfill is needed; existing rows simply stay
NULL, and readers default to "unknown".
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: Sequence[str] | str | None = None
depends_on: Sequence[str] | str | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE adjudications ADD COLUMN source TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE adjudications DROP COLUMN IF EXISTS source")
