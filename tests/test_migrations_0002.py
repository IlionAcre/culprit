"""Tests for Alembic revision 0002 (HNSW indexes deferred from revision
0001, plus `steps.actor`/`steps.summary` NOT NULL). Same split as
`test_migrations.py`: structural/offline-SQL checks run unconditionally,
everything asserting on live database state is `requires_db`-gated and skips
cleanly without `CULPRIT_TEST_DSN`.

Per AI_docs/INTEGRATION_ITEMS.md: there is no live Postgres on this
development machine, so every `requires_db` test below is written and
reviewed but has not actually been run against a real database.
"""

import os
import subprocess
import sys
from pathlib import Path

import psycopg
import pytest

from tests.conftest import requires_db

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MIGRATION_FILE = _REPO_ROOT / "migrations" / "versions" / "0002_hnsw_indexes_and_step_not_null.py"


def _connect():
    return psycopg.connect(os.environ["CULPRIT_TEST_DSN"], autocommit=True)


def _run_alembic(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=_REPO_ROOT,
        env=dict(os.environ),
        capture_output=True,
        text=True,
    )


def _run_alembic_offline(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault(
        "CULPRIT_TEST_DSN", "postgresql://user:pass@localhost:5432/culprit_offline"
    )
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_revision_0002_chains_onto_0001_not_a_second_root():
    text = _MIGRATION_FILE.read_text()

    assert 'revision: str = "0002"' in text
    assert 'down_revision: str | None = "0001"' in text


def test_revision_0001_is_not_amended_by_this_revision():
    """CLAUDE.md: revision 0001 stays frozen; new DDL is a new revision. Only
    checks executable lines, not the whole file: 0001's own docstring
    mentions "HNSW" in prose to explain the deferral, same reasoning as
    test_migrations.py's test_revision_0001_never_creates_an_hnsw_index."""
    original = (_REPO_ROOT / "migrations" / "versions" / "0001_initial_schema.py").read_text()
    executable_lines = [
        line for line in original.splitlines() if "op.execute" in line or "CREATE INDEX" in line
    ]

    assert not any("hnsw" in line.lower() for line in executable_lines)
    assert not any("ix_traces_task_embedding_hnsw" in line for line in executable_lines)
    assert not any("ix_diagnoses_card_embedding_hnsw" in line for line in executable_lines)


def test_revision_0002_task_embedding_index_is_partial_on_success_and_uses_cosine_ops():
    text = _MIGRATION_FILE.read_text()

    start = text.index("ix_traces_task_embedding_hnsw")
    end = text.index("ix_diagnoses_card_embedding_hnsw")
    block = text[start:end]

    assert "USING hnsw (task_embedding vector_cosine_ops)" in block
    assert "WITH (m = 16, ef_construction = 64)" in block
    assert "WHERE outcome = 'success'" in block


def test_revision_0002_card_embedding_index_is_unrestricted_and_uses_cosine_ops():
    text = _MIGRATION_FILE.read_text()

    start = text.index("ix_diagnoses_card_embedding_hnsw")
    end = text.index("def downgrade")
    block = text[start:end]

    assert "USING hnsw (card_embedding vector_cosine_ops)" in block
    # unrestricted: no WHERE clause anywhere between this index and the
    # steps backfill/NOT NULL statements that follow it
    assert "WHERE" not in block.split("steps SET actor")[0]


def test_revision_0002_backfills_steps_before_adding_not_null():
    """Ordering matters: SET NOT NULL against any pre-existing NULL row would
    fail outright, so the backfill UPDATEs must precede both ALTER COLUMN
    statements."""
    text = _MIGRATION_FILE.read_text()

    backfill_actor = text.index("UPDATE steps SET actor")
    backfill_summary = text.index("UPDATE steps SET summary")
    not_null_actor = text.index("ALTER COLUMN actor SET NOT NULL")
    not_null_summary = text.index("ALTER COLUMN summary SET NOT NULL")

    assert backfill_actor < not_null_actor
    assert backfill_summary < not_null_summary


def test_alembic_upgrade_to_0002_renders_valid_sql_offline_and_includes_new_ddl():
    result = _run_alembic_offline("upgrade", "0001:0002", "--sql")

    assert result.returncode == 0, result.stderr
    out = result.stdout.lower()
    assert "hnsw" in out
    assert "alter table steps" in out
    assert "not null" in out


def test_alembic_downgrade_to_0001_renders_valid_sql_offline():
    result = _run_alembic_offline("downgrade", "0002:0001", "--sql")

    assert result.returncode == 0, result.stderr
    out = result.stdout.lower()
    assert "drop index if exists ix_traces_task_embedding_hnsw" in out
    assert "drop index if exists ix_diagnoses_card_embedding_hnsw" in out
    assert "drop not null" in out


@requires_db
def test_migration_creates_both_hnsw_indexes():
    _run_alembic("upgrade", "head")

    with _connect() as conn:
        rows = conn.execute(
            "SELECT indexname, indexdef FROM pg_indexes "
            "WHERE indexname IN ('ix_traces_task_embedding_hnsw', 'ix_diagnoses_card_embedding_hnsw')"
        ).fetchall()

    names = {r[0] for r in rows}
    assert names == {"ix_traces_task_embedding_hnsw", "ix_diagnoses_card_embedding_hnsw"}
    defs = {r[0]: r[1] for r in rows}
    assert "hnsw" in defs["ix_traces_task_embedding_hnsw"].lower()
    assert "outcome" in defs["ix_traces_task_embedding_hnsw"].lower()
    assert "hnsw" in defs["ix_diagnoses_card_embedding_hnsw"].lower()


@requires_db
def test_migration_makes_steps_actor_and_summary_not_null():
    _run_alembic("upgrade", "head")

    with _connect() as conn:
        rows = conn.execute(
            "SELECT column_name, is_nullable FROM information_schema.columns "
            "WHERE table_name = 'steps' AND column_name IN ('actor', 'summary')"
        ).fetchall()

    assert {r[0]: r[1] for r in rows} == {"actor": "NO", "summary": "NO"}


@requires_db
def test_migration_backfills_preexisting_null_actor_and_summary_before_constraining(monkeypatch):
    """The realistic case this backfill exists for: apply 0001 only, insert a
    row with NULL actor/summary (as 0001's own nullable DDL allowed), then
    upgrade to 0002 and confirm the backfill ran rather than the ALTER TABLE
    simply failing against pre-existing NULLs."""
    _run_alembic("downgrade", "base")
    _run_alembic("upgrade", "0001")

    with _connect() as conn:
        conn.execute(
            "INSERT INTO traces (trace_id, source, outcome) VALUES ('t-nn', 'test', 'failure')"
        )
        conn.execute(
            "INSERT INTO steps (trace_id, step_index, span_id, kind, depth, tree_path, "
            "signature, start_ns, end_ns, duration_ms) VALUES "
            "('t-nn', 0, 's-0', 'llm', 0, '0', 'sig', 0, 1, 1.0)"
        )

    result = _run_alembic("upgrade", "0002")
    assert result.returncode == 0, result.stderr

    with _connect() as conn:
        actor, summary = conn.execute(
            "SELECT actor, summary FROM steps WHERE trace_id = 't-nn'"
        ).fetchone()

    assert actor == ""
    assert summary == ""
