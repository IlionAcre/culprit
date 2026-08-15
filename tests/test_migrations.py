"""Tests for Alembic revision 0001.

Structural checks (revision identifiers, statement ordering inside the
migration file, that offline SQL rendering succeeds) run unconditionally and
need no database. Everything that asserts on live database state is gated by
`requires_db`, so the default `uv run pytest` stays fully offline per the
project's global constraint and only skips (never fails) without
CULPRIT_TEST_DSN set.
"""

import os
import subprocess
import sys
from pathlib import Path

import psycopg
import pytest

from tests.conftest import requires_db

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MIGRATION_FILE = _REPO_ROOT / "migrations" / "versions" / "0001_initial_schema.py"


def _connect():
    """Bare psycopg connection for assertions. Deliberately does not import
    culprit.db: that seam is owned by a concurrent workstream and this test
    file must stand on its own regardless of that module's state."""
    return psycopg.connect(os.environ["CULPRIT_TEST_DSN"], autocommit=True)


def _run_alembic(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def _run_alembic_offline(*args: str) -> subprocess.CompletedProcess:
    """Like _run_alembic, but guarantees a DSN is present even when
    CULPRIT_TEST_DSN is unset, so `--sql` rendering (which never opens a
    connection) can be exercised in the fully offline default test run."""
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


def test_revision_0001_declares_no_down_revision_as_the_root_migration():
    """0001 must be the base of the migration chain: no earlier revision
    exists yet, so down_revision has to be None rather than a guessed id."""
    text = _MIGRATION_FILE.read_text()

    assert 'revision: str = "0001"' in text
    assert "down_revision: str | None = None" in text


def test_revision_0001_creates_the_extension_before_any_table():
    """Migration ordering matters: CREATE EXTENSION IF NOT EXISTS vector must
    precede every table, or a table using the vector type would fail to
    create on a fresh database."""
    text = _MIGRATION_FILE.read_text()

    extension_pos = text.index("CREATE EXTENSION IF NOT EXISTS vector")
    first_table_pos = text.index("CREATE TABLE")

    assert extension_pos < first_table_pos


def test_revision_0001_creates_every_table_before_any_index():
    """The second ordering rule: no CREATE INDEX may appear before the last
    CREATE TABLE, so index builds never race table creation within the same
    revision."""
    text = _MIGRATION_FILE.read_text()

    table_positions = [
        text.index(f'CREATE TABLE {name} (')
        for name in (
            "traces", "spans", "steps", "diagnoses", "signals",
            "divergences", "adjudications", "clusters", "benchmark_cases",
        )
    ]
    index_positions = [
        i for i in range(len(text)) if text.startswith("CREATE INDEX", i)
    ]

    assert index_positions, "expected at least one CREATE INDEX statement"
    assert max(table_positions) < min(index_positions)


def test_revision_0001_never_creates_an_hnsw_index():
    """HNSW indexes belong in revision 0002 so a large initial backfill is
    not slowed by index maintenance on every insert. This is the single
    easiest thing to accidentally regress, since it is tempting to add the
    vector index next to the vector column. Checked against the executable
    statements only (docstring prose mentions "HNSW" deliberately, to
    explain the deferral, so a whole-file substring check would false
    positive on its own documentation)."""
    text = _MIGRATION_FILE.read_text()
    executable_lines = [
        line for line in text.splitlines() if "op.execute" in line or "CREATE INDEX" in line
    ]

    assert not any("hnsw" in line.lower() for line in executable_lines)


def test_revision_0001_upgrade_sql_never_creates_an_hnsw_index():
    """Belt and suspenders: check the actual rendered SQL, not just the
    source file, in case a future refactor moves DDL into a helper that the
    source-text check above would miss. Targets "0001" explicitly, not
    "head": revision 0002 (tests/test_migrations_0002.py) legitimately adds
    HNSW indexes, so rendering all the way to head would false-positive here
    once 0002 exists."""
    result = _run_alembic_offline("upgrade", "0001", "--sql")

    assert result.returncode == 0, result.stderr
    assert "hnsw" not in result.stdout.lower()


def test_revision_0001_declares_the_spans_composite_primary_key():
    """OTel span ids are only unique within a trace. A single-column PK on
    span_id would collide across traces the moment two traces reuse an id,
    which real exporters do."""
    text = _MIGRATION_FILE.read_text()

    spans_block_start = text.index("CREATE TABLE spans")
    spans_block_end = text.index("CREATE TABLE steps")
    spans_block = text[spans_block_start:spans_block_end]

    assert "PRIMARY KEY (trace_id, span_id)" in spans_block


def test_revision_0001_never_puts_a_unique_constraint_on_diagnoses_trace_id():
    """diagnoses.trace_id is deliberately not unique: re-running analysis
    after improving a detector must produce a new diagnosis alongside the
    old, not overwrite it. A UNIQUE constraint here would silently break
    that contract."""
    text = _MIGRATION_FILE.read_text()

    diagnoses_start = text.index("CREATE TABLE diagnoses")
    diagnoses_end = text.index("CREATE TABLE signals")
    diagnoses_block = text[diagnoses_start:diagnoses_end]

    assert "UNIQUE" not in diagnoses_block


def test_revision_0001_declares_attributes_pruned_as_a_typed_boolean():
    """The retention decision belongs in this revision, not a later one:
    spans.attributes_pruned records whether the retention job already
    emptied this row's attributes."""
    text = _MIGRATION_FILE.read_text()

    assert "attributes_pruned BOOLEAN NOT NULL DEFAULT false" in text


def test_alembic_upgrade_head_renders_valid_sql_offline():
    """`alembic upgrade head --sql` renders SQL without connecting to a
    database (a placeholder DSN is enough for URL parsing). This is the
    closest thing to a real dry run available when no Postgres is reachable,
    and it runs unconditionally, unlike the requires_db tests below."""
    result = _run_alembic_offline("upgrade", "head", "--sql")

    assert result.returncode == 0, result.stderr
    assert "CREATE EXTENSION IF NOT EXISTS vector" in result.stdout
    assert "CREATE TABLE traces" in result.stdout
    assert "CREATE TABLE spans" in result.stdout
    assert "vector(384)" in result.stdout


def test_alembic_downgrade_to_base_renders_valid_sql_offline():
    """Same reasoning as the upgrade test: the downgrade path is exercised
    for SQL validity even without a database to run it against."""
    result = _run_alembic_offline("downgrade", "0001:base", "--sql")

    assert result.returncode == 0, result.stderr
    assert "DROP TABLE IF EXISTS traces" in result.stdout
    assert "DROP TABLE IF EXISTS benchmark_cases" in result.stdout


@requires_db
def test_migration_creates_pgvector_columns_at_384_dimensions():
    """384 is bge-small-en-v1.5's dimension. Both vector columns must match
    it and each other, because one loaded model serves both."""
    _run_alembic("upgrade", "head")

    with _connect() as conn:
        rows = conn.execute(
            "SELECT c.relname, a.atttypmod "
            "FROM pg_attribute a "
            "JOIN pg_class c ON c.oid = a.attrelid "
            "WHERE a.attname IN ('task_embedding', 'card_embedding') "
            "AND NOT a.attisdropped"
        ).fetchall()

    assert len(rows) == 2
    assert {r[1] for r in rows} == {384}


@requires_db
def test_migration_creates_every_table():
    """Coarse completeness check: every table named in the plan's Postgres
    schema section exists after upgrading to head."""
    _run_alembic("upgrade", "head")

    expected = {
        "traces", "spans", "steps", "diagnoses", "signals", "divergences",
        "adjudications", "clusters", "benchmark_cases",
    }

    with _connect() as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        ).fetchall()

    assert expected.issubset({r[0] for r in rows})


@requires_db
def test_spans_primary_key_is_composite_on_trace_id_and_span_id():
    """Enforced at the database, not just documented in the migration file:
    a duplicate span_id within a different trace_id must be insertable."""
    _run_alembic("upgrade", "head")

    with _connect() as conn:
        rows = conn.execute(
            "SELECT a.attname FROM pg_index i "
            "JOIN pg_attribute a ON a.attrelid = i.indrelid "
            "AND a.attnum = ANY(i.indkey) "
            "WHERE i.indrelid = 'spans'::regclass AND i.indisprimary "
            "ORDER BY a.attname"
        ).fetchall()

    assert {r[0] for r in rows} == {"trace_id", "span_id"}


@requires_db
def test_diagnoses_trace_id_has_no_unique_constraint():
    """Re-running analysis must be able to insert a second diagnoses row for
    the same trace_id without a constraint violation."""
    _run_alembic("upgrade", "head")

    with _connect() as conn:
        conn.execute(
            "INSERT INTO traces (trace_id, source, outcome) "
            "VALUES ('t-dup', 'test', 'failure')"
        )
        conn.execute(
            "INSERT INTO diagnoses (diagnosis_id, trace_id) VALUES "
            "('11111111-1111-1111-1111-111111111111', 't-dup')"
        )
        conn.execute(
            "INSERT INTO diagnoses (diagnosis_id, trace_id) VALUES "
            "('22222222-2222-2222-2222-222222222222', 't-dup')"
        )
        count = conn.execute(
            "SELECT count(*) FROM diagnoses WHERE trace_id = 't-dup'"
        ).fetchone()[0]

    assert count == 2


@requires_db
def test_migration_downgrade_drops_every_table():
    """The downgrade path must be a clean inverse: nothing from revision
    0001 should survive a downgrade to base."""
    _run_alembic("upgrade", "head")
    _run_alembic("downgrade", "base")

    with _connect() as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name != 'alembic_version'"
        ).fetchall()

    assert rows == []
