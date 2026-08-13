"""Alembic environment for culprit's hand-written migrations.

There are no SQLAlchemy ORM models in this project, and none should be added
here: the schema is mostly JSONB, bulk inserts, and pgvector operators, none
of which an ORM improves. Alembic is used purely as a migration runner over
plain DDL strings written directly in each revision (see
migrations/versions/0001_initial_schema.py), never `--autogenerate`, so
`target_metadata` below is deliberately None.

The connection URL is read from the environment rather than from
alembic.ini, because a checked-in default DSN either points at a real
database by accident or silently misleads a contributor into thinking one
exists. CULPRIT_TEST_DSN takes priority so the exact invocation this
project's tests and docs use, `CULPRIT_TEST_DSN=... alembic upgrade head`,
works with no other configuration. CULPRIT_DATABASE_URL is the
production/dev fallback for the same env var config.py's `database_url`
field is expected to read.
"""

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def _database_url() -> str:
    url = os.environ.get("CULPRIT_TEST_DSN") or os.environ.get("CULPRIT_DATABASE_URL")
    if not url:
        raise RuntimeError(
            "Set CULPRIT_TEST_DSN or CULPRIT_DATABASE_URL before running "
            "migrations. Neither was found in the environment."
        )
    # psycopg 3 is this project's driver; a plain postgresql:// DSN (the form
    # every other tool and doc example uses) needs the +psycopg suffix for
    # SQLAlchemy to pick the right dialect.
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    return url


def run_migrations_offline() -> None:
    """Render SQL without connecting. Used by `alembic upgrade head --sql`,
    which is how this migration's DDL is verified in environments with no
    Postgres available."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(
        configuration, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
