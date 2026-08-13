"""psycopg 3 connection pool with pgvector wired in.

`register_vector` has to run once per connection (it fetches the `vector`
type's OID from Postgres itself), so it is passed as the pool's `configure`
callback rather than called by hand after every checkout: the pool invokes
it automatically for every new connection it opens, including ones created
to replace a dropped one.

`open=False` keeps `make_pool` a pure constructor with no side effect at
call time: nothing dials the database until a caller explicitly opens the
pool (or enters it as a context manager). That is what lets `db.py` be
imported, and `ConnFn` reused by other modules, with no Postgres instance
running anywhere, including throughout the offline test suite.
"""

from typing import Callable

from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg_pool import ConnectionPool

ConnFn = Callable[[], Connection]


def make_pool(dsn: str, min_size: int = 1, max_size: int = 10) -> ConnectionPool:
    return ConnectionPool(
        conninfo=dsn,
        min_size=min_size,
        max_size=max_size,
        configure=register_vector,
        open=False,
    )
