"""RQ worker entrypoint. A separate process from the API, running the same
package, so `diagnose_trace_job`/`recluster_job` execute with identical
imports and code paths regardless of which process enqueued them. Analysis
takes minutes per trace, which is why it never runs inside a request (see
CLAUDE.md); `culprit worker` starts one of these.
"""

import logging

from rq import Worker

from culprit.logging_config import LOGGER_NAME
from culprit.queue import QUEUE_NAME, get_queue

logger = logging.getLogger(LOGGER_NAME)


def run_worker(*, burst: bool = False) -> None:
    """Start one RQ worker listening on the `culprit` queue. `burst=True`
    processes whatever is already queued and returns, used by tests and
    one-shot batch runs; `burst=False` (the default) blocks and polls
    forever, which is how `culprit worker` runs in production."""
    queue = get_queue()
    worker = Worker([queue], connection=queue.connection)
    logger.info(
        "worker starting", extra={"event": "worker_started", "queue": QUEUE_NAME}
    )
    worker.work(burst=burst)


if __name__ == "__main__":
    run_worker()
