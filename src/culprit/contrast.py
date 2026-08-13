"""L2 contrastive trajectory diff entrypoint: aligns one trace's steps
against successful reference runs and scores each step by how much of the
space of trajectories that would have succeeded it destroyed.

Stubbed here at its real signature during Phase 0 so downstream workstreams
can be written and tested against it before WS-D exists. WS-D fills the body
and additionally owns `fingerprint.py`, `signature.py`, `align.py`, and
`reference.py`.
"""

from culprit.db import ConnFn
from culprit.schemas import Step, Trace
from culprit.signals import ContrastResult, NeighborFn


def contrast(
    trace: Trace, steps: list[Step], *, neighbor_fn: NeighborFn, conn_fn: ConnFn
) -> ContrastResult:
    """Fingerprint the trace, resolve a reference pool via `neighbor_fn`,
    build or load the consensus profile and signature Markov model, align
    with Gotoh (affine-gap Needleman-Wunsch), and score each step's
    divergence via the forward/reverse residual-alignability passes.

    Owned by WS-D. `neighbor_fn` is injected rather than importing
    `store_traces` directly, so WS-D's own tests run against a dict-backed
    fake with no Postgres dependency; the concrete `NeighborFn` implementation
    lives in `store_traces.nearest_successful`, owned by WS-B. Abstains with
    `abstain_reason="insufficient_references"` when fewer than 3 references
    resolve, a contrastive method with two references produces confident
    nonsense.
    """
    raise NotImplementedError("owned by WS-D")
