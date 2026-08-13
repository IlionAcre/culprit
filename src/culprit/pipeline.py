"""L0-L5 orchestration entrypoint: the single function that runs the whole
analysis cascade for one trace and returns a `Diagnosis`.

Stubbed here at its real signature during Phase 0 so WS-G (service surface)
can build the FastAPI app, Typer CLI, and RQ job functions against a real
call site today, monkeypatching `diagnose` in its own tests rather than
waiting for five analysis layers to exist. The body is filled in exclusively
during Integration task I1, after WS-A through WS-F have all landed. Per
`CLAUDE.md`'s guardrails, no other workstream edits this file: a shared
orchestration point touched by two of the eight parallel workstreams at once
is exactly the merge conflict Phase 0 exists to avoid.
"""

from culprit.db import ConnFn
from culprit.embed import EmbedFn
from culprit.llm import CallFn
from culprit.schemas import Trace
from culprit.signals import Diagnosis


def diagnose(
    trace: Trace, *, conn_fn: ConnFn, call_fn: CallFn, embed_fn: EmbedFn
) -> Diagnosis:
    """Run L0 (normalize/linearize) through L3 (adjudication), persist the
    result, and return the resulting `Diagnosis`. L5 clustering runs
    separately as a batch recompute, not inline per diagnosis.

    Filled in during Integration task I1.
    """
    raise NotImplementedError("filled during Integration, task I1")
