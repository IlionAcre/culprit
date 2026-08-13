"""L3 targeted adjudication entrypoint: one LLM call per candidate step,
judging only the handful of steps L1 and L2 narrowed the trace down to.

Stubbed here at its real signature during Phase 0 so downstream workstreams
can be written and tested against it before WS-E exists. WS-E fills the body
and additionally owns `candidates.py`, `context_window.py`, `prompts.py`,
and `confidence.py`.
"""

from culprit.llm import CallFn
from culprit.schemas import Step, Trace
from culprit.signals import Adjudication, Candidate


def adjudicate(
    candidates: list[Candidate],
    trace: Trace,
    steps: list[Step],
    *,
    call_fn: CallFn,
    model: str,
) -> list[Adjudication]:
    """Build a bounded context packet per candidate (task goal, spine, zoom
    window, terminal failure evidence, capped at 12k tokens) and fan out one
    `call_fn` call per candidate with `ThreadPoolExecutor(max_workers=4)
    .map()`.

    Owned by WS-E. Per the per-item error isolation rule, a candidate whose
    call or parse fails must produce a sentinel abstained `Adjudication`
    rather than raising or dropping it; the other candidates still get
    judged.
    """
    raise NotImplementedError("owned by WS-E")
