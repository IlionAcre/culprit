"""L3 targeted adjudication: one LLM call per candidate step, judging only
the handful of steps L1 and L2 narrowed the trace down to.

Structured-output parsing mirrors Litmus's `llm_judge.py` (read-only
reference for this workstream) exactly: a de-fence regex before `json.loads`,
a private Pydantic model for the verdict shape, and a narrow `except` set
around parsing specifically. Litmus's own test suite
(`test_scoring_llm_judge.py`) found real Gemini responses wrap JSON in a
fence even when told not to, sometimes with prose before it or no internal
newline in the fence at all - the same regex handles all of those here.

Per-item error isolation (CLAUDE.md) is the other half of this module: a
candidate whose call or parse fails must produce a sentinel abstained
`Adjudication` with `error` set, never an exception that costs the other
candidates their independent judgment. `_process_candidate` mirrors
Litmus's `cli.py::_process_case` for exactly this reason - never raises, so
it is safe to call from inside a `ThreadPoolExecutor` worker with no extra
exception plumbing.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field, ValidationError

from culprit.confidence import agrees_with_l1, calibrate_confidence, cite_check
from culprit.context_window import ContextPacket, build_context_packet
from culprit.llm import CallFn
from culprit.prompts import build_prompt
from culprit.schemas import Span, Step, Trace
from culprit.signals import Adjudication, Candidate
from culprit.taxonomy import FailureClass

_DEFAULT_MAX_WORKERS = 4
_DEFAULT_TOKEN_BUDGET = 12_000

_CODE_FENCE_RE = re.compile(r"```(?:\w*\n)?(.*?)```", re.DOTALL)


class AdjudicationParseError(Exception):
    """Raised when the model's response cannot be parsed into a verdict.
    Always caught by `_process_candidate`'s broad `except`, which converts
    it into a sentinel `Adjudication` - never allowed to propagate and cost
    the other candidates their own independent judgment."""


class _AdjudicationVerdict(BaseModel):
    is_root_cause: bool
    failure_class: FailureClass
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    counterfactual: str
    cited_step_indices: list[int] = Field(default_factory=list)


def _strip_markdown_code_fence(text: str) -> str:
    """Searches for a fenced block anywhere in the text rather than assuming
    the fence markers are the first/last lines - copied deliberately from
    Litmus's proven-in-production version, which a real Gemini call caught
    failing on a single-line fence with no internal newline and on prose
    before the fence (see this module's docstring)."""
    match = _CODE_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _parse_verdict(raw_output: str) -> _AdjudicationVerdict:
    try:
        data = json.loads(_strip_markdown_code_fence(raw_output))
        return _AdjudicationVerdict(**data)
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        raise AdjudicationParseError(
            f"adjudication response could not be parsed as a verdict: {raw_output!r} ({e})"
        ) from e


def _sentinel_adjudication(candidate: Candidate, model: str, error: str) -> Adjudication:
    """The per-item error isolation contract for L3: a candidate whose call
    or parse failed still produces a real `Adjudication`, just one that
    abstains and names the failure, so `select_diagnosis` (confidence.py)
    can treat it as ineligible without special-casing exceptions."""
    return Adjudication(
        step_index=candidate.step_index,
        span_id=candidate.span_id,
        is_root_cause=False,
        failure_class=FailureClass.UNKNOWN.value,
        confidence=0.0,
        calibrated_confidence=0.0,
        rationale="",
        counterfactual="",
        cited_step_indices=[],
        abstained=True,
        model=model,
        prompt_tokens=None,
        completion_tokens=None,
        cost_usd=None,
        error=error,
    )


def _adjudication_from_verdict(
    candidate: Candidate, packet: ContextPacket, verdict: _AdjudicationVerdict,
    model: str, cost_usd: float | None,
    prompt_tokens: int | None, completion_tokens: int | None,
) -> Adjudication:
    """Builds a non-abstained Adjudication from the model's parsed verdict.
    Token counts are threaded through from CallFn so the narrow-then-adjudicate
    cost claim can be proven from the adjudications table rather than asserted
    (INTEGRATION_ITEMS.md backlog item 6)."""
    density = cite_check(verdict.cited_step_indices, packet.visible_step_indices)
    agreement = agrees_with_l1([s.category for s in candidate.signals], verdict.failure_class.value)
    calibrated = calibrate_confidence(verdict.confidence, candidate.prior, agreement, density)
    return Adjudication(
        step_index=candidate.step_index,
        span_id=candidate.span_id,
        is_root_cause=verdict.is_root_cause,
        failure_class=verdict.failure_class.value,
        confidence=verdict.confidence,
        calibrated_confidence=calibrated,
        rationale=verdict.rationale,
        counterfactual=verdict.counterfactual,
        cited_step_indices=list(verdict.cited_step_indices),
        abstained=False,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        error=None,
    )


def _process_candidate(
    candidate: Candidate, trace: Trace, steps: list[Step], *, call_fn: CallFn,
    model: str, token_budget: int, spans_by_id: "dict[str, Span] | None" = None,
) -> Adjudication:
    """Never raises - both the call and the parse happen inside one broad
    `try`, matching Litmus's `_process_case`. A candidate is independent of
    every other candidate by construction, so isolating failure here is
    what makes fanning this out across a `ThreadPoolExecutor` safe with no
    extra exception plumbing at the call site."""
    try:
        packet = build_context_packet(
            candidate, trace, steps, token_budget=token_budget, spans_by_id=spans_by_id,
        )
        prompt = build_prompt(candidate, packet)
        raw_output, _latency_ms, cost_usd, prompt_tokens, completion_tokens = call_fn(model, prompt)
        verdict = _parse_verdict(raw_output)
        return _adjudication_from_verdict(
            candidate, packet, verdict, model, cost_usd, prompt_tokens, completion_tokens,
        )
    except Exception as e:  # noqa: BLE001 - deliberately broad, see docstring
        return _sentinel_adjudication(candidate, model, error=f"{type(e).__name__}: {e}")


def adjudicate(
    candidates: list[Candidate],
    trace: Trace,
    steps: list[Step],
    *,
    call_fn: CallFn,
    model: str,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    token_budget: int = _DEFAULT_TOKEN_BUDGET,
    spans_by_id: "dict[str, Span] | None" = None,
) -> list[Adjudication]:
    """Build a bounded context packet per candidate and fan out one call per
    candidate. *Rejected: one call covering all candidates* - reintroduces
    the long-context degradation TRAIL/Who&When report and destroys the
    independence that makes cross-candidate confidence comparison
    meaningful (CLAUDE.md's "L3 adjudication" section). `max_workers<=1`
    runs sequentially, identical to Litmus's own concurrency-toggle
    convention, useful for tests that want deterministic ordering without
    thread-pool nondeterminism in play.

    `spans_by_id` is optional (default `None`) so every pre-Integration
    caller and test keeps working unchanged; `pipeline.py` passes the real
    dict so the zoom window's neighbor steps get real payload text instead
    of only `Step.summary` (INTEGRATION_ITEMS.md item 2, same fix already
    applied to `contrast()`)."""
    if not candidates:
        return []

    def _run(candidate: Candidate) -> Adjudication:
        return _process_candidate(
            candidate, trace, steps, call_fn=call_fn, model=model,
            token_budget=token_budget, spans_by_id=spans_by_id,
        )

    if max_workers <= 1:
        return [_run(c) for c in candidates]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_run, candidates))
