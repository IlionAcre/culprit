"""The Signal / Diagnosis contract: every type an analysis layer produces or
consumes, defined once so no downstream workstream invents its own shape.

Pydantic versus `@dataclass` follows the project-wide rule (config docstring
in `culprit.toml`, see also `schemas.py`): `@dataclass` for internal computed
results that never leave the process boundary, Pydantic for anything that
crosses an I/O boundary. Only `Diagnosis` crosses two (persisted as JSONB,
served over HTTP), so it alone is `BaseModel`; `Signal`, `DivergenceCandidate`,
`ContrastResult`, `Candidate`, and `Adjudication` are internal pipeline state
and stay `@dataclass`. Pydantic v2 accepts stdlib dataclasses natively as
nested field types, so `Diagnosis.signals: list[Signal]` needs no extra
adapter.

Two properties matter more than the field lists:

- **Per-item error isolation.** A detector, aligner, or adjudicator that
  raises must never take down the other 19 detectors, or the other 4
  candidates. Instead it produces exactly one sentinel result with `error`
  set and every other field a meaningless placeholder, and callers exclude
  it from ranking rather than propagating the exception. Applied at
  `Signal`, `DivergenceCandidate`, and `Adjudication` granularity here, and
  at trace level via `Diagnosis.degraded_layers`.

- **`Signal.category` is a hint, not a verdict.** L1 knows a tool call
  returned an empty result; it does not know whether that is what actually
  caused the failure five steps later. Only `Adjudication.failure_class`,
  set by L3 after looking at bounded context, is a verdict. Conflating the
  two turns deterministic detectors into false-positive machines, which is
  exactly the failure mode a narrow-then-adjudicate architecture exists to
  avoid.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel


@dataclass(frozen=True)
class Evidence:
    span_id: str
    step_index: int
    field: str  # dotted path, e.g. "payload.result_text"
    excerpt: str  # <= 400 chars, head-and-tail truncated
    numeric: float | None = None


@dataclass
class Signal:  # L1 emits
    detector: str
    step_index: int  # -1 for whole-trace or sentinel
    span_id: str | None
    severity: float  # 0.0 to 1.0
    category: str  # FailureClass hint, not a verdict
    message: str
    evidence: list[Evidence]
    error: str | None = None


@dataclass
class DivergenceCandidate:  # L2 emits
    step_index: int
    span_id: str
    divergence_score: float  # the ranked quantity, D(i)
    cliff_delta: float  # loss of achievable future at this step, C(i)
    surprisal: float  # nats, Markov model
    profile_surprisal: float  # nats, consensus profile
    reference_count: int
    observed_signature: str
    expected_signatures: list[tuple[str, float]]  # top-3 with probability
    nearest_reference_trace_ids: list[str]
    alignment_op: str  # "mismatch" | "insertion" | "deletion"
    error: str | None = None


@dataclass
class ContrastResult:
    candidates: list[DivergenceCandidate]
    reference_count: int
    abstained: bool
    abstain_reason: str | None


@dataclass
class Candidate:  # L3 consumes
    step_index: int
    span_id: str
    rank: int
    prior: float
    source: str  # "l1" | "l2" | "both" | "fallback" | later "l4"
    signals: list[Signal]
    divergence: DivergenceCandidate | None


@dataclass
class Adjudication:  # L3 emits, one per candidate
    step_index: int
    span_id: str
    is_root_cause: bool
    failure_class: str
    confidence: float  # model self-reported
    calibrated_confidence: float
    rationale: str
    counterfactual: str  # what a successful run would have done here
    cited_step_indices: list[int]
    abstained: bool
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    error: str | None = None
    source: str = "unknown"  # carried from Candidate.source (l1/l2/both/fallback/filler)


class Diagnosis(BaseModel):
    """Pydantic, not a dataclass, because it crosses two I/O boundaries:
    persisted as a `diagnoses` row and served over HTTP. Both `confidence`
    and `calibrated_confidence` are carried (not only the calibrated value)
    so a reader can see how far calibration moved the number, which is the
    whole point of measuring calibration in the first place."""

    diagnosis_id: str
    trace_id: str
    created_at: datetime
    root_cause_step_index: int | None
    root_cause_span_id: str | None
    failure_class: str | None
    confidence: float  # the winning Adjudication's raw confidence
    calibrated_confidence: float  # the winning Adjudication's calibrated value
    abstained: bool
    abstain_reason: str | None
    rationale: str
    counterfactual: str
    candidates_considered: int
    signals: list[Signal]
    divergences: list[DivergenceCandidate]
    adjudications: list[Adjudication]
    layer_versions: dict[str, str]
    # If L2 blew up, this still returns an L1+L3 diagnosis rather than
    # silently presenting a weaker answer as a full one; the response says
    # so explicitly instead of making the caller infer it from empty lists.
    degraded_layers: list[str]
    error: str | None = None


# signals.py owns this alias (rather than db.py) so WS-D can depend on the
# shape of "look up nearest successful reference traces" without importing
# store_traces and its Postgres/pgvector machinery directly.
# (query_embedding, k) -> ordered trace_ids of nearest successful runs
NeighborFn = Callable[[list[float], int], list[str]]
