"""TRAIL adapter. Reality check from Integration task I5: TRAIL does NOT ship
OTLP wire format. The public release (HuggingFace `PatronusAI/TRAIL`, mirrored
on ModelScope) is one nested JSON document per trace - `{trace_id, spans}`
where each span carries `child_spans`, ISO-8601 `timestamp`/`duration`, and a
flat `span_attributes` dict - plus one annotations file per trace with
`errors: [{category, location, evidence, description, impact}]`. The guessed
schema this adapter was first written against (`resourceSpans` OTLP envelopes,
`errors[].span_id`) matched nothing in the real data.

What survived the reality check: the spans' `span_attributes` are genuine
OpenInference flattened attributes (`openinference.span.kind`, indexed
`llm.input_messages.N.message.*`, `tool.name`, `input.value`/`output.value`),
so after flattening the tree into the raw-dict shape `otlp.decode` returns,
each span still goes through `normalize.normalize_span` (the openinference
vocab module claims it) and `linearize.linearize` - the same normalization and
linearization code production traces use. Only `otlp.decode` itself is
bypassed, because there is no OTLP envelope to decode; that is a smaller
change than the docstring above the original adapter implied, and the reason a
TRAIL score still means something.

**`TRAIL_CATEGORY_MAP` is an explicit dict constant, keyed by normalized
category string** (strip, lowercase, hyphens/underscores collapsed to spaces).
TRAIL's released label set is title-case human phrases with spelling and
casing variants in the wild ("Context Handling Failures" vs "Context Handling
Failure", one literal "Instruction non complience" typo, one leading-space
" Incorrect Problem Identification"); normalization plus explicit variant keys
keeps the dict reviewable while absorbing that mess. Every guessed key from
the pre-integration version was wrong - the real taxonomy shares not a single
string with the guesses. A category this dict has not seen maps to
`FailureClass.UNKNOWN` with a logged warning rather than raising.

**Judgment calls in the map, recorded so they can be argued with.** TRAIL's
categories are coarser and multi-sense compared to culprit's 21 classes;
several mappings are plurality calls, documented per key below. Environmental
categories (auth failures, timeouts, missing files, broken tools) describe the
world, not the agent, and have no honest culprit analogue - they map to
UNKNOWN explicitly rather than being forced into an agent-side class.

**One TRAIL trace with multiple annotated errors becomes multiple
`BenchmarkCase` rows** sharing one `trace`/`spans`/`steps` object graph, one
per annotation, sorted so the earliest becomes `is_primary=True` (120 of the
131 released traces carry more than one error). This is what lets
`bench_score.py` implement "credit a match against any annotated error, but
separately report whether the earliest was found" without a second,
benchmark-specific data structure.

Input file format: one JSON list of records, each
`{trace_id, split, spans: [nested span trees], errors: [...]}` - the merge of
the dataset's raw trace files and per-trace annotation files, produced by
`scripts/prepare_trail.py` (the dataset itself ships them as separate
directory trees, and one released annotation file has a literal trailing-comma
syntax error that the prepare script tolerates).
"""

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from culprit.benchmarks.base import BenchmarkCase
from culprit.linearize import linearize, orphan_span_ids
from culprit.logging_config import LOGGER_NAME
from culprit.normalize import normalize_span
from culprit.schemas import AgentPayload, Outcome, Span, Trace
from culprit.taxonomy import FailureClass

logger = logging.getLogger(LOGGER_NAME)

# TRAIL's released error taxonomy -> culprit's FailureClass. Keys are
# normalized (see `_normalize_category`). Deliberately many-to-one: the point
# is a defensible mapping, not a bijection. Revisable; this dict encodes the
# current best judgment after reading sampled descriptions of every category.
TRAIL_CATEGORY_MAP: dict[str, FailureClass] = {
    # Plurality sense is tool/code-call arguments with invalid structure
    # (88 of 177 sampled descriptions); a substantial minority is final-output
    # format violations, which would favor OUTPUT_SCHEMA_VIOLATION. Mixed bag.
    "formatting errors": FailureClass.MALFORMED_TOOL_INPUT,
    "formatting error": FailureClass.MALFORMED_TOOL_INPUT,
    # Violating an explicit task/system-prompt instruction.
    "instruction non compliance": FailureClass.CONSTRAINT_VIOLATION,
    "instruction non complience": FailureClass.CONSTRAINT_VIOLATION,  # typo in the released data
    # Mid-run abandonment of the agent's own plan, skipping planned steps.
    "goal deviation": FailureClass.PLAN_OMISSION,
    # Unsupported claims and fabrications in free-text reasoning.
    "language only": FailureClass.INFORMATION_FABRICATION,
    # Sampled descriptions are dominated by repeating the same failing call.
    "resource abuse": FailureClass.INFINITE_LOOP_OR_OSCILLATION,
    # Sampled descriptions are dominated by fabricated tool interactions.
    "tool related": FailureClass.HALLUCINATED_TOOL_OR_PARAMETER,
    "tool selection errors": FailureClass.WRONG_TOOL_SELECTED,
    "tool selection": FailureClass.WRONG_TOOL_SELECTED,
    "context handling failures": FailureClass.CONTEXT_LOSS,
    "context handling failure": FailureClass.CONTEXT_LOSS,
    # Planning/delegation coordination failures; the delegation-flavored
    # minority would fit HANDOFF_INFORMATION_LOSS better. Plurality call.
    "task orchestration": FailureClass.PLAN_OMISSION,
    "task orchestration error": FailureClass.PLAN_OMISSION,
    "task orchestration errors": FailureClass.PLAN_OMISSION,
    "poor information retrieval": FailureClass.RETRIEVAL_MISS,
    "incorrect problem identification": FailureClass.TASK_MISINTERPRETATION,
    # Misread a tool result (e.g. as success when it was not); closest
    # agent-side class, though culprit reserves this for explicit errors.
    "tool output misinterpretation": FailureClass.TOOL_FAILURE_UNHANDLED,
    # Used stale or wrong slices of its own history.
    "incorrect memory usage": FailureClass.CONTEXT_LOSS,
    "resource exhaustion": FailureClass.STEP_BUDGET_EXHAUSTED,
    # Environmental: no honest agent-side culprit class.
    "environment setup errors": FailureClass.UNKNOWN,
    "resource not found": FailureClass.UNKNOWN,
    "authentication errors": FailureClass.UNKNOWN,
    "service errors": FailureClass.UNKNOWN,
    "timeout issues": FailureClass.UNKNOWN,
    "tool definition issues": FailureClass.UNKNOWN,
}

_CATEGORY_SEP_RE = re.compile(r"[\s\-_]+")
# e.g. "PT1M48.75533S", "PT0.000161S", occasionally with hours.
_ISO_DURATION_RE = re.compile(
    r"^PT(?:(?P<h>\d+(?:\.\d+)?)H)?(?:(?P<m>\d+(?:\.\d+)?)M)?(?:(?P<s>\d+(?:\.\d+)?)S)?$"
)
_STATUS_CODE_MAP = {"ok": "STATUS_CODE_OK", "error": "STATUS_CODE_ERROR"}


def _normalize_category(category: str) -> str:
    return _CATEGORY_SEP_RE.sub(" ", category.strip().lower())


def _map_category(category: str) -> FailureClass:
    mapped = TRAIL_CATEGORY_MAP.get(_normalize_category(category))
    if mapped is None:
        logger.warning(
            "unmapped TRAIL error category, defaulting to unknown",
            extra={"event": "trail_category_unmapped", "category": category},
        )
        return FailureClass.UNKNOWN
    return mapped


def _parse_timestamp_ns(ts: str) -> int:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1e9)


def _parse_duration_ns(duration: str | None) -> int:
    if not duration:
        return 0
    m = _ISO_DURATION_RE.match(duration)
    if m is None:
        raise ValueError(f"unparseable ISO-8601 duration {duration!r}")
    seconds = sum(
        float(m.group(unit) or 0) * factor
        for unit, factor in (("h", 3600), ("m", 60), ("s", 1))
    )
    return int(seconds * 1e9)


def _flatten(node: dict, out: list[dict]) -> None:
    """One nested TRAIL span tree -> raw dicts in the exact shape
    `otlp.decode` returns (DFS pre-order; `linearize` re-derives the tree
    from `parent_span_id` anyway, so order here is only a readability aid)."""
    start_ns = _parse_timestamp_ns(node["timestamp"])
    out.append({
        "span_id": node["span_id"],
        "parent_span_id": node.get("parent_span_id"),
        "name": node.get("span_name", ""),
        "otel_kind": "SPAN_KIND_INTERNAL",
        "start_ns": start_ns,
        "end_ns": start_ns + _parse_duration_ns(node.get("duration")),
        "status_code": _STATUS_CODE_MAP.get(
            (node.get("status_code") or "").lower(), "STATUS_CODE_UNSET"
        ),
        "status_message": node.get("status_message") or None,
        "attributes": dict(node.get("span_attributes") or {}),
    })
    for child in node.get("child_spans") or []:
        _flatten(child, out)


def _trace_from(trace_id: str, spans: list[Span], steps, external_id: str) -> Trace:
    agent_name = next((s.payload.agent_name for s in spans if isinstance(s.payload, AgentPayload)), None)
    root = next((s for s in spans if s.parent_span_id is None), spans[0] if spans else None)
    return Trace(
        trace_id=trace_id, source="benchmark:trail", outcome=Outcome.FAILURE,
        agent_key=f"benchmark:trail:{agent_name}" if agent_name else None,
        task_key=f"benchmark:trail:{external_id}", task_goal=None, framework="benchmark:trail",
        root_span_id=root.span_id if root else None,
        span_count=len(spans), step_count=len(steps),
        metadata={"orphan_span_ids": orphan_span_ids(spans), "external_case_id": external_id},
    )


def _cases_from_record(record: dict) -> list[BenchmarkCase]:
    external_id = record["trace_id"]
    friendly_trace_id = f"benchmark:trail:{external_id}"

    raw_spans: list[dict] = []
    for root in record["spans"]:
        _flatten(root, raw_spans)
    # One released trace (SWE Bench 72822db6...) emits the same span twice
    # (identical span_id, name, timestamp, parent), which violates the
    # (trace_id, span_id) primary key at write time; keep the first
    # occurrence and drop exact-id re-emissions.
    seen_ids: set[str] = set()
    deduped: list[dict] = []
    for raw in raw_spans:
        if raw["span_id"] in seen_ids:
            logger.warning(
                "duplicate span id in TRAIL trace, keeping first occurrence",
                extra={"event": "trail_duplicate_span", "trace_id": external_id, "span_id": raw["span_id"]},
            )
            continue
        seen_ids.add(raw["span_id"])
        deduped.append(raw)
    spans = [normalize_span(raw, friendly_trace_id) for raw in deduped]
    steps = linearize(spans)
    trace = _trace_from(friendly_trace_id, spans, steps, external_id)

    # Maps a span id to the step it either *is* (a semantic step) or was
    # collapsed into (a non-semantic span folded per linearize.py rule 2), so
    # an annotated error landing on a non-semantic span still resolves to a
    # real step index instead of being silently dropped.
    step_index_by_span_id: dict[str, int] = {}
    for step in steps:
        step_index_by_span_id[step.span_id] = step.step_index
        for collapsed_id in step.collapsed_span_ids:
            step_index_by_span_id.setdefault(collapsed_id, step.step_index)

    annotations: list[tuple[int, str, FailureClass]] = []
    for error in record.get("errors", []):
        span_id = error.get("location")
        step_index = step_index_by_span_id.get(span_id)
        if step_index is None:
            logger.warning(
                "TRAIL error annotation references a span not present in the trace, skipped",
                extra={"event": "trail_annotation_orphaned", "trace_id": external_id, "span_id": span_id},
            )
            continue
        annotations.append((step_index, span_id, _map_category(error.get("category", ""))))

    annotations.sort(key=lambda a: a[0])
    return [
        BenchmarkCase(
            case_id=f"trail:{external_id}:{i}", trace=trace, spans=spans, steps=steps,
            ground_truth_step_index=step_index, ground_truth_span_id=span_id,
            ground_truth_failure_class=failure_class.value, is_primary=(i == 0),
        )
        for i, (step_index, span_id, failure_class) in enumerate(annotations)
    ]


def load_cases(path: Path) -> list[BenchmarkCase]:
    """One malformed record (bad nested span JSON, an annotation pointing at
    a span that does not exist) is logged and skipped rather than aborting
    the whole file - the same per-item isolation discipline as L1 detectors."""
    records = json.loads(path.read_text(encoding="utf-8"))
    cases: list[BenchmarkCase] = []
    for record in records:
        try:
            cases.extend(_cases_from_record(record))
        except Exception as e:  # noqa: BLE001 - per-record isolation, see docstring
            logger.warning(
                "TRAIL record failed to convert to a benchmark case, skipped",
                extra={"event": "trail_record_failed", "trace_id": record.get("trace_id"), "error": str(e)},
            )
    return cases
