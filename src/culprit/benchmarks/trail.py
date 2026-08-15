"""TRAIL adapter. TRAIL ships OpenTelemetry-format traces annotated with
per-span `(span_id, category)` error records, so this feeds the fixture's
embedded OTLP JSON straight through `otlp.decode` -> `normalize.normalize_span`
-> `linearize.linearize`, the exact same three calls the ingest endpoint
makes (`api.py`), rather than re-deriving spans/steps by hand. That is the
one cross-workstream import CLAUDE.md calls out, and the whole reason a TRAIL
score means anything: it exercised the real normalization and linearization
code, not a parallel path built only to please this adapter.

**`TRAIL_CATEGORY_MAP` is an explicit dict constant, not inferred at runtime.**
It encodes a genuine, revisable judgment call about how TRAIL's own error
taxonomy corresponds to `taxonomy.FailureClass`; a dict literal is reviewable
and diffable in a way an inference rule would not be. The category strings
below are TRAIL's published error categories as of this benchmark's
integration; a category this dict has not seen yet (a taxonomy revision on
TRAIL's side, or simply a typo in a fixture) maps to `FailureClass.UNKNOWN`
with a logged warning rather than raising - see `_map_category`.

**One TRAIL trace with multiple annotated errors becomes multiple
`BenchmarkCase` rows** sharing one `trace`/`spans`/`steps` object graph, one
per annotation, sorted so the earliest becomes `is_primary=True`. This is
what lets `bench_score.py` implement "credit a match against any annotated
error, but separately report whether the earliest was found" without a
second, benchmark-specific data structure - conflating the two would flatter
the system (CLAUDE.md, "Benchmark harness").
"""

import json
import logging
from pathlib import Path

from culprit.benchmarks.base import BenchmarkCase
from culprit.linearize import linearize, orphan_span_ids
from culprit.logging_config import LOGGER_NAME
from culprit.normalize import normalize_span
from culprit.otlp import decode
from culprit.schemas import AgentPayload, Outcome, Span, Trace
from culprit.taxonomy import FailureClass

logger = logging.getLogger(LOGGER_NAME)

# TRAIL's error taxonomy -> culprit's FailureClass. Deliberately many-to-one
# in places (e.g. TRAIL has no direct analogue of culprit's finer-grained
# verification split) since the point is a defensible mapping, not a
# bijection. Revisit at Integration against TRAIL's actual released label
# set; this is the reviewable starting point, not a frozen contract.
TRAIL_CATEGORY_MAP: dict[str, FailureClass] = {
    "wrong_tool_call": FailureClass.WRONG_TOOL_SELECTED,
    "incorrect_tool_parameters": FailureClass.MALFORMED_TOOL_INPUT,
    "tool_call_failure": FailureClass.TOOL_FAILURE_UNHANDLED,
    "hallucinated_tool_call": FailureClass.HALLUCINATED_TOOL_OR_PARAMETER,
    "poor_information_retrieval": FailureClass.RETRIEVAL_MISS,
    "context_handling_failure": FailureClass.CONTEXT_LOSS,
    "information_fabrication": FailureClass.INFORMATION_FABRICATION,
    "task_orchestration_error": FailureClass.PLAN_OMISSION,
    "goal_deviation": FailureClass.TASK_MISINTERPRETATION,
    "resource_abuse": FailureClass.STEP_BUDGET_EXHAUSTED,
    "looping_behavior": FailureClass.INFINITE_LOOP_OR_OSCILLATION,
    "premature_termination": FailureClass.PREMATURE_TERMINATION,
    "missing_verification": FailureClass.MISSING_VERIFICATION,
    "incorrect_verification": FailureClass.INCORRECT_VERIFICATION,
    "output_format_error": FailureClass.OUTPUT_SCHEMA_VIOLATION,
    "multi_agent_communication_error": FailureClass.HANDOFF_INFORMATION_LOSS,
}


def _map_category(category: str) -> FailureClass:
    mapped = TRAIL_CATEGORY_MAP.get(category)
    if mapped is None:
        logger.warning(
            "unmapped TRAIL error category, defaulting to unknown",
            extra={"event": "trail_category_unmapped", "category": category},
        )
        return FailureClass.UNKNOWN
    return mapped


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

    otlp_payload = json.dumps({"resourceSpans": record["resourceSpans"]}).encode()
    raw_spans = decode(otlp_payload, "application/json")
    spans = [normalize_span(raw, friendly_trace_id) for raw in raw_spans]
    steps = linearize(spans)
    trace = _trace_from(friendly_trace_id, spans, steps, external_id)

    # Maps a span id to the step it either *is* (a semantic step) or was
    # collapsed into (a framework span folded per linearize.py rule 2), so an
    # annotated error landing on a non-semantic span still resolves to a
    # real step index instead of being silently dropped.
    step_index_by_span_id: dict[str, int] = {}
    for step in steps:
        step_index_by_span_id[step.span_id] = step.step_index
        for collapsed_id in step.collapsed_span_ids:
            step_index_by_span_id.setdefault(collapsed_id, step.step_index)

    annotations: list[tuple[int, str, FailureClass]] = []
    for error in record.get("errors", []):
        span_id = error.get("span_id")
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
    """One malformed record (bad OTLP JSON, an annotation pointing at a span
    that does not exist) is logged and skipped rather than aborting the whole
    fixture file - the same per-item isolation discipline as L1 detectors."""
    records = json.loads(path.read_text())
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
