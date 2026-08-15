"""Who&When adapter. Who&When has no OTel form: ground truth is a flat list
of `(agent, message)` turns plus a `(mistake_agent, mistake_step,
mistake_reason)` annotation, so unlike TRAIL there is nothing for
`otlp.decode` to parse - this adapter synthesizes canonical `Span`/`Step`
objects directly, then puts them through the exact same `linearize.linearize`
the OTLP path uses, so a Who&When trace is still a canonical `Trace` fed
through real production code, not a benchmark-only shortcut.

**The synthetic root is `SpanKind.CHAIN`, not `AGENT`.** `linearize.py` rule
2 folds CHAIN spans into the *next* semantic step's `collapsed_span_ids`
rather than giving them their own step index. That is deliberate: if the
root were itself an AGENT span it would consume step 0, shifting every
message one step to the right and breaking the direct
`mistake_step == ground_truth_step_index` mapping the plan promises. Making
the root a non-semantic container is what makes "already linear" literally
true at the `Step` level, not just at the message level.

**Kind refinement from content markers.** Each message defaults to
`SpanKind.AGENT` (a turn of reasoning or delegation). A message whose content
looks like a function call (`tool_name(args)`) is refined to `SpanKind.TOOL`,
or `SpanKind.RETRIEVER` if the function name mentions search/retrieval - both
kinds `linearize.py` still treats as semantic, so the message-to-step 1:1
mapping survives the refinement. Known limitation, accepted rather than
worked around by editing frozen `linearize.py`: `actor` on a refined
non-AGENT step resolves through the flat parent chain to the synthetic root's
own name (`linearize._resolve_actor`'s contract), not the per-message agent
name; this does not affect any metric `bench_score.py` reports, all of which
are step/class/confidence based, never actor based.

**`ground_truth_failure_class` is always `FailureClass.UNKNOWN`.** Who&When's
ground truth is a free-text `mistake_reason`, not a labeled taxonomy, so
there is nothing to map (unlike TRAIL's explicit category dict). This is why
the plan calls Who&When "the cleaner primary benchmark for step-level
metrics" specifically, not for class accuracy.
"""

import json
import logging
import re
from pathlib import Path

from culprit.benchmarks.base import BenchmarkCase
from culprit.linearize import linearize
from culprit.logging_config import LOGGER_NAME
from culprit.schemas import (
    AgentPayload, Outcome, RetrievalPayload, Span, SpanKind, SpanStatus, ToolPayload, Trace,
)
from culprit.taxonomy import FailureClass

logger = logging.getLogger(LOGGER_NAME)

_CALL_MARKER = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)\s*$", re.DOTALL)
_RETRIEVAL_MARKERS = ("search", "retriev")
_STEP_DURATION_NS = 500_000
_STEP_SPACING_NS = 1_000_000


def _message_span(trace_id: str, root_id: str, case_id: str, i: int, agent: str, content: str) -> Span:
    span_id = f"{case_id}-{i}"
    start_ns = i * _STEP_SPACING_NS
    common = dict(
        trace_id=trace_id, span_id=span_id, parent_span_id=root_id, name=f"{agent}:{i}",
        status=SpanStatus.OK, status_message=None, start_ns=start_ns,
        end_ns=start_ns + _STEP_DURATION_NS, vocabulary="benchmark:who_and_when", attributes={},
    )

    match = _CALL_MARKER.match(content or "")
    if match is None:
        return Span(
            kind=SpanKind.AGENT,
            payload=AgentPayload(agent_name=agent, role="", input_text="", output_text=content, delegated_to=[]),
            **common,
        )

    func_name, args = match.group(1), match.group(2)
    if any(marker in func_name.lower() for marker in _RETRIEVAL_MARKERS):
        return Span(
            kind=SpanKind.RETRIEVER,
            payload=RetrievalPayload(query=args, documents=[], top_k=0),
            **common,
        )
    return Span(
        kind=SpanKind.TOOL,
        payload=ToolPayload(
            tool_name=func_name, call_id=f"{case_id}-call-{i}", arguments_json=args or "{}",
            arguments={}, result_text="", result_len=0, is_error=False, error_message=None,
        ),
        **common,
    )


def _case_from_record(record: dict) -> BenchmarkCase:
    case_id = record["case_id"]
    trace_id = f"benchmark:who_and_when:{case_id}"
    root_id = f"{case_id}-root"

    root = Span(
        trace_id=trace_id, span_id=root_id, parent_span_id=None, name="orchestrator",
        kind=SpanKind.CHAIN, status=SpanStatus.OK, status_message=None, start_ns=0,
        end_ns=len(record["messages"]) * _STEP_SPACING_NS, vocabulary="benchmark:who_and_when",
        attributes={}, payload=None,
    )
    messages = [
        _message_span(trace_id, root_id, case_id, i, m["agent"], m.get("content", ""))
        for i, m in enumerate(record["messages"])
    ]
    spans = [root] + messages
    steps = linearize(spans)

    mistake_step = int(record["mistake_step"])
    if not 0 <= mistake_step < len(steps):
        raise ValueError(f"mistake_step {mistake_step} out of range for {len(steps)} steps")
    ground_truth_span_id = steps[mistake_step].span_id

    trace = Trace(
        trace_id=trace_id, source="benchmark:who_and_when", outcome=Outcome.FAILURE,
        agent_key=None, task_key=trace_id, task_goal=record.get("task_goal"),
        framework="benchmark:who_and_when", root_span_id=root_id,
        span_count=len(spans), step_count=len(steps),
        metadata={
            "external_case_id": case_id,
            "mistake_agent": record.get("mistake_agent"),
            "mistake_reason": record.get("mistake_reason"),
        },
    )

    return BenchmarkCase(
        case_id=f"who_and_when:{case_id}", trace=trace, spans=spans, steps=steps,
        ground_truth_step_index=mistake_step, ground_truth_span_id=ground_truth_span_id,
        ground_truth_failure_class=FailureClass.UNKNOWN.value, is_primary=True,
    )


def load_cases(path: Path) -> list[BenchmarkCase]:
    """One malformed record (an out-of-range `mistake_step`, a missing
    field) is logged and skipped rather than aborting the whole fixture
    file, matching `trail.py`'s per-record isolation."""
    records = json.loads(path.read_text())
    cases: list[BenchmarkCase] = []
    for record in records:
        try:
            cases.append(_case_from_record(record))
        except Exception as e:  # noqa: BLE001 - per-record isolation, see docstring
            logger.warning(
                "Who&When record failed to convert to a benchmark case, skipped",
                extra={"event": "who_and_when_record_failed", "case_id": record.get("case_id"), "error": str(e)},
            )
    return cases
