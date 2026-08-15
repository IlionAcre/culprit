"""OTLP-to-persisted-Trace composition: the one place that assembles a raw
ingested payload into a canonical `Trace` plus its `Span`/`Step` sequence
and writes it, via `otlp.decode` (or the direct-upload JSON path) ->
`normalize_span` -> `linearize` -> `store_traces.write_trace`.

Flagged as an unbuilt gap by both WS-A (`store_traces.py`'s docstring) and
WS-G (`jobs.py`'s docstring): no frozen-phase contract composed those four
functions end to end, because no single workstream's contract called for
it - it needs `otlp.py`/`normalize.py`/`linearize.py` (WS-A) on one side and
`store_traces.write_trace` (WS-B) on the other. Integration task I2 fills
it in, matching `jobs.py`'s own flagged gap for `_default_ingest`.

**Two input shapes, one composition.** OTLP (protobuf or OTLP-JSON) carries
no trace-level `outcome`/`framework`/`task_goal` anywhere - OTel has no
standard field for any of the three - so `_ingest_otlp` infers them from
the decoded spans with documented, best-effort heuristics. The direct JSON
"raw_upload" envelope (`vocab/raw_upload.py`) carries all three explicitly
at the top level and `_ingest_raw_upload` reads them as-is, no heuristic
needed. The two shapes are told apart by payload content, not
`content_type` alone: real uploaders send both OTLP-JSON and the
raw_upload envelope as `application/json`, and only the body's top-level
keys distinguish them (`resourceSpans` vs a bare `spans` list), mirroring
the per-span distinguishing marker `vocab/raw_upload.py` already uses.
"""

import json
import logging
import uuid

from culprit.db import ConnFn
from culprit.embed import EmbedFn
from culprit.fingerprint import agent_key, task_key
from culprit.linearize import linearize, orphan_span_ids
from culprit.logging_config import LOGGER_NAME
from culprit.normalize import normalize_span
from culprit.otlp import decode
from culprit.schemas import AgentPayload, LlmPayload, Outcome, Span, SpanStatus, Step, Trace
from culprit.store_traces import write_trace

logger = logging.getLogger(LOGGER_NAME)


def _infer_outcome(spans: list[Span]) -> Outcome:
    """OTLP carries no trace-level outcome field; any ERROR-status span is
    the best signal available without a product-specific convention to read
    instead (e.g. a resource attribute no fixture or real exporter here
    defines). A documented heuristic, not a claim of correctness."""
    return Outcome.FAILURE if any(s.status == SpanStatus.ERROR for s in spans) else Outcome.SUCCESS


def _infer_task_goal(spans: list[Span]) -> str | None:
    """Best-effort: the first AGENT span's `input_text`, or failing that the
    first user-role message in the first LLM span with one. Neither is a
    standard OTel field; this exists because L2's reference-pool lookup and
    the L3 prompt both need *some* task_goal, and OTLP has nowhere canonical
    to put one."""
    for span in spans:
        if isinstance(span.payload, AgentPayload) and span.payload.input_text:
            return span.payload.input_text
    for span in spans:
        if isinstance(span.payload, LlmPayload):
            for message in span.payload.request_messages:
                if message.role == "user" and message.content:
                    return message.content
    return None


def _root_span_id(spans: list[Span]) -> str | None:
    root = next((s for s in spans if s.parent_span_id is None), None)
    return root.span_id if root is not None else (spans[0].span_id if spans else None)


def _finish_trace(
    trace_id: str, source: str, framework: str | None, task_goal: str | None,
    outcome: Outcome, spans: list[Span], steps: list[Step],
) -> Trace:
    """Builds the `Trace`, then fills `agent_key`/`task_key` in a second
    step: both fingerprint functions need fields (`framework`, `task_goal`)
    that only exist once the first `Trace` is built, so a single
    all-at-once constructor call cannot compute them - see
    `fingerprint.py`'s own docstring on the same dependency."""
    trace = Trace(
        trace_id=trace_id, source=source, outcome=outcome, task_goal=task_goal,
        framework=framework, root_span_id=_root_span_id(spans),
        span_count=len(spans), step_count=len(steps),
        metadata={"orphan_span_ids": orphan_span_ids(spans)},
    )
    return trace.model_copy(update={
        "agent_key": agent_key(trace, steps),
        "task_key": task_key(trace.task_goal),
    })


def _ingest_otlp(payload: bytes, content_type: str) -> tuple[Trace, list[Span], list[Step]]:
    raw_spans = decode(payload, content_type)
    trace_id = raw_spans[0]["trace_id"] if raw_spans else str(uuid.uuid4())
    spans = [normalize_span(raw, trace_id) for raw in raw_spans]
    steps = linearize(spans)
    trace = _finish_trace(
        trace_id, "otlp", None, _infer_task_goal(spans), _infer_outcome(spans), spans, steps,
    )
    return trace, spans, steps


def _ingest_raw_upload(payload: bytes) -> tuple[Trace, list[Span], list[Step]]:
    data = json.loads(payload)
    trace_id = data.get("trace_id") or str(uuid.uuid4())
    spans = [normalize_span(raw, trace_id) for raw in data.get("spans", [])]
    steps = linearize(spans)
    try:
        outcome = Outcome(data.get("outcome", "unknown"))
    except ValueError:
        outcome = Outcome.UNKNOWN
    trace = _finish_trace(
        trace_id, data.get("source") or "manual_upload", data.get("framework"),
        data.get("task_goal"), outcome, spans, steps,
    )
    return trace, spans, steps


def _is_raw_upload_envelope(payload: bytes, content_type_bare: str) -> bool:
    if content_type_bare != "application/json":
        return False
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return False
    return isinstance(data, dict) and "spans" in data and "resourceSpans" not in data


def ingest_payload(
    payload: bytes, content_type: str, conn_fn: ConnFn, *, embed_fn: EmbedFn | None = None,
) -> str:
    """Decode `payload`, normalize and linearize its spans, assemble a
    `Trace`, and persist all three via `store_traces.write_trace`. Returns
    the new `trace_id`. `embed_fn` is optional (default `None`, storing no
    `task_embedding`) so callers without an embedding model handy (most
    tests) can still exercise ingestion; `jobs.py`'s real caller always
    supplies one.
    """
    content_type_bare = content_type.split(";")[0].strip().lower()
    if _is_raw_upload_envelope(payload, content_type_bare):
        trace, spans, steps = _ingest_raw_upload(payload)
    else:
        trace, spans, steps = _ingest_otlp(payload, content_type)

    embedding = embed_fn([trace.task_goal or ""])[0] if embed_fn is not None else None
    write_trace(conn_fn, trace, spans, steps, embedding)
    logger.info(
        "trace ingested",
        extra={"event": "ingest_payload_completed", "trace_id": trace.trace_id, "span_count": len(spans)},
    )
    return trace.trace_id
