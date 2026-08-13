"""Raw span dict -> canonical `Span`. Detects the attribute vocabulary via
`vocab.registry` and dispatches to the matching module's `to_span()`. This is
the single point where per-span failures get isolated: a raise anywhere
inside a vocab module (bad JSON in an attribute, a missing required field)
must never abort ingestion of the trace's other spans, so it is caught here,
once, rather than duplicated in every vocab module. See CLAUDE.md's guardrail:
raw attributes are never discarded, even for a span whose vocabulary is
unrecognized or whose normalization raised - a later vocabulary module
depends on that data still being there.
"""

import logging

from culprit.logging_config import LOGGER_NAME
from culprit.schemas import Span, SpanKind, SpanStatus
from culprit.vocab import registry

logger = logging.getLogger(LOGGER_NAME)

_STATUS_CODE_MAP = {
    "STATUS_CODE_OK": SpanStatus.OK,
    "STATUS_CODE_ERROR": SpanStatus.ERROR,
}


def _fallback_status(raw: dict) -> SpanStatus:
    code = raw.get("status_code") or raw.get("status")
    return _STATUS_CODE_MAP.get(code, SpanStatus.UNSET)


def _fallback_span(raw: dict, trace_id: str, vocabulary: str, error: str) -> Span:
    """A span we could not normalize still becomes a valid `Span`: kind
    UNKNOWN (so linearize.py folds it into the enclosing step's
    collapsed_span_ids rather than inventing a bogus step), payload None,
    every raw attribute retained, normalize_error explaining why."""
    return Span(
        trace_id=trace_id,
        span_id=str(raw.get("span_id", "")),
        parent_span_id=raw.get("parent_span_id"),
        name=str(raw.get("name", "")),
        kind=SpanKind.UNKNOWN,
        status=_fallback_status(raw),
        status_message=raw.get("status_message"),
        start_ns=int(raw.get("start_ns", 0) or 0),
        end_ns=int(raw.get("end_ns", 0) or 0),
        vocabulary=vocabulary,
        attributes=dict(raw.get("attributes", {})),
        payload=None,
        normalize_error=error,
    )


def normalize_span(raw: dict, trace_id: str) -> Span:
    vocabulary = registry.detect(raw)
    if vocabulary == "unknown":
        return _fallback_span(raw, trace_id, vocabulary, "no vocabulary matched")

    module = registry.VOCABULARIES[vocabulary]
    try:
        return module.to_span(raw, trace_id)
    except Exception as e:
        logger.warning(
            "span failed to normalize",
            extra={
                "event": "normalize_span_failed",
                "vocabulary": vocabulary,
                "span_id": raw.get("span_id"),
                "error": str(e),
            },
        )
        return _fallback_span(raw, trace_id, vocabulary, str(e))
