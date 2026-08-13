"""Extension-point contract for attribute vocabularies: one module per
vocabulary (OTel GenAI, OpenInference, the direct-upload shape, and whatever
ships next), registered by name in `registry.py`.

`typing.Protocol`, not an ABC, per this project's plugin convention: nothing
subclasses anything, and any object exposing `name`, `matches`, and
`to_span` satisfies the contract structurally. That is what lets
`registry.py` treat every vocabulary the same way without importing the
modules' concrete classes.
"""

import json
from typing import Any, Protocol

from culprit.schemas import Span, SpanStatus


class VocabModule(Protocol):
    name: str

    def matches(self, raw: dict) -> bool:
        """Cheap, side-effect-free membership test on a single raw span
        dict. `registry.detect()` may call this on every registered module
        in turn, so it must never raise."""
        ...

    def to_span(self, raw: dict, trace_id: str) -> Span:
        """Build the canonical Span. May raise on malformed
        vocabulary-specific data (unparseable JSON in an attribute, a
        missing required field): `normalize.py` is the single place that
        catches it and degrades to `normalize_error`, so implementations do
        not need their own try/except around this."""
        ...


_OTLP_STATUS = {
    "STATUS_CODE_OK": SpanStatus.OK,
    "STATUS_CODE_ERROR": SpanStatus.ERROR,
    "STATUS_CODE_UNSET": SpanStatus.UNSET,
}


def otlp_status(code: str | None) -> SpanStatus:
    """Shared by both OTLP-transport vocab modules (otel_genai,
    openinference): they ride the same OTel `Status.code` enum, decoded to
    the same string by `otlp.py` regardless of wire format. The
    direct-upload vocabulary does not use this - its status strings already
    equal `SpanStatus`'s own values, so it constructs `SpanStatus` directly."""
    return _OTLP_STATUS.get(code or "", SpanStatus.UNSET)


def parse_json_attr(attrs: dict, key: str, default: Any) -> Any:
    """Both OTLP vocabularies carry structured data (message lists,
    retrieved documents, tool arguments) as a JSON-encoded string attribute
    rather than nested AnyValue - decoding it is identical work in both
    modules, so it lives here instead of twice. Returns `default` when the
    key is absent (a genuinely missing attribute is not an error); raises
    when the key is present but not valid JSON, since that is real evidence
    something upstream is broken and normalize.py needs to see it."""
    raw = attrs.get(key)
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"attribute {key!r} is not valid JSON: {e}") from e
