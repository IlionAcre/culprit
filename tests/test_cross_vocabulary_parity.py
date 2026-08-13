"""The definition-of-done test for the whole ingestion workstream (CLAUDE.md):
the OTel GenAI and OpenInference fixtures deliberately encode the same
logical agent run - same span order, actor, tool name, document count, and
timestamps - so that correct normalization must produce identical `Step`
sequences from both. Any drift here means the normalizers are encoding
vocabulary-specific noise into the canonical model instead of the actual
semantics, which is the one thing L1-L3 downstream can never see past.
"""

import json
from pathlib import Path

from culprit.linearize import linearize
from culprit.normalize import normalize_span
from culprit.otlp import decode

_FIXTURES = Path(__file__).parent / "fixtures" / "otlp"


def _steps_for(fixture_name: str) -> list:
    raw_bytes = (_FIXTURES / fixture_name).read_bytes()
    raw_spans = decode(raw_bytes, "application/json")
    spans = [normalize_span(raw, "shared-trace-id") for raw in raw_spans]
    return linearize(spans)


def _strip_span_identity(step):
    """Everything about a Step except the fields that are legitimately
    fixture-specific: the underlying span_id (each file mints its own) and
    whatever non-semantic span ids got collapsed into it (also
    fixture-specific ids, and here there are none to collapse in either
    fixture)."""
    return step.model_copy(update={"span_id": "", "collapsed_span_ids": []})


def test_otel_genai_and_openinference_fixtures_normalize_to_identical_steps():
    otel_steps = _steps_for("otel_genai_sample.json")
    openinference_steps = _steps_for("openinference_sample.json")

    assert len(otel_steps) == len(openinference_steps) == 5

    stripped_otel = [_strip_span_identity(s) for s in otel_steps]
    stripped_openinference = [_strip_span_identity(s) for s in openinference_steps]

    assert stripped_otel == stripped_openinference


def test_the_two_fixtures_do_not_trivially_share_span_ids():
    """Guards against the comparison above being vacuous - if the fixtures
    happened to reuse span ids, stripping span_id wouldn't be doing
    anything, and the test above would pass even if per-span normalization
    were badly broken elsewhere."""
    otel_ids = {s["spanId"] for s in _resource_spans("otel_genai_sample.json")}
    openinference_ids = {s["spanId"] for s in _resource_spans("openinference_sample.json")}

    assert otel_ids.isdisjoint(openinference_ids)


def _resource_spans(fixture_name: str) -> list[dict]:
    data = json.loads((_FIXTURES / fixture_name).read_bytes())
    return [
        span
        for rs in data["resourceSpans"]
        for ss in rs["scopeSpans"]
        for span in ss["spans"]
    ]


def test_each_step_kind_sequence_matches_the_documented_logical_run():
    """CLAUDE.md: an agent calling search_orders, retrieving 2 policy
    documents, then answering - invoke_agent/AGENT, chat/LLM,
    execute_tool/TOOL, retrieve_documents/RETRIEVER, chat/LLM."""
    from culprit.schemas import SpanKind

    expected_kinds = [SpanKind.AGENT, SpanKind.LLM, SpanKind.TOOL, SpanKind.RETRIEVER, SpanKind.LLM]

    assert [s.kind for s in _steps_for("otel_genai_sample.json")] == expected_kinds
    assert [s.kind for s in _steps_for("openinference_sample.json")] == expected_kinds
