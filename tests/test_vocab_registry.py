from culprit.vocab import registry


def test_detect_recognizes_otel_genai_by_its_operation_name_attribute():
    raw = {"attributes": {"gen_ai.operation.name": "chat"}}

    assert registry.detect(raw) == "otel_genai"


def test_detect_recognizes_openinference_by_its_span_kind_attribute():
    raw = {"attributes": {"openinference.span.kind": "LLM"}}

    assert registry.detect(raw) == "openinference"


def test_detect_recognizes_raw_upload_by_its_direct_upload_shape():
    raw = {"kind": "tool", "start_time": "2026-01-01T00:00:00Z", "end_time": "2026-01-01T00:00:01Z"}

    assert registry.detect(raw) == "raw_upload"


def test_detect_returns_unknown_for_an_unrecognized_vocabulary():
    """This is the guardrail path: an unrecognized span must never raise
    here, only be reported as "unknown" so normalize.py can set
    normalize_error while keeping every raw attribute."""
    raw = {"attributes": {"vendor.weird.key": 42}}

    assert registry.detect(raw) == "unknown"


def test_detect_never_raises_on_a_span_dict_missing_attributes_entirely():
    assert registry.detect({}) == "unknown"


def test_registry_registers_all_three_vocabularies():
    assert set(registry.VOCABULARIES) == {"otel_genai", "openinference", "raw_upload"}
