from pathlib import Path

from culprit.benchmarks.registry import BENCHMARKS

_FIXTURES = Path(__file__).parent / "fixtures" / "benchmarks"


def test_registry_has_exactly_the_two_supported_benchmarks():
    assert set(BENCHMARKS) == {"trail", "who_and_when"}


def test_every_registered_adapter_loads_its_own_3_record_sample_fixture():
    for name, load_cases in BENCHMARKS.items():
        fixture = _FIXTURES / f"{name}_sample.json"
        cases = load_cases(fixture)
        # 3 records per fixture; a record with multiple annotated errors
        # legitimately yields more than one case (see trail.py).
        assert len({c.trace.trace_id for c in cases}) == 3, (
            f"{name} sample fixture should convert all 3 records"
        )
