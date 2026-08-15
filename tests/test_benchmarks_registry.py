from pathlib import Path

from culprit.benchmarks.registry import BENCHMARKS

_FIXTURES = Path(__file__).parent / "fixtures" / "benchmarks"


def test_registry_has_exactly_the_two_supported_benchmarks():
    assert set(BENCHMARKS) == {"trail", "who_and_when"}


def test_every_registered_adapter_loads_its_own_3_record_sample_fixture():
    for name, load_cases in BENCHMARKS.items():
        fixture = _FIXTURES / f"{name}_sample.json"
        cases = load_cases(fixture)
        assert len(cases) == 3, f"{name} sample fixture should yield 3 cases"
