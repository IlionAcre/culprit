import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_log_file(monkeypatch, tmp_path):
    """Every test gets its own log file under tmp_path so a rotating file
    handler in one test cannot hold a Windows file handle open for another."""
    monkeypatch.setenv("CULPRIT_LOG_FILE", str(tmp_path / "culprit-test.jsonl"))


requires_db = pytest.mark.skipif(
    not os.environ.get("CULPRIT_TEST_DSN"),
    reason="needs CULPRIT_TEST_DSN; default test run stays fully offline",
)

SKIP_DATASET_TESTS_ENV = "CULPRIT_SKIP_DATASET_TESTS"


def require_dataset(path, prepare_script: str) -> None:
    """Skip a benchmark regression test unless its dataset is on disk.

    `data/` is gitignored, so these tests run for whoever prepared the
    benchmarks and skip for everyone else. That makes the suite's pass and
    skip counts machine-dependent, and README.md states one pair of numbers
    for every reader. `scripts/check_readme_counts.py` sets
    CULPRIT_SKIP_DATASET_TESTS so it always measures the counts a fresh
    clone produces, whether or not the machine running it has the data.
    """
    if os.environ.get(SKIP_DATASET_TESTS_ENV, "").strip():
        pytest.skip(f"{SKIP_DATASET_TESTS_ENV} set; measuring clean-clone counts")
    if not path.exists():
        pytest.skip(f"benchmark dataset {path} not found; run {prepare_script}")
