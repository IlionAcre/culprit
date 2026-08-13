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
