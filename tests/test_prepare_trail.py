"""Tests for scripts/prepare_trail.py's own CLI behavior: what it does
when the raw TRAIL dataset it is pointed at is missing or empty, versus
when it holds a real record. Runs the script as a subprocess (it is a
standalone script, not a package module) - see test_migrations.py for the
same pattern with alembic.
"""

import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "prepare_trail.py"


def _run(dataset_dir: Path, out_file: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), str(dataset_dir), str(out_file)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_missing_dataset_dir_exits_nonzero_and_names_the_dir(tmp_path):
    dataset_dir = tmp_path / "no_such_dir"
    out_file = tmp_path / "trail_all.json"

    result = _run(dataset_dir, out_file)

    assert result.returncode != 0
    assert str(dataset_dir) in result.stderr
    assert not out_file.exists()


def test_empty_dataset_dir_exits_nonzero_and_names_the_dir(tmp_path):
    dataset_dir = tmp_path / "empty"
    dataset_dir.mkdir()
    out_file = tmp_path / "trail_all.json"

    result = _run(dataset_dir, out_file)

    assert result.returncode != 0
    assert str(dataset_dir) in result.stderr
    assert not out_file.exists()


def test_one_valid_record_lands_in_the_output(tmp_path):
    dataset_dir = tmp_path / "trail"
    (dataset_dir / "GAIA").mkdir(parents=True)
    (dataset_dir / "processed_annotations_gaia").mkdir(parents=True)
    (dataset_dir / "GAIA" / "trace-001.json").write_text(
        json.dumps({"spans": [{"span_id": "s1"}]})
    )
    (dataset_dir / "processed_annotations_gaia" / "trace-001.json").write_text(
        json.dumps({"errors": [{"span_id": "s1"}]})
    )
    out_file = tmp_path / "trail_all.json"

    result = _run(dataset_dir, out_file)

    assert result.returncode == 0
    records = json.loads(out_file.read_text())
    assert len(records) == 1
    assert records[0]["trace_id"] == "trace-001"
    assert records[0]["split"] == "GAIA"
    assert records[0]["spans"] == [{"span_id": "s1"}]
    assert records[0]["errors"] == [{"span_id": "s1"}]
