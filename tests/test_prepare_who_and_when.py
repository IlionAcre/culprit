"""Tests for scripts/prepare_who_and_when.py's own CLI behavior: what it
does when the Who&When repo it is pointed at is missing or empty, versus
when it holds a real record. Runs the script as a subprocess (it is a
standalone script, not a package module) - see test_migrations.py for the
same pattern with alembic.
"""

import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "prepare_who_and_when.py"


def _run(repo_dir: Path, out_file: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), str(repo_dir), str(out_file)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_missing_repo_dir_exits_nonzero_and_names_the_dir(tmp_path):
    repo_dir = tmp_path / "no_such_dir"
    out_file = tmp_path / "who_and_when_all.json"

    result = _run(repo_dir, out_file)

    assert result.returncode != 0
    assert str(repo_dir) in result.stderr
    assert not out_file.exists()


def test_empty_repo_dir_exits_nonzero_and_names_the_dir(tmp_path):
    repo_dir = tmp_path / "empty"
    repo_dir.mkdir()
    out_file = tmp_path / "who_and_when_all.json"

    result = _run(repo_dir, out_file)

    assert result.returncode != 0
    assert str(repo_dir) in result.stderr
    assert not out_file.exists()


def test_one_valid_record_lands_in_the_output(tmp_path):
    repo_dir = tmp_path / "Who&When"
    (repo_dir / "Algorithm-Generated").mkdir(parents=True)
    (repo_dir / "Algorithm-Generated" / "1.json").write_text(
        json.dumps(
            {
                "question": "do the thing",
                "history": [{"name": "agent_a", "content": "hi"}],
                "mistake_step": "0",
                "mistake_agent": "agent_a",
                "mistake_reason": "wrong tool",
            }
        )
    )
    out_file = tmp_path / "who_and_when_all.json"

    result = _run(repo_dir, out_file)

    assert result.returncode == 0
    records = json.loads(out_file.read_text())
    assert len(records) == 1
    assert records[0]["case_id"] == "ag-1"
    assert records[0]["task_goal"] == "do the thing"
    assert records[0]["messages"] == [{"agent": "agent_a", "content": "hi"}]
    assert records[0]["mistake_step"] == "0"
