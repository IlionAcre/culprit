"""Verify that README.md test counts match the actual offline pytest suite.

This script runs the offline test suite with CULPRIT_TEST_DSN set to empty,
parses the test counts from the pytest output, compares them to the counts
claimed in README.md, and exits with status code 1 on any mismatch.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def parse_pytest_output(output: str) -> tuple[int, int, int]:
    """Parse passed, skipped, and collected counts from pytest output.

    If pytest outputs an explicit collected count, that count is used.
    Otherwise, collected is calculated as passed + skipped + failed + errors.
    """
    passed_match = re.search(r"(\d+)\s+passed\b", output)
    passed = int(passed_match.group(1)) if passed_match else 0

    skipped_match = re.search(r"(\d+)\s+skipped\b", output)
    skipped = int(skipped_match.group(1)) if skipped_match else 0

    failed_match = re.search(r"(\d+)\s+failed\b", output)
    failed = int(failed_match.group(1)) if failed_match else 0

    errors_match = re.search(r"(\d+)\s+errors?\b", output)
    errors = int(errors_match.group(1)) if errors_match else 0

    collected_match = re.search(r"collected\s+(\d+)\s+items\b", output) or re.search(
        r"(\d+)\s+tests?\s+collected\b", output
    )
    if collected_match:
        collected = int(collected_match.group(1))
    else:
        collected = passed + skipped + failed + errors

    if passed == 0 and skipped == 0 and collected == 0:
        raise ValueError(
            f"Could not parse test counts from pytest output:\n{output}"
        )

    return passed, skipped, collected


def parse_readme_counts(readme_text: str) -> tuple[int, int, int]:
    """Parse claimed passed, skipped, and collected counts from README.md."""
    match = re.search(
        r"Offline:\s*(\d+)\s+tests?\s+pass(?:ed)?,\s*(\d+)\s+skipped\s*\(\s*(\d+)\s+collected\s*\)",
        readme_text,
    )
    if not match:
        raise ValueError(
            "Could not find 'Offline: X tests pass, Y skipped (Z collected)' in README.md."
        )
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def run_offline_suite(repo_root: Path) -> tuple[str, int]:
    """Run the suite the way a fresh clone runs it, and return output and exit code.

    Two things make the counts machine-dependent, and both are pinned here so
    the number this script checks is the number a reader gets. Blanking
    CULPRIT_TEST_DSN skips the database-gated tests. Setting
    CULPRIT_SKIP_DATASET_TESTS skips the benchmark regression tests, which
    would otherwise run for whoever prepared the gitignored `data/` and skip
    for everyone else, so a maintainer's machine and CI disagreed by two.
    """
    env = os.environ.copy()
    env["CULPRIT_TEST_DSN"] = ""
    env["CULPRIT_SKIP_DATASET_TESTS"] = "1"
    uv_bin = shutil.which("uv") or "uv"
    proc = subprocess.run(
        [uv_bin, "run", "pytest", "-q"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    output = proc.stdout + "\n" + proc.stderr
    return output, proc.returncode


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    readme_path = repo_root / "README.md"

    if not readme_path.exists():
        print(f"Error: README.md not found at {readme_path}", file=sys.stderr)
        return 1

    readme_text = readme_path.read_text(encoding="utf-8")
    try:
        readme_passed, readme_skipped, readme_collected = parse_readme_counts(readme_text)
    except ValueError as exc:
        print(f"Error reading README.md claims: {exc}", file=sys.stderr)
        return 1

    print(
        "Running offline test suite via 'uv run pytest -q' with empty CULPRIT_TEST_DSN...",
        flush=True,
    )
    output, returncode = run_offline_suite(repo_root)

    # CI runs this script instead of a separate pytest step, so a red suite
    # has to fail here rather than being reported only as a count mismatch.
    if returncode != 0:
        print(output, file=sys.stderr)
        print(
            f"Error: offline test suite failed (pytest exit code {returncode}).",
            file=sys.stderr,
        )
        return 1

    try:
        suite_passed, suite_skipped, suite_collected = parse_pytest_output(output)
    except ValueError as exc:
        print(f"Error parsing test output: {exc}", file=sys.stderr)
        return 1

    if (suite_passed, suite_skipped, suite_collected) != (
        readme_passed,
        readme_skipped,
        readme_collected,
    ):
        print(
            f"Mismatch in test counts:\n"
            f"  Suite:  {suite_passed} passed, {suite_skipped} skipped ({suite_collected} collected)\n"
            f"  README: {readme_passed} passed, {readme_skipped} skipped ({readme_collected} collected)",
            file=sys.stderr,
        )
        return 1

    print(
        f"Test counts match: {suite_passed} passed, {suite_skipped} skipped ({suite_collected} collected)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
