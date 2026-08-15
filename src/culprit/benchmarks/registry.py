"""Benchmark name -> adapter registry. House pattern from
`detectors/registry.py`: a `Protocol` (`base.BenchmarkAdapter`) plus a
module-level dict of plain callables, not classes. Sole owner per CLAUDE.md's
"one owner per plugin registry" rule - only WS-H (this workstream) edits
this file.
"""

from culprit.benchmarks import trail, who_and_when
from culprit.benchmarks.base import BenchmarkAdapter

BENCHMARKS: dict[str, BenchmarkAdapter] = {
    "trail": trail.load_cases,
    "who_and_when": who_and_when.load_cases,
}
