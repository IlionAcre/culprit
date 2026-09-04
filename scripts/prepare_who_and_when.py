"""Merge the Who&When dataset (github.com/mingyin1/Agents_Failure_Attribution,
`Who&When/Algorithm-Generated/*.json` and `Who&When/Hand-Crafted/*.json`) into
the single-file record list the `benchmarks/who_and_when.py` adapter consumes.

Per-file schema reality vs the adapter's guessed record shape: the guessed
`{case_id, messages: [{agent, content}], mistake_step, ...}` matches almost
nothing literally - files carry `question` (not `task_goal`), `history`
(not `messages`), agent identity in `name` (Algorithm-Generated) or only in
`role` with "(thought)"/"(-> Agent)" decorations (Hand-Crafted), and
`mistake_step` as a string. What DOES hold: `mistake_step` is a direct 0-based
index into `history`, which is what the adapter's linear step mapping needs.
This script bridges the two; the adapter itself is unchanged.

Usage: uv run python scripts/prepare_who_and_when.py [repo_dir] [out_file]
"""

import json
import re
import sys
from pathlib import Path

_ROLE_DECORATION_RE = re.compile(r"\s*\(.*\)\s*")

SUBSETS = {"Algorithm-Generated": "ag", "Hand-Crafted": "hc"}


def main() -> None:
    repo_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/benchmarks/who_and_when/repo/Who&When")
    out_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/benchmarks/who_and_when/who_and_when_all.json")

    if not repo_dir.exists():
        print(
            f"{repo_dir} does not exist. The Who&When dataset is not there; "
            'see "Getting the benchmark datasets" in README.md.',
            file=sys.stderr,
        )
        sys.exit(1)

    records = []
    for subset, prefix in SUBSETS.items():
        for path in sorted((repo_dir / subset).glob("*.json"), key=lambda p: int(p.stem)):
            raw = json.loads(path.read_text(encoding="utf-8"))
            messages = [
                {
                    "agent": h.get("name") or _ROLE_DECORATION_RE.sub("", h.get("role", "unknown")),
                    "content": h.get("content", ""),
                }
                for h in raw["history"]
            ]
            records.append({
                "case_id": f"{prefix}-{path.stem}",
                "task_goal": raw.get("question"),
                "messages": messages,
                "mistake_step": raw["mistake_step"],
                "mistake_agent": raw.get("mistake_agent"),
                "mistake_reason": raw.get("mistake_reason"),
            })

    if not records:
        print(
            f"{repo_dir} holds no Who&When dataset (0 records found). "
            'See "Getting the benchmark datasets" in README.md.',
            file=sys.stderr,
        )
        sys.exit(1)

    out_file.write_text(json.dumps(records), encoding="utf-8")
    print(f"wrote {len(records)} records to {out_file}")


if __name__ == "__main__":
    main()
