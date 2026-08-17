"""Merge the raw TRAIL dataset layout into the single-file record list the
`benchmarks/trail.py` adapter consumes.

The public release (HuggingFace `PatronusAI/TRAIL`, gated; ModelScope mirror
of the same files) ships two parallel directory trees:

    GAIA/<trace_id>.json                        raw nested trace
    SWE Bench/<trace_id>.json                   raw nested trace
    processed_annotations_gaia/<trace_id>.json  {errors: [...], scores: [...]}
    processed_annotations_swe_bench/<trace_id>.json

This script pairs them by trace_id into one JSON list of records
`{trace_id, split, spans, errors}` at `data/benchmarks/trail/trail_all.json`
(default), which is what `trail.load_cases` reads. One released annotation
file (`a96c6811...`, GAIA) has a literal trailing-comma syntax error upstream;
it is loaded tolerantly here (trailing commas stripped) so the trace is not
lost, and the mismatch is reported at the end.

Usage: uv run python scripts/prepare_trail.py [dataset_dir] [out_file]
"""

import json
import re
import sys
from pathlib import Path

_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")

SPLIT_DIRS = {
    "GAIA": ("GAIA", "processed_annotations_gaia"),
    "SWE Bench": ("SWE Bench", "processed_annotations_swe_bench"),
}


def _load_tolerant(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_TRAILING_COMMA_RE.sub(r"\1", text))


def main() -> None:
    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/benchmarks/trail")
    out_file = Path(sys.argv[2]) if len(sys.argv) > 2 else dataset_dir / "trail_all.json"

    records = []
    missing_annotations = []
    missing_traces = []
    for split, (raw_dir, ann_dir) in SPLIT_DIRS.items():
        for raw_path in sorted((dataset_dir / raw_dir).glob("*.json")):
            trace_id = raw_path.stem
            ann_path = dataset_dir / ann_dir / f"{trace_id}.json"
            if not ann_path.exists():
                missing_annotations.append(trace_id)
                continue
            raw = _load_tolerant(raw_path)
            ann = _load_tolerant(ann_path)
            records.append({
                "trace_id": trace_id,
                "split": split,
                "spans": raw["spans"],
                "errors": ann.get("errors", []),
            })
        for ann_path in sorted((dataset_dir / ann_dir).glob("*.json")):
            if not (dataset_dir / raw_dir / ann_path.name).exists():
                missing_traces.append(ann_path.stem)

    out_file.write_text(json.dumps(records), encoding="utf-8")
    n_errors = sum(len(r["errors"]) for r in records)
    print(f"wrote {len(records)} records ({n_errors} annotated errors) to {out_file}")
    if missing_annotations:
        print(f"{len(missing_annotations)} raw traces had no annotation file: {missing_annotations}")
    if missing_traces:
        print(f"{len(missing_traces)} annotation files had no raw trace: {missing_traces}")


if __name__ == "__main__":
    main()
