"""End-to-end WS-D tests against `synth.py`/`synth_inject.py`. Every test
uses a dict-backed `neighbor_fn` and `reference_loader` fake, never
`store_traces` or a real Postgres connection, per the injection contract in
`signals.py.NeighborFn` and this module's own `ReferenceLoaderFn`.
"""

from culprit.contrast import contrast
from culprit.synth import SynthRun, successful_run
from culprit.synth_inject import INJECTION_KINDS, inject


def _pool(seeds: list[int]) -> dict[str, SynthRun]:
    return {f"synth-{s}": successful_run(s) for s in seeds}


def _fakes(runs: dict[str, SynthRun]):
    ids = list(runs.keys())

    def neighbor_fn(embedding, k):
        return ids[:k]

    def reference_loader(conn_fn, trace_id):
        return runs[trace_id].steps

    return neighbor_fn, reference_loader


def _fake_embed(texts):
    return [[0.0] * 4 for _ in texts]


def _run_contrast(run: SynthRun, runs: dict[str, SynthRun]):
    neighbor_fn, loader = _fakes(runs)
    return contrast(
        run.trace, run.steps, neighbor_fn=neighbor_fn, conn_fn=lambda: None,
        embed_fn=_fake_embed, reference_loader=loader,
    )


def test_abstains_below_three_references():
    runs = _pool([1, 2])
    run = successful_run(99)

    result = _run_contrast(run, runs)

    assert result.abstained
    assert result.abstain_reason == "insufficient_references"
    assert result.candidates == []


def test_empty_trace_abstains_without_touching_neighbor_fn():
    calls = []
    run = successful_run(1)

    def neighbor_fn(embedding, k):
        calls.append(k)
        return []

    result = contrast(
        run.trace, [], neighbor_fn=neighbor_fn, conn_fn=lambda: None,
        embed_fn=_fake_embed, reference_loader=lambda conn_fn, tid: [],
    )

    assert result.abstained
    assert result.abstain_reason == "empty_trace"
    assert calls == []


def test_identical_to_reference_pool_yields_zero_candidates():
    """DoD: identical sequences score maximum and yield zero divergence
    candidates. `run` is literally one of the references in its own pool."""
    seeds = list(range(1, 11))
    runs = _pool(seeds)
    run = runs[f"synth-{seeds[0]}"]

    result = _run_contrast(run, runs)

    assert not result.abstained
    assert result.reference_count == len(seeds)
    assert result.candidates == []


def test_top1_divergence_matches_injection_ground_truth_for_most_kinds():
    """DoD: top-1 divergence equals the injection index for >=80% of
    injection kinds."""
    seeds = list(range(1, 16))
    runs = _pool(seeds)

    hits, misses = 0, []
    for kind in sorted(INJECTION_KINDS):
        base = successful_run(1000)
        mutated, ground_truth = inject(base, kind, at_step=3)

        result = _run_contrast(mutated, runs)

        top1 = result.candidates[0].step_index if result.candidates else None
        if top1 == ground_truth:
            hits += 1
        else:
            misses.append((kind, ground_truth, top1))

    accuracy = hits / len(INJECTION_KINDS)
    assert accuracy >= 0.8, f"top-1 accuracy {accuracy:.2f} ({hits}/{len(INJECTION_KINDS)}), misses: {misses}"


def test_candidates_report_the_resolved_reference_count():
    seeds = list(range(1, 9))
    runs = _pool(seeds)
    base = successful_run(2000)
    mutated, _ = inject(base, "tool_error", at_step=3)

    result = _run_contrast(mutated, runs)

    assert not result.abstained
    for c in result.candidates:
        assert c.reference_count == len(seeds)


def test_candidates_are_capped_at_five_and_sorted_by_descending_score():
    seeds = list(range(1, 16))
    runs = _pool(seeds)
    base = successful_run(3000)
    mutated, _ = inject(base, "retry_storm", at_step=3)

    result = _run_contrast(mutated, runs)

    assert len(result.candidates) <= 5
    scores = [c.divergence_score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)


def test_alignment_op_is_one_of_the_documented_values():
    seeds = list(range(1, 12))
    runs = _pool(seeds)
    base = successful_run(4000)
    mutated, _ = inject(base, "hallucinated_tool", at_step=3)

    result = _run_contrast(mutated, runs)

    for c in result.candidates:
        assert c.alignment_op in ("match", "mismatch", "insertion", "deletion")
