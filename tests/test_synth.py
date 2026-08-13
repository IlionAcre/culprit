from culprit.synth import SynthRun, successful_run


def test_successful_runs_vary_in_length_and_order_across_seeds():
    """A reference pool of identical traces teaches the profile nothing, and
    every legitimate variation would then score as a divergence."""
    runs = [successful_run(seed=i) for i in range(20)]

    lengths = {r.trace.step_count for r in runs}
    assert len(lengths) > 1

    signatures_by_seed = [tuple(s.signature for s in r.steps) for r in runs]
    assert len(set(signatures_by_seed)) > 1


def test_successful_run_is_reproducible_for_the_same_seed():
    """Generation is seeded: rerunning with the same seed must not silently
    drift, or a benchmark comparing two runs of the harness would be
    comparing apples to noise."""
    first = successful_run(seed=7)
    second = successful_run(seed=7)

    assert [s.signature for s in first.steps] == [s.signature for s in second.steps]
    assert first.trace.step_count == second.trace.step_count


def test_successful_run_returns_a_bundle_with_trace_spans_and_steps_together():
    """Trace deliberately carries no spans or steps (schemas.py); every real
    consumer (run_detectors, contrast) needs all three, so successful_run
    hands them back together rather than requiring a separate lookup."""
    run = successful_run(seed=3)

    assert isinstance(run, SynthRun)
    assert len(run.spans) == len(run.steps) == run.trace.step_count
    assert run.trace.span_count == run.trace.step_count


def test_spans_and_steps_align_by_index():
    """Every injector in synth_inject.py indexes `spans[i]` and `steps[i]`
    together and assumes they describe the same event; this is the
    invariant that makes that safe."""
    run = successful_run(seed=1)

    assert all(span.span_id == step.span_id for span, step in zip(run.spans, run.steps))


def test_two_successful_runs_do_not_share_the_same_object_lists():
    """successful_run must not hand out references into any shared module
    state: two calls' step lists have to be independently mutable."""
    first = successful_run(seed=4)
    second = successful_run(seed=4)

    first.steps[0].summary = "mutated for this test only"

    assert second.steps[0].summary != "mutated for this test only"
