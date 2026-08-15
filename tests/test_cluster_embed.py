"""Tests for `cluster_embed.py`. The id-masking test is the one that matters
most (CLAUDE.md's L5 section): without it, cluster structure is dominated by
order ids and timestamps and every diagnosis becomes its own cluster.
"""

from culprit.cluster_embed import embed_cards, l2_normalize, render_card
from culprit.signals import DivergenceCandidate
from culprit.synth_results import make_diagnosis, make_divergence, make_signal


def test_two_diagnoses_differing_only_by_order_id_render_byte_identical_cards():
    d1 = make_diagnosis(
        rationale="Order ORD-48213 failed because tool result was empty for user 991823.",
    )
    d2 = make_diagnosis(
        rationale="Order ORD-99101 failed because tool result was empty for user 442.",
    )

    assert render_card(d1) == render_card(d2)


def test_uuid_and_date_in_rationale_are_masked():
    d1 = make_diagnosis(
        rationale="Seen at 2026-08-11T10:00:00Z with id 550e8400-e29b-41d4-a716-446655440000.",
    )
    d2 = make_diagnosis(
        rationale="Seen at 2026-09-30T22:15:03Z with id 123e4567-e89b-12d3-a456-426614174000.",
    )

    card = render_card(d1)
    assert "<UUID>" in card
    assert "<DATE>" in card
    assert "550e8400" not in card
    assert render_card(d1) == render_card(d2)


def test_step_signature_and_expected_are_masked_when_they_carry_ids():
    divergence = make_divergence(
        step_index=0,
        observed_signature="tool:agent:search_order_48213:ok",
        expected_signatures=[("tool:agent:search_order_99101:ok", 0.9)],
    )
    diagnosis = make_diagnosis(root_cause_step_index=0, divergences=[divergence])

    card = render_card(diagnosis)

    assert "48213" not in card
    assert "<ID>" in card


def test_render_card_picks_the_divergence_matching_root_cause_step_index():
    other_step = make_divergence(step_index=1, observed_signature="tool:agent:other_tool:ok")
    root_step = make_divergence(step_index=2, observed_signature="tool:agent:root_tool:ok")
    diagnosis = make_diagnosis(root_cause_step_index=2, divergences=[other_step, root_step])

    card = render_card(diagnosis)

    assert "root_tool" in card
    assert "other_tool" not in card


def test_render_card_falls_back_to_first_divergence_when_no_step_matches():
    d = make_diagnosis(
        root_cause_step_index=99,
        divergences=[make_divergence(step_index=0, observed_signature="tool:agent:fallback_tool:ok")],
    )

    assert "fallback_tool" in render_card(d)


def test_render_card_handles_no_divergences_or_signals_without_raising():
    d = make_diagnosis(divergences=[], signals=[])

    card = render_card(d)

    assert "observed: " in card
    assert "detectors: " in card


def test_detector_names_are_sorted_deduped_and_exclude_errored_signals():
    signals = [
        make_signal(detector="zeta_detector"),
        make_signal(detector="alpha_detector"),
        make_signal(detector="alpha_detector"),
        make_signal(detector="broken_detector", error="boom"),
    ]
    d = make_diagnosis(signals=signals)

    card = render_card(d)

    assert "detectors: alpha_detector, zeta_detector" in card
    assert "broken_detector" not in card


def test_rationale_is_truncated_to_300_chars_before_masking():
    d = make_diagnosis(rationale="x" * 500)

    card = render_card(d)

    rationale_line = next(line for line in card.splitlines() if line.startswith("rationale: "))
    assert len(rationale_line) == len("rationale: ") + 300


def test_l2_normalize_produces_unit_length_vector():
    normalized = l2_normalize([3.0, 4.0])

    assert abs(sum(x * x for x in normalized) ** 0.5 - 1.0) < 1e-9


def test_l2_normalize_handles_zero_vector_without_dividing_by_zero():
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_embed_cards_calls_embed_fn_once_with_every_card_and_normalizes():
    diagnoses = [make_diagnosis(), make_diagnosis()]
    seen_batches = []

    def fake_embed_fn(texts):
        seen_batches.append(texts)
        return [[1.0, 0.0] for _ in texts]

    vectors = embed_cards(diagnoses, embed_fn=fake_embed_fn)

    assert len(seen_batches) == 1  # one batched call, not one per diagnosis
    assert len(seen_batches[0]) == 2
    assert vectors == [[1.0, 0.0], [1.0, 0.0]]
