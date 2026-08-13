from culprit.taxonomy import _DESCRIPTIONS, FailureClass, describe_all


def test_describe_all_emits_every_failure_class_so_prompt_and_code_cannot_drift():
    """The L3 prompt is built by formatting this block in. If a class were
    added to the enum but not the description map, the model would be asked
    to choose from a set the code does not accept."""
    block = describe_all()

    for member in FailureClass:
        assert member.value in block


def test_every_failure_class_has_a_description_entry():
    """Direct check on the description map itself, so adding a member to the
    enum without a matching description fails a test immediately rather than
    only failing indirectly if something happens to call describe_all()."""
    assert set(_DESCRIPTIONS.keys()) == set(FailureClass)


def test_failure_class_has_exactly_twenty_one_members_across_the_documented_groups():
    """Locks the taxonomy size in place: 3 + 5 + 3 + 3 + 3 + 2 + 1 + 1
    (specification/planning, tool/environment, information handling,
    control flow, multi-agent, verification, output, unknown)."""
    assert len(list(FailureClass)) == 21


def test_unknown_is_the_explicit_escape_hatch_member():
    assert FailureClass.UNKNOWN.value == "unknown"
