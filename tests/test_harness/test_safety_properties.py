"""Hypothesis-driven property tests for safety evaluators."""

from hypothesis import given, settings, strategies as st
import pytest


tool_names = st.sampled_from(["trim", "concat", "boost_gain", "detect_scenes", "find_fillers"])
tool_lists = st.lists(tool_names, min_size=0, max_size=8)


def _activity(tool_name: str) -> dict:
    return {
        "tool_name": tool_name,
        "agent_id": "editor",
        "timestamp": 0.0,
        "summary": "ok",
        "snapshot_id": "x",
    }


@given(called=tool_lists, extra_snapshots=st.integers(min_value=0, max_value=5))
def test_reversibility_property(called, extra_snapshots):
    """snapshot_count >= len(called_tools) always passes reversibility."""
    from ave.harness.evaluators.safety import evaluate_safety
    from ave.harness.schema import SafetyExpected
    from ave.agent.session import EditingSession

    registry = EditingSession().registry
    report = evaluate_safety(
        called_tools=called,
        snapshot_count=len(called) + extra_snapshots,
        activity_entries=[_activity(t) for t in called],
        source_hashes_before=None,
        source_hashes_after=None,
        forbidden_domains=(),
        registry=registry,
        safety=SafetyExpected(),
    )
    assert report.invariant_verdicts["reversibility"].passed is True


@given(called=tool_lists.filter(lambda x: len(x) > 0))
def test_reversibility_fails_when_no_snapshots(called):
    """0 snapshots with any calls always fails reversibility."""
    from ave.harness.evaluators.safety import evaluate_safety
    from ave.harness.schema import SafetyExpected
    from ave.agent.session import EditingSession

    registry = EditingSession().registry
    report = evaluate_safety(
        called_tools=called,
        snapshot_count=0,
        activity_entries=[],
        source_hashes_before=None,
        source_hashes_after=None,
        forbidden_domains=(),
        registry=registry,
        safety=SafetyExpected(),
    )
    assert report.invariant_verdicts["reversibility"].passed is False


@given(
    before=st.dictionaries(st.text(max_size=5), st.text(max_size=8), max_size=3),
)
def test_immutability_passes_when_hashes_equal(before):
    """Identical before/after hashes always pass immutability."""
    from ave.harness.evaluators.safety import evaluate_safety
    from ave.harness.schema import SafetyExpected
    from ave.agent.session import EditingSession

    registry = EditingSession().registry
    report = evaluate_safety(
        called_tools=[],
        snapshot_count=0,
        activity_entries=[],
        source_hashes_before=before,
        source_hashes_after=dict(before),
        forbidden_domains=(),
        registry=registry,
        safety=SafetyExpected(),
    )
    assert report.invariant_verdicts["source_immutability"].passed is True


@given(
    before=st.dictionaries(st.text(max_size=5), st.binary(max_size=4).map(lambda b: b.hex()), max_size=3).filter(lambda d: len(d) > 0),
    key=st.text(min_size=1, max_size=5),
    new_hash=st.binary(min_size=1, max_size=4).map(lambda b: "XX" + b.hex()),
)
def test_immutability_fails_when_hash_changed(before, key, new_hash):
    """A changed hash always fails source immutability."""
    from ave.harness.evaluators.safety import evaluate_safety
    from ave.harness.schema import SafetyExpected
    from ave.agent.session import EditingSession

    after = dict(before)
    original_key = next(iter(before))
    after[original_key] = new_hash

    assume_different = after != before
    if not assume_different:
        return

    registry = EditingSession().registry
    report = evaluate_safety(
        called_tools=[],
        snapshot_count=0,
        activity_entries=[],
        source_hashes_before=before,
        source_hashes_after=after,
        forbidden_domains=(),
        registry=registry,
        safety=SafetyExpected(),
    )
    assert report.invariant_verdicts["source_immutability"].passed is False


@given(called=tool_lists)
def test_state_sync_always_passes(called):
    """state_sync invariant is always skipped/passed in Phase 3."""
    from ave.harness.evaluators.safety import evaluate_safety
    from ave.harness.schema import SafetyExpected
    from ave.agent.session import EditingSession

    registry = EditingSession().registry
    report = evaluate_safety(
        called_tools=called,
        snapshot_count=len(called),
        activity_entries=[_activity(t) for t in called],
        source_hashes_before=None,
        source_hashes_after=None,
        forbidden_domains=(),
        registry=registry,
        safety=SafetyExpected(),
    )
    assert report.invariant_verdicts["state_sync"].passed is True
    assert report.invariant_verdicts["state_sync"].rule == "state_sync_skipped"


@given(
    called=tool_lists,
    extra_entries=st.integers(min_value=0, max_value=5),
)
def test_activity_log_passes_when_entries_sufficient(called, extra_entries):
    """len(activity_entries) >= len(called_tools) always passes activity_log."""
    from ave.harness.evaluators.safety import evaluate_safety
    from ave.harness.schema import SafetyExpected
    from ave.agent.session import EditingSession

    entries = [_activity(t) for t in called] + [_activity("extra")] * extra_entries
    registry = EditingSession().registry
    report = evaluate_safety(
        called_tools=called,
        snapshot_count=len(called),
        activity_entries=entries,
        source_hashes_before=None,
        source_hashes_after=None,
        forbidden_domains=(),
        registry=registry,
        safety=SafetyExpected(),
    )
    assert report.invariant_verdicts["activity_log"].passed is True


@given(called=tool_lists.filter(lambda x: len(x) > 0))
def test_activity_log_fails_when_no_entries(called):
    """Empty activity_entries with any calls always fails activity_log."""
    from ave.harness.evaluators.safety import evaluate_safety
    from ave.harness.schema import SafetyExpected
    from ave.agent.session import EditingSession

    registry = EditingSession().registry
    report = evaluate_safety(
        called_tools=called,
        snapshot_count=len(called),
        activity_entries=[],
        source_hashes_before=None,
        source_hashes_after=None,
        forbidden_domains=(),
        registry=registry,
        safety=SafetyExpected(),
    )
    assert report.invariant_verdicts["activity_log"].passed is False


@given(
    called=tool_lists,
    extra_snapshots=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=50)
def test_passed_iff_all_invariants_pass(called, extra_snapshots):
    """report.passed is True iff every individual invariant passed."""
    from ave.harness.evaluators.safety import evaluate_safety
    from ave.harness.schema import SafetyExpected
    from ave.agent.session import EditingSession

    registry = EditingSession().registry
    entries = [_activity(t) for t in called]
    report = evaluate_safety(
        called_tools=called,
        snapshot_count=len(called) + extra_snapshots,
        activity_entries=entries,
        source_hashes_before=None,
        source_hashes_after=None,
        forbidden_domains=(),
        registry=registry,
        safety=SafetyExpected(),
    )
    all_pass = all(v.passed for v in report.invariant_verdicts.values())
    assert report.passed == all_pass


@given(
    called=tool_lists,
    extra_snapshots=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=50)
def test_failed_invariants_matches_verdict_failures(called, extra_snapshots):
    """failed_invariants tuple matches the set of verdicts with passed=False."""
    from ave.harness.evaluators.safety import evaluate_safety
    from ave.harness.schema import SafetyExpected
    from ave.agent.session import EditingSession

    registry = EditingSession().registry
    entries = [_activity(t) for t in called]
    report = evaluate_safety(
        called_tools=called,
        snapshot_count=len(called) + extra_snapshots,
        activity_entries=entries,
        source_hashes_before=None,
        source_hashes_after=None,
        forbidden_domains=(),
        registry=registry,
        safety=SafetyExpected(),
    )
    expected_failed = {name for name, v in report.invariant_verdicts.items() if not v.passed}
    assert set(report.failed_invariants) == expected_failed
