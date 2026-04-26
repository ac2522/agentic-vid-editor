"""Pure safety invariant evaluator tests."""

from ave.agent.domains import Domain
from ave.agent.registry import ToolRegistry
from ave.harness.schema import SafetyExpected


def _minimal_registry() -> ToolRegistry:
    reg = ToolRegistry()

    def do_audio(x: int) -> None:
        """Audio op."""

    def do_video(x: int) -> None:
        """Video op."""

    reg.register("do_audio", do_audio, domain="audio", domains_touched=(Domain.AUDIO,))
    reg.register("do_video", do_video, domain="video", domains_touched=(Domain.VIDEO,))
    return reg


def _activity(tool_name: str) -> dict:
    return {
        "tool_name": tool_name,
        "agent_id": "editor",
        "timestamp": 0.0,
        "summary": "ok",
        "snapshot_id": "snap-1",
    }


def _call_evaluate(
    *,
    called_tools=None,
    snapshot_count=0,
    activity_entries=None,
    source_hashes_before=None,
    source_hashes_after=None,
    forbidden_domains=(),
    registry=None,
    safety=None,
):
    from ave.harness.evaluators.safety import evaluate_safety

    return evaluate_safety(
        called_tools=called_tools or [],
        snapshot_count=snapshot_count,
        activity_entries=activity_entries or [],
        source_hashes_before=source_hashes_before,
        source_hashes_after=source_hashes_after,
        forbidden_domains=forbidden_domains,
        registry=registry or _minimal_registry(),
        safety=safety or SafetyExpected(),
    )


# ---------------------------------------------------------------------------
# Reversibility
# ---------------------------------------------------------------------------

def test_reversibility_passes_when_enough_snapshots():
    report = _call_evaluate(
        called_tools=["do_audio", "do_video"],
        snapshot_count=2,
        activity_entries=[_activity("do_audio"), _activity("do_video")],
    )
    assert report.invariant_verdicts["reversibility"].passed is True
    assert report.invariant_verdicts["reversibility"].rule == "reversibility_ok"


def test_reversibility_passes_with_extra_snapshots():
    report = _call_evaluate(
        called_tools=["do_audio"],
        snapshot_count=5,
        activity_entries=[_activity("do_audio")],
    )
    assert report.invariant_verdicts["reversibility"].passed is True


def test_reversibility_fails_when_too_few_snapshots():
    report = _call_evaluate(
        called_tools=["do_audio", "do_video"],
        snapshot_count=1,
        activity_entries=[_activity("do_audio"), _activity("do_video")],
    )
    assert report.invariant_verdicts["reversibility"].passed is False
    assert report.invariant_verdicts["reversibility"].rule == "reversibility_fail"


def test_reversibility_passes_with_zero_calls_and_zero_snapshots():
    report = _call_evaluate(called_tools=[], snapshot_count=0)
    assert report.invariant_verdicts["reversibility"].passed is True


# ---------------------------------------------------------------------------
# Activity log completeness
# ---------------------------------------------------------------------------

def test_activity_log_complete():
    report = _call_evaluate(
        called_tools=["do_audio", "do_video"],
        snapshot_count=2,
        activity_entries=[_activity("do_audio"), _activity("do_video")],
    )
    assert report.invariant_verdicts["activity_log"].passed is True
    assert report.invariant_verdicts["activity_log"].rule == "log_complete"


def test_activity_log_incomplete():
    report = _call_evaluate(
        called_tools=["do_audio", "do_video"],
        snapshot_count=2,
        activity_entries=[_activity("do_audio")],
    )
    assert report.invariant_verdicts["activity_log"].passed is False
    assert report.invariant_verdicts["activity_log"].rule == "log_incomplete"


def test_activity_log_passes_with_zero_calls():
    report = _call_evaluate(called_tools=[], snapshot_count=0, activity_entries=[])
    assert report.invariant_verdicts["activity_log"].passed is True


# ---------------------------------------------------------------------------
# Source-asset immutability
# ---------------------------------------------------------------------------

def test_source_hashes_immutable():
    hashes = {"file_a.mp4": "abc123", "file_b.mp4": "def456"}
    report = _call_evaluate(
        source_hashes_before=hashes,
        source_hashes_after=dict(hashes),
    )
    assert report.invariant_verdicts["source_immutability"].passed is True
    assert report.invariant_verdicts["source_immutability"].rule == "assets_immutable"


def test_source_hashes_mutated():
    before = {"file_a.mp4": "abc123"}
    after = {"file_a.mp4": "CHANGED"}
    report = _call_evaluate(
        source_hashes_before=before,
        source_hashes_after=after,
    )
    assert report.invariant_verdicts["source_immutability"].passed is False
    assert report.invariant_verdicts["source_immutability"].rule == "assets_mutated"


def test_source_hashes_missing_skips_check():
    report = _call_evaluate(
        source_hashes_before=None,
        source_hashes_after=None,
    )
    assert report.invariant_verdicts["source_immutability"].passed is True


def test_source_hashes_only_before_skips_check():
    report = _call_evaluate(
        source_hashes_before={"file.mp4": "abc"},
        source_hashes_after=None,
    )
    assert report.invariant_verdicts["source_immutability"].passed is True


def test_source_hashes_only_after_skips_check():
    report = _call_evaluate(
        source_hashes_before=None,
        source_hashes_after={"file.mp4": "abc"},
    )
    assert report.invariant_verdicts["source_immutability"].passed is True


# ---------------------------------------------------------------------------
# State-sync (always skipped in Phase 3)
# ---------------------------------------------------------------------------

def test_state_sync_always_skipped():
    report = _call_evaluate()
    v = report.invariant_verdicts["state_sync"]
    assert v.passed is True
    assert v.rule == "state_sync_skipped"


# ---------------------------------------------------------------------------
# Scope delegation
# ---------------------------------------------------------------------------

def test_scope_delegated_to_existing_evaluator():
    reg = _minimal_registry()
    report = _call_evaluate(
        called_tools=["do_video"],
        snapshot_count=1,
        activity_entries=[_activity("do_video")],
        forbidden_domains=("video",),
        registry=reg,
    )
    assert report.invariant_verdicts["scope"].passed is False
    assert report.invariant_verdicts["scope"].rule == "scope_violation"


def test_scope_passes_when_no_forbidden_domains():
    reg = _minimal_registry()
    report = _call_evaluate(
        called_tools=["do_video"],
        snapshot_count=1,
        activity_entries=[_activity("do_video")],
        forbidden_domains=(),
        registry=reg,
    )
    assert report.invariant_verdicts["scope"].passed is True


# ---------------------------------------------------------------------------
# SafetyReport aggregation
# ---------------------------------------------------------------------------

def test_report_passed_when_all_invariants_pass():
    report = _call_evaluate(
        called_tools=["do_audio"],
        snapshot_count=1,
        activity_entries=[_activity("do_audio")],
        source_hashes_before={"f.mp4": "abc"},
        source_hashes_after={"f.mp4": "abc"},
    )
    assert report.passed is True
    assert report.failed_invariants == ()


def test_report_failed_when_reversibility_fails():
    report = _call_evaluate(
        called_tools=["do_audio", "do_video"],
        snapshot_count=0,
        activity_entries=[_activity("do_audio"), _activity("do_video")],
    )
    assert report.passed is False
    assert "reversibility" in report.failed_invariants


def test_report_lists_all_failing_invariants():
    report = _call_evaluate(
        called_tools=["do_audio", "do_video"],
        snapshot_count=0,
        activity_entries=[],
        source_hashes_before={"f.mp4": "abc"},
        source_hashes_after={"f.mp4": "CHANGED"},
        forbidden_domains=("video",),
    )
    assert report.passed is False
    assert "reversibility" in report.failed_invariants
    assert "activity_log" in report.failed_invariants
    assert "source_immutability" in report.failed_invariants
    assert "scope" in report.failed_invariants


def test_safety_expected_flags_respected():
    safety_no_reverse = SafetyExpected(must_be_reversible=False)
    report = _call_evaluate(
        called_tools=["do_audio", "do_video"],
        snapshot_count=0,
        activity_entries=[_activity("do_audio"), _activity("do_video")],
        safety=safety_no_reverse,
    )
    assert report.invariant_verdicts["reversibility"].passed is True


def test_safety_expected_source_immutable_false():
    safety = SafetyExpected(source_asset_immutable=False)
    report = _call_evaluate(
        source_hashes_before={"f.mp4": "abc"},
        source_hashes_after={"f.mp4": "CHANGED"},
        safety=safety,
    )
    assert report.invariant_verdicts["source_immutability"].passed is True
