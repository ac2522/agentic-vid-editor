"""Tests for the state-sync summary builder."""

from pathlib import Path

from ave.agent.activity import ActivityLog
from ave.agent.session import EditingSession
from ave.agent.state_sync import StateSummary, build_state_summary
from ave.project.snapshots import SnapshotManager


def _setup_session(tmp_path: Path) -> tuple[EditingSession, ActivityLog, Path]:
    project = tmp_path / "proj"
    project.mkdir()
    xges = project / "project.xges"
    xges.write_text("<xges></xges>")
    log = ActivityLog(persist_path=tmp_path / "activity-log.jsonl")
    session = EditingSession(
        snapshot_manager=SnapshotManager(),
        activity_log=log,
        project_root=project,
    )
    session.load_project(xges)
    return session, log, xges


def test_summary_includes_timeline_loaded_state(tmp_path: Path):
    session, log, xges = _setup_session(tmp_path)
    summary = build_state_summary(session=session, activity_log=log, since_timestamp=0.0)
    assert isinstance(summary, StateSummary)
    assert "timeline_loaded" in summary.state_provisions


def test_summary_includes_recent_activity(tmp_path: Path):
    session, log, xges = _setup_session(tmp_path)
    log.append(agent_id="editor", tool_name="trim_clip", summary="trim a", snapshot_id="s1")
    log.append(agent_id="colorist", tool_name="apply_lut", summary="apply lut", snapshot_id="s2")

    summary = build_state_summary(session=session, activity_log=log, since_timestamp=0.0)
    assert len(summary.recent_entries) == 2
    assert summary.recent_entries[0].tool_name == "trim_clip"


def test_summary_respects_since_timestamp(tmp_path: Path):
    session, log, xges = _setup_session(tmp_path)
    log.append(agent_id="editor", tool_name="trim_clip", summary="old", snapshot_id="s1")
    cutoff = log.entries()[-1].timestamp
    log.append(agent_id="editor", tool_name="split_clip", summary="new", snapshot_id="s2")

    summary = build_state_summary(session=session, activity_log=log, since_timestamp=cutoff)
    assert len(summary.recent_entries) == 1
    assert summary.recent_entries[0].tool_name == "split_clip"


def test_render_produces_compact_text_block(tmp_path: Path):
    session, log, xges = _setup_session(tmp_path)
    log.append(agent_id="editor", tool_name="trim_clip", summary="trim a", snapshot_id="s1")

    summary = build_state_summary(session=session, activity_log=log, since_timestamp=0.0)
    text = summary.render()
    assert "STATE SUMMARY" in text
    assert "trim_clip" in text
    assert "editor" in text
    # Compact — target <= 300 tokens (~1200 chars as a rough bound)
    assert len(text) < 2000
