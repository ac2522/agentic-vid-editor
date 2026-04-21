"""Tests that ChatSession prepends a state summary to each user turn."""

from pathlib import Path

from ave.agent.activity import ActivityLog
from ave.agent.session import EditingSession
from ave.project.snapshots import SnapshotManager
from ave.web.chat import ChatSession


class _MinimalOrchestrator:
    """Orchestrator stub exposing just the `session` attribute ChatSession needs."""

    def __init__(self, session: EditingSession):
        self.session = session


class _FakeTimeline:
    pass


def _prep(tmp_path: Path) -> tuple[ChatSession, ActivityLog]:
    project = tmp_path / "proj"
    project.mkdir()
    xges = project / "project.xges"
    xges.write_text("<xges></xges>")

    log = ActivityLog(persist_path=tmp_path / "log.jsonl")
    ave_session = EditingSession(
        snapshot_manager=SnapshotManager(),
        activity_log=log,
        project_root=project,
    )
    ave_session.load_project(xges)

    orch = _MinimalOrchestrator(ave_session)
    chat = ChatSession(orch, _FakeTimeline())
    return chat, log


def test_prepare_user_content_prefixes_state_summary(tmp_path: Path):
    chat, log = _prep(tmp_path)
    log.append(agent_id="editor", tool_name="trim_clip", summary="trimmed intro", snapshot_id="s1")

    prepared = chat._prepare_user_content("please cut the end off")

    assert prepared.startswith("STATE SUMMARY")
    assert "trim_clip" in prepared
    assert "editor" in prepared
    # Original text preserved at the end
    assert prepared.endswith("please cut the end off")


def test_prepare_user_content_advances_last_summary_timestamp(tmp_path: Path):
    chat, log = _prep(tmp_path)

    before = chat._last_summary_timestamp
    chat._prepare_user_content("first")
    after = chat._last_summary_timestamp
    assert after > before


def test_prepare_user_content_without_session_returns_plain_text(tmp_path: Path):
    """If the orchestrator has no session (or activity log), the text passes through."""

    class _NoSession:
        pass

    chat = ChatSession(_NoSession(), _FakeTimeline())
    prepared = chat._prepare_user_content("just some text")
    assert prepared == "just some text"


def test_prepare_user_content_second_turn_only_shows_new_activity(tmp_path: Path):
    chat, log = _prep(tmp_path)
    log.append(agent_id="editor", tool_name="trim_clip", summary="t1", snapshot_id="s1")
    chat._prepare_user_content("turn 1")

    log.append(agent_id="colorist", tool_name="apply_lut", summary="t2", snapshot_id="s2")
    prepared = chat._prepare_user_content("turn 2")

    # trim_clip was in turn 1 and should NOT appear in turn 2
    assert "trim_clip" not in prepared
    # apply_lut IS the new activity
    assert "apply_lut" in prepared
