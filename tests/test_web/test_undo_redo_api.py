"""Tests for pure undo/redo handlers and the rollback WS event."""

from pathlib import Path

from ave.agent.activity import ActivityLog
from ave.agent.session import EditingSession
from ave.project.snapshots import SnapshotManager
from ave.web.api import redo_response, undo_response
from ave.web.chat import format_timeline_rollback


def _setup_session_with_completed_turn(tmp_path: Path) -> tuple[EditingSession, Path]:
    project = tmp_path / "proj"
    project.mkdir()
    xges = project / "project.xges"
    xges.write_text("<xges>initial</xges>")

    session = EditingSession(
        snapshot_manager=SnapshotManager(),
        activity_log=ActivityLog(persist_path=tmp_path / "log.jsonl"),
        project_root=project,
    )
    session.load_project(xges)
    session.begin_turn("turn-001")
    xges.write_text("<xges>after</xges>")
    session.end_turn("turn-001")
    return session, xges


def test_undo_response_ok(tmp_path: Path):
    session, xges = _setup_session_with_completed_turn(tmp_path)
    status, body = undo_response(session, "turn-001")
    assert status == 200
    assert body == {"ok": True, "turn_id": "turn-001", "direction": "undo"}
    assert xges.read_text() == "<xges>initial</xges>"


def test_redo_response_ok(tmp_path: Path):
    session, xges = _setup_session_with_completed_turn(tmp_path)
    undo_response(session, "turn-001")
    status, body = redo_response(session, "turn-001")
    assert status == 200
    assert body == {"ok": True, "turn_id": "turn-001", "direction": "redo"}
    assert xges.read_text() == "<xges>after</xges>"


def test_undo_response_missing_turn_id(tmp_path: Path):
    session, _ = _setup_session_with_completed_turn(tmp_path)
    status, body = undo_response(session, "")
    assert status == 400
    assert body["ok"] is False


def test_undo_response_unknown_turn(tmp_path: Path):
    session, _ = _setup_session_with_completed_turn(tmp_path)
    status, body = undo_response(session, "ghost")
    assert status == 404
    assert body["ok"] is False


def test_redo_response_unknown_turn(tmp_path: Path):
    session, _ = _setup_session_with_completed_turn(tmp_path)
    status, body = redo_response(session, "ghost")
    assert status == 404


def test_format_timeline_rollback_event():
    evt = format_timeline_rollback(turn_id="turn-001", direction="undo")
    assert evt == {
        "type": "timeline_rollback",
        "turn_id": "turn-001",
        "direction": "undo",
    }
