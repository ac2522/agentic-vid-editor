"""Turn lifecycle and undo/redo tests for EditingSession."""

from pathlib import Path

import pytest

from ave.agent.session import EditingSession, SessionError
from ave.project.snapshots import SnapshotManager


def _setup_session(tmp_path: Path) -> tuple[EditingSession, Path]:
    project = tmp_path / "proj"
    project.mkdir()
    xges = project / "project.xges"
    xges.write_text("<xges>initial</xges>")

    mgr = SnapshotManager()
    session = EditingSession(snapshot_manager=mgr, project_root=project)
    session.load_project(xges)
    return session, xges


def test_begin_turn_captures_checkpoint(tmp_path: Path):
    session, xges = _setup_session(tmp_path)
    session.begin_turn("turn-001")
    snaps = session.snapshot_manager.list_snapshots()
    assert any(s.label == "turn_checkpoint:turn-001" for s in snaps)


def test_end_turn_captures_post_turn(tmp_path: Path):
    session, xges = _setup_session(tmp_path)
    session.begin_turn("turn-001")
    xges.write_text("<xges>after-turn</xges>")
    session.end_turn("turn-001")
    snaps = session.snapshot_manager.list_snapshots()
    assert any(s.label == "post_turn:turn-001" for s in snaps)


def test_undo_turn_restores_pre_turn_state(tmp_path: Path):
    session, xges = _setup_session(tmp_path)
    session.begin_turn("turn-001")
    xges.write_text("<xges>after-turn</xges>")
    session.end_turn("turn-001")

    session.undo_turn("turn-001")
    assert xges.read_text() == "<xges>initial</xges>"


def test_redo_turn_restores_post_turn_state(tmp_path: Path):
    session, xges = _setup_session(tmp_path)
    session.begin_turn("turn-001")
    xges.write_text("<xges>after-turn</xges>")
    session.end_turn("turn-001")
    session.undo_turn("turn-001")

    session.redo_turn("turn-001")
    assert xges.read_text() == "<xges>after-turn</xges>"


def test_undo_without_project_raises(tmp_path: Path):
    session = EditingSession(snapshot_manager=SnapshotManager())
    with pytest.raises(SessionError, match="project"):
        session.undo_turn("turn-001")
