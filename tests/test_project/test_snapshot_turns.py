"""Tests for turn-level checkpoints on SnapshotManager."""

from pathlib import Path

import pytest

from ave.project.snapshots import SnapshotManager


def _write_xges(path: Path, content: str) -> None:
    path.write_text(content)


def test_capture_turn_checkpoint_tags_snapshot(tmp_path: Path):
    xges = tmp_path / "project.xges"
    _write_xges(xges, "<xges>pre</xges>")

    mgr = SnapshotManager()
    snap = mgr.capture_turn_checkpoint(
        xges_path=xges,
        turn_id="turn-001",
        provisions=frozenset({"timeline_loaded"}),
    )
    assert snap.label == "turn_checkpoint:turn-001"
    assert snap.turn_id == "turn-001"


def test_rollback_to_turn_restores_xges(tmp_path: Path):
    xges = tmp_path / "project.xges"
    _write_xges(xges, "<xges>pre</xges>")

    mgr = SnapshotManager()
    mgr.capture_turn_checkpoint(
        xges_path=xges,
        turn_id="turn-001",
        provisions=frozenset({"timeline_loaded"}),
    )

    _write_xges(xges, "<xges>post</xges>")
    snap, provs = mgr.rollback_to_turn(turn_id="turn-001", xges_path=xges)
    assert xges.read_text() == "<xges>pre</xges>"
    assert "timeline_loaded" in provs
    assert snap.turn_id == "turn-001"


def test_redo_turn_restores_post_turn_state(tmp_path: Path):
    xges = tmp_path / "project.xges"
    _write_xges(xges, "<xges>pre</xges>")

    mgr = SnapshotManager()
    mgr.capture_turn_checkpoint(
        xges_path=xges,
        turn_id="turn-001",
        provisions=frozenset(),
    )
    _write_xges(xges, "<xges>post</xges>")
    mgr.capture_post_turn(
        xges_path=xges,
        turn_id="turn-001",
        provisions=frozenset({"edited"}),
    )

    # Roll back
    mgr.rollback_to_turn(turn_id="turn-001", xges_path=xges)
    assert xges.read_text() == "<xges>pre</xges>"

    # Redo
    snap, provs = mgr.redo_turn(turn_id="turn-001", xges_path=xges)
    assert xges.read_text() == "<xges>post</xges>"
    assert "edited" in provs


def test_rollback_to_unknown_turn_raises(tmp_path: Path):
    mgr = SnapshotManager()
    with pytest.raises(KeyError, match="turn-nope"):
        mgr.rollback_to_turn(turn_id="turn-nope", xges_path=tmp_path / "p.xges")
