"""Tests for XGES snapshot manager — undo/rollback for timeline state."""

from __future__ import annotations

from pathlib import Path

import pytest

from ave.project.snapshots import Snapshot, SnapshotManager, SnapshotSummary


class TestSnapshotCreation:
    """Test Snapshot data model."""

    def test_create_snapshot(self):
        snap = Snapshot(
            snapshot_id="abc-123",
            timestamp=1000.0,
            label="before trim",
            xges_content="<ges>...</ges>",
            provisions=frozenset({"timeline_loaded", "clip_exists"}),
            tool_name="trim",
        )
        assert snap.snapshot_id == "abc-123"
        assert snap.label == "before trim"
        assert snap.xges_content == "<ges>...</ges>"
        assert snap.provisions == frozenset({"timeline_loaded", "clip_exists"})
        assert snap.tool_name == "trim"

    def test_snapshot_immutable(self):
        snap = Snapshot(
            snapshot_id="abc-123",
            timestamp=1000.0,
            label="before trim",
            xges_content="<ges/>",
            provisions=frozenset(),
            tool_name=None,
        )
        with pytest.raises(AttributeError):
            snap.label = "modified"


class TestSnapshotSummary:
    """Test compact snapshot summary for LLM consumption."""

    def test_create_summary(self):
        summary = SnapshotSummary(
            snapshot_id="abc-123",
            label="before trim",
            tool_name="trim",
            timestamp=1000.0,
        )
        assert summary.snapshot_id == "abc-123"
        assert summary.label == "before trim"


class TestSnapshotManager:
    """Test snapshot capture, restore, and eviction."""

    @pytest.fixture()
    def xges_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "project.xges"
        p.write_text("<ges version='1.0'><timeline/></ges>")
        return p

    @pytest.fixture()
    def mgr(self) -> SnapshotManager:
        return SnapshotManager(max_snapshots=5)

    def test_capture_returns_snapshot(self, mgr, xges_file):
        snap = mgr.capture(
            xges_file,
            label="test capture",
            provisions=frozenset({"timeline_loaded"}),
            tool_name="trim",
        )
        assert isinstance(snap, Snapshot)
        assert snap.label == "test capture"
        assert snap.xges_content == "<ges version='1.0'><timeline/></ges>"
        assert snap.provisions == frozenset({"timeline_loaded"})

    def test_capture_reads_file_content(self, mgr, xges_file):
        xges_file.write_text("<ges><timeline><layer/></timeline></ges>")
        snap = mgr.capture(xges_file, "after edit", frozenset())
        assert "<layer/>" in snap.xges_content

    def test_list_snapshots_returns_summaries(self, mgr, xges_file):
        mgr.capture(xges_file, "snap 1", frozenset())
        mgr.capture(xges_file, "snap 2", frozenset())
        summaries = mgr.list_snapshots()
        assert len(summaries) == 2
        assert all(isinstance(s, SnapshotSummary) for s in summaries)
        assert summaries[0].label == "snap 1"
        assert summaries[1].label == "snap 2"

    def test_restore_by_id(self, mgr, xges_file):
        original_content = xges_file.read_text()
        snap = mgr.capture(xges_file, "before edit", frozenset({"timeline_loaded"}))

        # Simulate an edit
        xges_file.write_text("<ges><timeline><modified/></timeline></ges>")
        assert "modified" in xges_file.read_text()

        # Restore
        restored, provisions = mgr.restore(snap.snapshot_id, xges_file)
        assert xges_file.read_text() == original_content
        assert provisions == frozenset({"timeline_loaded"})
        assert restored.snapshot_id == snap.snapshot_id

    def test_restore_latest(self, mgr, xges_file):
        mgr.capture(xges_file, "snap 1", frozenset({"a"}))
        xges_file.write_text("<ges>v2</ges>")
        mgr.capture(xges_file, "snap 2", frozenset({"a", "b"}))
        xges_file.write_text("<ges>v3</ges>")

        result = mgr.restore_latest(xges_file)
        assert result is not None
        restored, provisions = result
        assert restored.label == "snap 2"
        assert xges_file.read_text() == "<ges>v2</ges>"
        assert provisions == frozenset({"a", "b"})

    def test_restore_latest_empty(self, mgr, xges_file):
        result = mgr.restore_latest(xges_file)
        assert result is None

    def test_restore_unknown_id_raises(self, mgr, xges_file):
        with pytest.raises(KeyError, match="not found"):
            mgr.restore("nonexistent-id", xges_file)

    def test_eviction_at_max_snapshots(self, mgr, xges_file):
        """Oldest snapshot evicted when max exceeded."""
        # mgr has max_snapshots=5
        ids = []
        for i in range(6):
            xges_file.write_text(f"<ges>v{i}</ges>")
            snap = mgr.capture(xges_file, f"snap {i}", frozenset())
            ids.append(snap.snapshot_id)

        summaries = mgr.list_snapshots()
        assert len(summaries) == 5
        # Oldest (snap 0) should be evicted
        summary_ids = {s.snapshot_id for s in summaries}
        assert ids[0] not in summary_ids
        assert ids[5] in summary_ids

    def test_eviction_order_preserves_newest(self, mgr, xges_file):
        """After eviction, the newest snapshots are preserved."""
        for i in range(10):
            xges_file.write_text(f"<ges>v{i}</ges>")
            mgr.capture(xges_file, f"snap {i}", frozenset())

        summaries = mgr.list_snapshots()
        assert len(summaries) == 5
        labels = [s.label for s in summaries]
        assert labels == [f"snap {i}" for i in range(5, 10)]

    def test_clear_removes_all(self, mgr, xges_file):
        mgr.capture(xges_file, "snap 1", frozenset())
        mgr.capture(xges_file, "snap 2", frozenset())
        count = mgr.clear()
        assert count == 2
        assert mgr.list_snapshots() == []

    def test_clear_empty_returns_zero(self, mgr):
        assert mgr.clear() == 0


class TestSnapshotPersistence:
    """Test optional disk persistence for crash recovery."""

    @pytest.fixture()
    def xges_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "project.xges"
        p.write_text("<ges><timeline/></ges>")
        return p

    def test_persist_to_dir_writes_files(self, tmp_path, xges_file):
        persist_dir = tmp_path / "snapshots"
        persist_dir.mkdir()
        mgr = SnapshotManager(max_snapshots=5, persist_dir=persist_dir)

        snap = mgr.capture(xges_file, "persisted", frozenset())
        expected_file = persist_dir / f"{snap.snapshot_id}.xges"
        assert expected_file.exists()
        assert expected_file.read_text() == "<ges><timeline/></ges>"

    def test_persist_dir_none_skips_disk(self, tmp_path, xges_file):
        mgr = SnapshotManager(max_snapshots=5, persist_dir=None)
        snap = mgr.capture(xges_file, "in-memory", frozenset())
        # No crash, no disk files
        assert snap.snapshot_id is not None

    def test_eviction_cleans_persisted_file(self, tmp_path, xges_file):
        persist_dir = tmp_path / "snapshots"
        persist_dir.mkdir()
        mgr = SnapshotManager(max_snapshots=2, persist_dir=persist_dir)

        snap1 = mgr.capture(xges_file, "snap 1", frozenset())
        mgr.capture(xges_file, "snap 2", frozenset())
        mgr.capture(xges_file, "snap 3", frozenset())

        # snap1 should be evicted and its file cleaned up
        assert not (persist_dir / f"{snap1.snapshot_id}.xges").exists()
        assert len(list(persist_dir.glob("*.xges"))) == 2


class TestSnapshotProvisionRestoration:
    """Critical: provisions must be restored alongside XGES content."""

    @pytest.fixture()
    def xges_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "project.xges"
        p.write_text("<ges>original</ges>")
        return p

    def test_restore_returns_original_provisions(self, xges_file):
        mgr = SnapshotManager(max_snapshots=5)
        original_provisions = frozenset({"timeline_loaded", "clip_exists"})
        snap = mgr.capture(xges_file, "before edit", original_provisions, tool_name="trim")

        # Simulate state mutation
        xges_file.write_text("<ges>modified</ges>")

        _, restored_provisions = mgr.restore(snap.snapshot_id, xges_file)
        assert restored_provisions == original_provisions

    def test_restore_latest_returns_provisions(self, xges_file):
        mgr = SnapshotManager(max_snapshots=5)
        mgr.capture(xges_file, "v1", frozenset({"a"}))
        xges_file.write_text("<ges>v2</ges>")
        mgr.capture(xges_file, "v2", frozenset({"a", "b"}))
        xges_file.write_text("<ges>v3</ges>")

        _, provisions = mgr.restore_latest(xges_file)
        assert provisions == frozenset({"a", "b"})
