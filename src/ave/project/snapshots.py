"""XGES snapshot manager — undo/rollback for timeline state.

Captures full XGES content before each tool call, enabling rollback
on failure. Snapshots include SessionState provisions so that both
timeline and session state are restored together.
"""

from __future__ import annotations

import uuid
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Snapshot:
    """Immutable record of timeline state at a point in time."""

    snapshot_id: str
    timestamp: float  # wall-clock time.time(), NOT GES nanoseconds
    label: str
    xges_content: str
    provisions: frozenset[str]
    tool_name: str | None = None
    turn_id: str | None = None


@dataclass(frozen=True)
class SnapshotSummary:
    """Compact snapshot summary for LLM consumption (~30 tokens)."""

    snapshot_id: str
    label: str
    tool_name: str | None
    timestamp: float


class SnapshotManager:
    """Manages XGES snapshots for undo/rollback.

    - Captures full XGES content + session provisions before each edit
    - Restores both XGES file and provisions on rollback
    - Evicts oldest snapshots when max_snapshots exceeded
    - Optional disk persistence via persist_dir for crash recovery
    """

    def __init__(
        self,
        max_snapshots: int = 50,
        persist_dir: Path | None = None,
    ) -> None:
        self._snapshots: list[Snapshot] = []
        self._max_snapshots = max_snapshots
        self._persist_dir = persist_dir

    def capture(
        self,
        xges_path: Path,
        label: str,
        provisions: frozenset[str],
        tool_name: str | None = None,
    ) -> Snapshot:
        """Capture current XGES state as a snapshot.

        Reads the XGES file content and stores it with the current provisions.
        If persist_dir is set, also writes to disk.
        Evicts oldest snapshot if at capacity.
        """
        # Evict oldest if at capacity (evict BEFORE adding)
        while len(self._snapshots) >= self._max_snapshots:
            evicted = self._snapshots.pop(0)
            self._cleanup_persisted(evicted.snapshot_id)

        content = Path(xges_path).read_text()
        snap = Snapshot(
            snapshot_id=str(uuid.uuid4()),
            timestamp=time.time(),
            label=label,
            xges_content=content,
            provisions=frozenset(provisions),
            tool_name=tool_name,
        )
        self._snapshots.append(snap)

        if self._persist_dir is not None:
            persist_path = self._persist_dir / f"{snap.snapshot_id}.xges"
            persist_path.write_text(content)

        return snap

    def restore(
        self,
        snapshot_id: str,
        xges_path: Path,
    ) -> tuple[Snapshot, frozenset[str]]:
        """Restore XGES content from a snapshot.

        Writes the snapshot's XGES content back to the file.
        Returns (snapshot, provisions) — caller must restore SessionState.

        Raises KeyError if snapshot_id not found.
        """
        snap = self._find(snapshot_id)
        Path(xges_path).write_text(snap.xges_content)
        return snap, snap.provisions

    def restore_latest(
        self,
        xges_path: Path,
    ) -> tuple[Snapshot, frozenset[str]] | None:
        """Restore the most recent snapshot.

        Returns None if no snapshots exist.
        """
        if not self._snapshots:
            return None
        snap = self._snapshots[-1]
        Path(xges_path).write_text(snap.xges_content)
        return snap, snap.provisions

    def list_snapshots(self) -> list[SnapshotSummary]:
        """Return compact summaries of all snapshots."""
        return [
            SnapshotSummary(
                snapshot_id=s.snapshot_id,
                label=s.label,
                tool_name=s.tool_name,
                timestamp=s.timestamp,
            )
            for s in self._snapshots
        ]

    def clear(self) -> int:
        """Remove all snapshots. Returns count cleared."""
        count = len(self._snapshots)
        if self._persist_dir is not None:
            for s in self._snapshots:
                self._cleanup_persisted(s.snapshot_id)
        self._snapshots.clear()
        return count

    def _find(self, snapshot_id: str) -> Snapshot:
        """Find snapshot by ID. Raises KeyError if not found."""
        for s in self._snapshots:
            if s.snapshot_id == snapshot_id:
                return s
        raise KeyError(f"Snapshot not found: {snapshot_id}")

    def capture_turn_checkpoint(
        self,
        xges_path: Path,
        turn_id: str,
        provisions: frozenset[str],
    ) -> Snapshot:
        """Capture a pre-turn checkpoint (state before the user's turn runs)."""
        while len(self._snapshots) >= self._max_snapshots:
            evicted = self._snapshots.pop(0)
            self._cleanup_persisted(evicted.snapshot_id)

        content = Path(xges_path).read_text()
        snap = Snapshot(
            snapshot_id=str(uuid.uuid4()),
            timestamp=time.time(),
            label=f"turn_checkpoint:{turn_id}",
            xges_content=content,
            provisions=frozenset(provisions),
            tool_name=None,
            turn_id=turn_id,
        )
        self._snapshots.append(snap)
        if self._persist_dir is not None:
            (self._persist_dir / f"{snap.snapshot_id}.xges").write_text(content)
        return snap

    def capture_post_turn(
        self,
        xges_path: Path,
        turn_id: str,
        provisions: frozenset[str],
    ) -> Snapshot:
        """Capture the post-turn state (for redo)."""
        while len(self._snapshots) >= self._max_snapshots:
            evicted = self._snapshots.pop(0)
            self._cleanup_persisted(evicted.snapshot_id)

        content = Path(xges_path).read_text()
        snap = Snapshot(
            snapshot_id=str(uuid.uuid4()),
            timestamp=time.time(),
            label=f"post_turn:{turn_id}",
            xges_content=content,
            provisions=frozenset(provisions),
            tool_name=None,
            turn_id=turn_id,
        )
        self._snapshots.append(snap)
        if self._persist_dir is not None:
            (self._persist_dir / f"{snap.snapshot_id}.xges").write_text(content)
        return snap

    def rollback_to_turn(
        self,
        turn_id: str,
        xges_path: Path,
    ) -> tuple[Snapshot, frozenset[str]]:
        """Restore the pre-turn checkpoint (undo)."""
        for s in self._snapshots:
            if s.turn_id == turn_id and s.label.startswith("turn_checkpoint:"):
                Path(xges_path).write_text(s.xges_content)
                return s, s.provisions
        raise KeyError(f"No turn checkpoint for {turn_id!r}")

    def redo_turn(
        self,
        turn_id: str,
        xges_path: Path,
    ) -> tuple[Snapshot, frozenset[str]]:
        """Restore the post-turn checkpoint (redo)."""
        for s in self._snapshots:
            if s.turn_id == turn_id and s.label.startswith("post_turn:"):
                Path(xges_path).write_text(s.xges_content)
                return s, s.provisions
        raise KeyError(f"No post-turn checkpoint for {turn_id!r}")

    def _cleanup_persisted(self, snapshot_id: str) -> None:
        """Remove persisted file for a snapshot, if it exists."""
        if self._persist_dir is None:
            return
        path = self._persist_dir / f"{snapshot_id}.xges"
        if path.exists():
            path.unlink()
