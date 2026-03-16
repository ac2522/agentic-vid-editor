"""Segment cache manager for timeline preview rendering.

Tracks segment render state (dirty -> rendering -> clean) and handles
invalidation when the timeline is edited. Pure logic — no GES dependency.
All times in nanoseconds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class SegmentState(Enum):
    DIRTY = "dirty"
    RENDERING = "rendering"
    CLEAN = "clean"


class CacheError(Exception):
    """Raised when cache operations fail."""


@dataclass
class CachedSegment:
    index: int
    start_ns: int
    end_ns: int
    state: SegmentState = SegmentState.DIRTY
    file_path: Path | None = None


class SegmentCache:
    """Manages cached timeline segments.

    Tracks segment state: dirty -> rendering -> clean.
    Handles invalidation on timeline edits.
    Supports viewport-priority dirty segment ordering.
    """

    def __init__(self, cache_dir: Path, timeline_id: str = "default") -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._timeline_id = timeline_id
        self._segments: dict[int, CachedSegment] = {}

    def register_segments(self, boundaries: list[tuple[int, int, int]]) -> None:
        """Register segments from (index, start_ns, end_ns) tuples. All start DIRTY."""
        for index, start_ns, end_ns in boundaries:
            self._segments[index] = CachedSegment(
                index=index, start_ns=start_ns, end_ns=end_ns
            )

    def _get_segment(self, index: int) -> CachedSegment:
        if index not in self._segments:
            raise CacheError(f"Unknown segment index: {index}")
        return self._segments[index]

    def get_state(self, index: int) -> SegmentState:
        return self._get_segment(index).state

    def mark_rendering(self, index: int) -> None:
        seg = self._get_segment(index)
        if seg.state != SegmentState.DIRTY:
            raise CacheError(
                f"Cannot mark segment {index} as RENDERING: "
                f"current state is {seg.state.value}, expected DIRTY"
            )
        seg.state = SegmentState.RENDERING

    def mark_clean(self, index: int, file_path: Path) -> None:
        file_path = Path(file_path)
        if not file_path.exists():
            raise CacheError(f"File does not exist: {file_path}")
        seg = self._get_segment(index)
        if seg.state != SegmentState.RENDERING:
            raise CacheError(
                f"Cannot mark segment {index} as CLEAN: "
                f"current state is {seg.state.value}, expected RENDERING"
            )
        seg.state = SegmentState.CLEAN
        seg.file_path = file_path

    def invalidate_segment(self, index: int) -> None:
        """Mark segment DIRTY, remove cached file if exists."""
        seg = self._get_segment(index)
        if seg.file_path is not None and seg.file_path.exists():
            seg.file_path.unlink()
        seg.state = SegmentState.DIRTY
        seg.file_path = None

    def invalidate_range(self, start_ns: int, end_ns: int) -> None:
        """Invalidate all segments overlapping the time range.

        Overlap: seg.start_ns < range_end_ns and seg.end_ns > range_start_ns
        """
        for seg in self._segments.values():
            if seg.start_ns < end_ns and seg.end_ns > start_ns:
                if seg.file_path is not None and seg.file_path.exists():
                    seg.file_path.unlink()
                seg.state = SegmentState.DIRTY
                seg.file_path = None

    def invalidate_all(self) -> None:
        for seg in self._segments.values():
            if seg.file_path is not None and seg.file_path.exists():
                seg.file_path.unlink()
            seg.state = SegmentState.DIRTY
            seg.file_path = None

    def get_dirty_segments(
        self,
        viewport_start_ns: int | None = None,
        viewport_end_ns: int | None = None,
    ) -> list[CachedSegment]:
        """Get DIRTY segments, optionally prioritizing those in viewport."""
        dirty = [s for s in self._segments.values() if s.state == SegmentState.DIRTY]

        if viewport_start_ns is not None and viewport_end_ns is not None:
            in_viewport = []
            outside = []
            for seg in dirty:
                if seg.start_ns < viewport_end_ns and seg.end_ns > viewport_start_ns:
                    in_viewport.append(seg)
                else:
                    outside.append(seg)
            in_viewport.sort(key=lambda s: s.index)
            outside.sort(key=lambda s: s.index)
            return in_viewport + outside
        else:
            dirty.sort(key=lambda s: s.index)
            return dirty

    def get_segment_path(self, index: int) -> Path:
        """Get file path for a CLEAN segment. Raises CacheError if not clean."""
        seg = self._get_segment(index)
        if seg.state != SegmentState.CLEAN:
            raise CacheError(f"Segment {index} is not CLEAN (state={seg.state.value})")
        assert seg.file_path is not None
        return seg.file_path

    def segment_count(self) -> dict[str, int]:
        """Return {"total": N, "dirty": N, "rendering": N, "clean": N}."""
        counts = {"total": len(self._segments), "dirty": 0, "rendering": 0, "clean": 0}
        for seg in self._segments.values():
            counts[seg.state.value] += 1
        return counts

    def save_state(self, path: Path) -> None:
        """Persist cache state to JSON."""
        data = {
            "timeline_id": self._timeline_id,
            "segments": [
                {
                    "index": seg.index,
                    "start_ns": seg.start_ns,
                    "end_ns": seg.end_ns,
                    "state": seg.state.value,
                    "file_path": str(seg.file_path) if seg.file_path else None,
                }
                for seg in self._segments.values()
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load_state(cls, path: Path, cache_dir: Path) -> SegmentCache:
        """Load cache state from JSON."""
        data = json.loads(Path(path).read_text())
        cache = cls(cache_dir, timeline_id=data.get("timeline_id", "default"))
        for seg_data in data["segments"]:
            file_path = Path(seg_data["file_path"]) if seg_data["file_path"] else None
            cache._segments[seg_data["index"]] = CachedSegment(
                index=seg_data["index"],
                start_ns=seg_data["start_ns"],
                end_ns=seg_data["end_ns"],
                state=SegmentState(seg_data["state"]),
                file_path=file_path,
            )
        return cache
