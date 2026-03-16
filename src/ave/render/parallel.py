"""Parallel segment rendering scheduler.

Pure queue data structure — does NOT execute renders.
Manages job state transitions and priority ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RenderJob:
    """A single segment render job."""

    segment_id: str
    start_ns: int
    stop_ns: int
    priority: int = 0  # Lower = higher priority
    status: str = "pending"  # pending | rendering | complete | failed
    output_path: Path | None = None
    error: str | None = None


class RenderScheduler:
    """Priority-ordered parallel segment rendering scheduler.

    Pure queue data structure — does NOT execute renders.
    Manages job state transitions and priority ordering.

    Priority: segments near playhead first, then outward.
    Concurrency: next_batch() returns up to max_workers jobs.
    """

    def __init__(self, max_workers: int = 2) -> None:
        self._queue: list[RenderJob] = []
        self._active: dict[str, RenderJob] = {}
        self._completed: dict[str, RenderJob] = {}
        self._failed: dict[str, RenderJob] = {}
        self._max_workers = max_workers
        self._known_ids: set[str] = set()

    def enqueue(self, jobs: list[RenderJob]) -> int:
        """Add jobs to queue. Returns count added. Skips duplicates."""
        added = 0
        for job in jobs:
            if job.segment_id in self._known_ids:
                continue
            self._known_ids.add(job.segment_id)
            self._queue.append(job)
            added += 1
        # Keep queue sorted by priority (lowest first)
        self._queue.sort(key=lambda j: j.priority)
        return added

    def next_batch(self) -> list[RenderJob]:
        """Get next batch of up to max_workers jobs.

        Moves them from queue to active. Returns empty if at capacity.
        """
        available_slots = self._max_workers - len(self._active)
        if available_slots <= 0:
            return []

        batch: list[RenderJob] = []
        for _ in range(min(available_slots, len(self._queue))):
            job = self._queue.pop(0)
            job.status = "rendering"
            self._active[job.segment_id] = job
            batch.append(job)
        return batch

    def mark_complete(self, segment_id: str, output_path: Path) -> None:
        """Mark an active job as complete."""
        if segment_id not in self._active:
            raise KeyError(f"No active job with segment_id={segment_id!r}")
        job = self._active.pop(segment_id)
        job.status = "complete"
        job.output_path = output_path
        self._completed[segment_id] = job

    def mark_failed(self, segment_id: str, error: str) -> None:
        """Mark an active job as failed."""
        if segment_id not in self._active:
            raise KeyError(f"No active job with segment_id={segment_id!r}")
        job = self._active.pop(segment_id)
        job.status = "failed"
        job.error = error
        self._failed[segment_id] = job

    def pending_count(self) -> int:
        return len(self._queue)

    def active_count(self) -> int:
        return len(self._active)

    def completed_count(self) -> int:
        return len(self._completed)

    def failed_count(self) -> int:
        return len(self._failed)

    @staticmethod
    def prioritize_by_playhead(
        jobs: list[RenderJob], playhead_ns: int
    ) -> list[RenderJob]:
        """Sort jobs by distance from playhead (closest first).

        Distance is measured from playhead to nearest edge of the segment.
        """

        def _distance(job: RenderJob) -> int:
            if job.start_ns <= playhead_ns <= job.stop_ns:
                return 0
            return min(abs(playhead_ns - job.start_ns), abs(playhead_ns - job.stop_ns))

        return sorted(jobs, key=_distance)
