"""Tests for parallel segment rendering scheduler."""

from pathlib import Path

import pytest

from ave.render.parallel import RenderJob, RenderScheduler


class TestRenderJob:
    def test_defaults(self):
        job = RenderJob(segment_id="s1", start_ns=0, stop_ns=1_000_000_000)
        assert job.segment_id == "s1"
        assert job.start_ns == 0
        assert job.stop_ns == 1_000_000_000
        assert job.priority == 0
        assert job.status == "pending"
        assert job.output_path is None
        assert job.error is None


class TestRenderScheduler:
    def _make_jobs(self, n: int, priority_start: int = 0) -> list[RenderJob]:
        return [
            RenderJob(
                segment_id=f"seg-{i}",
                start_ns=i * 1_000_000_000,
                stop_ns=(i + 1) * 1_000_000_000,
                priority=priority_start + i,
            )
            for i in range(n)
        ]

    def test_enqueue_adds_jobs(self):
        sched = RenderScheduler(max_workers=2)
        jobs = self._make_jobs(3)
        count = sched.enqueue(jobs)
        assert count == 3
        assert sched.pending_count() == 3

    def test_enqueue_skips_duplicates(self):
        sched = RenderScheduler(max_workers=2)
        jobs = self._make_jobs(2)
        sched.enqueue(jobs)
        # Re-enqueue same ids
        count = sched.enqueue(jobs)
        assert count == 0
        assert sched.pending_count() == 2

    def test_next_batch_returns_up_to_max_workers(self):
        sched = RenderScheduler(max_workers=2)
        sched.enqueue(self._make_jobs(5))
        batch = sched.next_batch()
        assert len(batch) == 2

    def test_next_batch_returns_empty_when_at_capacity(self):
        sched = RenderScheduler(max_workers=2)
        sched.enqueue(self._make_jobs(5))
        sched.next_batch()  # fills active slots
        batch = sched.next_batch()
        assert batch == []

    def test_next_batch_moves_jobs_to_active(self):
        sched = RenderScheduler(max_workers=2)
        sched.enqueue(self._make_jobs(3))
        batch = sched.next_batch()
        assert sched.active_count() == 2
        assert sched.pending_count() == 1
        for job in batch:
            assert job.status == "rendering"

    def test_mark_complete(self):
        sched = RenderScheduler(max_workers=2)
        sched.enqueue(self._make_jobs(2))
        sched.next_batch()
        out = Path("/tmp/seg-0.mxf")
        sched.mark_complete("seg-0", out)
        assert sched.active_count() == 1
        assert sched.completed_count() == 1

    def test_mark_failed(self):
        sched = RenderScheduler(max_workers=2)
        sched.enqueue(self._make_jobs(2))
        sched.next_batch()
        sched.mark_failed("seg-0", "render error")
        assert sched.active_count() == 1
        assert sched.failed_count() == 1

    def test_mark_complete_unknown_id_raises(self):
        sched = RenderScheduler(max_workers=2)
        with pytest.raises(KeyError):
            sched.mark_complete("nonexistent", Path("/tmp/x.mxf"))

    def test_counts(self):
        sched = RenderScheduler(max_workers=1)
        sched.enqueue(self._make_jobs(3))
        assert sched.pending_count() == 3
        assert sched.active_count() == 0
        assert sched.completed_count() == 0
        assert sched.failed_count() == 0

        sched.next_batch()
        assert sched.pending_count() == 2
        assert sched.active_count() == 1

        sched.mark_complete("seg-0", Path("/tmp/out.mxf"))
        assert sched.completed_count() == 1
        assert sched.active_count() == 0

        sched.next_batch()
        sched.mark_failed("seg-1", "oops")
        assert sched.failed_count() == 1

    def test_prioritize_by_playhead(self):
        jobs = [
            RenderJob(segment_id="a", start_ns=0, stop_ns=1_000_000_000),
            RenderJob(segment_id="b", start_ns=5_000_000_000, stop_ns=6_000_000_000),
            RenderJob(segment_id="c", start_ns=2_000_000_000, stop_ns=3_000_000_000),
        ]
        playhead = 2_500_000_000  # middle of segment c
        sorted_jobs = RenderScheduler.prioritize_by_playhead(jobs, playhead)
        # c is closest (contains playhead), then a, then b
        assert sorted_jobs[0].segment_id == "c"
        assert sorted_jobs[-1].segment_id == "b"

    def test_dequeue_in_priority_order(self):
        sched = RenderScheduler(max_workers=3)
        jobs = [
            RenderJob(segment_id="low", start_ns=0, stop_ns=1_000_000_000, priority=10),
            RenderJob(segment_id="high", start_ns=0, stop_ns=1_000_000_000, priority=0),
            RenderJob(segment_id="mid", start_ns=0, stop_ns=1_000_000_000, priority=5),
        ]
        sched.enqueue(jobs)
        batch = sched.next_batch()
        ids = [j.segment_id for j in batch]
        assert ids == ["high", "mid", "low"]
