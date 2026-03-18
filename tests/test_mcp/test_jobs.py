"""Tests for async job tracking."""

from __future__ import annotations

from ave.mcp.jobs import JobTracker, JobStatus


class TestJobTracker:
    def test_create_job(self):
        tracker = JobTracker()
        job = tracker.create("edit_video")
        assert job.status == JobStatus.PENDING
        assert job.progress == 0.0
        assert len(job.job_id) == 12

    def test_get_job(self):
        tracker = JobTracker()
        job = tracker.create("edit_video")
        retrieved = tracker.get(job.job_id)
        assert retrieved is not None
        assert retrieved.job_id == job.job_id

    def test_get_unknown_returns_none(self):
        tracker = JobTracker()
        assert tracker.get("nonexistent") is None

    def test_update_progress(self):
        tracker = JobTracker()
        job = tracker.create("rotoscope")
        tracker.update_progress(job.job_id, 0.5)
        updated = tracker.get(job.job_id)
        assert updated.progress == 0.5
        assert updated.status == JobStatus.RUNNING

    def test_complete_job(self):
        tracker = JobTracker()
        job = tracker.create("edit_video")
        tracker.complete(job.job_id, {"success": True})
        completed = tracker.get(job.job_id)
        assert completed.status == JobStatus.COMPLETED
        assert completed.progress == 1.0
        assert completed.result == {"success": True}

    def test_fail_job(self):
        tracker = JobTracker()
        job = tracker.create("rotoscope")
        tracker.fail(job.job_id, "GPU out of memory")
        failed = tracker.get(job.job_id)
        assert failed.status == JobStatus.FAILED
        assert failed.error == "GPU out of memory"

    def test_list_jobs(self):
        tracker = JobTracker()
        tracker.create("job1")
        tracker.create("job2")
        tracker.create("job3")
        jobs = tracker.list_jobs()
        assert len(jobs) == 3

    def test_to_dict(self):
        tracker = JobTracker()
        job = tracker.create("test")
        d = job.to_dict()
        assert "job_id" in d
        assert "status" in d
        assert "elapsed_seconds" in d
