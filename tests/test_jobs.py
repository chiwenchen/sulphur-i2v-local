"""Tests for app.jobs."""

from __future__ import annotations

import pytest

from app.jobs import JobStatus, JobTracker


@pytest.mark.asyncio
async def test_claim_succeeds_when_idle() -> None:
    tracker = JobTracker()
    job = await tracker.try_claim("a fluffy cat", {"steps": 8}, "/tmp/in.png")
    assert job is not None
    assert job.status == JobStatus.QUEUED
    assert tracker.is_busy
    assert tracker.active_job_id == job.job_id


@pytest.mark.asyncio
async def test_claim_returns_none_when_busy() -> None:
    tracker = JobTracker()
    first = await tracker.try_claim("a", {}, "/tmp/a.png")
    assert first is not None
    second = await tracker.try_claim("b", {}, "/tmp/b.png")
    assert second is None


@pytest.mark.asyncio
async def test_mark_done_releases_lock() -> None:
    tracker = JobTracker()
    job = await tracker.try_claim("x", {}, "/tmp/x.png")
    assert job is not None
    await tracker.mark_running(job.job_id)
    await tracker.mark_done(job.job_id, "/outputs/x.mp4")
    assert not tracker.is_busy
    stored = tracker.get(job.job_id)
    assert stored is not None
    assert stored.status == JobStatus.DONE
    assert stored.video_url == "/outputs/x.mp4"
    assert stored.progress == 1.0


@pytest.mark.asyncio
async def test_mark_error_releases_lock() -> None:
    tracker = JobTracker()
    job = await tracker.try_claim("x", {}, "/tmp/x.png")
    assert job is not None
    await tracker.mark_error(job.job_id, "OOM")
    assert not tracker.is_busy
    stored = tracker.get(job.job_id)
    assert stored is not None
    assert stored.status == JobStatus.ERROR
    assert stored.error == "OOM"


@pytest.mark.asyncio
async def test_progress_clamped() -> None:
    tracker = JobTracker()
    job = await tracker.try_claim("x", {}, "/tmp/x.png")
    assert job is not None
    await tracker.update_progress(job.job_id, 1.5)
    assert tracker.get(job.job_id).progress == 1.0  # type: ignore[union-attr]
    await tracker.update_progress(job.job_id, -0.2)
    assert tracker.get(job.job_id).progress == 0.0  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_to_dict_shape() -> None:
    tracker = JobTracker()
    job = await tracker.try_claim("prompt-here", {"steps": 8}, "/tmp/i.png")
    assert job is not None
    d = job.to_dict()
    assert set(d.keys()) >= {
        "job_id", "status", "progress", "video_url", "error",
        "prompt", "created_at", "started_at", "completed_at",
    }
