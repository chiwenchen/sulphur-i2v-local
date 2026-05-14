"""In-memory job tracker for image-to-video generation.

Single-job design: only one generation runs at a time. The tracker exposes a
lock so the API layer can return 409 when another job is already in flight.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class Job:
    """A single I2V generation job."""

    job_id: str
    prompt: str
    params: dict[str, Any]
    input_image_path: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    video_url: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "progress": self.progress,
            "video_url": self.video_url,
            "error": self.error,
            "prompt": self.prompt,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class JobTracker:
    """In-memory single-job tracker.

    Thread-safety: protected by asyncio.Lock for the single-job invariant.
    All state operations should be called from a single event loop.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()
        self._active_job_id: str | None = None

    @property
    def is_busy(self) -> bool:
        return self._active_job_id is not None

    @property
    def active_job_id(self) -> str | None:
        return self._active_job_id

    async def try_claim(
        self, prompt: str, params: dict[str, Any], input_image_path: str
    ) -> Job | None:
        """Try to start a new job. Returns the new Job, or None if busy."""
        async with self._lock:
            if self._active_job_id is not None:
                return None
            job = Job(
                job_id=str(uuid.uuid4()),
                prompt=prompt,
                params=params,
                input_image_path=input_image_path,
            )
            self._jobs[job.job_id] = job
            self._active_job_id = job.job_id
            return job

    async def mark_running(self, job_id: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != JobStatus.QUEUED:
                return
            job.status = JobStatus.RUNNING
            job.started_at = time.time()

    async def update_progress(self, job_id: str, progress: float) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.progress = max(0.0, min(1.0, progress))

    async def mark_done(self, job_id: str, video_url: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JobStatus.DONE
            job.progress = 1.0
            job.video_url = video_url
            job.completed_at = time.time()
            if self._active_job_id == job_id:
                self._active_job_id = None

    async def mark_error(self, job_id: str, error: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JobStatus.ERROR
            job.error = error
            job.completed_at = time.time()
            if self._active_job_id == job_id:
                self._active_job_id = None

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)
