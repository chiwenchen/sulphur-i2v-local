"""Tests for the FastAPI server (with mocked ComfyUI backend)."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.jobs import JobStatus, JobTracker
from app.server import create_app


def _make_png_bytes(size: tuple[int, int] = (64, 64), color: tuple[int, int, int] = (200, 100, 50)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client() -> TestClient:
    app = create_app(tracker=JobTracker())
    return TestClient(app)


def test_health_returns_busy_false_initially(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["busy"] is False
    assert body["active_job_id"] is None


def test_generate_returns_202_with_job_id(client: TestClient) -> None:
    # Patch the background task so it doesn't try to contact ComfyUI
    with patch("app.server._run_job", new_callable=AsyncMock) as mocked:
        r = client.post(
            "/api/generate",
            data={"prompt": "a fluffy cat"},
            files={"image": ("input.png", _make_png_bytes(), "image/png")},
        )
        assert r.status_code == 202, r.text
        body = r.json()
        assert "job_id" in body
        assert body["status"] == "queued"
        # And the background task should have been scheduled
        mocked.assert_called_once()


def test_generate_409_when_busy(client: TestClient) -> None:
    """Second submission while a job is in flight gets 409."""
    with patch("app.server._run_job", new_callable=AsyncMock):
        first = client.post(
            "/api/generate",
            data={"prompt": "first"},
            files={"image": ("a.png", _make_png_bytes(), "image/png")},
        )
        assert first.status_code == 202
        second = client.post(
            "/api/generate",
            data={"prompt": "second"},
            files={"image": ("b.png", _make_png_bytes(), "image/png")},
        )
        assert second.status_code == 409


def test_generate_requires_prompt(client: TestClient) -> None:
    r = client.post(
        "/api/generate",
        data={"prompt": ""},
        files={"image": ("a.png", _make_png_bytes(), "image/png")},
    )
    assert r.status_code == 422


def test_generate_rejects_non_image(client: TestClient) -> None:
    r = client.post(
        "/api/generate",
        data={"prompt": "x"},
        files={"image": ("a.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 422


def test_get_job_404_unknown(client: TestClient) -> None:
    r = client.get("/api/jobs/does-not-exist")
    assert r.status_code == 404


def test_get_job_after_submit(client: TestClient) -> None:
    with patch("app.server._run_job", new_callable=AsyncMock):
        submit = client.post(
            "/api/generate",
            data={"prompt": "x"},
            files={"image": ("a.png", _make_png_bytes(), "image/png")},
        )
        job_id = submit.json()["job_id"]
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == job_id
        assert body["status"] == JobStatus.QUEUED.value


def test_serve_outputs_404_when_missing(client: TestClient) -> None:
    r = client.get("/outputs/nope.mp4")
    assert r.status_code == 404


def test_outputs_forbids_traversal(client: TestClient) -> None:
    r = client.get("/outputs/../app/server.py")
    # Either 404 (resolved outside) or 403 (explicit block) is fine
    assert r.status_code in (403, 404)
