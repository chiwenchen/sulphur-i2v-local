"""Tests for app.pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from app.pipeline import (
    ComfyUIClient,
    ComfyUIError,
    build_i2v_workflow,
)


class TestBuildWorkflow:
    def test_workflow_has_required_nodes(self) -> None:
        wf = build_i2v_workflow(
            image_filename="test.png",
            prompt="a cat",
            seed=42,
        )
        node_types = {n["class_type"] for n in wf.values()}
        # The minimum set of capabilities required for LTX-2.3 I2V
        assert "UnetLoaderGGUF" in node_types
        assert "DualCLIPLoaderGGUF" in node_types
        assert "VAELoader" in node_types
        assert "LoadImage" in node_types
        assert "CLIPTextEncode" in node_types
        assert "LTXVImgToVideo" in node_types
        assert "LTXVConditioning" in node_types
        assert "SamplerCustomAdvanced" in node_types
        assert "VAEDecode" in node_types
        assert "SaveVideo" in node_types

    def test_prompt_injected(self) -> None:
        wf = build_i2v_workflow(
            image_filename="img.png", prompt="a fluffy cat playing piano", seed=1
        )
        positive = next(
            n for n in wf.values()
            if n["class_type"] == "CLIPTextEncode" and n["inputs"]["text"] == "a fluffy cat playing piano"
        )
        assert positive is not None

    def test_seed_injected(self) -> None:
        wf = build_i2v_workflow(image_filename="i.png", prompt="x", seed=12345)
        noise = next(n for n in wf.values() if n["class_type"] == "RandomNoise")
        assert noise["inputs"]["noise_seed"] == 12345

    def test_dimensions_injected(self) -> None:
        wf = build_i2v_workflow(
            image_filename="i.png",
            prompt="x",
            seed=0,
            width=384,
            height=256,
            length=33,
        )
        i2v = next(n for n in wf.values() if n["class_type"] == "LTXVImgToVideo")
        assert i2v["inputs"]["width"] == 384
        assert i2v["inputs"]["height"] == 256
        assert i2v["inputs"]["length"] == 33

    def test_image_filename_injected(self) -> None:
        wf = build_i2v_workflow(image_filename="myinput.png", prompt="x", seed=0)
        load = next(n for n in wf.values() if n["class_type"] == "LoadImage")
        assert load["inputs"]["image"] == "myinput.png"

    def test_node_references_resolve(self) -> None:
        """Every input that is a list reference [node_id, output_idx] must point to a real node."""
        wf = build_i2v_workflow(image_filename="i.png", prompt="x", seed=0)
        for node_id, node in wf.items():
            for k, v in node["inputs"].items():
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                    referenced = v[0]
                    assert referenced in wf, (
                        f"Node {node_id} field {k} references missing node {referenced!r}"
                    )


class TestComfyUIClient:
    """Tests for the HTTP client. Uses httpx MockTransport to avoid network."""

    def _mock_client(self, handler) -> ComfyUIClient:
        transport = httpx.MockTransport(handler)
        client = ComfyUIClient(base_url="http://127.0.0.1:8188")
        client._client = httpx.Client(base_url="http://127.0.0.1:8188", transport=transport)
        return client

    def test_health(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.url.path == "/system_stats"
            return httpx.Response(200, json={"system": {"comfyui_version": "0.21.1"}})

        client = self._mock_client(handler)
        stats = client.health()
        assert "system" in stats

    def test_submit_returns_prompt_id(self) -> None:
        import json as _json

        def handler(req: httpx.Request) -> httpx.Response:
            assert req.url.path == "/prompt"
            payload = _json.loads(req.read())
            assert "prompt" in payload
            assert "client_id" in payload
            return httpx.Response(200, json={"prompt_id": "abc-123", "number": 1})

        client = self._mock_client(handler)
        wf = build_i2v_workflow(image_filename="i.png", prompt="x", seed=0)
        pid = client.submit(wf)
        assert pid == "abc-123"

    def test_submit_raises_on_400(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text="bad workflow")

        client = self._mock_client(handler)
        wf = build_i2v_workflow(image_filename="i.png", prompt="x", seed=0)
        with pytest.raises(ComfyUIError, match="400"):
            client.submit(wf)

    def test_history_returns_none_for_unknown(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        client = self._mock_client(handler)
        assert client.history("unknown-id") is None

    def test_wait_for_success(self) -> None:
        call_count = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if req.url.path == "/queue":
                return httpx.Response(200, json={"queue_running": [], "queue_pending": []})
            if req.url.path.startswith("/history/"):
                # Pretend it's done after a single call
                return httpx.Response(
                    200,
                    json={
                        "abc-1": {
                            "status": {"completed": True, "status_str": "success"},
                            "outputs": {"15": {"videos": [{"filename": "out.mp4", "subfolder": "", "type": "output"}]}},
                        }
                    },
                )
            return httpx.Response(404)

        client = self._mock_client(handler)
        entry = client.wait_for("abc-1", poll_interval=0.01, max_wait=5)
        assert entry["status"]["completed"] is True

    def test_wait_for_error_raises(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            if req.url.path.startswith("/history/"):
                return httpx.Response(
                    200,
                    json={
                        "abc-2": {
                            "status": {
                                "completed": False,
                                "status_str": "error",
                                "messages": [["execution_error", {"node_id": "1", "exception_message": "OOM"}]],
                            }
                        }
                    },
                )
            return httpx.Response(200, json={"queue_running": [], "queue_pending": []})

        client = self._mock_client(handler)
        with pytest.raises(ComfyUIError, match="execution failed"):
            client.wait_for("abc-2", poll_interval=0.01, max_wait=5)

    def test_find_output_video(self) -> None:
        client = self._mock_client(lambda r: httpx.Response(404))
        entry = {
            "status": {"completed": True},
            "outputs": {
                "save_video": {
                    "videos": [
                        {"filename": "i2v_00001_.mp4", "subfolder": "", "type": "output"}
                    ]
                }
            },
        }
        path = client.find_output_video(entry)
        assert path is not None
        assert path.name == "i2v_00001_.mp4"

    def test_find_output_video_from_images_key(self) -> None:
        """SaveVideo node actually puts the mp4 in `images` not `videos`."""
        client = self._mock_client(lambda r: httpx.Response(404))
        entry = {
            "status": {"completed": True},
            "outputs": {
                "save_video": {
                    "images": [
                        {"filename": "probe_00001_.mp4", "subfolder": "", "type": "output"}
                    ],
                    "animated": [True],
                }
            },
        }
        path = client.find_output_video(entry)
        assert path is not None
        assert path.name == "probe_00001_.mp4"

    def test_find_output_video_ignores_still_images(self) -> None:
        client = self._mock_client(lambda r: httpx.Response(404))
        entry = {
            "outputs": {
                "save_image": {"images": [{"filename": "preview.png", "type": "output"}]}
            },
        }
        assert client.find_output_video(entry) is None
