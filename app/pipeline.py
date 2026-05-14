"""ComfyUI HTTP client wrapper for LTX-2.3 image-to-video generation.

The pipeline submits a workflow JSON to a running ComfyUI subprocess via
its HTTP API, polls for completion, and returns the path to the generated mp4.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

import httpx
from PIL import Image

logger = logging.getLogger(__name__)


# Path inside ComfyUI's models/ tree (filename only, not absolute path).
DEFAULT_TRANSFORMER_GGUF = "10Eros_v1-Q3_K_S.gguf"
DEFAULT_VAE = "LTX23_video_vae_bf16.safetensors"
# In ComfyUI 0.21+, CLIPType.LTXV is the Gemma 3 12B / LTX-AV text encoder pair.
DEFAULT_TEXT_ENCODER = "gemma-3-12b-it-Q4_K_S.gguf"
DEFAULT_TEXT_PROJECTION = "ltx-2.3_text_projection_bf16.safetensors"
# Distilled LoRA shipped with Sulphur-2-base; lets us use ~15 steps instead of 30+.
DEFAULT_DISTILL_LORA = "ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors"
DEFAULT_LORA_STRENGTH = 0.5

# Inference defaults tuned for 16 GB Apple Silicon + LTX-2.3 (dev variant + distilled LoRA).
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 320
DEFAULT_LENGTH = 49      # frames; LTX wants 9 + 8k
DEFAULT_STEPS = 15       # with distilled LoRA at 0.5; needs 25-30+ without LoRA
DEFAULT_GUIDANCE = 3.5   # LTX-2.3 official workflow uses 3.6
DEFAULT_FRAME_RATE = 24

# LTX-2.3 scheduler hyperparameters (from official ComfyUI template, not 0.9 defaults).
LTX23_MAX_SHIFT = 2.72
LTX23_BASE_SHIFT = 0.8
LTX23_TERMINAL = 0.0


class ProgressCallback(Protocol):
    def __call__(self, progress: float) -> None: ...


class ComfyUIError(RuntimeError):
    """Raised when the ComfyUI backend returns an error or unexpected state."""


def build_i2v_workflow(
    *,
    image_filename: str,
    prompt: str,
    negative_prompt: str = "",
    seed: int,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    length: int = DEFAULT_LENGTH,
    steps: int = DEFAULT_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    frame_rate: int = DEFAULT_FRAME_RATE,
    transformer_gguf: str = DEFAULT_TRANSFORMER_GGUF,
    vae: str = DEFAULT_VAE,
    text_encoder: str = DEFAULT_TEXT_ENCODER,
    text_projection: str = DEFAULT_TEXT_PROJECTION,
    distill_lora: str | None = DEFAULT_DISTILL_LORA,
    lora_strength: float = DEFAULT_LORA_STRENGTH,
    output_prefix: str = "i2v",
) -> dict[str, dict[str, Any]]:
    """Build an LTX-2.3 I2V workflow JSON in ComfyUI API format.

    Node graph:
        Unet (GGUF) ─> [LoRA] ─┐
                                ├─> SamplerCustomAdvanced ─> VAEDecode ─> CreateVideo ─> SaveVideo
        CLIP ─> Encode(pos/neg) ─> LTXVImgToVideo ─> LTXVConditioning ─┘
        Image ─┘                       ▲
        VAE ─────────────────────────  │
                                        │
                          RandomNoise + KSamplerSelect + LTXVScheduler + CFGGuider
    """
    workflow: dict[str, dict[str, Any]] = {
        "unet_loader": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": transformer_gguf},
        },
        **(
            {
                "lora_loader": {
                    "class_type": "LoraLoaderModelOnly",
                    "inputs": {
                        "model": ["unet_loader", 0],
                        "lora_name": distill_lora,
                        "strength_model": lora_strength,
                    },
                }
            }
            if distill_lora
            else {}
        ),
        "text_encoder_loader": {
            # DualCLIPLoaderGGUF (from city96/ComfyUI-GGUF) handles GGUF + safetensors
            # mixed loading. type="ltxv" → CLIPType.LTXV → Gemma 3 12B pipeline.
            "class_type": "DualCLIPLoaderGGUF",
            "inputs": {
                "clip_name1": text_encoder,
                "clip_name2": text_projection,
                "type": "ltxv",
            },
        },
        "vae_loader": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae},
        },
        "load_image": {
            "class_type": "LoadImage",
            "inputs": {"image": image_filename},
        },
        "encode_positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["text_encoder_loader", 0], "text": prompt},
        },
        "encode_negative": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["text_encoder_loader", 0], "text": negative_prompt},
        },
        "img_to_video": {
            "class_type": "LTXVImgToVideo",
            "inputs": {
                "positive": ["encode_positive", 0],
                "negative": ["encode_negative", 0],
                "vae": ["vae_loader", 0],
                "image": ["load_image", 0],
                "width": width,
                "height": height,
                "length": length,
                "batch_size": 1,
                "strength": 1.0,
            },
        },
        "conditioning": {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["img_to_video", 0],
                "negative": ["img_to_video", 1],
                "frame_rate": frame_rate,
            },
        },
        "noise": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        "sampler_select": {
            "class_type": "KSamplerSelect",
            # euler_ancestral matches the official LTX-2.3 ComfyUI template
            "inputs": {"sampler_name": "euler_ancestral"},
        },
        "scheduler": {
            "class_type": "LTXVScheduler",
            # LTX-2.3 official scheduler hyperparameters; 0.9.x defaults give
            # under-denoised, posterized output on the new architecture.
            "inputs": {
                "steps": steps,
                "max_shift": LTX23_MAX_SHIFT,
                "base_shift": LTX23_BASE_SHIFT,
                "stretch": True,
                "terminal": LTX23_TERMINAL,
            },
        },
        "guider": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["lora_loader", 0] if distill_lora else ["unet_loader", 0],
                "positive": ["conditioning", 0],
                "negative": ["conditioning", 1],
                "cfg": guidance,
            },
        },
        "sample": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["noise", 0],
                "guider": ["guider", 0],
                "sampler": ["sampler_select", 0],
                "sigmas": ["scheduler", 0],
                "latent_image": ["img_to_video", 2],
            },
        },
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["sample", 0], "vae": ["vae_loader", 0]},
        },
        "create_video": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["decode", 0], "fps": frame_rate},
        },
        "save_video": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["create_video", 0],
                "filename_prefix": output_prefix,
                "format": "mp4",
                "codec": "h264",
            },
        },
    }
    return workflow


class ComfyUIClient:
    """Thin HTTP client around the ComfyUI server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self._client_id = str(uuid.uuid4())

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ComfyUIClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def health(self) -> dict[str, Any]:
        r = self._client.get("/system_stats")
        r.raise_for_status()
        return r.json()

    def upload_image(self, src_path: Path, *, dest_name: str | None = None) -> str:
        """Upload an image to ComfyUI's /upload/image endpoint. Returns server-side filename."""
        with open(src_path, "rb") as f:
            files = {"image": (dest_name or src_path.name, f, "image/png")}
            r = self._client.post("/upload/image", files=files, data={"overwrite": "true"})
        r.raise_for_status()
        body = r.json()
        # Successful response includes "name" with the saved filename.
        name = body.get("name")
        if not name:
            raise ComfyUIError(f"upload/image returned unexpected body: {body}")
        return name

    def submit(self, workflow: dict[str, dict[str, Any]]) -> str:
        """Submit a workflow. Returns the prompt_id."""
        payload = {"prompt": workflow, "client_id": self._client_id}
        r = self._client.post("/prompt", json=payload, timeout=60.0)
        if r.status_code != 200:
            raise ComfyUIError(f"submit returned {r.status_code}: {r.text}")
        body = r.json()
        pid = body.get("prompt_id")
        if not pid:
            raise ComfyUIError(f"submit response missing prompt_id: {body}")
        return pid

    def history(self, prompt_id: str) -> dict[str, Any] | None:
        r = self._client.get(f"/history/{prompt_id}", timeout=30.0)
        r.raise_for_status()
        body = r.json()
        return body.get(prompt_id)

    def queue_status(self) -> dict[str, Any]:
        r = self._client.get("/queue")
        r.raise_for_status()
        return r.json()

    def wait_for(
        self,
        prompt_id: str,
        *,
        poll_interval: float = 2.0,
        max_wait: float = 3600.0,
        progress_cb: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Poll /history until the prompt is done or errored. Returns the history entry."""
        start = time.time()
        last_progress = 0.0
        while True:
            entry = self.history(prompt_id)
            if entry is not None:
                status = entry.get("status", {})
                if status.get("completed") or status.get("status_str") == "success":
                    if progress_cb:
                        progress_cb(1.0)
                    return entry
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    raise ComfyUIError(f"workflow execution failed: {msgs}")
            # Estimate progress from queue position
            queue = self.queue_status()
            running = queue.get("queue_running", [])
            if running and progress_cb:
                # No fine-grained progress in /queue; just bump while running
                last_progress = min(0.9, last_progress + 0.05)
                progress_cb(last_progress)
            if time.time() - start > max_wait:
                raise ComfyUIError(f"timed out after {max_wait}s waiting for {prompt_id}")
            time.sleep(poll_interval)

    def find_output_video(self, entry: dict[str, Any]) -> Path | None:
        """Extract the saved video path from a completed history entry.

        ComfyUI's `SaveVideo` node returns its mp4 in the `images` key (alongside
        `animated: True`), not `videos`. We probe multiple candidate keys for
        forward compatibility.
        """
        outputs = entry.get("outputs", {}) or {}
        for node_output in outputs.values():
            # Prefer animated images (SaveVideo output) over still images.
            for key in ("videos", "gifs", "images"):
                items = node_output.get(key) or []
                for v in items:
                    if not isinstance(v, dict):
                        continue
                    fname = v.get("filename")
                    if not fname or not str(fname).lower().endswith((".mp4", ".webm", ".mov", ".gif")):
                        continue
                    sub = v.get("subfolder", "")
                    vtype = v.get("type", "output")
                    return Path(vtype) / sub / fname if sub else Path(vtype) / fname
        return None


def save_pil_image(img: Image.Image, dest: Path) -> Path:
    """Save a PIL image to a path as PNG. Returns the path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, format="PNG")
    return dest
