"""Feasibility probe for LTX-2.3 I2V on Apple Silicon via ComfyUI.

Verifies end-to-end that:
  1. ComfyUI starts and accepts a workflow
  2. The Q3_K_S GGUF transformer loads via UnetLoaderGGUF
  3. The LTX-AV text encoder + T5XXL fp8 + projection load via LTXAVTextEncoderLoader
  4. The VAE loads
  5. Minimal I2V inference (length=9, steps=4, 256x256) produces an mp4

Usage:
    # First start ComfyUI in another terminal:
    #   PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python vendor/comfyui/main.py \
    #     --listen 127.0.0.1 --port 8188 \
    #     --output-directory ./outputs/comfyui
    #
    # Then:
    uv run python scripts/probe.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from PIL import Image, ImageDraw  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.pipeline import ComfyUIClient, ComfyUIError, build_i2v_workflow  # noqa: E402

PROBE_OUT = REPO_ROOT / "outputs" / "probe"
PROBE_OUT.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[probe] {msg}", flush=True)


def make_test_image(path: Path) -> Path:
    """Create a small RGB test image with a recognizable gradient."""
    img = Image.new("RGB", (256, 256), color=(40, 80, 160))
    draw = ImageDraw.Draw(img)
    # Diagonal stripes so we can tell motion in video
    for i in range(0, 256, 16):
        draw.line([(i, 0), (0, i)], fill=(220, 200, 100), width=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return path


def main() -> int:
    base_url = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
    log(f"connecting to ComfyUI at {base_url}")

    with ComfyUIClient(base_url=base_url, timeout=30.0) as client:
        # 1. Health check
        try:
            stats = client.health()
        except Exception as exc:  # noqa: BLE001
            log(f"FAIL: ComfyUI not reachable at {base_url}: {exc}")
            log("Did you start it? See module docstring.")
            return 2
        device = stats.get("devices", [{}])[0]
        log(f"  device:    {device.get('type')} ({device.get('name')})")
        log(f"  vram:      {device.get('vram_total', 0) / 1024**3:.1f} GB total,"
            f" {device.get('vram_free', 0) / 1024**3:.1f} GB free")

        # 2. Prepare test image
        img_path = make_test_image(PROBE_OUT / "input.png")
        log(f"  fixture:   {img_path}")
        uploaded_name = client.upload_image(img_path)
        log(f"  uploaded:  {uploaded_name}")

        # 3. Build minimal workflow
        wf = build_i2v_workflow(
            image_filename=uploaded_name,
            prompt="slow zoom on the texture, cinematic, calm",
            negative_prompt="static, blurry",
            seed=42,
            width=256,
            height=256,
            length=9,        # minimum for LTXVImgToVideo
            steps=4,         # minimum
            guidance=3.0,
            frame_rate=24,
            output_prefix="probe",
        )
        log(f"  workflow:  {len(wf)} nodes")

        # 4. Submit
        try:
            prompt_id = client.submit(wf)
        except ComfyUIError as exc:
            log(f"FAIL: submit rejected by ComfyUI: {exc}")
            return 3
        log(f"  prompt_id: {prompt_id}")

        # 5. Wait for completion
        last = [0.0, time.time()]

        def progress_cb(p: float) -> None:
            now = time.time()
            if p != last[0] or now - last[1] > 30:
                log(f"  progress:  {p*100:5.1f}%  (elapsed {now - last[1]:.0f}s)")
                last[0] = p
                last[1] = now

        t0 = time.time()
        try:
            entry = client.wait_for(
                prompt_id, poll_interval=4.0, max_wait=1800, progress_cb=progress_cb
            )
        except ComfyUIError as exc:
            log(f"FAIL: workflow execution error: {exc}")
            return 4
        elapsed = time.time() - t0
        log(f"  finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")

        # 6. Verify output mp4 exists
        rel = client.find_output_video(entry)
        if rel is None:
            log(f"FAIL: no video in outputs. entry={entry}")
            return 5
        log(f"  output:    {rel}")
        out_dir = REPO_ROOT / "outputs" / "comfyui"
        candidates = list(out_dir.rglob(rel.name))
        if not candidates:
            log(f"FAIL: saved video file not found in {out_dir}")
            return 6
        mp4 = candidates[0]
        size = mp4.stat().st_size
        log(f"  file:      {mp4} ({size} bytes)")
        if size < 1000:
            log("FAIL: output mp4 is suspiciously small")
            return 7

    log("PROBE PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] FATAL: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)
