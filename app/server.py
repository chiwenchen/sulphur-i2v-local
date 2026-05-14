"""FastAPI server: thin wrapper that submits I2V workflows to ComfyUI."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.jobs import Job, JobTracker
from app.pipeline import (
    DEFAULT_FRAME_RATE,
    DEFAULT_GUIDANCE,
    DEFAULT_HEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    ComfyUIClient,
    ComfyUIError,
    build_i2v_workflow,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
OUTPUTS_DIR = REPO_ROOT / "outputs"
INPUTS_DIR = OUTPUTS_DIR / "inputs"
COMFYUI_OUTPUTS_DIR = OUTPUTS_DIR / "comfyui"

COMFYUI_BASE_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
INPUTS_DIR.mkdir(parents=True, exist_ok=True)
COMFYUI_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def create_app(tracker: JobTracker | None = None) -> FastAPI:
    app = FastAPI(title="Sulphur-2 / LTX-2.3 Image-to-Video")
    app.state.tracker = tracker or JobTracker()

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def root() -> HTMLResponse:
        index = STATIC_DIR / "index.html"
        if not index.exists():
            return HTMLResponse("<h1>index.html missing</h1>", status_code=500)
        return HTMLResponse(index.read_text(encoding="utf-8"))

    @app.get("/api/health")
    async def health() -> dict:
        comfy_ok = False
        try:
            with ComfyUIClient(base_url=COMFYUI_BASE_URL, timeout=2.0) as c:
                c.health()
                comfy_ok = True
        except Exception as exc:  # noqa: BLE001
            logger.debug("comfyui health check failed: %s", exc)
        return {
            "ok": True,
            "comfyui_reachable": comfy_ok,
            "busy": app.state.tracker.is_busy,
            "active_job_id": app.state.tracker.active_job_id,
        }

    @app.post("/api/generate")
    async def generate(
        image: UploadFile = File(...),
        prompt: str = Form(...),
        negative_prompt: str = Form(""),
        seed: int = Form(-1),
        width: int = Form(DEFAULT_WIDTH),
        height: int = Form(DEFAULT_HEIGHT),
        length: int = Form(DEFAULT_LENGTH),
        steps: int = Form(DEFAULT_STEPS),
        guidance: float = Form(DEFAULT_GUIDANCE),
        frame_rate: int = Form(DEFAULT_FRAME_RATE),
    ) -> JSONResponse:
        if not prompt.strip():
            raise HTTPException(status_code=422, detail="prompt is required")
        if image.content_type and not image.content_type.startswith("image/"):
            raise HTTPException(status_code=422, detail=f"not an image: {image.content_type}")

        # Save the uploaded image to a stable location
        input_id = uuid.uuid4().hex
        ext = (image.filename or "input.png").rsplit(".", 1)[-1].lower()
        if ext not in {"png", "jpg", "jpeg", "webp"}:
            ext = "png"
        input_path = INPUTS_DIR / f"{input_id}.{ext}"
        with open(input_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed if seed >= 0 else int.from_bytes(os.urandom(4), "big"),
            "width": width,
            "height": height,
            "length": length,
            "steps": steps,
            "guidance": guidance,
            "frame_rate": frame_rate,
        }

        job = await app.state.tracker.try_claim(prompt, params, str(input_path))
        if job is None:
            raise HTTPException(
                status_code=409,
                detail="another generation is already in flight; wait for it to finish",
            )

        asyncio.create_task(_run_job(app, job, input_path))

        return JSONResponse(
            {"job_id": job.job_id, "status": job.status.value},
            status_code=202,
        )

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict:
        job = app.state.tracker.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job.to_dict()

    @app.get("/outputs/{path:path}")
    async def serve_output(path: str) -> FileResponse:
        # Constrain to OUTPUTS_DIR to avoid traversal
        full = (OUTPUTS_DIR / path).resolve()
        if not str(full).startswith(str(OUTPUTS_DIR.resolve())):
            raise HTTPException(status_code=403, detail="forbidden")
        if not full.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(str(full))

    return app


async def _run_job(app: FastAPI, job: Job, input_path: Path) -> None:
    """Background task: drive the ComfyUI workflow for a single job."""
    tracker: JobTracker = app.state.tracker
    try:
        await tracker.mark_running(job.job_id)
        loop = asyncio.get_event_loop()

        def _do_generation() -> str:
            client = ComfyUIClient(base_url=COMFYUI_BASE_URL, timeout=60.0)
            try:
                # Upload the image to ComfyUI
                uploaded_name = client.upload_image(input_path, dest_name=input_path.name)
                p = job.params
                wf = build_i2v_workflow(
                    image_filename=uploaded_name,
                    prompt=p["prompt"],
                    negative_prompt=p.get("negative_prompt", ""),
                    seed=p["seed"],
                    width=p["width"],
                    height=p["height"],
                    length=p["length"],
                    steps=p["steps"],
                    guidance=p["guidance"],
                    frame_rate=p["frame_rate"],
                    output_prefix=f"i2v_{job.job_id[:8]}",
                )
                prompt_id = client.submit(wf)

                def progress_cb(p: float) -> None:
                    asyncio.run_coroutine_threadsafe(
                        tracker.update_progress(job.job_id, p), loop
                    )

                entry = client.wait_for(prompt_id, poll_interval=3.0, max_wait=3600, progress_cb=progress_cb)
                rel = client.find_output_video(entry)
                if rel is None:
                    raise ComfyUIError(f"no video in outputs: {entry.get('outputs')}")
                # The relative path is like "output/i2v_xxx_00001_.mp4"
                # ComfyUI's --output-directory is COMFYUI_OUTPUTS_DIR so video lives there
                src = COMFYUI_OUTPUTS_DIR / rel.name
                if not src.exists():
                    # Sometimes the saver creates with a subfolder
                    candidates = list(COMFYUI_OUTPUTS_DIR.rglob(rel.name))
                    if candidates:
                        src = candidates[0]
                    else:
                        raise ComfyUIError(f"saved video not found at {src}")
                # Move into the served-outputs root for a cleaner URL
                dst = OUTPUTS_DIR / f"{job.job_id}.mp4"
                shutil.copy2(src, dst)
                return f"/outputs/{job.job_id}.mp4"
            finally:
                client.close()

        video_url = await loop.run_in_executor(None, _do_generation)
        await tracker.mark_done(job.job_id, video_url)

    except Exception as exc:  # noqa: BLE001
        logger.exception("job %s failed", job.job_id)
        await tracker.mark_error(job.job_id, f"{type(exc).__name__}: {exc}")


def main() -> int:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
