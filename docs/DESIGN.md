# LTX-2.3 Image-to-Video Local Web App — Design

**Date:** 2026-05-14
**Owner:** cwchen2000@gmail.com
**Target hardware:** MacBook Pro, Apple M5, 16 GB unified memory, 10-core GPU, Metal 4

## 0. Pivot note (2026-05-14)

Original plan was to run `SulphurAI/Sulphur-2-base` via the `diffusers` library.
Investigation found:

1. The `Sulphur-2-base` repo is not in `diffusers` format — loose `.safetensors`
   files + ComfyUI workflow JSONs only.
2. All Sulphur-2-base weight variants are **27–43 GB**, too large for 16 GB unified
   memory even with offloading.

**Pivot:** Use the community-quantized GGUF `vantagewithai/LTX2.3-10Eros-GGUF`
(same LTX 2.3 21B base, Q3_K_S, **10.3 GB**), running through a headless **ComfyUI**
subprocess. The VAE + text encoder come from `Sulphur-2-base` (separate small files).

## 1. Goal

Build a local web application that runs an LTX 2.3-class Image-to-Video model on a
MacBook Pro M5 with 16 GB RAM. Single-user, runs entirely on `localhost`. No cloud,
no auth.

## 2. Non-Goals

- Text-to-Video (Sulphur-2 supports it; out of scope for v1)
- Multi-user / authentication / sharing
- Persistent history, gallery, favorites
- Job queue UI (single-job in-memory tracker only)
- Training / fine-tuning
- Mobile-responsive UI

## 3. Hard Constraints

- **16 GB unified memory** is borderline for a 9B-parameter video diffusion model. RAM optimizations are mandatory, not optional.
- First-run model download is ~10 GB+. UI must communicate this.
- Per-video inference is expected to take **5–15 minutes** on M5; UI must not block.
- PyTorch MPS does not yet cover every op used by LTX-class video pipelines. `PYTORCH_ENABLE_MPS_FALLBACK=1` is required.

## 4. Architecture

```
┌───────────────────────────────────────────────┐
│ Browser (localhost:7860)                      │
│  index.html + app.js + Tailwind CDN           │
└────────────┬──────────────────────────────────┘
             │ multipart POST /api/generate
             │ GET /api/jobs/{id}
             │ GET /outputs/{uuid}.mp4
             ▼
┌───────────────────────────────────────────────┐
│ FastAPI on :7860 (uvicorn, single worker)     │
│  server.py     ← routes                       │
│  jobs.py       ← in-memory job tracker        │
│  pipeline.py   ← ComfyUI HTTP client          │
└────────────┬──────────────────────────────────┘
             │ POST /prompt (workflow JSON)
             │ GET  /history/{prompt_id}
             │ WS   /ws (progress)
             ▼
┌───────────────────────────────────────────────┐
│ ComfyUI subprocess on :8188 (headless)        │
│  custom_nodes/ComfyUI-GGUF                    │
│  custom_nodes/10S-Comfy-nodes (if needed)     │
│  models/diffusion_models/LTX2.3-10Eros-Q3_K_S │
│  models/vae/  models/text_encoders/           │
└───────────────────────────────────────────────┘
```

## 5. API contract

### `POST /api/generate`
- Content-Type: `multipart/form-data`
- Fields:
  - `image` (file, required) — input image, JPEG/PNG, max 4 MB
  - `prompt` (str, required) — text prompt
  - `seed` (int, optional, default: random)
  - `num_inference_steps` (int, optional, default: 30)
  - `guidance_scale` (float, optional, default: 3.0)
  - `num_frames` (int, optional, default: 65, max: 121)
- Returns: `{"job_id": "uuid", "status": "queued"}`
- Status: 409 if another job is already running

### `GET /api/jobs/{job_id}`
- Returns:
  ```json
  {
    "job_id": "...",
    "status": "queued|running|done|error",
    "progress": 0.42,
    "video_url": "/outputs/{uuid}.mp4" | null,
    "error": "..." | null,
    "created_at": "...",
    "completed_at": "..." | null
  }
  ```

### `GET /outputs/{filename}`
- Serves generated mp4 from disk.

### `GET /api/health`
- Returns `{"status": "ok", "model_loaded": bool}`

## 6. Pipeline contract

```python
class I2VPipeline:
    def __init__(self, model_id: str, device: str = "mps") -> None: ...
    def load(self) -> None: ...  # idempotent, loads weights
    @property
    def ready(self) -> bool: ...
    def generate(
        self,
        image: PIL.Image.Image,
        prompt: str,
        *,
        seed: int | None,
        num_inference_steps: int,
        guidance_scale: float,
        num_frames: int,
        output_path: Path,
        progress_callback: Callable[[float], None] | None,
    ) -> Path: ...  # returns mp4 path
```

## 7. Frontend behavior

- **Idle state**: empty image dropzone, empty prompt textarea, advanced-params collapsed, "Generate" button enabled
- **Generating state**: dropzone + textarea + button disabled; progress bar (indeterminate if callback unavailable, else 0–100%); message "預估 5–15 分鐘，請勿關閉分頁"
- **Done state**: `<video controls>` with the generated mp4; download button; "Generate another" button to reset
- **Error state**: red banner with error message; "重試" button

Polling: every 5 s while job is in `queued|running`; backoff to 10 s after 5 minutes elapsed.

## 8. Storage

- `outputs/{uuid}.mp4` — generated videos (gitignored)
- `outputs/{uuid}.input.png` — original input image (kept for reproducibility, gitignored)
- No database. Job state lives in-memory; lost on restart. v1 deliberately accepts this.

## 9. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| MPS missing ops for LTX pipeline | High | `PYTORCH_ENABLE_MPS_FALLBACK=1`; if still broken, drop to full CPU (very slow) |
| OOM during inference (16 GB tight) | High | sequential CPU offload + attention/VAE slicing; offer fp16; auto-shrink to num_frames=49 + 512×320 on retry |
| `SulphurAI/Sulphur-2-base` not directly loadable via `diffusers.LTXImageToVideoPipeline` (custom weights) | Medium | Fallback A: use base LTX pipeline with Sulphur weights merged manually. Fallback B: switch backend to `llama.cpp llama-server` and adapt. |
| Model download fails / partial | Medium | Use HF Hub's resume support; surface progress to UI on first launch |
| ffmpeg not installed | Low | Use `imageio-ffmpeg` (bundles ffmpeg binary) instead of system ffmpeg |
| Single-job lock leaks (job crashes, lock not released) | Low | `try/finally` around inference; on server restart, lock is reset (in-memory) |

## 10. Testing strategy

### Step 0 — Feasibility probe (BLOCKING)
Standalone script `scripts/probe.py`. Loads model, runs 10-step inference on a 256×256 solid-color image, asserts mp4 output exists and is readable. **If this fails, the whole architecture pivots — do not write web app code first.**

### Backend unit tests (`tests/test_api.py`)
- Mock `I2VPipeline`. Assert routes return correct status codes, contracts, and lock behavior. Fast (<1 s).

### Backend pipeline test (`tests/test_pipeline.py`, marked `@pytest.mark.slow`)
- Real model load + minimal inference (lowest possible num_frames, smallest resolution). Asserts mp4 produced and `ffprobe` reports valid stream. Opt-in via `pytest -m slow`. May take 5+ min.

### E2E (Playwright via `/qa` skill)
- Open `localhost:7860`
- Upload a fixture image
- Type a prompt
- Click Generate
- Wait up to 20 min for status=done
- Assert `<video>` has non-empty `src`
- Download mp4, ffprobe sanity check

### Manual smoke
- Open the produced mp4 in QuickTime, confirm it actually plays as a video (not a single frame).

## 11. Project layout

```
sulphur-i2v-local/
├── README.md
├── pyproject.toml          # uv-managed
├── .python-version         # 3.11
├── .gitignore              # outputs/, .venv, __pycache__, *.mp4
├── scripts/
│   └── probe.py            # feasibility probe (step 0)
├── app/
│   ├── __init__.py
│   ├── __main__.py         # uvicorn entry: python -m app
│   ├── server.py
│   ├── pipeline.py
│   ├── jobs.py
│   └── static/
│       ├── index.html
│       ├── app.js
│       └── style.css
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_pipeline.py
│   └── fixtures/
│       └── test_input.png  # 256×256 solid color
└── outputs/                # gitignored
```

## 12. Execution plan

1. Write this design doc + commit
2. Create `sulphur-i2v-local/` + git init + GitHub private repo + `/github-repo-init`
3. Branch `feat/i2v-mvp`
4. **Feasibility probe** — go/no-go gate
5. `pipeline.py` + `tests/test_pipeline.py` (TDD, mocked first)
6. `server.py` + `jobs.py` + `tests/test_api.py`
7. Frontend static three files
8. Run pytest (unit always; slow integration opt-in once)
9. Boot server, run Playwright E2E via `/qa`
10. PR → merge → status report (PR#, status, test report, E2E report)

## 13. Open questions (acceptable to defer)

- Optional: prompt enhancer (requires loading a second 4 B+ LLM via LM Studio). **Deferred** for v1 — too much RAM pressure.
- Optional: model quantization (Q8_0 GGUF) for tighter memory. **Deferred** — try bf16 first; only swap if OOM is unavoidable.
- Optional: deploying to a Fly machine for sharing. **Out of scope** — local-only by design.
