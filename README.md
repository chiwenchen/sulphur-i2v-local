# Sulphur-2 Image-to-Video — Local Web App

Local web app that runs [`SulphurAI/Sulphur-2-base`](https://huggingface.co/SulphurAI/Sulphur-2-base) for **Image-to-Video** generation, on macOS (Apple Silicon).

Single-user, runs entirely on `localhost`. No cloud, no auth.

## Hardware target

- MacBook Pro, Apple M5 (or M-series), 16 GB unified memory minimum
- macOS with Metal 4 / PyTorch MPS

> 16 GB is borderline. Close other heavy apps (Docker Desktop, Chrome with many tabs, Slack) before generating.

## Setup

```bash
uv sync --extra dev
```

First run will download model weights from Hugging Face Hub (~10 GB+). Subsequent runs use the cache.

## Run

```bash
uv run python -m app
```

Open `http://localhost:7860`.

## Test

```bash
uv run pytest                # fast unit tests
uv run pytest -m slow        # opt-in: real inference, may take 5+ min
```

## Feasibility probe

If you hit issues, run the probe to verify the model loads and inference works:

```bash
uv run python scripts/probe.py
```

## Design

See [`docs/DESIGN.md`](docs/DESIGN.md).
