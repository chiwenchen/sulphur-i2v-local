"""Download all model components needed for LTX-2.3 I2V via ComfyUI.

Total: ~19 GB. Resumable (uses HuggingFace Hub cache).

Usage:
    uv run python scripts/download_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "vendor" / "comfyui" / "models"

DOWNLOADS: list[dict] = [
    {
        "repo_id": "vantagewithai/LTX2.3-10Eros-GGUF",
        "filename": "10Eros_v1-Q3_K_S.gguf",
        "dest_dir": MODELS_DIR / "unet",
        "human_size": "10.3 GB",
        "purpose": "transformer (quantized)",
    },
    {
        "repo_id": "Kijai/LTX2.3_comfy",
        "filename": "vae/LTX23_video_vae_bf16.safetensors",
        "dest_dir": MODELS_DIR / "vae",
        "human_size": "1.4 GB",
        "purpose": "video VAE",
    },
    {
        "repo_id": "Kijai/LTX2.3_comfy",
        "filename": "text_encoders/ltx-2.3_text_projection_bf16.safetensors",
        "dest_dir": MODELS_DIR / "text_encoders",
        "human_size": "2.2 GB",
        "purpose": "LTX-2.3 text projection layer",
    },
    {
        "repo_id": "Lightricks/T5-XXL-8bit",
        "filename": "model-00001-of-00002.safetensors",
        "dest_dir": MODELS_DIR / "text_encoders" / "t5xxl-8bit",
        "human_size": "4.8 GB",
        "purpose": "T5-XXL text encoder (shard 1/2)",
    },
    {
        "repo_id": "Lightricks/T5-XXL-8bit",
        "filename": "model-00002-of-00002.safetensors",
        "dest_dir": MODELS_DIR / "text_encoders" / "t5xxl-8bit",
        "human_size": "0.86 GB",
        "purpose": "T5-XXL text encoder (shard 2/2)",
    },
]


def main() -> int:
    print(f"Downloading {len(DOWNLOADS)} files to {MODELS_DIR}", flush=True)
    for i, d in enumerate(DOWNLOADS, 1):
        d["dest_dir"].mkdir(parents=True, exist_ok=True)
        print(f"\n[{i}/{len(DOWNLOADS)}] {d['repo_id']} :: {d['filename']}", flush=True)
        print(f"  purpose: {d['purpose']}  size: {d['human_size']}", flush=True)
        try:
            path = hf_hub_download(
                repo_id=d["repo_id"],
                filename=d["filename"],
                local_dir=str(d["dest_dir"]),
            )
            sz_gb = Path(path).stat().st_size / 1024**3
            print(f"  -> {path}  ({sz_gb:.2f} GB)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            return 1
    print("\nAll downloads complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
