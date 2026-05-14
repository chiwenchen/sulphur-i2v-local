"""Feasibility probe for SulphurAI/Sulphur-2-base on Apple Silicon (MPS).

Goal: prove (or disprove) that we can load the model and run a minimal
inference on this hardware before building the full web app.

Usage:
    uv run python scripts/probe.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

# CRITICAL: enable CPU fallback for ops missing on MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from PIL import Image

MODEL_ID = "SulphurAI/Sulphur-2-base"
PROBE_OUT = Path("outputs/probe")
PROBE_OUT.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[probe] {msg}", flush=True)


def probe_env() -> None:
    log(f"python: {sys.version.split()[0]}")
    log(f"torch:  {torch.__version__}")
    log(f"mps available: {torch.backends.mps.is_available()}")
    log(f"mps built:     {torch.backends.mps.is_built()}")
    try:
        import diffusers

        log(f"diffusers: {diffusers.__version__}")
    except ImportError as exc:
        log(f"diffusers import FAILED: {exc}")
        raise


def probe_model_card() -> None:
    """Inspect the model card / config to figure out what pipeline class to use."""
    log("--- inspecting model repo ---")
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        info = api.model_info(MODEL_ID)
    except Exception as exc:
        log(f"model_info FAILED: {exc}")
        raise

    log(f"pipeline_tag: {info.pipeline_tag}")
    log(f"library_name: {info.library_name}")
    log(f"tags: {info.tags}")
    files = [f.rfilename for f in (info.siblings or [])]
    log(f"file count: {len(files)}")
    for f in sorted(files)[:30]:
        log(f"  - {f}")
    if len(files) > 30:
        log(f"  ... and {len(files) - 30} more")


def probe_load_pipeline():
    """Try several strategies to load the model."""
    from diffusers import DiffusionPipeline

    strategies = [
        ("DiffusionPipeline.from_pretrained bf16 + trust_remote_code", {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }),
        ("DiffusionPipeline.from_pretrained fp16 + trust_remote_code", {
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
        }),
        ("DiffusionPipeline.from_pretrained fp32 + trust_remote_code", {
            "torch_dtype": torch.float32,
            "trust_remote_code": True,
        }),
    ]

    last_err: Exception | None = None
    for name, kwargs in strategies:
        log(f"--- trying: {name} ---")
        try:
            t0 = time.time()
            pipe = DiffusionPipeline.from_pretrained(MODEL_ID, **kwargs)
            log(f"loaded in {time.time() - t0:.1f}s, class={type(pipe).__name__}")
            return pipe
        except Exception as exc:
            log(f"FAILED: {type(exc).__name__}: {exc}")
            last_err = exc
            traceback.print_exc()

    raise RuntimeError(f"All load strategies failed; last: {last_err}")


def probe_pipeline_signature(pipe) -> None:
    log("--- pipeline introspection ---")
    log(f"class: {type(pipe).__name__}")
    log(f"module: {type(pipe).__module__}")
    components = getattr(pipe, "components", None)
    if isinstance(components, dict):
        for k, v in components.items():
            log(f"  component {k}: {type(v).__name__ if v is not None else None}")
    callable_obj = pipe.__call__
    import inspect

    try:
        sig = inspect.signature(callable_obj)
        log(f"__call__ signature: {sig}")
    except (TypeError, ValueError) as exc:
        log(f"could not introspect signature: {exc}")


def probe_move_to_mps(pipe):
    log("--- moving pipeline to MPS ---")
    try:
        pipe = pipe.to("mps")
        log("moved to mps OK")
    except Exception as exc:
        log(f"to(mps) FAILED: {exc}; trying enable_sequential_cpu_offload")
        try:
            pipe.enable_sequential_cpu_offload()
            log("sequential_cpu_offload OK")
        except Exception as exc2:
            log(f"cpu_offload FAILED: {exc2}; will run on CPU")
    for opt in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
        fn = getattr(pipe, opt, None)
        if callable(fn):
            try:
                fn()
                log(f"{opt}() OK")
            except Exception as exc:
                log(f"{opt} FAILED: {exc}")
    return pipe


def probe_minimal_inference(pipe) -> None:
    log("--- minimal inference ---")
    # Create a tiny solid-color image
    img = Image.new("RGB", (256, 256), color=(80, 120, 200))
    fixture_path = PROBE_OUT / "input.png"
    img.save(fixture_path)
    log(f"saved fixture image: {fixture_path}")

    # Try to figure out call signature. LTX I2V typically wants `image=` and `prompt=`.
    import inspect

    sig = inspect.signature(pipe.__call__)
    params = sig.parameters
    log(f"pipeline params: {list(params.keys())}")

    call_kwargs: dict = {"prompt": "a slow camera pan over a calm blue field"}
    # I2V pipelines accept `image`
    if "image" in params:
        call_kwargs["image"] = img
    # Constrain everything to be cheap
    if "num_inference_steps" in params:
        call_kwargs["num_inference_steps"] = 6
    if "num_frames" in params:
        call_kwargs["num_frames"] = 17  # very short
    if "height" in params:
        call_kwargs["height"] = 256
    if "width" in params:
        call_kwargs["width"] = 256
    if "guidance_scale" in params:
        call_kwargs["guidance_scale"] = 3.0

    log(f"call kwargs: { {k: type(v).__name__ for k, v in call_kwargs.items()} }")

    t0 = time.time()
    try:
        result = pipe(**call_kwargs)
    except Exception as exc:
        log(f"inference FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    log(f"inference returned in {time.time() - t0:.1f}s, type={type(result).__name__}")

    # Inspect result
    for attr in ("frames", "videos", "video", "images"):
        val = getattr(result, attr, None)
        if val is not None:
            log(f"result.{attr}: type={type(val).__name__}, "
                f"len={len(val) if hasattr(val, '__len__') else 'n/a'}")

    # Try to save output as mp4
    frames = getattr(result, "frames", None) or getattr(result, "videos", None)
    if frames is None:
        log("WARNING: no frames-like attribute on result; cannot write mp4")
        return

    # frames is typically a list of lists of PIL images, shape [batch][frame]
    try:
        seq = frames[0] if isinstance(frames, (list, tuple)) else frames
        log(f"frame-seq type={type(seq).__name__}, "
            f"len={len(seq) if hasattr(seq, '__len__') else 'n/a'}")
        if hasattr(seq, "__len__") and len(seq) > 0:
            f0 = seq[0]
            log(f"frame[0] type={type(f0).__name__}, "
                f"size={getattr(f0, 'size', None)}")
    except Exception as exc:
        log(f"frame introspection failed: {exc}")

    out_path = PROBE_OUT / "probe.mp4"
    try:
        from diffusers.utils import export_to_video

        export_to_video(frames[0] if isinstance(frames, (list, tuple)) else frames,
                        str(out_path), fps=8)
        log(f"WROTE {out_path} (size={out_path.stat().st_size} bytes)")
    except Exception as exc:
        log(f"export_to_video failed: {exc}")
        traceback.print_exc()


def main() -> int:
    probe_env()
    probe_model_card()
    pipe = probe_load_pipeline()
    probe_pipeline_signature(pipe)
    pipe = probe_move_to_mps(pipe)
    probe_minimal_inference(pipe)
    log("PROBE PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[probe] FATAL: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)
