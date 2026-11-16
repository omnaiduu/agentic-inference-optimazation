"""Shared Modal setup: model id, GPU choice, images, and the HF weight cache."""
import os

import modal

MODEL_ID = "google/gemma-4-E2B-it"
# Pin the exact weights so every experiment uses the same model files.
MODEL_REVISION = os.environ.get("AIO_MODEL_REVISION", "main")

# Default GPU: L40S (~$1.95/h, Ada w/ FP8, 48 GB) - L4 was often unavailable. Override: AIO_GPU=H100 modal run ...
GPU = os.environ.get("AIO_GPU", "L40S")

VLLM_VERSION = "0.33.0"

HF_HOME = "/hf"
hf_cache = modal.Volume.from_name("aio-hf-cache", create_if_missing=True)

_env = {"HF_HOME": HF_HOME, "HF_XET_HIGH_PERFORMANCE": "1"}

# Light image: only for reading model metadata on CPU.
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub[hf_xet]==2.4.0")
    .env(_env)
    .add_local_python_source("aio")
)

# GPU image, following Modal's own vLLM guide (modal.com/docs/examples/vllm_inference):
# NVIDIA's CUDA *devel* image (includes nvcc, which vLLM's FlashInfer sampler
# JIT-compiles with) + Modal's standalone Python + pip-installed vLLM.
# CUDA 13.0.3 matches the official vLLM 0.33.0 release build.
# The same image runs our own engine, so both sides use identical torch/CUDA.
gpu_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_xet]", "aiohttp")
    .env(_env)
    .add_local_python_source("aio")
)
