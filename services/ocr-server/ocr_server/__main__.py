# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and launch the isolated Hugging Face OCR inference container."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from xr_ai_logging import setup_logging
from xr_ai_vllm import _docker, load_config, resolve_model_cache


def run() -> None:
    setup_logging("ocr-server")
    cfg, yaml_dir, ready_file = load_config()
    port = int(cfg.get("port", 8112))
    length = int(cfg.get("infer_length", 1024))
    if not 1 <= port <= 65535 or length not in (640, 1024):
        raise ValueError("port must be 1..65535; infer_length must be 640 or 1024")
    arch = str(cfg["cuda_arch"])
    if arch not in {"8.9", "12.0", "12.1"}:
        raise ValueError("cuda_arch must be 8.9 (Ada), 12.0 (Blackwell), or 12.1 (Spark)")
    image = f"xr-ai-nemotron-ocr:sm{arch.replace('.', '')}"
    context = Path(__file__).resolve().parent
    # BuildKit attestations contain timestamps and otherwise change the local
    # image ID on every cached build, defeating persistent-container reuse.
    subprocess.run([
        "docker", "build", "--provenance=false", "--build-arg", f"TORCH_CUDA_ARCH_LIST={arch}",
        "-t", image, str(context),
    ], check=True)
    # A source or dependency change rebuilds the image and invalidates reuse.
    image_id = subprocess.check_output([
        "docker", "image", "inspect", "--format", "{{.Id}}", image,
    ], text=True).strip()
    cache = resolve_model_cache(cfg, yaml_dir, default="../../models/ocr")
    name = str(cfg.get("container_name", "xr-ai-ocr"))
    gpu = str(cfg.get("cuda_visible_devices", "0"))
    fingerprint = _docker.launch_fingerprint({
        "image": image_id, "port": port, "cache": str(cache), "gpu": gpu,
        "infer_length": length,
        "hf_token_digest": _docker.credential_digest(os.environ.get("HF_TOKEN")),
    })
    argv = [
        "docker", "run", "--name", name,
        "--label", f"xr-ai-vllm.port={port}",
        "--label", f"{_docker._CONFIG_LABEL}={fingerprint}",
        "--runtime", "nvidia", "--ipc", "host",
        "-e", f"NVIDIA_VISIBLE_DEVICES={gpu}",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "-e", "HF_TOKEN", "-e", f"OCR_INFER_LENGTH={length}",
        "-p", f"{port}:8000", "-v", f"{cache}:/models", image,
    ]
    _docker.run_container(
        argv=argv, image=image, container_name=name, log_prefix="ocr",
        port=port, health_url=f"http://127.0.0.1:{port}/v1/health/ready",
        launch_banner=f"Loading multilingual Hugging Face OCR on GPU {gpu}",
        reuse_banner=f"OCR already serving on :{port}",
        ready_banner=f"OCR ready on :{port}", ready_file=ready_file,
    )


if __name__ == "__main__":
    run()
