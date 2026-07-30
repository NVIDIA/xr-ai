# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
nim_server: generic launcher for a self-hosted NVIDIA NIM container.

One command serves any NIM image; the YAML picks which. Orchestrators list
one ``Process`` row per NIM with a distinct ``config=`` (the launcher's
usual pattern for shared commands). Dispatches through
``xr_ai_vllm.serve_nim``: NGC login, image pull, engine download into the
cache volume, and readiness on ``/v1/health/ready`` all happen there.

Accepts ``--config <path>.yaml`` (auto-passed by xr-ai-launcher).

Config keys
-----------
    image:                 str   NIM image, e.g. nvcr.io/nim/meta/llama-3.1-8b-instruct:latest (required).
    http_port:             int   HTTP port: OpenAI API for LLM/VLM NIMs,
                                 health-only for speech NIMs (required).
    grpc_port:             int   gRPC port (Riva speech NIMs only).
    container_name:        str   Docker name (default: xr-ai-nim-<image basename>).
    nim_cache:             str   Engine/weights cache volume, relative to this
                                 YAML (default: ../../models/nim).
    cuda_visible_devices:  str   GPU filter for this container.
    env:                   map   Extra NIM_* env vars passed verbatim.

Requires ``NGC_API_KEY`` (nvcr.io pull + engine download).
"""
import re
import sys
from pathlib import Path

from loguru import logger
from xr_ai_logging import setup_logging
from xr_ai_vllm import load_config, serve_nim

_DEFAULT_NIM_CACHE = "../../models/nim"


def _default_container_name(image: str) -> str:
    base = image.rsplit("/", 1)[-1].split(":", 1)[0]
    return "xr-ai-nim-" + re.sub(r"[^a-zA-Z0-9_.-]", "-", base)


def run() -> None:
    setup_logging("nim-server")

    cfg, yaml_dir, ready_file = load_config()

    if not cfg.get("image"):
        logger.error("'image' is required in config")
        sys.exit(1)
    if not cfg.get("http_port"):
        logger.error("'http_port' is required in config")
        sys.exit(1)

    image          = str(cfg["image"])
    http_port      = int(cfg["http_port"])
    grpc_port      = int(cfg["grpc_port"]) if cfg.get("grpc_port") else None
    container_name = cfg.get("container_name") or _default_container_name(image)
    cuda_devices   = cfg.get("cuda_visible_devices")

    nim_cache = Path(cfg.get("nim_cache", _DEFAULT_NIM_CACHE))
    if not nim_cache.is_absolute():
        nim_cache = (yaml_dir / nim_cache).resolve()

    serve_nim(
        image=image,
        container_name=container_name,
        log_prefix=container_name.removeprefix("xr-ai-nim-"),
        http_port=http_port,
        grpc_port=grpc_port,
        nim_cache=nim_cache,
        cuda_visible_devices=str(cuda_devices) if cuda_devices is not None else None,
        extra_env={str(k): str(v) for k, v in (cfg.get("env") or {}).items()},
        ready_file=ready_file,
    )


if __name__ == "__main__":
    run()
