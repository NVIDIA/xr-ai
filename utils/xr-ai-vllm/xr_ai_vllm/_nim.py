# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIM container backend: self-hosted NVIDIA NIM microservices.

A NIM image (``nvcr.io/nim/<org>/<model>``) bakes in its own entrypoint:
on first start it authenticates to NGC with ``NGC_API_KEY``, downloads the
GPU-matched optimized engine into ``NIM_CACHE_PATH``, and serves. LLM/VLM
NIMs expose the OpenAI-compatible API on ``NIM_HTTP_API_PORT``; Riva speech
NIMs additionally serve gRPC on ``NIM_GRPC_API_PORT``. All expose readiness
at ``/v1/health/ready`` on the HTTP port, which gates startup here.

Container lifecycle (reuse, restart, signals, logs) is shared with the vLLM
docker backend via :func:`._docker.run_container`.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from . import _docker

log = logging.getLogger(__name__)


def build_nim_run_argv(
    *,
    image: str,
    container_name: str,
    http_port: int,
    grpc_port: int | None,
    nim_cache: Path,
    cuda_visible_devices: str | None,
    extra_env: dict[str, str] | None,
) -> list[str]:
    """Build the foreground ``docker run …`` argv for a NIM container.

    No command override: the NIM entrypoint is the server. Unlike the vLLM
    argv builder, this uses bridge networking with explicit ``-p`` maps (see
    the inline comment); the port label and ``--runtime nvidia`` choices do
    mirror it. ``NGC_API_KEY`` is passed by name only, so the value stays off
    the ps-visible argv; docker reads it from the wrapper's environment.
    """
    argv: list[str] = ["docker", "run"]
    argv += ["--name", container_name]
    argv += ["--label", f"xr-ai-vllm.port={http_port}"]
    # Bridge networking with explicit -p maps to each family's documented
    # internal defaults. Env-var port overrides (NIM_HTTP_API_PORT) are
    # honored inconsistently across NIM images, and host networking makes the
    # image's internal default collide with whatever already owns that host
    # port.
    if grpc_port is not None:
        # Riva speech NIM: gRPC on 50051, HTTP (health) on 9000.
        argv += ["-p", f"{grpc_port}:50051", "-p", f"{http_port}:9000"]
    else:
        # LLM/VLM NIM: OpenAI API on 8000.
        argv += ["-p", f"{http_port}:8000"]
    argv += ["--ipc", "host"]
    argv += ["--runtime", "nvidia"]
    argv += ["-e", f"NVIDIA_VISIBLE_DEVICES={cuda_visible_devices or 'all'}"]
    argv += ["-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility"]

    argv += ["-e", "NGC_API_KEY"]

    env_vars: dict[str, str] = {
        "NIM_CACHE_PATH": "/opt/nim/.cache",
    }
    if extra_env:
        env_vars.update(extra_env)
    for key, val in env_vars.items():
        argv += ["-e", f"{key}={val}"]

    argv += ["-v", f"{nim_cache}:/opt/nim/.cache"]
    # No -u override: NIM images differ in their baked-in user. Riva speech
    # NIMs run as root and write image-internal paths (/opt/nim/workspace)
    # that a forced host uid cannot (Permission denied at startup); LLM/VLM
    # NIMs run as uid 1000. Cache files may show as root-owned on the host.

    argv.append(image)
    return argv


def serve_nim(
    *,
    image: str,
    container_name: str,
    log_prefix: str,
    http_port: int,
    grpc_port: int | None = None,
    nim_cache: Path,
    cuda_visible_devices: str | None = None,
    extra_env: dict[str, str] | None = None,
    ready_file: Path | None = None,
) -> None:
    """Pull (if needed) and run a NIM container, blocking until stopped.

    Readiness is ``/v1/health/ready`` on *http_port*. First start includes
    the NGC engine download (multi-GB), so expect a long cold start; the
    mounted *nim_cache* makes subsequent starts fast.
    """
    ngc_api_key = os.environ.get("NGC_API_KEY", "").strip()
    if not ngc_api_key:
        log.error(
            "NIM containers require NGC_API_KEY (nvcr.io pull + engine "
            "download); get one at https://ngc.nvidia.com/setup/api-key"
        )
        sys.exit(1)
    # The argv passes NGC_API_KEY by name only; docker resolves it from the
    # environment this process hands the `docker run` subprocess.
    os.environ["NGC_API_KEY"] = ngc_api_key

    # Per-container, world-writable cache subdir: images run as different
    # baked-in users (see the -u rationale in build_nim_run_argv), and in a
    # shared dir one image's root-owned subtree blocks the next image's
    # non-root writes.
    nim_cache = nim_cache / container_name
    nim_cache.mkdir(parents=True, exist_ok=True)
    nim_cache.chmod(0o777)
    endpoint = (
        f"grpc localhost:{grpc_port} (health http:{http_port})"
        if grpc_port is not None else f"http://localhost:{http_port}/v1"
    )
    argv = build_nim_run_argv(
        image=image,
        container_name=container_name,
        http_port=http_port,
        grpc_port=grpc_port,
        nim_cache=nim_cache,
        cuda_visible_devices=cuda_visible_devices,
        extra_env=extra_env,
    )
    _docker.run_container(
        argv=argv,
        image=image,
        container_name=container_name,
        log_prefix=log_prefix,
        port=http_port,
        health_url=f"http://127.0.0.1:{http_port}/v1/health/ready",
        launch_banner=(
            f"Launching NIM  image={image}  container={container_name}  "
            f"{endpoint} (first start downloads the optimized engine "
            f"from NGC, multi-GB)"
        ),
        reuse_banner=f"NIM already serving ({endpoint}), reusing",
        ready_banner=f"Ready  →  {endpoint}  (docker: {container_name})",
        ready_file=ready_file,
    )
