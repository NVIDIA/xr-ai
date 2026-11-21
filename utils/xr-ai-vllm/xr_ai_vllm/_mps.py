# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA MPS client setup for the vLLM Docker backend."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from xr_ai_launcher import CUDA_MPS_ENV_KEYS, cuda_mps_identity

log = logging.getLogger(__name__)



def docker_options(
    env: dict[str, str], *, model_cache: Path, extra_pip: list[str] | None,
) -> tuple[list[str], dict]:
    """Normalize *env* in place, probe MPS, and return Docker options and identity."""
    raw = env.get("CUDA_MPS_PIPE_DIRECTORY", "")
    if not raw:
        for key in CUDA_MPS_ENV_KEYS:
            env.pop(key, None)
        return [], {}
    if extra_pip:
        raise ValueError("extra_pip is not supported with CUDA MPS; install packages in the image")
    effective = cuda_mps_identity(env)
    pipe = effective["pipe_directory"]
    uid, gid = os.getuid(), os.getgid()
    env["CUDA_MPS_PIPE_DIRECTORY"] = pipe
    for key, field in (
        ("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "active_thread_percentage"),
        ("CUDA_MPS_CLIENT_PRIORITY", "client_priority"),
    ):
        if key in env:
            env[key] = str(effective[field])
    for directory in (model_cache, model_cache / "hub", model_cache / "xet"):
        if directory.exists() and not os.access(directory, os.W_OK | os.X_OK):
            log.warning(
                "CUDA MPS UID %s cannot write model cache %s. Cached weights may load, "
                "but downloads can fail. Ask the cache owner to grant write access "
                "(for example, correct ownership with chown), or select a writable cache. "
                "This check is advisory; nested permissions and concurrent writers can differ.",
                uid, directory,
            )
    env.setdefault("HOME", f"/tmp/xr-ai-mps-{uid}")
    env.setdefault("XDG_CACHE_HOME", env["HOME"] + "/.cache")
    env.setdefault("HF_MODULES_CACHE", env["HOME"] + "/hf-modules")
    return ["--user", f"{uid}:{gid}", "-v", f"{pipe}:{pipe}"], {
        "user": f"{uid}:{gid}", "controller": effective["controller"],
    }
