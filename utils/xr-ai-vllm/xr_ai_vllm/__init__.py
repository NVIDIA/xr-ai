# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
xr-ai-vllm — pluggable vLLM backend for xr-ai inference services.

Lets each vLLM-backed service host vllm via either:

* `pip`    — pip-installed `vllm` CLI in the wrapper's venv (default; today's behavior).
* `docker` — `docker run nvcr.io/nvidia/vllm:<tag> vllm serve …` (NGC container).

The choice is per-server, set via `vllm_backend: pip|docker` in the service's
YAML. Both paths honor identical config keys (model, ports, vllm flags); only
the runtime hosting vllm differs.

Stdlib-only by contract — no vllm or other heavy deps imported here, so the
docker path stays light even when pip vllm is not installed.

Typical usage from a service wrapper::

    from xr_ai_vllm import serve, DEFAULT_IMAGE

    serve(
        backend=cfg.get("vllm_backend", "pip"),
        persistent=True,
        image=cfg.get("vllm_image", DEFAULT_IMAGE),
        container_name="xr-ai-vllm-vlm-server",
        log_prefix="vlm_server",
        model=model,
        extra_serve_args=[
            "--served-model-name", served_name,
            "--max-num-seqs", str(max_seqs),
            ...
        ],
        host=host, port=port,
        model_cache=model_cache,
        hf_token=os.environ.get("HF_TOKEN"),
        cuda_visible_devices=cfg.get("cuda_visible_devices"),
        ready_file=ns.ready_file,
    )
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from . import _docker, _pip
from ._config import (
    gpu_compute_major,
    load_config,
    resolve_model_cache,
    setup_hf_env,
)
from ._config import (
    gpu_memory_utilization as gpu_memory_utilization,
)
from ._nim import serve_nim

log = logging.getLogger(__name__)

DEFAULT_IMAGE = "nvcr.io/nvidia/vllm:26.04-py3"
"""Default NGC vLLM image used when a service does not override ``vllm_image``.

Individual services may pin a newer image when required by their model.
"""


def serve(
    *,
    backend: str,
    persistent: bool,
    image: str = DEFAULT_IMAGE,
    container_name: str,
    log_prefix: str,
    model: str,
    extra_serve_args: list[str],
    host: str,
    port: int,
    model_cache: Path,
    hf_token: str | None = None,
    cuda_visible_devices: str | None = None,
    extra_env: dict[str, str] | None = None,
    extra_pip: list[str] | None = None,
    ready_file: Path | None = None,
) -> None:
    """Launch vLLM via *backend* (`"pip"` or `"docker"`).

    *extra_serve_args* are the flags appended after `vllm serve <model>` —
    everything past the model id (e.g. ``--served-model-name``,
    ``--max-num-seqs``, ``--reasoning-parser``, …). They are passed verbatim,
    so caller-side flag construction is unchanged from the per-service
    wrappers' previous inline argv.

    *persistent* controls the pip-mode lifecycle only:

    * ``True``  — vLLM pip process starts in a new session so it survives
      wrapper restarts.  Cleanup is via `stop_persistent_servers`.
    * ``False`` — die with the wrapper.

    For the docker backend *persistent* is ignored: the container always
    runs foreground with ``start_new_session=True``, so it escapes the
    launcher's process group regardless. Use
    ``Process(..., launch_mode="persist")`` in the orchestrator ``main.py``
    to tell the launcher not to kill the wrapper on shutdown.

    *container_name* is only consulted in docker mode. Use a stable,
    service-specific name (e.g. ``xr-ai-vllm-<entry-point>``) so the stop
    helper can find it.

    *extra_pip* is docker-mode only: a list of pip-installable package
    specs that get installed into the container right before ``vllm
    serve`` runs (same shell line that already installs ``hf_transfer``).
    Use it for models whose architecture imports a wheel the NGC image
    doesn't bundle — e.g. ``["mamba-ssm", "causal-conv1d"]`` for
    Nemotron-Omni's hybrid SSM backbone. Silently ignored in pip mode
    (deps belong in the wrapper's pyproject.toml there).
    """
    vllm_argv: list[str] = [
        "vllm", "serve", model,
        "--host", host,
        "--port", str(port),
    ]
    vllm_argv += list(extra_serve_args)

    if backend == "pip":
        _pip.run(
            persistent=persistent,
            log_prefix=log_prefix,
            vllm_argv=vllm_argv,
            host=host,
            port=port,
            ready_file=ready_file,
        )
    elif backend == "docker":
        _docker.run(
            image=image,
            container_name=container_name,
            log_prefix=log_prefix,
            vllm_argv=vllm_argv,
            host=host,
            port=port,
            model_cache=model_cache,
            hf_token=hf_token,
            cuda_visible_devices=cuda_visible_devices,
            extra_env=extra_env,
            extra_pip=extra_pip,
            ready_file=ready_file,
        )
    else:
        raise ValueError(
            f"unknown vllm_backend: {backend!r} (expected 'pip' or 'docker')"
        )


def stop_persistent_servers(
    services: list[tuple[str, int]],
) -> bool:
    """Stop persisted servers and report whether every discovered server stopped.

    *services* is a list of ``(label, port)`` tuples.  For each entry:

    1. Look for a docker container labelled ``xr-ai-vllm.port=<port>``
       (stamped at start time by the vLLM wrapper) and ``docker stop`` it.
    2. Fall back to port → pid → SIGTERM → SIGKILL for pip-mode vLLM or
       in-process servers (e.g. STT).

    A missing container and listener is already stopped. Output is print-style
    with ``[<label>] …`` prefixes. Discovery errors and listeners that are not
    identified as xr-ai services fail closed without sending a signal.
    """
    import signal
    import time

    success = True
    found = False
    for label, port in services:
        container_name, container_checked = _docker.container_on_port_checked(port)
        if not container_checked:
            print(f"  [{label}] cannot inspect :{port} ownership — not stopping", flush=True)
            success = False
            continue

        if container_name:
            found = True
            print(f"  [{label}] stopping container {container_name}…", flush=True)
            if _docker.stop_container(container_name):
                # Remove the container so the next launch goes through a full
                # `docker run` and picks up YAML config changes (without removal,
                # `docker start` reuses the old container with its baked-in
                # argv — stale --limit-mm-per-prompt, extra_pip, etc.).
                #
                # Tradeoff acknowledged: `_docker.run`'s start-on-existing path
                # skipped `pip install` (hf_transfer, plus any extra_pip like
                # mamba-ssm / causal-conv1d for Nemotron-Omni). Forcing rm means
                # those reinstall on every restart — a few-second extra cost on
                # warm cache, in exchange for config edits actually applying.
                # Acceptable: persistent-servers exists to skip MODEL reloads,
                # not pip reinstalls.
                if _docker.remove_container(container_name):
                    print(f"  [{label}] stopped and removed", flush=True)
                else:
                    print(f"  [{label}] stopped (rm failed — "
                          f"run `docker rm {container_name}` to apply config changes)",
                          flush=True)
            else:
                print(f"  [{label}] docker stop failed — check `docker ps -a`",
                      flush=True)
                success = False
            continue

        pid, pid_checked, listening = _docker.pid_on_port_checked(port)
        if not pid_checked or (listening and pid is None):
            print(f"  [{label}] cannot inspect :{port} ownership — not stopping", flush=True)
            success = False
            continue
        if not listening:
            continue

        assert pid is not None
        found = True
        if not _docker.is_xr_ai_server_process(pid, label, port):
            print(f"  [{label}] listener on :{port} is not an xr-ai server — not stopping",
                  flush=True)
            success = False
            continue

        print(f"  [{label}] stopping (pid={pid}, port={port})…", flush=True)
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(40):
                time.sleep(0.5)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    print(f"  [{label}] stopped", flush=True)
                    break
            else:
                print(f"  [{label}] force-killing", flush=True)
                os.kill(pid, signal.SIGKILL)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    print(f"  [{label}] stopped", flush=True)
                else:
                    print(f"  [{label}] still running after SIGKILL", flush=True)
                    success = False
        except ProcessLookupError:
            print(f"  [{label}] already gone", flush=True)

    if not found:
        print("  No persistent servers found running.", flush=True)

    return success


__all__ = [
    "serve",
    "serve_nim",
    "stop_persistent_servers",
    "DEFAULT_IMAGE",
    "resolve_model_cache",
    "load_config",
    "setup_hf_env",
    "gpu_compute_major",
]
