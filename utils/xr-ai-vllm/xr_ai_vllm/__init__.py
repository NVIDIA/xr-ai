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
import urllib.request
from pathlib import Path

from . import _docker, _pip

log = logging.getLogger(__name__)

# Default NGC image. Override per-server via `vllm_image:` in YAML.
DEFAULT_IMAGE = "nvcr.io/nvidia/vllm:26.04-py3"


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
    ready_file: Path | None = None,
) -> None:
    """Launch vLLM via *backend* (`"pip"` or `"docker"`).

    *extra_serve_args* are the flags appended after `vllm serve <model>` —
    everything past the model id (e.g. ``--served-model-name``,
    ``--max-num-seqs``, ``--reasoning-parser``, …). They are passed verbatim,
    so caller-side flag construction is unchanged from the per-service
    wrappers' previous inline argv.

    *persistent* picks the lifecycle:

    * ``True``  — survive wrapper restarts (vLLM in a new session group / a
      detached docker container). Cleanup is via `stop_persistent_servers`.
    * ``False`` — die with the wrapper. SIGTERM on the wrapper takes vLLM
      with it.

    *container_name* is only consulted in docker mode. Use a stable,
    service-specific name (e.g. ``xr-ai-vllm-<entry-point>``) so the stop
    helper can find it.
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
            persistent=persistent,
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
            ready_file=ready_file,
        )
    else:
        raise ValueError(
            f"unknown vllm_backend: {backend!r} (expected 'pip' or 'docker')"
        )


def stop_persistent_servers(
    services: list[tuple[str, int, str | None]],
) -> None:
    """Stop persisted vLLM servers; safe to call when nothing is running.

    *services* is a list of ``(label, port, container_name | None)`` tuples.
    For each entry:

    1. Probe ``http://127.0.0.1:<port>/health``. Skip if not reachable.
    2. If *container_name* is given and that docker container exists,
       ``docker stop`` it (escalates to ``docker kill`` after 20 s).
    3. Otherwise (or if docker stop reports no such container), fall back
       to the port → pid → SIGTERM → SIGKILL path used for pip-mode vLLM.

    Output is print-style with `[<label>] …` prefixes so it matches the
    existing `--stop` UX.
    """
    import signal
    import time

    found = False
    for label, port, container_name in services:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as r:
                if r.status != 200:
                    continue
        except Exception:
            continue

        found = True

        if container_name and _docker.container_exists(container_name):
            print(f"  [{label}] stopping container {container_name} (port={port})…",
                  flush=True)
            if _docker.stop_container(container_name):
                print(f"  [{label}] stopped", flush=True)
            else:
                print(f"  [{label}] docker stop failed — check `docker ps -a`",
                      flush=True)
            continue

        pid = _docker.pid_on_port(port)
        if pid is None:
            print(f"  [{label}] running on :{port} but no PID found — "
                  f"kill manually", flush=True)
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
        except ProcessLookupError:
            print(f"  [{label}] already gone", flush=True)

    if not found:
        print("  No persistent vLLM servers found running.", flush=True)


__all__ = ["serve", "stop_persistent_servers", "DEFAULT_IMAGE"]
