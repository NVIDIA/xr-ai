# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
xr-ai-nemo-runtime — opt-in NGC NeMo container backend for the in-process
NeMo servers (stt-server, magpie TTS).

The NeMo servers load torch+NeMo in the wrapper's venv and inherit the host's
cuDNN/CUDA via LD_LIBRARY_PATH, which aborts at torch import on hosts whose
system cuDNN differs from torch's bundled one. This runs the same FastAPI
server inside an NGC NeMo image instead, so torch+nemo+cuDNN all come from the
container and the host's libraries are irrelevant.

Opt in per-server via `backend: docker` in the service YAML (default `pip` =
today's in-venv behavior, unchanged). This is a BESPOKE NeMo docker path,
deliberately separate from `xr_ai_vllm` (the vLLM `vllm serve` binary differs
from our `python -m <server>` entry, and the in-container packaging differs).

Stdlib-only by contract — no torch/nemo/fastapi imported here.

Typical usage from a NeMo server's ``run()``::

    from xr_ai_nemo_runtime import run_nemo_docker, DEFAULT_IMAGE

    run_nemo_docker(
        image=cfg.get("nemo_image", DEFAULT_IMAGE),
        container_name="xr-ai-nemo-stt-server",
        log_prefix="stt_server",
        server_module="stt_server",
        server_pkg_dir=Path(__file__).resolve().parent.parent,
        config_path=ns.config.resolve(),
        model_cache=model_cache,
        nemo_cache_dir=model_cache / "nemo",
        host=host, port=port,
        hf_token=os.environ.get("HF_TOKEN"),
        cuda_visible_devices=cfg.get("cuda_visible_devices"),
        extra_pip=["python-multipart"],
        ready_file=ns.ready_file,
    )
"""
from __future__ import annotations

from pathlib import Path

from . import _docker

# Example NGC NeMo image. Override per-server via `nemo_image:` in YAML.
# NOTE: confirm a tag available to your NGC account — image tags roll over and
# this default may not be pullable. `docker pull <tag>` before relying on it.
DEFAULT_IMAGE = "nvcr.io/nvidia/nemo:25.04"


def run_nemo_docker(
    *,
    image: str = DEFAULT_IMAGE,
    container_name: str,
    log_prefix: str,
    server_module: str,
    server_pkg_dir: Path,
    config_path: Path,
    model_cache: Path,
    nemo_cache_dir: Path | None = None,
    host: str,
    port: int,
    hf_token: str | None = None,
    cuda_visible_devices: str | None = None,
    extra_pip: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    ready_file: Path | None = None,
) -> None:
    """Run a NeMo FastAPI server inside an NGC NeMo container.

    *server_module* is the module run via ``python -m <module> --_serve`` in
    the container (e.g. ``"stt_server"``). The ``--_serve`` flag must
    short-circuit the server's own backend dispatch so the container does not
    re-spawn itself.

    *server_pkg_dir* is the directory that CONTAINS the server package (so it
    goes on PYTHONPATH). It must live under the repo tree that gets
    bind-mounted; the repo root is derived from it.

    *config_path* must be inside the bind-mounted repo (the server reads it
    in-container). *model_cache* is mounted read-write so weight downloads
    persist on the host. *nemo_cache_dir* sets ``NEMO_CACHE_DIR`` in the
    container (stt sets this today).

    *extra_pip* are the light deps the server needs that the NeMo image lacks
    beyond the shared base (fastapi/uvicorn/hf_transfer/loguru/pyyaml) — e.g.
    ``["python-multipart"]`` for stt, ``["soundfile", "numpy"]`` for magpie.
    Never the server's own pyproject: that drags nemo_toolkit/torch and
    conflicts with the image's.
    """
    repo_root = _find_repo_root(server_pkg_dir)
    _docker.run(
        image=image,
        container_name=container_name,
        log_prefix=log_prefix,
        server_module=server_module,
        repo_root=repo_root,
        server_pkg_dir=server_pkg_dir,
        config_path=config_path,
        model_cache=model_cache,
        nemo_cache_dir=nemo_cache_dir,
        host=host,
        port=port,
        hf_token=hf_token,
        cuda_visible_devices=cuda_visible_devices,
        extra_pip=extra_pip,
        extra_env=extra_env,
        ready_file=ready_file,
    )


def _find_repo_root(start: Path) -> Path:
    """Walk up from *start* to the repo root (the dir holding AGENTS.md).

    The whole repo is bind-mounted so both the server package and
    ``utils/xr-ai-logging`` are visible in the container. Falls back to the
    filesystem root's child if no marker is found (mount is still valid; only
    the xr-ai-logging path would be wrong, which surfaces immediately).
    """
    start = start.resolve()
    for parent in [start, *start.parents]:
        if (parent / "AGENTS.md").exists():
            return parent
    return start.parents[-1]


__all__ = ["run_nemo_docker", "DEFAULT_IMAGE", "_docker"]
