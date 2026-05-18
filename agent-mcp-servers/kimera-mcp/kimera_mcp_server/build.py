# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the ``kimera_vio`` docker image on demand.

Both stages of the build are bootstrapped automatically the first time
the MCP server starts:

1. ``kimera_vio_deps`` — Kimera-VIO's own ``Dockerfile_20_04``, which
   compiles GTSAM, OpenGV, DBoW2, Kimera-RPGO into an Ubuntu 20.04
   base.  Needs the Kimera-VIO source tree on the host as the docker
   build context.

2. ``kimera_vio`` — our overlay (``scripts/Dockerfile.kimera``) that
   builds Kimera-VIO itself plus the ``kimera_live_vio`` streaming
   wrapper.  Build context is ``scripts/`` so the wrapper source ships
   in alongside it.

Both builds stream output to the host stderr so the operator can watch
progress.  Once the image exists, subsequent server starts skip
straight to ``docker run``.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

from loguru import logger


# Default upstream URL for the Kimera-VIO source tree.  Overridable
# via the ``kimera_vio_repo`` yaml key for offline / forked builds.
DEFAULT_REPO = "https://github.com/MIT-SPARK/Kimera-VIO.git"

def _scripts_dir() -> pathlib.Path:
    """Locate the docker build context bundled with the package.

    Two layouts are supported so the same code path works in both
    editable installs (``uv sync`` from this repo) and wheel installs
    (where ``scripts/`` is force-included as ``kimera_mcp_server/_scripts/``)."""
    here = pathlib.Path(__file__).resolve().parent
    # Wheel layout: kimera_mcp_server/_scripts/
    wheel_dir = here / "_scripts"
    if (wheel_dir / "Dockerfile.kimera").exists():
        return wheel_dir
    # Editable layout: ../scripts/ next to the package directory.
    editable_dir = here.parent / "scripts"
    if (editable_dir / "Dockerfile.kimera").exists():
        return editable_dir
    raise RuntimeError(
        "could not locate kimera-mcp scripts/ directory next to package "
        f"(checked {wheel_dir} and {editable_dir})"
    )


def _docker_available() -> None:
    if shutil.which("docker") is None:
        raise RuntimeError(
            "kimera-mcp needs `docker` on PATH to build / run the "
            "Kimera-VIO container.  Install Docker Engine first."
        )


def _image_exists(name: str) -> bool:
    out = subprocess.run(
        ["docker", "images", "-q", name],
        check=False, capture_output=True, timeout=10,
    )
    return bool(out.stdout.strip())


def _ensure_kimera_source(src_cache: pathlib.Path,
                          repo_url: str) -> pathlib.Path:
    """Clone Kimera-VIO into ``src_cache`` if it isn't already there.
    Returns the path that should be used as the docker build context
    for the deps image."""
    if shutil.which("git") is None:
        raise RuntimeError(
            "kimera-mcp needs `git` on PATH to fetch Kimera-VIO source. "
            "Install git or pre-populate {} yourself.".format(src_cache)
        )
    if (src_cache / "Dockerfile_20_04").exists():
        return src_cache
    src_cache.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[kimera-build] cloning {} → {}", repo_url, src_cache)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(src_cache)],
        check=True, timeout=300,
    )
    return src_cache


def _docker_build(context: pathlib.Path, dockerfile: pathlib.Path,
                  tag: str) -> None:
    """Run ``docker build`` and stream its output to stderr."""
    cmd = ["docker", "build",
           "-f", str(dockerfile),
           "-t", tag,
           str(context)]
    logger.info("[kimera-build] {}", " ".join(cmd))
    # Don't capture — we want the build progress to land in the
    # operator's terminal so a 30-minute deps build isn't silent.
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        raise RuntimeError(
            f"docker build failed (rc={rc}) for tag {tag!r}.  "
            "See the docker build output above for the error."
        )


def ensure_image(
    *,
    image_tag:     str           = "kimera_vio",
    deps_tag:      str           = "kimera_vio_deps",
    src_cache:     pathlib.Path  = pathlib.Path("~/.cache/xr-ai/Kimera-VIO"),
    repo_url:      str           = DEFAULT_REPO,
) -> None:
    """Build ``kimera_vio_deps`` and ``kimera_vio`` if they don't
    already exist.  Idempotent: subsequent calls are no-ops once the
    image is on the host."""
    _docker_available()
    if _image_exists(image_tag):
        return

    src_cache = pathlib.Path(src_cache).expanduser()

    if not _image_exists(deps_tag):
        logger.warning(
            "[kimera-build] {} image missing — building from source. "
            "First-time build can take 30+ minutes.", deps_tag,
        )
        src = _ensure_kimera_source(src_cache, repo_url)
        _docker_build(
            context=src,
            dockerfile=src / "Dockerfile_20_04",
            tag=deps_tag,
        )

    logger.warning(
        "[kimera-build] {} image missing — building overlay.", image_tag,
    )
    scripts_dir = _scripts_dir()
    _docker_build(
        context=scripts_dir,
        dockerfile=scripts_dir / "Dockerfile.kimera",
        tag=image_tag,
    )
    logger.info("[kimera-build] images ready: {} (deps: {})",
                image_tag, deps_tag)
