# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install DROID-SLAM on demand.

``droid_slam`` isn't on PyPI — the upstream Princeton repo ships C++/
CUDA extensions that have to be compiled against the active PyTorch.
On first server start, if ``import droid_slam`` fails, we shell out to
``scripts/setup_droid.sh`` which:

1. Clones princeton-vl/DROID-SLAM into ``~/.cache/xr-ai/DROID-SLAM``.
2. Runs ``python setup.py install`` inside the clone — compiles CUDA
   extensions against the current venv's torch and installs the
   ``droid_slam`` package *into the active venv* (not system Python).
3. Downloads the pretrained checkpoint (~250 MB) to
   ``~/.cache/xr-ai/droid.pth``.

Mirrors the auto-build pattern that ``kimera_mcp_server.build`` uses
for the kimera docker image so first-run ``uv run slam_example`` works
without any operator-side install step on a CUDA host.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import pathlib
import shutil
import subprocess

from loguru import logger


def _scripts_dir() -> pathlib.Path:
    """Locate the setup script bundled with the package.

    Two layouts are supported so the same code path works in both
    editable installs (``uv sync`` from this repo) and wheel installs
    (where ``scripts/`` is force-included as ``droid_mcp_server/_scripts/``)."""
    here = pathlib.Path(__file__).resolve().parent
    wheel_dir = here / "_scripts"
    if (wheel_dir / "setup_droid.sh").exists():
        return wheel_dir
    editable_dir = here.parent / "scripts"
    if (editable_dir / "setup_droid.sh").exists():
        return editable_dir
    raise RuntimeError(
        "could not locate droid-mcp scripts/ directory next to package "
        f"(checked {wheel_dir} and {editable_dir})"
    )


def _droid_importable() -> bool:
    """True iff ``import droid_slam`` would succeed in this venv."""
    return importlib.util.find_spec("droid_slam") is not None


def ensure_droid_installed() -> None:
    """If ``droid_slam`` isn't importable, run ``setup_droid.sh`` to
    clone, compile, and install it.  Idempotent — subsequent calls are
    no-ops once the package is in the venv."""
    if _droid_importable():
        return
    if shutil.which("bash") is None:
        raise RuntimeError(
            "droid-mcp needs `bash` on PATH to run setup_droid.sh.  "
            "Install bash or pre-install droid_slam yourself.",
        )

    scripts_dir = _scripts_dir()
    script = scripts_dir / "setup_droid.sh"
    logger.warning(
        "[droid-build] droid_slam not installed — running setup_droid.sh.  "
        "This clones princeton-vl/DROID-SLAM, compiles its CUDA extensions "
        "against the active torch, and downloads ~250 MB of weights.  "
        "Needs nvcc + g++ on PATH; can take 5-10 minutes on first run.",
    )

    env = os.environ.copy()
    # Make sure setup.py finds the active venv's python (which is the
    # same python this MCP server is running under).  Without VIRTUAL_ENV
    # / PATH already set by `uv run`, this would fall back to system
    # python and install into the wrong place.
    rc = subprocess.run(
        ["bash", str(script)],
        check=False, env=env,
    ).returncode
    if rc != 0:
        raise RuntimeError(
            f"setup_droid.sh failed (rc={rc}).  See its output above for "
            "the underlying error — common causes: nvcc not on PATH "
            "(install the CUDA toolkit matching your torch wheel), or "
            "the gdown download being rate-limited (re-run with the env "
            "var DROID_WEIGHTS_URL pointing at a mirror).",
        )

    # Invalidate Python's module cache so the freshly installed package
    # is picked up on the very next import call without a restart.
    importlib.invalidate_caches()
    if not _droid_importable():
        raise RuntimeError(
            "setup_droid.sh reported success but `droid_slam` is still "
            "not importable.  Inspect ~/.cache/xr-ai/DROID-SLAM/ and the "
            "venv's site-packages to debug.",
        )
    logger.info("[droid-build] droid_slam installed successfully")
