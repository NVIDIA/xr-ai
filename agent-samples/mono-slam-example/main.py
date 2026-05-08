# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
mono-slam-example orchestrator — monocular SLAM pose logger.

How to run (from agent-samples/mono-slam-example/):
    uv sync && uv run mono_slam_example

On first run the orchestrator clones DPVO into ../../deps/dpvo/ and extracts
Eigen 3.4.0 into its thirdparty/ directory.  The worker's `uv sync` then
builds DPVO's CUDA extensions against the local torch.  Both bootstrap
steps no-op on subsequent runs.

A CUDA-capable GPU and CUDA toolkit (>= 12.1, with nvcc on PATH) are
required.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from loguru import logger
from xr_ai_launcher import Process, run_stack
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent

_PROCESSES: list[Process] = [
    Process("hub",    "../../server-runtime", "xr_media_hub",
            config="yaml/xr_media_hub.yaml"),
    Process("worker", "worker",               "mono_slam_example_worker",
            config="yaml/mono_slam_example_worker.yaml"),
    # viz subscribes to pose updates from the worker via the hub data channel.
    # Starts after the worker so IPC is ready before the first pose arrives.
    Process("viz",    "viz",                  "mono_slam_example_viz",
            config="yaml/mono_slam_example_viz.yaml"),
]


# ── DPVO source bootstrap ─────────────────────────────────────────────────────
#
# DPVO is not on PyPI.  The worker's pyproject lists it as an editable path
# source at deps/dpvo/.  These helpers materialise that path before the
# worker's `uv sync` runs.  Mirrors xr-render-demo's _ensure_lovr_bin pattern.

_DPVO_REPO     = "https://github.com/princeton-vl/DPVO.git"
_DPVO_DIR      = (_BASE / "../../deps/dpvo").resolve()
_EIGEN_REPO    = "https://gitlab.com/libeigen/eigen.git"
_EIGEN_VERSION = "3.4.0"
# Pinned commit for tag 3.4.0.  GitLab's archive-zip endpoint is
# non-deterministic (files inside the zip are repacked per request), so we
# clone the tag instead and trust git's content-addressed integrity.
_EIGEN_COMMIT  = "3147391d946bb4b6c68edd901f2add6ac1f31f8c"


def _ensure_nvcc_on_path() -> None:
    """Make nvcc reachable so DPVO's CUDA build picks up the toolchain."""
    if shutil.which("nvcc"):
        return
    for candidate in ("/usr/local/cuda/bin",
                      "/usr/local/cuda-12.6/bin",
                      "/usr/local/cuda-12.1/bin"):
        if Path(candidate, "nvcc").exists():
            os.environ["PATH"] = f"{candidate}:{os.environ.get('PATH', '')}"
            logger.info("Added {} to PATH for CUDA build", candidate)
            return
    sys.exit(
        "\n  mono-slam-example: nvcc not found.\n"
        "  Install CUDA toolkit (>= 12.1) and add its bin/ to PATH, then\n"
        "  re-run.\n"
    )


def _ensure_dpvo_source() -> None:
    """Clone DPVO and extract Eigen into deps/dpvo/ if not already present."""
    if not (_DPVO_DIR / ".git").exists():
        logger.info("DPVO not found — cloning {} into {}", _DPVO_REPO, _DPVO_DIR)
        _DPVO_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", _DPVO_REPO, str(_DPVO_DIR)],
            check=True,
        )
    else:
        logger.debug("DPVO already cloned at {}", _DPVO_DIR)

    eigen_dir = _DPVO_DIR / "thirdparty" / f"eigen-{_EIGEN_VERSION}"
    if (eigen_dir / "Eigen").is_dir():
        logger.debug("Eigen already present at {}", eigen_dir)
        return

    logger.info("Eigen {} not found — cloning {}", _EIGEN_VERSION, _EIGEN_REPO)
    eigen_dir.parent.mkdir(parents=True, exist_ok=True)
    # Wipe any leftover partial dir so git clone has a clean target.
    if eigen_dir.exists():
        shutil.rmtree(eigen_dir)
    # Shallow-clone the tag, then verify the resolved commit matches our pin —
    # protects against the (extremely unlikely) case of an upstream tag move.
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "--branch", _EIGEN_VERSION,
         _EIGEN_REPO, str(eigen_dir)],
        check=True,
    )
    head = subprocess.check_output(
        ["git", "-C", str(eigen_dir), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if head != _EIGEN_COMMIT:
        sys.exit(
            f"\n  [setup] Eigen commit mismatch.\n"
            f"    expected: {_EIGEN_COMMIT}\n"
            f"    got:      {head}\n"
            f"    tag {_EIGEN_VERSION} appears to have moved upstream.\n"
        )
    logger.info("Eigen cloned to {}", eigen_dir)


def run() -> None:
    setup_logging("orchestrator", namespace="mono-slam-example")
    _ensure_nvcc_on_path()
    _ensure_dpvo_source()
    run_stack(_PROCESSES, _BASE)


if __name__ == "__main__":
    run()
