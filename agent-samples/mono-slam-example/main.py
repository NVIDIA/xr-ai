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
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
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
_EIGEN_VERSION = "3.4.0"
_EIGEN_SHA256  = "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72"
_EIGEN_URL     = (
    f"https://gitlab.com/libeigen/eigen/-/archive/{_EIGEN_VERSION}/"
    f"eigen-{_EIGEN_VERSION}.zip"
)


def _dl_progress(block_num: int, block_size: int, total_size: int) -> None:
    # Carriage-return progress is intentionally raw print() — loguru records
    # are line-oriented and would emit a fresh line per update, defeating
    # the in-place spinner.  Same approach as xr-render-demo.
    if total_size > 0:
        pct = min(100, block_num * block_size * 100 // total_size)
        print(f"\r  [setup]   {pct}%   ", end="", flush=True)


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
    if eigen_dir.exists():
        logger.debug("Eigen already present at {}", eigen_dir)
        return

    logger.info("Eigen {} not found — downloading", _EIGEN_VERSION)
    eigen_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = eigen_dir.parent / "eigen.zip.partial"
    try:
        urllib.request.urlretrieve(_EIGEN_URL, partial, _dl_progress)
        print()  # finish progress line
    except Exception as exc:
        partial.unlink(missing_ok=True)
        sys.exit(f"\n  [setup] Eigen download failed: {exc}\n")

    actual = hashlib.sha256(partial.read_bytes()).hexdigest()
    if actual != _EIGEN_SHA256:
        partial.unlink(missing_ok=True)
        sys.exit(
            f"\n  [setup] Eigen SHA256 mismatch.\n"
            f"    expected: {_EIGEN_SHA256}\n"
            f"    got:      {actual}\n"
        )

    with zipfile.ZipFile(partial) as zf:
        zf.extractall(eigen_dir.parent)
    partial.unlink()

    # Some Eigen archives extract as eigen-<hash>/ rather than eigen-3.4.0/;
    # rename if needed so DPVO's setup.py finds the path it expects.
    if not eigen_dir.exists():
        extracted = next(
            (d for d in eigen_dir.parent.iterdir()
             if d.is_dir() and d.name.startswith("eigen-")),
            None,
        )
        if extracted is not None:
            extracted.rename(eigen_dir)

    logger.info("Eigen extracted to {}", eigen_dir)


def run() -> None:
    setup_logging("orchestrator", namespace="mono-slam-example")
    _ensure_nvcc_on_path()
    _ensure_dpvo_source()
    run_stack(_PROCESSES, _BASE)


if __name__ == "__main__":
    run()
