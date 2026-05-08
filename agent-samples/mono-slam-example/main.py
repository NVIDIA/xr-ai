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


_MIN_CUDA = (12, 1)


def _nvcc_major_minor(nvcc: str) -> tuple[int, int] | None:
    """Parse `<nvcc> --version` and return (major, minor) or None on failure.

    Output format includes a line like:
        Cuda compilation tools, release 12.1, V12.1.66
    """
    try:
        out = subprocess.check_output([nvcc, "--version"], text=True,
                                      stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    for line in out.splitlines():
        if "release" in line:
            tail = line.split("release", 1)[1].strip().split(",", 1)[0]
            parts = tail.split(".")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                return int(parts[0]), int(parts[1])
    return None


def _ensure_nvcc_on_path() -> None:
    """Make a CUDA-{_MIN_CUDA}+-capable nvcc reachable for DPVO's build.

    DPVO compiles CUDA extensions against the torch we install (cu121).
    Torch's build step refuses to use a host nvcc whose major version
    differs from torch's CUDA version, so we must put a 12.x toolchain
    first on PATH even if a 11.x one is the system default.
    """
    current = shutil.which("nvcc")
    if current:
        ver = _nvcc_major_minor(current)
        if ver is not None and ver >= _MIN_CUDA:
            logger.debug("nvcc {}.{} already on PATH at {}", ver[0], ver[1], current)
            return
        logger.warning(
            "nvcc on PATH ({}) is {}.{} — need >= {}.{}; searching for a "
            "newer install", current, *(ver or (0, 0)), *_MIN_CUDA,
        )

    # Common locations for side-by-side CUDA installs.  Prefer the highest
    # qualifying version; sort by parsed (major, minor) descending.
    candidates: list[tuple[tuple[int, int], str]] = []
    for path in sorted(Path("/usr/local").glob("cuda-*")):
        nvcc = path / "bin" / "nvcc"
        if nvcc.exists():
            ver = _nvcc_major_minor(str(nvcc))
            if ver is not None and ver >= _MIN_CUDA:
                candidates.append((ver, str(path / "bin")))
    candidates.sort(reverse=True)

    if candidates:
        chosen = candidates[0][1]
        os.environ["PATH"] = f"{chosen}:{os.environ.get('PATH', '')}"
        logger.info("Prepending {} to PATH for CUDA build", chosen)
        return

    sys.exit(
        f"\n  mono-slam-example: no CUDA >= {_MIN_CUDA[0]}.{_MIN_CUDA[1]} found.\n"
        f"  DPVO compiles its CUDA extensions against torch 2.3.1+cu121, which\n"
        f"  requires the CUDA {_MIN_CUDA[0]}.{_MIN_CUDA[1]}+ toolkit.\n"
        f"  Install it from https://developer.nvidia.com/cuda-toolkit and ensure\n"
        f"  its bin/ is on PATH (or installs to /usr/local/cuda-12.x/), then re-run.\n"
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
