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


# torch 2.3.1+cu121's cpp_extension build refuses any nvcc whose major
# version differs from torch's CUDA major (12).  Both 11.x and 13.x fail.
# Within major 12 we want the highest minor we can find that's >= 12.1.
_REQUIRED_CUDA_MAJOR = 12
_MIN_CUDA_MINOR      = 1


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


def _cuda_version_acceptable(ver: tuple[int, int]) -> bool:
    """torch 2.3.1+cu121 accepts only major == 12 with minor >= 12.1."""
    return ver[0] == _REQUIRED_CUDA_MAJOR and ver[1] >= _MIN_CUDA_MINOR


def _ensure_nvcc_on_path() -> None:
    """Make a CUDA-12.x nvcc reachable for DPVO's build.

    DPVO compiles CUDA extensions against torch 2.3.1+cu121, whose build
    step (`torch.utils.cpp_extension._check_cuda_version`) errors out if
    the host nvcc's major version differs from torch's CUDA major (12).
    A host with CUDA 11.x first on PATH (or 13.x — already shipping on
    some distros) needs us to find a side-by-side 12.x install.
    """
    current = shutil.which("nvcc")
    if current:
        ver = _nvcc_major_minor(current)
        if ver is not None and _cuda_version_acceptable(ver):
            logger.debug("nvcc {}.{} already on PATH at {}", ver[0], ver[1], current)
            return
        logger.warning(
            "nvcc on PATH ({}) is {}.{} — need {}.{}+ within major {}; "
            "searching for a side-by-side install",
            current, *(ver or (0, 0)),
            _REQUIRED_CUDA_MAJOR, _MIN_CUDA_MINOR, _REQUIRED_CUDA_MAJOR,
        )

    # Common locations for side-by-side CUDA installs.  Prefer the highest
    # qualifying minor within the required major; sort by version desc.
    candidates: list[tuple[tuple[int, int], str]] = []
    for path in sorted(Path("/usr/local").glob("cuda-*")):
        nvcc = path / "bin" / "nvcc"
        if nvcc.exists():
            ver = _nvcc_major_minor(str(nvcc))
            if ver is not None and _cuda_version_acceptable(ver):
                candidates.append((ver, str(path / "bin")))
    candidates.sort(reverse=True)

    if candidates:
        chosen = candidates[0][1]
        os.environ["PATH"] = f"{chosen}:{os.environ.get('PATH', '')}"
        logger.info("Prepending {} to PATH for CUDA build", chosen)
        return

    sys.exit(
        f"\n  mono-slam-example: no CUDA {_REQUIRED_CUDA_MAJOR}.x install with\n"
        f"  minor >= {_MIN_CUDA_MINOR} was found.  DPVO compiles its CUDA\n"
        f"  extensions against torch 2.3.1+cu121, which requires nvcc with\n"
        f"  major == {_REQUIRED_CUDA_MAJOR}; both 11.x and 13.x are rejected\n"
        f"  by torch's build-time CUDA-version check.\n\n"
        f"  Install CUDA {_REQUIRED_CUDA_MAJOR}.{_MIN_CUDA_MINOR}+ from\n"
        f"  https://developer.nvidia.com/cuda-toolkit-archive — installing it\n"
        f"  side-by-side under /usr/local/cuda-{_REQUIRED_CUDA_MAJOR}.x/ is\n"
        f"  enough; the bootstrap will pick it up automatically without\n"
        f"  disturbing your existing default toolkit.\n"
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
