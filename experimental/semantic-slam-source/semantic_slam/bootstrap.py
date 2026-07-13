# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First-launch dependency bootstrap.

The heavy model stack (chamferdist, pytorch3d, gradslam, segment-anything) and
the SAM weights are not part of the default ``uv sync`` because they build CUDA
extensions from source and are fetched from git. ``ensure_setup`` checks whether
they are present and, if not, runs ``scripts/setup_env.sh`` to install them into
the uv-managed venv -- so the very first ``SemanticSLAM(...)`` transparently sets
everything up.

Disable the auto-run with ``SEMANTIC_SLAM_AUTO_SETUP=0`` (e.g. in CI), in which
case a missing dependency raises with the exact command to run by hand.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

# Modules that only exist after scripts/setup_env.sh has run.
_REQUIRED = (
    "chamferdist",
    "pytorch3d",
    "gradslam",
    "segment_anything",
    "groundingdino",
    "ram",
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SETUP_SCRIPT = _REPO_ROOT / "scripts" / "setup_env.sh"


def _missing():
    """Return the subset of the model stack that is not importable."""
    missing = []
    for mod in _REQUIRED:
        if importlib.util.find_spec(mod) is None:
            missing.append(mod)
    return missing


def ensure_setup(*, force=False):
    """Ensure the model stack is installed, running setup_env.sh if needed.

    Idempotent and cheap on the happy path (just import-spec lookups). Honors
    ``SEMANTIC_SLAM_AUTO_SETUP`` (default ``"1"``); set to ``"0"`` to turn the
    auto-install off and fail loudly instead.
    """
    if not force and not _missing():
        return

    auto = os.environ.get("SEMANTIC_SLAM_AUTO_SETUP", "1") != "0"
    if not auto:
        raise RuntimeError(
            f"semantic-slam model stack not installed (missing: {_missing()}). "
            f"Run: {_SETUP_SCRIPT}  (or set SEMANTIC_SLAM_AUTO_SETUP=1 to "
            "auto-install on launch)."
        )

    if not _SETUP_SCRIPT.exists():
        raise RuntimeError(f"setup script not found: {_SETUP_SCRIPT}")

    print(
        f"[semantic-slam] first-launch setup: installing model stack via "
        f"{_SETUP_SCRIPT} ...",
        file=sys.stderr,
    )
    subprocess.run(["bash", str(_SETUP_SCRIPT)], check=True)

    # Drop any negative import caches so the freshly-installed modules resolve.
    importlib.invalidate_caches()
    still_missing = _missing()
    if still_missing:
        raise RuntimeError(
            f"setup ran but these are still missing: {still_missing}. "
            f"See output of {_SETUP_SCRIPT}."
        )
