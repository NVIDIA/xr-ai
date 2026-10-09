# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run inside the vLLM container to resolve its mounted Hugging Face cache.

``local`` prints a cached path or exits with CACHE_MISS; explicit-offline misses
fail. ``download`` fetches and syncs before CUDA starts. Only the path goes to
stdout. Snapshot contents are left to the runtime loader.
"""
from __future__ import annotations

import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

CACHE_MISS = 3


def resolve_local(repo_id: str, revision: str | None) -> Path | None:
    """Return a local model directory or cached snapshot, without downloading."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    if os.path.isdir(repo_id):
        return Path(repo_id)
    try:
        return Path(snapshot_download(
            repo_id=repo_id, revision=revision, local_files_only=True,
        ))
    except LocalEntryNotFoundError:
        return None


def main(argv: list[str]) -> int:
    mode, repo_id, revision = argv
    revision = revision or None
    from huggingface_hub import constants, snapshot_download

    # Hub or download logs must not enter the shell's model_path assignment.
    with redirect_stdout(sys.stderr):
        if mode == "local":
            snapshot = resolve_local(repo_id, revision)
            if snapshot is None:
                if not constants.HF_HUB_OFFLINE:
                    return CACHE_MISS
                print(
                    f"error: {repo_id} (revision {revision or 'main'}) is not cached "
                    f"under {constants.HF_HUB_CACHE}, and HF_HUB_OFFLINE is set. "
                    "Start once with network access to download it, or unset HF_HUB_OFFLINE."
                )
                return 1
        elif mode == "download":
            print(f"Downloading {repo_id} (revision {revision or 'main'}) before CUDA startup")
            snapshot = Path(snapshot_download(repo_id=repo_id, revision=revision))
            # Finish writeback before CUDA allocation on Spark's shared memory.
            os.sync()
        else:
            raise ValueError(f"unknown mode {mode!r}")
    print(snapshot)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
