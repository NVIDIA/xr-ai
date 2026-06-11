# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=0.24"]
# ///
"""
Pre-download every model weight the samples use, into the shared ``models/``
cache — so a machine can run offline (or as an internet-outage backup).

This is a standalone convenience script, not part of the orchestrator. It
fetches the full set across all GPU profiles, including the Omni model, so the
cache is complete regardless of which profile a given machine auto-detects.

Run it once while you have a connection:

    uv run agent-samples/model-servers/download_weights.py

By default it writes to ``<repo>/models`` (the same cache the servers use via
``model_cache`` / ``HF_HOME``). Override with ``--dest`` or set ``HF_HOME``.
Gated NVIDIA repositories need a token: ``export HF_TOKEN=<your token>`` (or log
in with ``huggingface-cli login``) before running.

    uv run agent-samples/model-servers/download_weights.py --dest /mnt/usb/models

Re-running is safe: ``snapshot_download`` resumes and skips files already
present, so the destination doubles as a portable, resumable backup.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Repository root: agent-samples/model-servers/download_weights.py -> parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DEST = _REPO_ROOT / "models"

# (repo id, server it backs, optional allow_patterns to avoid pulling a whole
# multi-language repo). Both Omni variants are included so the cache works on
# Blackwell (NVFP4) and Ada/Hopper (FP8) without a second download trip.
_MODELS: list[tuple[str, str, list[str] | None]] = [
    ("nvidia/Cosmos-Reason1-7B",                              "vlm",                  None),
    ("nvidia/parakeet-tdt-0.6b-v3",                           "stt",                  None),
    ("nvidia/Llama-3.1-Nemotron-Nano-8B-v1",                  "llm (llama_nemotron)", None),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",             "agent-llm (nemotron3_nano)", None),
    ("nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",   "omni (Blackwell / NVFP4)", None),
    ("nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",     "omni (Ada / Hopper / FP8)", None),
    ("nvidia/magpie_tts_multilingual_357m",                  "tts (magpie)",         None),
    # piper-voices holds every language; pull only the default sample voice.
    ("rhasspy/piper-voices",                                  "tts (piper)",          ["en/en_US/lessac/medium/*"]),
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pre-download all sample model weights for offline use.")
    p.add_argument("--dest", type=Path, default=None,
                   help=f"Destination cache dir (default: $HF_HOME or {_DEFAULT_DEST}).")
    p.add_argument("--only", action="append", default=[],
                   help="Download only repo ids containing this substring (repeatable).")
    ns = p.parse_args(argv)

    dest = ns.dest or (Path(os.environ["HF_HOME"]) if os.environ.get("HF_HOME") else _DEFAULT_DEST)
    dest = dest.expanduser().resolve()
    # Point the HF cache at dest so the layout matches what the servers read.
    os.environ.setdefault("HF_HOME", str(dest))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is required. Run via `uv run` (it installs it), "
              "or `pip install huggingface_hub`.", file=sys.stderr)
        return 2

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("WARNING: no HF_TOKEN set — gated NVIDIA repos will 401. "
              "Set HF_TOKEN or run `huggingface-cli login` first.", file=sys.stderr)

    targets = [m for m in _MODELS if not ns.only or any(s in m[0] for s in ns.only)]
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(targets)} model repo(s) into {dest}\n")

    failures: list[tuple[str, str]] = []
    for repo_id, server, patterns in targets:
        print(f"→ {repo_id}  [{server}]" + (f"  (subset: {patterns})" if patterns else ""))
        try:
            snapshot_download(repo_id=repo_id, allow_patterns=patterns, resume_download=True)
            print(f"  done: {repo_id}\n")
        except Exception as exc:  # noqa: BLE001 — best-effort; report and continue
            print(f"  FAILED: {repo_id} — {exc}\n", file=sys.stderr)
            failures.append((repo_id, str(exc)))

    if failures:
        print(f"\n{len(failures)} repo(s) failed:", file=sys.stderr)
        for repo_id, err in failures:
            print(f"  - {repo_id}: {err}", file=sys.stderr)
        print("\nGated repos need access granted on huggingface.co plus a valid "
              "HF_TOKEN. Re-run to resume the rest.", file=sys.stderr)
        return 1

    print(f"\nAll {len(targets)} model repo(s) cached under {dest}. "
          f"Point the servers at it with HF_HOME or the per-profile model_cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
