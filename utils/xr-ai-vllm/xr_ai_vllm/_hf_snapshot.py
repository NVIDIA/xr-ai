# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local-first Hugging Face snapshot resolution for the vLLM container.

The Docker backend runs this file's source inside the container as
``python3 -c <source> <mode> <repo_id> <revision>`` against the mounted
``HF_HOME``. It therefore imports nothing from ``xr_ai_vllm`` and defers
``huggingface_hub``, which the host package does not depend on, to call time.

``local`` prints the cached snapshot directory and makes no network request.
When the snapshot is absent or incomplete, it exits with :data:`CACHE_MISS`,
or fails with an error naming the model and revision if ``HF_HUB_OFFLINE`` is
set. ``download`` fetches the snapshot, flushes it to disk, and prints its
directory. Only the directory is written to stdout.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

CACHE_MISS = 3
# vLLM's default loader serves Mistral-format weights when present, otherwise
# the first of these patterns with files; it ignores every other format. Model
# classes may override the patterns (Cosmos3 reads ``[tv]*er/*.safetensors``),
# but the root safetensors index still selects the files the loader reads.
_MISTRAL_WEIGHTS = ("consolidated*.safetensors", "consolidated.safetensors.index.json")
_SAFETENSORS_INDEX = "model.safetensors.index.json"
_WEIGHT_FORMATS = (
    ("*.safetensors", _SAFETENSORS_INDEX),
    ("*.bin", "pytorch_model.bin.index.json"),
    ("*.pt", None),
)
# Trainer outputs vLLM drops from the .bin and .pt formats.
_NOT_FOR_INFERENCE = (
    "training_args.bin",
    "optimizer.bin",
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
)
# Hub shard names encode their own count: ``<stem>-00001-of-00004.<ext>``.
_SHARD_NAME = re.compile(r"^(?P<stem>.+)-(?P<i>\d+)-of-(?P<n>\d+)\.(?P<ext>[a-z]+)$")
# Each entry is one complete tokenizer definition the loaders accept.
_TOKENIZER_FORMATS = (
    ("tokenizer.json",),
    ("tokenizer.model",),
    ("tekken.json",),
    ("vocab.json", "merges.txt"),
    ("vocab.txt",),
)
_PROCESSOR_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)
# config.json sections that mean the model consumes non-text input.
_MODALITY_CONFIGS = ("vision_config", "audio_config", "sound_config", "speech_config")
# Configs whose ``auto_map`` names remote-code modules the loaders import.
_AUTO_MAP_CONFIGS = ("config.json", "tokenizer_config.json", *_PROCESSOR_FILES)
# The relative imports transformers follows when it loads remote code.
_RELATIVE_IMPORT = re.compile(
    r"^\s*(?:import\s+\.(\S+)\s*$|from\s+\.(\S+)\s+import)", re.MULTILINE
)


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _missing_remote_code(snapshot: Path) -> list[str]:
    """Return the remote-code modules that *snapshot* lacks.

    ``auto_map`` values are ``module.Class``, a list of them (slow and fast
    tokenizers, either possibly null), or ``repo--module.Class`` for code
    hosted in another repository, which this snapshot need not contain.
    Each module present is scanned for the relative imports transformers
    also fetches, so helper modules are required too.
    """
    pending: list[str] = []
    for name in _AUTO_MAP_CONFIGS:
        data = _read_json(snapshot / name) or {}
        auto_map = data.get("auto_map")
        if not isinstance(auto_map, dict):
            continue
        for value in auto_map.values():
            for ref in value if isinstance(value, list) else [value]:
                if isinstance(ref, str) and "--" not in ref and "." in ref:
                    pending.append(ref.rsplit(".", 1)[0].replace(".", "/") + ".py")
    seen: set[str] = set()
    missing: list[str] = []
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        path = snapshot / module
        if not path.is_file():
            missing.append(module)
            continue
        package = Path(module).parent
        for match in _RELATIVE_IMPORT.finditer(path.read_text(errors="replace")):
            imported = match[1] or match[2]
            if not imported.startswith("."):  # parent-package imports leave the repo
                pending.append(str(package / (imported.replace(".", "/") + ".py")))
    return sorted(missing)


def _selected_weights(snapshot: Path) -> tuple[list[Path], str | None]:
    """Return the weight files and index name vLLM's default loader selects.

    Root files decide the format as they do for vLLM; subfolders are searched
    only when the root has no weights, for models whose class overrides the
    load patterns but ships no root index.
    """
    mistral = list(snapshot.glob(_MISTRAL_WEIGHTS[0]))
    if mistral:
        return mistral, _MISTRAL_WEIGHTS[1]
    if (snapshot / _SAFETENSORS_INDEX).exists():
        return [], _SAFETENSORS_INDEX
    for search in (snapshot.glob, snapshot.rglob):
        for pattern, index in _WEIGHT_FORMATS:
            files = list(search(pattern))
            if files:
                if pattern != "*.safetensors":
                    files = [f for f in files if f.name not in _NOT_FOR_INFERENCE]
                return files, index
    return [], None


def _missing_weights(snapshot: Path) -> list[str]:
    """Return the shards of the selected weight format that *snapshot* lacks.

    The format's index lists its shards. An interrupted download can leave
    the index unlinked, so the shard names themselves (``-i-of-n``) are the
    fallback evidence of how many files exist. Other formats are not loaded,
    so their completeness does not matter.
    """
    files, index_name = _selected_weights(snapshot)
    index = snapshot / index_name if index_name else None
    if index is not None and index.exists():
        data = _read_json(index)
        if data is None:
            return [index.name]
        shards = set(data.get("weight_map", {}).values())
        return sorted(s for s in shards if not (snapshot / s).is_file())
    if not files:
        return ["model weights (*.safetensors, *.bin or *.pt)"]

    groups: dict[tuple[str, str, int], set[int]] = {}
    for path in files:
        match = _SHARD_NAME.match(path.name)
        if match is not None:
            stem = str(path.relative_to(snapshot).with_name(match["stem"]))
            key = (stem, match["ext"], int(match["n"]))
            groups.setdefault(key, set()).add(int(match["i"]))
    missing: list[str] = []
    for (stem, ext, count), present in sorted(groups.items()):
        absent = set(range(1, count + 1)) - present
        if absent:
            missing.append(f"{stem}: {len(absent)} of {count} .{ext} shards")
    return missing


def missing_files(snapshot: Path) -> list[str]:
    """Return the files a vLLM model needs that *snapshot* lacks.

    The Hub links a file into a snapshot only after its blob is complete, but
    an interrupted download leaves other files unlinked, and older Hub
    releases keep no file listing that ``local_files_only`` could check. The
    contract is therefore: ``config.json``, a complete tokenizer definition,
    processor config when ``config.json`` declares non-text modalities, every
    remote-code module an ``auto_map`` names, and every shard of the weight
    format vLLM's default loader selects.
    """
    config = snapshot / "config.json"
    if not config.is_file():
        return ["config.json"]
    missing: list[str] = []
    if not any(
        all((snapshot / name).is_file() for name in fmt) for fmt in _TOKENIZER_FORMATS
    ):
        missing.append("tokenizer files (e.g. tokenizer.json)")
    modalities = _read_json(config) or {}
    if any(key in modalities for key in _MODALITY_CONFIGS) and not any(
        (snapshot / name).is_file() for name in _PROCESSOR_FILES
    ):
        missing.append("processor config (preprocessor_config.json)")
    return missing + _missing_remote_code(snapshot) + _missing_weights(snapshot)


def _describe(repo_id: str, revision: str | None) -> str:
    return f"{repo_id} (revision {revision or 'main'})"


def resolve_local(repo_id: str, revision: str | None) -> Path | None:
    """Return the complete cached snapshot directory, or ``None``."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    if os.path.isdir(repo_id):
        return Path(repo_id)  # configured as a local model directory
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=repo_id, revision=revision, local_files_only=True
            )
        )
    except LocalEntryNotFoundError:
        return None
    missing = missing_files(snapshot)
    if missing:
        print(
            f"Ignoring incomplete cached snapshot of {_describe(repo_id, revision)}; "
            f"missing: {', '.join(missing)}",
            file=sys.stderr,
        )
        return None
    return snapshot


def download(repo_id: str, revision: str | None) -> Path:
    """Download the snapshot and flush it to disk before CUDA starts."""
    from huggingface_hub import snapshot_download

    print(
        f"Downloading Hugging Face snapshot {_describe(repo_id, revision)} "
        "before CUDA startup",
        file=sys.stderr,
    )
    snapshot = Path(snapshot_download(repo_id=repo_id, revision=revision))
    missing = missing_files(snapshot)
    if missing:
        raise RuntimeError(
            f"downloaded snapshot of {_describe(repo_id, revision)} at {snapshot} "
            f"is missing: {', '.join(missing)}"
        )
    # Spark's CPU and GPU share one memory pool; finish writeback before vLLM
    # initializes CUDA so the two allocations cannot overlap.
    os.sync()
    return snapshot


def main(argv: list[str]) -> int:
    mode, repo_id, revision = argv
    revision = revision or None
    stdout, sys.stdout = sys.stdout, sys.stderr
    try:
        if mode == "local":
            snapshot = resolve_local(repo_id, revision)
            if snapshot is None:
                from huggingface_hub import constants

                if not constants.HF_HUB_OFFLINE:
                    return CACHE_MISS
                print(
                    f"error: {_describe(repo_id, revision)} is not fully cached "
                    f"under {constants.HF_HUB_CACHE}, and HF_HUB_OFFLINE is set. "
                    "Start once with network access to download it, or unset "
                    "HF_HUB_OFFLINE.",
                    file=sys.stderr,
                )
                return 1
            print(f"Using cached snapshot {snapshot}", file=sys.stderr)
        elif mode == "download":
            snapshot = download(repo_id, revision)
        else:
            raise ValueError(f"unknown mode {mode!r}")
    finally:
        sys.stdout = stdout
    print(snapshot)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
