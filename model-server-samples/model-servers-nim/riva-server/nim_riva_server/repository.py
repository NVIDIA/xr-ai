# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container-side export/reload bootstrap for the sample's pinned Riva NIMs.

Only the Python standard library is used; this file runs inside NVIDIA's image.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import zlib
from pathlib import Path


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _inventory(directory: Path) -> dict:
    return {
        path.name: {"size": path.stat().st_size, "sha256": _digest(path)}
        for path in sorted(directory.glob("*.tar.gz"))
    }


def _complete(directory: Path, contract: dict) -> bool:
    try:
        manifest = json.loads((directory / "complete.json").read_text())
        if manifest["contract"] != contract or manifest["archives"] != _inventory(directory):
            return False
        # A previous validator may have recorded checksums for malformed bytes.
        # Always apply the current structural checks, including on legacy hits.
        _validate_archives(directory, set(manifest["archives"]))
        return True
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, tarfile.TarError,
            EOFError, zlib.error):
        return False


def _validate_export(directory: Path, workspace: Path) -> None:
    # These images can return success even when riva-deploy fails. Require an
    # intact archive for every RMIR in addition to checking the process status.
    expected = {f"{path.stem}.tar.gz" for path in workspace.rglob("*.rmir")}
    _validate_archives(directory, expected)


def _validate_archives(directory: Path, expected: set[str]) -> None:
    actual = {path.name for path in directory.glob("*.tar.gz")}
    if not expected or actual != expected:
        raise RuntimeError(f"Riva export incomplete: expected {sorted(expected)}, found {sorted(actual)}")
    for name in sorted(expected):
        with tarfile.open(directory / name, "r:*") as archive:
            members = archive.getmembers()
            configs = [m for m in members if m.isfile() and m.name.endswith("/config.pbtxt") and m.size]
            engines = [m for m in members if m.isfile() and m.name.endswith((".plan", ".engine")) and m.size]
            if not configs or not engines:
                raise RuntimeError(f"Riva export {name} lacks model configurations or TensorRT engines")
            # getmembers() can silently stop at a malformed later header.
            # Recheck from the next expected header, including the block that
            # stopped parsing: only two zero end blocks and zero padding may
            # remain. Reading through EOF also verifies compression trailers.
            archive.fileobj.seek(archive.offset)
            padding = 0
            while chunk := archive.fileobj.read(1024 * 1024):
                if chunk.strip(b"\0"):
                    raise RuntimeError(f"Riva export {name} has an invalid tar trailer or header")
                padding += len(chunk)
            if padding < 2 * tarfile.BLOCKSIZE or padding % tarfile.BLOCKSIZE:
                raise RuntimeError(f"Riva export {name} has incomplete tar end blocks")


def _legacy_repository(root: Path, contract: dict) -> Path | None:
    # The original bootstrap produced format 1. Later format changes must not
    # adopt its exports, even when the hardware and model settings match.
    if contract.get("build_format") != 1:
        return None
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        try:
            legacy = json.loads((directory / "complete.json").read_text())["contract"]
            if not isinstance(legacy, dict) or "bootstrap" not in legacy or "build_format" in legacy:
                continue
            compatible = {k: v for k, v in legacy.items() if k != "bootstrap"}
            compatible["build_format"] = 1
            key = hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()
            if compatible != contract or directory.name != key:
                continue
            # Skip an old writer rather than racing its publication or waiting
            # while holding the new key's lock. Never inspect .building exports.
            with (root / f"{key}.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if _complete(directory, legacy):
                    return directory
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def prepare_repository(root: Path, contract: dict, command: list[str], env: dict[str, str]) -> Path:
    """Publish complete exports atomically; failed or interrupted builds stay unusable."""
    root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    repository = root / key
    staging = root / f"{key}.building"
    with (root / f"{key}.lock").open("a") as lock:
        print(f"[riva-cache] Waiting for repository lock {key}", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _complete(repository, contract):
            print(f"[riva-cache] Reusing compiled repository {repository}", flush=True)
            return repository
        legacy = _legacy_repository(root, contract)
        if legacy is not None:
            # Reuse in place: running containers may still refer to this path.
            print(f"[riva-cache] Reusing validated legacy repository {legacy}", flush=True)
            return legacy
        # Only this key's unpublished staging area is discarded. Other model
        # profiles and known-good exports remain available after --stop.
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        print(f"[riva-cache] Building compiled repository {repository}", flush=True)
        try:
            # nimlib's downloader in these images hardcodes /opt/nim/workspace
            # even when NIM_WORKSPACE is set. Keep its native build workspace;
            # only serving (which disables downloads) may use a fresh one.
            workspace = Path(env.get("NIM_WORKSPACE", "/opt/nim/workspace"))
            shutil.rmtree(workspace, ignore_errors=True)
            workspace.mkdir(parents=True, exist_ok=True)
            build_env = dict(env, NIM_WORKSPACE=str(workspace), NIM_EXPORT_PATH=str(staging),
                             NIM_DISABLE_MODEL_DOWNLOAD="false",
                             NIM_USE_MULTIPROCESSING_FOR_INFERENCE="true")
            subprocess.run(command, env=build_env, check=True)
            _validate_export(staging, workspace)
            (staging / "complete.json").write_text(json.dumps({
                "contract": contract, "archives": _inventory(staging),
            }, sort_keys=True))
            if repository.exists():
                shutil.rmtree(repository)
            staging.rename(repository)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return repository


def run() -> None:
    """Build missing engines, then exec NVIDIA's server with the exported repository."""
    env = dict(os.environ)
    command = json.loads(env.pop("XR_AI_RIVA_COMMAND"))
    contract = json.loads(env.pop("XR_AI_RIVA_CONTRACT"))
    env.pop("XR_AI_RIVA_BOOTSTRAP", None)
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader,nounits",
    ], text=True).strip()
    if not gpu or "[N/A]" in gpu:
        raise RuntimeError("cannot identify the visible GPU for the Riva repository cache")
    contract.update({"gpu": gpu, "architecture": platform.machine()})
    repository = prepare_repository(
        Path(env["NIM_CACHE_PATH"]) / "riva-repositories", contract, command, env,
    )
    # Build and serve share a container, but must not share a workspace: an
    # RMIR left in the serving workspace forces riva-deploy -f to run again.
    env.update(NIM_WORKSPACE=tempfile.mkdtemp(prefix="riva-serve-"),
               NIM_EXPORT_PATH=str(repository), NIM_DISABLE_MODEL_DOWNLOAD="true")
    print("[riva-cache] Starting Riva from the compiled repository", flush=True)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    run()
