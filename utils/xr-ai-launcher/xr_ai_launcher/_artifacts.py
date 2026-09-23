# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stdlib helpers for preparing and recording local artifacts."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

_MANIFEST_VERSION = 1


class _PreparationInterrupted(BaseException):
    def __init__(self, signum: int) -> None:
        self.signum = signum


@dataclass(frozen=True)
class ArtifactManifest:
    """A validated artifact root and its recorded files."""

    root: Path
    """Base path used to resolve recorded relative file names."""

    files: tuple[tuple[str, int], ...]
    """Relative file names paired with their byte sizes."""

    @property
    def size(self) -> int:
        """Total recorded byte size."""
        return sum(size for _, size in self.files)


def format_size(size: int | None) -> str:
    """Format a byte count with IEC units."""
    if size is None:
        return "unknown"
    value = float(size)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise RuntimeError("artifact size unit selection failed")


def path_size(path: Path) -> int:
    """Return the byte size of files below a path without double-counting links."""
    if path.is_file():
        return path.stat().st_size
    total = 0
    seen: set[tuple[int, int]] = set()
    for child in path.rglob("*"):
        try:
            stat = child.stat()
        except OSError:
            continue
        if not child.is_file():
            continue
        identity = (stat.st_dev, stat.st_ino)
        if identity not in seen:
            seen.add(identity)
            total += stat.st_size
    return total


def report_prepare_status(name: str, state: str, size: int | None) -> None:
    """Flush a human-readable artifact name, state, and size to stdout."""
    print(f"[prepare] {name}: {state} (size: {format_size(size)})", flush=True)


def docker_image_size(image: str) -> int | None:
    """Return a local Docker image's bytes, or ``None`` if inspection fails."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Size}}", image],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError):
        return None


def repair_hf_snapshot(
    repo_id: str,
    cache_dir: Path,
    *,
    revision: str = "main",
    token: str | None = None,
) -> Path:
    """Repair a Hub branch or tag and reject stale offline cache fallback.

    The remote revision, complete file-size inventory, returned snapshot, and
    local revision ref must agree before this returns. Content changes that
    preserve every file size are outside the artifact-manifest contract. The
    Hugging Face dependency is imported lazily for services that use this path.
    """
    from huggingface_hub import HfApi, snapshot_download

    revision_parts = revision.split("/")
    if (
        not revision
        or revision.startswith("/")
        or any(part in {"", ".", ".."} for part in revision_parts)
        or (
            len(revision) == 40
            and all(character in "0123456789abcdef" for character in revision)
        )
    ):
        raise ValueError("Hugging Face repair revision must be a branch or tag")

    api = HfApi(token=token)
    info = api.repo_info(repo_id=repo_id, revision=revision)
    resolved = getattr(info, "sha", None)
    if not isinstance(resolved, str) or not resolved:
        raise RuntimeError(f"Hugging Face returned no revision for {repo_id}")
    expected = {
        str(entry.path): int(entry.size)
        for entry in api.list_repo_tree(
            repo_id=repo_id,
            revision=resolved,
            recursive=True,
        )
        if getattr(entry, "path", None) is not None
        and getattr(entry, "size", None) is not None
    }
    if not expected:
        raise RuntimeError(
            f"Hugging Face returned no files for {repo_id}@{resolved}"
        )

    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            force_download=True,
        )
    )
    if snapshot.name != resolved:
        raise RuntimeError(
            f"Hugging Face repair returned {snapshot.name} instead of {resolved} "
            f"for {repo_id}"
        )
    for filename, expected_size in expected.items():
        parts = filename.split("/")
        if (
            not filename
            or filename.startswith("/")
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise RuntimeError(
                f"Hugging Face returned an unsafe path for {repo_id}: {filename!r}"
            )
        artifact = snapshot.joinpath(*parts)
        if not artifact.is_file() or artifact.stat().st_size != expected_size:
            raise RuntimeError(
                f"Hugging Face repair did not restore {repo_id}/{filename}"
            )

    ref = snapshot.parent.parent / "refs" / revision
    try:
        cached_revision = ref.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(
            f"Hugging Face repair did not publish {repo_id}@{revision}"
        ) from exc
    if cached_revision != resolved:
        raise RuntimeError(
            f"Hugging Face repair left {repo_id}@{revision} at {cached_revision!r}, "
            f"expected {resolved}"
        )
    return snapshot


def prepare_or_exit(service: str, action: Callable[[], None]) -> None:
    """Run preparation with service-prefixed errors and exit codes 130/143.

    Repeated SIGTERM signals are ignored while an interrupted action unwinds.
    The prior handler is restored, or the default when Python cannot recover it.
    """
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    restore_sigterm = previous_sigterm if previous_sigterm is not None else signal.SIG_DFL
    interrupted = False

    def interrupt(signum: int, _frame: object) -> None:
        nonlocal interrupted
        if interrupted:
            return
        interrupted = True
        raise _PreparationInterrupted(signum)

    try:
        try:
            signal.signal(signal.SIGTERM, interrupt)
            action()
        finally:
            signal.signal(signal.SIGTERM, restore_sigterm)
    except _PreparationInterrupted as exc:
        print(
            f"[{service}] artifact preparation interrupted by signal {exc.signum}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(128 + exc.signum) from None
    except KeyboardInterrupt:
        print(f"[{service}] artifact preparation interrupted", file=sys.stderr, flush=True)
        raise SystemExit(130) from None
    except Exception as exc:
        detail = str(exc) or type(exc).__name__
        raise SystemExit(f"[{service}] {detail}") from None
    finally:
        # A signal can interrupt the first restoration attempt itself.
        if signal.getsignal(signal.SIGTERM) is interrupt:
            signal.signal(signal.SIGTERM, restore_sigterm)


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _artifact_path(root: Path, relative: str) -> Path:
    return root if relative == "." else root / relative


def _validated_manifest(root: Path, files: object) -> ArtifactManifest | None:
    if not isinstance(files, list) or not files:
        return None
    validated: list[tuple[str, int]] = []
    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, list) or len(entry) != 2:
            return None
        relative, size = entry
        if (
            not isinstance(relative, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or relative in seen
        ):
            return None
        relative_path = Path(relative)
        if relative != "." and (relative_path.is_absolute() or ".." in relative_path.parts):
            return None
        artifact = _artifact_path(root, relative)
        try:
            if not artifact.is_file() or artifact.stat().st_size != size:
                return None
        except OSError:
            return None
        seen.add(relative)
        validated.append((relative, size))
    return ArtifactManifest(root=root, files=tuple(validated))


def read_artifact_manifest(
    marker: Path,
    *,
    expected_root: Path | None = None,
) -> ArtifactManifest | None:
    """Read a complete manifest, returning ``None`` for stale or invalid data."""
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload["version"] != _MANIFEST_VERSION:
            return None
        root = _absolute(Path(payload["artifact_root"]))
        files = payload["files"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if expected_root is not None and root != _absolute(expected_root):
        return None
    return _validated_manifest(root, files)


def write_artifact_manifest(
    marker: Path,
    root: Path,
    artifacts: Iterable[Path],
) -> ArtifactManifest:
    """Validate and atomically record an explicit set of prepared files."""
    normalized_root = _absolute(root)
    files: list[list[object]] = []
    for artifact in sorted({_absolute(path) for path in artifacts}):
        try:
            if artifact == normalized_root and normalized_root.is_file():
                relative = "."
            else:
                relative = str(artifact.relative_to(normalized_root))
        except ValueError as exc:
            raise RuntimeError(
                f"prepared artifact {artifact} is outside {normalized_root}"
            ) from exc
        try:
            size = artifact.stat().st_size
        except OSError as exc:
            raise RuntimeError(f"prepared artifact is unavailable: {artifact}") from exc
        if not artifact.is_file():
            raise RuntimeError(f"prepared artifact is not a file: {artifact}")
        files.append([relative, size])
    manifest = _validated_manifest(normalized_root, files)
    if manifest is None:
        raise RuntimeError(f"prepared artifact set under {normalized_root} is empty")

    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_root": str(manifest.root),
        "files": [list(entry) for entry in manifest.files],
        "version": _MANIFEST_VERSION,
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=marker.parent,
            prefix=f".{marker.name}.",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(payload, output, sort_keys=True)
        temporary.replace(marker)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest
