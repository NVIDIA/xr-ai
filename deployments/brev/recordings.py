# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""List finalized session bundles and stream them without following filesystem links."""

import json
import os
import shutil
import stat
import zipfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path.home() / ".local/share/xr-ai/captures"


@contextmanager
def open_file(relative):
    path = Path(relative)
    if path.is_absolute() or not path.parts or len(path.parts) > 20 or any(
        part in {".", ".."} or "\\" in part or ":" in part for part in path.parts
    ):
        raise ValueError("invalid recording path")
    # Walk beneath the capture root using directory descriptors. NOFOLLOW also
    # protects against a directory being replaced by a symlink during download.
    directory = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("recording asset is not a regular file")
            yield stream
    finally:
        os.close(directory)


def manifest(session):
    if session in {"", "."}:
        raise ValueError("recording session is required")
    with open_file(Path(session) / "manifest.json") as stream:
        raw = stream.read(1048577)
    if len(raw) > 1048576:
        raise ValueError("recording manifest is too large")
    value = json.loads(raw)
    if not isinstance(value, dict) or "complete" not in value:
        raise ValueError("recording is not finalized")
    return value


def recordings():
    values = []
    if ROOT.is_symlink():
        return values
    for path in sorted(ROOT.glob("**/manifest.json"), reverse=True):
        session = path.parent.relative_to(ROOT).as_posix()
        try:
            value = manifest(session)
        except (OSError, ValueError):
            continue
        values.append({"id": session, "participant": str(value.get("participant_id", "")),
                       "complete": value["complete"] is True})
    return values


def files(session):
    # Request IDs select only server-enumerated bundles; they never become paths.
    session = next((item["id"] for item in recordings() if item["id"] == session), None)
    if session is None:
        raise ValueError("finalized recording not found")
    manifest(session)
    root = ROOT.resolve()
    folder = (ROOT / session).resolve()
    if not folder.is_relative_to(root):
        raise ValueError("recording escapes the capture root")
    # No recursive symlink traversal. Each file is opened again through open_file
    # when streaming, so the names collected here cannot bypass its safety checks.
    result = []
    for directory, _, names in os.walk(folder, followlinks=False):
        for name in sorted(names):
            path = (Path(directory) / name).relative_to(root)
            with open_file(path):
                pass
            result.append(path)
    return result


def write_zip(output, paths):
    # ZIP_STORED avoids recompressing media; zipfile supports non-seekable HTTP output.
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in paths:
            with open_file(path) as source, archive.open(path.as_posix(), "w", force_zip64=True) as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
