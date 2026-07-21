# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the timestamped H.264 chunks written by XR Media Hub."""

import json
from pathlib import Path

from loguru import logger
from xr_ai_nat.functions._rpc import RPCError


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)


class ChunkStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _check(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise RPCError(
                f"Path escapes recordings directory: {path}",
                code="path_escape",
            )
        return resolved

    def _participant_dir(self, participant_id: str) -> Path | None:
        if not self.root.exists():
            return None
        canonical = self._check(self.root / safe_name(participant_id))
        if canonical.is_dir() and self._matches(canonical, participant_id):
            return canonical
        for path in sorted(self.root.iterdir()):
            directory = self._check(path)
            if directory.is_dir() and self._matches(directory, participant_id):
                return directory
        return None

    def _identity(self, directory: Path) -> str | None:
        identity = self._check(directory / ".identity")
        if identity.exists():
            return identity.read_text(encoding="utf-8")
        return None

    def _matches(self, directory: Path, participant_id: str) -> bool:
        identity = self._identity(directory)
        if identity is not None:
            return identity == participant_id
        return directory.name == participant_id

    def participants(self) -> list[str]:
        if not self.root.exists():
            return []
        participants = []
        for path in sorted(self.root.iterdir()):
            directory = self._check(path)
            if not directory.is_dir():
                continue
            participants.append(self._identity(directory) or directory.name)
        return participants

    def _metadata(self, chunk: Path) -> dict:
        sidecar = self._check(chunk.with_suffix(".json"))
        if sidecar.exists():
            try:
                return json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                logger.warning("Ignoring corrupt video metadata {}: {}", sidecar, error)
        timestamp = int(chunk.stem)
        return {
            "start_us": timestamp,
            "end_us": timestamp,
            "size_bytes": chunk.stat().st_size,
        }

    def chunks(self, participant_id: str) -> list[tuple[Path, dict]]:
        directory = self._participant_dir(participant_id)
        if directory is None:
            return []
        paths = sorted(
            (self._check(path) for path in directory.glob("*.264")),
            key=lambda path: int(path.stem),
        )
        return [(path, self._metadata(path)) for path in paths]

    def stats(self, participant_id: str) -> dict:
        chunks = self.chunks(participant_id)
        if not chunks:
            raise RPCError(f"No video chunks for {participant_id!r}", code="not_found")
        total = sum(int(meta.get("size_bytes", path.stat().st_size)) for path, meta in chunks)
        return {
            "participant_id": participant_id,
            "num_chunks": len(chunks),
            "total_bytes": total,
            "avg_chunk_bytes": total // len(chunks),
            "earliest_us": int(chunks[0][1].get("start_us", chunks[0][0].stem)),
            "latest_us": int(chunks[-1][1].get("end_us", chunks[-1][0].stem)),
        }

    def query(self, participant_id: str, start_us: int, end_us: int) -> bytes:
        selected: list[Path] = []
        anchor: Path | None = None
        for path, metadata in self.chunks(participant_id):
            chunk_start = int(metadata.get("start_us", path.stem))
            chunk_end = int(metadata.get("end_us", chunk_start))
            if chunk_start <= end_us and chunk_end >= start_us:
                selected.append(path)
            elif chunk_start < start_us:
                anchor = path
        if not selected and anchor is not None:
            selected.append(anchor)
        if not selected:
            raise RPCError("No video in requested time window", code="not_found")
        return b"".join(self._check(path).read_bytes() for path in selected)

    def frame_chunk(self, participant_id: str, timestamp_us: int) -> tuple[Path, dict]:
        chunks = self.chunks(participant_id)
        if not chunks:
            raise RPCError(f"No recorded video for {participant_id!r}", code="not_found")
        for path, metadata in chunks:
            start = int(metadata.get("start_us", path.stem))
            end = int(metadata.get("end_us", start))
            if start <= timestamp_us <= end:
                return path, metadata
        return min(
            chunks,
            key=lambda item: abs(int(item[1].get("start_us", item[0].stem)) - timestamp_us),
        )
