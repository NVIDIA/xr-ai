# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Save guide-run evidence and execution events for inspection and evaluation.

One owned writer orders all files. Cancellation of an inference or participant
cannot abandon its queued writes; backend shutdown drains the writer after
observers and controls stop. A trace is diagnostic data, never execution state.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

from loguru import logger


class RunTrace:
    def __init__(self, directory: Path | None, guide) -> None:
        self.directory = directory
        self.guide = guide
        self._queue = asyncio.Queue()
        self._writer = None
        self._closed = False
        self._failed: set[str] = set()

    def emit(self, run_id: str | None, kind: str, *, image: bytes | None = None, **data) -> None:
        if self.directory is None or run_id is None or self._closed:
            return
        event = {"kind": kind, "timestamp_us": time.time_ns() // 1000, **data}
        self._queue.put_nowait((run_id, event, image))
        if self._writer is None:
            self._writer = asyncio.create_task(self._write())

    async def _write(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is None:
                    return
                try:
                    await asyncio.to_thread(self._save, *item)
                except (OSError, ValueError, TypeError):
                    self._failed.add(item[0])
                    logger.exception("Guide trace write failed run={}", item[0])
                    try:
                        await asyncio.to_thread(self._mark_incomplete, item[0])
                    except (OSError, ValueError, TypeError):
                        logger.exception("Could not mark incomplete guide trace run={}", item[0])
            finally:
                self._queue.task_done()

    def _mark_incomplete(self, run_id: str) -> None:
        path = self.directory / run_id / "packet.json"
        if path.exists():
            packet = json.loads(path.read_text())
            packet["status"] = "incomplete"
            self._replace(path, json.dumps(packet, indent=2).encode())

    @staticmethod
    def _replace(path: Path, content: bytes) -> None:
        # A failed write must not leave a truncated frame that a later event
        # mistakes for an already saved content hash.
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)

    def _save(self, run_id, event, image) -> None:
        directory = self.directory / run_id
        directory.mkdir(parents=True, exist_ok=True)
        if event["kind"] == "run_started":
            self._replace(directory / "guide.yaml", self.guide.content)
            packet = {"schema_version": 1, "kind": "guide-run", "status": "active",
                      "guide_sha256": self.guide.sha256, "events": "events.jsonl"}
            self._replace(directory / "packet.json", json.dumps(packet, indent=2).encode())
        if image is not None:
            digest = hashlib.sha256(image).hexdigest()
            frames = directory / "frames"
            frames.mkdir(exist_ok=True)
            # Bytes retain the original format; content sniffing is used on replay.
            target = frames / digest
            if not target.exists():
                self._replace(target, image)
            event["image"] = {"path": str(target.relative_to(directory)), "sha256": digest}
        with (directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        if event["kind"] == "run_ended":
            packet = json.loads((directory / "packet.json").read_text())
            packet.update(status="incomplete" if run_id in self._failed else "ended", reason=event["reason"])
            self._replace(directory / "packet.json", json.dumps(packet, indent=2).encode())

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._writer is not None:
                self._queue.put_nowait(None)
        if self._writer is not None:
            cancelled = False
            while not self._writer.done():
                try:
                    await asyncio.shield(self._writer)
                except asyncio.CancelledError:
                    cancelled = True
            self._writer.result()
            if cancelled:
                raise asyncio.CancelledError
