# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-scoped persistent text-memory tool."""

import asyncio
import json
from pathlib import Path
from threading import Lock

from pydantic import BaseModel, Field, field_validator

from .tools import Tool
from .types import StrictRequest


class AddTranscriptRequest(StrictRequest):
    source_id: str
    timestamp_us: int
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class AddTranscriptResult(BaseModel):
    ok: bool = True


class TextMemoryTool(Tool[AddTranscriptRequest, AddTranscriptResult]):
    """Append timestamped text to participant-scoped JSONL storage."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._root = self.directory.resolve()
        self._lock = Lock()
        super().__init__(
            "add_transcript",
            "Append one timestamped text segment to persistent memory.",
            AddTranscriptRequest,
            AddTranscriptResult,
            self._append,
        )

    @staticmethod
    def _safe(source_id: str) -> str:
        return "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in source_id
        )

    def _path(self, source_id: str) -> Path:
        stem = self._safe(source_id)
        suffix = 1
        while True:
            candidate = stem if suffix == 1 else f"{stem}_{suffix}"
            identity = (self.directory / f"{candidate}.identity").resolve()
            data = (self.directory / f"{candidate}.jsonl").resolve()
            if not identity.is_relative_to(self._root) or not data.is_relative_to(self._root):
                raise ValueError("transcript path escapes storage directory")
            if identity.exists() and identity.read_text(encoding="utf-8") == source_id:
                return data
            if suffix == 1 and data.exists() and not identity.exists() and source_id == stem:
                identity.write_text(source_id, encoding="utf-8")
                return data
            if not identity.exists() and not data.exists():
                identity.write_text(source_id, encoding="utf-8")
                return data
            suffix += 1

    async def _append(self, request: AddTranscriptRequest) -> AddTranscriptResult:
        await asyncio.to_thread(self._append_sync, request)
        return AddTranscriptResult()

    def _append_sync(self, request: AddTranscriptRequest) -> None:
        with self._lock:
            with self._path(request.source_id).open("a", encoding="utf-8") as output:
                output.write(
                    json.dumps(
                        {
                            "timestamp_us": request.timestamp_us,
                            "text": request.text,
                        }
                    )
                    + "\n"
                )


__all__ = ["AddTranscriptRequest", "AddTranscriptResult", "TextMemoryTool"]
