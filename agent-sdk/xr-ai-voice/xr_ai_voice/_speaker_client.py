# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private voice policy over the model package's speaker streaming transport."""

from __future__ import annotations

import asyncio
from dataclasses import fields

from xr_ai_models._speaker_stream import _make_speaker_stream, _speaker_available, _SpeakerStream
from xr_ai_voicegate._conversation import _ConversationConfig
from xr_ai_voicegate._speaker import _SpeakerConfig

from ._readiness import wait_for_services


async def _select_speaker_asr(cfg: _SpeakerConfig | None) -> bool:
    if cfg is None:
        return False
    client = _SpeakerClient(cfg)
    available = await client._available()
    if available is False and cfg.backend == "auto":
        return False
    if available is not True:
        async def ready() -> bool:
            return await client._available() is True
        await wait_for_services({"speaker ASR": ready})
    return True


class _SpeakerClient:
    def __init__(self, cfg: _SpeakerConfig) -> None:
        self._cfg = cfg
        self._sessions: dict[str, _SpeakerStream] = {}

    async def _available(self) -> bool | None:
        return await _speaker_available(self._cfg.base_url, self._cfg.timeout_s)

    async def _feed(self, pid: str, audio: bytes, pts_us: int) -> list[dict]:
        stream = self._sessions.get(pid)
        if stream is None:
            stream = _make_speaker_stream(self._cfg.base_url, self._cfg.timeout_s)
            # Store before opening: cancellation after server allocation must
            # still close the connection and release its inference state.
            self._sessions[pid] = stream
            names = [f.name for f in fields(_ConversationConfig)] + [
                "activity_threshold", "silence_duration", "max_utterance_s",
            ]
            await stream._open({name: getattr(self._cfg, name) for name in names}, pts_us)
        return await stream._feed(audio)

    async def _forget(self, pid: str) -> None:
        stream = self._sessions.pop(pid, None)
        if stream is not None:
            await stream._close()

    async def _close(self) -> None:
        await asyncio.gather(*(self._forget(pid) for pid in list(self._sessions)), return_exceptions=True)
