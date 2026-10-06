# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private, local msgpack transport to the shared speaker-conditioned ASR process."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import msgpack
import zmq
import zmq.asyncio
from xr_ai_voicegate._speaker import _SpeakerConfig


async def _select_speaker_asr(cfg: _SpeakerConfig | None) -> bool:
    if cfg is None:
        return False
    available = await _SpeakerClient(cfg)._available()
    if not available and cfg.backend == "required":
        raise RuntimeError("speaker ASR is required but unavailable; start the diarization model stack")
    return available


class _SpeakerClient:
    def __init__(self, cfg: _SpeakerConfig) -> None:
        self._cfg = cfg
        self._sessions: dict[str, str] = {}
        self._retired: dict[str, None] = {}

    async def _request(self, body: dict) -> dict:
        # A fresh REQ socket makes timeouts recoverable without reusing a socket
        # still waiting for its previous reply.
        socket = zmq.asyncio.Context.instance().socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.IMMEDIATE, 1)
        socket.connect(self._cfg.endpoint)
        try:
            async with asyncio.timeout(self._cfg.timeout_s):
                await socket.send(msgpack.packb(body, use_bin_type=True))
                result = msgpack.unpackb(await socket.recv(), raw=False)
            if result.get("error"):
                raise RuntimeError(result["error"])
            return result
        finally:
            socket.close()

    async def _available(self) -> bool:
        if not Path(self._cfg.endpoint.removeprefix("ipc://")).exists():
            return False
        try:
            async with asyncio.timeout(min(2.0, self._cfg.timeout_s)):
                status = await self._request({"op": "status"})
        except TimeoutError:
            # A stale socket is not evidence that a speech backend is ready.
            return False
        if status.get("service") != "xr-ai-speaker-stt" or status.get("protocol") != 1:
            raise RuntimeError("speaker endpoint belongs to an incompatible service")
        return True

    async def _feed(self, pid: str, audio: bytes, pts_us: int) -> list[dict]:
        session = self._sessions.get(pid)
        if session is None:
            # Retry uncertain closes before allocating new inference state.
            # Keep this bounded so an unavailable service does not delay audio.
            try:
                async with asyncio.timeout(min(0.25, self._cfg.timeout_s)):
                    for retired in list(self._retired):
                        await self._close_session(retired)
            except Exception:
                pass
            session = uuid4().hex
            # Keep the identity before sending: cancellation can arrive after
            # the server allocates state but before its reply reaches us.
            self._sessions[pid] = session
            await self._request({"op": "open", "session": session, "config": asdict(self._cfg)})
        result = await self._request({"op": "audio", "session": session, "audio": audio, "pts_us": pts_us})
        return result["events"]

    async def _forget(self, pid: str) -> None:
        session = self._sessions.pop(pid, None)
        if session is not None:
            self._retired[session] = None
            # The service expires abandoned sessions. Bound local retry state
            # too, including identities from requests that never reached it.
            if len(self._retired) > 64:
                self._retired.pop(next(iter(self._retired)))
            await self._close_session(session)

    async def _close_session(self, session: str) -> None:
        await self._request({"op": "close", "session": session})
        self._retired.pop(session, None)

    async def _close(self) -> None:
        await asyncio.gather(
            *(self._forget(pid) for pid in list(self._sessions)), return_exceptions=True,
        )
        await asyncio.gather(
            *(self._close_session(session) for session in list(self._retired)), return_exceptions=True,
        )
