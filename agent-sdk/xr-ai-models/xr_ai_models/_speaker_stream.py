# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private transport for the repository's speaker-conditioned streaming service."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from websockets.asyncio.client import ClientConnection, connect


def _check_identity(status: dict[str, Any]) -> None:
    if status.get("service") != "xr-ai-speaker-stt" or status.get("protocol") != 3:
        raise RuntimeError("speaker endpoint belongs to an incompatible service")


async def _speaker_available(base_url: str, timeout_s: float) -> bool:
    async with httpx.AsyncClient(timeout=min(2.0, timeout_s)) as client:
        try:
            response = await client.get(f"{base_url.rstrip('/')}/health")
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
        response.raise_for_status()
        _check_identity(response.json())
        return True


class _SpeakerStream:
    """One connection owns one enrollment and ordered PCM timeline."""

    def __init__(self, base_url: str, timeout_s: float) -> None:
        self._url = base_url.rstrip('/').replace("http://", "ws://", 1).replace("https://", "wss://", 1)
        self._timeout_s = timeout_s
        self._socket: ClientConnection | None = None

    async def _open(self, config: dict[str, Any], audio_origin_us: int) -> None:
        async with asyncio.timeout(self._timeout_s):
            self._socket = await connect(
                f"{self._url}/v1/audio/transcriptions/stream",
                open_timeout=self._timeout_s, close_timeout=self._timeout_s,
            )
            await self._socket.send(json.dumps({"config": config, "audio_origin_us": audio_origin_us}))
            _check_identity(await self._response())

    async def _response(self) -> dict[str, Any]:
        assert self._socket is not None
        result = json.loads(await self._socket.recv())
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result

    async def _feed(self, audio: bytes) -> list[dict[str, Any]]:
        assert self._socket is not None
        async with asyncio.timeout(self._timeout_s):
            await self._socket.send(audio)
            return (await self._response())["events"]

    async def _close(self) -> None:
        socket, self._socket = self._socket, None
        if socket is not None:
            try:
                await socket.close()
            except BaseException:
                # The caller bounds shutdown. Cancellation must still sever
                # the connection that owns the server's inference session.
                socket.transport.abort()
                raise


def _make_speaker_stream(base_url: str, timeout_s: float) -> _SpeakerStream:
    return _SpeakerStream(base_url, timeout_s)
