# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward Riva PCM while owning synthesis and disconnect cleanup."""
from __future__ import annotations

import asyncio
from contextlib import aclosing

import grpc
from fastapi.responses import JSONResponse, StreamingResponse


class SpeechStreamResponse(StreamingResponse):
    def __init__(self, backend, text: str, rate: int, lock: asyncio.Lock, *, trailing_silence: bytes = b""):
        async def audio():
            # Hold the lock through RPC cancellation and worker cleanup as well
            # as generation. A disconnected client must not overlap the next RPC.
            async with lock:
                has_audio = False
                async with aclosing(backend.stream_pcm(text, timeout=120)) as chunks:
                    async for chunk in chunks:
                        if len(chunk) % 2:
                            raise ValueError("NIM returned invalid mono PCM")
                        if chunk:
                            has_audio = True
                            yield chunk
                # Silence belongs to the audio timeline; sleeping here can be
                # hidden by playback buffering. Never pad failed or empty RPCs.
                if has_audio and trailing_silence:
                    yield trailing_silence

        super().__init__(audio(), media_type="audio/pcm", headers={
            "X-Audio-Sample-Rate": str(rate), "X-Audio-Channels": "1",
        })

    async def stream_response(self, send):
        async with aclosing(self.body_iterator):
            # Fetch the first audio before HTTP 200 so initial NIM failures keep
            # their useful status codes. Later failures terminate the response.
            try:
                first = await anext(self.body_iterator, None)
            except (grpc.RpcError, TimeoutError, ValueError) as exc:
                if isinstance(exc, TimeoutError):
                    error = JSONResponse({"detail": "NIM request timed out"}, status_code=504)
                else:
                    error = JSONResponse({"detail": "Speech NIM request failed"}, status_code=502)
                await send({"type": "http.response.start", "status": error.status_code,
                            "headers": error.raw_headers})
                await send({"type": "http.response.body", "body": error.body})
                return
            await send({"type": "http.response.start", "status": self.status_code,
                        "headers": self.raw_headers})
            if first is not None:
                await send({"type": "http.response.body", "body": first, "more_body": True})
            async for chunk in self.body_iterator:
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def __call__(self, scope, receive, send):
        # Watch disconnects even while waiting for the lock or first audio.
        # Starlette's ASGI 2.4 path otherwise detects them only on a failed send.
        sender = asyncio.create_task(self.stream_response(send))
        disconnected = asyncio.create_task(self.listen_for_disconnect(receive))
        try:
            done, _ = await asyncio.wait((sender, disconnected), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                await task
        finally:
            for task in (sender, disconnected):
                if not task.done():
                    task.cancel()
            await asyncio.gather(sender, disconnected, return_exceptions=True)
