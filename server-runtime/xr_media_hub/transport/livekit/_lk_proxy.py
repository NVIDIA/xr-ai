# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared LiveKit proxy helpers used by both the standalone web server
(``_web_server.py``) and the token server (``_token_server.py``).

Both expose the same /rtc surface to browser/headset clients —
``GET /rtc/validate`` over HTTP and ``WS /rtc`` for signaling — so a
TLS-terminating front can reach a plaintext LiveKit signaling port
without mixed-content errors. The HTTP shells differ; the LiveKit-
facing proxy logic does not.
"""
from __future__ import annotations

import asyncio

import httpx
import websockets
from fastapi import Request, WebSocket
from fastapi.responses import Response
from loguru import logger


async def proxy_validate(
    client: httpx.AsyncClient,
    lk_internal_http: str,
    request: Request,
) -> Response:
    """Forward ``GET /rtc/validate?<qs>`` to LiveKit's internal HTTP."""
    qs = str(request.url.query)
    r = await client.get(f"{lk_internal_http}/rtc/validate?{qs}")
    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type", "text/plain"),
    )


async def pump_rtc_ws(client_ws: WebSocket, lk_internal_ws: str) -> None:
    """Bidirectional WebSocket pump between *client_ws* and LiveKit signaling.

    Same-origin /rtc proxy: lets browser clients on TLS pages reach a plain
    LiveKit signaling port without the mixed-content error.
    """
    qs = client_ws.scope.get("query_string", b"").decode()
    target = f"{lk_internal_ws}/rtc?{qs}"
    await client_ws.accept()
    try:
        async with websockets.connect(target, max_size=None) as lk_ws:

            async def c2l() -> None:
                try:
                    while True:
                        msg = await client_ws.receive()
                        msg_type = msg.get("type")
                        if msg_type == "websocket.disconnect":
                            break
                        # Only websocket.receive carries data — any other
                        # event type (lifespan ack, ping/pong) shouldn't
                        # land here, but skip explicitly so a future ASGI
                        # change doesn't accidentally forward control
                        # frames upstream as bogus payload.
                        if msg_type != "websocket.receive":
                            continue
                        if msg.get("bytes"):
                            await lk_ws.send(msg["bytes"])
                        elif msg.get("text"):
                            await lk_ws.send(msg["text"])
                except Exception as exc:
                    # Client-side receive/send can fail during normal disconnect
                    # races; keep teardown best-effort but retain debug visibility.
                    logger.debug("WS proxy /rtc c2l ended with error: {}", exc)
                finally:
                    await lk_ws.close()

            async def l2c() -> None:
                try:
                    async for frame in lk_ws:
                        if isinstance(frame, bytes):
                            await client_ws.send_bytes(frame)
                        else:
                            await client_ws.send_text(frame)
                except Exception:
                    # Either side closing the websocket during teardown can
                    # raise here; suppression is intentional to let shutdown
                    # complete without noisy expected-disconnect errors.
                    pass

            await asyncio.gather(c2l(), l2c(), return_exceptions=True)
    except Exception as exc:
        logger.debug("WS proxy /rtc error: {}", exc)
    finally:
        try:
            await client_ws.close()
        except Exception:
            # Best-effort cleanup: client may already be closed/disconnected.
            # Intentionally ignore close-time errors in shutdown path.
            pass
