# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correlated asynchronous server for private XR services."""

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

import zmq
import zmq.asyncio

from .protocol import PROTOCOL_VERSION, RPCError, pack, unpack

Dispatch = Callable[[str, dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]]


class RPCServer:
    """ROUTER server dispatching correlated requests without framework coupling."""

    def __init__(self, endpoint: str, dispatch: Dispatch) -> None:
        self.endpoint = endpoint
        self._dispatch = dispatch
        self._socket: zmq.asyncio.Socket | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._send_lock = asyncio.Lock()

    async def serve(self, *, ready: Callable[[], None] | None = None) -> None:
        if self._socket is not None:
            raise RuntimeError("RPC server is already running")
        socket = zmq.asyncio.Context.instance().socket(zmq.ROUTER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(self.endpoint)
        self._socket = socket
        if ready is not None:
            ready()
        try:
            while True:
                frames = await socket.recv_multipart()
                if len(frames) != 2:
                    continue
                identity, payload = frames
                task = asyncio.create_task(
                    self._handle(identity, payload),
                    name="xr-service-rpc-request",
                )
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        finally:
            await self.close()

    async def _handle(self, identity: bytes, payload: bytes) -> None:
        request_id = ""
        try:
            request = unpack(payload)
            request_id = str(request.get("request_id") or "")
            if request.get("version") != PROTOCOL_VERSION:
                raise RPCError("unsupported RPC protocol version", code="unsupported_version")
            operation = request.get("operation")
            arguments = request.get("arguments")
            if not isinstance(operation, str) or not isinstance(arguments, dict):
                raise RPCError("invalid RPC request", code="invalid_request")
            result = self._dispatch(operation, arguments)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, dict):
                raise RPCError("RPC handler must return a map", code="invalid_result")
            response = {
                "version": PROTOCOL_VERSION,
                "request_id": request_id,
                "ok": True,
                "result": result,
            }
        except RPCError as exc:
            response = {
                "version": PROTOCOL_VERSION,
                "request_id": request_id,
                "ok": False,
                "error": {"code": exc.code, "message": str(exc)},
            }
        except Exception as exc:
            response = {
                "version": PROTOCOL_VERSION,
                "request_id": request_id,
                "ok": False,
                "error": {"code": "internal_error", "message": str(exc)},
            }
        if self._socket is not None:
            async with self._send_lock:
                await self._socket.send_multipart([identity, pack(response)])

    async def close(self) -> None:
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        socket, self._socket = self._socket, None
        if socket is not None:
            socket.close(linger=0)
