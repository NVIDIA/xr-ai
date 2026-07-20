# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correlated asynchronous client for private XR services."""

import asyncio
import uuid
from contextlib import suppress
from typing import Any

import zmq
import zmq.asyncio

from .protocol import PROTOCOL_VERSION, RPCError, pack, unpack


class RPCClient:
    """Concurrent DEALER client with request correlation and bounded calls."""

    def __init__(self, endpoint: str, *, timeout_s: float = 10.0) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self._socket: zmq.asyncio.Socket | None = None
        self._receiver: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._send_lock = asyncio.Lock()

    def _ensure_started(self) -> None:
        if self._socket is not None:
            return
        socket = zmq.asyncio.Context.instance().socket(zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.endpoint)
        self._socket = socket
        self._receiver = asyncio.create_task(self._receive(), name="xr-service-rpc-client")

    async def call(
        self,
        operation: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        self._ensure_started()
        assert self._socket is not None
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        request = {
            "version": PROTOCOL_VERSION,
            "request_id": request_id,
            "operation": operation,
            "arguments": arguments or {},
        }
        try:
            async with self._send_lock:
                await self._socket.send(pack(request))
            try:
                response = await asyncio.wait_for(
                    future,
                    timeout=self.timeout_s if timeout_s is None else timeout_s,
                )
            except TimeoutError as exc:
                raise RPCError(
                    f"{operation} timed out calling {self.endpoint}",
                    code="timeout",
                ) from exc
        finally:
            self._pending.pop(request_id, None)

        if not response.get("ok"):
            error = response.get("error") or {}
            raise RPCError(
                str(error.get("message") or "remote operation failed"),
                code=str(error.get("code") or "remote_error"),
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise RPCError("RPC result must contain a map", code="invalid_response")
        return result

    async def _receive(self) -> None:
        assert self._socket is not None
        try:
            while True:
                response = unpack(await self._socket.recv())
                request_id = response.get("request_id")
                if not isinstance(request_id, str):
                    continue
                future = self._pending.get(request_id)
                if future is not None and not future.done():
                    future.set_result(response)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = RPCError(f"RPC receive failed: {exc}", code="connection_error")
            for future in tuple(self._pending.values()):
                if not future.done():
                    future.set_exception(error)
            socket, self._socket = self._socket, None
            self._receiver = None
            if socket is not None:
                socket.close(linger=0)

    async def close(self) -> None:
        receiver, self._receiver = self._receiver, None
        if receiver is not None:
            receiver.cancel()
            with suppress(asyncio.CancelledError):
                await receiver
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(RPCError("RPC client closed", code="closed"))
        self._pending.clear()
        socket, self._socket = self._socket, None
        if socket is not None:
            socket.close(linger=0)
